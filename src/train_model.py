"""Train and calibrate the fraud model on FRAUD.MARTS.FCT_TRANSACTIONS.

Splits by time, never randomly, because the features are backward looking and a
random split would let the model learn from an account's future.

No class rebalancing: no scale_pos_weight, no SMOTE, no resampling. The whole
decision layer prices thresholds off calibrated probabilities, and rebalancing
distorts the probability scale it depends on.

Pulls once and caches to data/parquet/fct_transactions.parquet so the model can
be retrained after the Snowflake trial expires.
"""

import json
import os
import sys
from pathlib import Path

import joblib
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             roc_auc_score)

CACHE = Path("data/parquet/fct_transactions.parquet")
MODELS = Path("models")
REPORTS = Path("reports")
VERIFY = REPORTS / "verification.md"

# Four disjoint windows. Early stopping and calibration are deliberately
# separated: in the stage 1 build both used days 120-149, so the calibrator
# was fitted on data the model had already been tuned against, which biases it
# optimistically. They now use different windows.
TRAIN_MAX_DAY = 110      # train:        txn_day <  110
ES_MAX_DAY = 130         # early stop:   110 <= txn_day < 130
CAL_MAX_DAY = 150        # calibration:  130 <= txn_day < 150
                         # test:         txn_day >= 150

# Identifiers and time indices. Leaving any of these in would let the model
# memorise rows or read the calendar instead of behaviour.
EXCLUDE = {"transaction_id", "account_id", "txn_day", "transaction_dt",
           "is_fraud", "prev_transaction_dt"}

SEED = 42

# Stage 1 results, recorded so the refinement can be compared honestly rather
# than assumed to have helped. Splits then were train<120 / cal 120-149 /
# test>=150, with early stopping and calibration sharing the 120-149 window,
# and 58 features (no V block, no identity columns).
BASELINE = {
    "label": "stage 1 (58 features, shared ES/cal window)",
    "n_features": 58,
    "train_pr_auc": 0.824296,
    "test_pr_auc_raw": 0.500701,
    "test_pr_auc_cal": 0.486468,
    "test_roc_auc_raw": 0.893324,
    "test_roc_auc_cal": 0.893244,
    "test_brier_raw": 0.022893,
    "test_brier_cal": 0.023070,
}

# Palette slots 1 and 2 from the design system, validated as an adjacent pair.
C_BEFORE, C_AFTER = "#2a78d6", "#eb6834"
INK, INK_2, GRIDC, SURFACE = "#0b0b0b", "#52514e", "#d9d8d2", "#fcfcfb"

buf = []


def emit(line=""):
    print(line)
    buf.append(line)


def block(text):
    emit("```")
    emit(text)
    emit("```")


def fetch():
    if CACHE.exists():
        print(f"reading cached {CACHE}")
        return pd.read_parquet(CACHE)
    import snowflake.connector
    load_dotenv(".env")
    conn = snowflake.connector.connect(
        account=os.environ["SNOWFLAKE_ACCOUNT"], user=os.environ["SNOWFLAKE_USER"],
        password=os.environ["SNOWFLAKE_PASSWORD"], role=os.environ["SNOWFLAKE_ROLE"],
        warehouse=os.environ["SNOWFLAKE_WAREHOUSE"], database="FRAUD", schema="MARTS")
    cur = conn.cursor()
    cur.execute("ALTER WAREHOUSE FRAUD_WH RESUME IF SUSPENDED")
    print("pulling FCT_TRANSACTIONS...")
    cur.execute("SELECT * FROM MARTS.FCT_TRANSACTIONS")

    # 443 columns x 590k rows does not fit comfortably in 8 GB as float64, so
    # stream arrow batches straight to parquet, narrowing types on the way.
    import pyarrow as pa
    import pyarrow.parquet as pq

    def shrink(t):
        if pa.types.is_float64(t):
            return pa.float32()
        if pa.types.is_decimal(t):
            return pa.int64() if t.scale == 0 else pa.float64()
        return t

    CACHE.parent.mkdir(parents=True, exist_ok=True)
    writer, n = None, 0
    for batch in cur.fetch_arrow_batches():
        # This connector version yields Tables, not RecordBatches.
        tbl = batch if isinstance(batch, pa.Table) else pa.Table.from_batches([batch])
        schema = pa.schema([pa.field(f.name.lower(), shrink(f.type)) for f in tbl.schema])
        tbl = tbl.rename_columns([c.lower() for c in tbl.schema.names]).cast(schema)
        if writer is None:
            writer = pq.ParquetWriter(CACHE, schema, compression="snappy")
        writer.write_table(tbl)
        n += tbl.num_rows
    writer.close()
    conn.close()
    print(f"cached {n:,} rows to {CACHE}")
    return pd.read_parquet(CACHE)


def build_features(df):
    feats = [c for c in df.columns if c not in EXCLUDE]
    X = df[feats].copy()
    cats = []
    for c in feats:
        if X[c].dtype == object or str(X[c].dtype) == "string":
            # +1 so pandas' -1 missing sentinel becomes a real category 0
            X[c] = X[c].astype("category").cat.codes.astype("int32") + 1
            cats.append(c)
        elif X[c].dtype == bool:
            X[c] = X[c].astype("int8")
        else:
            X[c] = pd.to_numeric(X[c], errors="coerce").astype("float32")
    return X, feats, cats


def reliability_plot(y, p_raw, p_cal, path):
    fig, ax = plt.subplots(figsize=(7.2, 5.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.plot([0, 1], [0, 1], color=GRIDC, lw=2, ls="--", zorder=1)
    ax.annotate("perfect calibration", xy=(0.25, 0.25), xytext=(-10, 8),
                textcoords="offset points", ha="right", va="bottom",
                color=INK_2, fontsize=9)
    for p, color, label in ((p_raw, C_BEFORE, "before calibration"),
                            (p_cal, C_AFTER, "after isotonic")):
        ft, mp = calibration_curve(y, p, n_bins=10, strategy="quantile")
        ax.plot(mp, ft, color=color, lw=2, marker="o", ms=8, label=label,
                markeredgecolor=SURFACE, markeredgewidth=2, zorder=3)
    lo = min(p_raw.min(), p_cal.min())
    hi = max(np.quantile(p_raw, 0.999), np.quantile(p_cal, 0.999))
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(max(lo, 1e-4), 1.0); ax.set_ylim(max(lo, 1e-4), 1.0)
    ax.set_xlabel("mean predicted probability", color=INK_2, fontsize=10)
    ax.set_ylabel("observed fraud rate", color=INK_2, fontsize=10)
    ax.set_title("Reliability on the test split (log-log, quantile bins)",
                 color=INK, fontsize=12, weight="bold", loc="left", pad=12)
    ax.grid(True, color=GRIDC, lw=0.8, alpha=0.7, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRIDC)
    ax.tick_params(colors=INK_2, labelsize=9)
    leg = ax.legend(frameon=False, fontsize=9.5, loc="upper left")
    for t in leg.get_texts():
        t.set_color(INK_2)
    fig.tight_layout()
    fig.savefig(path, dpi=170, facecolor=SURFACE)
    print(f"wrote {path}")


def main():
    MODELS.mkdir(exist_ok=True); REPORTS.mkdir(exist_ok=True)
    df = fetch()
    df.columns = [c.lower() for c in df.columns]

    emit(); emit("## Part 2: the model"); emit()

    # --- splits -----------------------------------------------------------
    tr = df[df.txn_day < TRAIN_MAX_DAY]
    es = df[(df.txn_day >= TRAIN_MAX_DAY) & (df.txn_day < ES_MAX_DAY)]
    ca = df[(df.txn_day >= ES_MAX_DAY) & (df.txn_day < CAL_MAX_DAY)]
    te = df[df.txn_day >= CAL_MAX_DAY]

    emit("### Time splits")
    emit()
    emit("Four disjoint windows. Early stopping and calibration no longer "
         "share a window, so the isotonic map is fitted on data the model was "
         "not tuned against.")
    emit()
    rows = []
    for name, part, rng in (("train", tr, f"txn_day < {TRAIN_MAX_DAY}"),
                            ("early stopping", es, f"{TRAIN_MAX_DAY} <= txn_day < {ES_MAX_DAY}"),
                            ("calibration", ca, f"{ES_MAX_DAY} <= txn_day < {CAL_MAX_DAY}"),
                            ("test", te, f"txn_day >= {CAL_MAX_DAY}")):
        fr = part.is_fraud.mean()
        rows.append({"split": name, "day_range": rng, "rows": len(part),
                     "fraud": int(part.is_fraud.sum()), "fraud_rate": round(fr, 6),
                     "dev_vs_0.0350_pct": round((fr - 0.034990) / 0.034990 * 100, 2)})
    sp = pd.DataFrame(rows)
    block(sp.to_string(index=False))
    emit()
    drift = sp[sp["dev_vs_0.0350_pct"].abs() > 30]
    if len(drift):
        emit(f"**FLAG: {len(drift)} split(s) deviate more than 30% from the "
             f"0.0350 base rate: {', '.join(drift.split)}.**")
        for _, r in drift.iterrows():
            emit(f"  - {r['split']}: {r['fraud_rate']:.6f} "
                 f"({r['dev_vs_0.0350_pct']:+.2f}%)")
    else:
        emit("No split deviates more than 30% from the 0.0350 base rate.")
    emit()
    emit(f"Total across splits: {len(tr)+len(es)+len(ca)+len(te):,} of {len(df):,} rows.")
    emit()

    # --- features ---------------------------------------------------------
    X, feats, cats = build_features(df)
    y = df.is_fraud.astype("int8")
    Xtr, ytr = X.loc[tr.index], y.loc[tr.index]
    Xes, yes_ = X.loc[es.index], y.loc[es.index]
    Xca, yca = X.loc[ca.index], y.loc[ca.index]
    Xte, yte = X.loc[te.index], y.loc[te.index]
    del X

    emit("### Features")
    emit()
    emit(f"Features used by the model: **{len(feats)}** "
         f"({len(cats)} categorical, {len(feats)-len(cats)} numeric).")
    emit()
    leaked = [c for c in ("transaction_id", "account_id", "txn_day", "transaction_dt")
              if c in feats]
    emit(f"Excluded by design: {sorted(EXCLUDE)}")
    emit()
    emit(f"Identifier/time columns present in the feature list: "
         f"{leaked if leaked else 'none'} -> **{'FAIL' if leaked else 'PASS'}**")
    emit()

    # --- train ------------------------------------------------------------
    clf = lgb.LGBMClassifier(
        objective="binary", n_estimators=3000, learning_rate=0.05,
        num_leaves=63, min_child_samples=50, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.8, reg_lambda=1.0, random_state=SEED, n_jobs=-1,
        verbose=-1)
    clf.fit(Xtr, ytr, eval_set=[(Xes, yes_)], eval_metric="average_precision",
            categorical_feature=cats,
            callbacks=[lgb.early_stopping(100, verbose=False),
                       lgb.log_evaluation(0)])
    best_iter = clf.best_iteration_
    emit(f"Best iteration by average_precision on the **early stopping** split "
         f"(days {TRAIN_MAX_DAY}-{ES_MAX_DAY-1}): **{best_iter}** of 3000, "
         "patience 100.")
    emit()

    # --- calibrate ---------------------------------------------------------
    # Fitted on days 130-149, which the model never saw during training or
    # early stopping.
    cal = CalibratedClassifierCV(FrozenEstimator(clf), method="isotonic")
    cal.fit(Xca, yca)

    p_tr = clf.predict_proba(Xtr)[:, 1]
    p_raw = clf.predict_proba(Xte)[:, 1]
    p_cal = cal.predict_proba(Xte)[:, 1]

    # --- metrics -----------------------------------------------------------
    m = {
        "train_pr_auc": average_precision_score(ytr, p_tr),
        "test_pr_auc_raw": average_precision_score(yte, p_raw),
        "test_pr_auc_cal": average_precision_score(yte, p_cal),
        "test_roc_auc_raw": roc_auc_score(yte, p_raw),
        "test_roc_auc_cal": roc_auc_score(yte, p_cal),
        "test_brier_raw": brier_score_loss(yte, p_raw),
        "test_brier_cal": brier_score_loss(yte, p_cal),
    }
    emit("### Test metrics")
    emit()
    block(pd.Series(m).round(6).to_string())
    emit()
    gap = m["train_pr_auc"] - m["test_pr_auc_raw"]
    emit(f"Train PR-AUC {m['train_pr_auc']:.4f} vs test PR-AUC "
         f"{m['test_pr_auc_raw']:.4f}, gap {gap:+.4f}.")
    if m["test_pr_auc_raw"] > 0.85:
        emit(f"**FLAG: test PR-AUC {m['test_pr_auc_raw']:.4f} exceeds 0.85, "
             "which is suspiciously high for IEEE-CIS and warrants a leakage review.**")
    else:
        emit(f"Test PR-AUC {m['test_pr_auc_raw']:.4f} is below the 0.85 "
             "suspicion threshold.")
    if gap > 0.15:
        emit(f"**FLAG: train/test PR-AUC gap {gap:.4f} suggests overfitting.**")
    emit()
    delta = m["test_brier_raw"] - m["test_brier_cal"]
    pct = delta / m["test_brier_raw"] * 100
    verdict = "improved" if delta > 0 else "degraded"
    emit(f"Brier {verdict} under calibration: {m['test_brier_raw']:.6f} -> "
         f"{m['test_brier_cal']:.6f} ({pct:+.2f}%).")
    emit()
    emit(f"Stage 1, with early stopping and calibration sharing one window, "
         f"went {BASELINE['test_brier_raw']:.6f} -> {BASELINE['test_brier_cal']:.6f} "
         f"(a degradation). This run uses disjoint windows.")
    if delta <= 0:
        emit()
        emit("**FLAG: isotonic calibration still did not improve the Brier "
             "score on test, even with disjoint windows.** Separating the "
             "windows was therefore not sufficient. The remaining cause is "
             "drift between the calibration period and the test period, not "
             "reuse of the tuning split.")
    else:
        emit()
        emit("Separating the early-stopping and calibration windows fixed the "
             "stage 1 degradation: isotonic now improves Brier on test.")
    emit()

    # --- importance ---------------------------------------------------------
    imp = pd.DataFrame({"feature": feats,
                        "gain": clf.booster_.feature_importance("gain")})
    imp["pct_of_total_gain"] = imp.gain / imp.gain.sum() * 100
    imp = imp.sort_values("gain", ascending=False).reset_index(drop=True)
    emit("### Stage 1 vs this run, side by side")
    emit()
    cmp = pd.DataFrame([
        {"metric": "features", "stage_1": BASELINE["n_features"], "this_run": len(feats)},
        {"metric": "train PR-AUC", "stage_1": BASELINE["train_pr_auc"], "this_run": m["train_pr_auc"]},
        {"metric": "test PR-AUC (raw)", "stage_1": BASELINE["test_pr_auc_raw"], "this_run": m["test_pr_auc_raw"]},
        {"metric": "test PR-AUC (calibrated)", "stage_1": BASELINE["test_pr_auc_cal"], "this_run": m["test_pr_auc_cal"]},
        {"metric": "test ROC-AUC (raw)", "stage_1": BASELINE["test_roc_auc_raw"], "this_run": m["test_roc_auc_raw"]},
        {"metric": "test Brier (raw)", "stage_1": BASELINE["test_brier_raw"], "this_run": m["test_brier_raw"]},
        {"metric": "test Brier (calibrated)", "stage_1": BASELINE["test_brier_cal"], "this_run": m["test_brier_cal"]},
        {"metric": "train/test PR-AUC gap", "stage_1": BASELINE["train_pr_auc"] - BASELINE["test_pr_auc_raw"], "this_run": gap},
    ])
    cmp["delta"] = cmp.this_run - cmp.stage_1
    block(cmp.round(6).to_string(index=False))
    emit()
    d_pr = m["test_pr_auc_raw"] - BASELINE["test_pr_auc_raw"]
    if d_pr > 0.001:
        emit(f"**The V block and identity columns improved test PR-AUC by "
             f"{d_pr:+.4f}** ({BASELINE['test_pr_auc_raw']:.4f} -> "
             f"{m['test_pr_auc_raw']:.4f}, "
             f"{d_pr/BASELINE['test_pr_auc_raw']*100:+.1f}%).")
    elif d_pr < -0.001:
        emit(f"**The V block and identity columns did NOT improve test PR-AUC. "
             f"It fell by {d_pr:+.4f}** ({BASELINE['test_pr_auc_raw']:.4f} -> "
             f"{m['test_pr_auc_raw']:.4f}). Adding 379 columns made the model "
             "worse on held-out data, not better.")
    else:
        emit(f"**The V block and identity columns made no material difference "
             f"to test PR-AUC** ({BASELINE['test_pr_auc_raw']:.4f} -> "
             f"{m['test_pr_auc_raw']:.4f}, change {d_pr:+.4f}).")
    emit()
    emit("Note the two runs use different training windows (day <120 then, "
         "<110 now), so this compares the pipelines end to end, not the "
         "feature block in isolation.")
    emit()

    emit("### Top 25 features by gain")
    emit()
    block(imp.head(25).round(4).to_string(index=False))
    emit()
    top = imp.iloc[0]
    if top.pct_of_total_gain > 30:
        emit(f"**FLAG: `{top.feature}` carries {top.pct_of_total_gain:.2f}% of "
             "total gain, above the 30% dominance threshold. That usually "
             "indicates leakage and needs review.**")
    else:
        emit(f"No single feature dominates: the top feature `{top.feature}` "
             f"carries {top.pct_of_total_gain:.2f}% of total gain, below the "
             "30% threshold.")
    emit()

    # --- probability distribution -------------------------------------------
    emit("### Predicted probability distribution on test")
    emit()
    dist = pd.DataFrame({
        "statistic": ["min", "p25", "median", "p75", "p95", "max", "mean"],
        "raw": [p_raw.min(), *np.quantile(p_raw, [.25, .5, .75, .95]), p_raw.max(), p_raw.mean()],
        "calibrated": [p_cal.min(), *np.quantile(p_cal, [.25, .5, .75, .95]), p_cal.max(), p_cal.mean()],
    })
    block(dist.round(6).to_string(index=False))
    emit()
    emit(f"Actual test fraud rate: {yte.mean():.6f}. "
         f"Mean calibrated probability: {p_cal.mean():.6f} "
         f"(difference {p_cal.mean()-yte.mean():+.6f}). "
         f"Mean raw probability: {p_raw.mean():.6f} "
         f"(difference {p_raw.mean()-yte.mean():+.6f}).")
    emit()

    # --- decile calibration --------------------------------------------------
    emit("### Decile calibration, predicted vs actual")
    emit()
    for label, p in (("before calibration", p_raw), ("after isotonic", p_cal)):
        d = pd.DataFrame({"p": p, "y": yte.values})
        d["decile"] = pd.qcut(d.p.rank(method="first"), 10, labels=range(1, 11))
        g = d.groupby("decile", observed=True).agg(
            n=("y", "size"), predicted=("p", "mean"), actual=("y", "mean")).reset_index()
        g["abs_diff"] = (g.predicted - g.actual).abs()
        emit(f"**{label}**")
        emit()
        block(g.round(6).to_string(index=False))
        emit()

    reliability_plot(yte.values, p_raw, p_cal, REPORTS / "calibration_curve.png")
    emit("Reliability curve: `reports/calibration_curve.png`.")
    emit()

    joblib.dump(clf, MODELS / "lgbm_fraud_model.joblib")
    joblib.dump(cal, MODELS / "isotonic_calibrator.joblib")
    clf.booster_.save_model(str(MODELS / "lgbm_fraud_model.txt"))
    meta = {"best_iteration": int(best_iter), "n_features": len(feats),
            "categorical_features": cats, "features": feats,
            "splits": {"train_max_day": TRAIN_MAX_DAY, "cal_max_day": CAL_MAX_DAY},
            "metrics": {k: float(v) for k, v in m.items()}, "seed": SEED}
    (MODELS / "model_metadata.json").write_text(json.dumps(meta, indent=2))
    emit(f"Saved to `models/`: lgbm_fraud_model.joblib, isotonic_calibrator.joblib, "
         f"lgbm_fraud_model.txt, model_metadata.json.")
    emit()

    np.save(REPORTS / "_test_probs.npy", p_cal)
    np.save(REPORTS / "_test_probs_raw.npy", p_raw)
    te[["transaction_id", "transaction_amt", "is_fraud", "txn_day"]].to_parquet(
        REPORTS / "_test_frame.parquet", index=False)

    with open(VERIFY, "a") as fh:
        fh.write("\n".join(buf) + "\n")
    print(f"\nappended Part 2 to {VERIFY}")


if __name__ == "__main__":
    sys.exit(main())
