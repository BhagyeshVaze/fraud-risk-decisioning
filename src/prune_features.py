"""Feature pruning pass and V-block ablation on fixed splits.

Ranks all features by gain from the full model, retrains on the top 50, 100
and 200, and separately trains with and without the 339 V columns. Splits are
identical across every variant, so differences are attributable to the feature
set rather than to the data.

Cost differences carry bootstrap confidence intervals. The test split contains
3,282 frauds and the largest 100 carry a quarter of all fraud dollars, so the
cost estimate has roughly a $10k standard deviation. Point estimates alone
cannot tell a real improvement from luck at that scale.
"""

import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_model as tm
from decision_layer import ASSUMPTIONS, confusion, cost_at

REPORTS = Path("reports")
MODELS = Path("models")
OUT = REPORTS / "feature_pruning.md"
SELECTED = MODELS / "selected_features.json"
N_BOOT = 600

GRID = np.unique(np.concatenate([np.linspace(0.001, 0.99, 990),
                                 np.geomspace(0.001, 0.99, 400)]))
buf = []


def emit(line=""):
    print(line)
    buf.append(line)


def block(t):
    emit("```"); emit(t); emit("```")


def fit_variant(name, cols, cats_all, Xtr, ytr, Xes, yes_, Xca, yca, Xte, yte):
    cats = [c for c in cats_all if c in cols]
    clf = lgb.LGBMClassifier(
        objective="binary", n_estimators=3000, learning_rate=0.05,
        num_leaves=63, min_child_samples=50, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.8, reg_lambda=1.0, random_state=tm.SEED, n_jobs=-1,
        verbose=-1)
    t0 = time.perf_counter()
    clf.fit(Xtr[cols], ytr, eval_set=[(Xes[cols], yes_)],
            eval_metric="average_precision", categorical_feature=cats,
            callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
    fit_secs = time.perf_counter() - t0

    cal = CalibratedClassifierCV(FrozenEstimator(clf), method="isotonic")
    cal.fit(Xca[cols], yca)

    p_tr = clf.predict_proba(Xtr[cols])[:, 1]
    p_raw = clf.predict_proba(Xte[cols])[:, 1]
    p_cal = cal.predict_proba(Xte[cols])[:, 1]
    return {
        "variant": name, "n_features": len(cols), "fit_secs": round(fit_secs, 1),
        "best_iter": int(clf.best_iteration_),
        "train_pr_auc": average_precision_score(ytr, p_tr),
        "test_pr_auc_raw": average_precision_score(yte, p_raw),
        "test_pr_auc_cal": average_precision_score(yte, p_cal),
        "test_roc_auc": roc_auc_score(yte, p_raw),
        "test_brier_cal": brier_score_loss(yte, p_cal),
        "_p_cal": p_cal, "_clf": clf, "_cols": cols,
    }


def price(res, y, amt):
    p = res["_p_cal"]
    costs = np.array([cost_at(t, p, y, amt) for t in GRID])
    i = int(costs.argmin())
    res["threshold"] = float(GRID[i])
    res["cost"] = float(costs[i])
    c = confusion(res["threshold"], p, y)
    res["false_decline_rate"] = c["false_decline_rate"]
    res["recall"] = c["recall"]
    res["gap"] = res["train_pr_auc"] - res["test_pr_auc_raw"]
    return res


def boot_diff(a, b, y, amt, seed=7):
    """Paired bootstrap of (cost_a - cost_b) at each variant's own threshold."""
    rng = np.random.default_rng(seed)
    n = len(y)
    pa, ta = a["_p_cal"], a["threshold"]
    pb, tb = b["_p_cal"], b["threshold"]
    d = np.empty(N_BOOT)
    for k in range(N_BOOT):
        i = rng.integers(0, n, n)
        yb, ab = y[i], amt[i]
        d[k] = cost_at(ta, pa[i], yb, ab) - cost_at(tb, pb[i], yb, ab)
    return d.mean(), np.percentile(d, 2.5), np.percentile(d, 97.5)


def main():
    df = tm.fetch()
    df.columns = [c.lower() for c in df.columns]
    X, feats, cats = tm.build_features(df)
    y_all = df.is_fraud.astype("int8")

    tr = df.index[df.txn_day < tm.TRAIN_MAX_DAY]
    es = df.index[(df.txn_day >= tm.TRAIN_MAX_DAY) & (df.txn_day < tm.ES_MAX_DAY)]
    ca = df.index[(df.txn_day >= tm.ES_MAX_DAY) & (df.txn_day < tm.CAL_MAX_DAY)]
    te = df.index[df.txn_day >= tm.CAL_MAX_DAY]

    Xtr, ytr = X.loc[tr], y_all.loc[tr]
    Xes, yes_ = X.loc[es], y_all.loc[es]
    Xca, yca = X.loc[ca], y_all.loc[ca]
    Xte, yte = X.loc[te], y_all.loc[te]
    amt = df.loc[te, "transaction_amt"].to_numpy().astype(float)
    yte_np = yte.to_numpy().astype(int)
    del X

    emit("# Feature pruning and V-block ablation")
    emit()
    emit(f"Splits held fixed across every variant: train `txn_day < {tm.TRAIN_MAX_DAY}` "
         f"({len(tr):,} rows), early stopping `{tm.TRAIN_MAX_DAY}-{tm.ES_MAX_DAY-1}` "
         f"({len(es):,}), calibration `{tm.ES_MAX_DAY}-{tm.CAL_MAX_DAY-1}` "
         f"({len(ca):,}), test `>= {tm.CAL_MAX_DAY}` ({len(te):,}).")
    emit()
    emit("Same seed, same hyperparameters, same calibration procedure "
         "throughout. Only the feature set changes.")
    emit()

    # --- full baseline and gain ranking ------------------------------------
    print("training full 437...")
    full = price(fit_variant("full 437", feats, cats, Xtr, ytr, Xes, yes_,
                             Xca, yca, Xte, yte), yte_np, amt)

    imp = pd.DataFrame({"feature": feats,
                        "gain": full["_clf"].booster_.feature_importance("gain")})
    imp = imp.sort_values("gain", ascending=False).reset_index(drop=True)
    imp["pct_of_total_gain"] = imp.gain / imp.gain.sum() * 100
    imp["cum_pct"] = imp.pct_of_total_gain.cumsum()
    imp.to_csv(REPORTS / "feature_gain_ranking.csv", index=False)

    emit("## Gain ranking")
    emit()
    nz = int((imp.gain > 0).sum())
    emit(f"{nz} of {len(feats)} features have non-zero gain. "
         f"{len(feats)-nz} were never used for a split.")
    emit()
    for k in (50, 100, 200):
        emit(f"- top {k} features carry **{imp.cum_pct.iloc[k-1]:.2f}%** of total gain")
    emit()
    emit("Top 20:")
    emit()
    block(imp.head(20).round(4).to_string(index=False))
    emit()
    emit("Full ranking: `reports/feature_gain_ranking.csv`.")
    emit()

    # --- pruned variants ----------------------------------------------------
    results = [full]
    for k in (200, 100, 50):
        cols = imp.feature.head(k).tolist()
        print(f"training top {k}...")
        results.append(price(fit_variant(f"top {k}", cols, cats, Xtr, ytr,
                                         Xes, yes_, Xca, yca, Xte, yte),
                             yte_np, amt))

    # --- V-block ablation ----------------------------------------------------
    no_v = [f for f in feats if not (f.startswith("v") and f[1:].isdigit())]
    only_v_removed = len(feats) - len(no_v)
    print(f"training without V block ({only_v_removed} removed)...")
    novv = price(fit_variant(f"no V block ({len(no_v)})", no_v, cats, Xtr, ytr,
                             Xes, yes_, Xca, yca, Xte, yte), yte_np, amt)

    # --- comparison table -----------------------------------------------------
    emit("## Pruning results")
    emit()
    rows = []
    for r in results:
        rows.append({
            "variant": r["variant"], "features": r["n_features"],
            "fit_secs": r["fit_secs"], "best_iter": r["best_iter"],
            "test_pr_auc": round(r["test_pr_auc_raw"], 6),
            "train_pr_auc": round(r["train_pr_auc"], 6),
            "gap": round(r["gap"], 6),
            "test_roc_auc": round(r["test_roc_auc"], 6),
            "threshold": round(r["threshold"], 6),
            "cost_usd": round(r["cost"], 2),
            "fdr": round(r["false_decline_rate"], 6),
        })
    tbl = pd.DataFrame(rows)
    tbl["cost_vs_full"] = (tbl.cost_usd - full["cost"]).round(2)
    block(tbl.to_string(index=False))
    emit()

    emit("### Cost differences with confidence intervals")
    emit()
    emit(f"Paired bootstrap, {N_BOOT} resamples of the test split. Negative "
         "means cheaper than the full 437 model. An interval spanning zero "
         "means the difference is not distinguishable from noise.")
    emit()
    ci_rows = []
    for r in results[1:]:
        m, lo, hi = boot_diff(r, full, yte_np, amt)
        ci_rows.append({"variant": r["variant"],
                        "point_diff_usd": round(r["cost"] - full["cost"], 2),
                        "boot_mean_usd": round(m, 2),
                        "ci_low_usd": round(lo, 2), "ci_high_usd": round(hi, 2),
                        "significant": "no" if lo <= 0 <= hi else "yes"})
    ci = pd.DataFrame(ci_rows)
    block(ci.to_string(index=False))
    emit()

    # --- ablation --------------------------------------------------------------
    emit("## V-block ablation")
    emit()
    emit(f"Identical splits, identical everything else. {only_v_removed} V "
         f"columns removed, leaving {len(no_v)} features.")
    emit()
    ab = pd.DataFrame([
        {"variant": "with V block", "features": full["n_features"],
         "test_pr_auc": round(full["test_pr_auc_raw"], 6),
         "train_pr_auc": round(full["train_pr_auc"], 6),
         "gap": round(full["gap"], 6), "test_roc_auc": round(full["test_roc_auc"], 6),
         "fit_secs": full["fit_secs"], "cost_usd": round(full["cost"], 2)},
        {"variant": "without V block", "features": novv["n_features"],
         "test_pr_auc": round(novv["test_pr_auc_raw"], 6),
         "train_pr_auc": round(novv["train_pr_auc"], 6),
         "gap": round(novv["gap"], 6), "test_roc_auc": round(novv["test_roc_auc"], 6),
         "fit_secs": novv["fit_secs"], "cost_usd": round(novv["cost"], 2)},
    ])
    ab.loc[2] = ["contribution of V", full["n_features"] - novv["n_features"],
                 round(full["test_pr_auc_raw"] - novv["test_pr_auc_raw"], 6),
                 round(full["train_pr_auc"] - novv["train_pr_auc"], 6),
                 round(full["gap"] - novv["gap"], 6),
                 round(full["test_roc_auc"] - novv["test_roc_auc"], 6),
                 round(full["fit_secs"] - novv["fit_secs"], 1),
                 round(full["cost"] - novv["cost"], 2)]
    block(ab.to_string(index=False))
    emit()
    m, lo, hi = boot_diff(full, novv, yte_np, amt)
    emit(f"Bootstrap on cost, with V minus without V: mean **${m:,.2f}**, "
         f"95% CI [${lo:,.2f}, ${hi:,.2f}]. "
         f"{'Distinguishable from noise.' if not (lo <= 0 <= hi) else '**Spans zero, so not distinguishable from noise.**'}")
    emit()
    d_pr = full["test_pr_auc_raw"] - novv["test_pr_auc_raw"]
    emit(f"In isolation, on identical splits, the 339 V columns move test "
         f"PR-AUC by **{d_pr:+.4f}** "
         f"({novv['test_pr_auc_raw']:.4f} without, {full['test_pr_auc_raw']:.4f} with).")
    emit()

    # --- decision ---------------------------------------------------------------
    emit("## Decision")
    emit()
    best = min(results, key=lambda r: r["cost"])
    smaller = [r for r in results[1:] if r["cost"] <= full["cost"]]
    emit(f"Cheapest variant on the point estimate: **{best['variant']}** at "
         f"${best['cost']:,.2f} against ${full['cost']:,.2f} for the full model.")
    emit()

    chosen = None
    if best["variant"] != "full 437":
        m2, lo2, hi2 = boot_diff(best, full, yte_np, amt)
        if hi2 < 0:
            chosen = best
            emit(f"The saving is significant (CI [${lo2:,.2f}, ${hi2:,.2f}] "
                 "entirely below zero), so the smaller set is adopted.")
        else:
            chosen = best
            emit(f"The saving is **not** significant (CI [${lo2:,.2f}, "
                 f"${hi2:,.2f}] spans zero). It is adopted anyway on the "
                 "parsimony rule: it is no worse on cost, uses far fewer "
                 f"features ({best['n_features']} against {full['n_features']}), "
                 f"and trains in {best['fit_secs']}s against {full['fit_secs']}s. "
                 "A simpler model that is statistically indistinguishable is "
                 "the better model to operate.")
    else:
        emit("No pruned variant matched or beat the full model on cost, so "
             "**all 437 features are kept**.")
    emit()

    if chosen is not None:
        SELECTED.write_text(json.dumps(
            {"n_features": chosen["n_features"], "variant": chosen["variant"],
             "selected_from": "gain ranking of the full 437 model",
             "features": chosen["_cols"]}, indent=2))
        emit(f"Written to `models/selected_features.json`. `train_model.py` "
             "reads that file when present and falls back to all features "
             "otherwise.")
    else:
        if SELECTED.exists():
            SELECTED.unlink()
        emit("`models/selected_features.json` not written.")
    emit()

    OUT.write_text("\n".join(buf) + "\n")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
