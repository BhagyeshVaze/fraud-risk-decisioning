"""Experiments behind the improvement pass. Results in reports/.

Usage: python src/experiments.py {cost|proxy|prune|improve|split}

All offline against the cached parquet; none needs Snowflake.
"""

import json
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import core
from core import COSTS, MODELS, REPORTS, Report

SWAP = ["prior_txn_count", "prior_amt_sum", "prior_amt_mean", "txn_count_1h",
        "txn_count_24h", "txn_count_7d", "amt_to_prior_mean_ratio",
        "seconds_since_prev_txn"]
RAW_COLS = ["TransactionID", "TransactionDT", "TransactionAmt", "txn_day",
            "card1", "card2", "card3", "card5", "addr1", "D1", "P_emaildomain"]
D_NORM = [f"d{i}" for i in list(range(1, 9)) + list(range(10, 16))]  # D9 is a time fraction
REG_PARAMS = dict(num_leaves=31, min_child_samples=200, colsample_bytree=0.5,
                  subsample=0.7, reg_lambda=20.0, reg_alpha=1.0, learning_rate=0.03)
INK, INK_2, GRIDC, SURFACE = "#0b0b0b", "#52514e", "#d9d8d2", "#fcfcfb"


def load_selected():
    sel = json.loads((MODELS / "selected_features.json").read_text())["features"]
    need = sorted(set(sel) | {"transaction_id", "txn_day", "is_fraud", "transaction_amt"})
    return sel, pd.read_parquet(core.FCT, columns=need)


def load_raw_aligned(ids):
    raw = pd.read_parquet(core.DATA / "transactions.parquet", columns=RAW_COLS)
    raw = raw.set_index("TransactionID").loc[ids].reset_index()
    assert (raw.TransactionID.to_numpy() == np.asarray(ids)).all()
    return raw


# --------------------------------------------------------------------------- #
# cost sensitivity
# --------------------------------------------------------------------------- #

def cost():
    R = Report(REPORTS / "cost_sensitivity.md")
    p = np.load(REPORTS / "_test_probs.npy")
    te = pd.read_parquet(REPORTS / "_test_frame.parquet")
    y, amt = te.is_fraud.to_numpy().astype(int), te.transaction_amt.to_numpy().astype(float)

    R("# Cost assumption sensitivity"); R()
    R("Every dollar figure rests on five constants that were chosen, not "
      "measured. **If they are wrong, does the recommendation change?**"); R()
    R("Churn probability and lifetime value enter only as their product, the "
      "cost of a false decline, $10.00 at the stated 5% and $200. That term "
      "generates $913,540 of the $1,414,895 decline-everything cost."); R()
    t0, c0 = core.sweep(p, y, amt)[:2]
    R(f"Baseline: false-decline cost $10.00, optimal threshold **{t0:.4f}**, "
      f"cost **${c0:,.0f}**."); R()

    R("## Sweeping the cost of a false decline"); R()
    rows = []
    for fd in [0.5, 1, 2, 4, 6, 8, 10, 15, 20, 30, 50, 80, 120]:
        c = dict(COSTS, churn_prob_after_false_decline=fd / COSTS["account_lifetime_value_usd"])
        t, ct = core.sweep(p, y, amt, c)[:2]
        base = core.cost_at(t0, p, y, amt, c)
        rows.append({"false_decline_cost_usd": fd,
                     "implied_churn_at_200_ltv": round(fd / 200, 4),
                     "optimal_threshold": round(t, 4), "cost_at_own_optimum": round(ct, 0),
                     "cost_at_baseline_thr": round(base, 0), "regret_usd": round(base - ct, 0),
                     "saving_vs_0.5": round(core.cost_at(0.5, p, y, amt, c) - base, 0)})
    sw = pd.DataFrame(rows); R.table(sw); R()
    R("`regret_usd` is the cost of keeping the baseline threshold when the true "
      "false-decline cost is that row. A threshold is only wrong if using it is "
      "expensive."); R()
    flips = sw[sw["saving_vs_0.5"] <= 0]
    R(f"**The recommendation flips** at ${flips.false_decline_cost_usd.min():.2f} "
      "or above: past that the baseline threshold no longer beats the 0.5 cutoff."
      if len(flips) else
      "Across the entire swept range the baseline threshold beats the 0.5 cutoff."); R()

    R("## Churn probability by lifetime value"); R()
    churns = np.array([0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.20])
    ltvs = np.array([50, 100, 150, 200, 300, 500, 800])
    thr_s, reg_s = np.zeros((len(churns), len(ltvs))), np.zeros((len(churns), len(ltvs)))
    for i, ch in enumerate(churns):
        for j, lt in enumerate(ltvs):
            c = dict(COSTS, churn_prob_after_false_decline=ch, account_lifetime_value_usd=lt)
            t, ct = core.sweep(p, y, amt, c)[:2]
            thr_s[i, j], reg_s[i, j] = t, core.cost_at(t0, p, y, amt, c) - ct
    idx = [f"churn {c:.0%}" for c in churns]; col = [f"LTV ${l}" for l in ltvs]
    R("Optimal threshold at each combination:"); R()
    R.block(pd.DataFrame(thr_s, index=idx, columns=col).round(4).to_string()); R()
    R(f"Regret from using the baseline {t0:.4f} threshold instead:"); R()
    R.block(pd.DataFrame(reg_s, index=idx, columns=col).round(0).to_string()); R()
    R(f"Threshold ranges {thr_s.min():.4f} to {thr_s.max():.4f}, a "
      f"{thr_s.max()/thr_s.min():.0f}x spread. Regret ranges ${reg_s.min():,.0f} "
      f"to ${reg_s.max():,.0f}."); R()
    R(f"**{(reg_s < 9900).mean()*100:.0f}% of the grid has regret below $9,900**, "
      "the measured one-standard-deviation noise on the cost estimate. In that "
      "region getting the constants wrong costs nothing detectable."); R()

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), facecolor=SURFACE)
    ax = axes[0]; ax.set_facecolor(SURFACE)
    ax.plot(sw.false_decline_cost_usd, sw.optimal_threshold, color="#2a78d6", lw=2,
            marker="o", ms=8, markeredgecolor=SURFACE, markeredgewidth=2)
    ax.axvline(10, color="#eb6834", lw=2, ls="--")
    ax.annotate("stated assumption\n$10.00", xy=(10, sw.optimal_threshold.iloc[6]),
                xytext=(-10, 34), textcoords="offset points", ha="right",
                color=INK_2, fontsize=9, weight="bold")
    ax.set_xscale("log"); ax.set_yscale("log")
    _style(ax, "Optimal threshold against the false-decline cost",
           "assumed cost of one false decline (USD, log)", "optimal decline threshold (log)")
    ax = axes[1]; ax.set_facecolor(SURFACE)
    im = ax.imshow(reg_s / 1000, cmap="YlOrBr", aspect="auto", origin="lower")
    ax.set_xticks(range(len(ltvs))); ax.set_xticklabels([f"${l}" for l in ltvs], fontsize=9)
    ax.set_yticks(range(len(churns))); ax.set_yticklabels([f"{c:.0%}" for c in churns], fontsize=9)
    for i in range(len(churns)):
        for j in range(len(ltvs)):
            v = reg_s[i, j] / 1000
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=8.5,
                    color=INK if v < reg_s.max() / 1000 * 0.6 else SURFACE)
    ax.set_xlabel("account lifetime value", color=INK_2, fontsize=10)
    ax.set_ylabel("churn probability after a false decline", color=INK_2, fontsize=10)
    ax.set_title(f"Regret from keeping the {t0:.3f} threshold (thousands USD)",
                 color=INK, fontsize=12, weight="bold", loc="left", pad=12)
    ax.tick_params(colors=INK_2)
    fig.colorbar(im, ax=ax, shrink=0.85, label="regret ($k)")
    fig.tight_layout(); fig.savefig(REPORTS / "cost_sensitivity.png", dpi=170, facecolor=SURFACE)
    R("Chart: `reports/cost_sensitivity.png`."); R()
    R.write()


def _style(ax, title, xlabel, ylabel):
    ax.set_xlabel(xlabel, color=INK_2, fontsize=10)
    ax.set_ylabel(ylabel, color=INK_2, fontsize=10)
    ax.set_title(title, color=INK, fontsize=12, weight="bold", loc="left", pad=12)
    ax.grid(True, color=GRIDC, lw=0.8, alpha=0.7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRIDC)
    ax.tick_params(colors=INK_2, labelsize=9)


# --------------------------------------------------------------------------- #
# account proxy
# --------------------------------------------------------------------------- #

def proxy():
    R = Report(REPORTS / "proxy_candidates.md")
    sel, fct = load_selected()
    raw = load_raw_aligned(fct.transaction_id)
    y = fct.is_fraud.to_numpy().astype(int)
    day, amt = fct.txn_day.to_numpy(), fct.transaction_amt.to_numpy().astype(float)

    R("# Account proxy candidates on the walk-forward harness"); R()
    R("Four non-overlapping 16-day test windows (days 118-181). Threshold chosen "
      "out of sample. Everything except the eight behavioural features is held "
      "identical."); R()

    per, frames = {}, []
    for name, fn in core.CANDIDATES.items():
        print(f"\n=== {name} ===")
        blk = core.build_proxy_block(raw, fn)
        X = fct[sel].copy()
        for c in SWAP:
            X[c] = blk[c].to_numpy()
        df = core.run_config(name, X, y, day, amt)
        per[name] = df; frames.append(df)
        del X, blk

    res = pd.concat(frames, ignore_index=True)
    res.to_csv(REPORTS / "proxy_walkforward_folds.csv", index=False)
    R("## Per-fold results"); R()
    R.table(res[["config", "fold_test_start", "n_test", "n_fraud_test", "test_pr_auc",
                 "threshold", "cost", "cost_at_half", "false_decline_rate"]].round(5)); R()

    agg = res.groupby("config").agg(
        pr_auc_mean=("test_pr_auc", "mean"), pr_auc_sd=("test_pr_auc", "std"),
        roc_auc_mean=("test_roc_auc", "mean"), gap_mean=("train_pr_auc", "mean"),
        cost_total=("cost", "sum"), cost_mean=("cost", "mean"), cost_sd=("cost", "std"),
        fdr_mean=("false_decline_rate", "mean"), caught=("fraud_dollars_caught", "sum"),
        fraud_total=("fraud_dollars_total", "sum"), fit_secs=("fit_secs", "mean"))
    agg["gap_mean"] -= agg.pr_auc_mean
    agg["pct_fraud_caught"] = agg.caught / agg.fraud_total * 100
    agg = agg.sort_values("cost_total")
    R("## Summary across the four folds"); R(); R.table(agg.round(5), index=True); R()

    base = "stage1 (card1+addr1+start)"
    R("## Paired comparison against the shipped proxy"); R()
    R("Only four pairs, so the t statistic is indicative. The per-fold count is "
      "the honest view."); R()
    rows = []
    for name, df in per.items():
        if name == base:
            continue
        c = core.paired_fold_test(df, per[base], "cost")
        pa = core.paired_fold_test(df, per[base], "test_pr_auc")
        rows.append({"candidate": name, "cost_diff_total": round(c["mean_diff"] * c["n_folds"], 0),
                     "cost_diff_per_fold": round(c["mean_diff"], 0),
                     "cost_sd_across_folds": round(c["sd_diff"], 0), "cost_t": round(c["t"], 2),
                     "folds_cheaper": int((c["per_fold"] < 0).sum()),
                     "pr_auc_diff": round(pa["mean_diff"], 5), "pr_auc_t": round(pa["t"], 2)})
    R.table(pd.DataFrame(rows).sort_values("cost_diff_total")); R()
    best = agg.index[0]
    R(f"Cheapest across all four folds: **{best}** at ${agg.cost_total.iloc[0]:,.0f}, "
      f"against ${agg.loc[base,'cost_total']:,.0f} for the shipped proxy."); R()
    R.write()


# --------------------------------------------------------------------------- #
# feature pruning
# --------------------------------------------------------------------------- #

def _fit_variant(name, cols, cats_all, parts):
    Xtr, ytr, Xes, yes_, Xca, yca, Xte, yte = parts
    import lightgbm as lgb, time
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.frozen import FrozenEstimator
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
    cats = [c for c in cats_all if c in cols]
    clf = lgb.LGBMClassifier(**core.LGB_PARAMS)
    t0 = time.perf_counter()
    clf.fit(Xtr[cols], ytr, eval_set=[(Xes[cols], yes_)], eval_metric="average_precision",
            categorical_feature=cats,
            callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
    secs = time.perf_counter() - t0
    cal = CalibratedClassifierCV(FrozenEstimator(clf), method="isotonic")
    cal.fit(Xca[cols], yca)
    p_cal = cal.predict_proba(Xte[cols])[:, 1]
    return {"variant": name, "n_features": len(cols), "fit_secs": round(secs, 1),
            "best_iter": int(clf.best_iteration_),
            "train_pr_auc": average_precision_score(ytr, clf.predict_proba(Xtr[cols])[:, 1]),
            "test_pr_auc_raw": average_precision_score(yte, clf.predict_proba(Xte[cols])[:, 1]),
            "test_pr_auc_cal": average_precision_score(yte, p_cal),
            "test_roc_auc": roc_auc_score(yte, clf.predict_proba(Xte[cols])[:, 1]),
            "test_brier_cal": brier_score_loss(yte, p_cal),
            "_p": p_cal, "_clf": clf, "_cols": cols}


def prune():
    R = Report(REPORTS / "feature_pruning.md")
    df = pd.read_parquet(core.FCT)
    df.columns = [c.lower() for c in df.columns]
    feats = [c for c in df.columns
             if c not in {"transaction_id", "account_id", "txn_day", "transaction_dt",
                          "is_fraud", "prev_transaction_dt"}]
    X, cats = core.encode(df[feats])
    y = df.is_fraud.astype("int8")
    tr = df.index[df.txn_day < core.TRAIN_MAX_DAY]
    es = df.index[(df.txn_day >= core.TRAIN_MAX_DAY) & (df.txn_day < core.ES_MAX_DAY)]
    ca = df.index[(df.txn_day >= core.ES_MAX_DAY) & (df.txn_day < core.CAL_MAX_DAY)]
    te = df.index[df.txn_day >= core.CAL_MAX_DAY]
    parts = (X.loc[tr], y.loc[tr], X.loc[es], y.loc[es], X.loc[ca], y.loc[ca],
             X.loc[te], y.loc[te])
    amt = df.loc[te, "transaction_amt"].to_numpy().astype(float)
    yte = y.loc[te].to_numpy().astype(int)
    del X

    R("# Feature pruning and V-block ablation"); R()
    R("Splits held fixed across every variant. Same seed, same hyperparameters, "
      "same calibration. Only the feature set changes."); R()

    def price(r):
        t, c, _ = core.sweep(r["_p"], yte, amt)
        r["threshold"], r["cost"] = t, c
        r["false_decline_rate"] = core.confusion(t, r["_p"], yte)["false_decline_rate"]
        r["gap"] = r["train_pr_auc"] - r["test_pr_auc_raw"]
        return r

    print("training full 437...")
    full = price(_fit_variant("full 437", feats, cats, parts))
    imp = pd.DataFrame({"feature": feats,
                        "gain": full["_clf"].booster_.feature_importance("gain")})
    imp = imp.sort_values("gain", ascending=False).reset_index(drop=True)
    imp["pct_of_total_gain"] = imp.gain / imp.gain.sum() * 100
    imp["cum_pct"] = imp.pct_of_total_gain.cumsum()
    imp.to_csv(REPORTS / "feature_gain_ranking.csv", index=False)

    R("## Gain ranking"); R()
    nz = int((imp.gain > 0).sum())
    R(f"{nz} of {len(feats)} features have non-zero gain; {len(feats)-nz} were "
      "never used for a split."); R()
    for k in (50, 100, 200):
        R(f"- top {k} carry **{imp.cum_pct.iloc[k-1]:.2f}%** of total gain")
    R(); R("Top 20:"); R(); R.table(imp.head(20).round(4)); R()

    results = [full]
    for k in (200, 100, 50):
        print(f"training top {k}...")
        results.append(price(_fit_variant(f"top {k}", imp.feature.head(k).tolist(), cats, parts)))
    no_v = [f for f in feats if not (f.startswith("v") and f[1:].isdigit())]
    print(f"training without V block ({len(feats)-len(no_v)} removed)...")
    novv = price(_fit_variant(f"no V block ({len(no_v)})", no_v, cats, parts))

    R("## Pruning results"); R()
    tbl = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in results])
    tbl["cost_vs_full"] = (tbl.cost - full["cost"]).round(2)
    R.table(tbl.round(6)); R()

    R("### Cost differences with confidence intervals"); R()
    R("Paired bootstrap, 600 resamples. Negative means cheaper than full 437. An "
      "interval spanning zero is not distinguishable from noise."); R()
    rows = []
    for r in results[1:]:
        m, lo, hi = core.boot_cost_diff(r["_p"], r["threshold"], full["_p"],
                                        full["threshold"], yte, amt)
        rows.append({"variant": r["variant"], "point_diff_usd": round(r["cost"] - full["cost"], 2),
                     "boot_mean_usd": round(m, 2), "ci_low_usd": round(lo, 2),
                     "ci_high_usd": round(hi, 2), "significant": "no" if lo <= 0 <= hi else "yes"})
    R.table(pd.DataFrame(rows)); R()

    R("## V-block ablation"); R()
    ab = pd.DataFrame([
        {"variant": "with V block", "features": full["n_features"],
         "test_pr_auc": round(full["test_pr_auc_raw"], 6), "gap": round(full["gap"], 6),
         "fit_secs": full["fit_secs"], "cost_usd": round(full["cost"], 2)},
        {"variant": "without V block", "features": novv["n_features"],
         "test_pr_auc": round(novv["test_pr_auc_raw"], 6), "gap": round(novv["gap"], 6),
         "fit_secs": novv["fit_secs"], "cost_usd": round(novv["cost"], 2)}])
    R.table(ab); R()
    m, lo, hi = core.boot_cost_diff(full["_p"], full["threshold"], novv["_p"],
                                    novv["threshold"], yte, amt)
    R(f"Bootstrap on cost, with V minus without V: mean **${m:,.2f}**, 95% CI "
      f"[${lo:,.2f}, ${hi:,.2f}]. "
      f"{'Distinguishable from noise.' if not (lo <= 0 <= hi) else '**Spans zero.**'}"); R()
    R(f"In isolation the 339 V columns move test PR-AUC by "
      f"**{full['test_pr_auc_raw']-novv['test_pr_auc_raw']:+.4f}**."); R()

    R("## Decision"); R()
    best = min(results, key=lambda r: r["cost"])
    R(f"Cheapest on the point estimate: **{best['variant']}** at ${best['cost']:,.2f} "
      f"against ${full['cost']:,.2f} for the full model."); R()
    if best["variant"] != "full 437":
        _, lo2, hi2 = core.boot_cost_diff(best["_p"], best["threshold"], full["_p"],
                                          full["threshold"], yte, amt)
        R(f"The saving is {'significant' if hi2 < 0 else '**not** significant'} "
          f"(CI [${lo2:,.2f}, ${hi2:,.2f}]). Adopted on parsimony: no worse on "
          f"cost, {best['n_features']} features against {full['n_features']}, "
          f"{best['fit_secs']}s against {full['fit_secs']}s."); R()
        (MODELS / "selected_features.json").write_text(json.dumps(
            {"n_features": best["n_features"], "variant": best["variant"],
             "selected_from": "gain ranking of the full 437 model",
             "features": best["_cols"]}, indent=2))
        R("Written to `models/selected_features.json`.")
    else:
        R("No pruned variant matched or beat the full model, so all 437 are kept.")
    R(); R.write()


# --------------------------------------------------------------------------- #
# model improvements
# --------------------------------------------------------------------------- #

def _count_encoder(min_card=20):
    def transform(X, pre_mask):
        X = X.copy()
        for c in list(X.columns):
            if X[c].dtype == object or str(X[c].dtype) == "string":
                if X.loc[pre_mask, c].nunique(dropna=True) >= min_card:
                    X[c + "_cnt"] = X[c].map(X.loc[pre_mask, c].value_counts()).astype("float32")
        return X
    return transform


def improve():
    R = Report(REPORTS / "model_improvements.md")
    sel, fct = load_selected()
    y = fct.is_fraud.to_numpy().astype(int)
    day, amt = fct.txn_day.to_numpy(), fct.transaction_amt.to_numpy().astype(float)
    X0 = fct[sel].copy()

    Xd = X0.copy()
    n_d = 0
    for c in D_NORM:
        if c in Xd.columns:
            Xd[c] = fct.txn_day.to_numpy() - pd.to_numeric(Xd[c], errors="coerce")
            n_d += 1

    R("# Model improvements"); R()
    R("Four non-overlapping 16-day test windows, days 118-181. Threshold chosen "
      "out of sample. Identical folds for every variant."); R()
    pre = day < 118
    ce_cols = [c for c in X0.columns if (X0[c].dtype == object)
               and X0.loc[pre, c].nunique(dropna=True) >= 20]
    R(f"Intervention 1 normalises **{n_d}** D columns (D9 excluded: it is a "
      "time-of-day fraction, not a day offset)."); R()
    R(f"Intervention 2 count-encodes **{len(ce_cols)}** high-cardinality "
      f"categoricals."); R()
    R(f"Intervention 3 parameters: {REG_PARAMS}."); R()

    ce = _count_encoder()
    variants = [("A baseline", X0, None, None), ("B D-normalised", Xd, None, None),
                ("C count-encoded", X0, None, ce), ("D regularised", X0, REG_PARAMS, None),
                ("BC D-norm + count", Xd, None, ce), ("BD D-norm + reg", Xd, REG_PARAMS, None),
                ("CD count + reg", X0, REG_PARAMS, ce), ("BCD all three", Xd, REG_PARAMS, ce)]
    store, frames = {}, []
    for name, X, params, tf in variants:
        print(f"\n=== {name} ===")
        df = core.run_config(name, X, y, day, amt, params=params, transform=tf)
        store[name] = df; frames.append(df)
    res = pd.concat(frames, ignore_index=True)
    res.to_csv(REPORTS / "model_improvement_folds.csv", index=False)

    R("## Per-fold results"); R()
    R.table(res[["config", "fold_test_start", "test_pr_auc", "train_pr_auc",
                 "threshold", "cost", "false_decline_rate", "fit_secs"]].round(5)); R()
    agg = res.groupby("config").agg(
        pr_auc_mean=("test_pr_auc", "mean"), pr_auc_sd=("test_pr_auc", "std"),
        roc_auc_mean=("test_roc_auc", "mean"), train_pr_auc=("train_pr_auc", "mean"),
        brier=("test_brier_cal", "mean"), cost_total=("cost", "sum"),
        cost_sd=("cost", "std"), fdr=("false_decline_rate", "mean"),
        caught=("fraud_dollars_caught", "sum"), fraud_total=("fraud_dollars_total", "sum"),
        fit_secs=("fit_secs", "mean"))
    agg["gap"] = agg.train_pr_auc - agg.pr_auc_mean
    agg["pct_fraud_caught"] = agg.caught / agg.fraud_total * 100
    agg = agg[["pr_auc_mean", "pr_auc_sd", "roc_auc_mean", "gap", "brier", "cost_total",
               "cost_sd", "fdr", "pct_fraud_caught", "fit_secs"]].sort_values("cost_total")
    R("## Summary, sorted by total cost"); R(); R.table(agg.round(5), index=True); R()

    R("## Paired against baseline, by fold"); R()
    rows = []
    for name, df in store.items():
        if name == "A baseline":
            continue
        c = core.paired_fold_test(df, store["A baseline"], "cost")
        p = core.paired_fold_test(df, store["A baseline"], "test_pr_auc")
        g = core.paired_fold_test(df, store["A baseline"], "train_pr_auc")
        rows.append({"variant": name, "cost_diff_total": round(c["mean_diff"] * c["n_folds"], 0),
                     "cost_t": round(c["t"], 2), "folds_cheaper": int((c["per_fold"] < 0).sum()),
                     "pr_auc_diff": round(p["mean_diff"], 5), "pr_auc_t": round(p["t"], 2),
                     "folds_better_pr": int((p["per_fold"] > 0).sum()),
                     "gap_change": round(g["mean_diff"] - p["mean_diff"], 5)})
    R.table(pd.DataFrame(rows).sort_values("cost_diff_total")); R()
    R("Counts are out of 4. With four folds a t statistic is indicative; "
      "consistency across folds is the more trustworthy signal."); R()
    R.write()


# --------------------------------------------------------------------------- #
# split comparison
# --------------------------------------------------------------------------- #

def split():
    from sklearn.linear_model import LogisticRegression
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.frozen import FrozenEstimator
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
    import lightgbm as lgb

    R = Report(REPORTS / "split_comparison.md")
    N_BOOT = 1000
    sel, fct = load_selected()
    fct = pd.read_parquet(core.FCT, columns=sorted(set(sel) | {"txn_day", "is_fraud", "account_id"}))
    Xraw = fct[sel]
    X, cats = core.encode(Xraw)
    y, day, n = fct.is_fraud.to_numpy().astype(int), fct.txn_day.to_numpy(), len(X)

    tr_t = day < 110
    es_t = (day >= 110) & (day < 130)
    ca_t = (day >= 130) & (day < 150)
    te_t = day >= 150
    props = np.array([tr_t.sum(), es_t.sum(), ca_t.sum(), te_t.sum()]) / n
    p8 = np.array([0.644, 0.105, 0.091]); p8 = p8 / p8.sum() * 0.8
    props80 = np.array([p8[0], p8[1], p8[2], 0.20])

    def masks(p, seed):
        u = np.random.default_rng(seed).permutation(n)
        cuts, out, prev = (np.cumsum(p) * n).astype(int), [], 0
        for c in cuts:
            m = np.zeros(n, bool); m[u[prev:c]] = True
            out.append(m); prev = c
        return out

    def logit_matrix(tr):
        parts = []
        for c in Xraw.columns:
            s = Xraw[c]
            if s.dtype == object or str(s.dtype) == "string":
                if s[tr].nunique(dropna=True) <= 30:
                    parts.append(pd.get_dummies(s, prefix=c, dummy_na=True, dtype=np.float32))
                else:
                    parts.append(pd.DataFrame({c + "_cnt": s.map(s[tr].value_counts()).astype(np.float32)}))
            else:
                v = pd.to_numeric(s, errors="coerce").astype(np.float32)
                parts.append(pd.DataFrame({c: v.fillna(v[tr].median())}))
        Z = pd.concat(parts, axis=1)
        return ((Z - Z[tr].mean()) / Z[tr].std().replace(0, 1)).fillna(0.0)

    def fit(tr, es, ca, te, model="lgbm"):
        if model == "lgbm":
            clf = lgb.LGBMClassifier(**core.LGB_PARAMS)
            clf.fit(X[tr], y[tr], eval_set=[(X[es], y[es])], eval_metric="average_precision",
                    categorical_feature=cats,
                    callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
            Xf = X
        else:
            Xf = logit_matrix(tr)
            clf = LogisticRegression(max_iter=3000, C=1.0, solver="lbfgs").fit(Xf[tr], y[tr])
        cal = CalibratedClassifierCV(FrozenEstimator(clf), method="isotonic")
        cal.fit(Xf[ca], y[ca])
        return clf.predict_proba(Xf[te])[:, 1], cal.predict_proba(Xf[te])[:, 1], y[te]

    runs = {}
    print("A time split, LightGBM ..."); runs["A time split, LightGBM"] = fit(tr_t, es_t, ca_t, te_t)
    print("B random size-matched ..."); runs["B random size-matched, LightGBM"] = fit(*masks(props, core.SEED))
    print("C random 80/20 ..."); runs["C random 80/20, LightGBM"] = fit(*masks(props80, core.SEED))
    print("D logistic ..."); runs["D time split, logistic regression"] = fit(tr_t, es_t, ca_t, te_t, "logit")

    def met(praw, pcal, yy):
        return {"pr_auc": average_precision_score(yy, praw),
                "roc_auc": roc_auc_score(yy, praw), "brier": brier_score_loss(yy, pcal)}

    def boot(praw, pcal, yy, seed=7):
        r = np.random.default_rng(seed); out = {"pr_auc": [], "roc_auc": [], "brier": []}
        for _ in range(N_BOOT):
            i = r.integers(0, len(yy), len(yy))
            if yy[i].sum() == 0:
                continue
            out["pr_auc"].append(average_precision_score(yy[i], praw[i]))
            out["roc_auc"].append(roc_auc_score(yy[i], praw[i]))
            out["brier"].append(brier_score_loss(yy[i], pcal[i]))
        return {k: np.array(v) for k, v in out.items()}

    R("# Split comparison and a logistic baseline"); R()
    R("Identical features (200), identical hyperparameters, identical isotonic "
      "calibration. Only the split rule changes between A, B and C. D changes "
      "only the model."); R()
    R(f"Time-split proportions, which run B reproduces exactly: train "
      f"{props[0]:.4f}, early stopping {props[1]:.4f}, calibration {props[2]:.4f}, "
      f"test {props[3]:.4f}."); R()
    R("**The time-split figure remains the number this project reports.**"); R()
    R("## Results"); R()
    R.table(pd.DataFrame([{"run": k, "n_test": len(v[2]), "n_fraud": int(v[2].sum()),
                           "fraud_rate": round(v[2].mean(), 5),
                           **{m: round(x, 4 if m != "brier" else 5) for m, x in met(*v).items()}}
                          for k, v in runs.items()])); R()

    bs = {k: boot(*v) for k, v in runs.items()}
    A = "A time split, LightGBM"
    R("## Bootstrapped differences against the time split"); R()
    R(f"{N_BOOT} resamples. Unpaired, because the random splits use different rows."); R()
    rows = []
    for other in ["B random size-matched, LightGBM", "C random 80/20, LightGBM"]:
        ma, mb = met(*runs[other]), met(*runs[A])
        for k in ("pr_auc", "roc_auc", "brier"):
            lo, hi = np.percentile(bs[other][k] - bs[A][k], [2.5, 97.5])
            rows.append({"comparison": other.split(",")[0] + " minus A", "metric": k,
                         "point": round(ma[k] - mb[k], 4), "ci_low": round(lo, 4),
                         "ci_high": round(hi, 4), "spans_zero": "yes" if lo < 0 < hi else "no"})
    R.table(pd.DataFrame(rows)); R()
    for other, lab in (("B random size-matched, LightGBM", "size-matched"),
                       ("C random 80/20, LightGBM", "80/20")):
        ma, mb = met(*runs[other]), met(*runs[A])
        R(f"- Random {lab}: PR-AUC {mb['pr_auc']:.4f} to {ma['pr_auc']:.4f}, "
          f"**{(ma['pr_auc']/mb['pr_auc']-1)*100:+.1f}%**. ROC-AUC "
          f"{mb['roc_auc']:.4f} to {ma['roc_auc']:.4f}.")
    R()
    ids = fct.account_id
    te_r = masks(props, core.SEED)[3]; tr_r = ~te_r
    R(f"Mechanism: under the random split **{len(set(ids[te_r]) & set(ids[tr_r])):,} "
      f"of {ids[te_r].nunique():,} test accounts "
      f"({len(set(ids[te_r]) & set(ids[tr_r]))/ids[te_r].nunique()*100:.1f}%)** also "
      f"appear in training. Under the time split it is "
      f"{len(set(ids[te_t]) & set(ids[tr_t]))/ids[te_t].nunique()*100:.1f}%, and none "
      "of those are future transactions."); R()

    R("## Logistic baseline, paired bootstrap on identical test rows"); R()
    pg, cg, yg = runs[A]; pl, cl, _ = runs["D time split, logistic regression"]
    r = np.random.default_rng(11); dp = {"pr_auc": [], "roc_auc": [], "brier": []}
    for _ in range(N_BOOT):
        i = r.integers(0, len(yg), len(yg))
        if yg[i].sum() == 0:
            continue
        dp["pr_auc"].append(average_precision_score(yg[i], pg[i]) - average_precision_score(yg[i], pl[i]))
        dp["roc_auc"].append(roc_auc_score(yg[i], pg[i]) - roc_auc_score(yg[i], pl[i]))
        dp["brier"].append(brier_score_loss(yg[i], cg[i]) - brier_score_loss(yg[i], cl[i]))
    mg, ml = met(pg, cg, yg), met(pl, cl, yg)
    R.table(pd.DataFrame([{"comparison": "LightGBM minus logistic", "metric": k,
                           "point": round(mg[k] - ml[k], 4),
                           "ci_low": round(np.percentile(dp[k], 2.5), 4),
                           "ci_high": round(np.percentile(dp[k], 97.5), 4),
                           "spans_zero": "yes" if np.percentile(dp[k], 2.5) < 0 < np.percentile(dp[k], 97.5) else "no"}
                          for k in ("pr_auc", "roc_auc", "brier")])); R()
    R("Logistic regression needs imputation, scaling and explicit encoding that "
      "LightGBM does not. That is inherent to the comparison: needing the "
      "preprocessing is part of the cost of a linear model."); R()

    leak = met(*runs["B random size-matched, LightGBM"])["pr_auc"] - mg["pr_auc"]
    gain = mg["pr_auc"] - ml["pr_auc"]
    R("## The comparison worth noticing"); R()
    R.block(f"leakage from a random split adds   {leak:+.4f} PR-AUC\n"
            f"logistic -> gradient boosting adds {gain:+.4f} PR-AUC"); R()
    R(f"**Switching to a random split buys more apparent performance ({leak:+.4f}) "
      f"than the entire jump from logistic regression to a tuned gradient "
      f"boosting model ({gain:+.4f}).** One is a modelling result; the other is "
      "measurement error."); R()
    R.write()


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "cost"
    {"cost": cost, "proxy": proxy, "prune": prune, "improve": improve, "split": split}[cmd]()
