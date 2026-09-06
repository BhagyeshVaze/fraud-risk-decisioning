"""Three model interventions, tested independently then in combination.

Each is motivated by a specific measured failure of the current model, not by
a generic checklist. All are evaluated on the same walk-forward harness with
the same folds, so differences are attributable to the intervention.

INTERVENTION 1: D-column normalisation.
    D1-D15 (excluding D9, which is a time-of-day fraction) are "days since some
    prior event". As raw counters they drift with calendar time, so a split
    learned at day 80 means something different at day 170. Because our
    evaluation is a forward time split, this is a direct cause of the 0.39
    train/test PR-AUC gap. Replacing D with (txn_day - D) converts a moving
    offset into the fixed day the event happened, which is stationary.

INTERVENTION 2: count encoding of high-cardinality categoricals.
    device_info, the email domains and id_31 have hundreds to thousands of
    levels. LightGBM's categorical splitter can isolate rare levels and
    memorise them, which inflates train PR-AUC without generalising. Replacing
    identity with frequency keeps the useful signal (rare values behave
    differently) while removing the ability to memorise a specific level.
    Counts are fitted on pre-test rows only.

INTERVENTION 3: stronger regularisation.
    The current model uses num_leaves 63 and min_child_samples 50 with a 0.39
    gap. Shallower trees, larger leaves, more column subsampling and a higher
    L2 penalty attack the gap directly.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import featurelib as fl
import walkforward as wf

REPORTS = Path("reports")
SWAP = ["prior_txn_count", "prior_amt_sum", "prior_amt_mean", "txn_count_1h",
        "txn_count_24h", "txn_count_7d", "amt_to_prior_mean_ratio",
        "seconds_since_prev_txn"]
D_NORM = [f"d{i}" for i in list(range(1, 9)) + list(range(10, 16))]  # D9 excluded

REG_PARAMS = dict(num_leaves=31, min_child_samples=200, colsample_bytree=0.5,
                  subsample=0.7, reg_lambda=20.0, reg_alpha=1.0,
                  learning_rate=0.03)

buf = []
def emit(s=""):
    print(s); buf.append(s)
def block(s):
    emit("```"); emit(s); emit("```")


def apply_dnorm(X, txn_day):
    X = X.copy()
    n = 0
    for c in D_NORM:
        if c in X.columns:
            X[c] = txn_day - pd.to_numeric(X[c], errors="coerce")
            n += 1
    return X, n


def make_count_encoder(min_card=20):
    def transform(X, pre_mask):
        X = X.copy()
        for c in list(X.columns):
            if X[c].dtype == object or str(X[c].dtype) == "string":
                if X.loc[pre_mask, c].nunique(dropna=True) >= min_card:
                    vc = X.loc[pre_mask, c].value_counts()
                    X[c + "_cnt"] = X[c].map(vc).astype("float32")
        return X
    return transform


def count_encoded_cols(X, pre_mask, min_card=20):
    return [c for c in X.columns
            if (X[c].dtype == object or str(X[c].dtype) == "string")
            and X.loc[pre_mask, c].nunique(dropna=True) >= min_card]


def main(proxy_name=None):
    sel = json.load(open("models/selected_features.json"))["features"]
    need = sorted(set(sel) | {"transaction_id", "txn_day", "is_fraud", "transaction_amt"})
    fct = pd.read_parquet("data/parquet/fct_transactions.parquet", columns=need)

    y = fct.is_fraud.to_numpy().astype(int)
    day = fct.txn_day.to_numpy()
    amt = fct.transaction_amt.to_numpy().astype(float)
    X0 = fct[sel].copy()

    emit("# Model improvements")
    emit()
    if proxy_name:
        raw = pd.read_parquet("data/parquet/transactions.parquet",
            columns=["TransactionID", "TransactionDT", "TransactionAmt", "txn_day",
                     "card1", "card2", "card3", "card5", "addr1", "D1", "P_emaildomain"])
        raw = raw.set_index("TransactionID").loc[fct.transaction_id].reset_index()
        blk = fl.build_proxy_block(raw, fl.CANDIDATES[proxy_name])
        for c in SWAP:
            X0[c] = blk[c].to_numpy()
        emit(f"Built on the **{proxy_name}** account proxy, carried forward from "
             "the proxy comparison.")
        emit()

    emit("Four non-overlapping 16-day test windows, days 118-181. Threshold "
         "chosen out of sample. Identical folds for every variant.")
    emit()

    Xd, n_d = apply_dnorm(X0, fct.txn_day.to_numpy())
    pre_all = day < 118
    ce_cols = count_encoded_cols(X0, pre_all)
    emit(f"Intervention 1 normalises **{n_d}** D columns present in the feature set.")
    emit(f"Intervention 2 count-encodes **{len(ce_cols)}** high-cardinality "
         f"categoricals: {', '.join(ce_cols)}.")
    emit(f"Intervention 3 parameters: {REG_PARAMS}.")
    emit()

    ce = make_count_encoder()
    variants = [
        ("A baseline", X0, None, None),
        ("B D-normalised", Xd, None, None),
        ("C count-encoded", X0, None, ce),
        ("D regularised", X0, REG_PARAMS, None),
        ("BC D-norm + count", Xd, None, ce),
        ("BD D-norm + reg", Xd, REG_PARAMS, None),
        ("CD count + reg", X0, REG_PARAMS, ce),
        ("BCD all three", Xd, REG_PARAMS, ce),
    ]

    frames, store = [], {}
    for name, X, params, tf in variants:
        print(f"\n=== {name} ===")
        df, _ = wf.run_config(name, X, y, day, amt, params=params, transform=tf)
        store[name] = df
        frames.append(df)

    res = pd.concat(frames, ignore_index=True)
    res.to_csv(REPORTS / "model_improvement_folds.csv", index=False)

    emit("## Per-fold results")
    emit()
    block(res[["config", "fold_test_start", "test_pr_auc", "train_pr_auc",
               "threshold", "cost", "false_decline_rate", "fit_secs"]]
          .round(5).to_string(index=False))
    emit()

    agg = res.groupby("config").agg(
        pr_auc_mean=("test_pr_auc", "mean"), pr_auc_sd=("test_pr_auc", "std"),
        roc_auc_mean=("test_roc_auc", "mean"),
        train_pr_auc=("train_pr_auc", "mean"),
        brier=("test_brier_cal", "mean"),
        cost_total=("cost", "sum"), cost_sd=("cost", "std"),
        fdr=("false_decline_rate", "mean"),
        caught=("fraud_dollars_caught", "sum"),
        fraud_total=("fraud_dollars_total", "sum"),
        fit_secs=("fit_secs", "mean"))
    agg["gap"] = agg.train_pr_auc - agg.pr_auc_mean
    agg["pct_fraud_caught"] = agg.caught / agg.fraud_total * 100
    agg = agg[["pr_auc_mean", "pr_auc_sd", "roc_auc_mean", "gap", "brier",
               "cost_total", "cost_sd", "fdr", "pct_fraud_caught", "fit_secs"]]
    agg = agg.sort_values("cost_total")

    emit("## Summary, sorted by total cost across the four folds")
    emit()
    block(agg.round(5).to_string())
    emit()

    base = "A baseline"
    emit("## Paired against baseline, by fold")
    emit()
    rows = []
    for name, df in store.items():
        if name == base:
            continue
        c = wf.paired_fold_test(df, store[base], "cost")
        p = wf.paired_fold_test(df, store[base], "test_pr_auc")
        g = wf.paired_fold_test(df, store[base], "train_pr_auc")
        rows.append({"variant": name,
                     "cost_diff_total": round(c["mean_diff"] * c["n_folds"], 0),
                     "cost_t": round(c["t"], 2),
                     "folds_cheaper": int((c["per_fold"] < 0).sum()),
                     "pr_auc_diff": round(p["mean_diff"], 5),
                     "pr_auc_t": round(p["t"], 2),
                     "folds_better_pr": int((p["per_fold"] > 0).sum()),
                     "gap_change": round(g["mean_diff"] - p["mean_diff"], 5)})
    cmp = pd.DataFrame(rows).sort_values("cost_diff_total")
    block(cmp.to_string(index=False))
    emit()
    emit("`folds_cheaper` and `folds_better_pr` are out of 4. With only four "
         "folds a t statistic is indicative; consistency across folds is the "
         "more trustworthy signal.")
    emit()
    (REPORTS / "model_improvements.md").write_text("\n".join(buf) + "\n")
    print(f"\nwrote {REPORTS/'model_improvements.md'}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
