"""Rank candidate account proxies on the walk-forward harness."""

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

buf = []
def emit(s=""):
    print(s); buf.append(s)
def block(s):
    emit("```"); emit(s); emit("```")


def main():
    sel = json.load(open("models/selected_features.json"))["features"]
    need = sorted(set(sel) | {"transaction_id", "txn_day", "is_fraud", "transaction_amt"})
    fct = pd.read_parquet("data/parquet/fct_transactions.parquet", columns=need)
    raw = pd.read_parquet("data/parquet/transactions.parquet",
        columns=["TransactionID", "TransactionDT", "TransactionAmt", "txn_day",
                 "card1", "card2", "card3", "card5", "addr1", "D1", "P_emaildomain"])
    raw = raw.set_index("TransactionID").loc[fct.transaction_id].reset_index()
    assert (raw.TransactionID.to_numpy() == fct.transaction_id.to_numpy()).all()

    y = fct.is_fraud.to_numpy().astype(int)
    day = fct.txn_day.to_numpy()
    amt = fct.transaction_amt.to_numpy().astype(float)

    emit("# Account proxy candidates on the walk-forward harness")
    emit()
    emit("Four non-overlapping 16-day test windows (days 118-181). Threshold "
         "chosen out of sample on cross-fitted calibration probabilities, then "
         "applied to test. Everything except the eight behavioural features is "
         "held identical across candidates.")
    emit()

    all_rows, per_cand = [], {}
    for name, fn in fl.CANDIDATES.items():
        print(f"\n=== {name} ===")
        blk = fl.build_proxy_block(raw, fn)
        X = fct[sel].copy()
        for c in SWAP:
            X[c] = blk[c].to_numpy()
        df, _ = wf.run_config(name, X, y, day, amt)
        per_cand[name] = df
        all_rows.append(df)
        del X, blk

    res = pd.concat(all_rows, ignore_index=True)
    res.to_csv(REPORTS / "proxy_walkforward_folds.csv", index=False)

    emit("## Per-fold results")
    emit()
    block(res[["config", "fold_test_start", "n_test", "n_fraud_test",
               "test_pr_auc", "threshold", "cost", "cost_at_half",
               "false_decline_rate"]].round(5).to_string(index=False))
    emit()

    emit("## Summary across the four folds")
    emit()
    agg = res.groupby("config").agg(
        pr_auc_mean=("test_pr_auc", "mean"), pr_auc_sd=("test_pr_auc", "std"),
        roc_auc_mean=("test_roc_auc", "mean"),
        gap_mean=("train_pr_auc", "mean"),
        cost_total=("cost", "sum"), cost_mean=("cost", "mean"),
        cost_sd=("cost", "std"),
        fdr_mean=("false_decline_rate", "mean"),
        caught=("fraud_dollars_caught", "sum"),
        fraud_total=("fraud_dollars_total", "sum"),
        fit_secs=("fit_secs", "mean"))
    agg["gap_mean"] = agg.gap_mean - agg.pr_auc_mean
    agg["pct_fraud_caught"] = agg.caught / agg.fraud_total * 100
    agg = agg.sort_values("cost_total")
    block(agg.round(5).to_string())
    emit()

    base = "stage1 (card1+addr1+start)"
    emit("## Paired comparison against the shipped proxy")
    emit()
    emit("Paired by fold, so each candidate is compared to stage 1 on the same "
         "four test windows. Only four pairs, so the t statistic is indicative, "
         "not decisive; the per-fold column is the honest view.")
    emit()
    rows = []
    for name, df in per_cand.items():
        if name == base:
            continue
        r = wf.paired_fold_test(df, per_cand[base], "cost")
        p = wf.paired_fold_test(df, per_cand[base], "test_pr_auc")
        rows.append({"candidate": name,
                     "cost_diff_total": round(r["mean_diff"] * r["n_folds"], 0),
                     "cost_diff_per_fold": round(r["mean_diff"], 0),
                     "cost_sd_across_folds": round(r["sd_diff"], 0),
                     "cost_t": round(r["t"], 2),
                     "folds_cheaper": int((r["per_fold"] < 0).sum()),
                     "pr_auc_diff": round(p["mean_diff"], 5),
                     "pr_auc_t": round(p["t"], 2)})
    cmp = pd.DataFrame(rows).sort_values("cost_diff_total")
    block(cmp.to_string(index=False))
    emit()
    best = agg.index[0]
    emit(f"Cheapest total cost across all four folds: **{best}** at "
         f"${agg.cost_total.iloc[0]:,.0f}, against "
         f"${agg.loc[base, 'cost_total']:,.0f} for the shipped proxy "
         f"(difference ${agg.cost_total.iloc[0] - agg.loc[base,'cost_total']:,.0f}).")
    emit()
    (REPORTS / "proxy_candidates.md").write_text("\n".join(buf) + "\n")
    print(f"\nwrote {REPORTS/'proxy_candidates.md'}")


if __name__ == "__main__":
    main()
