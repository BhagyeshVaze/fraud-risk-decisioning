"""Turn calibrated probabilities into a decline threshold priced in dollars.

Every dollar figure below rests on the cost assumptions in ASSUMPTIONS. They
are assumptions, not measurements: they are plausible mid-market card-not-
present values, not numbers observed in this dataset. IEEE-CIS carries no
chargeback fees, no margin, and no churn data. Change them and the optimal
threshold moves, which is the point of the sensitivity section.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPORTS = Path("reports")
VERIFY = REPORTS / "verification.md"
RESULTS = REPORTS / "stage1_results.md"

# Stage 1 decision-layer results, for the side-by-side.
BASELINE = {
    "threshold": 0.0710,
    "cost_opt": 309028.40,
    "cost_naive": 434126.66,
    "cost_approve_all": 577294.40,
    "cost_decline_all": 1414894.91,
    "fp": 6099, "tp": 2146, "fn": 1136, "tn": 85255,
    "precision": 0.260279, "recall": 0.653870, "fdr": 0.066762,
    "fraud_dollars_caught": 319336.86,
}

ASSUMPTIONS = {
    "chargeback_fee_usd": 25.0,
    "margin_rate": 0.025,
    "churn_prob_after_false_decline": 0.05,
    "account_lifetime_value_usd": 200.0,
    "manual_review_cost_usd": 2.0,
}

# Palette slots 1, 2, 3, documented as validating all-pairs in both modes.
C_CURVE, C_NAIVE, C_OPT = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK_2, GRIDC, SURFACE = "#0b0b0b", "#52514e", "#d9d8d2", "#fcfcfb"

buf = []


def emit(line=""):
    print(line)
    buf.append(line)


def block(text):
    emit("```"); emit(text); emit("```")


def cost_at(threshold, p, y, amt, a=ASSUMPTIONS):
    """Total dollar cost of declining every transaction with p >= threshold.

    fraud approved  -> lose the goods plus a chargeback fee
    legit declined  -> lose the margin, risk churn, and pay for the review
    fraud declined  -> pay for the review only
    legit approved  -> no cost
    """
    declined = p >= threshold
    fraud = y == 1

    fn = declined.__invert__() & fraud          # fraud approved
    fp = declined & ~fraud                      # legit declined
    tp = declined & fraud                       # fraud declined

    fraud_loss = amt[fn].sum() + a["chargeback_fee_usd"] * fn.sum()
    lost_margin = a["margin_rate"] * amt[fp].sum()
    churn = a["churn_prob_after_false_decline"] * a["account_lifetime_value_usd"] * fp.sum()
    review = a["manual_review_cost_usd"] * declined.sum()
    return fraud_loss + lost_margin + churn + review


def confusion(threshold, p, y):
    declined = p >= threshold
    fraud = y == 1
    tp = int((declined & fraud).sum())
    fp = int((declined & ~fraud).sum())
    fn = int((~declined & fraud).sum())
    tn = int((~declined & ~fraud).sum())
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    false_decline_rate = fp / (fp + tn) if fp + tn else float("nan")
    return dict(threshold=threshold, tp=tp, fp=fp, fn=fn, tn=tn,
                precision=precision, recall=recall,
                false_decline_rate=false_decline_rate,
                declined=tp + fp, approved=fn + tn)


def main():
    p = np.load(REPORTS / "_test_probs.npy")
    te = pd.read_parquet(REPORTS / "_test_frame.parquet")
    y = te.is_fraud.to_numpy().astype(int)
    amt = te.transaction_amt.to_numpy().astype(float)

    emit(); emit("## Part 3: the decision layer"); emit()
    emit("### Cost assumptions")
    emit()
    emit("These are **assumptions, not measured facts**. IEEE-CIS contains no "
         "chargeback, margin, or churn data, so these are plausible "
         "mid-market card-not-present values chosen to make the tradeoff "
         "explicit. Every dollar figure below inherits their uncertainty.")
    emit()
    block(json.dumps(ASSUMPTIONS, indent=2))
    emit()
    emit("Cost model, applied per transaction:")
    emit()
    emit("| Outcome | Cost |")
    emit("| --- | --- |")
    emit("| fraud approved (FN) | transaction amount + $25 chargeback fee |")
    emit("| legit declined (FP) | 2.5% margin lost + 5% x $200 churn + $2 review |")
    emit("| fraud declined (TP) | $2 review |")
    emit("| legit approved (TN) | $0 |")
    emit()

    # --- sweep -------------------------------------------------------------
    grid = np.unique(np.concatenate([
        np.linspace(0.001, 0.99, 990),
        np.geomspace(0.001, 0.99, 400)]))
    costs = np.array([cost_at(t, p, y, amt) for t in grid])
    i_opt = int(costs.argmin())
    t_opt, c_opt = float(grid[i_opt]), float(costs[i_opt])

    c_naive = cost_at(0.5, p, y, amt)
    c_approve_all = cost_at(1.01, p, y, amt)
    c_decline_all = cost_at(0.0, p, y, amt)

    total_dollars = amt.sum()
    fraud_dollars = amt[y == 1].sum()

    emit("### Threshold sweep")
    emit()
    emit(f"Swept {len(grid):,} thresholds from 0.001 to 0.99.")
    emit()
    emit(f"**Chosen threshold: {t_opt:.6f}**")
    emit()
    sweep = pd.DataFrame([
        {"policy": "optimum", "threshold": round(t_opt, 6), "total_cost_usd": round(c_opt, 2)},
        {"policy": "naive 0.5", "threshold": 0.5, "total_cost_usd": round(c_naive, 2)},
        {"policy": "approve everything", "threshold": float("inf"), "total_cost_usd": round(c_approve_all, 2)},
        {"policy": "decline everything", "threshold": 0.0, "total_cost_usd": round(c_decline_all, 2)},
    ])
    sweep["vs_optimum_usd"] = (sweep.total_cost_usd - c_opt).round(2)
    block(sweep.to_string(index=False))
    emit()

    emit("### Sensitivity to the threshold")
    emit()
    sens = pd.DataFrame([
        {"threshold": round(t, 6), "label": lab, "total_cost_usd": round(cost_at(t, p, y, amt), 2)}
        for t, lab in ((t_opt * 0.8, "optimum -20%"), (t_opt, "optimum"),
                       (t_opt * 1.2, "optimum +20%"))])
    sens["delta_vs_optimum_usd"] = (sens.total_cost_usd - c_opt).round(2)
    sens["delta_pct"] = (sens.delta_vs_optimum_usd / c_opt * 100).round(3)
    block(sens.to_string(index=False))
    emit()
    worst = sens.delta_pct.abs().max()
    emit(f"Moving the threshold 20% in either direction changes total cost by "
         f"at most {worst:.3f}%. The optimum is {'flat' if worst < 2 else 'sharp'}, "
         f"so {'small errors in the cost assumptions do not move the answer much' if worst < 2 else 'the threshold needs care'}.")
    emit()

    # --- confusion ----------------------------------------------------------
    emit("### Confusion matrices, raw counts")
    emit()
    o_ = confusion(t_opt, p, y)
    cm = pd.DataFrame([o_, confusion(0.5, p, y)])
    cm.insert(0, "policy", ["optimum", "naive 0.5"])
    block(cm.round(6).to_string(index=False))
    emit()

    # --- dollars -------------------------------------------------------------
    emit("### Fraud dollars")
    emit()
    dec_opt = p >= t_opt
    caught_opt = amt[dec_opt & (y == 1)].sum()
    dec_naive = p >= 0.5
    caught_naive = amt[dec_naive & (y == 1)].sum()
    dollars = pd.DataFrame([
        {"metric": "total transaction dollars in test", "usd": round(total_dollars, 2)},
        {"metric": "total fraud dollars in test", "usd": round(fraud_dollars, 2)},
        {"metric": "fraud dollars caught at optimum", "usd": round(caught_opt, 2)},
        {"metric": "fraud dollars caught at 0.5", "usd": round(caught_naive, 2)},
    ])
    dollars["pct_of_fraud_dollars"] = (dollars.usd / fraud_dollars * 100).round(3)
    block(dollars.to_string(index=False))
    emit()
    emit(f"Share of fraud dollars caught at the chosen threshold: "
         f"**{caught_opt / fraud_dollars * 100:.2f}%**")
    emit(f"Fraud dollars as a share of all test dollars: "
         f"{fraud_dollars / total_dollars * 100:.2f}%")
    emit()

    # --- extremes reconciliation -------------------------------------------
    emit("### Reconciling the extremes")
    emit()
    emit("Component arithmetic for the two boundary policies, so the cost "
         "function can be checked by hand at the extremes.")
    emit()
    n_all = len(y)
    n_fraud = int((y == 1).sum())
    n_legit = n_all - n_fraud
    legit_dollars = amt[y == 0].sum()
    a = ASSUMPTIONS

    dm = a["margin_rate"] * legit_dollars
    dc = a["churn_prob_after_false_decline"] * a["account_lifetime_value_usd"] * n_legit
    dr = a["manual_review_cost_usd"] * n_all
    emit("**Decline everything** (threshold 0.0): FN = 0, TP = all "
         f"{n_fraud:,} frauds, FP = all {n_legit:,} legitimate.")
    emit()
    block(
        f"lost margin   0.025 x ${legit_dollars:,.2f}   = ${dm:>13,.2f}\n"
        f"churn         0.05 x $200 x {n_legit:,}       = ${dc:>13,.2f}\n"
        f"review        $2 x {n_all:,} declines         = ${dr:>13,.2f}\n"
        f"fraud loss    every fraud declined            = ${0:>13,.2f}\n"
        f"{'-'*54}\n"
        f"TOTAL                                          = ${dm+dc+dr:>13,.2f}\n"
        f"computed by cost_at(0.0)                       = ${c_decline_all:>13,.2f}")
    emit()
    emit(f"Margin plus churn alone is ${dm+dc:,.2f}. The remaining "
         f"${dr:,.2f} is the review term: declining everything means reviewing "
         f"all {n_all:,} transactions, fraudulent and legitimate alike. That "
         "term is the difference between the two figures.")
    emit()

    am = amt[y == 1].sum()
    ac = a["chargeback_fee_usd"] * n_fraud
    emit("**Approve everything** (threshold above 1): TP = FP = 0, "
         f"FN = all {n_fraud:,} frauds.")
    emit()
    block(
        f"fraud loss    ${am:,.2f} of goods            = ${am:>13,.2f}\n"
        f"chargebacks   $25 x {n_fraud:,}                  = ${ac:>13,.2f}\n"
        f"review        $2 x 0 declines                 = ${0:>13,.2f}\n"
        f"{'-'*54}\n"
        f"TOTAL                                          = ${am+ac:>13,.2f}\n"
        f"computed by cost_at(1.01)                      = ${c_approve_all:>13,.2f}")
    emit()

    # --- stage 1 comparison ---------------------------------------------------
    emit("### Stage 1 vs this run, decision layer")
    emit()
    cmp = pd.DataFrame([
        {"metric": "chosen threshold", "stage_1": BASELINE["threshold"], "this_run": round(t_opt, 6)},
        {"metric": "total cost at optimum", "stage_1": BASELINE["cost_opt"], "this_run": round(c_opt, 2)},
        {"metric": "total cost at 0.5", "stage_1": BASELINE["cost_naive"], "this_run": round(c_naive, 2)},
        {"metric": "saving vs 0.5", "stage_1": round(BASELINE["cost_naive"] - BASELINE["cost_opt"], 2), "this_run": round(c_naive - c_opt, 2)},
        {"metric": "saving vs approve-all", "stage_1": round(BASELINE["cost_approve_all"] - BASELINE["cost_opt"], 2), "this_run": round(c_approve_all - c_opt, 2)},
        {"metric": "false decline rate", "stage_1": BASELINE["fdr"], "this_run": round(o_["false_decline_rate"], 6)},
        {"metric": "precision", "stage_1": BASELINE["precision"], "this_run": round(o_["precision"], 6)},
        {"metric": "recall", "stage_1": BASELINE["recall"], "this_run": round(o_["recall"], 6)},
        {"metric": "fraud dollars caught", "stage_1": BASELINE["fraud_dollars_caught"], "this_run": round(caught_opt, 2)},
    ])
    cmp["delta"] = (cmp.this_run - cmp.stage_1).round(6)
    block(cmp.to_string(index=False))
    emit()

    # --- plot ----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.4, 5.4), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.plot(grid, costs / 1000, color=C_CURVE, lw=2, zorder=3)
    # The 0.5 marker sits near the right edge, so its label is anchored
    # inward to avoid overflowing the axes.
    for t, c, color, lab, dx, ha in (
            (t_opt, c_opt, C_OPT, f"optimum {t_opt:.3f}", 8, "left"),
            (0.5, c_naive, C_NAIVE, "naive 0.5", -8, "right")):
        ax.axvline(t, color=color, lw=2, ls="--", zorder=2)
        ax.plot([t], [c / 1000], marker="o", ms=9, color=color,
                markeredgecolor=SURFACE, markeredgewidth=2, zorder=4)
        ax.annotate(f"{lab}\n${c/1000:,.1f}k", xy=(t, c / 1000),
                    xytext=(dx, 16), textcoords="offset points", ha=ha,
                    color=INK_2, fontsize=9, weight="bold")
    ax.set_xscale("log")
    ax.set_xlabel("decline threshold (calibrated probability, log scale)",
                  color=INK_2, fontsize=10)
    ax.set_ylabel("total cost on test split (thousands USD)", color=INK_2, fontsize=10)
    ax.set_title("Cost against decline threshold", color=INK, fontsize=12,
                 weight="bold", loc="left", pad=12)
    ax.grid(True, color=GRIDC, lw=0.8, alpha=0.7, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRIDC)
    ax.tick_params(colors=INK_2, labelsize=9)
    fig.tight_layout()
    fig.savefig(REPORTS / "cost_vs_threshold.png", dpi=170, facecolor=SURFACE)
    emit("Cost curve: `reports/cost_vs_threshold.png`.")
    emit()

    with open(VERIFY, "a") as fh:
        fh.write("\n".join(buf) + "\n")
    print(f"appended Part 3 to {VERIFY}")

    # --- results doc ----------------------------------------------------------
    o = confusion(t_opt, p, y)
    n = confusion(0.5, p, y)
    meta = json.loads(Path("models/model_metadata.json").read_text())
    m = meta["metrics"]
    RESULTS.write_text(f"""# Stage 1 results

Fraud risk decisioning on IEEE-CIS, evaluated on a held-out time split
(txn_day >= 150, {len(te):,} transactions, {int(y.sum()):,} fraudulent).

## Headline

At a decline threshold of **{t_opt:.4f}**, the model prevents
**${caught_opt:,.0f}** of the **${fraud_dollars:,.0f}** of fraud in the test
split ({caught_opt/fraud_dollars*100:.1f}%), while declining
**{o['fp']:,}** legitimate transactions
({o['false_decline_rate']*100:.2f}% of all legitimate traffic).

Total modelled cost is **${c_opt:,.0f}**, against **${c_naive:,.0f}** at the
naive 0.5 cutoff and **${c_approve_all:,.0f}** if every transaction is approved.

| Policy | Total cost | Saving vs this policy |
| --- | --- | --- |
| Chosen threshold {t_opt:.4f} | ${c_opt:,.0f} | - |
| Naive 0.5 cutoff | ${c_naive:,.0f} | ${c_naive-c_opt:,.0f} |
| Approve everything | ${c_approve_all:,.0f} | ${c_approve_all-c_opt:,.0f} |
| Decline everything | ${c_decline_all:,.0f} | ${c_decline_all-c_opt:,.0f} |

## Decision quality

| Metric | At {t_opt:.4f} | At 0.50 |
| --- | --- | --- |
| True positives (fraud caught) | {o['tp']:,} | {n['tp']:,} |
| False positives (legit declined) | {o['fp']:,} | {n['fp']:,} |
| False negatives (fraud missed) | {o['fn']:,} | {n['fn']:,} |
| True negatives | {o['tn']:,} | {n['tn']:,} |
| Precision | {o['precision']:.4f} | {n['precision']:.4f} |
| Recall | {o['recall']:.4f} | {n['recall']:.4f} |
| False decline rate | {o['false_decline_rate']*100:.3f}% | {n['false_decline_rate']*100:.3f}% |

## Model quality

| Metric | Value |
| --- | --- |
| PR-AUC (test, calibrated) | {m['test_pr_auc_cal']:.4f} |
| PR-AUC (test, raw) | {m['test_pr_auc_raw']:.4f} |
| ROC-AUC (test) | {m['test_roc_auc_cal']:.4f} |
| Brier (test, raw) | {m['test_brier_raw']:.6f} |
| Brier (test, calibrated) | {m['test_brier_cal']:.6f} |

Isotonic calibration did not improve the Brier score on the test split. The
calibrator is fitted on days 120-149 and applied to days 150+, and the same
window is used for early stopping. See `verification.md` for the decile tables.

## Cost assumptions

Stated as assumptions, not measurements. IEEE-CIS has no chargeback, margin,
or churn data.

| Assumption | Value |
| --- | --- |
| Chargeback fee | $25.00 |
| Margin rate | 2.5% |
| Churn probability after a false decline | 5% |
| Account lifetime value | $200.00 |
| Manual review cost | $2.00 |

Cost model: fraud approved costs the transaction amount plus the chargeback
fee; a legitimate transaction declined costs the lost margin plus the churn-
weighted lifetime value plus a review; fraud declined costs a review.

Sensitivity: moving the threshold 20% either way changes total cost by at most
{worst:.3f}%, so the optimum is {'flat and robust to modest errors in these assumptions' if worst < 2 else 'sharp and sensitive to these assumptions'}.

Charts: `cost_vs_threshold.png`, `calibration_curve.png`.
""")
    print(f"wrote {RESULTS}")


if __name__ == "__main__":
    main()
