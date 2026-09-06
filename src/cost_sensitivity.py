"""How much of the stage 1 answer is driven by the invented cost constants.

The headline threshold and dollar figure are a function of five numbers nobody
measured. This sweeps the two that dominate, churn probability and account
lifetime value, and reports how the optimal threshold and the saving move.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from decision_layer import ASSUMPTIONS as BASE

REPORTS = Path("reports")
GRID = np.unique(np.concatenate([np.linspace(0.001, 0.99, 990),
                                 np.geomspace(0.001, 0.99, 400)]))
INK, INK_2, GRIDC, SURFACE = "#0b0b0b", "#52514e", "#d9d8d2", "#fcfcfb"

buf = []


def emit(s=""):
    print(s); buf.append(s)


def block(s):
    emit("```"); emit(s); emit("```")


def cost(mask, y, amt, a):
    fn = ~mask & (y == 1); fp = mask & (y == 0)
    return (amt[fn].sum() + a["chargeback_fee_usd"] * fn.sum()
            + a["margin_rate"] * amt[fp].sum()
            + a["churn_prob_after_false_decline"] * a["account_lifetime_value_usd"] * fp.sum()
            + a["manual_review_cost_usd"] * mask.sum())


def optimum(p, y, amt, a):
    c = np.array([cost(p >= t, y, amt, a) for t in GRID])
    i = int(c.argmin())
    return float(GRID[i]), float(c[i])


def main():
    p = np.load(REPORTS / "_test_probs.npy")
    te = pd.read_parquet(REPORTS / "_test_frame.parquet")
    y = te.is_fraud.to_numpy().astype(int)
    amt = te.transaction_amt.to_numpy().astype(float)

    emit("# Cost assumption sensitivity")
    emit()
    emit("Every dollar figure in this project rests on five constants that were "
         "chosen, not measured. This asks a narrow question: **if those "
         "constants are wrong, does the recommendation change?**")
    emit()
    emit("Two of the five dominate. Churn probability and account lifetime "
         "value enter only as their product, the expected cost of a false "
         "decline, which is $10.00 at the stated 5% and $200. That term "
         "generates $913,540 of the $1,414,895 decline-everything cost, so it "
         "is the assumption worth stressing.")
    emit()
    t0, c0 = optimum(p, y, amt, BASE)
    emit(f"Baseline: churn 5%, LTV $200, false-decline cost $10.00, "
         f"optimal threshold **{t0:.4f}**, cost **${c0:,.0f}**.")
    emit()

    # ---- 1D sweep on the product -----------------------------------------
    emit("## Sweeping the cost of a false decline")
    emit()
    rows = []
    for fd in [0.5, 1, 2, 4, 6, 8, 10, 15, 20, 30, 50, 80, 120]:
        a = dict(BASE, churn_prob_after_false_decline=fd / BASE["account_lifetime_value_usd"])
        t, c = optimum(p, y, amt, a)
        base_c = cost(p >= t0, y, amt, a)
        naive_c = cost(p >= 0.5, y, amt, a)
        rows.append({"false_decline_cost_usd": fd,
                     "implied_churn_at_200_ltv": round(fd / 200, 4),
                     "optimal_threshold": round(t, 4),
                     "cost_at_own_optimum": round(c, 0),
                     "cost_at_baseline_thr": round(base_c, 0),
                     "regret_usd": round(base_c - c, 0),
                     "saving_vs_0.5": round(naive_c - base_c, 0)})
    sw = pd.DataFrame(rows)
    block(sw.to_string(index=False))
    emit()
    emit("`regret_usd` is what it costs to keep using the baseline threshold "
         f"of {t0:.4f} when the true false-decline cost is the value in that "
         "row. It is the number that matters: a threshold is only wrong if "
         "using it is expensive.")
    emit()
    worst = sw.loc[sw.regret_usd.idxmax()]
    emit(f"Worst regret across the swept range: **${worst.regret_usd:,.0f}** at a "
         f"false-decline cost of ${worst.false_decline_cost_usd:.2f} "
         f"({worst['implied_churn_at_200_ltv']*100:.2f}% churn at $200 LTV).")
    flips = sw[sw["saving_vs_0.5"] <= 0]
    if len(flips):
        emit()
        emit(f"**The recommendation flips** at a false-decline cost of "
             f"${flips.false_decline_cost_usd.min():.2f} or above: past that "
             "point the baseline threshold is no longer better than the naive "
             "0.5 cutoff.")
    else:
        emit()
        emit("Across the entire swept range the baseline threshold beats the "
             "0.5 cutoff. The recommendation to move off 0.5 does not depend "
             "on these constants.")
    emit()

    # ---- 2D surface --------------------------------------------------------
    emit("## Two-dimensional surface, churn probability by lifetime value")
    emit()
    churns = np.array([0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.20])
    ltvs = np.array([50, 100, 150, 200, 300, 500, 800])
    thr_s = np.zeros((len(churns), len(ltvs)))
    reg_s = np.zeros_like(thr_s)
    for i, ch in enumerate(churns):
        for j, lt in enumerate(ltvs):
            a = dict(BASE, churn_prob_after_false_decline=ch,
                     account_lifetime_value_usd=lt)
            t, c = optimum(p, y, amt, a)
            thr_s[i, j] = t
            reg_s[i, j] = cost(p >= t0, y, amt, a) - c
    tt = pd.DataFrame(thr_s, index=[f"churn {c:.0%}" for c in churns],
                      columns=[f"LTV ${l}" for l in ltvs]).round(4)
    rr = pd.DataFrame(reg_s, index=tt.index, columns=tt.columns).round(0)
    emit("Optimal threshold at each combination:")
    emit(); block(tt.to_string()); emit()
    emit(f"Regret in dollars from using the baseline {t0:.4f} threshold instead:")
    emit(); block(rr.to_string()); emit()
    emit(f"Threshold ranges from {thr_s.min():.4f} to {thr_s.max():.4f} across "
         f"the grid, a {thr_s.max()/thr_s.min():.0f}x spread. Regret ranges "
         f"from ${reg_s.min():,.0f} to ${reg_s.max():,.0f}.")
    emit()
    within = (reg_s < 9900).mean() * 100
    emit(f"**{within:.0f}% of the grid has regret below $9,900**, the measured "
         "one-standard-deviation noise on the cost estimate. Within that "
         "region the baseline threshold is indistinguishable from the "
         "assumption-specific optimum, so getting the constants wrong there "
         "costs nothing detectable.")
    emit()

    # ---- plot ----------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), facecolor=SURFACE)
    ax = axes[0]
    ax.set_facecolor(SURFACE)
    ax.plot(sw.false_decline_cost_usd, sw.optimal_threshold, color="#2a78d6",
            lw=2, marker="o", ms=8, markeredgecolor=SURFACE, markeredgewidth=2)
    ax.axvline(10, color="#eb6834", lw=2, ls="--")
    ax.annotate("stated assumption\n$10.00", xy=(10, sw.optimal_threshold.iloc[6]),
                xytext=(-10, 34), textcoords="offset points", ha="right",
                color=INK_2, fontsize=9, weight="bold")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("assumed cost of one false decline (USD, log)", color=INK_2, fontsize=10)
    ax.set_ylabel("optimal decline threshold (log)", color=INK_2, fontsize=10)
    ax.set_title("Optimal threshold against the false-decline cost",
                 color=INK, fontsize=12, weight="bold", loc="left", pad=12)
    ax.grid(True, color=GRIDC, lw=0.8, alpha=0.7)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color(GRIDC)
    ax.tick_params(colors=INK_2, labelsize=9)

    ax = axes[1]
    ax.set_facecolor(SURFACE)
    im = ax.imshow(reg_s / 1000, cmap="YlOrBr", aspect="auto", origin="lower")
    ax.set_xticks(range(len(ltvs))); ax.set_xticklabels([f"${l}" for l in ltvs], fontsize=9)
    ax.set_yticks(range(len(churns))); ax.set_yticklabels([f"{c:.0%}" for c in churns], fontsize=9)
    ax.set_xlabel("account lifetime value", color=INK_2, fontsize=10)
    ax.set_ylabel("churn probability after a false decline", color=INK_2, fontsize=10)
    ax.set_title(f"Regret from keeping the {t0:.3f} threshold (thousands USD)",
                 color=INK, fontsize=12, weight="bold", loc="left", pad=12)
    for i in range(len(churns)):
        for j in range(len(ltvs)):
            v = reg_s[i, j] / 1000
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=8.5,
                    color=INK if v < reg_s.max()/1000*0.6 else SURFACE)
    ax.tick_params(colors=INK_2)
    fig.colorbar(im, ax=ax, shrink=0.85, label="regret ($k)")
    fig.tight_layout()
    fig.savefig(REPORTS / "cost_sensitivity.png", dpi=170, facecolor=SURFACE)
    emit("Chart: `reports/cost_sensitivity.png`.")
    emit()

    (REPORTS / "cost_sensitivity.md").write_text("\n".join(buf) + "\n")
    print(f"\nwrote {REPORTS/'cost_sensitivity.md'}")


if __name__ == "__main__":
    main()
