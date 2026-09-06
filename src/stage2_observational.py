"""Stage 2, observational half. Implements section 10 of the pre-registration.

Simulates a non-random rollout steered by account risk, then tries to recover
the effect the way a post-hoc analyst would, and compares against the truth
that a replay makes available.

One clarification against the registration is logged in the deviations table:
section 10 named the randomised estimate as "the benchmark truth". In a replay
both potential outcomes are known for every account, so the exact ATE is
computable and is the correct target. The randomised estimate is one noisy draw
at that target. Bias is reported against both, with the exact quantity primary.

A second point the registration did not anticipate: propensity matching
estimates the effect on the treated (ATT), and the treatment effect here is
strongly heterogeneous. Under a risk-steered rollout the ATT differs from the
ATE by construction, so comparing an ATT estimate against the ATE would
attribute estimand mismatch to confounding. Both targets are reported.
"""

import hashlib
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import brentq
from sklearn.linear_model import LogisticRegression

REPORTS = Path("reports")
OUT = REPORTS / "stage2_observational_results.md"
SALT = "stage2-v1"
BETAS = [0.5, 1.5, 3.0, -0.5, -1.5, -3.0]
CALIPER_SD = 0.2
N_BOOT = 400
SEED = 42

C_A, C_B, C_C = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK_2, GRIDC, SURFACE = "#0b0b0b", "#52514e", "#d9d8d2", "#fcfcfb"

buf = []
def emit(s=""):
    print(s); buf.append(s)
def block(s):
    emit("```"); emit(s); emit("```")


def hash_u(account_id, tag):
    return int(hashlib.md5(f"{SALT}:{tag}:{account_id}".encode()).hexdigest(), 16) / 2**128


def smd(x, t):
    a, b = x[t == 1], x[t == 0]
    sd = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return (a.mean() - b.mean()) / sd if sd > 0 else 0.0


def match_att(Y, T, X, u_tie):
    """1:1 nearest neighbour on the propensity logit, with replacement.

    Returns ATT, diagnostics and the matched pair differences.
    """
    ps = LogisticRegression(max_iter=2000, C=1.0).fit(X, T).predict_proba(X)[:, 1]
    ps = np.clip(ps, 1e-6, 1 - 1e-6)
    logit = np.log(ps / (1 - ps))
    caliper = CALIPER_SD * logit.std(ddof=1)

    ti = np.flatnonzero(T == 1)
    ci = np.flatnonzero(T == 0)
    # common support: drop treated outside the control logit range
    lo, hi = logit[ci].min(), logit[ci].max()
    keep = (logit[ti] >= lo) & (logit[ti] <= hi)
    dropped_support = int((~keep).sum())
    ti = ti[keep]

    order = np.argsort(logit[ci], kind="stable")
    cs, cl = ci[order], logit[ci][order]
    pos = np.searchsorted(cl, logit[ti])
    best = np.empty(len(ti), dtype=np.int64)
    bestd = np.full(len(ti), np.inf)
    for off in (-1, 0):
        j = np.clip(pos + off, 0, len(cl) - 1)
        d = np.abs(cl[j] - logit[ti])
        take = d < bestd
        best[take], bestd[take] = cs[j][take], d[take]

    within = bestd <= caliper
    dropped_caliper = int((~within).sum())
    ti, mc = ti[within], best[within]

    diffs = Y[ti] - Y[mc]
    att = diffs.mean()
    ess = len(np.unique(mc))
    return {
        "att": att, "diffs": diffs, "ps": ps, "logit": logit,
        "treated_used": len(ti), "matched_controls_unique": ess,
        "dropped_support": dropped_support, "dropped_caliper": dropped_caliper,
        "caliper": caliper, "treated_idx": ti, "control_idx": mc,
    }


def boot_att(Y, T, X, n_boot=N_BOOT, seed=SEED):
    """Bootstrap over treated units. Not formally valid for NN matching
    (Abadie and Imbens 2008); reported as indicative only."""
    rng = np.random.default_rng(seed)
    res = match_att(Y, T, X, None)
    d = res["diffs"]
    reps = np.array([rng.choice(d, len(d), replace=True).mean() for _ in range(n_boot)])
    return np.percentile(reps, 2.5), np.percentile(reps, 97.5)


def rosenbaum(diffs, gammas=(1.0, 1.1, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0, 6.0)):
    """Rosenbaum bounds on the Wilcoxon signed-rank test for matched pairs."""
    d = diffs[diffs != 0]
    if len(d) < 10:
        return pd.DataFrame()
    r = stats.rankdata(np.abs(d))
    W = r[d < 0].sum()          # effect is negative (cheaper), so rank the negatives
    S, S2 = r.sum(), (r**2).sum()
    rows = []
    for g in gammas:
        pplus = g / (1 + g)
        pminus = 1 / (1 + g)
        # worst case against the finding: expectation as small as possible
        z_worst = (W - pminus * S) / np.sqrt(pminus * (1 - pminus) * S2)
        z_best = (W - pplus * S) / np.sqrt(pplus * (1 - pplus) * S2)
        rows.append({"gamma": g,
                     "p_upper_bound": float(1 - stats.norm.cdf(z_worst)),
                     "p_lower_bound": float(1 - stats.norm.cdf(z_best))})
    return pd.DataFrame(rows)


def main():
    acc = pd.read_parquet(REPORTS / "_stage2_accounts.parquet")
    truth_r = json.load(open(REPORTS / "_stage2_truth.json"))

    df = pd.read_parquet("data/parquet/fct_transactions.parquet",
        columns=["transaction_id", "account_id", "txn_day", "transaction_amt",
                 "has_identity", "product_cd", "card6"])
    te = df[df.txn_day >= 150]
    first = te.sort_values(["account_id", "transaction_id"]).groupby("account_id").first()
    acc["first_amt"] = first.transaction_amt
    acc["first_hasid"] = first.has_identity.astype(float)
    acc["first_prod"] = first.product_cd.astype(str)
    acc["first_card6"] = first.card6.astype(str)

    y_chal = acc.y_chal.to_numpy()
    y_champ = acc.y_champ.to_numpy()
    tau = y_chal - y_champ
    risk = acc.risk.to_numpy()
    z = (risk - risk.mean()) / risk.std(ddof=1)
    N = len(acc)

    ATE = tau.mean()

    base_cols = ["pre_value", "pre_n", "pre_fraud", "has_pre_period",
                 "n_txn", "tot_amt", "first_amt", "first_hasid"]
    dummies = pd.get_dummies(acc[["first_prod", "first_card6"]], dummy_na=True, dtype=float)
    Xobs = np.column_stack([acc[base_cols].to_numpy(dtype=float), dummies.to_numpy()])
    Xobs = (Xobs - Xobs.mean(0)) / np.where(Xobs.std(0) > 0, Xobs.std(0), 1)
    Xora = np.column_stack([Xobs, (risk - risk.mean()) / risk.std(ddof=1)])
    cov_names = base_cols + list(dummies.columns)

    emit("# Stage 2, observational half")
    emit()
    emit("Implements section 10 of `reports/stage2_preregistration.md`, "
         "committed at `e5bb4f5`. Simulates a non-random rollout steered by "
         "account risk, then recovers the effect the way a post-hoc analyst "
         "would and compares against the truth.")
    emit()

    emit("## The two targets, and why both are needed")
    emit()
    emit(f"Because this is a replay, both potential outcomes are known for "
         f"every account, so the **exact ATE is computable: "
         f"${ATE:+.4f} per account** (${ATE*N:+,.0f} scaled). The randomised "
         f"arm estimated ${truth_r['effect_per_account']:+.4f} "
         f"(SE ${truth_r['se']:.4f}), one noisy draw at that quantity.")
    emit()
    emit("Propensity matching estimates the effect **on the treated** (ATT). "
         "The effect here is strongly heterogeneous: $0.00 in risk quintiles 1 "
         "to 3 and -$16.29 in quintile 5. Under a rollout steered by risk the "
         "treated group is not a random sample, so the ATT genuinely differs "
         "from the ATE. Judging an ATT estimate against the ATE would charge "
         "estimand mismatch to confounding. Both targets are therefore "
         "reported for every scenario, and the ATT is the one PSM is "
         "responsible for.")
    emit()

    all_rows, diag_rows, ros_store, ps_store = [], [], {}, {}
    for beta in BETAS:
        # Solve alpha so the realised treated share is 50%. Note the
        # parenthesisation: the mean is over the sigmoid, not over its
        # denominator.
        def share(a, b=beta):
            return (1.0 / (1.0 + np.exp(-(a + b * z)))).mean()
        f = lambda a: share(a) - 0.5
        alpha = brentq(f, -50, 50)
        pr = 1 / (1 + np.exp(-(alpha + beta * z)))
        u = np.array([hash_u(a, f"obs{beta}") for a in acc.index])
        T = (u < pr).astype(int)
        Y = np.where(T == 1, y_chal, y_champ)

        ATT = tau[T == 1].mean()
        naive = Y[T == 1].mean() - Y[T == 0].mean()
        mo = match_att(Y, T, Xobs, u)
        mr = match_att(Y, T, Xora, u)
        lo_o, hi_o = boot_att(Y, T, Xobs)
        lo_r, hi_r = boot_att(Y, T, Xora)

        label = f"beta {beta:+.1f} ({'risk-seeking' if beta > 0 else 'risk-averse'})"
        ps_store[label] = (mo["ps"], T)
        d_ = mo["diffs"]
        ros_store[label] = (rosenbaum(d_), {
            "pairs": len(d_),
            "zero_pct": (d_ == 0).mean() * 100,
            "nonzero_neg_pct": (d_[d_ != 0] < 0).mean() * 100 if (d_ != 0).any() else np.nan,
            "top1pct_share": np.sort(np.abs(d_))[-max(len(d_)//100, 1):].sum() / max(np.abs(d_).sum(), 1e-9) * 100,
            "mean": d_.mean(), "median": float(np.median(d_))})

        for est, val, lo, hi, tgt, tv in (
                ("naive difference", naive, np.nan, np.nan, "ATT", ATT),
                ("PSM, observed covariates", mo["att"], lo_o, hi_o, "ATT", ATT),
                ("PSM, oracle (includes risk)", mr["att"], lo_r, hi_r, "ATT", ATT)):
            all_rows.append({
                "scenario": label, "beta": beta, "estimator": est,
                "estimate": round(val, 4), "ci_low": round(lo, 4) if lo == lo else np.nan,
                "ci_high": round(hi, 4) if hi == hi else np.nan,
                "true_ATT": round(ATT, 4), "true_ATE": round(ATE, 4),
                "bias_vs_ATT": round(val - tv, 4),
                "bias_pct_vs_ATT": round((val - tv) / abs(tv) * 100, 1) if tv else np.nan,
                "bias_vs_ATE": round(val - ATE, 4),
                "sign_correct": "yes" if np.sign(val) == np.sign(ATE) else "NO"})

        for nm, m in (("observed", mo), ("oracle", mr)):
            X = Xobs if nm == "observed" else Xora
            pre = np.mean([abs(smd(X[:, j], T)) for j in range(X.shape[1])])
            Tm = np.zeros(len(Y)); 
            sel = np.concatenate([m["treated_idx"], m["control_idx"]])
            lab = np.concatenate([np.ones(len(m["treated_idx"])), np.zeros(len(m["control_idx"]))])
            post = np.mean([abs(smd(X[sel, j], lab)) for j in range(X.shape[1])])
            diag_rows.append({
                "scenario": label, "psm": nm,
                "treated_matched": m["treated_used"],
                "unique_controls": m["matched_controls_unique"],
                "dropped_support": m["dropped_support"],
                "dropped_caliper": m["dropped_caliper"],
                "caliper": round(m["caliper"], 4),
                "mean_abs_SMD_before": round(pre, 4),
                "mean_abs_SMD_after": round(post, 4)})

    res = pd.DataFrame(all_rows)
    dia = pd.DataFrame(diag_rows)
    res.to_csv(REPORTS / "stage2_observational_estimates.csv", index=False)
    dia.to_csv(REPORTS / "stage2_observational_diagnostics.csv", index=False)

    emit("## Results by scenario")
    emit()
    emit(f"Exact ATE across all accounts: **${ATE:+.4f} per account**. "
         "Negative favours the challenger.")
    emit()
    for label in res.scenario.unique():
        s = res[res.scenario == label]
        emit(f"**{label}**, true ATT ${s.true_ATT.iloc[0]:+.4f}")
        emit()
        block(s[["estimator", "estimate", "ci_low", "ci_high", "bias_vs_ATT",
                 "bias_pct_vs_ATT", "bias_vs_ATE", "sign_correct"]].to_string(index=False))
        emit()

    emit("## Bias summary")
    emit()
    piv = res.pivot_table(index="beta", columns="estimator", values="bias_vs_ATT")
    block(piv.round(4).to_string())
    emit()
    emit("Bias against the ATT, in dollars per account. The oracle column is "
         "the control: it uses the same matching machinery plus the one "
         "variable that actually drove assignment.")
    emit()

    emit("## Matching diagnostics")
    emit()
    block(dia.to_string(index=False))
    emit()

    emit("## Rosenbaum sensitivity, PSM on observed covariates")
    emit()
    emit("How large an unmeasured confounder, expressed as an odds ratio on "
         "treatment assignment, would be needed before the matched-pair "
         "conclusion could be overturned.")
    emit()
    rows = []
    for label, (tab, st) in ros_store.items():
        g = "n/a"
        if len(tab):
            crit = tab[tab.p_upper_bound > 0.05]
            g = crit.gamma.min() if len(crit) else ">6.0"
        rows.append({"scenario": label, "gamma_to_overturn": g,
                     "matched_pairs": st["pairs"],
                     "pct_pairs_exactly_zero": round(st["zero_pct"], 1),
                     "pct_of_nonzero_pairs_negative": round(st["nonzero_neg_pct"], 1),
                     "pct_of_total_abs_diff_in_top_1pct": round(st["top1pct_share"], 1),
                     "mean_pair_diff": round(st["mean"], 4),
                     "median_pair_diff": round(st["median"], 4)})
    block(pd.DataFrame(rows).to_string(index=False))
    emit()
    emit("**The Rosenbaum bounds are uninformative here, and the reason is "
         "worth stating rather than hiding.** Around 91% of matched pairs have "
         "an outcome difference of exactly zero, because most accounts are "
         "never declined under either policy. Of the pairs that do differ, a "
         "minority are negative, while the top 1% of absolute differences "
         "carries roughly two thirds of the total mass. A signed-rank "
         "statistic counts pairs; this effect lives entirely in the size of a "
         "few of them. So the rank test reports non-significance in scenarios "
         "where the mean difference is large and its bootstrap interval "
         "excludes zero.")
    emit()
    emit("This is a limitation of rank-based sensitivity analysis on a "
         "zero-inflated, tail-dominated outcome, not evidence about "
         "confounding. Reported because the pre-registration required "
         "Rosenbaum sensitivity, and a required diagnostic that turns out to "
         "be inapplicable should be shown to be inapplicable rather than "
         "quietly dropped.")
    emit()

    # --- plot -----------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0), facecolor=SURFACE)
    ax = axes[0]; ax.set_facecolor(SURFACE)
    for est, col, mk in (("naive difference", C_B, "o"),
                         ("PSM, observed covariates", C_A, "s"),
                         ("PSM, oracle (includes risk)", C_C, "^")):
        s = res[res.estimator == est].sort_values("beta")
        ax.plot(s.beta, s.bias_vs_ATT, color=col, lw=2, marker=mk, ms=9,
                markeredgecolor=SURFACE, markeredgewidth=2, label=est)
    ax.axhline(0, color=GRIDC, lw=2, ls="--")
    ax.set_xlabel("confounding strength beta (negative = risk-averse rollout)",
                  color=INK_2, fontsize=10)
    ax.set_ylabel("bias against the true ATT ($ per account)", color=INK_2, fontsize=10)
    ax.set_title("Bias by estimator and confounding strength", color=INK,
                 fontsize=12, weight="bold", loc="left", pad=12)
    ax.grid(True, color=GRIDC, lw=0.8, alpha=0.7)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"): ax.spines[sp].set_color(GRIDC)
    ax.tick_params(colors=INK_2, labelsize=9)
    leg = ax.legend(frameon=False, fontsize=9)
    for t in leg.get_texts(): t.set_color(INK_2)

    ax = axes[1]; ax.set_facecolor(SURFACE)
    ps, T = ps_store["beta +3.0 (risk-seeking)"]
    bins = np.linspace(0, 1, 41)
    ax.hist(ps[T == 1], bins=bins, alpha=0.65, color=C_A, label="treated")
    ax.hist(ps[T == 0], bins=bins, alpha=0.65, color=C_B, label="control")
    ax.set_xlabel("estimated propensity score", color=INK_2, fontsize=10)
    ax.set_ylabel("accounts", color=INK_2, fontsize=10)
    ax.set_title("Propensity overlap, strongest confounding (beta +3.0)",
                 color=INK, fontsize=12, weight="bold", loc="left", pad=12)
    ax.grid(True, color=GRIDC, lw=0.8, alpha=0.7, axis="y")
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"): ax.spines[sp].set_color(GRIDC)
    ax.tick_params(colors=INK_2, labelsize=9)
    leg = ax.legend(frameon=False, fontsize=9)
    for t in leg.get_texts(): t.set_color(INK_2)
    fig.tight_layout()
    fig.savefig(REPORTS / "stage2_observational.png", dpi=170, facecolor=SURFACE)
    emit("Chart: `reports/stage2_observational.png`.")
    emit()

    OUT.write_text("\n".join(buf) + "\n")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
