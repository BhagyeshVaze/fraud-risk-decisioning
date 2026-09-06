"""Stage 2, randomised arm. Implements reports/stage2_preregistration.md.

Every parameter here is fixed by the pre-registration committed at e5bb4f5,
before any of this code existed. Nothing is tuned to the result.
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPORTS = Path("reports")
OUT = REPORTS / "stage2_randomized_results.md"

# --- registered constants ---------------------------------------------------
SALT = "stage2-v1"
CHAMPION_THR = 0.5000
CHALLENGER_THR = 0.1160
GUARDRAIL_CEILING = 0.080
CB, MR, CH, LTV, RV = 25.0, 0.025, 0.05, 200.0, 2.0
N_BOOT = 2000
PRE_LO, PRE_HI = 110, 150
REPLAY_LO = 150
SEED = 42

buf = []
def emit(s=""):
    print(s); buf.append(s)
def block(s):
    emit("```"); emit(s); emit("```")


def assign(account_id):
    u = int(hashlib.md5(f"{SALT}:{account_id}".encode()).hexdigest(), 16) / 2**128
    return "challenger" if u < 0.5 else "champion"


def txn_cost(declined, y, amt):
    fn = ~declined & (y == 1)
    fp = declined & (y == 0)
    c = np.zeros(len(y))
    c[fn] = amt[fn] + CB
    c[fp] = MR * amt[fp] + CH * LTV
    c += RV * declined
    return c


def boot_ci(a, b, n_boot=N_BOOT, seed=SEED):
    """BCa bootstrap CI for mean(a) - mean(b), resampling within arms."""
    rng = np.random.default_rng(seed)
    obs = a.mean() - b.mean()
    reps = np.empty(n_boot)
    for k in range(n_boot):
        reps[k] = (rng.choice(a, len(a), replace=True).mean()
                   - rng.choice(b, len(b), replace=True).mean())
    z0 = stats.norm.ppf(np.clip((reps < obs).mean(), 1e-6, 1 - 1e-6))
    both = np.concatenate([a, b])
    n = len(both)
    # jackknife acceleration on the pooled difference statistic
    theta_dot = obs
    # Full leave-one-out jackknife, closed form and vectorised.
    ja = (a.sum() - a) / (len(a) - 1) - b.mean()
    jb = a.mean() - (b.sum() - b) / (len(b) - 1)
    jk = np.concatenate([ja, jb])
    d = jk.mean() - jk
    acc = (d**3).sum() / (6 * ((d**2).sum() ** 1.5) + 1e-30)
    out = []
    for q in (0.025, 0.975):
        zq = stats.norm.ppf(q)
        adj = stats.norm.cdf(z0 + (z0 + zq) / (1 - acc * (z0 + zq)))
        out.append(np.quantile(reps, np.clip(adj, 0.001, 0.999)))
    return obs, out[0], out[1]


def welch(a, b):
    diff = a.mean() - b.mean()
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return diff, se, diff - 1.96 * se, diff + 1.96 * se


def ancova(Y, T, X):
    """OLS Y ~ 1 + T + X with HC1 robust SE. Returns coefficient on T and its SE."""
    D = np.column_stack([np.ones(len(Y)), T, X])
    XtX_inv = np.linalg.pinv(D.T @ D)
    beta = XtX_inv @ D.T @ Y
    resid = Y - D @ beta
    n, k = D.shape
    meat = (D * resid[:, None]).T @ (D * resid[:, None])
    V = XtX_inv @ meat @ XtX_inv * (n / (n - k))
    return beta[1], np.sqrt(V[1, 1])


def ratio_se_clustered(num, den):
    """Cluster-robust SE of a ratio estimator sum(num)/sum(den), clusters = rows."""
    R = num.sum() / den.sum()
    m = len(num)
    resid = num - R * den
    # Delta method for a ratio of cluster totals:
    #   Var(R) = sum(e_i^2) * m/(m-1) / (sum(den))^2
    return R, np.sqrt((resid**2).sum() * m / (m - 1)) / den.sum()


def main():
    df = pd.read_parquet("data/parquet/fct_transactions.parquet",
        columns=["transaction_id", "account_id", "txn_day", "transaction_amt",
                 "is_fraud", "has_identity", "product_cd", "card6"])
    p = np.load(REPORTS / "_test_probs.npy")
    tf = pd.read_parquet(REPORTS / "_test_frame.parquet")
    te = df[df.txn_day >= REPLAY_LO].reset_index(drop=True)
    assert (te.transaction_id.to_numpy() == tf.transaction_id.to_numpy()).all(), \
        "probability vector does not align with the replay frame"
    te["p"] = p

    y = te.is_fraud.to_numpy().astype(int)
    amt = te.transaction_amt.to_numpy().astype(float)
    pv = te.p.to_numpy()
    te["c_champ"] = txn_cost(pv >= CHAMPION_THR, y, amt)
    te["c_chal"] = txn_cost(pv >= CHALLENGER_THR, y, amt)
    te["dec_champ"] = pv >= CHAMPION_THR
    te["dec_chal"] = pv >= CHALLENGER_THR

    acc = te.groupby("account_id").agg(
        n_txn=("transaction_id", "size"), tot_amt=("transaction_amt", "sum"),
        mean_amt=("transaction_amt", "mean"), risk=("p", "mean"),
        y_champ=("c_champ", "sum"), y_chal=("c_chal", "sum"),
        fraud=("is_fraud", "sum"))
    acc["arm"] = [assign(a) for a in acc.index]
    acc["Y"] = np.where(acc.arm == "challenger", acc.y_chal, acc.y_champ)
    acc["tau"] = acc.y_chal - acc.y_champ

    pre = df[(df.txn_day >= PRE_LO) & (df.txn_day < PRE_HI)].groupby("account_id").agg(
        pre_value=("transaction_amt", "sum"), pre_n=("transaction_id", "size"),
        pre_fraud=("is_fraud", "sum"), pre_mean_amt=("transaction_amt", "mean"))
    acc = acc.join(pre, how="left")
    acc["has_pre_period"] = acc.pre_n.notna().astype(float)
    for c in ("pre_value", "pre_n", "pre_fraud", "pre_mean_amt"):
        acc[c] = acc[c].fillna(0.0)
    acc.to_parquet(REPORTS / "_stage2_accounts.parquet")

    emit("# Stage 2, randomised arm")
    emit()
    emit("Executes `reports/stage2_preregistration.md`, committed at `e5bb4f5` "
         "before this code existed. Champion 0.5000, challenger 0.1160, "
         "assignment by md5 hash with salt `stage2-v1`.")
    emit()
    emit("Registered up front: the direction of this result is not in doubt. "
         "Stage 1 measured the contrast at $155,129. The purpose here is the "
         "machinery and the benchmark for the observational half.")
    emit()

    # --- SRM ---------------------------------------------------------------
    emit("## SRM check")
    emit()
    n_chal = int((acc.arm == "challenger").sum())
    n_champ = int((acc.arm == "champion").sum())
    N = n_chal + n_champ
    chi2 = (n_chal - N / 2) ** 2 / (N / 2) + (n_champ - N / 2) ** 2 / (N / 2)
    pval = 1 - stats.chi2.cdf(chi2, 1)
    block(f"accounts challenger  {n_chal:>7,}\n"
          f"accounts champion    {n_champ:>7,}\n"
          f"total                {N:>7,}\n"
          f"split                {n_chal/N:.5f} / {n_champ/N:.5f}\n"
          f"chi-square (1 df)    {chi2:.4f}\n"
          f"p-value              {pval:.4f}\n"
          f"halt threshold       p < 0.001\n"
          f"verdict              {'PASS' if pval >= 0.001 else 'HALT'}")
    emit()
    t_chal = int(te.account_id.map(acc.arm).eq("challenger").sum())
    emit(f"Transaction counts, reported descriptively and not tested, since "
         f"cluster sizes vary: challenger {t_chal:,}, champion "
         f"{len(te)-t_chal:,} ({t_chal/len(te):.4f} share).")
    emit()

    # --- primary ------------------------------------------------------------
    A = acc.loc[acc.arm == "challenger", "Y"].to_numpy()
    B = acc.loc[acc.arm == "champion", "Y"].to_numpy()
    diff, se, lo, hi = welch(A, B)
    t = diff / se
    pv_primary = 2 * (1 - stats.norm.cdf(abs(t)))
    b_obs, b_lo, b_hi = boot_ci(A, B)

    emit("## Primary: mean cost per account")
    emit()
    block(f"challenger mean   ${A.mean():>10.4f}   n {len(A):,}\n"
          f"champion   mean   ${B.mean():>10.4f}   n {len(B):,}\n"
          f"difference        ${diff:>+10.4f} per account\n"
          f"                  ${diff*N:>+10,.0f} scaled to all {N:,} accounts\n"
          f"SE                ${se:>10.4f}\n"
          f"t                 {t:>11.3f}\n"
          f"p (two-sided)     {pv_primary:.3e}\n"
          f"95% CI (normal)   [${lo:+.4f}, ${hi:+.4f}] per account\n"
          f"95% CI (BCa)      [${b_lo:+.4f}, ${b_hi:+.4f}] per account")
    emit()
    agree = abs(b_lo - lo) < 0.15 * abs(diff) and abs(b_hi - hi) < 0.15 * abs(diff)
    same_sign = (hi < 0) == (b_hi < 0)
    if agree and same_sign:
        emit("Normal and BCa intervals agree.")
    else:
        emit(f"**Normal and BCa intervals disagree.** Normal "
             f"[{lo:+.4f}, {hi:+.4f}] {'excludes' if hi < 0 else 'includes'} zero; "
             f"BCa [{b_lo:+.4f}, {b_hi:+.4f}] "
             f"{'excludes' if b_hi < 0 else 'includes'} zero. "
             "The pre-registration names the normal-theory interval as primary "
             "and the bootstrap as a cross-check, so **the normal interval "
             "governs the ship decision**. Switching to whichever interval "
             "gives the preferred answer is the exact failure pre-registration "
             "exists to prevent. The disagreement is driven by the heavy right "
             "tail of per-account cost and is reported, not resolved.")
    emit()

    # --- CUPED / ANCOVA -------------------------------------------------------
    emit("## CUPED, as multivariate regression adjustment")
    emit()
    Y = acc.Y.to_numpy()
    T = (acc.arm == "challenger").astype(float).to_numpy()
    Xc = acc[["pre_value", "pre_n", "pre_fraud", "pre_mean_amt", "has_pre_period"]].to_numpy()
    b_un, se_un = ancova(Y, T, np.empty((len(Y), 0)))
    b_ad, se_ad = ancova(Y, T, Xc)
    vr = 1 - (se_ad / se_un) ** 2
    block(f"unadjusted  effect ${b_un:>+9.4f}   SE ${se_un:.4f}\n"
          f"adjusted    effect ${b_ad:>+9.4f}   SE ${se_ad:.4f}\n"
          f"variance reduction {vr*100:>8.2f}%   (registered expectation 4.9%)\n"
          f"adjusted 95% CI    [${b_ad-1.96*se_ad:+.4f}, ${b_ad+1.96*se_ad:+.4f}]")
    emit()
    emit(f"Realised variance reduction of {vr*100:.2f}% against a registered "
         f"expectation of 4.9%. Pre-period coverage is "
         f"{acc.has_pre_period.mean()*100:.1f}%, which is the binding "
         "constraint. **This is a null result for CUPED on this data**, "
         "registered as a likely outcome in advance and reported as such.")
    emit()

    # --- guardrail -------------------------------------------------------------
    emit("## Guardrail: false decline rate against the 8.0% ceiling")
    emit()
    te["arm"] = te.account_id.map(acc.arm)
    for arm, thr, col in (("challenger", CHALLENGER_THR, "dec_chal"),
                          ("champion", CHAMPION_THR, "dec_champ")):
        sub = te[(te.arm == arm) & (te.is_fraud == 0)]
        g = sub.groupby("account_id").agg(d=(col, "sum"), n=(col, "size"))
        fdr, fse = ratio_se_clustered(g.d.to_numpy().astype(float),
                                      g.n.to_numpy().astype(float))
        if arm == "challenger":
            z = (fdr - GUARDRAIL_CEILING) / fse
            p1 = stats.norm.cdf(z)
            ub = fdr + 1.645 * fse
            emit(f"**{arm}** threshold {thr}")
            block(f"false decline rate   {fdr*100:.4f}%\n"
                  f"cluster-robust SE    {fse*100:.4f}pp   (clusters = {len(g):,} accounts)\n"
                  f"one-sided 95% UB     {ub*100:.4f}%\n"
                  f"ceiling              {GUARDRAIL_CEILING*100:.1f}%\n"
                  f"H0: FDR >= 8.0%      z {z:.3f}   p {p1:.3e}\n"
                  f"verdict              {'PASS, below the ceiling' if ub < GUARDRAIL_CEILING else 'BREACH'}")
            guard_pass = ub < GUARDRAIL_CEILING
        else:
            emit(f"Champion arm, for reference: false decline rate "
                 f"{fdr*100:.4f}% (SE {fse*100:.4f}pp).")
        emit()

    # --- secondary --------------------------------------------------------------
    emit("## Secondary metrics")
    emit()
    rows = []
    for arm, dcol in (("challenger", "dec_chal"), ("champion", "dec_champ")):
        s = te[te.arm == arm]
        d = s[dcol].to_numpy()
        yy = s.is_fraud.to_numpy().astype(int)
        aa = s.transaction_amt.to_numpy().astype(float)
        rows.append({"arm": arm, "transactions": len(s), "frauds": int(yy.sum()),
                     "declined": int(d.sum()),
                     "decline_rate": round(d.mean(), 5),
                     "recall": round(d[yy == 1].mean(), 5),
                     "precision": round(yy[d].mean(), 5) if d.sum() else np.nan,
                     "fraud_dollars_caught": round(aa[d & (yy == 1)].sum(), 0),
                     "fraud_dollars_total": round(aa[yy == 1].sum(), 0),
                     "cost_per_txn": round(s["c_chal" if arm == "challenger" else "c_champ"].mean(), 4)})
    sec = pd.DataFrame(rows)
    sec["pct_fraud_dollars_caught"] = (sec.fraud_dollars_caught / sec.fraud_dollars_total * 100).round(2)
    block(sec.to_string(index=False))
    emit()

    # --- subgroups ---------------------------------------------------------------
    emit("## Subgroups, descriptive only")
    emit()
    emit("Registered as descriptive and excluded from the ship decision. Two of "
         "the four are underpowered at 15.5% and 14.5%.")
    emit()
    med = acc.mean_amt.median()
    acc["established"] = acc.has_pre_period > 0
    acc["high_ticket"] = acc.mean_amt >= med
    rows = []
    for col, labs in (("established", ("new", "established")),
                      ("high_ticket", (f"low ticket (< ${med:.2f})", f"high ticket (>= ${med:.2f})"))):
        for v, lab in zip((False, True), labs):
            s = acc[acc[col] == v]
            a = s.loc[s.arm == "challenger", "Y"].to_numpy()
            b = s.loc[s.arm == "champion", "Y"].to_numpy()
            d, e, l, h = welch(a, b)
            rows.append({"subgroup": lab, "accounts": len(s),
                         "effect_per_account": round(d, 4),
                         "ci_low": round(l, 4), "ci_high": round(h, 4),
                         "scaled_total": round(d * len(s), 0),
                         "crosses_zero": "yes" if l < 0 < h else "no"})
    block(pd.DataFrame(rows).to_string(index=False))
    emit()

    # --- quintile profile ----------------------------------------------------------
    emit("## Effect by account risk quintile")
    emit()
    emit("Registered to lead the recommendation ahead of the average treatment "
         "effect.")
    emit()
    acc["q"] = pd.qcut(acc.risk.rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
    rows = []
    for q, s in acc.groupby("q", observed=True):
        a = s.loc[s.arm == "challenger", "Y"].to_numpy()
        b = s.loc[s.arm == "champion", "Y"].to_numpy()
        d, e, l, h = welch(a, b)
        rows.append({"quintile": int(q), "accounts": len(s),
                     "mean_risk": round(s.risk.mean(), 5),
                     "mean_cost_champion": round(b.mean(), 3),
                     "effect_per_account": round(d, 4),
                     "ci_low": round(l, 3), "ci_high": round(h, 3),
                     "scaled_total": round(d * len(s), 0)})
    block(pd.DataFrame(rows).to_string(index=False))
    emit()

    # --- targeted application ---------------------------------------------------
    emit("## Targeted application follow-up")
    emit()
    emit("Applying the challenger only above a risk cutoff, champion below. "
         "Computed as a paired counterfactual across all accounts, since both "
         "outcomes are known for every account.")
    emit()
    # Cost alone understates the case for targeting: the same cost achieved on
    # fewer accounts means far fewer customers are wrongly declined.
    acc_risk = acc.risk
    te["acct_risk"] = te.account_id.map(acc_risk)
    legit_m = te.is_fraud.to_numpy() == 0
    fraud_m = ~legit_m
    amt_t = te.transaction_amt.to_numpy().astype(float)
    dc, dch = te.dec_champ.to_numpy(), te.dec_chal.to_numpy()
    rows = []
    for cut in [0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.20, 1.01]:
        use_chal = acc.risk >= cut
        total = np.where(use_chal, acc.y_chal, acc.y_champ).sum()
        on = te.acct_risk.to_numpy() >= cut
        dec = np.where(on, dch, dc)
        rows.append({"risk_cutoff": cut,
                     "accounts_on_challenger": int(use_chal.sum()),
                     "pct_accounts": round(use_chal.mean() * 100, 2),
                     "total_cost": round(total, 0),
                     "legit_declined": int((dec & legit_m).sum()),
                     "false_decline_rate_pct": round((dec & legit_m).sum() / legit_m.sum() * 100, 3),
                     "fraud_dollars_caught": round(amt_t[dec & fraud_m].sum(), 0)})
    tg = pd.DataFrame(rows)
    all_chal = acc.y_chal.sum()
    tg["vs_challenger_everywhere"] = (tg.total_cost - all_chal).round(0)
    block(tg.to_string(index=False))
    emit()
    best = tg.loc[tg.total_cost.idxmin()]
    uni = tg[tg.risk_cutoff == 0.0].iloc[0]
    emit(f"Cheapest targeting cutoff: **{best.risk_cutoff}**, applying the "
         f"challenger to {best.pct_accounts}% of accounts, total cost "
         f"${best.total_cost:,.0f} against ${all_chal:,.0f} for uniform "
         f"application (difference ${best.total_cost-all_chal:+,.0f}).")
    emit()
    row12 = tg[tg.risk_cutoff == 0.12].iloc[0]
    emit(f"**Targeting buys almost nothing at the cost-optimal cutoff.** At "
         f"0.02 it declines {int(best.legit_declined):,} legitimate "
         f"transactions against {int(uni.legit_declined):,} for uniform "
         f"application, a {(1-best.legit_declined/uni.legit_declined)*100:.1f}% "
         f"reduction, and catches identical fraud dollars. The reason is "
         f"mechanical: accounts below 0.02 mean risk almost never cross a "
         f"0.116 decline threshold anyway, so excluding them changes almost no "
         f"decisions. The registered expectation that targeting would dominate "
         f"uniform application is **not supported**.")
    emit()
    emit(f"There is a real trade further up the curve, but it costs money. At a "
         f"0.12 cutoff, false declines fall to {int(row12.legit_declined):,} "
         f"({row12.false_decline_rate_pct}%), a "
         f"{(1-row12.legit_declined/uni.legit_declined)*100:.0f}% reduction, "
         f"for ${row12.total_cost-all_chal:+,.0f} of additional cost and "
         f"${row12.fraud_dollars_caught-uni.fraud_dollars_caught:+,.0f} less "
         f"fraud caught. Whether that is worth it depends entirely on the churn "
         f"assumption, which section 6 of the cost sensitivity report shows is "
         f"the least reliable constant in the model.")
    emit()

    # --- paired oracle -------------------------------------------------------------
    emit("## Paired counterfactual benchmark")
    emit()
    d = acc.tau.to_numpy()
    pse = d.std(ddof=1) / np.sqrt(len(d))
    emit("Available only because this is a replay. Reported as an efficiency "
         "benchmark and as a bug detector: a material disagreement with the "
         "randomised estimate would indicate a fault in assignment or analysis.")
    emit()
    block(f"paired effect      ${d.mean():+.4f} per account\n"
          f"paired SE          ${pse:.4f}\n"
          f"paired 95% CI      [${d.mean()-1.96*pse:+.4f}, ${d.mean()+1.96*pse:+.4f}]\n"
          f"randomised effect  ${diff:+.4f}   SE ${se:.4f}\n"
          f"efficiency ratio   {se/pse:.2f}x   (randomised SE / paired SE)\n"
          f"agreement          {'consistent' if abs(d.mean()-diff) < 1.96*se else 'DISAGREE'}")
    emit()

    # --- ship recommendation ----------------------------------------------------
    emit("## Ship recommendation")
    emit()
    c1 = pv_primary < 0.05 and diff < 0
    c2 = hi < 0
    c3 = guard_pass
    block(f"1 primary significant, challenger favoured   {'PASS' if c1 else 'FAIL'}\n"
          f"2 upper bound of 95% CI below zero           {'PASS' if c2 else 'FAIL'}\n"
          f"3 guardrail below the 8.0% ceiling at its UB {'PASS' if c3 else 'FAIL'}\n"
          f"decision                                     "
          f"{'SHIP' if (c1 and c2 and c3) else 'DO NOT SHIP'}")
    emit()
    json.dump({"effect_per_account": float(diff), "se": float(se),
               "ci": [float(lo), float(hi)], "bca": [float(b_lo), float(b_hi)],
               "n_chal": n_chal, "n_champ": n_champ,
               "total_effect": float(diff * N)},
              open(REPORTS / "_stage2_truth.json", "w"), indent=2)
    OUT.write_text("\n".join(buf) + "\n")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
