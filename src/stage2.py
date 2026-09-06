"""Stage 2: pre-registered replay experiment and observational comparison.

Usage: python src/stage2.py {randomized|observational}

Every parameter is fixed by reports/stage2_preregistration.md, committed at
e5bb4f5 before this code existed. Nothing here is tuned to the result.
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
from scipy import stats
from scipy.optimize import brentq
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import core
from core import COSTS, REPORTS, Report

SALT = "stage2-v1"
CHAMPION_THR, CHALLENGER_THR = 0.5000, 0.1160
GUARDRAIL_CEILING = 0.080
BETAS = [0.5, 1.5, 3.0, -0.5, -1.5, -3.0]
CALIPER_SD, N_BOOT_PSM = 0.2, 400
PRE_LO, PRE_HI, REPLAY_LO = 110, 150, 150

C_A, C_B, C_C = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK_2, GRIDC, SURFACE = "#0b0b0b", "#52514e", "#d9d8d2", "#fcfcfb"

FCT_COLS = ["transaction_id", "account_id", "txn_day", "transaction_amt",
            "is_fraud", "has_identity", "product_cd", "card6"]


def load_replay():
    df = pd.read_parquet(core.FCT, columns=FCT_COLS)
    p = np.load(REPORTS / "_test_probs.npy")
    tf = pd.read_parquet(REPORTS / "_test_frame.parquet")
    te = df[df.txn_day >= REPLAY_LO].reset_index(drop=True)
    assert (te.transaction_id.to_numpy() == tf.transaction_id.to_numpy()).all(), \
        "probability vector does not align with the replay frame"
    te["p"] = p
    return df, te


# --------------------------------------------------------------------------- #
# randomised arm
# --------------------------------------------------------------------------- #

def randomized():
    R = Report(REPORTS / "stage2_randomized_results.md")
    df, te = load_replay()
    y = te.is_fraud.to_numpy().astype(int)
    amt = te.transaction_amt.to_numpy().astype(float)
    pv = te.p.to_numpy()
    te["c_champ"] = core.txn_cost(pv >= CHAMPION_THR, y, amt)
    te["c_chal"] = core.txn_cost(pv >= CHALLENGER_THR, y, amt)
    te["dec_champ"], te["dec_chal"] = pv >= CHAMPION_THR, pv >= CHALLENGER_THR

    acc = te.groupby("account_id").agg(
        n_txn=("transaction_id", "size"), tot_amt=("transaction_amt", "sum"),
        mean_amt=("transaction_amt", "mean"), risk=("p", "mean"),
        y_champ=("c_champ", "sum"), y_chal=("c_chal", "sum"), fraud=("is_fraud", "sum"))
    acc["arm"] = ["challenger" if core.account_hash(a, SALT) < 0.5 else "champion"
                  for a in acc.index]
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

    R("# Stage 2, randomised arm"); R()
    R("Executes `reports/stage2_preregistration.md`, committed at `e5bb4f5` "
      "before this code existed. Champion 0.5000, challenger 0.1160, assignment "
      "by md5 hash with salt `stage2-v1`."); R()
    R("Registered up front: the direction is not in doubt. The purpose is the "
      "machinery and the benchmark for the observational half."); R()

    R("## SRM check"); R()
    n_chal = int((acc.arm == "challenger").sum()); N = len(acc); n_champ = N - n_chal
    chi2 = (n_chal - N / 2) ** 2 / (N / 2) + (n_champ - N / 2) ** 2 / (N / 2)
    pval = 1 - stats.chi2.cdf(chi2, 1)
    R.block(f"accounts challenger  {n_chal:>7,}\naccounts champion    {n_champ:>7,}\n"
            f"total                {N:>7,}\nsplit                {n_chal/N:.5f} / {n_champ/N:.5f}\n"
            f"chi-square (1 df)    {chi2:.4f}\np-value              {pval:.4f}\n"
            f"halt threshold       p < 0.001\n"
            f"verdict              {'PASS' if pval >= 0.001 else 'HALT'}"); R()
    te["arm"] = te.account_id.map(acc.arm)
    t_chal = int((te.arm == "challenger").sum())
    R(f"Transaction counts, descriptive only since cluster sizes vary: challenger "
      f"{t_chal:,}, champion {len(te)-t_chal:,}."); R()

    A = acc.loc[acc.arm == "challenger", "Y"].to_numpy()
    B = acc.loc[acc.arm == "champion", "Y"].to_numpy()
    diff, se, lo, hi = core.welch(A, B)
    t = diff / se
    p_primary = 2 * (1 - stats.norm.cdf(abs(t)))
    _, b_lo, b_hi = core.bca_ci(A, B)

    R("## Primary: mean cost per account"); R()
    R.block(f"challenger mean   ${A.mean():>10.4f}   n {len(A):,}\n"
            f"champion   mean   ${B.mean():>10.4f}   n {len(B):,}\n"
            f"difference        ${diff:>+10.4f} per account\n"
            f"                  ${diff*N:>+10,.0f} scaled to all {N:,} accounts\n"
            f"SE                ${se:>10.4f}\nt                 {t:>11.3f}\n"
            f"p (two-sided)     {p_primary:.3e}\n"
            f"95% CI (normal)   [${lo:+.4f}, ${hi:+.4f}] per account\n"
            f"95% CI (BCa)      [${b_lo:+.4f}, ${b_hi:+.4f}] per account"); R()
    if abs(b_lo - lo) < 0.15 * abs(diff) and (hi < 0) == (b_hi < 0):
        R("Normal and BCa intervals agree.")
    else:
        R(f"**Normal and BCa intervals disagree.** Normal [{lo:+.4f}, {hi:+.4f}] "
          f"{'excludes' if hi < 0 else 'includes'} zero; BCa [{b_lo:+.4f}, {b_hi:+.4f}] "
          f"{'excludes' if b_hi < 0 else 'includes'} zero. The pre-registration "
          "names the normal interval as primary, so **it governs the ship "
          "decision**. Switching to whichever interval gives the preferred answer "
          "is what pre-registration exists to prevent.")
    R()

    R("## CUPED, as multivariate regression adjustment"); R()
    Y = acc.Y.to_numpy()
    T = (acc.arm == "challenger").astype(float).to_numpy()
    Xc = acc[["pre_value", "pre_n", "pre_fraud", "pre_mean_amt", "has_pre_period"]].to_numpy()

    def ancova(Y, T, X):
        D = np.column_stack([np.ones(len(Y)), T, X])
        inv = np.linalg.pinv(D.T @ D)
        beta = inv @ D.T @ Y
        r = Y - D @ beta
        V = inv @ ((D * r[:, None]).T @ (D * r[:, None])) @ inv * (len(Y) / (len(Y) - D.shape[1]))
        return beta[1], np.sqrt(V[1, 1])

    b_un, se_un = ancova(Y, T, np.empty((len(Y), 0)))
    b_ad, se_ad = ancova(Y, T, Xc)
    vr = 1 - (se_ad / se_un) ** 2
    R.block(f"unadjusted  effect ${b_un:>+9.4f}   SE ${se_un:.4f}\n"
            f"adjusted    effect ${b_ad:>+9.4f}   SE ${se_ad:.4f}\n"
            f"variance reduction {vr*100:>8.2f}%   (registered expectation 4.9%)"); R()
    R(f"Realised reduction {vr*100:.2f}% against a registered 4.9%. Pre-period "
      f"coverage is {acc.has_pre_period.mean()*100:.1f}%, the binding constraint. "
      "**A null result for CUPED on this data**, registered as likely in advance."); R()

    R("## Guardrail: false decline rate against the 8.0% ceiling"); R()
    guard_pass = False
    for arm, col in (("challenger", "dec_chal"), ("champion", "dec_champ")):
        sub = te[(te.arm == arm) & (te.is_fraud == 0)]
        g = sub.groupby("account_id").agg(d=(col, "sum"), n=(col, "size"))
        fdr, fse = core.ratio_se_clustered(g.d.to_numpy().astype(float),
                                           g.n.to_numpy().astype(float))
        if arm == "challenger":
            ub = fdr + 1.645 * fse
            z = (fdr - GUARDRAIL_CEILING) / fse
            guard_pass = ub < GUARDRAIL_CEILING
            R(f"**{arm}** threshold {CHALLENGER_THR}"); R()
            R.block(f"false decline rate   {fdr*100:.4f}%\n"
                    f"cluster-robust SE    {fse*100:.4f}pp   (clusters = {len(g):,} accounts)\n"
                    f"one-sided 95% UB     {ub*100:.4f}%\nceiling              8.0%\n"
                    f"H0: FDR >= 8.0%      z {z:.3f}   p {stats.norm.cdf(z):.3e}\n"
                    f"verdict              {'PASS, below the ceiling' if guard_pass else 'BREACH'}")
        else:
            R(f"Champion arm, for reference: false decline rate {fdr*100:.4f}% "
              f"(SE {fse*100:.4f}pp).")
        R()

    R("## Secondary metrics"); R()
    rows = []
    for arm, dcol, ccol in (("challenger", "dec_chal", "c_chal"),
                            ("champion", "dec_champ", "c_champ")):
        s = te[te.arm == arm]
        d, yy = s[dcol].to_numpy(), s.is_fraud.to_numpy().astype(int)
        aa = s.transaction_amt.to_numpy().astype(float)
        rows.append({"arm": arm, "transactions": len(s), "frauds": int(yy.sum()),
                     "declined": int(d.sum()), "decline_rate": round(d.mean(), 5),
                     "recall": round(d[yy == 1].mean(), 5),
                     "precision": round(yy[d].mean(), 5) if d.sum() else np.nan,
                     "fraud_dollars_caught": round(aa[d & (yy == 1)].sum(), 0),
                     "fraud_dollars_total": round(aa[yy == 1].sum(), 0),
                     "cost_per_txn": round(s[ccol].mean(), 4)})
    sec = pd.DataFrame(rows)
    sec["pct_fraud_dollars_caught"] = (sec.fraud_dollars_caught / sec.fraud_dollars_total * 100).round(2)
    R.table(sec); R()

    R("## Subgroups, descriptive only"); R()
    R("Registered as descriptive and excluded from the ship decision. Two of the "
      "four are underpowered at 15.5% and 14.5%."); R()
    med = acc.mean_amt.median()
    acc["established"], acc["high_ticket"] = acc.has_pre_period > 0, acc.mean_amt >= med
    rows = []
    for col, labs in (("established", ("new", "established")),
                      ("high_ticket", (f"low ticket (< ${med:.2f})", f"high ticket (>= ${med:.2f})"))):
        for v, lab in zip((False, True), labs):
            s = acc[acc[col] == v]
            d, _, l, h = core.welch(s.loc[s.arm == "challenger", "Y"].to_numpy(),
                                    s.loc[s.arm == "champion", "Y"].to_numpy())
            rows.append({"subgroup": lab, "accounts": len(s), "effect_per_account": round(d, 4),
                         "ci_low": round(l, 4), "ci_high": round(h, 4),
                         "scaled_total": round(d * len(s), 0),
                         "crosses_zero": "yes" if l < 0 < h else "no"})
    R.table(pd.DataFrame(rows)); R()

    R("## Effect by account risk quintile"); R()
    R("Registered to lead the recommendation ahead of the average treatment effect."); R()
    acc["q"] = pd.qcut(acc.risk.rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
    rows = []
    for q, s in acc.groupby("q", observed=True):
        b = s.loc[s.arm == "champion", "Y"].to_numpy()
        d, _, l, h = core.welch(s.loc[s.arm == "challenger", "Y"].to_numpy(), b)
        rows.append({"quintile": int(q), "accounts": len(s), "mean_risk": round(s.risk.mean(), 5),
                     "mean_cost_champion": round(b.mean(), 3), "effect_per_account": round(d, 4),
                     "ci_low": round(l, 3), "ci_high": round(h, 3),
                     "scaled_total": round(d * len(s), 0)})
    R.table(pd.DataFrame(rows)); R()

    R("## Targeted application follow-up"); R()
    R("Challenger only above a risk cutoff, champion below. A paired "
      "counterfactual, since both outcomes are known for every account."); R()
    te["acct_risk"] = te.account_id.map(acc.risk)
    legit_m = te.is_fraud.to_numpy() == 0
    amt_t = te.transaction_amt.to_numpy().astype(float)
    dc, dch = te.dec_champ.to_numpy(), te.dec_chal.to_numpy()
    rows = []
    for cut in [0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.20, 1.01]:
        use = acc.risk >= cut
        dec = np.where(te.acct_risk.to_numpy() >= cut, dch, dc)
        rows.append({"risk_cutoff": cut, "accounts_on_challenger": int(use.sum()),
                     "pct_accounts": round(use.mean() * 100, 2),
                     "total_cost": round(np.where(use, acc.y_chal, acc.y_champ).sum(), 0),
                     "legit_declined": int((dec & legit_m).sum()),
                     "false_decline_rate_pct": round((dec & legit_m).sum() / legit_m.sum() * 100, 3),
                     "fraud_dollars_caught": round(amt_t[dec & ~legit_m].sum(), 0)})
    tg = pd.DataFrame(rows)
    all_chal = acc.y_chal.sum()
    tg["vs_challenger_everywhere"] = (tg.total_cost - all_chal).round(0)
    R.table(tg); R()
    best, uni = tg.loc[tg.total_cost.idxmin()], tg[tg.risk_cutoff == 0.0].iloc[0]
    r12 = tg[tg.risk_cutoff == 0.12].iloc[0]
    R(f"Cheapest cutoff: **{best.risk_cutoff}**, challenger on {best.pct_accounts}% "
      f"of accounts, ${best.total_cost:,.0f} against ${all_chal:,.0f} uniform."); R()
    R(f"**Targeting buys almost nothing at the cost-optimal cutoff.** It declines "
      f"{int(best.legit_declined):,} legitimate transactions against "
      f"{int(uni.legit_declined):,}, a "
      f"{(1-best.legit_declined/uni.legit_declined)*100:.1f}% reduction, catching "
      "identical fraud dollars. Accounts below 0.02 mean risk rarely cross a 0.116 "
      "threshold anyway. The registered expectation that targeting would dominate "
      "is **not supported**."); R()
    R(f"There is a real trade higher up, but it costs money. At 0.12, false "
      f"declines fall to {int(r12.legit_declined):,} "
      f"({(1-r12.legit_declined/uni.legit_declined)*100:.0f}% lower) for "
      f"${r12.total_cost-all_chal:+,.0f} of cost and "
      f"${r12.fraud_dollars_caught-uni.fraud_dollars_caught:+,.0f} less fraud caught."); R()

    R("## Paired counterfactual benchmark"); R()
    d = acc.tau.to_numpy(); pse = d.std(ddof=1) / np.sqrt(len(d))
    R("Available only because this is a replay. An efficiency benchmark and a bug "
      "detector: material disagreement would indicate a fault in assignment or "
      "analysis."); R()
    R.block(f"paired effect      ${d.mean():+.4f} per account\npaired SE          ${pse:.4f}\n"
            f"paired 95% CI      [${d.mean()-1.96*pse:+.4f}, ${d.mean()+1.96*pse:+.4f}]\n"
            f"randomised effect  ${diff:+.4f}   SE ${se:.4f}\n"
            f"efficiency ratio   {se/pse:.2f}x\n"
            f"agreement          {'consistent' if abs(d.mean()-diff) < 1.96*se else 'DISAGREE'}"); R()

    R("## Ship recommendation"); R()
    c1, c2, c3 = p_primary < 0.05 and diff < 0, hi < 0, guard_pass
    R.block(f"1 primary significant, challenger favoured   {'PASS' if c1 else 'FAIL'}\n"
            f"2 upper bound of 95% CI below zero           {'PASS' if c2 else 'FAIL'}\n"
            f"3 guardrail below the 8.0% ceiling at its UB {'PASS' if c3 else 'FAIL'}\n"
            f"decision                                     "
            f"{'SHIP' if all((c1, c2, c3)) else 'DO NOT SHIP'}"); R()
    json.dump({"effect_per_account": float(diff), "se": float(se),
               "ci": [float(lo), float(hi)], "bca": [float(b_lo), float(b_hi)],
               "n_chal": n_chal, "n_champ": n_champ, "total_effect": float(diff * N)},
              open(REPORTS / "_stage2_truth.json", "w"), indent=2)
    R.write()


# --------------------------------------------------------------------------- #
# observational half
# --------------------------------------------------------------------------- #

def _match_att(Y, T, X):
    """1:1 nearest neighbour on the propensity logit, with replacement."""
    ps = np.clip(LogisticRegression(max_iter=2000).fit(X, T).predict_proba(X)[:, 1],
                 1e-6, 1 - 1e-6)
    logit = np.log(ps / (1 - ps))
    caliper = CALIPER_SD * logit.std(ddof=1)
    ti, ci = np.flatnonzero(T == 1), np.flatnonzero(T == 0)
    keep = (logit[ti] >= logit[ci].min()) & (logit[ti] <= logit[ci].max())
    dropped_support = int((~keep).sum()); ti = ti[keep]

    order = np.argsort(logit[ci], kind="stable")
    cs, cl = ci[order], logit[ci][order]
    pos = np.searchsorted(cl, logit[ti])
    best = np.empty(len(ti), np.int64); bestd = np.full(len(ti), np.inf)
    for off in (-1, 0):
        j = np.clip(pos + off, 0, len(cl) - 1)
        d = np.abs(cl[j] - logit[ti])
        take = d < bestd
        best[take], bestd[take] = cs[j][take], d[take]
    within = bestd <= caliper
    dropped_caliper = int((~within).sum())
    ti, mc = ti[within], best[within]
    diffs = Y[ti] - Y[mc]
    return {"att": diffs.mean(), "diffs": diffs, "ps": ps, "treated_used": len(ti),
            "matched_controls_unique": len(np.unique(mc)),
            "dropped_support": dropped_support, "dropped_caliper": dropped_caliper,
            "caliper": caliper, "treated_idx": ti, "control_idx": mc}


def _boot_att(Y, T, X, seed=core.SEED):
    """Bootstrap over treated units. Not formally valid for NN matching
    (Abadie and Imbens 2008); indicative only."""
    rng = np.random.default_rng(seed)
    d = _match_att(Y, T, X)["diffs"]
    reps = np.array([rng.choice(d, len(d), True).mean() for _ in range(N_BOOT_PSM)])
    return np.percentile(reps, 2.5), np.percentile(reps, 97.5)


def _rosenbaum(diffs, gammas=(1.0, 1.1, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0, 6.0)):
    d = diffs[diffs != 0]
    if len(d) < 10:
        return pd.DataFrame()
    r = stats.rankdata(np.abs(d))
    W, S, S2 = r[d < 0].sum(), r.sum(), (r ** 2).sum()
    rows = []
    for g in gammas:
        pp, pm = g / (1 + g), 1 / (1 + g)
        zw = (W - pm * S) / np.sqrt(pm * (1 - pm) * S2)
        zb = (W - pp * S) / np.sqrt(pp * (1 - pp) * S2)
        rows.append({"gamma": g, "p_upper_bound": float(1 - stats.norm.cdf(zw)),
                     "p_lower_bound": float(1 - stats.norm.cdf(zb))})
    return pd.DataFrame(rows)


def observational():
    R = Report(REPORTS / "stage2_observational_results.md")
    acc = pd.read_parquet(REPORTS / "_stage2_accounts.parquet")
    truth_r = json.load(open(REPORTS / "_stage2_truth.json"))
    df = pd.read_parquet(core.FCT, columns=FCT_COLS)
    te = df[df.txn_day >= REPLAY_LO]
    first = te.sort_values(["account_id", "transaction_id"]).groupby("account_id").first()
    acc["first_amt"], acc["first_hasid"] = first.transaction_amt, first.has_identity.astype(float)
    acc["first_prod"], acc["first_card6"] = first.product_cd.astype(str), first.card6.astype(str)

    y_chal, y_champ = acc.y_chal.to_numpy(), acc.y_champ.to_numpy()
    tau, risk, N = y_chal - y_champ, acc.risk.to_numpy(), len(acc)
    z = (risk - risk.mean()) / risk.std(ddof=1)
    ATE = tau.mean()

    base = ["pre_value", "pre_n", "pre_fraud", "has_pre_period", "n_txn", "tot_amt",
            "first_amt", "first_hasid"]
    dm = pd.get_dummies(acc[["first_prod", "first_card6"]], dummy_na=True, dtype=float)
    Xobs = np.column_stack([acc[base].to_numpy(float), dm.to_numpy()])
    Xobs = (Xobs - Xobs.mean(0)) / np.where(Xobs.std(0) > 0, Xobs.std(0), 1)
    Xora = np.column_stack([Xobs, z])

    R("# Stage 2, observational half"); R()
    R("Implements section 10 of the pre-registration. Simulates a non-random "
      "rollout steered by account risk, then recovers the effect the way a "
      "post-hoc analyst would."); R()
    R("## The two targets, and why both are needed"); R()
    R(f"Both potential outcomes are known for every account, so the **exact ATE "
      f"is computable: ${ATE:+.4f} per account** (${ATE*N:+,.0f} scaled). The "
      f"randomised arm estimated ${truth_r['effect_per_account']:+.4f} "
      f"(SE ${truth_r['se']:.4f}), one noisy draw at that quantity."); R()
    R("Matching estimates the effect **on the treated**. The effect is strongly "
      "heterogeneous, so under a risk-steered rollout the ATT differs from the "
      "ATE by construction. Judging an ATT estimate against the ATE would charge "
      "estimand mismatch to confounding. Both are reported."); R()

    all_rows, diag_rows, ros = [], [], {}
    ps_store = {}
    for beta in BETAS:
        alpha = brentq(lambda a: (1.0 / (1.0 + np.exp(-(a + beta * z)))).mean() - 0.5, -50, 50)
        pr = 1 / (1 + np.exp(-(alpha + beta * z)))
        u = np.array([core.account_hash(a, SALT, f"obs{beta}") for a in acc.index])
        T = (u < pr).astype(int)
        Y = np.where(T == 1, y_chal, y_champ)
        ATT = tau[T == 1].mean()
        naive = Y[T == 1].mean() - Y[T == 0].mean()
        mo, mr = _match_att(Y, T, Xobs), _match_att(Y, T, Xora)
        lo_o, hi_o = _boot_att(Y, T, Xobs)
        lo_r, hi_r = _boot_att(Y, T, Xora)
        label = f"beta {beta:+.1f} ({'risk-seeking' if beta > 0 else 'risk-averse'})"
        ps_store[label] = (mo["ps"], T)
        d_ = mo["diffs"]
        ros[label] = (_rosenbaum(d_), {
            "pairs": len(d_), "zero_pct": (d_ == 0).mean() * 100,
            "nonzero_neg_pct": (d_[d_ != 0] < 0).mean() * 100 if (d_ != 0).any() else np.nan,
            "top1pct_share": np.sort(np.abs(d_))[-max(len(d_) // 100, 1):].sum()
                             / max(np.abs(d_).sum(), 1e-9) * 100,
            "mean": d_.mean(), "median": float(np.median(d_))})

        for est, val, lo, hi in (("naive difference", naive, np.nan, np.nan),
                                 ("PSM, observed covariates", mo["att"], lo_o, hi_o),
                                 ("PSM, oracle (includes risk)", mr["att"], lo_r, hi_r)):
            all_rows.append({"scenario": label, "beta": beta, "estimator": est,
                             "estimate": round(val, 4),
                             "ci_low": round(lo, 4) if lo == lo else np.nan,
                             "ci_high": round(hi, 4) if hi == hi else np.nan,
                             "true_ATT": round(ATT, 4), "true_ATE": round(ATE, 4),
                             "bias_vs_ATT": round(val - ATT, 4),
                             "bias_pct_vs_ATT": round((val - ATT) / abs(ATT) * 100, 1) if ATT else np.nan,
                             "bias_vs_ATE": round(val - ATE, 4),
                             "sign_correct": "yes" if np.sign(val) == np.sign(ATE) else "NO"})
        for nm, m, Xm in (("observed", mo, Xobs), ("oracle", mr, Xora)):
            sel = np.concatenate([m["treated_idx"], m["control_idx"]])
            lab = np.concatenate([np.ones(len(m["treated_idx"])), np.zeros(len(m["control_idx"]))])
            diag_rows.append({
                "scenario": label, "psm": nm, "treated_matched": m["treated_used"],
                "unique_controls": m["matched_controls_unique"],
                "dropped_support": m["dropped_support"], "dropped_caliper": m["dropped_caliper"],
                "caliper": round(m["caliper"], 4),
                "mean_abs_SMD_before": round(np.mean([abs(core.smd(Xm[:, j], T))
                                                      for j in range(Xm.shape[1])]), 4),
                "mean_abs_SMD_after": round(np.mean([abs(core.smd(Xm[sel, j], lab))
                                                     for j in range(Xm.shape[1])]), 4)})

    res, dia = pd.DataFrame(all_rows), pd.DataFrame(diag_rows)
    res.to_csv(REPORTS / "stage2_observational_estimates.csv", index=False)
    dia.to_csv(REPORTS / "stage2_observational_diagnostics.csv", index=False)

    R("## Results by scenario"); R()
    R(f"Exact ATE across all accounts: **${ATE:+.4f} per account**."); R()
    for label in res.scenario.unique():
        s = res[res.scenario == label]
        R(f"**{label}**, true ATT ${s.true_ATT.iloc[0]:+.4f}"); R()
        R.table(s[["estimator", "estimate", "ci_low", "ci_high", "bias_vs_ATT",
                   "bias_pct_vs_ATT", "bias_vs_ATE", "sign_correct"]]); R()

    R("## Bias summary"); R()
    R.block(res.pivot_table(index="beta", columns="estimator",
                            values="bias_vs_ATT").round(4).to_string()); R()
    R("Bias against the ATT, dollars per account. The oracle column is the "
      "control: the same machinery plus the one variable that drove assignment."); R()

    R("## Matching diagnostics"); R(); R.table(dia); R()

    R("## Rosenbaum sensitivity, PSM on observed covariates"); R()
    rows = []
    for label, (tab, st) in ros.items():
        g = "n/a"
        if len(tab):
            crit = tab[tab.p_upper_bound > 0.05]
            g = crit.gamma.min() if len(crit) else ">6.0"
        rows.append({"scenario": label, "gamma_to_overturn": g, "matched_pairs": st["pairs"],
                     "pct_pairs_exactly_zero": round(st["zero_pct"], 1),
                     "pct_of_nonzero_pairs_negative": round(st["nonzero_neg_pct"], 1),
                     "pct_of_total_abs_diff_in_top_1pct": round(st["top1pct_share"], 1),
                     "mean_pair_diff": round(st["mean"], 4),
                     "median_pair_diff": round(st["median"], 4)})
    R.table(pd.DataFrame(rows)); R()
    R("**The Rosenbaum bounds are uninformative here.** 83-95% of matched pairs "
      "differ by exactly zero, and the top 1% of absolute differences carries "
      "over half the mass. A signed-rank statistic counts pairs; this effect "
      "lives in the size of a few. Reported as inapplicable rather than dropped, "
      "since the pre-registration required it."); R()

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0), facecolor=SURFACE)
    ax = axes[0]; ax.set_facecolor(SURFACE)
    for est, col, mk in (("naive difference", C_B, "o"),
                         ("PSM, observed covariates", C_A, "s"),
                         ("PSM, oracle (includes risk)", C_C, "^")):
        s = res[res.estimator == est].sort_values("beta")
        ax.plot(s.beta, s.bias_vs_ATT, color=col, lw=2, marker=mk, ms=9,
                markeredgecolor=SURFACE, markeredgewidth=2, label=est)
    ax.axhline(0, color=GRIDC, lw=2, ls="--")
    _st(ax, "Bias by estimator and confounding strength",
        "confounding strength beta (negative = risk-averse rollout)",
        "bias against the true ATT ($ per account)")
    leg = ax.legend(frameon=False, fontsize=9)
    for t in leg.get_texts():
        t.set_color(INK_2)
    ax = axes[1]; ax.set_facecolor(SURFACE)
    ps, T = ps_store["beta +3.0 (risk-seeking)"]
    bins = np.linspace(0, 1, 41)
    ax.hist(ps[T == 1], bins=bins, alpha=0.65, color=C_A, label="treated")
    ax.hist(ps[T == 0], bins=bins, alpha=0.65, color=C_B, label="control")
    _st(ax, "Propensity overlap, strongest confounding (beta +3.0)",
        "estimated propensity score", "accounts")
    leg = ax.legend(frameon=False, fontsize=9)
    for t in leg.get_texts():
        t.set_color(INK_2)
    fig.tight_layout()
    fig.savefig(REPORTS / "stage2_observational.png", dpi=170, facecolor=SURFACE)
    R("Chart: `reports/stage2_observational.png`."); R()
    R.write()


def _st(ax, title, xlabel, ylabel):
    ax.set_xlabel(xlabel, color=INK_2, fontsize=10)
    ax.set_ylabel(ylabel, color=INK_2, fontsize=10)
    ax.set_title(title, color=INK, fontsize=12, weight="bold", loc="left", pad=12)
    ax.grid(True, color=GRIDC, lw=0.8, alpha=0.7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRIDC)
    ax.tick_params(colors=INK_2, labelsize=9)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "randomized"
    {"randomized": randomized, "observational": observational}[cmd]()
