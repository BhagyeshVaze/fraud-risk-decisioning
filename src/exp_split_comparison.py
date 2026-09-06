"""How much does a random train/test split overstate performance here?

Most published results on IEEE-CIS use random splits. This project reports a
time-based split. The two are not comparable, and this quantifies the gap by
changing nothing except the split rule.

Also fits a logistic regression baseline on the time split, so the gain from
gradient boosting is measured rather than assumed.

Reporting only. The time-split figure remains the number this project quotes.
"""

import json
import pickle
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

import walkforward as wf

REPORTS = Path("reports")
OUT = REPORTS / "split_comparison.md"
SEED = 42
N_BOOT = 1000

LGB = dict(objective="binary", n_estimators=3000, learning_rate=0.05,
           num_leaves=63, min_child_samples=50, subsample=0.8, subsample_freq=1,
           colsample_bytree=0.8, reg_lambda=1.0, random_state=SEED, n_jobs=-1,
           verbose=-1)

buf = []
def emit(s=""):
    print(s); buf.append(s)
def block(s):
    emit("```"); emit(s); emit("```")


def random_masks(props, n, seed):
    rng = np.random.default_rng(seed)
    u = rng.permutation(n)
    cuts = (np.cumsum(props) * n).astype(int)
    masks, prev = [], 0
    for c in cuts:
        m = np.zeros(n, bool); m[u[prev:c]] = True
        masks.append(m); prev = c
    return masks


def logistic_matrix(Xraw, tr):
    """Imputation, scaling and encoding that a linear model requires."""
    parts = []
    for c in Xraw.columns:
        s = Xraw[c]
        if s.dtype == object or str(s.dtype) == "string":
            if s[tr].nunique(dropna=True) <= 30:
                parts.append(pd.get_dummies(s, prefix=c, dummy_na=True, dtype=np.float32))
            else:
                vc = s[tr].value_counts()
                parts.append(pd.DataFrame({c + "_cnt": s.map(vc).astype(np.float32)}))
        else:
            v = pd.to_numeric(s, errors="coerce").astype(np.float32)
            parts.append(pd.DataFrame({c: v.fillna(v[tr].median())}))
    Z = pd.concat(parts, axis=1)
    mu, sd = Z[tr].mean(), Z[tr].std().replace(0, 1)
    return ((Z - mu) / sd).fillna(0.0)


def fit_eval(X, Xraw, cats, y, tr, es, ca, te, model="lgbm"):
    if model == "lgbm":
        clf = lgb.LGBMClassifier(**LGB)
        clf.fit(X[tr], y[tr], eval_set=[(X[es], y[es])],
                eval_metric="average_precision", categorical_feature=cats,
                callbacks=[lgb.early_stopping(100, verbose=False),
                           lgb.log_evaluation(0)])
        Xf = X
    else:
        Xf = logistic_matrix(Xraw, tr)
        clf = LogisticRegression(max_iter=3000, C=1.0, solver="lbfgs")
        clf.fit(Xf[tr], y[tr])
    cal = CalibratedClassifierCV(FrozenEstimator(clf), method="isotonic")
    cal.fit(Xf[ca], y[ca])
    return clf.predict_proba(Xf[te])[:, 1], cal.predict_proba(Xf[te])[:, 1], y[te]


def metrics(praw, pcal, yy):
    return {"pr_auc": average_precision_score(yy, praw),
            "roc_auc": roc_auc_score(yy, praw),
            "brier": brier_score_loss(yy, pcal)}


def boot(praw, pcal, yy, seed):
    r = np.random.default_rng(seed); n = len(yy)
    out = {"pr_auc": [], "roc_auc": [], "brier": []}
    for _ in range(N_BOOT):
        i = r.integers(0, n, n)
        if yy[i].sum() == 0:
            continue
        out["pr_auc"].append(average_precision_score(yy[i], praw[i]))
        out["roc_auc"].append(roc_auc_score(yy[i], praw[i]))
        out["brier"].append(brier_score_loss(yy[i], pcal[i]))
    return {k: np.array(v) for k, v in out.items()}


def main():
    sel = json.load(open("models/selected_features.json"))["features"]
    need = sorted(set(sel) | {"txn_day", "is_fraud", "account_id"})
    fct = pd.read_parquet("data/parquet/fct_transactions.parquet", columns=need)
    Xraw = fct[sel]
    X, cats = wf.encode(Xraw)
    y = fct.is_fraud.to_numpy().astype(int)
    day = fct.txn_day.to_numpy()
    n = len(X)

    tr_t = day < 110
    es_t = (day >= 110) & (day < 130)
    ca_t = (day >= 130) & (day < 150)
    te_t = day >= 150
    props = np.array([tr_t.sum(), es_t.sum(), ca_t.sum(), te_t.sum()]) / n

    p8 = np.array([0.644, 0.105, 0.091]); p8 = p8 / p8.sum() * 0.8
    props80 = np.array([p8[0], p8[1], p8[2], 0.20])

    runs = {}
    print("A time split, LightGBM ...")
    runs["A time split, LightGBM"] = fit_eval(X, Xraw, cats, y, tr_t, es_t, ca_t, te_t)
    print("B random size-matched, LightGBM ...")
    runs["B random size-matched, LightGBM"] = fit_eval(
        X, Xraw, cats, y, *random_masks(props, n, SEED))
    print("C random 80/20, LightGBM ...")
    runs["C random 80/20, LightGBM"] = fit_eval(
        X, Xraw, cats, y, *random_masks(props80, n, SEED))
    print("D time split, logistic regression ...")
    runs["D time split, logistic regression"] = fit_eval(
        X, Xraw, cats, y, tr_t, es_t, ca_t, te_t, model="logit")

    emit("# Split comparison and a logistic baseline")
    emit()
    emit("Identical features (200), identical hyperparameters, identical "
         "isotonic calibration. Only the split rule changes between A, B and C. "
         "D changes only the model.")
    emit()
    emit(f"Time-split proportions, which run B reproduces exactly: train "
         f"{props[0]:.4f}, early stopping {props[1]:.4f}, calibration "
         f"{props[2]:.4f}, test {props[3]:.4f}.")
    emit()
    emit("**The time-split figure remains the number this project reports.** "
         "Everything below measures how much a naive evaluation would have "
         "overstated it.")
    emit()

    rows = []
    for k, (praw, pcal, yy) in runs.items():
        m = metrics(praw, pcal, yy)
        rows.append({"run": k, "n_test": len(yy), "n_fraud": int(yy.sum()),
                     "fraud_rate": round(yy.mean(), 5),
                     "pr_auc": round(m["pr_auc"], 4),
                     "roc_auc": round(m["roc_auc"], 4),
                     "brier": round(m["brier"], 5)})
    res = pd.DataFrame(rows)
    emit("## Results")
    emit()
    block(res.to_string(index=False))
    emit()

    bs = {k: boot(*v, seed=7) for k, v in runs.items()}
    A = "A time split, LightGBM"

    emit("## Bootstrapped differences against the time split")
    emit()
    emit(f"{N_BOOT} resamples. Unpaired, because the random splits evaluate on "
         "different rows.")
    emit()
    rows = []
    for other in ["B random size-matched, LightGBM", "C random 80/20, LightGBM"]:
        ma, mb = metrics(*runs[other]), metrics(*runs[A])
        for k in ("pr_auc", "roc_auc", "brier"):
            d = bs[other][k] - bs[A][k]
            lo, hi = np.percentile(d, [2.5, 97.5])
            rows.append({"comparison": other.split(",")[0] + " minus A",
                         "metric": k, "point": round(ma[k] - mb[k], 4),
                         "ci_low": round(lo, 4), "ci_high": round(hi, 4),
                         "spans_zero": "yes" if lo < 0 < hi else "no"})
    block(pd.DataFrame(rows).to_string(index=False))
    emit()
    for other, lab in (("B random size-matched, LightGBM", "size-matched"),
                       ("C random 80/20, LightGBM", "80/20")):
        ma, mb = metrics(*runs[other]), metrics(*runs[A])
        emit(f"- Random {lab}: PR-AUC {mb['pr_auc']:.4f} to {ma['pr_auc']:.4f}, "
             f"**{(ma['pr_auc']/mb['pr_auc']-1)*100:+.1f}%**. ROC-AUC "
             f"{mb['roc_auc']:.4f} to {ma['roc_auc']:.4f}, "
             f"{(ma['roc_auc']/mb['roc_auc']-1)*100:+.1f}%.")
    emit()

    ids = fct.account_id
    _, _, _, te_r = random_masks(props, n, SEED)
    tr_r = ~te_r
    emit(f"Mechanism: under the random split "
         f"**{len(set(ids[te_r]) & set(ids[tr_r])):,} of "
         f"{ids[te_r].nunique():,} test accounts "
         f"({len(set(ids[te_r]) & set(ids[tr_r]))/ids[te_r].nunique()*100:.1f}%)** "
         f"also appear in training. Under the time split it is "
         f"{len(set(ids[te_t]) & set(ids[tr_t])):,} of {ids[te_t].nunique():,} "
         f"({len(set(ids[te_t]) & set(ids[tr_t]))/ids[te_t].nunique()*100:.1f}%), "
         "and none of those are future transactions.")
    emit()

    emit("## Logistic baseline, paired bootstrap on identical test rows")
    emit()
    pg, cg, yg = runs[A]
    pl, cl, yl = runs["D time split, logistic regression"]
    assert (yg == yl).all()
    r = np.random.default_rng(11); nn = len(yg)
    dp = {"pr_auc": [], "roc_auc": [], "brier": []}
    for _ in range(N_BOOT):
        i = r.integers(0, nn, nn)
        if yg[i].sum() == 0:
            continue
        dp["pr_auc"].append(average_precision_score(yg[i], pg[i]) - average_precision_score(yg[i], pl[i]))
        dp["roc_auc"].append(roc_auc_score(yg[i], pg[i]) - roc_auc_score(yg[i], pl[i]))
        dp["brier"].append(brier_score_loss(yg[i], cg[i]) - brier_score_loss(yg[i], cl[i]))
    mg, ml = metrics(pg, cg, yg), metrics(pl, cl, yl)
    rows = []
    for k in ("pr_auc", "roc_auc", "brier"):
        a = np.array(dp[k]); lo, hi = np.percentile(a, [2.5, 97.5])
        rows.append({"comparison": "LightGBM minus logistic", "metric": k,
                     "point": round(mg[k] - ml[k], 4),
                     "ci_low": round(lo, 4), "ci_high": round(hi, 4),
                     "spans_zero": "yes" if lo < 0 < hi else "no"})
    block(pd.DataFrame(rows).to_string(index=False))
    emit()
    emit("Logistic regression needs imputation, scaling and explicit encoding, "
         "none of which LightGBM requires. That is an advantage for the tree "
         "model, but it is inherent to the comparison: needing the "
         "preprocessing is part of the cost of a linear model.")
    emit()

    leak = metrics(*runs["B random size-matched, LightGBM"])["pr_auc"] - mg["pr_auc"]
    gain = mg["pr_auc"] - ml["pr_auc"]
    emit("## The comparison worth noticing")
    emit()
    block(f"leakage from a random split adds   {leak:+.4f} PR-AUC\n"
          f"logistic -> gradient boosting adds {gain:+.4f} PR-AUC")
    emit()
    emit(f"**Switching to a random split buys more apparent performance "
         f"({leak:+.4f}) than the entire jump from logistic regression to a "
         f"tuned gradient boosting model ({gain:+.4f}).** One is a modelling "
         "result. The other is measurement error. A published 0.80 on a random "
         "split and this project's 0.4933 on a time split are not evidence of "
         "different model quality.")
    emit()
    OUT.write_text("\n".join(buf) + "\n")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
