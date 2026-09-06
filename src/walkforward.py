"""Walk-forward evaluation harness.

Stage 1 evaluated on a single 32-day window and chose the decline threshold on
that same window. Both are fixed here:

1. Four non-overlapping 16-day test windows spanning days 118-181, each with
   its own train / early-stopping / calibration windows behind it. Reporting
   four folds gives a between-fold spread instead of one number.

2. The threshold is chosen out of sample. Isotonic is cross-fitted within the
   calibration window to produce out-of-fold calibrated probabilities, the
   threshold is swept on those, and only then is it applied to the test
   window. The stage 1 protocol swept on test, which is optimistic. The
   test-optimal (oracle) threshold is still reported, so the size of that
   optimism is visible rather than hidden.

Fold layout, for test start T: train < T-32, early stop [T-32, T-16),
calibrate [T-16, T), test [T, T+16).
"""

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
from decision_layer import ASSUMPTIONS as A

FOLD_STARTS = (118, 134, 150, 166)
TEST_LEN = 16
CAL_LEN = 16
ES_LEN = 16
SEED = 42

GRID = np.unique(np.concatenate([np.linspace(0.001, 0.99, 990),
                                 np.geomspace(0.001, 0.99, 400)]))

LGB_PARAMS = dict(
    objective="binary", n_estimators=3000, learning_rate=0.05, num_leaves=63,
    min_child_samples=50, subsample=0.8, subsample_freq=1,
    colsample_bytree=0.8, reg_lambda=1.0, random_state=SEED, n_jobs=-1,
    verbose=-1)


def cost_of(mask, y, amt):
    """Total dollar cost of declining where mask is True."""
    fn = ~mask & (y == 1)
    fp = mask & (y == 0)
    return (amt[fn].sum() + A["chargeback_fee_usd"] * fn.sum()
            + A["margin_rate"] * amt[fp].sum()
            + A["churn_prob_after_false_decline"] * A["account_lifetime_value_usd"] * fp.sum()
            + A["manual_review_cost_usd"] * mask.sum())


def best_threshold(p, y, amt):
    costs = np.array([cost_of(p >= t, y, amt) for t in GRID])
    i = int(costs.argmin())
    return float(GRID[i]), float(costs[i])


def encode(X):
    """Object columns to integer codes; LightGBM is told which are categorical."""
    X = X.copy()
    cats = []
    for c in X.columns:
        if X[c].dtype == object or str(X[c].dtype) == "string":
            X[c] = X[c].astype("category").cat.codes.astype("int32") + 1
            cats.append(c)
        elif X[c].dtype == bool:
            X[c] = X[c].astype("int8")
        else:
            X[c] = pd.to_numeric(X[c], errors="coerce").astype("float32")
    return X, cats


def run_fold(X, y, day, amt, T, params=None, verbose=False, transform=None):
    tr = day < T - ES_LEN - CAL_LEN
    es = (day >= T - ES_LEN - CAL_LEN) & (day < T - CAL_LEN)
    ca = (day >= T - CAL_LEN) & (day < T)
    te = (day >= T) & (day < T + TEST_LEN)

    # Any fitted transform (count encoders, etc.) sees only pre-test rows, so
    # nothing from the evaluation window leaks into the features.
    if transform is not None:
        X = transform(X, tr | es | ca)

    Xc, cats = encode(X)
    clf = lgb.LGBMClassifier(**{**LGB_PARAMS, **(params or {})})
    t0 = time.perf_counter()
    clf.fit(Xc[tr], y[tr], eval_set=[(Xc[es], y[es])],
            eval_metric="average_precision", categorical_feature=cats,
            callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
    fit_secs = time.perf_counter() - t0

    # cross-fitted isotonic inside the calibration window, so the threshold is
    # picked on probabilities the calibrator did not see
    ca_idx = np.flatnonzero(ca)
    rng = np.random.default_rng(SEED)
    half = rng.permutation(len(ca_idx)) < len(ca_idx) // 2
    p_ca_oof = np.empty(len(ca_idx))
    for m in (half, ~half):
        fit_idx, score_idx = ca_idx[m], ca_idx[~m]
        c = CalibratedClassifierCV(FrozenEstimator(clf), method="isotonic")
        c.fit(Xc.iloc[fit_idx], y[fit_idx])
        p_ca_oof[~m] = c.predict_proba(Xc.iloc[score_idx])[:, 1]

    thr, _ = best_threshold(p_ca_oof, y[ca_idx], amt[ca_idx])

    cal = CalibratedClassifierCV(FrozenEstimator(clf), method="isotonic")
    cal.fit(Xc[ca], y[ca])
    p_te = cal.predict_proba(Xc[te])[:, 1]
    p_raw = clf.predict_proba(Xc[te])[:, 1]
    p_tr = clf.predict_proba(Xc[tr])[:, 1]

    yte, ate = y[te], amt[te]
    thr_oracle, cost_oracle = best_threshold(p_te, yte, ate)
    dec = p_te >= thr
    legit = yte == 0

    return {
        "fold_test_start": T, "n_train": int(tr.sum()), "n_test": int(te.sum()),
        "n_fraud_test": int(yte.sum()), "fit_secs": round(fit_secs, 1),
        "best_iter": int(clf.best_iteration_),
        "test_pr_auc": average_precision_score(yte, p_raw),
        "test_pr_auc_cal": average_precision_score(yte, p_te),
        "test_roc_auc": roc_auc_score(yte, p_raw),
        "test_brier_cal": brier_score_loss(yte, p_te),
        "train_pr_auc": average_precision_score(y[tr], p_tr),
        "threshold": thr,
        "cost": cost_of(dec, yte, ate),
        "cost_at_half": cost_of(p_te >= 0.5, yte, ate),
        "cost_approve_all": cost_of(np.zeros(len(yte), bool), yte, ate),
        "cost_oracle": cost_oracle, "threshold_oracle": thr_oracle,
        "fraud_dollars_caught": float(ate[dec & (yte == 1)].sum()),
        "fraud_dollars_total": float(ate[yte == 1].sum()),
        "false_decline_rate": float(dec[legit].mean()),
        "recall": float(dec[yte == 1].mean()),
        "_p_te": p_te, "_y_te": yte, "_amt_te": ate, "_thr": thr, "_clf": clf,
    }


def run_config(name, X, y, day, amt, params=None, folds=FOLD_STARTS, keep=False,
               transform=None):
    rows, detail = [], []
    for T in folds:
        r = run_fold(X, y, day, amt, T, params, transform=transform)
        detail.append(r)
        rows.append({k: v for k, v in r.items() if not k.startswith("_")})
        print(f"  {name} fold T={T}: PR-AUC {r['test_pr_auc']:.4f} "
              f"cost ${r['cost']:,.0f} thr {r['threshold']:.4f} ({r['fit_secs']}s)")
    df = pd.DataFrame(rows)
    df.insert(0, "config", name)
    return (df, detail) if keep else (df, None)


def summarise(df):
    """Fold-level mean and spread, plus a pooled cost total."""
    num = df.select_dtypes("number")
    out = pd.DataFrame({"mean": num.mean(), "sd": num.std(),
                        "min": num.min(), "max": num.max()}).round(6)
    return out


def paired_fold_test(a, b, col="cost"):
    """Paired comparison across folds. Few folds, so report the spread too."""
    d = a[col].to_numpy() - b[col].to_numpy()
    n = len(d)
    se = d.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
    return {"mean_diff": d.mean(), "sd_diff": d.std(ddof=1) if n > 1 else np.nan,
            "se": se, "t": d.mean() / se if se and se > 0 else np.nan,
            "n_folds": n, "per_fold": d}
