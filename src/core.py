"""Shared library: cost model, feature engine, walk-forward harness, reporting.

Rationale for the design decisions here lives in reports/, not in comments.
"""

import hashlib
import os
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

REPORTS = Path("reports")
MODELS = Path("models")
DATA = Path("data/parquet")
FCT = DATA / "fct_transactions.parquet"
SEED = 42

# Assumptions, not measurements. Sensitivity in reports/cost_sensitivity.md.
COSTS = {"chargeback_fee_usd": 25.0, "margin_rate": 0.025,
         "churn_prob_after_false_decline": 0.05,
         "account_lifetime_value_usd": 200.0, "manual_review_cost_usd": 2.0}

THRESHOLD_GRID = np.unique(np.concatenate([np.linspace(0.001, 0.99, 990),
                                           np.geomspace(0.001, 0.99, 400)]))

LGB_PARAMS = dict(objective="binary", n_estimators=3000, learning_rate=0.05,
                  num_leaves=63, min_child_samples=50, subsample=0.8,
                  subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                  random_state=SEED, n_jobs=-1, verbose=-1)

# Production split. Early stopping and calibration use different windows so the
# isotonic map is not fitted on data the model was tuned against.
TRAIN_MAX_DAY, ES_MAX_DAY, CAL_MAX_DAY = 110, 130, 150


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #

class Report:
    """Collects markdown, prints as it goes, writes once at the end."""

    def __init__(self, path=None):
        self.path, self.lines = Path(path) if path else None, []

    def __call__(self, s=""):
        print(s)
        self.lines.append(s)

    def block(self, s):
        self("```"); self(s); self("```")

    def table(self, df, index=False):
        self.block(df.to_string(index=index))

    def write(self, mode="w"):
        with open(self.path, mode) as fh:
            fh.write("\n".join(self.lines) + "\n")
        print(f"\nwrote {self.path}")


# --------------------------------------------------------------------------- #
# cost model
# --------------------------------------------------------------------------- #

def txn_cost(declined, y, amt, c=COSTS):
    """Per-transaction dollar cost of a decline decision."""
    fn, fp = ~declined & (y == 1), declined & (y == 0)
    out = np.zeros(len(y))
    out[fn] = amt[fn] + c["chargeback_fee_usd"]
    out[fp] = (c["margin_rate"] * amt[fp]
               + c["churn_prob_after_false_decline"] * c["account_lifetime_value_usd"])
    return out + c["manual_review_cost_usd"] * declined


def total_cost(declined, y, amt, c=COSTS):
    return txn_cost(declined, y, amt, c).sum()


def cost_at(threshold, p, y, amt, c=COSTS):
    return total_cost(p >= threshold, y, amt, c)


def sweep(p, y, amt, c=COSTS, grid=THRESHOLD_GRID):
    """Cheapest threshold and its cost."""
    costs = np.array([cost_at(t, p, y, amt, c) for t in grid])
    i = int(costs.argmin())
    return float(grid[i]), float(costs[i]), costs


def confusion(threshold, p, y):
    d, f = p >= threshold, y == 1
    tp, fp = int((d & f).sum()), int((d & ~f).sum())
    fn, tn = int((~d & f).sum()), int((~d & ~f).sum())
    return dict(threshold=threshold, tp=tp, fp=fp, fn=fn, tn=tn,
                precision=tp / (tp + fp) if tp + fp else np.nan,
                recall=tp / (tp + fn) if tp + fn else np.nan,
                false_decline_rate=fp / (fp + tn) if fp + tn else np.nan,
                declined=tp + fp, approved=fn + tn)


# --------------------------------------------------------------------------- #
# statistics
# --------------------------------------------------------------------------- #

def welch(a, b):
    d = a.mean() - b.mean()
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return d, se, d - 1.96 * se, d + 1.96 * se


def bca_ci(a, b, n_boot=2000, seed=SEED):
    """BCa interval for mean(a) - mean(b), resampling within each arm."""
    rng = np.random.default_rng(seed)
    obs = a.mean() - b.mean()
    reps = np.array([rng.choice(a, len(a), True).mean() - rng.choice(b, len(b), True).mean()
                     for _ in range(n_boot)])
    z0 = stats.norm.ppf(np.clip((reps < obs).mean(), 1e-6, 1 - 1e-6))
    jk = np.concatenate([(a.sum() - a) / (len(a) - 1) - b.mean(),
                         a.mean() - (b.sum() - b) / (len(b) - 1)])
    d = jk.mean() - jk
    acc = (d ** 3).sum() / (6 * ((d ** 2).sum() ** 1.5) + 1e-30)
    out = []
    for q in (0.025, 0.975):
        zq = stats.norm.ppf(q)
        adj = stats.norm.cdf(z0 + (z0 + zq) / (1 - acc * (z0 + zq)))
        out.append(np.quantile(reps, np.clip(adj, 0.001, 0.999)))
    return obs, out[0], out[1]


def boot_cost_diff(pa, ta, pb, tb, y, amt, n_boot=600, seed=7):
    """Paired bootstrap of cost(policy a) - cost(policy b) on the same rows."""
    rng = np.random.default_rng(seed)
    d = np.empty(n_boot)
    for k in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        d[k] = cost_at(ta, pa[i], y[i], amt[i]) - cost_at(tb, pb[i], y[i], amt[i])
    return d.mean(), np.percentile(d, 2.5), np.percentile(d, 97.5)


def ratio_se_clustered(num, den):
    """Delta-method SE for sum(num)/sum(den) with clusters as rows."""
    R, m = num.sum() / den.sum(), len(num)
    resid = num - R * den
    return R, np.sqrt((resid ** 2).sum() * m / (m - 1)) / den.sum()


def smd(x, t):
    a, b = x[t == 1], x[t == 0]
    sd = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return (a.mean() - b.mean()) / sd if sd > 0 else 0.0


# --------------------------------------------------------------------------- #
# features
# --------------------------------------------------------------------------- #

def encode(X):
    """Object columns to integer codes; the caller tells LightGBM which."""
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


SECONDS = {"1h": 3600, "24h": 86400, "7d": 604800}
_BIG = 1 << 25  # exceeds max transaction_dt, so account*_BIG + dt is unique

PROXY_COLS = ["account_id", "account_key_tier", "is_degraded_key",
              "prior_txn_count", "prior_amt_sum", "prior_amt_mean",
              "txn_count_1h", "txn_count_24h", "txn_count_7d",
              "prev_transaction_dt", "amt_to_prior_mean_ratio",
              "seconds_since_prev_txn"]


def window_features(account_id, transaction_dt, transaction_amt):
    """Strictly-prior aggregates per account.

    Mirrors the SQL `RANGE BETWEEN <bound> PRECEDING AND 1 PRECEDING`, so rows
    sharing a timestamp with the current row are excluded too. searchsorted
    rather than groupby-shift, because a shift would include those peers.
    """
    idx = account_id.index
    code = pd.factorize(account_id, sort=False)[0].astype(np.int64)
    dt = transaction_dt.to_numpy().astype(np.int64)
    amt = transaction_amt.to_numpy().astype(np.float64)

    key = code * _BIG + dt
    order = np.argsort(key, kind="stable")
    key_s, amt_s, dt_s = key[order], amt[order], dt[order]
    csum = np.concatenate([[0.0], np.cumsum(amt_s)])

    base = code * _BIG
    start = np.searchsorted(key_s, base, side="left")
    pos = np.searchsorted(key_s, key, side="left")

    count = (pos - start).astype(np.int64)
    total = csum[pos] - csum[start]
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(count > 0, total / count, np.nan)
    total = np.where(count > 0, total, np.nan)

    out = {"prior_txn_count": count, "prior_amt_sum": total,
           "prior_amt_mean": mean,
           "prev_transaction_dt": np.where(count > 0, dt_s[np.maximum(pos - 1, 0)], np.nan)}
    for label, w in SECONDS.items():
        lo = np.searchsorted(key_s, base + np.maximum(dt - w, 0), side="left")
        out[f"txn_count_{label}"] = (pos - lo).astype(np.int64)

    df = pd.DataFrame(out, index=idx)
    m = df.prior_amt_mean.to_numpy()
    df["amt_to_prior_mean_ratio"] = np.where((m != 0) & np.isfinite(m), amt / m, np.nan)
    df["seconds_since_prev_txn"] = dt - df.prev_transaction_dt
    return df


def _as_int(s):
    return pd.to_numeric(s, errors="coerce").astype("Int64")


def key_stage1(raw):
    """Shipped proxy: card1 + addr1 + (txn_day - D1), degrading by tier."""
    card1, addr1 = _as_int(raw["card1"]), _as_int(raw["addr1"])
    start = _as_int(raw["txn_day"]) - _as_int(raw["D1"])
    c, a, s = card1.notna(), addr1.notna(), start.notna()
    conds = [c & a & s, c & s, c & a, c]
    tier = np.select(conds, ["full", "no_addr", "no_card_start", "card_only"],
                     default="singleton")
    aid = np.select(conds, [
        "F|" + card1.astype(str) + "|" + addr1.astype(str) + "|" + start.astype(str),
        "N|" + card1.astype(str) + "|" + start.astype(str),
        "D|" + card1.astype(str) + "|" + addr1.astype(str),
        "C|" + card1.astype(str)],
        default="X|" + _as_int(raw["TransactionID"]).astype(str))
    return pd.Series(aid, index=raw.index), pd.Series(tier, index=raw.index)


def _tiered_key(raw, parts, prefix):
    """Longest available prefix of `parts` becomes the key; depth is the tier."""
    n = len(raw)
    aid = pd.Series("X|" + _as_int(raw["TransactionID"]).astype(str), index=raw.index)
    tier = pd.Series("singleton", index=raw.index)
    depth, ok = np.zeros(n, dtype=int), np.ones(n, dtype=bool)
    for i, (_, col, _r) in enumerate(parts):
        ok &= col.notna().to_numpy()
        depth = np.where(ok, i + 1, depth)
    for d in range(len(parts), 0, -1):
        m = depth == d
        if not m.any():
            continue
        s = pd.Series(f"{prefix}{d}", index=raw.index)
        for _, col, _r in parts[:d]:
            s = s + "|" + col.astype(str)
        aid, tier = aid.where(~m, s), tier.where(~m, f"depth_{d}")
    return aid, tier


def _start_day(raw):
    return _as_int(raw["txn_day"]) - _as_int(raw["D1"])


CANDIDATES = {
    "stage1 (card1+addr1+start)": key_stage1,
    "card1 only": lambda r: _tiered_key(r, [("card1", _as_int(r["card1"]), True)], "K"),
    "card1+start": lambda r: _tiered_key(r, [
        ("card1", _as_int(r["card1"]), True), ("start", _start_day(r), False)], "S"),
    "card1+addr1+start+card2": lambda r: _tiered_key(r, [
        ("card1", _as_int(r["card1"]), True), ("addr1", _as_int(r["addr1"]), False),
        ("start", _start_day(r), False), ("card2", _as_int(r["card2"]), False)], "W"),
    "card1+addr1+start+email": lambda r: _tiered_key(r, [
        ("card1", _as_int(r["card1"]), True), ("addr1", _as_int(r["addr1"]), False),
        ("start", _start_day(r), False),
        ("email", r["P_emaildomain"].where(r["P_emaildomain"].notna()), False)], "E"),
    "card1+card2+card3+card5+addr1+start": lambda r: _tiered_key(r, [
        ("card1", _as_int(r["card1"]), True), ("card2", _as_int(r["card2"]), False),
        ("card3", _as_int(r["card3"]), False), ("card5", _as_int(r["card5"]), False),
        ("addr1", _as_int(r["addr1"]), False), ("start", _start_day(r), False)], "M"),
}


def build_proxy_block(raw, key_fn):
    aid, tier = key_fn(raw)
    f = window_features(aid, raw["TransactionDT"], raw["TransactionAmt"])
    f.insert(0, "account_id", aid.to_numpy())
    f.insert(1, "account_key_tier", tier.to_numpy())
    f.insert(2, "is_degraded_key", (tier != "full").to_numpy())
    return f


# --------------------------------------------------------------------------- #
# walk-forward harness
# --------------------------------------------------------------------------- #

FOLD_STARTS = (118, 134, 150, 166)
TEST_LEN = ES_LEN = CAL_LEN = 16


def run_fold(X, y, day, amt, T, params=None, transform=None, test_len=None):
    """One fold. Threshold is picked out of sample on cross-fitted calibration."""
    tl = test_len or TEST_LEN
    tr = day < T - ES_LEN - CAL_LEN
    es = (day >= T - ES_LEN - CAL_LEN) & (day < T - CAL_LEN)
    ca = (day >= T - CAL_LEN) & (day < T)
    te = (day >= T) & (day < T + tl)

    if transform is not None:
        X = transform(X, tr | es | ca)
    Xc, cats = encode(X)

    clf = lgb.LGBMClassifier(**{**LGB_PARAMS, **(params or {})})
    t0 = time.perf_counter()
    clf.fit(Xc[tr], y[tr], eval_set=[(Xc[es], y[es])], eval_metric="average_precision",
            categorical_feature=cats,
            callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
    fit_secs = time.perf_counter() - t0

    ca_idx = np.flatnonzero(ca)
    half = np.random.default_rng(SEED).permutation(len(ca_idx)) < len(ca_idx) // 2
    p_oof = np.empty(len(ca_idx))
    for m in (half, ~half):
        c = CalibratedClassifierCV(FrozenEstimator(clf), method="isotonic")
        c.fit(Xc.iloc[ca_idx[m]], y[ca_idx[m]])
        p_oof[~m] = c.predict_proba(Xc.iloc[ca_idx[~m]])[:, 1]
    thr, _, _ = sweep(p_oof, y[ca_idx], amt[ca_idx])

    cal = CalibratedClassifierCV(FrozenEstimator(clf), method="isotonic")
    cal.fit(Xc[ca], y[ca])
    p_te, p_raw = cal.predict_proba(Xc[te])[:, 1], clf.predict_proba(Xc[te])[:, 1]
    yte, ate, dec = y[te], amt[te], None
    dec = p_te >= thr
    thr_o, cost_o, _ = sweep(p_te, yte, ate)

    return {
        "fold_test_start": T, "n_train": int(tr.sum()), "n_test": int(te.sum()),
        "n_fraud_test": int(yte.sum()), "fit_secs": round(fit_secs, 1),
        "best_iter": int(clf.best_iteration_),
        "test_pr_auc": average_precision_score(yte, p_raw),
        "test_pr_auc_cal": average_precision_score(yte, p_te),
        "test_roc_auc": roc_auc_score(yte, p_raw),
        "test_brier_cal": brier_score_loss(yte, p_te),
        "train_pr_auc": average_precision_score(y[tr], clf.predict_proba(Xc[tr])[:, 1]),
        "threshold": thr, "cost": total_cost(dec, yte, ate),
        "cost_at_half": cost_at(0.5, p_te, yte, ate),
        "cost_approve_all": total_cost(np.zeros(len(yte), bool), yte, ate),
        "cost_oracle": cost_o, "threshold_oracle": thr_o,
        "fraud_dollars_caught": float(ate[dec & (yte == 1)].sum()),
        "fraud_dollars_total": float(ate[yte == 1].sum()),
        "false_decline_rate": float(dec[yte == 0].mean()),
        "recall": float(dec[yte == 1].mean()), "_clf": clf,
    }


def run_config(name, X, y, day, amt, params=None, folds=FOLD_STARTS, transform=None):
    rows = []
    for T in folds:
        r = run_fold(X, y, day, amt, T, params, transform)
        rows.append({k: v for k, v in r.items() if not k.startswith("_")})
        print(f"  {name} T={T}: PR-AUC {r['test_pr_auc']:.4f} "
              f"cost ${r['cost']:,.0f} thr {r['threshold']:.4f} ({r['fit_secs']}s)")
    df = pd.DataFrame(rows)
    df.insert(0, "config", name)
    return df


def paired_fold_test(a, b, col="cost"):
    d = a[col].to_numpy() - b[col].to_numpy()
    n = len(d)
    se = d.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
    return {"mean_diff": d.mean(), "sd_diff": d.std(ddof=1) if n > 1 else np.nan,
            "se": se, "t": d.mean() / se if se else np.nan,
            "n_folds": n, "per_fold": d}


# --------------------------------------------------------------------------- #
# snowflake
# --------------------------------------------------------------------------- #

def connect(schema="MARTS", warehouse=True):
    import snowflake.connector
    from dotenv import load_dotenv
    load_dotenv(".env")
    kw = dict(account=os.environ["SNOWFLAKE_ACCOUNT"], user=os.environ["SNOWFLAKE_USER"],
              password=os.environ["SNOWFLAKE_PASSWORD"], role=os.environ["SNOWFLAKE_ROLE"],
              database=os.environ.get("SNOWFLAKE_DATABASE", "FRAUD"), schema=schema)
    if warehouse:
        kw["warehouse"] = os.environ["SNOWFLAKE_WAREHOUSE"]
    conn = snowflake.connector.connect(**kw)
    if warehouse:
        conn.cursor().execute("ALTER WAREHOUSE FRAUD_WH RESUME IF SUSPENDED")
    return conn


def suspend_warehouse():
    conn = connect(warehouse=False)
    cur = conn.cursor()
    try:
        cur.execute("ALTER WAREHOUSE FRAUD_WH SUSPEND")
        print("warehouse suspended")
    except Exception:
        print("warehouse already suspended")
    cur.execute("SHOW WAREHOUSES LIKE 'FRAUD_WH'")
    r = dict(zip([d[0] for d in cur.description], cur.fetchone()))
    print(f"state={r['state']} size={r['size']} auto_suspend={r['auto_suspend']}s")
    conn.close()


def query(cur, sql):
    cur.execute(sql)
    return pd.DataFrame(cur.fetchall(), columns=[d[0] for d in cur.description])


def account_hash(account_id, salt, tag=""):
    key = f"{salt}:{tag}:{account_id}" if tag else f"{salt}:{account_id}"
    return int(hashlib.md5(key.encode()).hexdigest(), 16) / 2 ** 128
