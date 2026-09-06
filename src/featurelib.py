"""Local reimplementation of the account key and backward-looking windows.

The dbt model is the source of truth. This exists so candidate account proxies
can be iterated in seconds against the cached parquet instead of a warehouse
round trip per candidate. `verify_against_dbt()` asserts the two agree, and no
result from this module should be trusted until that passes.

Window semantics mirror the SQL exactly:

    RANGE BETWEEN <bound> PRECEDING AND 1 PRECEDING

which means strictly earlier transaction_dt, and therefore also excludes rows
sharing a timestamp with the current row. That is implemented here with
searchsorted over a combined (account, transaction_dt) key rather than a
groupby-shift, because a shift would include same-timestamp peers.
"""

import hashlib

import numpy as np
import pandas as pd

SECONDS = {"1h": 3600, "24h": 86400, "7d": 604800}
BIG = 1 << 25  # exceeds max transaction_dt (~15.8M), so account*BIG+dt is unique

PROXY_COLS = [
    "account_id", "account_key_tier", "is_degraded_key",
    "prior_txn_count", "prior_amt_sum", "prior_amt_mean",
    "txn_count_1h", "txn_count_24h", "txn_count_7d",
    "prev_transaction_dt", "amt_to_prior_mean_ratio", "seconds_since_prev_txn",
]


# --------------------------------------------------------------------------
# account key candidates
# --------------------------------------------------------------------------

def _as_int(s):
    return pd.to_numeric(s, errors="coerce").astype("Int64")


def key_stage1(raw):
    """The shipped proxy: card1 + addr1 + (txn_day - D1), with degradation tiers."""
    card1 = _as_int(raw["card1"])
    addr1 = _as_int(raw["addr1"])
    start = _as_int(raw["txn_day"]) - _as_int(raw["D1"])

    have_c, have_a, have_s = card1.notna(), addr1.notna(), start.notna()
    tier = np.select(
        [have_c & have_a & have_s, have_c & have_s, have_c & have_a, have_c],
        ["full", "no_addr", "no_card_start", "card_only"], default="singleton")
    aid = np.select(
        [have_c & have_a & have_s, have_c & have_s, have_c & have_a, have_c],
        ["F|" + card1.astype(str) + "|" + addr1.astype(str) + "|" + start.astype(str),
         "N|" + card1.astype(str) + "|" + start.astype(str),
         "D|" + card1.astype(str) + "|" + addr1.astype(str),
         "C|" + card1.astype(str)],
        default="X|" + _as_int(raw["TransactionID"]).astype(str))
    return pd.Series(aid, index=raw.index), pd.Series(tier, index=raw.index)


def _tiered_key(raw, parts, prefix):
    """Build a key from `parts`, degrading by dropping trailing optional parts.

    parts is a list of (label, Series, required) in priority order. A row uses
    the longest prefix of available parts; tier records how far it got.
    """
    n = len(raw)
    aid = pd.Series("X|" + _as_int(raw["TransactionID"]).astype(str), index=raw.index)
    tier = pd.Series("singleton", index=raw.index)

    avail = [p[1].notna() for p in parts]
    # longest available prefix per row
    depth = np.zeros(n, dtype=int)
    ok = np.ones(n, dtype=bool)
    for i, a in enumerate(avail):
        ok = ok & a.to_numpy()
        depth = np.where(ok, i + 1, depth)

    for d in range(len(parts), 0, -1):
        m = depth == d
        if not m.any():
            continue
        s = pd.Series(f"{prefix}{d}", index=raw.index)
        for _, col, _req in parts[:d]:
            s = s + "|" + col.astype(str)
        aid = aid.where(~m, s)
        tier = tier.where(~m, f"depth_{d}")
    return aid, tier


def key_card_only(raw):
    return _tiered_key(raw, [("card1", _as_int(raw["card1"]), True)], "K")


def key_wide(raw):
    """card1 + card2 + addr1 + card start day. More components, finer grouping."""
    start = _as_int(raw["txn_day"]) - _as_int(raw["D1"])
    return _tiered_key(raw, [
        ("card1", _as_int(raw["card1"]), True),
        ("addr1", _as_int(raw["addr1"]), False),
        ("start", start, False),
        ("card2", _as_int(raw["card2"]), False),
    ], "W")


def key_card_start(raw):
    """card1 + card start day only. Drops addr1, which is null 11% of the time."""
    start = _as_int(raw["txn_day"]) - _as_int(raw["D1"])
    return _tiered_key(raw, [
        ("card1", _as_int(raw["card1"]), True),
        ("start", start, False),
    ], "S")


def key_card_email(raw):
    """card1 + addr1 + card start + purchaser email domain."""
    start = _as_int(raw["txn_day"]) - _as_int(raw["D1"])
    email = raw["P_emaildomain"].where(raw["P_emaildomain"].notna())
    return _tiered_key(raw, [
        ("card1", _as_int(raw["card1"]), True),
        ("addr1", _as_int(raw["addr1"]), False),
        ("start", start, False),
        ("email", email, False),
    ], "E")


def key_full_card(raw):
    """card1 + card2 + card3 + card5 + addr1 + card start. Maximum specificity."""
    start = _as_int(raw["txn_day"]) - _as_int(raw["D1"])
    return _tiered_key(raw, [
        ("card1", _as_int(raw["card1"]), True),
        ("card2", _as_int(raw["card2"]), False),
        ("card3", _as_int(raw["card3"]), False),
        ("card5", _as_int(raw["card5"]), False),
        ("addr1", _as_int(raw["addr1"]), False),
        ("start", start, False),
    ], "M")


CANDIDATES = {
    "stage1 (card1+addr1+start)": key_stage1,
    "card1 only": key_card_only,
    "card1+start": key_card_start,
    "card1+addr1+start+card2": key_wide,
    "card1+addr1+start+email": key_card_email,
    "card1+card2+card3+card5+addr1+start": key_full_card,
}


# --------------------------------------------------------------------------
# backward-looking windows
# --------------------------------------------------------------------------

def window_features(account_id, transaction_dt, transaction_amt):
    """Strictly-prior aggregates per account, matching the SQL RANGE frames.

    Returns a DataFrame indexed like the inputs.
    """
    idx = account_id.index
    acct_code = pd.factorize(account_id, sort=False)[0].astype(np.int64)
    dt = transaction_dt.to_numpy().astype(np.int64)
    amt = transaction_amt.to_numpy().astype(np.float64)

    key = acct_code * BIG + dt
    order = np.argsort(key, kind="stable")
    key_s, amt_s, dt_s = key[order], amt[order], dt[order]

    # exclusive prefix sum over the sorted rows
    csum = np.concatenate([[0.0], np.cumsum(amt_s)])

    acct_base = acct_code * BIG
    start = np.searchsorted(key_s, acct_base, side="left")          # first row of account
    pos = np.searchsorted(key_s, key, side="left")                  # first row with dt == cur

    prior_count = (pos - start).astype(np.int64)
    prior_sum = csum[pos] - csum[start]
    with np.errstate(invalid="ignore", divide="ignore"):
        prior_mean = np.where(prior_count > 0, prior_sum / prior_count, np.nan)
    prior_sum = np.where(prior_count > 0, prior_sum, np.nan)

    prev_dt = np.where(prior_count > 0, dt_s[np.maximum(pos - 1, 0)], np.nan)

    out = {
        "prior_txn_count": prior_count,
        "prior_amt_sum": prior_sum,
        "prior_amt_mean": prior_mean,
        "prev_transaction_dt": prev_dt,
    }
    for label, w in SECONDS.items():
        lo = np.maximum(dt - w, 0)
        pos_lo = np.searchsorted(key_s, acct_base + lo, side="left")
        out[f"txn_count_{label}"] = (pos - pos_lo).astype(np.int64)

    df = pd.DataFrame(out, index=idx)
    df["amt_to_prior_mean_ratio"] = np.where(
        (df.prior_amt_mean.to_numpy() != 0) & np.isfinite(df.prior_amt_mean.to_numpy()),
        amt / df.prior_amt_mean.to_numpy(), np.nan)
    df["seconds_since_prev_txn"] = dt - df.prev_transaction_dt
    return df


def build_proxy_block(raw, key_fn):
    """account_id, tier, degraded flag and all window features for one candidate."""
    aid, tier = key_fn(raw)
    feats = window_features(aid, raw["TransactionDT"], raw["TransactionAmt"])
    feats.insert(0, "account_id", aid.to_numpy())
    feats.insert(1, "account_key_tier", tier.to_numpy())
    feats.insert(2, "is_degraded_key", (tier != "full").to_numpy()
                 if "full" in set(tier.unique())
                 else (tier != f"depth_{tier.str.extract(r'depth_(\d+)')[0].astype(float).max():.0f}").to_numpy())
    return feats


def verify_against_dbt(raw, fct):
    """Assert the local engine reproduces the shipped dbt output exactly."""
    blk = build_proxy_block(raw, key_stage1)
    blk.index = raw["TransactionID"].to_numpy()
    ref = fct.set_index("transaction_id")
    blk = blk.loc[ref.index]

    report = []
    for c in ["account_id", "account_key_tier", "prior_txn_count", "txn_count_1h",
              "txn_count_24h", "txn_count_7d", "prior_amt_sum", "prior_amt_mean",
              "prev_transaction_dt", "seconds_since_prev_txn",
              "amt_to_prior_mean_ratio"]:
        a, b = blk[c], ref[c]
        if a.dtype == object:
            match = (a.astype(str) == b.astype(str)).mean()
            worst = np.nan
        else:
            av, bv = a.to_numpy(dtype="float64"), b.to_numpy(dtype="float64")
            both_nan = np.isnan(av) & np.isnan(bv)
            close = np.isclose(av, bv, rtol=1e-6, atol=1e-6, equal_nan=True)
            match = (close | both_nan).mean()
            d = np.abs(av - bv)
            worst = np.nanmax(np.where(both_nan, 0, d)) if len(d) else 0.0
        report.append({"column": c, "match_rate": match, "max_abs_diff": worst})
    return pd.DataFrame(report)
