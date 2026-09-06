{{ config(
    materialized='incremental',
    unique_key='transaction_id',
    incremental_strategy='merge',
    on_schema_change='sync_all_columns'
) }}

-- One row per transaction, with backward-looking behavioural features per
-- constructed account, plus full passthrough of the V block (V1-V339) and
-- the identity block (id_01-id_38, DeviceType, DeviceInfo).
--
-- LEAKAGE CONTROL. Every window below uses:
--     RANGE BETWEEN <bound> PRECEDING AND 1 PRECEDING
-- The "AND 1 PRECEDING" end bound is what makes these strictly backward
-- looking. RANGE is used rather than ROWS deliberately: RANGE compares
-- transaction_dt values, so transactions sharing a timestamp with the current
-- row are excluded too, not merely the current row itself. A frame ending at
-- CURRENT ROW would fold the present into its own feature and inflate every
-- downstream metric.
--
-- Consequence to expect: the first transaction of every account has NULL for
-- all prior-window features, because it genuinely has no history. That is
-- correct, not a defect.
--
-- INCREMENTAL. Windows are computed over full history first, then the result
-- is filtered to new txn_day values. Restricting the input to new days would
-- truncate each account's history and silently corrupt the features, so the
-- model reads everything on every run and only the write is incremental.

WITH base AS (

    SELECT
        t.*,
        k.account_id,
        k.account_key_tier,
        k.is_degraded_key,
        i.transaction_id IS NOT NULL                      AS has_identity,

        -- Identity passthrough. Explicit rather than i.* so transaction_id
        -- does not collide with the transactions side.
        {% for n in range(1, 39) %}i.id_{{ "%02d"|format(n) }},
        {% endfor %}i.device_type,
        i.device_info
    FROM {{ ref('stg_transactions') }} t
    INNER JOIN {{ ref('int_account_keys') }} k
        ON k.transaction_id = t.transaction_id
    LEFT JOIN {{ ref('stg_identity') }} i
        ON i.transaction_id = t.transaction_id

),

featured AS (

    SELECT
        transaction_id,
        account_id,
        account_key_tier,
        is_degraded_key,
        txn_day,
        transaction_dt,
        is_fraud,

        transaction_amt,
        LN(transaction_amt + 1)                           AS log_amount,

        -- Strictly prior aggregates over the whole account history.
        COUNT(*) OVER (
            PARTITION BY account_id ORDER BY transaction_dt
            RANGE BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        )                                                 AS prior_txn_count,

        SUM(transaction_amt) OVER (
            PARTITION BY account_id ORDER BY transaction_dt
            RANGE BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        )                                                 AS prior_amt_sum,

        AVG(transaction_amt) OVER (
            PARTITION BY account_id ORDER BY transaction_dt
            RANGE BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        )                                                 AS prior_amt_mean,

        -- Trailing velocity counts. 3600s, 86400s, 604800s.
        COUNT(*) OVER (
            PARTITION BY account_id ORDER BY transaction_dt
            RANGE BETWEEN 3600 PRECEDING AND 1 PRECEDING
        )                                                 AS txn_count_1h,

        COUNT(*) OVER (
            PARTITION BY account_id ORDER BY transaction_dt
            RANGE BETWEEN 86400 PRECEDING AND 1 PRECEDING
        )                                                 AS txn_count_24h,

        COUNT(*) OVER (
            PARTITION BY account_id ORDER BY transaction_dt
            RANGE BETWEEN 604800 PRECEDING AND 1 PRECEDING
        )                                                 AS txn_count_7d,

        -- Timestamp of the most recent strictly earlier transaction.
        MAX(transaction_dt) OVER (
            PARTITION BY account_id ORDER BY transaction_dt
            RANGE BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        )                                                 AS prev_transaction_dt,

        txn_hour,
        txn_dow,
        has_identity,

        product_cd,
        card4,
        card6,
        p_emaildomain,
        r_emaildomain,

        {% for i in range(1, 15) %}c{{ i }},
        {% endfor %}
        {% for i in range(1, 16) %}d{{ i }},
        {% endfor %}

        -- Time-normalised D columns.
        --
        -- D1-D15 are "days since some prior event", so as raw counters they
        -- drift with calendar time: a split learned at day 80 does not mean
        -- the same thing at day 170. Because evaluation is a forward time
        -- split, that drift costs real accuracy. Subtracting from txn_day
        -- converts each into the fixed day the event happened, which is
        -- stationary, and incidentally lets the model form its own grouping
        -- of transactions sharing a card history.
        --
        -- Measured on four walk-forward folds: +0.0228 test PR-AUC, better in
        -- 4 folds out of 4, and $38,140 lower cost. See
        -- reports/model_improvements.md.
        --
        -- D9 is deliberately excluded. It ranges 0 to 0.958 and is a
        -- time-of-day fraction, not a day offset, so normalising it would be
        -- meaningless.
        {% for i in [1,2,3,4,5,6,7,8,10,11,12,13,14,15] %}txn_day - d{{ i }} AS d{{ i }}_norm,
        {% endfor %}
        {% for i in range(1, 10) %}m{{ i }},
        {% endfor %}

        -- Vesta's engineered block. 339 columns, passed through untouched.
        {% for i in range(1, 340) %}v{{ i }},
        {% endfor %}

        -- Identity block. Null for the 75.6% of transactions with no
        -- matching identity row, which is why has_identity is a feature.
        {% for n in range(1, 39) %}id_{{ "%02d"|format(n) }},
        {% endfor %}device_type,
        device_info

    FROM base

)

SELECT
    *,
    transaction_amt / NULLIF(prior_amt_mean, 0)           AS amt_to_prior_mean_ratio,
    transaction_dt - prev_transaction_dt                  AS seconds_since_prev_txn
FROM featured

{% if is_incremental() %}
WHERE txn_day > (SELECT COALESCE(MAX(txn_day), -1) FROM {{ this }})
{% endif %}
