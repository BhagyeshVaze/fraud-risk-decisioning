{{ config(materialized='table') }}

-- One row per constructed account. Aggregates are over the account's whole
-- history, so unlike fct_transactions these intentionally include every
-- transaction. This is a descriptive dimension, not a model feature source.

SELECT
    account_id,
    ANY_VALUE(account_key_tier)                           AS account_key_tier,
    BOOLOR_AGG(is_degraded_key)                           AS is_degraded_key,

    MIN(txn_day)                                          AS first_seen_day,
    MAX(txn_day)                                          AS last_seen_day,
    MAX(txn_day) - MIN(txn_day)                           AS active_day_span,

    COUNT(*)                                              AS txn_count,
    SUM(transaction_amt)                                  AS total_amount,
    AVG(transaction_amt)                                  AS mean_amount,

    SUM(is_fraud)                                         AS fraud_count,
    AVG(is_fraud)                                         AS fraud_rate

FROM {{ ref('fct_transactions') }}
GROUP BY account_id
