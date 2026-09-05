-- Direct assertion that the trailing-window features are backward looking.
--
-- For the first transaction of every account there is by definition no prior
-- history, so prior_txn_count must be 0 and all trailing counts must be 0.
-- If any window frame ended at CURRENT ROW instead of 1 PRECEDING, the
-- current row would count itself and these would be 1. The test passes when
-- it returns no rows.

WITH first_txn AS (
    SELECT
        transaction_id,
        prior_txn_count,
        txn_count_1h,
        txn_count_24h,
        txn_count_7d,
        prior_amt_sum,
        ROW_NUMBER() OVER (PARTITION BY account_id ORDER BY transaction_dt) AS rn
    FROM {{ ref('fct_transactions') }}
)
SELECT *
FROM first_txn
WHERE rn = 1
  AND (
        prior_txn_count != 0
     OR txn_count_1h    != 0
     OR txn_count_24h   != 0
     OR txn_count_7d    != 0
     OR prior_amt_sum IS NOT NULL
  )
