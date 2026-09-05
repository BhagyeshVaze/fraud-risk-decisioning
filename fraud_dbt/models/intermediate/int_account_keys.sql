{{ config(materialized='view') }}

-- IEEE-CIS has no account identifier, so this constructs a proxy.
--
-- D1 is documented as days since the card first appeared, so txn_day - d1 is
-- the day the card started, which is stable across that card's transactions.
-- Combining it with card1 and addr1 gives a key that groups transactions
-- plausibly belonging to one cardholder.
--
-- This is a proxy, not ground truth. Any component can be null, so the key
-- degrades through named tiers rather than silently producing nulls or
-- collapsing unrelated rows together. Tier is carried downstream so the
-- degraded share stays measurable.

WITH base AS (

    SELECT
        transaction_id,
        transaction_dt,
        txn_day,
        card1,
        CAST(addr1 AS NUMBER(38,0))                       AS addr1_int,
        CASE
            WHEN d1 IS NOT NULL
            THEN txn_day - CAST(d1 AS NUMBER(38,0))
        END                                               AS card_start_day
    FROM {{ ref('stg_transactions') }}

)

SELECT
    transaction_id,
    transaction_dt,
    txn_day,
    card1,
    addr1_int                                             AS addr1,
    card_start_day,

    CASE
        WHEN card1 IS NOT NULL AND addr1_int IS NOT NULL AND card_start_day IS NOT NULL
            THEN 'full'
        WHEN card1 IS NOT NULL AND card_start_day IS NOT NULL
            THEN 'no_addr'
        WHEN card1 IS NOT NULL AND addr1_int IS NOT NULL
            THEN 'no_card_start'
        WHEN card1 IS NOT NULL
            THEN 'card_only'
        ELSE 'singleton'
    END                                                   AS account_key_tier,

    CASE
        WHEN card1 IS NOT NULL AND addr1_int IS NOT NULL AND card_start_day IS NOT NULL
            THEN 'F|' || card1 || '|' || addr1_int || '|' || card_start_day
        WHEN card1 IS NOT NULL AND card_start_day IS NOT NULL
            THEN 'N|' || card1 || '|' || card_start_day
        WHEN card1 IS NOT NULL AND addr1_int IS NOT NULL
            THEN 'D|' || card1 || '|' || addr1_int
        WHEN card1 IS NOT NULL
            THEN 'C|' || card1
        ELSE 'X|' || transaction_id
    END                                                   AS account_id,

    CASE
        WHEN card1 IS NOT NULL AND addr1_int IS NOT NULL AND card_start_day IS NOT NULL
            THEN FALSE
        ELSE TRUE
    END                                                   AS is_degraded_key

FROM base
