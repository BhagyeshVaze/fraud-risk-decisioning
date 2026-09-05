-- fct_transactions must contain exactly the 590,540 source transactions.
-- A join or incremental bug that dropped or duplicated rows would show here.
-- The test passes when it returns no rows.

SELECT
    {{ 590540 }}   AS expected_rows,
    COUNT(*)       AS actual_rows
FROM {{ ref('fct_transactions') }}
HAVING COUNT(*) != 590540
