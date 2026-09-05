# Fraud Risk Decisioning System

**At the cost-optimal decline threshold this system prevents $319,337 of the
$495,244 in fraud on a held-out 30-day test period, catching 64.5% of fraud
dollars while declining 6.7% of legitimate transactions. Total modelled cost
is $309,028, which is $125,098 cheaper than a naive 0.5 cutoff and $268,266
cheaper than approving everything.**

Those figures rest on stated cost assumptions, not measured ones. See
[Cost assumptions](#cost-assumptions).

## What it does and what it found

The system scores card-not-present transactions for fraud, converts the score
into a calibrated probability, and converts that probability into an
approve-or-decline decision by pricing every possible threshold in dollars.

The interesting part is the third step. A fraud model is usually judged on
PR-AUC, but nobody is paid in PR-AUC. Once each outcome has a price, the
question becomes where to set the threshold, and the answer is not 0.5. It is
0.0710. The naive cutoff looks respectable on precision (0.80 against 0.26)
and is 40% more expensive, because it optimizes the wrong thing: it avoids
false declines that cost about $12 each while waving through frauds that cost
about $175 each.

Three findings worth stating plainly:

1. **The optimum is far from 0.5 and cost is asymmetric around it.** Being too
   strict is much more expensive than being slightly too loose, because the
   decline side of the curve carries churn and lost margin on a very large
   population of legitimate traffic.
2. **Calibration did not improve the Brier score on test** (0.022893 to
   0.023070). It improves the middle deciles, where the threshold sits, and
   overshoots the top decile. The cause is that the isotonic map is fitted on
   days 120 to 149 and applied to days 150 and later.
3. **No single feature dominates.** The top feature carries 11.1% of total
   gain, well under the 30% level that usually indicates leakage.

## Results

Evaluated on `txn_day >= 150`: 94,636 transactions, 3,282 fraudulent.

| Policy | Total cost | Fraud dollars caught | Legit declined | Saving vs optimum |
| --- | --- | --- | --- | --- |
| **Chosen threshold 0.0710** | **$309,028** | $319,337 (64.5%) | 6,099 (6.68%) | - |
| Naive 0.5 cutoff | $434,127 | $123,196 (24.9%) | 266 (0.29%) | $125,098 |
| Approve everything | $577,294 | $0 (0%) | 0 (0%) | $268,266 |
| Decline everything | $1,414,895 | $495,244 (100%) | 91,354 (100%) | $1,105,866 |

| Decision quality | At 0.0710 | At 0.50 |
| --- | --- | --- |
| Fraud caught (TP) | 2,146 | 1,056 |
| Legit declined (FP) | 6,099 | 266 |
| Fraud missed (FN) | 1,136 | 2,226 |
| Precision | 0.2603 | 0.7988 |
| Recall | 0.6539 | 0.3218 |
| False decline rate | 6.676% | 0.291% |

| Model quality | Value |
| --- | --- |
| PR-AUC, test, raw | 0.5007 |
| PR-AUC, test, calibrated | 0.4865 |
| ROC-AUC, test | 0.8932 |
| Brier, test, raw | 0.022893 |
| Brier, test, calibrated | 0.023070 |
| PR-AUC, train | 0.8243 |

The train-to-test PR-AUC gap of 0.32 is flagged as overfitting in
[verification.md](reports/verification.md). It is expected here: the trailing
window features are far richer for accounts with long histories, which are
overrepresented in the earlier training period.

Sensitivity: moving the threshold 20% in either direction changes total cost by
at most 3.45%, so the answer is not knife-edge, but it is not flat either.

## Architecture

```
S3 (ca-central-1)  ->  Snowflake external stage  ->  FRAUD.RAW
                                                        |
                                             dbt: STAGING (views)
                                                        |
                                             dbt: MARTS (tables)
                                                        |
                                    LightGBM -> isotonic -> threshold sweep
```

![lineage](reports/dag.png)

| Layer | Object | Rows |
| --- | --- | --- |
| RAW | `RAW_TRANSACTIONS` | 590,540 |
| RAW | `RAW_IDENTITY` | 144,233 |
| STAGING | `stg_transactions`, `stg_identity`, `int_account_keys` | views |
| MARTS | `fct_transactions` | 590,540 |
| MARTS | `dim_accounts` | 217,850 |

Interactive lineage is in [reports/dbt_docs/index.html](reports/dbt_docs/index.html),
exported so it outlives the Snowflake trial. 17 dbt tests pass, including an
exact row-count assertion and a direct anti-leakage assertion.

**Leakage control.** Every window feature uses
`RANGE BETWEEN <bound> PRECEDING AND 1 PRECEDING`, partitioned by account and
ordered by `transaction_dt`. `RANGE` rather than `ROWS` is deliberate: it
excludes transactions sharing a timestamp with the current row, not just the
current row. This is verified three ways, in
[verification.md](reports/verification.md): a dbt test asserting that every
account's first transaction has zero prior counts, an independent pandas
recomputation on sampled accounts, and printed transaction sequences.

## Data and its limitations

IEEE-CIS Fraud Detection, 590,540 transactions over 182 days. Four limitations
shape everything above.

**The account identifier is constructed, not given.** IEEE-CIS has no account
or customer id. The proxy concatenates `card1`, `addr1`, and
`txn_day - D1`, where D1 is days since the card first appeared, so the third
component is a stable card start date. This yields 217,850 accounts. 88.7% of
rows get the full key; 11.3% fall back to a degraded tier, tracked explicitly
in `account_key_tier`. **If the proxy is wrong, every behavioural feature is
wrong**, and there is no way to validate it against ground truth.

**Account sizes are heavily skewed.** The median account has 1 transaction,
because 57.5% of cards appear exactly once. Weighted by transaction, which is
what the features actually see, the median account has 5. 21.2% of
transactions will never have prior-window history.

**The competition test set is unlabeled**, so the split is a time split within
the labeled training data only: train below day 120, calibration 120 to 149,
test 150 and later. There is no evaluation against the competition's own
holdout, and the numbers here are not comparable to leaderboard scores.

**Identity data covers only 24.4% of transactions.** Every identity column is
null for the other 75.6%. `has_identity` is a feature in its own right, since
its absence is informative.

Two smaller notes: `TransactionDT` is seconds from an unstated origin, so
`txn_hour` and `txn_dow` are relative and no real calendar date is recoverable.
And `card1` is an anonymized card identifier, not a card number.

## Cost assumptions

**These are assumptions, not facts.** IEEE-CIS contains no chargeback, margin,
or churn data. They are plausible mid-market card-not-present values chosen to
make the tradeoff explicit. Every dollar figure in this README inherits their
uncertainty.

| Assumption | Value |
| --- | --- |
| Chargeback fee | $25.00 |
| Margin rate | 2.5% |
| Churn probability after a false decline | 5% |
| Account lifetime value | $200.00 |
| Manual review cost | $2.00 |

| Outcome | Modelled cost |
| --- | --- |
| Fraud approved | transaction amount + $25 |
| Legitimate declined | 2.5% margin + 5% x $200 churn + $2 review |
| Fraud declined | $2 review |
| Legitimate approved | $0 |

![cost curve](reports/cost_vs_threshold.png)

## How to reproduce

Requires Python 3.10 or later. Credentials go in `.env` at the repo root, which
is gitignored; see the variable names in `src/load_to_snowflake.py`.

```bash
python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
```

```bash
python src/prepare_data.py && python src/load_to_snowflake.py
```

```bash
cd fraud_dbt && set -a && source ../.env && set +a && dbt build && dbt docs generate
```

```bash
python src/verify_part1.py && python src/train_model.py && python src/decision_layer.py
```

The model stage reads `data/parquet/fct_transactions.parquet` if it exists and
only queries Snowflake otherwise, so **steps 2 and 3 can be skipped entirely
once that file exists**. This is deliberate: the Snowflake account is a 30-day
trial and the modelling work has to outlive it.

## What is not done yet

- **No unplanned-versus-planned decline logic.** Every decline is treated the
  same. A real system routes some to step-up authentication rather than a hard
  decline, which changes the cost model substantially.
- **The account proxy is unvalidated.** It should be sanity-checked against
  `card2`, `card5`, and device fingerprints before anyone trusts it.
- **No drift monitoring.** The calibration degradation between days 120 to 149
  and 150 and later is exactly the signal a monitor should watch.
- **Calibration reuses the early-stopping split.** The calibrator is fitted on
  the same days used to early-stop the model, which biases it optimistically.
  A fourth disjoint split would fix this.
- **No feature selection.** All 58 features go in. The V-columns are untouched.
- **No CI, no scheduling, no serving.** There is no orchestration, no API, and
  no monitoring of live scores.
- **Cost assumptions are unvalidated.** They should come from finance, not from
  a README.
