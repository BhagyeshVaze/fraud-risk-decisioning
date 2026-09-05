# Fraud Risk Decisioning System

**At the cost-optimal decline threshold this system prevents $339,844 of the
$495,244 in fraud on a held-out 30-day test period, catching 68.6% of fraud
dollars while declining 7.3% of legitimate transactions. Total modelled cost
is $298,574, which is $147,082 cheaper than a naive 0.5 cutoff and $278,721
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
0.0793. The naive cutoff looks far better on precision (0.74 against 0.25) and
is 49% more expensive, because it optimizes the wrong thing: it avoids false
declines that cost about $12 each while waving through frauds that cost about
$176 each.

Four findings worth stating plainly:

1. **The optimum is far from 0.5 and cost is asymmetric around it.** Being too
   strict is roughly nine times more costly than being equally too loose
   (+6.2% against +0.7% for a 20% threshold move). The decline side carries
   churn and lost margin across a very large legitimate population.
2. **The V block and identity columns helped, but only slightly.** Adding 379
   columns moved test PR-AUC from 0.5007 to 0.5086, **+0.0079, about +1.6%**.
   They are described as carrying most of Vesta's engineered signal; on this
   split they are worth about one and a half percent. They did translate into
   real money, $10,455 lower cost, because the threshold sweep converts small
   ranking gains into dollars efficiently.
3. **Calibration still does not improve the Brier score**, even after early
   stopping and calibration were moved to disjoint windows. The degradation
   shrank from -0.77% to -0.11%, so window reuse was most of the problem, but
   not all of it. What remains is drift between the calibration period and the
   test period.
4. **No single feature dominates.** The top feature carries 6.05% of total
   gain, well under the 30% level that usually indicates leakage.

## Results

Evaluated on `txn_day >= 150`: 94,636 transactions, 3,282 fraudulent.

| Policy | Total cost | Fraud dollars caught | Legit declined | Saving vs optimum |
| --- | --- | --- | --- | --- |
| **Chosen threshold 0.0793** | **$298,574** | $339,844 (68.6%) | 6,652 (7.28%) | - |
| Naive 0.5 cutoff | $445,656 | $111,687 (22.6%) | 388 (0.42%) | $147,082 |
| Approve everything | $577,294 | $0 (0%) | 0 (0%) | $278,721 |
| Decline everything | $1,414,895 | $495,244 (100%) | 91,354 (100%) | $1,116,321 |

| Decision quality | At 0.0793 | At 0.50 |
| --- | --- | --- |
| Fraud caught (TP) | 2,213 | 1,127 |
| Legit declined (FP) | 6,652 | 388 |
| Fraud missed (FN) | 1,069 | 2,155 |
| True negatives | 84,702 | 90,966 |
| Precision | 0.2496 | 0.7439 |
| Recall | 0.6743 | 0.3434 |
| False decline rate | 7.282% | 0.425% |

### This build against the previous one

| Metric | Stage 1 (58 features) | Now (437 features) | Change |
| --- | --- | --- | --- |
| Test PR-AUC, raw | 0.5007 | 0.5086 | **+0.0079** |
| Test PR-AUC, calibrated | 0.4865 | 0.4901 | +0.0037 |
| Test ROC-AUC | 0.8933 | 0.8863 | **-0.0070** |
| Test Brier, raw | 0.022893 | 0.022956 | +0.000063 |
| Test Brier, calibrated | 0.023070 | 0.022980 | -0.000090 |
| Train PR-AUC | 0.8243 | 0.8798 | +0.0555 |
| Train/test PR-AUC gap | 0.3236 | 0.3712 | **+0.0476** |
| Chosen threshold | 0.0710 | 0.0793 | +0.0083 |
| False decline rate | 6.676% | 7.282% | +0.606pp |
| Total cost | $309,028 | $298,574 | **-$10,455** |
| Saving vs 0.5 | $125,098 | $147,082 | +$21,984 |

Read honestly: the extra columns bought a small ranking improvement and about
$10.5k of cost, at the price of a **wider overfitting gap and slightly worse
ROC-AUC**. PR-AUC is the metric that matters at this operating point, since the
threshold sits in the high-precision tail, so the trade is worth taking. But
"most of Vesta's engineered signal" overstates what these columns delivered on
this split. The two runs also use different training windows (day <120 then,
<110 now), so this compares pipelines end to end, not the feature block in
isolation.

## Model quality

| Metric | Value |
| --- | --- |
| PR-AUC, test, raw | 0.5086 |
| PR-AUC, test, calibrated | 0.4901 |
| ROC-AUC, test | 0.8863 |
| Brier, test, raw | 0.022956 |
| Brier, test, calibrated | 0.022980 |
| PR-AUC, train | 0.8798 |
| Best iteration | 421 of 3000 |

Sensitivity: 20% tighter costs +6.24%, 20% looser costs +0.70%. Err loose.

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

| Layer | Object | Rows | Columns |
| --- | --- | --- | --- |
| RAW | `RAW_TRANSACTIONS` | 590,540 | 395 |
| RAW | `RAW_IDENTITY` | 144,233 | 41 |
| STAGING | `stg_transactions`, `stg_identity`, `int_account_keys` | views | |
| MARTS | `fct_transactions` | 590,540 | 443 |
| MARTS | `dim_accounts` | 217,850 | 11 |

`fct_transactions` carries 443 columns: the behavioural features, the C, D and
M blocks, the full V block (V1-V339), and the identity block (id_01-id_38,
DeviceType, DeviceInfo). 437 of them reach the model; `transaction_id`,
`account_id`, `txn_day`, `transaction_dt`, `prev_transaction_dt` and the label
are excluded.

Interactive lineage is in [reports/dbt_docs/index.html](reports/dbt_docs/index.html),
exported so it outlives the Snowflake trial. 17 dbt tests pass, including an
exact row-count assertion and a direct anti-leakage assertion.

**Leakage control.** Every window feature uses
`RANGE BETWEEN <bound> PRECEDING AND 1 PRECEDING`, partitioned by account and
ordered by `transaction_dt`. `RANGE` rather than `ROWS` is deliberate: it
excludes transactions sharing a timestamp with the current row, not just the
current row. Verified three ways in
[verification.md](reports/verification.md): a dbt test asserting every
account's first transaction has zero prior counts, an independent pandas
recomputation on sampled accounts, and printed transaction sequences.

**Time splits.** Four disjoint windows, so early stopping and calibration
never share data:

| Split | Days | Rows | Fraud rate |
| --- | --- | --- | --- |
| train | < 110 | 380,100 | 0.03418 |
| early stopping | 110-129 | 62,266 | 0.04084 |
| calibration | 130-149 | 53,538 | 0.03448 |
| test | >= 150 | 94,636 | 0.03468 |

The early-stopping window runs 16.7% hot against the 0.0350 base rate. That is
inside the 30% tolerance but it is the noisiest of the four.

## Data and its limitations

IEEE-CIS Fraud Detection, 590,540 transactions over 182 days. Four limitations
shape everything above.

**The account identifier is constructed, not given.** IEEE-CIS has no account
or customer id. The proxy concatenates `card1`, `addr1`, and `txn_day - D1`,
where D1 is days since the card first appeared, so the third component is a
stable card start date. This yields 217,850 accounts. 88.7% of rows get the
full key; 11.3% fall back to a degraded tier, tracked in `account_key_tier`.
**If the proxy is wrong, every behavioural feature is wrong**, and there is no
ground truth to check it against.

**Account sizes are heavily skewed.** The median account has 1 transaction,
because 57.5% of cards appear exactly once. Weighted by transaction, which is
what the features actually see, the median account has 5. 21.2% of
transactions will never have prior-window history.

**The competition test set is unlabeled**, so the split is a time split within
the labeled training data only. There is no evaluation against the
competition's own holdout, and these numbers are not comparable to leaderboard
scores.

**Identity data covers only 24.4% of transactions.** All 40 identity columns
are null for the other 75.6%. `has_identity` is a feature in its own right,
since the absence is informative.

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

Worked arithmetic for both boundary policies is in
[verification.md](reports/verification.md) under "Reconciling the extremes",
so the cost function can be checked by hand.

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
trial and the modelling work has to outlive it. The cached table is 443 columns
wide, so the pull streams Arrow batches to parquet in float32 rather than
materialising it in memory; peak usage is about 1.6 GB.

## What is not done yet

- **No step-up authentication path.** Every decline is a hard decline. A real
  system routes some to additional verification, which changes the cost model
  substantially.
- **The account proxy is unvalidated.** It should be checked against `card2`,
  `card5`, and device fingerprints before anyone trusts it.
- **Calibration drift is unresolved.** Separating the early-stopping and
  calibration windows shrank the Brier degradation from -0.77% to -0.11% but
  did not eliminate it. The remaining cause is period-to-period drift, which
  argues for refitting the calibrator on a rolling recent window.
- **No drift monitoring**, which is what would catch the above in production.
- **No feature selection.** All 437 features go in, including 339 V columns
  whose marginal contribution is small. Pruning would cut training time and
  probably narrow the overfitting gap.
- **No CI, no scheduling, no serving.** No orchestration, no API, no monitoring
  of live scores.
- **Cost assumptions are unvalidated.** They should come from finance, not from
  a README.
