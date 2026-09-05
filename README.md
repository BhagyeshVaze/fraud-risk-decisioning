# Fraud Risk Decisioning System

**At the cost-optimal decline threshold this system prevents $328,658 of the
$495,244 in fraud on a held-out 30-day test period, catching 66.4% of fraud
dollars while declining 6.3% of legitimate transactions. Total modelled cost
is $296,570, which is $144,514 cheaper than a naive 0.5 cutoff and $280,724
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
0.1011. The naive cutoff looks far better on precision (0.74 against 0.27) and
is 49% more expensive, because it optimizes the wrong thing: it avoids false
declines that cost about $12 each while waving through frauds that cost about
$176 each.

Four findings worth stating plainly:

1. **The optimum is far from 0.5, and the shape around it is not stable.** At
   this operating point a threshold 20% higher (declining fewer) costs +5.2%,
   while 20% lower (declining more) costs only +0.9%. At the previous
   437-feature operating point the asymmetry ran the *other* way. Two models
   that are statistically indistinguishable on cost disagree about which
   direction is the dangerous one, which is a warning not to over-read the
   shape of a curve estimated from 3,282 frauds.
2. **The V block genuinely matters.** A clean ablation on identical splits
   shows the 339 V columns are worth **+0.0275 test PR-AUC and $17,314 of
   cost**, with a bootstrap CI of [-$27,894, -$6,624] that does not span zero.
   An earlier comparison put the figure at +0.0079, but that one was
   confounded by different training windows and understated them.
3. **Most features are dead weight, but not all of them.** Pruning to the top
   200 by gain is statistically indistinguishable from all 437 on cost and
   trains 19% faster, so it is what ships. Pruning further is not free: the
   top 100 costs **$11,011 more** and the top 50 **$22,672 more**, both
   significant.
4. **Measurement noise is roughly $10,000.** A bootstrap of the cost figure has
   a standard deviation near $9.9k, so any single change worth less than that
   cannot be distinguished from luck on this test set. Every cost comparison
   in `reports/feature_pruning.md` carries a confidence interval for that
   reason.

## Results

Evaluated on `txn_day >= 150`: 94,636 transactions, 3,282 fraudulent.

| Policy | Total cost | Fraud dollars caught | Legit declined | Saving vs optimum |
| --- | --- | --- | --- | --- |
| **Chosen threshold 0.1011** | **$296,570** | $328,658 (66.4%) | 5,734 (6.28%) | - |
| Naive 0.5 cutoff | $441,085 | $116,633 (23.6%) | 392 (0.43%) | $144,514 |
| Approve everything | $577,294 | $0 (0%) | 0 (0%) | $280,724 |
| Decline everything | $1,414,895 | $495,244 (100%) | 91,354 (100%) | $1,118,325 |

| Decision quality | At 0.1011 | At 0.50 |
| --- | --- | --- |
| Fraud caught (TP) | 2,139 | 1,127 |
| Legit declined (FP) | 5,734 | 392 |
| Fraud missed (FN) | 1,143 | 2,155 |
| True negatives | 85,620 | 90,962 |
| Precision | 0.2717 | 0.7419 |
| Recall | 0.6517 | 0.3434 |
| False decline rate | 6.277% | 0.429% |

### Feature-set comparison, identical splits

Every row uses the same four time windows, seed, and hyperparameters. Only the
feature set changes. Cost differences are against the full 437 model, with a
600-resample paired bootstrap.

| Feature set | Test PR-AUC | Gap | Fit time | Cost | vs 437 | 95% CI | Significant |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Full 437 | 0.5086 | 0.3712 | 35.9s | $298,574 | - | - | - |
| **Top 200 (shipped)** | 0.5015 | 0.3949 | **29.0s** | **$296,570** | **-$2,003** | [-$9,817, +$5,528] | no |
| Top 100 | 0.4902 | 0.3890 | 16.4s | $309,584 | +$11,011 | [+$1,715, +$20,682] | **yes** |
| Top 50 | 0.4888 | 0.3548 | 11.7s | $321,245 | +$22,672 | [+$10,675, +$33,319] | **yes** |

Top 200 is cheaper on the point estimate but the difference is inside the noise
band, so the honest claim is that it is **no worse** than the full set while
using 54% fewer features and training 19% faster. That is why it ships. Cutting
to 100 or 50 is significantly worse and is not a viable trade.

The top 200 features carry 89.4% of total gain; 148 of the 437 were never used
for a split at all.

### V-block ablation

| | Features | Test PR-AUC | Train PR-AUC | ROC-AUC | Fit time | Cost |
| --- | --- | --- | --- | --- | --- | --- |
| With V block | 437 | 0.5086 | 0.8798 | 0.8863 | 35.9s | $298,574 |
| Without V block | 98 | 0.4811 | 0.8642 | 0.8804 | 13.9s | $315,980 |
| **Contribution of V** | **339** | **+0.0275** | +0.0156 | +0.0060 | +22.0s | **-$17,406** |

Bootstrap on the cost difference: mean **-$17,314**, 95% CI
[-$27,894, -$6,624]. Entirely below zero, so this one is real. The V block is
the single most valuable feature group in the build.

## Model quality

| Metric | Value |
| --- | --- |
| PR-AUC, test, raw | 0.5015 |
| PR-AUC, test, calibrated | 0.4794 |
| ROC-AUC, test | 0.8839 |
| Brier, test, raw | 0.023162 |
| Brier, test, calibrated | 0.023176 |
| PR-AUC, train | 0.8964 |
| Best iteration | 494 of 3000 |
| Features | 200 of 437 available |

Sensitivity: a threshold 20% lower (0.0809, declining more) costs +0.87%; 20%
higher (0.1213, declining fewer) costs +5.16%. At the 437-feature operating
point the asymmetry ran the opposite way (+6.24% lower, +0.70% higher). Two
models that cannot be told apart on cost disagree on which direction is
riskier, so the local shape of the curve should not be treated as a finding.

**Caveat on the measurement.** A bootstrap of the cost at the chosen threshold
has a standard deviation of about $9,900 and a 95% interval roughly $38,000
wide. The top 100 frauds carry 25.1% of all fraud dollars in a test set of only
3,282 frauds, so the total moves substantially on where a few large
transactions fall. Treat any single improvement smaller than about $10,000 as
unproven until it is confirmed across multiple test periods.

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
DeviceType, DeviceInfo). 437 are eligible for the model; `transaction_id`,
`account_id`, `txn_day`, `transaction_dt`, `prev_transaction_dt` and the label
are excluded. **The shipped model uses the top 200 of those 437 by gain**,
listed in `models/selected_features.json` and produced by
`src/prune_features.py`. Delete that file to fall back to all 437.

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
- **Pruning was selected on the test split.** The top-200 set was chosen by
  comparing test cost across four candidates, so the choice is mildly
  optimistic. A clean confirmation needs a fresh holdout or walk-forward folds.
- **The overfitting gap did not close.** Pruning to 200 features left it at
  0.3949, slightly wider than 0.3712 at 437. Regularization and hyperparameter
  tuning, not feature count, are the remaining levers.
- **No CI, no scheduling, no serving.** No orchestration, no API, no monitoring
  of live scores.
- **Cost assumptions are unvalidated.** They should come from finance, not from
  a README.
