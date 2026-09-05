# Stage 1 results

Fraud risk decisioning on IEEE-CIS, evaluated on a held-out time split
(txn_day >= 150, 94,636 transactions, 3,282 fraudulent).

## Headline

At a decline threshold of **0.0710**, the model prevents
**$319,337** of the **$495,244** of fraud in the test
split (64.5%), while declining
**6,099** legitimate transactions
(6.68% of all legitimate traffic).

Total modelled cost is **$309,028**, against **$434,127** at the
naive 0.5 cutoff and **$577,294** if every transaction is approved.

| Policy | Total cost | Saving vs this policy |
| --- | --- | --- |
| Chosen threshold 0.0710 | $309,028 | - |
| Naive 0.5 cutoff | $434,127 | $125,098 |
| Approve everything | $577,294 | $268,266 |
| Decline everything | $1,414,895 | $1,105,867 |

## Decision quality

| Metric | At 0.0710 | At 0.50 |
| --- | --- | --- |
| True positives (fraud caught) | 2,146 | 1,056 |
| False positives (legit declined) | 6,099 | 266 |
| False negatives (fraud missed) | 1,136 | 2,226 |
| True negatives | 85,255 | 91,088 |
| Precision | 0.2603 | 0.7988 |
| Recall | 0.6539 | 0.3218 |
| False decline rate | 6.676% | 0.291% |

## Model quality

| Metric | Value |
| --- | --- |
| PR-AUC (test, calibrated) | 0.4865 |
| PR-AUC (test, raw) | 0.5007 |
| ROC-AUC (test) | 0.8932 |
| Brier (test, raw) | 0.022893 |
| Brier (test, calibrated) | 0.023070 |

Isotonic calibration did not improve the Brier score on the test split. The
calibrator is fitted on days 120-149 and applied to days 150+, and the same
window is used for early stopping. See `verification.md` for the decile tables.

## Cost assumptions

Stated as assumptions, not measurements. IEEE-CIS has no chargeback, margin,
or churn data.

| Assumption | Value |
| --- | --- |
| Chargeback fee | $25.00 |
| Margin rate | 2.5% |
| Churn probability after a false decline | 5% |
| Account lifetime value | $200.00 |
| Manual review cost | $2.00 |

Cost model: fraud approved costs the transaction amount plus the chargeback
fee; a legitimate transaction declined costs the lost margin plus the churn-
weighted lifetime value plus a review; fraud declined costs a review.

Sensitivity: moving the threshold 20% either way changes total cost by at most
3.451%, so the optimum is sharp and sensitive to these assumptions.

Charts: `cost_vs_threshold.png`, `calibration_curve.png`.
