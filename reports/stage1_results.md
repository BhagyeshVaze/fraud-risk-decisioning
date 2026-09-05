# Stage 1 results

Fraud risk decisioning on IEEE-CIS, evaluated on a held-out time split
(txn_day >= 150, 94,636 transactions, 3,282 fraudulent).

## Headline

At a decline threshold of **0.0793**, the model prevents
**$339,844** of the **$495,244** of fraud in the test
split (68.6%), while declining
**6,652** legitimate transactions
(7.28% of all legitimate traffic).

Total modelled cost is **$298,574**, against **$445,656** at the
naive 0.5 cutoff and **$577,294** if every transaction is approved.

| Policy | Total cost | Saving vs this policy |
| --- | --- | --- |
| Chosen threshold 0.0793 | $298,574 | - |
| Naive 0.5 cutoff | $445,656 | $147,082 |
| Approve everything | $577,294 | $278,721 |
| Decline everything | $1,414,895 | $1,116,321 |

## Decision quality

| Metric | At 0.0793 | At 0.50 |
| --- | --- | --- |
| True positives (fraud caught) | 2,213 | 1,127 |
| False positives (legit declined) | 6,652 | 388 |
| False negatives (fraud missed) | 1,069 | 2,155 |
| True negatives | 84,702 | 90,966 |
| Precision | 0.2496 | 0.7439 |
| Recall | 0.6743 | 0.3434 |
| False decline rate | 7.282% | 0.425% |

## Model quality

| Metric | Value |
| --- | --- |
| PR-AUC (test, calibrated) | 0.4901 |
| PR-AUC (test, raw) | 0.5086 |
| ROC-AUC (test) | 0.8860 |
| Brier (test, raw) | 0.022956 |
| Brier (test, calibrated) | 0.022980 |

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
6.241%, so the optimum is sharp and sensitive to these assumptions.

Charts: `cost_vs_threshold.png`, `calibration_curve.png`.
