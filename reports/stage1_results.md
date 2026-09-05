# Stage 1 results

Fraud risk decisioning on IEEE-CIS, evaluated on a held-out time split
(txn_day >= 150, 94,636 transactions, 3,282 fraudulent).

## Headline

At a decline threshold of **0.1011**, the model prevents
**$328,658** of the **$495,244** of fraud in the test
split (66.4%), while declining
**5,734** legitimate transactions
(6.28% of all legitimate traffic).

Total modelled cost is **$296,570**, against **$441,085** at the
naive 0.5 cutoff and **$577,294** if every transaction is approved.

| Policy | Total cost | Saving vs this policy |
| --- | --- | --- |
| Chosen threshold 0.1011 | $296,570 | - |
| Naive 0.5 cutoff | $441,085 | $144,514 |
| Approve everything | $577,294 | $280,724 |
| Decline everything | $1,414,895 | $1,118,325 |

## Decision quality

| Metric | At 0.1011 | At 0.50 |
| --- | --- | --- |
| True positives (fraud caught) | 2,139 | 1,127 |
| False positives (legit declined) | 5,734 | 392 |
| False negatives (fraud missed) | 1,143 | 2,155 |
| True negatives | 85,620 | 90,962 |
| Precision | 0.2717 | 0.7419 |
| Recall | 0.6517 | 0.3434 |
| False decline rate | 6.277% | 0.429% |

## Model quality

| Metric | Value |
| --- | --- |
| PR-AUC (test, calibrated) | 0.4794 |
| PR-AUC (test, raw) | 0.5015 |
| ROC-AUC (test) | 0.8838 |
| Brier (test, raw) | 0.023162 |
| Brier (test, calibrated) | 0.023176 |

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
5.155%, so the optimum is sharp and sensitive to these assumptions.

Charts: `cost_vs_threshold.png`, `calibration_curve.png`.
