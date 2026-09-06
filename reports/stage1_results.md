# Stage 1 results

Evaluated on a held-out time split (txn_day >= 150, 94,636 transactions,
3,282 fraudulent).

## Headline

At a decline threshold of **0.1160**, the model prevents **$341,479**
of the **$495,244** of fraud in the test split
(69.0%), while declining **5,960** legitimate
transactions (6.52% of legitimate traffic).

Total modelled cost is **$291,334**, against **$446,463** at the naive
0.5 cutoff and **$577,294** if every transaction is approved.

| Policy | Total cost | Saving vs this policy |
| --- | --- | --- |
| Chosen threshold 0.1160 | $291,334 | - |
| Naive 0.5 cutoff | $446,463 | $155,129 |
| Approve everything | $577,294 | $285,961 |
| Decline everything | $1,414,895 | $1,123,561 |

## Decision quality

| Metric | At 0.1160 | At 0.50 |
| --- | --- | --- |
| True positives | 2,137 | 992 |
| False positives | 5,960 | 323 |
| False negatives | 1,145 | 2,290 |
| True negatives | 85,394 | 91,031 |
| Precision | 0.2639 | 0.7544 |
| Recall | 0.6511 | 0.3023 |
| False decline rate | 6.524% | 0.354% |

## Model quality

| Metric | Value |
| --- | --- |
| PR-AUC (test, raw) | 0.4933 |
| PR-AUC (test, calibrated) | 0.4720 |
| ROC-AUC (test) | 0.8926 |
| Brier (test, raw) | 0.024119 |
| Brier (test, calibrated) | 0.023636 |

## Cost assumptions

Assumptions, not measurements.

| Assumption | Value |
| --- | --- |
| Chargeback fee | $25.00 |
| Margin rate | 2.5% |
| Churn probability after a false decline | 5% |
| Account lifetime value | $200.00 |
| Manual review cost | $2.00 |

Sensitivity: moving the threshold 20% either way changes total cost by at most
4.602%. Full surface in `cost_sensitivity.md`.

Charts: `cost_vs_threshold.png`, `calibration_curve.png`.
