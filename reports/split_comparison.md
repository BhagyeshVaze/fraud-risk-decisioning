# Split comparison and a logistic baseline

Identical features (200), identical hyperparameters, identical isotonic calibration. Only the split rule changes between A, B and C. D changes only the model.

Time-split proportions, which run B reproduces exactly: train 0.6436, early stopping 0.1054, calibration 0.0907, test 0.1603.

**The time-split figure remains the number this project reports.** Everything below measures how much a naive evaluation would have overstated it.

## Results

```
                              run  n_test  n_fraud  fraud_rate  pr_auc  roc_auc   brier
           A time split, LightGBM   94636     3282     0.03468  0.4933   0.8926 0.02364
  B random size-matched, LightGBM   94637     3337     0.03526  0.8250   0.9628 0.01154
         C random 80/20, LightGBM  118108     4207     0.03562  0.8237   0.9631 0.01174
D time split, logistic regression   94636     3282     0.03468  0.1863   0.8279 0.03790
```

## Bootstrapped differences against the time split

1000 resamples. Unpaired, because the random splits evaluate on different rows.

```
                   comparison  metric   point  ci_low  ci_high spans_zero
B random size-matched minus A  pr_auc  0.3317  0.3113   0.3519         no
B random size-matched minus A roc_auc  0.0702  0.0632   0.0782         no
B random size-matched minus A   brier -0.0121 -0.0131  -0.0111         no
       C random 80/20 minus A  pr_auc  0.3305  0.3111   0.3501         no
       C random 80/20 minus A roc_auc  0.0705  0.0634   0.0781         no
       C random 80/20 minus A   brier -0.0119 -0.0128  -0.0109         no
```

- Random size-matched: PR-AUC 0.4933 to 0.8250, **+67.3%**. ROC-AUC 0.8926 to 0.9628, +7.9%.
- Random 80/20: PR-AUC 0.4933 to 0.8237, **+67.0%**. ROC-AUC 0.8926 to 0.9631, +7.9%.

Mechanism: under the random split **43,683 of 64,857 test accounts (67.4%)** also appear in training. Under the time split it is 14,790 of 47,734 (31.0%), and none of those are future transactions.

## Logistic baseline, paired bootstrap on identical test rows

```
             comparison  metric   point  ci_low  ci_high spans_zero
LightGBM minus logistic  pr_auc  0.3070  0.2939   0.3201         no
LightGBM minus logistic roc_auc  0.0647  0.0577   0.0713         no
LightGBM minus logistic   brier -0.0143 -0.0150  -0.0135         no
```

Logistic regression needs imputation, scaling and explicit encoding, none of which LightGBM requires. That is an advantage for the tree model, but it is inherent to the comparison: needing the preprocessing is part of the cost of a linear model.

## The comparison worth noticing

```
leakage from a random split adds   +0.3317 PR-AUC
logistic -> gradient boosting adds +0.3070 PR-AUC
```

**Switching to a random split buys more apparent performance (+0.3317) than the entire jump from logistic regression to a tuned gradient boosting model (+0.3070).** One is a modelling result. The other is measurement error. A published 0.80 on a random split and this project's 0.4933 on a time split are not evidence of different model quality.

