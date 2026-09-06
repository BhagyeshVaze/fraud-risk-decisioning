# Account proxy candidates on the walk-forward harness

Four non-overlapping 16-day test windows (days 118-181). Threshold chosen out of sample on cross-fitted calibration probabilities, then applied to test. Everything except the eight behavioural features is held identical across candidates.

## Per-fold results

```
                             config  fold_test_start  n_test  n_fraud_test  test_pr_auc  threshold         cost  cost_at_half  false_decline_rate
         stage1 (card1+addr1+start)              118   49312          1822      0.52747    0.07200 182584.25650  263305.16537             0.07202
         stage1 (card1+addr1+start)              134   42470          1402      0.44028    0.07400 137355.24242  212742.33367             0.05172
         stage1 (card1+addr1+start)              150   48044          1473      0.48182    0.08900 147796.47016  186005.81054             0.07326
         stage1 (card1+addr1+start)              166   43838          1690      0.55658    0.09000 137043.52901  217190.36060             0.05837
                         card1 only              118   49312          1822      0.54554    0.07798 177590.30008  271156.43991             0.06119
                         card1 only              134   42470          1402      0.45493    0.07400 136380.50086  200046.17682             0.05688
                         card1 only              150   48044          1473      0.49986    0.06122 142000.60577  177617.17269             0.07376
                         card1 only              166   43838          1690      0.55078    0.07664 139039.28309  212896.02084             0.04615
                        card1+start              118   49312          1822      0.53596    0.06337 186724.17674  272817.84936             0.07867
                        card1+start              134   42470          1402      0.44186    0.08700 141964.93179  201118.83370             0.03984
                        card1+start              150   48044          1473      0.48220    0.05900 150303.07049  185163.05995             0.09074
                        card1+start              166   43838          1690      0.54045    0.07600 142703.58222  235761.06599             0.04235
            card1+addr1+start+card2              118   49312          1822      0.53183    0.07600 186548.39258  262252.33921             0.07119
            card1+addr1+start+card2              134   42470          1402      0.44578    0.07000 142095.36383  208536.16998             0.08549
            card1+addr1+start+card2              150   48044          1473      0.48997    0.08200 146332.02199  182901.14743             0.05581
            card1+addr1+start+card2              166   43838          1690      0.55133    0.07664 136854.29996  224664.25814             0.05713
            card1+addr1+start+email              118   49312          1822      0.53887    0.07277 176877.88047  266817.07586             0.08960
            card1+addr1+start+email              134   42470          1402      0.45032    0.08000 137490.53363  200539.06445             0.06036
            card1+addr1+start+email              150   48044          1473      0.49406    0.08356 142963.67325  186072.85807             0.05441
            card1+addr1+start+email              166   43838          1690      0.54256    0.05800 137156.40578  229394.14719             0.06596
card1+card2+card3+card5+addr1+start              118   49312          1822      0.53531    0.05900 184994.98967  267376.70786             0.09370
card1+card2+card3+card5+addr1+start              134   42470          1402      0.44470    0.07000 141957.17526  196383.79589             0.07841
card1+card2+card3+card5+addr1+start              150   48044          1473      0.49107    0.07000 148887.84566  183334.78773             0.08909
card1+card2+card3+card5+addr1+start              166   43838          1690      0.54540    0.08356 139647.90593  213237.28798             0.05092
```

## Summary across the four folds

```
                                     pr_auc_mean  pr_auc_sd  roc_auc_mean  gap_mean    cost_total     cost_mean      cost_sd  fdr_mean        caught   fraud_total  fit_secs  pct_fraud_caught
config                                                                                                                                                                                        
card1+addr1+start+email                  0.50645    0.04343       0.88921   0.37817  594488.49314  148622.12329  19024.37767   0.06758  676716.79229  1.007748e+06    23.550          67.15139
card1 only                               0.51278    0.04484       0.89614   0.39832  595010.68980  148752.67245  19361.64365   0.05949  653740.12309  1.007748e+06    27.575          64.87139
stage1 (card1+addr1+start)               0.50154    0.05113       0.88961   0.37037  604779.49809  151194.87452  21514.63417   0.06384  658147.46926  1.007748e+06    23.825          65.30874
card1+addr1+start+card2                  0.50473    0.04690       0.88934   0.38372  611830.07835  152957.51959  22726.95959   0.06741  657402.32220  1.007748e+06    26.050          65.23480
card1+card2+card3+card5+addr1+start      0.50412    0.04611       0.89091   0.36951  615487.91652  153871.97913  21116.89830   0.07803  685312.11928  1.007748e+06    22.150          68.00432
card1+start                              0.50012    0.04700       0.89085   0.37122  621695.76124  155423.94031  21204.40622   0.06290  641240.27721  1.007748e+06    22.125          63.63102
```

## Paired comparison against the shipped proxy

Paired by fold, so each candidate is compared to stage 1 on the same four test windows. Only four pairs, so the t statistic is indicative, not decisive; the per-fold column is the honest view.

```
                          candidate  cost_diff_total  cost_diff_per_fold  cost_sd_across_folds  cost_t  folds_cheaper  pr_auc_diff  pr_auc_t
            card1+addr1+start+email         -10291.0             -2573.0                3134.0   -1.64              2      0.00492      0.78
                         card1 only          -9769.0             -2442.0                3634.0   -1.34              3      0.01124      1.96
            card1+addr1+start+card2           7051.0              1763.0                3052.0    1.16              2      0.00319      1.09
card1+card2+card3+card5+addr1+start          10708.0              2677.0                1449.0    3.70              0      0.00258      0.55
                        card1+start          16916.0              4229.0                1312.0    6.44              0     -0.00142     -0.27
```

Cheapest total cost across all four folds: **card1+addr1+start+email** at $594,488, against $604,779 for the shipped proxy (difference $-10,291).

