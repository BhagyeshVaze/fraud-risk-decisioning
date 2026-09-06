# Stage 2, observational half

Implements section 10 of the pre-registration. Simulates a non-random rollout steered by account risk, then recovers the effect the way a post-hoc analyst would.

## The two targets, and why both are needed

Both potential outcomes are known for every account, so the **exact ATE is computable: $-3.2499 per account** ($-155,129 scaled). The randomised arm estimated $-1.7745 (SE $0.9469), one noisy draw at that quantity.

Matching estimates the effect **on the treated**. The effect is strongly heterogeneous, so under a risk-steered rollout the ATT differs from the ATE by construction. Judging an ATT estimate against the ATE would charge estimand mismatch to confounding. Both are reported.

## Results by scenario

Exact ATE across all accounts: **$-3.2499 per account**.

**beta +0.5 (risk-seeking)**, true ATT $-5.3830

```
                  estimator  estimate   ci_low  ci_high  bias_vs_ATT  bias_pct_vs_ATT  bias_vs_ATE sign_correct
           naive difference    0.5158      NaN      NaN       5.8988            109.6       3.7657           NO
   PSM, observed covariates   -1.3822  -3.2974   0.2076       4.0008             74.3       1.8676          yes
PSM, oracle (includes risk)   -9.4297 -12.1480  -6.5752      -4.0467            -75.2      -6.1798          yes
```

**beta +1.5 (risk-seeking)**, true ATT $-6.1864

```
                  estimator  estimate  ci_low  ci_high  bias_vs_ATT  bias_pct_vs_ATT  bias_vs_ATE sign_correct
           naive difference    5.6138     NaN      NaN      11.8002            190.7       8.8637           NO
   PSM, observed covariates    4.1011  2.6733   5.5606      10.2875            166.3       7.3509           NO
PSM, oracle (includes risk)   -0.8856 -2.3848   0.6225       5.3008             85.7       2.3643          yes
```

**beta +3.0 (risk-seeking)**, true ATT $-6.4688

```
                  estimator  estimate  ci_low  ci_high  bias_vs_ATT  bias_pct_vs_ATT  bias_vs_ATE sign_correct
           naive difference    8.1273     NaN      NaN      14.5960            225.6      11.3771           NO
   PSM, observed covariates    6.3299  5.0154   7.5309      12.7987            197.9       9.5798           NO
PSM, oracle (includes risk)    1.4367 -0.2785   2.9938       7.9055            122.2       4.6866           NO
```

**beta -0.5 (risk-averse)**, true ATT $-0.9709

```
                  estimator  estimate  ci_low  ci_high  bias_vs_ATT  bias_pct_vs_ATT  bias_vs_ATE sign_correct
           naive difference   -7.9247     NaN      NaN      -6.9538           -716.2      -4.6748          yes
   PSM, observed covariates   -4.5365 -6.1900  -3.0688      -3.5656           -367.3      -1.2866          yes
PSM, oracle (includes risk)   -0.9321 -2.0326   0.4033       0.0388              4.0       2.3178          yes
```

**beta -1.5 (risk-averse)**, true ATT $-0.1318

```
                  estimator  estimate  ci_low  ci_high  bias_vs_ATT  bias_pct_vs_ATT  bias_vs_ATE sign_correct
           naive difference  -10.7969     NaN      NaN     -10.6651          -8091.2      -7.5470          yes
   PSM, observed covariates   -4.5762 -5.9806  -3.1892      -4.4444          -3371.8      -1.3263          yes
PSM, oracle (includes risk)    0.2786 -0.7426   1.2842       0.4104            311.4       3.5285           NO
```

**beta -3.0 (risk-averse)**, true ATT $-0.1177

```
                  estimator  estimate  ci_low  ci_high  bias_vs_ATT  bias_pct_vs_ATT  bias_vs_ATE sign_correct
           naive difference  -12.6438     NaN      NaN     -12.5261         -10639.8      -9.3939          yes
   PSM, observed covariates   -5.9158 -7.4914  -4.3984      -5.7981          -4924.9      -2.6659          yes
PSM, oracle (includes risk)    0.3061 -0.6292   1.1335       0.4239            360.0       3.5560           NO
```

## Bias summary

```
estimator  PSM, observed covariates  PSM, oracle (includes risk)  naive difference
beta                                                                              
-3.0                        -5.7981                       0.4239          -12.5261
-1.5                        -4.4444                       0.4104          -10.6651
-0.5                        -3.5656                       0.0388           -6.9538
 0.5                         4.0008                      -4.0467            5.8988
 1.5                        10.2875                       5.3008           11.8002
 3.0                        12.7987                       7.9055           14.5960
```

Bias against the ATT, dollars per account. The oracle column is the control: the same machinery plus the one variable that drove assignment.

## Matching diagnostics

```
                scenario      psm  treated_matched  unique_controls  dropped_support  dropped_caliper  caliper  mean_abs_SMD_before  mean_abs_SMD_after
beta +0.5 (risk-seeking) observed            23866             7723                0               14   0.0244               0.0313              0.0089
beta +0.5 (risk-seeking)   oracle            23699             9865              117               64   0.0989               0.0446              0.0152
beta +1.5 (risk-seeking) observed            23855             7633                0               12   0.0418               0.0450              0.0110
beta +1.5 (risk-seeking)   oracle            23011             9292              802               54   0.3042               0.0642              0.0113
beta +3.0 (risk-seeking) observed            23907             7367                0               20   0.0587               0.0548              0.0115
beta +3.0 (risk-seeking)   oracle            22629             8707             1255               43   0.4157               0.0763              0.0172
 beta -0.5 (risk-averse) observed            24010             7700                0                4   0.0214               0.0231              0.0084
 beta -0.5 (risk-averse)   oracle            24012             9875                1                1   0.0947               0.0366              0.0107
 beta -1.5 (risk-averse) observed            23831             7575                3                3   0.0422               0.0431              0.0169
 beta -1.5 (risk-averse)   oracle            23837             9438                0                0   0.2870               0.0619              0.0126
 beta -3.0 (risk-averse) observed            23758             7466                2                1   0.0543               0.0579              0.0122
 beta -3.0 (risk-averse)   oracle            23751             8798               10                0   0.4133               0.0792              0.0123
```

## Rosenbaum sensitivity, PSM on observed covariates

```
                scenario gamma_to_overturn  matched_pairs  pct_pairs_exactly_zero  pct_of_nonzero_pairs_negative  pct_of_total_abs_diff_in_top_1pct  mean_pair_diff  median_pair_diff
beta +0.5 (risk-seeking)               1.0          23866                    86.5                           15.8                               62.4         -1.3822               0.0
beta +1.5 (risk-seeking)               1.0          23855                    84.5                            6.9                               58.6          4.1011               0.0
beta +3.0 (risk-seeking)               1.0          23907                    83.4                            5.6                               55.2          6.3299               0.0
 beta -0.5 (risk-averse)               1.0          24010                    91.4                           37.8                               68.0         -4.5365               0.0
 beta -1.5 (risk-averse)              >6.0          23831                    93.7                           54.9                               72.5         -4.5762               0.0
 beta -3.0 (risk-averse)              >6.0          23758                    95.0                           69.5                               78.8         -5.9158               0.0
```

**The Rosenbaum bounds are uninformative here.** 83-95% of matched pairs differ by exactly zero, and the top 1% of absolute differences carries over half the mass. A signed-rank statistic counts pairs; this effect lives in the size of a few. Reported as inapplicable rather than dropped, since the pre-registration required it.

Chart: `reports/stage2_observational.png`.

