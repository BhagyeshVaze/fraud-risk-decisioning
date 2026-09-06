# Feature pruning and V-block ablation

Splits held fixed across every variant: train `txn_day < 110` (380,100 rows), early stopping `110-129` (62,266), calibration `130-149` (53,538), test `>= 150` (94,636).

Same seed, same hyperparameters, same calibration procedure throughout. Only the feature set changes.

## Gain ranking

394 of 437 features have non-zero gain. 43 were never used for a split.

- top 50 features carry **74.60%** of total gain
- top 100 features carry **88.83%** of total gain
- top 200 features carry **97.45%** of total gain

Top 20:

```
        feature       gain  pct_of_total_gain  cum_pct
           v257 41998.0309             6.0543   6.0543
           v258 41303.0121             5.9541  12.0083
             c1 35300.7517             5.0888  17.0971
    device_info 33526.0268             4.8330  21.9301
            c13 26667.9166             3.8443  25.7745
            c14 26650.0655             3.8418  29.6162
           v294 21391.6356             3.0837  32.6999
  r_emaildomain 15935.9021             2.2973  34.9972
             d2 15704.8176             2.2639  37.2611
  p_emaildomain 15331.2679             2.2101  39.4712
transaction_amt 14710.6371             2.1206  41.5919
          id_31 14141.8335             2.0386  43.6305
            d15 11535.7103             1.6629  45.2934
           v189  8881.8374             1.2804  46.5738
 prior_amt_mean  8766.4907             1.2637  47.8375
            d10  8399.5411             1.2108  49.0484
          card6  8027.5098             1.1572  50.2056
             d1  7867.0181             1.1341  51.3397
             d4  7865.1058             1.1338  52.4735
            c11  7816.7037             1.1268  53.6003
```

Full ranking: `reports/feature_gain_ranking.csv`.

## Pruning results

```
 variant  features  fit_secs  best_iter  test_pr_auc  train_pr_auc      gap  test_roc_auc  threshold  cost_usd      fdr  cost_vs_full
full 437       437      35.9        421     0.508558      0.879783 0.371225      0.886324   0.079339 298573.68 0.072816          0.00
 top 200       200      29.0        494     0.501532      0.896429 0.394897      0.883890   0.101064 296570.28 0.062767      -2003.40
 top 100       100      16.4        431     0.490221      0.879252 0.389031      0.880786   0.088000 309584.31 0.066937      11010.63
  top 50        50      11.7        380     0.488805      0.843598 0.354793      0.880961   0.097000 321245.46 0.067474      22671.78
```

### Cost differences with confidence intervals

Paired bootstrap, 600 resamples of the test split. Negative means cheaper than the full 437 model. An interval spanning zero means the difference is not distinguishable from noise.

```
variant  point_diff_usd  boot_mean_usd  ci_low_usd  ci_high_usd significant
top 200        -2003.39       -2162.72    -9817.02      5527.79          no
top 100        11010.63       10744.51     1715.06     20681.90         yes
 top 50        22671.78       22603.97    10675.28     33319.04         yes
```

## V-block ablation

Identical splits, identical everything else. 339 V columns removed, leaving 98 features.

```
          variant  features  test_pr_auc  train_pr_auc       gap  test_roc_auc  fit_secs  cost_usd
     with V block       437     0.508558      0.879783  0.371225      0.886324      35.9 298573.68
  without V block        98     0.481073      0.864178  0.383105      0.880355      13.9 315979.95
contribution of V       339     0.027485      0.015606 -0.011880      0.005969      22.0 -17406.27
```

Bootstrap on cost, with V minus without V: mean **$-17,313.86**, 95% CI [$-27,893.96, $-6,624.06]. Distinguishable from noise.

In isolation, on identical splits, the 339 V columns move test PR-AUC by **+0.0275** (0.4811 without, 0.5086 with).

## Decision

Cheapest variant on the point estimate: **top 200** at $296,570.28 against $298,573.68 for the full model.

The saving is **not** significant (CI [$-9,817.02, $5,527.79] spans zero). It is adopted anyway on the parsimony rule: it is no worse on cost, uses far fewer features (200 against 437), and trains in 29.0s against 35.9s. A simpler model that is statistically indistinguishable is the better model to operate.

Written to `models/selected_features.json`. `src/model.py train` reads that file when present and falls back to all features otherwise.

