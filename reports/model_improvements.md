# Model improvements

Four non-overlapping 16-day test windows, days 118-181. Threshold chosen out of sample. Identical folds for every variant.

Intervention 1 normalises **14** D columns present in the feature set.
Intervention 2 count-encodes **6** high-cardinality categoricals: device_info, r_emaildomain, p_emaildomain, id_31, id_30, id_33.
Intervention 3 parameters: {'num_leaves': 31, 'min_child_samples': 200, 'colsample_bytree': 0.5, 'subsample': 0.7, 'reg_lambda': 20.0, 'reg_alpha': 1.0, 'learning_rate': 0.03}.

## Per-fold results

```
           config  fold_test_start  test_pr_auc  train_pr_auc  threshold         cost  false_decline_rate  fit_secs
       A baseline              118      0.52749       0.87996    0.07200 182584.25650             0.07202      14.4
       A baseline              134      0.44028       0.83740    0.07400 137355.24242             0.05172      13.5
       A baseline              150      0.48179       0.88057    0.08300 147796.47016             0.07326      20.8
       A baseline              166      0.55658       0.88971    0.09000 137043.52901             0.05837      25.3
   B D-normalised              118      0.54704       0.94702    0.07500 157596.43670             0.07875      22.1
   B D-normalised              134      0.47721       0.88859    0.07600 128740.11919             0.05484      16.5
   B D-normalised              150      0.50947       0.95547    0.08955 136946.56083             0.05871      31.8
   B D-normalised              166      0.56353       0.94794    0.08200 143356.05583             0.03768      36.2
  C count-encoded              118      0.53851       0.84620    0.07600 184245.06606             0.05730      11.9
  C count-encoded              134      0.45333       0.85581    0.07400 137690.54548             0.06416      15.7
  C count-encoded              150      0.49243       0.90262    0.10462 144451.70452             0.05177      32.6
  C count-encoded              166      0.55021       0.89881    0.07600 145655.91477             0.04318      38.5
    D regularised              118      0.52723       0.83150    0.05400 187604.18523             0.09431      72.6
    D regularised              134      0.44239       0.82238    0.04600 147866.73580             0.09728      86.3
    D regularised              150      0.47411       0.82281    0.08900 148160.65797             0.04894     102.0
    D regularised              166      0.53594       0.85201    0.08400 138888.12909             0.04866     155.2
BC D-norm + count              118      0.54491       0.92296    0.05100 162254.52793             0.08429      25.4
BC D-norm + count              134      0.45999       0.87492    0.06200 130327.77773             0.06682      21.5
BC D-norm + count              150      0.51690       0.95557    0.09763 135441.10008             0.05104      44.6
BC D-norm + count              166      0.56699       0.96570    0.08356 138495.12963             0.05433      40.1
  BD D-norm + reg              118      0.54877       0.88422    0.07400 165301.35257             0.08090      74.1
  BD D-norm + reg              134      0.47529       0.86640    0.04800 139291.56787             0.10514      83.8
  BD D-norm + reg              150      0.49671       0.88748    0.09500 141356.31298             0.06373     128.3
  BD D-norm + reg              166      0.56099       0.89145    0.08356 133261.58797             0.05466     182.2
   CD count + reg              118      0.53119       0.82776    0.05062 192099.39322             0.10375      73.9
   CD count + reg              134      0.44242       0.79007    0.08600 138489.69460             0.05177      66.9
   CD count + reg              150      0.48224       0.83021    0.08500 146364.66435             0.07518     114.1
   CD count + reg              166      0.54900       0.82785    0.06674 133285.34361             0.04942     126.7
    BCD all three              118      0.54821       0.88902    0.07277 164551.49583             0.07844     108.1
    BCD all three              134      0.47227       0.86423    0.05812 131679.30300             0.07790     108.3
    BCD all three              150      0.50105       0.89770    0.06791 145582.58768             0.06899     200.7
    BCD all three              166      0.56155       0.89157    0.09400 136276.46071             0.03761     188.7
```

## Summary, sorted by total cost across the four folds

```
                   pr_auc_mean  pr_auc_sd  roc_auc_mean      gap    brier    cost_total      cost_sd      fdr  pct_fraud_caught  fit_secs
config                                                                                                                                   
BC D-norm + count      0.52220    0.04626       0.89957  0.40759  0.02257  566518.53537  14156.75663  0.06412          68.88228    32.900
B D-normalised         0.52431    0.03870       0.90078  0.41044  0.02249  566639.17255  12192.70887  0.05749          67.17920    26.650
BCD all three          0.52077    0.04146       0.89427  0.36486  0.02266  578089.84722  14551.39550  0.06573          68.29046   151.450
BD D-norm + reg        0.52044    0.04102       0.89450  0.36195  0.02269  579210.82138  14090.68469  0.07611          70.60390   117.100
A baseline             0.50154    0.05114       0.88964  0.37037  0.02307  604779.49809  21514.63417  0.06384          65.30874    18.500
CD count + reg         0.50121    0.04830       0.88514  0.31776  0.02321  610239.09577  26902.48856  0.07003          66.34996    95.400
C count-encoded        0.50862    0.04450       0.89126  0.36724  0.02290  612043.23082  21115.88681  0.05410          61.85504    24.675
D regularised          0.49492    0.04442       0.88282  0.33726  0.02334  622519.70810  21746.24951  0.07230          65.57345   104.025
```

## Paired against baseline, by fold

```
          variant  cost_diff_total  cost_t  folds_cheaper  pr_auc_diff  pr_auc_t  folds_better_pr  gap_change
BC D-norm + count         -38261.0   -2.09              3      0.02066      3.97                4     0.03722
   B D-normalised         -38140.0   -1.49              3      0.02278      3.58                4     0.04007
    BCD all three         -26690.0   -1.70              4      0.01923      3.47                4    -0.00551
  BD D-norm + reg         -25569.0   -1.59              3      0.01890      2.96                4    -0.00843
   CD count + reg           5460.0    0.47              2     -0.00032     -0.13                3    -0.05262
  C count-encoded           7264.0    0.73              1      0.00708      1.57                3    -0.00313
    D regularised          17740.0    1.97              0     -0.00662     -1.29                1    -0.03311
```

`folds_cheaper` and `folds_better_pr` are out of 4. With only four folds a t statistic is indicative; consistency across folds is the more trustworthy signal.

