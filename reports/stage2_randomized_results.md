# Stage 2, randomised arm

Executes `reports/stage2_preregistration.md`, committed at `e5bb4f5` before this code existed. Champion 0.5000, challenger 0.1160, assignment by md5 hash with salt `stage2-v1`.

Registered up front: the direction is not in doubt. The purpose is the machinery and the benchmark for the observational half.

## SRM check

```
accounts challenger   23,649
accounts champion     24,085
total                 47,734
split                0.49543 / 0.50457
chi-square (1 df)    3.9824
p-value              0.0460
halt threshold       p < 0.001
verdict              PASS
```

Transaction counts, descriptive only since cluster sizes vary: challenger 47,740, champion 46,896.

## Primary: mean cost per account

```
challenger mean   $    6.7229   n 23,649
champion   mean   $    8.4974   n 24,085
difference        $   -1.7745 per account
                  $   -84,705 scaled to all 47,734 accounts
SE                $    0.9469
t                      -1.874
p (two-sided)     6.092e-02
95% CI (normal)   [$-3.6304, $+0.0813] per account
95% CI (BCa)      [$-3.7550, $-0.0856] per account
```

**Normal and BCa intervals disagree.** Normal [-3.6304, +0.0813] includes zero; BCa [-3.7550, -0.0856] excludes zero. The pre-registration names the normal interval as primary, so **it governs the ship decision**. Switching to whichever interval gives the preferred answer is what pre-registration exists to prevent.

## CUPED, as multivariate regression adjustment

```
unadjusted  effect $  -1.7745   SE $0.9469
adjusted    effect $  -1.9331   SE $0.9306
variance reduction     3.40%   (registered expectation 4.9%)
```

Realised reduction 3.40% against a registered 4.9%. Pre-period coverage is 31.6%, the binding constraint. **A null result for CUPED on this data**, registered as likely in advance.

## Guardrail: false decline rate against the 8.0% ceiling

**challenger** threshold 0.116

```
false decline rate   6.3615%
cluster-robust SE    0.2936pp   (clusters = 23,143 accounts)
one-sided 95% UB     6.8446%
ceiling              8.0%
H0: FDR >= 8.0%      z -5.580   p 1.203e-08
verdict              PASS, below the ceiling
```

Champion arm, for reference: false decline rate 0.3436% (SE 0.0412pp).

## Secondary metrics

```
       arm  transactions  frauds  declined  decline_rate  recall  precision  fraud_dollars_caught  fraud_dollars_total  cost_per_txn  pct_fraud_dollars_caught
challenger         47740    1792      4058       0.08500 0.63337    0.27969              173697.0             261888.0        3.3303                     66.32
  champion         46896    1490       649       0.01384 0.33087    0.75963               57382.0             233357.0        4.3641                     24.59
```

## Subgroups, descriptive only

Registered as descriptive and excluded from the ship decision. Two of the four are underpowered at 15.5% and 14.5%.

```
               subgroup  accounts  effect_per_account  ci_low  ci_high  scaled_total crosses_zero
                    new     32628             -3.0442 -5.4734  -0.6149      -99325.0           no
            established     15106              0.9288 -1.7036   3.5612       14030.0          yes
  low ticket (< $78.33)     23858             -0.0599 -0.9788   0.8591       -1429.0          yes
high ticket (>= $78.33)     23876             -3.5605 -7.1580   0.0371      -85010.0          yes
```

## Effect by account risk quintile

Registered to lead the recommendation ahead of the average treatment effect.

```
 quintile  accounts  mean_risk  mean_cost_champion  effect_per_account  ci_low  ci_high  scaled_total
        1      9547    0.00243               0.199              0.2324  -0.102    0.567        2219.0
        2      9547    0.00373               0.232              0.6715   0.021    1.322        6411.0
        3      9546    0.00654               1.082              0.5473  -0.524    1.618        5224.0
        4      9547    0.01441               2.960              0.8317  -1.529    3.192        7940.0
        5      9547    0.11996              37.895            -10.9339 -19.732   -2.135     -104386.0
```

## Targeted application follow-up

Challenger only above a risk cutoff, champion below. A paired counterfactual, since both outcomes are known for every account.

```
 risk_cutoff  accounts_on_challenger  pct_accounts  total_cost  legit_declined  false_decline_rate_pct  fraud_dollars_caught  vs_challenger_everywhere
       0.000                   47734        100.00    291334.0            5960                   6.524              341479.0                       0.0
       0.005                   27108         56.79    291334.0            5960                   6.524              341479.0                       0.0
       0.010                   18595         38.96    291334.0            5960                   6.524              341479.0                       0.0
       0.020                   10433         21.86    291260.0            5955                   6.519              341479.0                     -74.0
       0.030                    7735         16.20    291545.0            5906                   6.465              340445.0                     211.0
       0.050                    5190         10.87    293007.0            5702                   6.242              336676.0                    1673.0
       0.080                    3634          7.61    293842.0            5065                   5.544              326801.0                    2508.0
       0.120                    2481          5.20    294221.0            3939                   4.312              309789.0                    2887.0
       0.200                    1238          2.59    325083.0            1746                   1.911              247841.0                   33749.0
       1.010                       0          0.00    446463.0             323                   0.354              113617.0                  155129.0
```

Cheapest cutoff: **0.02**, challenger on 21.86% of accounts, $291,260 against $291,334 uniform.

**Targeting buys almost nothing at the cost-optimal cutoff.** It declines 5,955 legitimate transactions against 5,960, a 0.1% reduction, catching identical fraud dollars. Accounts below 0.02 mean risk rarely cross a 0.116 threshold anyway. The registered expectation that targeting would dominate is **not supported**.

There is a real trade higher up, but it costs money. At 0.12, false declines fall to 3,939 (34% lower) for $+2,887 of cost and $-31,690 less fraud caught.

## Paired counterfactual benchmark

Available only because this is a replay. An efficiency benchmark and a bug detector: material disagreement would indicate a fault in assignment or analysis.

```
paired effect      $-3.2499 per account
paired SE          $0.4690
paired 95% CI      [$-4.1691, $-2.3306]
randomised effect  $-1.7745   SE $0.9469
efficiency ratio   2.02x
agreement          consistent
```

## Ship recommendation

```
1 primary significant, challenger favoured   FAIL
2 upper bound of 95% CI below zero           FAIL
3 guardrail below the 8.0% ceiling at its UB PASS
decision                                     DO NOT SHIP
```

