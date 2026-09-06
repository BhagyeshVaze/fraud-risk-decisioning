# Cost assumption sensitivity

Every dollar figure in this project rests on five constants that were chosen, not measured. This asks a narrow question: **if those constants are wrong, does the recommendation change?**

Two of the five dominate. Churn probability and account lifetime value enter only as their product, the expected cost of a false decline, which is $10.00 at the stated 5% and $200. That term generates $913,540 of the $1,414,895 decline-everything cost, so it is the assumption worth stressing.

Baseline: churn 5%, LTV $200, false-decline cost $10.00, optimal threshold **0.1011**, cost **$296,570**.

## Sweeping the cost of a false decline

```
 false_decline_cost_usd  implied_churn_at_200_ltv  optimal_threshold  cost_at_own_optimum  cost_at_baseline_thr  regret_usd  saving_vs_0.5
                    0.5                    0.0025             0.0440             223270.0              242097.0     18827.0       195263.0
                    1.0                    0.0050             0.0440             228523.0              244964.0     16441.0       192592.0
                    2.0                    0.0100             0.0440             239029.0              250698.0     11669.0       187250.0
                    4.0                    0.0200             0.0679             256668.0              262166.0      5498.0       176566.0
                    6.0                    0.0300             0.0790             271968.0              273634.0      1667.0       165882.0
                    8.0                    0.0400             0.1011             285102.0              285102.0         0.0       155198.0
                   10.0                    0.0500             0.1011             296570.0              296570.0         0.0       144514.0
                   15.0                    0.0750             0.1050             323916.0              325240.0      1325.0       117804.0
                   20.0                    0.1000             0.1356             346629.0              353910.0      7281.0        91094.0
                   30.0                    0.1500             0.1356             381449.0              411250.0     29801.0        37674.0
                   50.0                    0.2500             0.2160             418502.0              525930.0    107428.0       -69166.0
                   80.0                    0.4000             0.3210             461203.0              697950.0    236748.0      -229426.0
                  120.0                    0.6000             0.3440             483071.0              927310.0    444239.0      -443106.0
```

`regret_usd` is what it costs to keep using the baseline threshold of 0.1011 when the true false-decline cost is the value in that row. It is the number that matters: a threshold is only wrong if using it is expensive.

Worst regret across the swept range: **$444,239** at a false-decline cost of $120.00 (60.00% churn at $200 LTV).

**The recommendation flips** at a false-decline cost of $50.00 or above: past that point the baseline threshold is no longer better than the naive 0.5 cutoff.

## Two-dimensional surface, churn probability by lifetime value

Optimal threshold at each combination:

```
           LTV $50  LTV $100  LTV $150  LTV $200  LTV $300  LTV $500  LTV $800
churn 1%    0.0440    0.0440    0.0440    0.0440    0.0679    0.0679    0.1011
churn 2%    0.0440    0.0440    0.0679    0.0679    0.0790    0.1011    0.1050
churn 3%    0.0440    0.0679    0.0679    0.0790    0.1011    0.1050    0.1356
churn 5%    0.0440    0.0679    0.0790    0.1011    0.1050    0.1356    0.2160
churn 8%    0.0679    0.1011    0.1011    0.1050    0.1356    0.2160    0.2160
churn 12%   0.0790    0.1011    0.1050    0.1356    0.2160    0.2160    0.3440
churn 20%   0.1011    0.1356    0.1356    0.2160    0.2160    0.3440    0.6440
```

Regret in dollars from using the baseline 0.1011 threshold instead:

```
           LTV $50  LTV $100  LTV $150  LTV $200  LTV $300  LTV $500  LTV $800
churn 1%   18827.0   16441.0   14055.0   11669.0    7937.0    3059.0       0.0
churn 2%   16441.0   11669.0    7937.0    5498.0    1667.0       0.0    2190.0
churn 3%   14055.0    7937.0    4278.0    1667.0       0.0    1325.0   16289.0
churn 5%    9283.0    3059.0      80.0       0.0    1325.0   18541.0   66518.0
churn 8%    5498.0       0.0       0.0    2190.0   16289.0   66518.0  164702.0
churn 12%   1667.0       0.0    3920.0   16289.0   50154.0  148338.0  317087.0
churn 20%      0.0    7281.0   29801.0   66518.0  148338.0  338279.0  657567.0
```

Threshold ranges from 0.0440 to 0.6440 across the grid, a 15x spread. Regret ranges from $0 to $657,567.

**55% of the grid has regret below $9,900**, the measured one-standard-deviation noise on the cost estimate. Within that region the baseline threshold is indistinguishable from the assumption-specific optimum, so getting the constants wrong there costs nothing detectable.

Chart: `reports/cost_sensitivity.png`.

