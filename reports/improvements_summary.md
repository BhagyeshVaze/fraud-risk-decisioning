# Improvement pass: what was tried, what was kept, what was rejected

Six experiments. Two changes shipped, four rejected. Every decision below is
tied to a measurement, and the rejections are recorded because a rejected
hypothesis is as informative as an accepted one.

## The three infrastructure improvements

### 1. Cost assumption sensitivity: SHIPPED

`src/cost_sensitivity.py`, `reports/cost_sensitivity.md`.

The headline rested on five invented constants. Churn probability and lifetime
value enter only through their product, the cost of one false decline, which
generates $913,540 of the $1,414,895 decline-everything cost.

- Sweeping that product from $0.50 to $120, the optimal threshold ranges from
  0.044 to 0.344.
- Over a 7x7 churn-by-LTV grid the threshold spans **0.044 to 0.644, a 15x
  range**.
- **55% of the grid has regret below $9,900**, the measured noise floor, so in
  that region using the shipped threshold costs nothing detectable.
- The qualitative recommendation survives everywhere plausible: only at a
  false-decline cost of **$50 or more** (25% churn at $200 LTV) does the naive
  0.5 cutoff become competitive.

Conclusion: the direction of the recommendation is robust; the specific number
is not. Both statements are now in the README.

### 2. Walk-forward evaluation harness: SHIPPED

`src/walkforward.py`.

Replaces a single 32-day test window with four non-overlapping 16-day windows
spanning days 118-181, each with its own train, early-stopping and calibration
windows behind it.

It also fixes a protocol flaw carried since stage 1: the decline threshold was
previously swept on the test split, which is optimistic. The threshold is now
chosen on **cross-fitted out-of-fold calibration probabilities** and only then
applied to test. The test-optimal threshold is still reported so the size of
the old optimism stays visible.

This harness produced every comparison below.

### 3. Better account proxy: REJECTED

`src/exp_account_proxy.py`, `reports/proxy_candidates.md`.

Six candidate keys, four folds each.

| Candidate | Cost vs stage 1 | t | Folds cheaper | PR-AUC diff |
| --- | --- | --- | --- | --- |
| card1+addr1+start+email | -$10,291 | -1.64 | 2 of 4 | +0.0049 |
| card1 only | -$9,769 | -1.34 | 3 of 4 | +0.0112 |
| card1+addr1+start+card2 | +$7,051 | 1.16 | 2 of 4 | +0.0032 |
| card1+card2+card3+card5+addr1+start | +$10,708 | 3.70 | 0 of 4 | +0.0026 |
| card1+start | +$16,916 | 6.44 | 0 of 4 | -0.0014 |

**Kept the stage 1 proxy.** The two candidates that beat it do so by about $10k
across four folds on a $600k base, with t statistics near -1.5 and fold wins of
2 and 3 out of 4. That is inside the noise. The two that lose, lose convincingly.

I had predicted this would be the largest available gain. It was not, and the
reason is instructive: only **eight** of the 200 model features derive from the
account key, so changing the key perturbs 4% of the inputs. Published solutions
that turned on account identity built dozens of aggregations over it. The lever
is how much is built on the key, not the key itself.

One structural caution: `card1` alone puts 15.2% of its multi-transaction
accounts in a state where they contain both fraud and legitimate transactions,
against 3.4% for stage 1. It scores well while being wrong as a customer proxy.

## The three model interventions

All on identical folds, identical seeds, only the intervention varying.

| Variant | Test PR-AUC | Gap | Cost (4 folds) | vs baseline | Folds better PR |
| --- | --- | --- | --- | --- | --- |
| **B D-normalised** | **0.5243** | 0.4104 | **$566,639** | **-$38,140** | **4 of 4** |
| BC D-norm + count | 0.5222 | 0.4076 | $566,519 | -$38,261 | 4 of 4 |
| BCD all three | 0.5208 | 0.3649 | $578,090 | -$26,690 | 4 of 4 |
| BD D-norm + reg | 0.5204 | 0.3620 | $579,211 | -$25,569 | 4 of 4 |
| A baseline | 0.5015 | 0.3704 | $604,779 | - | - |
| CD count + reg | 0.5012 | 0.3178 | $610,239 | +$5,460 | 3 of 4 |
| C count-encoded | 0.5086 | 0.3672 | $612,043 | +$7,264 | 3 of 4 |
| D regularised | 0.4949 | 0.3373 | $622,520 | +$17,740 | 1 of 4 |

### Intervention 1, D-column normalisation: SHIPPED

D1-D15 are "days since a prior event". As raw counters they drift with calendar
time, so a split learned at day 80 does not mean the same thing at day 170.
Replacing each with `txn_day - D` gives the fixed day the event happened, which
is stationary. D9 is excluded: it ranges 0 to 0.958 and is a time-of-day
fraction, not a day offset.

Result: **+0.0228 test PR-AUC, better in 4 folds out of 4, t = 3.58**, and
$38,140 lower cost. This is the most consistent effect measured anywhere in the
project.

A secondary mechanism probably matters as much as stationarity: `txn_day - D1`
is the card's start date, so normalising hands the model raw material to form
its own transaction groupings. That connects to the proxy result. Giving the
model the ingredients worked; handing it a different pre-built key did not.

### Intervention 2, count encoding of high-cardinality categoricals: REJECTED

Frequency encoding for the 18 categoricals with 20 or more levels, fitted on
pre-test rows only. Alone it gained +0.0071 PR-AUC but **cost $7,264 more** and
was cheaper in only 1 fold of 4.

Combined with D-normalisation it is a coin flip: BC is $121 cheaper than B
alone on a $566,639 base, **0.021%**, with slightly worse PR-AUC (-0.0021) and
19% slower training. Not adopted; B alone is simpler and no worse.

### Intervention 3, stronger regularisation: REJECTED, and it taught the most

Shallower trees, larger leaves, heavier subsampling and L2. It did exactly what
it was designed to do: **it closed the gap more than anything else tested**,
from 0.370 to 0.337 alone and to 0.318 combined with count encoding.

It was also **the worst variant**: -0.0066 PR-AUC and **$17,740 more cost**,
cheaper in 0 folds of 4.

Ranking every variant by gap gives almost the reverse of ranking by cost. The
best performer, D-normalisation, has the **widest** gap of any variant at
0.4104. I flagged the 0.39 train/test gap as a problem repeatedly across this
project and recommended attacking it. That was wrong. On this problem the gap
measures how much structure the model can find in training data, not how badly
it generalises, and optimising it directly makes the model worse.

## One honest disagreement

On the four-fold harness, D-normalisation improves PR-AUC in every fold. On the
single production split (train < 110, test >= 150) it makes PR-AUC **worse**,
0.5015 to 0.4933, while still improving cost.

I tested the obvious explanation, that normalisation creates absolute dates
which extrapolate badly at long forecast horizons, by holding folds fixed and
varying only the test length:

| Test window | Raw D PR-AUC | Normalised PR-AUC | Raw cost | Normalised cost |
| --- | --- | --- | --- | --- |
| 16 days | 0.5097 | 0.5349 | $323,476 | $296,016 |
| 32 days | 0.4958 | 0.5142 | $613,638 | $582,466 |

The hypothesis is rejected: normalisation wins at both horizons, and the
advantage only narrows slightly. The production configuration differs from the
harness only in its train and early-stopping boundaries (day 110 against 118),
and that small change flips the sign of the PR-AUC comparison. That is itself
evidence for how unreliable a single split is, which is the argument for the
harness in the first place.

Shipped on the weight of evidence: seven of eight fold-level PR-AUC comparisons
favour normalisation, and cost improves in every configuration tested including
production.

## Net effect on the shipped model

| Metric | Before | After | Change |
| --- | --- | --- | --- |
| Total cost, production split | $296,570 | **$291,334** | **-$5,237** |
| Saving vs the 0.5 cutoff | $144,514 | **$155,129** | +$10,615 |
| Fraud dollars caught | 66.36% | **68.95%** | +2.59pp |
| False decline rate | 6.277% | 6.524% | +0.247pp |
| Test PR-AUC | 0.5015 | 0.4933 | -0.0082 |
| Test ROC-AUC | 0.8839 | 0.8926 | +0.0087 |
| Brier under calibration | degraded 0.06% | **improved 2.00%** | fixed |
| Train/test gap | 0.3949 | 0.4516 | +0.0567 |

The cost improvement of $5,237 on the production split is inside the $9,900
noise band and should not be claimed as proven on its own. The four-fold
evidence behind it, -$38,140 with 4 of 4 folds favourable on PR-AUC, is what
justifies the change.

A side effect worth noting: isotonic calibration now **improves** the Brier
score on test for the first time in this project, 0.024119 to 0.023636. That
problem persisted through two earlier attempts to fix it by changing the split
structure. It was resolved by a feature transform instead.
