# Stage 2 design spec: counterfactual replay experiment

**Status: draft for review. No implementation until this is signed off.**

Every number in this document was computed from
`data/parquet/fct_transactions.parquet` and the stage 1 calibrated
probabilities. Nothing here is a placeholder.

---

## 0. Three things I need decided before I build

The design as briefed has a power problem, and two of the requested techniques
do not earn their place on this data. Details are in sections 6 and 7; the
decisions are:

1. **The replay window is too short.** On days 150-181 the headline contrast
   has **69.9% power** and an MDE of $163,114 against a true effect of
   $144,514. The fix is to retrain with earlier split boundaries so the replay
   window starts at day 130, which gives 84.4% power. That costs a retrain and
   makes stage 2 numbers non-comparable to stage 1. My recommendation is to do
   it.
2. **CUPED buys almost nothing here (4.3% variance reduction).** I would still
   implement it, because a null result honestly reported is a legitimate
   finding, but it should not be presented as a variance-reduction win.
3. **The per-transaction rule cannot be an evaluated arm.** That contrast has
   **6.6% power**. Including it as a confirmatory arm would be
   pre-registering a coin flip. It should be exploratory only, or dropped.

---

## 1. What this is

A counterfactual replay on a frozen dataset. Labels are known, so for any
policy we can compute exactly what it would have cost on real outcomes. No
outcomes are simulated or imputed. This is off-policy evaluation, not a live
experiment, and the README will say so.

What replay **can** establish: whether the cost difference between policies is
distinguishable from sampling noise, and how large it is with an interval.

What replay **cannot** establish: churn behaviour after a decline, fraudster
retry behaviour, customer support load, or any response to being declined.
Those are assumed by the cost model, not measured. The $10 per false decline
churn term remains an assumption in stage 2 exactly as it was in stage 1.

---

## 2. Unit of randomization

**`account_id`**, the constructed proxy from `int_account_keys`.

Rationale, as briefed: fraud rings share cards and devices, so a transaction
belonging to a ring is not independent of its siblings. Transaction-level
assignment would place related transactions in different arms, letting
treatment leak across the network and violating SUTVA.

The proxy is imperfect. That does not undermine its use as a randomization
unit: assignment only needs to be *coarser than* the dependence structure, not
correct. Where the proxy over-splits a real cardholder, we get a conservative
test (some correlated units land in different arms). Where it over-merges, we
lose a little precision. Both are documented.

Measured cluster structure on days 150-181:

| Property | Value |
| --- | --- |
| Transactions | 94,636 |
| Clusters (accounts) | 47,734 |
| Mean cluster size | 1.983 |
| Median cluster size | 1 |
| p95 / max cluster size | 5 / 1,400 |
| Singleton clusters | 30,081 (63.0%) |
| Transactions in multi-transaction clusters | 68.2% |
| **ICC of transaction-level cost within account** | **0.582** |
| **Design effect** `1 + (m0-1) x ICC` | **1.571** |

The ICC of 0.582 is high and vindicates clustering: transactions within a
constructed account are strongly correlated in cost. Transaction-level
randomization would understate standard errors by roughly the square root of
the design effect, about 25%.

**Assignment mechanism.** Deterministic and auditable:
`u = md5(f"{SALT}:{account_id}").int / 2**128`, arm by fixed cutpoints on `u`.
`SALT` is fixed in the pre-registration. Deterministic hashing means assignment
is reproducible, independent of row order, and re-derivable by a reviewer
without stored state.

---

## 3. Arms

| Arm | Policy | Allocation |
| --- | --- | --- |
| **A. Control** | Fixed threshold 0.50, the naive incumbent | 50% |
| **B. Treatment** | Cost-optimal global threshold from stage 1 | 50% |

Two arms, not three. The stage 1 threshold is refit on the training data only
and then frozen before assignment; it is not re-optimized inside the replay
window, which would be optimizing on the evaluation set.

**Excluded as a confirmatory arm: the per-transaction expected-cost rule.**
Its contrast against B has a true effect of $11,199 against an MDE of $84,759,
giving **6.6% power**. Pre-registering it would guarantee an inconclusive
result. It will be reported as a pre-specified *exploratory* comparison with
its interval, explicitly labelled underpowered and not part of the ship
decision.

---

## 4. Timeline and replay window

Sequential analysis with five looks at equal information fractions.

| Look | Replay day | Information fraction |
| --- | --- | --- |
| 1 | 156 | 0.20 |
| 2 | 163 | 0.40 |
| 3 | 169 | 0.60 |
| 4 | 175 | 0.80 |
| 5 (final) | 181 | 1.00 |

A replay has no real cost to waiting, so interim looks are methodological
practice rather than operational necessity. The spec states this rather than
pretending the looks buy speed.

**Window options.** The replay must start after the model's calibration window
or the arms are evaluated on data the model saw.

| Replay window | Days | Transactions | Accounts | MDE (total) | Power |
| --- | --- | --- | --- | --- | --- |
| day >= 150 (as briefed) | 32 | 94,636 | 47,734 | $163,113 | **69.9%** |
| day >= 140 | 42 | 121,163 | 58,291 | $180,250 | 78.3% |
| **day >= 130 (recommended)** | 52 | 148,174 | 68,399 | $195,254 | **84.4%** |
| day >= 120 | 62 | 179,939 | 79,519 | $210,528 | 89.3% |

Windows before day 150 require retraining with earlier train, early-stopping
and calibration boundaries. Recommendation is **day >= 130**: train `< 90`,
early stopping `90-109`, calibration `110-129`, replay `130-181`. That clears
80% power with margin and keeps 90 days of training data.

---

## 5. Metric hierarchy

**Primary (one, pre-registered, drives the ship decision)**

- **Mean total modelled cost per account**, in dollars, over the replay window.
  Per-transaction cost is defined exactly as in stage 1: fraud approved costs
  amount + $25; legitimate declined costs 2.5% of amount + $10 churn-weighted
  LTV + $2 review; fraud declined costs $2; legitimate approved costs $0.
  Summed to the account, which is the randomization unit.

**Secondary (reported with intervals, do not drive the decision)**

- Fraud dollars caught, and as a share of fraud dollars present
- Recall and precision at the arm's operating threshold
- Count and rate of declined transactions
- Mean cost per transaction (sanity cross-check against the primary)

**Guardrail (pre-registered, can veto a ship)**

- **False decline rate**, meaning legitimate transactions declined divided by
  legitimate transactions.

Important: the treatment **raises** false declines roughly 14.6-fold by design,
from 0.429% to 6.277%. A "no worse than control" guardrail would fail on
purpose and is meaningless here. The guardrail must be an **absolute ceiling
agreed in advance by whoever owns the customer relationship**, tested one-sided
against the ceiling, not against control.

Placeholder pending that decision: ceiling of **8.0%**, tested as
`H0: FDR >= 8.0%` versus `H1: FDR < 8.0%` at α = 0.05 one-sided. Measured
precision on this window is tight (SE 0.080pp), so the guardrail is
well-powered whatever ceiling is chosen. **This number is a business input, not
a statistical one, and I should not be the one to pick it.**

---

## 6. Power and MDE

Two-arm cluster-randomized, equal allocation, unit of analysis = account.
`SE(difference in means) = 2 x sd / sqrt(N)`.

Measured per-account cost under control: mean $9.240, **sd $133.242**, max
$11,270, skew 33.3, kurtosis 1,727.

| Contrast | True effect / account | SE | MDE @80% | Power |
| --- | --- | --- | --- | --- |
| **B vs A (primary)** | -$3.0275 | $1.2197 | $3.4171 | **69.9%** |
| Per-txn rule vs B (exploratory) | -$0.2346 | $0.6338 | $1.7757 | **6.6%** |

At the briefed window the primary contrast is underpowered: the MDE of
$163,114 in total cost exceeds the effect we already know is present
($144,514). The experiment would fail to reject roughly 30% of the time on an
effect that is real. Section 4 gives the fix.

**On the heavy tail and the CLT.** Per-account cost has skew 33.3 and kurtosis
1,727, which looks disqualifying for a normal-approximation interval. It is
not, at this sample size: the sampling distribution of the mean has skew
`33.3/sqrt(23,867) = 0.21` and excess kurtosis `1,727/23,867 = 0.07`. Both are
mild, so normal-theory intervals are defensible. A BCa bootstrap will be run
alongside as a check, and any material disagreement between the two will be
reported rather than silently resolved.

---

## 7. Variance reduction

**CUPED.** Covariate = account activity in the 20 days before the replay
window. Adjusted outcome `Y* = Y - θ(X - E[X])` with `θ = Cov(Y,X)/Var(X)`.

Measured on the current window:

| Covariate | Coverage | corr with outcome | Variance reduction |
| --- | --- | --- | --- |
| pre-period transaction count | 22.4% | +0.024 | 0.06% |
| pre-period spend | 22.4% | +0.034 | 0.12% |
| **pre-period fraud count** | 22.4% | **+0.208** | **4.33%** |

Best case lifts power from 69.9% to 71.8%. **CUPED does not work well here**,
for two structural reasons: only 22.4% of replay-window accounts appear in the
pre-period at all, so the covariate is zero for three quarters of units; and
cost is driven by rare large frauds, which are close to unpredictable from an
account's own history.

I propose implementing it anyway and reporting the measured reduction
honestly. A pre-registered variance-reduction technique that does not reduce
variance is a real finding about this data, and burying it would be worse than
reporting it.

**Winsorization is rejected.** Capping the outcome at p99.9 ($1,919) cuts sd
from $133.24 to $88.22 and lifts power to 96.3%. It also changes the estimand:
the large frauds it truncates are precisely the dollars the system exists to
prevent. It will be reported as a pre-specified robustness check only, never as
the primary.

**The paired benchmark.** Because this is a replay, every account's outcome is
known under *both* policies. A within-account paired estimator is available for
free and is far more efficient:

| Estimator | SE / account | t | MDE (total) |
| --- | --- | --- | --- |
| Randomized, between-account | $1.2197 | -2.5 | $163,113 |
| **Paired, within-account** | **$0.4760** | **-6.4** | **$63,658** |

The randomized design is **2.6x less efficient** and deliberately discards
information a live experiment could never have had. That is the correct choice
if the goal is to rehearse the analysis a live test would require, which it is.
But the spec should be honest that it is a rehearsal cost, so the paired
estimate will be reported alongside as an oracle benchmark. If the randomized
and paired estimates disagree materially, that indicates a randomization or
analysis bug, which makes it a useful diagnostic as well as an honest
disclosure.

---

## 8. Analysis plan

**Estimator.** Difference in mean cost per account, B minus A. Because the unit
of analysis equals the unit of randomization, a two-sample comparison is
already cluster-robust; no clustering correction is needed for the primary.

**Cluster-robust standard errors.** Required for the secondary transaction-level
specifications, where covariate adjustment happens at transaction grain. OLS of
transaction cost on arm plus covariates, **CR2 (bias-reduced) standard errors
clustered on `account_id`**, Satterthwaite degrees of freedom. With ~47,700
clusters the small-sample correction is immaterial, but CR2 costs nothing and
removes an objection. A wild cluster bootstrap will be reported for the primary
contrast as a robustness check.

**SRM check.** Chi-square goodness of fit on **account counts** against the
intended 50/50, halting threshold p < 0.001, run at every look.

A nuance worth stating: SRM must be tested on the **randomization unit**, not on
transactions. Transaction counts will differ between arms by chance because
cluster sizes vary, and testing those would produce false alarms. Transaction
imbalance will be reported descriptively and interpreted against the expected
spread from the cluster size distribution, not tested.

**Sequential testing.** Lan-DeMets alpha spending with an O'Brien-Fleming
boundary, two-sided α = 0.05 over five looks. Approximate nominal thresholds:

| Look | Information | Approx. nominal two-sided α |
| --- | --- | --- |
| 1 | 0.20 | 0.000005 |
| 2 | 0.40 | 0.0013 |
| 3 | 0.60 | 0.0084 |
| 4 | 0.80 | 0.0225 |
| 5 | 1.00 | 0.0413 |

Exact boundaries computed by the spending function at implementation. O'Brien-
Fleming is chosen over Pocock because it spends almost nothing early and
preserves nearly the full α for the final look, which matters when the design
is already near the power margin. If a boundary is crossed at an interim look,
the reported effect will be **bias-adjusted**, since naive estimates at early
stopping are inflated.

**Guardrail test.** One-sided test of the false decline rate against the agreed
ceiling, evaluated at every look. Guardrail breach vetoes a ship regardless of
the primary result.

---

## 9. What gets pre-registered

Written to `reports/stage2_preregistration.md`, committed, and **hashed into
the commit log before any outcome is computed**. The commit hash is the
timestamp; the analysis code will refuse to run unless the pre-registration
file matches its recorded hash.

Contents:

1. Hypotheses, primary and guardrail, stated directionally
2. Arms, allocation, assignment hash function and `SALT`
3. Replay window and the exact split boundaries used to train the frozen model
4. Primary metric definition including the full cost function and its constants
5. Secondary and exploratory metrics, labelled as such, with the per-transaction
   rule explicitly marked underpowered
6. Guardrail ceiling and its one-sided test
7. Power calculation, MDE, and the assumed sd, as recorded above
8. Alpha spending function, number and timing of looks, boundary type
9. CUPED covariate, chosen in advance, with the expected reduction stated up
   front so a null result cannot be reframed afterwards
10. Exclusion rules: none planned. If any arise they must be arm-blind and
    documented as a deviation
11. The ship decision rule in section 10, stated before any outcome is seen
12. A deviations log, appended to if anything changes after registration

---

## 10. Ship decision rule

Pre-registered, evaluated at the final look or at an interim boundary crossing:

**Ship the treatment if all three hold**

1. The primary contrast crosses its alpha-spending boundary in favour of the
   treatment
2. The **upper bound** of the 95% CI on cost difference is below zero, that is
   the treatment is cheaper even in the pessimistic tail
3. The guardrail false decline rate is below the agreed ceiling at its upper
   confidence bound

**Do not ship if** the guardrail is breached, regardless of cost.

**Declare inconclusive if** the primary fails to cross its boundary. Given the
power analysis, an inconclusive result is a live possibility at the briefed
window and does **not** mean the treatment is worthless; it means this design
could not resolve it. That distinction is pre-registered so it cannot be
relitigated after the fact.

The recommendation will be reported as an interval, for example "ships at an
estimated saving of $X per account, 95% CI [$L, $U]", never as a point
estimate.

---

## 11. Threats to validity

| Threat | Handling |
| --- | --- |
| Account proxy is constructed, may over- or under-merge | Documented. Over-splitting is conservative; over-merging costs precision. Sensitivity: rerun assignment on `card1` alone as a coarser unit. |
| Only 22.4% of replay accounts have pre-period data | CUPED weak, reported honestly. Not compensated for elsewhere. |
| Heavy-tailed outcome | CLT verified adequate at this n (section 6). BCa bootstrap as cross-check. Winsorized robustness check reported separately. |
| Threshold was chosen on the stage 1 test split | The replay window must not overlap it. This is the main argument for moving the window to day >= 130 and retraining. |
| No behavioural response in a replay | Stated as a hard limitation. Churn, retry, and support load remain assumptions. |
| Multiple looks inflate type I error | Alpha spending, with bias adjustment on early stopping. |
| Cost constants are unvalidated | The entire result is conditional on them. A sensitivity surface over churn probability and LTV will accompany the ship recommendation. |

---

## 12. Open decisions for review

1. **Move the replay window to day >= 130 and retrain?** My recommendation is
   yes. Without it the primary contrast runs at 69.9% power.
2. **What is the false decline ceiling?** A business input. The 8.0%
   placeholder is mine and should be replaced.
3. **Keep CUPED given a measured 4.3% reduction?** My recommendation is keep and
   report the null honestly.
4. **Per-transaction rule: exploratory arm or drop entirely?** My
   recommendation is report as exploratory, excluded from the ship decision.
5. **Two arms or add a third?** A third arm splits allocation and costs power
   on the primary. My recommendation is two.
