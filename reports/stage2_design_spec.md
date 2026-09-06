# Stage 2 design spec: replay experiment and observational comparison

**Status: draft for review. No implementation until sign-off.**

Every figure below was computed from the shipped D-normalised model and the
cached replay data. Nothing is a placeholder or a textbook default.

Supersedes the earlier draft, which was written against the pre-normalisation
model and included sequential testing.

---

## 0. Read this first: five things that need a decision

1. **"Champion = current threshold" leaves no viable challenger.** Against
   0.1160, every candidate policy has 5% to 32% power. The only adequately
   powered contrast is against the naive 0.5 rule. Section 3 proposes
   reframing champion as the pre-model incumbent. Without that, the experiment
   cannot conclude anything.
2. **The requested CUPED covariate is the weakest one available.** Pre-period
   account value gives **0.09%** variance reduction. Pre-period fraud count
   gives 4.6%. Neither is material, and the reason is that only 31.6% of
   replay accounts exist in the pre-period at all.
3. **A non-inferiority guardrail on false declines will fail by construction.**
   The challenger declines 6.52% of legitimate transactions against the
   champion's 0.35%, a deliberate +6.17pp. Non-inferiority is the wrong test
   shape. It needs to be an absolute ceiling.
4. **Two of the four pre-registered subgroups are underpowered** at 15.5% and
   14.5%. They can be registered as descriptive, but not as tests.
5. **The treatment effect is entirely concentrated in the top risk quintile.**
   Quintiles 1 to 3 have an effect of exactly zero. This makes the average
   treatment effect a poor summary and it is the single most important fact
   for the observational design.

---

## 1. Randomisation unit and mechanism

**Unit: `account_id`**, the constructed proxy from `int_account_keys`.

Justification is dependence, not identity: fraud rings share cards and devices,
so transactions within a constructed account are not independent. Measured
ICC of transaction-level cost within account is 0.582, design effect 1.571.
Randomising transactions would understate standard errors by about 25%.

The proxy is imperfect and documented as such. For randomisation that is
tolerable: assignment only needs to be coarser than the dependence structure.
Where it over-splits a true cardholder the test becomes conservative; where it
over-merges we lose precision. Neither biases the estimate.

Replay window structure, days 150-181:

| Property | Value |
| --- | --- |
| Transactions | 94,636 |
| Accounts (clusters) | 47,734 |
| Mean / median / p95 / max cluster size | 1.983 / 1 / 5 / 1,400 |
| ICC of cost within account | 0.582 |

**Mechanism.** Deterministic hash, not a random number generator:
`u = md5(f"{SALT}:{account_id}").int / 2**128`, assign to challenger if
`u < 0.5`. `SALT` is fixed in the pre-registration. This is reproducible,
independent of row order, requires no stored assignment table, and a reviewer
can re-derive every assignment from the account id alone.

---

## 2. Timeline and pre-period

| Window | Days | Purpose |
| --- | --- | --- |
| Model training | < 110 | frozen before anything below |
| Early stopping | 110-129 | frozen |
| Calibration and threshold selection | 130-149 | champion and challenger thresholds fixed here |
| **Pre-period** | **110-149** | CUPED covariate, subgroup definitions, PSM covariates |
| **Replay** | **150-181** | the experiment |

The pre-period deliberately overlaps the model's training tail. That is safe
for covariate construction, since covariates are account descriptions rather
than model outputs, but it means pre-period fraud counts are labels the model
already saw. Registered as a known limitation.

**Pre-period coverage is 31.6%.** Only 15,106 of 47,734 replay accounts have
any activity in days 110-149. This single number is why CUPED underperforms
and why the propensity model in section 8 has so little to work with. It is
the most consequential weakness in the design.

---

## 3. Arms, and how the challenger threshold is chosen

**Thresholds are chosen on the calibration window (days 130-149) and frozen
before assignment.** Neither is optimised on the replay window. This is the
protocol adopted in the improvement pass: sweep on cross-fitted out-of-fold
calibration probabilities, then apply. Re-optimising inside the replay window
would test a policy fitted to its own evaluation data.

### The power problem with "champion = current threshold"

Measured effects against champion 0.1160, on 47,734 accounts with a
per-account cost standard deviation of $66.75 and SE of $0.6110:

| Candidate challenger | Total effect | Per account | Power |
| --- | --- | --- | --- |
| Per-transaction expected-cost rule | +$9,535 | +$0.200 | **5.1%** |
| Threshold 0.080 | +$16,659 | +$0.349 | 8.2% |
| Threshold 0.160 | +$18,930 | +$0.397 | 9.5% |
| Threshold 0.050 | +$35,202 | +$0.738 | 22.6% |
| Threshold 0.250 | +$43,578 | +$0.913 | 32.1% |
| Naive 0.500 | +$155,129 | +$3.250 | **100%** |

Nothing near the current threshold is detectable. Reaching 80% power on the
0.250 contrast would need roughly 168,000 accounts; the entire 182-day dataset
contains 217,850, and only about 90,000 are reachable without retraining.
Extending the window does not rescue these.

### Proposed arms

| Arm | Policy | Threshold | Allocation |
| --- | --- | --- | --- |
| **Champion** | incumbent fixed rule, pre-model | 0.5000 | 50% |
| **Challenger** | model-optimised, cost-swept on days 130-149 | 0.1160 | 50% |

This reframes champion as the **pre-model incumbent** rather than the currently
shipped threshold. The business question becomes "should the fixed rule be
replaced by the model-optimised threshold", which is the decision stage 1
actually recommends, and it is the only version of this experiment that can
reach a conclusion.

I am flagging honestly that the outcome is not in doubt: stage 1 already
measured this contrast at $155,129. The value of stage 2 is the analysis
machinery and the observational comparison, not the answer. If you want a
genuinely uncertain contrast, the honest options are to accept 30% power on the
0.250 challenger, or to abandon the randomised framing and use the paired
counterfactual estimator, which is 2.6x more efficient.

---

## 4. Metric hierarchy

**Primary, one metric, drives the recommendation**

- Mean total modelled cost per account over the replay window, in dollars.
  Per-transaction cost is the stage 1 function: fraud approved costs amount
  plus $25; legitimate declined costs 2.5% of amount plus $10 churn-weighted
  LTV plus $2 review; fraud declined costs $2; legitimate approved costs $0.
  Summed to the account, which is the randomisation unit.

**Secondary, reported with intervals, do not drive the decision**

- Fraud dollars caught and share of fraud dollars present
- Recall, precision
- Declined transaction count and rate
- Mean cost per transaction, as a cross-check on the primary

**Guardrail**

- False decline rate: legitimate transactions declined over legitimate
  transactions.

**The guardrail cannot be a non-inferiority test.** Measured rates are 0.354%
for the champion and 6.524% for the challenger, a deliberate 18-fold increase
that is the mechanism by which the challenger catches more fraud. A
non-inferiority test against the champion fails at any sensible margin, and
would fail on a policy we believe is correct.

The guardrail must be an **absolute ceiling** set by whoever owns the customer
relationship, tested one-sided as `H0: FDR >= ceiling` against
`H1: FDR < ceiling`. Placeholder **8.0%**, which the challenger clears with
measured SE of 0.08pp. **This is a business input and I should not be choosing
it.**

---

## 5. Power calculation

Two arms, equal allocation, unit of analysis equals unit of randomisation, so
`SE(difference) = 2 x sd / sqrt(N)`.

**Assumptions, stated because they drive the answer**

1. N = 47,734 accounts, 50/50, realised split within SRM tolerance.
2. Per-account cost standard deviation of **$128.5**, measured under the
   champion arm. Using the champion is conservative: under the challenger the
   sd is $66.75, because blocking large frauds removes the heavy tail. Power
   computed on the pooled or challenger sd would look better; the champion
   figure is the pessimistic one.
3. Two-sided α = 0.05, no multiplicity adjustment on the primary.
4. Normal approximation for the difference in means.
5. Effect size taken as the measured replay effect of -$3.2499 per account.

**Result**

| Quantity | Value |
| --- | --- |
| SE of the difference | $1.1825 per account |
| MDE at 80% power | $3.3128 per account, **$158,132 total** |
| True effect | -$3.2499 per account, -$155,129 total |
| **Power at the true effect** | **78.5%** |

**The design is marginally underpowered.** The MDE of $158,132 slightly exceeds
the effect of $155,129, so at 80% this contrast is just out of reach and lands
at 78.5%. It will usually reject, but roughly one run in five would not. This
should be stated in the pre-registration rather than discovered afterwards.

**On the heavy tail.** Per-account cost is severely skewed. At n = 23,867 per
arm the sampling distribution of the mean is nonetheless close to normal
(skew of the mean around 0.2), so normal intervals are defensible. A BCa
bootstrap runs alongside and any material disagreement gets reported.

---

## 6. Variance reduction, and why CUPED barely helps

CUPED adjusts `Y* = Y - θ(X - E[X])` with `θ = Cov(Y,X)/Var(X)`. Measured on
the actual data:

| Covariate | Pre-period | Correlation | Variance reduction | MDE after |
| --- | --- | --- | --- | --- |
| **Account value (requested)** | 40 days | **+0.0298** | **0.09%** | $158,061 |
| Transaction count | 40 days | +0.0168 | 0.03% | $158,109 |
| Mean ticket | 40 days | +0.0120 | 0.01% | $158,120 |
| Fraud count | 40 days | +0.2141 | 4.58% | $154,466 |
| Composite, all four | 40 days | +0.2205 | **4.86%** | $154,268 |

**The requested covariate, pre-period account value, is the weakest of the
four.** It moves the MDE by $71 out of $158,132.

Two structural reasons: 68.4% of replay accounts have no pre-period at all, so
the covariate is zero for two thirds of units; and cost is driven by rare large
frauds, which an account's own spending history barely predicts.

**Recommendation:** register the four-covariate composite rather than account
value alone, and state the expected 4.9% reduction in advance so a null cannot
be reframed after the fact. Implement it, report it honestly, and do not
present it as a win.

---

## 7. Pre-registered subgroups

Definitions fixed in advance, computed from the pre-period only.

- **New**: no transactions in days 110-149. **Established**: at least one.
- **Ticket**: account mean transaction amount in the replay window, split at
  the pooled median of **$78.33**. Registered exactly, since a median computed
  post hoc would be a researcher degree of freedom.

Measured effects and power:

| Subgroup | Accounts | Effect per account | Total | Power |
| --- | --- | --- | --- | --- |
| New | 32,628 | -$4.067 | -$132,697 | 73.6% |
| Established | 15,106 | -$1.485 | -$22,432 | **15.5%** |
| Low ticket (< $78.33) | 23,858 | -$0.496 | -$11,830 | **14.5%** |
| High ticket (>= $78.33) | 23,876 | -$6.002 | -$143,299 | 74.3% |

Two of the four are underpowered and cannot support a test. They will be
registered as **descriptive**, reported with intervals, and explicitly excluded
from any claim. With Bonferroni across four contrasts at α = 0.0125 even the
powered two drop below 70%, so the subgroup analysis is registered as
**secondary and exploratory throughout**, with no subgroup permitted to change
the ship decision.

The substantive expectation, registered in advance: **the benefit is
concentrated in new and high-ticket accounts**, and established low-ticket
accounts see nearly nothing.

---

## 8. The observational simulation

The part that matters most, and the measurements above shape it heavily.

### The fact that drives the design

Effect by account risk quintile, where risk is the account's mean predicted
probability in the replay window:

| Risk quintile | Accounts | Mean risk | Mean cost, champion | Mean effect |
| --- | --- | --- | --- | --- |
| 1 | 9,547 | 0.0024 | $0.32 | **$0.00** |
| 2 | 9,547 | 0.0037 | $0.57 | **$0.00** |
| 3 | 9,546 | 0.0065 | $1.35 | **$0.00** |
| 4 | 9,547 | 0.0144 | $3.34 | +$0.04 |
| 5 | 9,547 | 0.1200 | $41.19 | **-$16.29** |

**The entire treatment effect lives in the top quintile, and cost varies 130x
across quintiles.** A confounder that steers assignment by risk therefore moves
both the outcome and the effect simultaneously. This is close to a worst case
for naive observational estimation, which makes it an excellent demonstration.

### Assignment mechanism

Non-random rollout by risk profile:

```
P(challenger | account) = sigmoid( alpha + beta * z(account_risk) )
```

`alpha` is solved numerically so the realised treated share is 50%, keeping the
observational and randomised analyses comparable on sample size. `beta`
controls confounding strength, registered at three levels: **0.5 weak, 1.5
moderate, 3.0 strong**.

Two directions, both registered, because they bias in opposite ways:

- **Risk-seeking rollout** (`beta > 0`): the stricter policy goes to risky
  accounts first. Realistic, and the common instinct in fraud teams.
- **Risk-averse rollout** (`beta < 0`): rollout starts on safe accounts to
  limit blast radius. Equally realistic and equally common.

Expected directions, registered in advance so the result cannot be
rationalised afterwards. Risk-seeking puts treatment where cost is highest, so
the naive comparison should make the challenger look **worse** than it is. The
true effect is negative, so the bias attenuates or reverses the measured
benefit. Risk-averse should make the challenger look **better** than it is on
levels while the true effect in that population is near zero.

### What the analyst is allowed to see

The confounder is the account's replay-window risk. In a real post-hoc analysis
that is not available: the analyst has history, not the scores the rollout team
used. The registered covariate set is therefore:

- Pre-period aggregates: value, transaction count, fraud count, has-pre-period
  indicator
- Account size in the replay window: transaction count, total amount
- First replay transaction attributes: amount, has_identity, product code, card
  type

**How much of the confounder this recovers, measured:**

| Covariate set | R-squared predicting account risk | Residual confounding |
| --- | --- | --- |
| Pre-period only | 0.0218 | 97.8% |
| Pre-period + account size | 0.0274 | 97.3% |
| Pre + size + first transaction | **0.0680** | **93.2%** |

Even the richest realistic set explains 6.8% of the confounder. Propensity
matching cannot fix what it cannot see, so substantial residual bias is
expected. That is the finding, not a flaw in the setup.

### Estimators compared

Four, run against the same simulated rollout:

1. **Naive difference in means.** No adjustment. The upper bound on bias.
2. **PSM on observed covariates.** Logistic propensity, 1:1 nearest neighbour
   with replacement, caliper 0.2 pooled SD of the logit, common-support
   trimming. The realistic estimate.
3. **PSM including the true confounder (oracle).** Identical, plus account
   risk. Should recover the truth. This is the positive control that separates
   "the method is broken" from "the analyst could not observe the confounder",
   and without it the exercise cannot make that distinction.
4. **Randomised benchmark.** The section 3 result. The truth.

### Diagnostics, reported whether or not they flatter the result

- Standardised mean differences before and after matching, all covariates
- Propensity overlap, plotted, with the trimmed share
- Number matched, number discarded, effective sample size
- Bias of each estimator against the randomised truth, in dollars and percent
- Bias decomposition against `beta`, showing how it scales with confounding
- Rosenbaum sensitivity: how large an unmeasured confounder would have to be to
  overturn the PSM conclusion

### Expected conclusion, registered in advance

Naive is badly biased; PSM on realistic covariates removes a small fraction of
that bias because it observes 6.8% of the confounder; oracle PSM recovers the
truth. The lesson is not that PSM is useless but that **PSM is only as good as
the covariates, and in this setting the covariates are nearly empty.** Writing
this down now means the result cannot be presented as a surprise later.

---

## 9. What gets pre-registered

`reports/stage2_preregistration.md`, committed before any outcome is computed,
with its hash recorded so the analysis refuses to run against a modified file.

1. Hypotheses, primary and guardrail, stated directionally
2. Arms, allocation, hash function and `SALT`
3. Replay window, pre-period, and the frozen model's split boundaries
4. Primary metric with the full cost function and its five constants
5. Secondary metrics; subgroups marked secondary and exploratory
6. Guardrail ceiling and its one-sided test
7. Power, MDE, the assumed sd, and the acknowledged 78.5%
8. CUPED covariate chosen in advance, with the expected 4.9% stated up front
9. Exclusions: none planned; any that arise must be arm-blind and logged
10. Observational simulation: assignment model, all three `beta` values, both
    directions, covariate sets, matching parameters
11. Expected directions of bias, written before running
12. Ship decision rule from section 10
13. Deviations log

---

## 10. Ship recommendation

One page, produced from the randomised analysis only. Ship if all three hold:

1. The primary contrast is significant in favour of the challenger at α = 0.05
2. The **upper bound** of the 95% CI on cost difference is below zero
3. The guardrail false decline rate is below the agreed ceiling at its upper
   confidence bound

Do not ship if the guardrail is breached, whatever the cost result. Declare
inconclusive if the primary does not reach significance, which at 78.5% power
happens roughly one run in five and does **not** mean the challenger is
worthless.

Reported as an interval throughout, never a point estimate.

---

## 11. Weaknesses, including ones not raised in the brief

| Weakness | Severity | Handling |
| --- | --- | --- |
| No viable challenger against the current threshold | **blocking** | Section 3 reframes champion as the pre-model incumbent |
| Power 78.5%, MDE exceeds the true effect | high | Stated in the pre-registration, not discovered later |
| Requested CUPED covariate gives 0.09% | high | Register the composite instead; expect 4.9% |
| Pre-period covers only 31.6% of accounts | high | Guts CUPED and PSM alike; no fix available |
| Non-inferiority guardrail fails by construction | high | Replace with an absolute ceiling, needs a business input |
| Two of four subgroups underpowered | medium | Registered as descriptive, cannot drive the decision |
| Effect concentrated in one quintile | medium | ATE is a poor summary; report the quintile profile alongside |
| Account proxy is constructed | medium | Documented; sensitivity on `card1` alone as a coarser unit |
| Pre-period overlaps model training | low | Covariates are account descriptions, but pre-period fraud labels were seen by the model |
| The randomised design discards the paired counterfactual | low | Paired estimator reported as an oracle benchmark and a bug detector |
| Outcome of the primary is not in doubt | disclosure | Stated plainly; the value of stage 2 is the machinery |

Two things worth raising that were not in the brief.

**The ATE is close to meaningless here.** With zero effect in three quintiles
and -$16.29 in the fifth, a single average describes no account in the
population. The recommendation should carry the quintile profile, and the
obvious follow-up is a targeted policy that applies the challenger only where
it acts. That is a stronger result than the ATE and it falls out of the same
analysis.

**The observational simulation is the more valuable half and deserves more
room.** The randomised arm re-derives a number stage 1 already has. The
observational arm answers something genuinely unknown: how wrong would we have
been had this been rolled out non-randomly and analysed post hoc. If effort has
to be traded, trade it toward section 8.
