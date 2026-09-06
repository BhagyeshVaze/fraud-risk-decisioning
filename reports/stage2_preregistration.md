# Stage 2 pre-registration

**Committed before any stage 2 result was computed.** The commit that
introduces this file contains no analysis code and no outputs. Its timestamp
is the registration timestamp. Any change after that commit must appear in the
deviations log in section 13 rather than being edited in silently.

Design rationale and the measurements behind every number here are in
`reports/stage2_design_spec.md`.

---

## 1. What this is, and what it is not

A counterfactual replay on a frozen dataset. Labels are known, so for any
policy the exact cost it would have incurred on real outcomes is computable.
No outcome is simulated or imputed. This is off-policy evaluation, not a live
experiment, and the README will say so.

**Registered disclosure: the direction of the primary result is not in doubt.**
Stage 1 already measured this contrast at $155,129 in favour of the challenger.
The value of stage 2 is the analysis machinery and, principally, the
observational comparison in section 10. This is written down now so that a
confirmatory result cannot later be presented as a discovery.

What replay cannot establish: churn after a decline, fraudster retry behaviour,
support load, or any behavioural response. Those remain assumptions of the cost
model.

---

## 2. Hypotheses

**Primary.** H0: mean per-account cost is equal under both policies.
H1 (two-sided, alpha = 0.05): it differs. Registered expectation: the
challenger is cheaper.

**Guardrail.** H0: challenger false decline rate >= 8.0%.
H1 (one-sided, alpha = 0.05): it is below 8.0%.

---

## 3. Arms, allocation, assignment

| Arm | Policy | Threshold | Allocation |
| --- | --- | --- | --- |
| Champion | pre-model incumbent fixed rule | 0.5000 | 50% |
| Challenger | model-optimised, swept on days 130-149 | 0.1160 | 50% |

Champion is the **pre-model incumbent**, not the currently shipped threshold.
Registered reason: against the shipped 0.1160 every candidate challenger has
5.1% to 32.1% power, so no such contrast can conclude. Full table in the spec.

Both thresholds were fixed on the calibration window (days 130-149) using
cross-fitted out-of-fold calibrated probabilities, before assignment. Neither
is optimised on the replay window.

**Randomisation unit: `account_id`.** Justified by dependence: measured ICC of
transaction-level cost within account is 0.582, design effect 1.571.

**Assignment mechanism, fixed here:**

```
SALT = "stage2-v1"
u    = int(md5(f"{SALT}:{account_id}").hexdigest(), 16) / 2**128
arm  = "challenger" if u < 0.5 else "champion"
```

Deterministic, order-independent, reproducible by any reviewer from the account
id alone. No assignment table is stored.

---

## 4. Windows

| Window | Days | Purpose |
| --- | --- | --- |
| Model training | < 110 | frozen |
| Early stopping | 110-129 | frozen |
| Calibration and threshold selection | 130-149 | frozen |
| Pre-period | 110-149 | CUPED covariates, subgroups, PSM covariates |
| Replay | 150-181 | the experiment |

Replay window: 94,636 transactions, 3,282 frauds, 47,734 accounts.

**Registered limitation.** Pre-period coverage is 31.6%: only 15,106 of 47,734
replay accounts have any activity in days 110-149. This weakens CUPED and the
propensity model, and no fix is available within this dataset.

**Registered limitation.** The pre-period overlaps the model's training tail,
so pre-period fraud counts are labels the model has seen. Covariates are
account descriptions rather than model outputs, but the overlap is recorded.

---

## 5. Primary metric

Mean total modelled cost per account over the replay window, in dollars.
Per-transaction cost, with constants fixed here:

| Outcome | Cost |
| --- | --- |
| Fraud approved | transaction amount + $25.00 |
| Legitimate declined | 0.025 x amount + 0.05 x $200.00 + $2.00 |
| Fraud declined | $2.00 |
| Legitimate approved | $0.00 |

Constants: chargeback fee $25.00, margin rate 0.025, churn probability 0.05,
account lifetime value $200.00, manual review cost $2.00. These are
assumptions, not measurements. Sensitivity is already characterised in
`reports/cost_sensitivity.md`.

Costs are summed to the account, which is the randomisation unit.

**Estimand:** difference in mean per-account cost, challenger minus champion.
Negative favours the challenger.

---

## 6. Secondary metrics

Reported with intervals. None drives the ship decision.

- Fraud dollars caught, and share of fraud dollars present
- Recall, precision
- Declined transaction count and rate
- Mean cost per transaction, as a cross-check on the primary
- Effect by account risk quintile (see section 11)

---

## 7. Guardrail

**False decline rate**: legitimate transactions declined divided by legitimate
transactions.

**Absolute ceiling of 8.0%**, registered now, before any result. Tested
one-sided, `H0: FDR >= 0.080` against `H1: FDR < 0.080`, alpha = 0.05, using
standard errors clustered on `account_id`.

Registered reason for a ceiling rather than non-inferiority: the challenger
declines 6.524% of legitimate transactions against the champion's 0.354%, an
18-fold increase that is the mechanism by which it catches more fraud. A
non-inferiority test against the champion would fail on a policy believed to be
correct. The 8.0% figure is a placeholder for a business input and is recorded
as such.

A guardrail breach vetoes a ship regardless of the cost result.

---

## 8. Power

Two arms, equal allocation, unit of analysis equals unit of randomisation, so
`SE(difference) = 2 x sd / sqrt(N)`.

Assumptions, registered:

1. N = 47,734 accounts, 50/50 within SRM tolerance
2. Per-account cost sd of **$128.50**, measured under the champion arm. This is
   the conservative choice: under the challenger the sd is $66.75, and power
   computed on that would look better
3. Two-sided alpha = 0.05, no multiplicity adjustment on the primary
4. Normal approximation for the difference in means
5. Effect size -$3.2499 per account, the measured replay effect

| Quantity | Value |
| --- | --- |
| SE of the difference | $1.1825 per account |
| MDE at 80% power | $3.3128 per account, **$158,132 total** |
| True effect | -$3.2499 per account, **-$155,129 total** |
| **Power at the true effect** | **78.5%** |

**Registered plainly and not rounded up: this design is marginally
underpowered.** The MDE of $158,132 exceeds the true effect of $155,129.
Roughly one run in five would fail to reject. An inconclusive result therefore
does not mean the challenger is worthless.

---

## 9. Analysis plan

**Primary estimator.** Difference in mean per-account cost. Because the unit of
analysis equals the unit of randomisation, a two-sample comparison is already
cluster-robust and no clustering correction is required.

**Intervals.** Normal-theory 95% CI as primary; BCa bootstrap with 2,000
resamples as a cross-check. Any material disagreement is reported, not
silently resolved.

**Cluster-robust standard errors.** Required for the transaction-level
secondary specifications, where adjustment happens at transaction grain: OLS of
transaction cost on arm plus covariates with **CR2** standard errors clustered
on `account_id` and Satterthwaite degrees of freedom.

**SRM check.** Chi-square goodness of fit on **account counts** against the
intended 50/50. Halting threshold p < 0.001.

Registered nuance: SRM is tested on the randomisation unit only. Transaction
counts will differ between arms by chance because cluster sizes vary; testing
those would produce false alarms. Transaction imbalance is reported
descriptively and never tested.

**CUPED.** Implemented as multivariate regression adjustment (ANCOVA), which
generalises single-covariate CUPED, using the **composite of all four**
pre-period covariates:

- `pre_value`: total transaction amount in days 110-149
- `pre_n`: transaction count in days 110-149
- `pre_fraud`: fraud count in days 110-149
- `pre_mean_amt`: mean transaction amount in days 110-149

Accounts absent from the pre-period take zero on all four, with a
`has_pre_period` indicator included.

**Registered expectation, stated before running: 4.9% variance reduction.**
Measured correlations are +0.0298 for account value alone, +0.2141 for fraud
count, and +0.2205 for the composite. The single covariate originally proposed,
account value, gives 0.09% and is deliberately not used alone. If the realised
reduction is near this expectation, that is a null result for CUPED on this
data and will be reported as one.

**Exclusions.** None planned. Any that arise must be arm-blind and logged in
section 13.

---

## 10. Observational simulation

The principal deliverable.

**Assignment.** A non-random rollout steered by risk profile:

```
P(challenger | account) = sigmoid( alpha + beta * z(account_risk) )
account_risk = mean predicted probability over the account's replay transactions
```

`alpha` is solved numerically so the realised treated share is 50%, keeping
sample sizes comparable to the randomised arm. `z` is standardisation over
accounts.

**Registered scenarios:** `beta` in {0.5, 1.5, 3.0} and {-0.5, -1.5, -3.0}, six
in total. Positive is a risk-seeking rollout (strict policy to risky accounts
first); negative is risk-averse (start on safe accounts to limit blast radius).
Both are realistic and they bias in opposite directions.

**Covariates the analyst may use.** The confounder is replay-window risk, which
a real post-hoc analyst does not have. The registered observed set is:

- `pre_value`, `pre_n`, `pre_fraud`, `has_pre_period`
- Account size in replay: transaction count, total amount
- First replay transaction: amount, has_identity, product code, card type

Measured explanatory power of this set for the confounder: **R-squared 0.0680**,
leaving 93.2% of the confounder unobserved.

**Estimators, all four run on the same simulated rollout:**

1. Naive difference in means, no adjustment
2. PSM on the observed covariates above
3. **Oracle PSM**, identical but including the true confounder `account_risk`
4. The randomised result from sections 3 to 9, as the benchmark truth

Estimator 3 is registered as essential, not optional. Without it the exercise
cannot distinguish "propensity matching is unreliable" from "the analyst could
not observe the confounder", and that distinction is the point.

**Matching specification, fixed here:** logistic propensity model; 1:1 nearest
neighbour with replacement; caliper 0.2 times the pooled standard deviation of
the logit of the propensity score; common-support trimming; ties broken by the
deterministic hash of section 3.

**Diagnostics, reported whether or not they flatter the result:**

- Standardised mean differences for every covariate, before and after matching
- Propensity overlap, plotted, with the trimmed share
- Number matched, number discarded, effective sample size
- Bias of each estimator against the randomised truth, in dollars and percent
- Bias as a function of `beta`
- Rosenbaum sensitivity: the size of unmeasured confounding needed to overturn
  the PSM conclusion

**Registered expected directions, written before running:**

- Risk-seeking (`beta > 0`) places treatment where cost is highest, so the
  naive estimate should make the challenger look **worse** than it is,
  attenuating or reversing the true negative effect.
- Risk-averse (`beta < 0`) should make the challenger look **better** than it
  is on levels, while the true effect in that population is near zero.
- Naive is most biased; PSM on observed covariates removes only a small part of
  that bias because it sees 6.8% of the confounder; oracle PSM recovers the
  truth.
- Bias magnitude increases monotonically with `|beta|`.

If any of these expectations is contradicted, the contradiction is the finding
and will be reported as such.

---

## 11. Reporting the recommendation

The one-page ship recommendation will **lead with the effect by account risk
quintile, not the average treatment effect.** Registered reason: the effect is
$0.00 in quintiles 1 to 3, +$0.04 in quintile 4 and -$16.29 in quintile 5,
while mean cost varies 130-fold across quintiles. A single average describes no
account in the population.

A **targeted-application follow-up** is registered as part of the same
analysis: the cost of applying the challenger only to accounts above a risk
cutoff, compared against applying it to everyone. This falls out of the same
computation and is expected to dominate uniform application.

All results reported as intervals, never as point estimates alone. Any claimed
difference carries a bootstrap confidence interval. Differences smaller than
the measured noise floor will be stated as indistinguishable from noise rather
than reported as improvements.

---

## 12. Ship decision rule

Ship the challenger only if all three hold:

1. The primary contrast is significant at alpha = 0.05 in the challenger's
   favour
2. The **upper bound** of the 95% CI on the cost difference is below zero
3. The guardrail false decline rate is below 8.0% at its upper confidence bound

**Do not ship** if the guardrail is breached, whatever the cost result.

**Declare inconclusive** if the primary does not reach significance. At 78.5%
power this happens roughly one run in five and does not mean the challenger is
worthless. This distinction is registered so it cannot be relitigated later.

Subgroups are **descriptive only** and are excluded from this rule. Two of the
four are underpowered at 15.5% and 14.5%.

---

## 13. Deviations log

Any departure from this document after the registering commit is recorded here
with its reason.

| Date | Deviation | Reason |
| --- | --- | --- |
| 2026-09-06 | Section 10 named the randomised estimate as "the benchmark truth". Bias is instead reported primarily against the **exact ATE** (-$3.2499), with the randomised estimate also shown. | In a replay both potential outcomes are known for every account, so the ATE is computable exactly rather than estimated. The randomised figure is one noisy draw at that quantity and using it as the target would conflate sampling error with estimator bias. Clarification, not a change of target. |
| 2026-09-06 | Added the **true ATT** alongside the true ATE as an evaluation target. | Propensity matching estimates the effect on the treated. The effect is strongly heterogeneous, so under a risk-steered rollout the ATT differs from the ATE by construction. Scoring an ATT estimator against the ATE would charge estimand mismatch to confounding. Not anticipated in the registration. |
| 2026-09-06 | Rosenbaum sensitivity reported as **inapplicable** rather than as a bound. | 83-95% of matched pairs have an outcome difference of exactly zero and the top 1% of differences carries 55-79% of the mass. A signed-rank statistic counts pairs while the effect lives in the size of a few. Shown to be inapplicable rather than dropped. |
| 2026-09-06 | Three code defects found and fixed during the randomised run: a factor-of-sqrt(m) error in the clustered ratio SE, a truncated BCa jackknife, and logic that would have declared the bootstrap interval authoritative on disagreement. | Bugs, not design changes. The third would have violated the registration and is noted because it was caught before it affected a reported result. |
