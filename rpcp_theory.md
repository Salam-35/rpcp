# Paper A Revised Plan: Robust Concept Learning from Noisy Class-Level Priors

## Working Title

**Reliability-Weighted Concept Learning from Noisy Class-Level Priors**

## One-Sentence Thesis

Class-level concept priors are useful weak supervision, but they are often noisy; this work studies when prior-guided concept learning fails under corrupted priors and proposes a reliability-weighted extension that detects and downweights unreliable prior entries.

## Central Research Question

Given image-label pairs `(x, y)` and a table of class-level concept priors

```text
Pi_tilde[m, y] ~= P(concept_m present | class y),
```

can we train an image-level concept predictor without using per-image concept labels during training, while also estimating which class-concept prior entries are reliable?

The revised paper does **not** claim that class-level priors alone fully identify true per-image concepts. Instead, it makes a narrower and testable claim:

> Reliability weighting improves robustness when class-level priors are noisy, and learned reliability scores can recover known corrupted prior entries under controlled and clinically motivated corruption settings.

---

# 1. Motivation and Problem Framing

## 1.1 Background

Concept Bottleneck Models (CBMs) normally require per-image concept annotations. In medical imaging, these annotations are expensive because they require expert review.

Prior-guided Concept Predictor (PCP) reduces this burden by using only class-level concept priors. Instead of labeling every image with concepts, a domain expert may provide statements like:

```text
P(irregular_pigment_network | melanoma) = 0.85
P(irregular_pigment_network | nevus)    = 0.20
```

PCP-style training then encourages average predicted concepts within each class to match these class-level priors.

## 1.2 Gap

PCP assumes the prior table is sufficiently correct. This is a strong assumption.

In real medical settings, priors may be noisy because they can come from:

- a single expert consultation,
- small datasets,
- literature summaries with different patient populations,
- LLM-generated clinical descriptions,
- ambiguous or subjective concepts,
- rare classes with weak evidence,
- inter-rater disagreement.

If a prior is wrong, PCP has no mechanism to know that it should trust the prior less.

## 1.3 Revised Contribution

This paper contributes:

1. **Problem formalization:** noisy class-level priors for weakly supervised concept learning.
2. **Method:** a reliability-weighted PCP objective that downweights suspicious prior entries.
3. **Controlled benchmark:** synthetic prior-corruption tests where the true corrupted entries are known.
4. **Reliability audit:** direct evaluation of learned reliability against corruption masks, concept F1, and multi-rater disagreement.
5. **Identifiability analysis:** a clear statement of when the problem cannot be solved from class-level priors alone.

---

# 2. Key Claim Boundaries

## 2.1 What This Paper Claims

This paper claims:

- R-PCP is more robust than PCP when prior entries are corrupted.
- Reliability scores can identify many corrupted prior entries under controlled corruption.
- Prior separation affects recoverability: if two classes have nearly identical prior signatures, concept learning becomes weakly identifiable.
- A small amount of external reliability evidence, such as multi-source priors or a held-out concept audit set, makes reliability estimation substantially more defensible.

## 2.2 What This Paper Does Not Claim

This paper does **not** claim:

- class-level priors uniquely identify per-image concepts;
- model-prior agreement alone proves a prior is correct;
- R-PCP solves all reasoning shortcuts;
- concept predictions are clinically valid without external audit.

These limitations should be stated explicitly in the abstract, method, and discussion.

---

# 3. Mathematical Setup

## 3.1 Data

Training data:

```text
D_train = {(x_i, y_i)} for i = 1,...,N
```

where:

- `x_i` is an image,
- `y_i in {1,...,K}` is the class label,
- no per-image concept labels are used for main training.

Concept set:

```text
C = {c_1, ..., c_M}
```

Prior table:

```text
Pi_tilde in [0,1]^(M x K)
Pi_tilde[m, y] ~= P(c_m = 1 | y)
```

Optional evaluation-only concept labels:

```text
c_i in {0,1}^M
```

These are hidden during training and used only to evaluate concept quality.

## 3.2 Concept Predictor

The model predicts:

```text
c_hat_theta(x) = sigmoid(g_theta(x)) in [0,1]^M
```

For class `y`, the model-implied class-level concept mean is:

```text
Pi_hat_theta[m, y] = E_x|y [c_hat_theta,m(x)]
```

In practice, this is estimated with batch or epoch-level class means.

## 3.3 Class-Level Moment Constraint

If priors were clean, a natural weak-supervision constraint would be:

```text
E_x|y [c_hat_m(x)] ~= Pi_tilde[m, y]
```

This is a population moment constraint. It is connected to posterior regularization and learning from label proportions.

## 3.4 Noisy Prior Model

We model the observed prior as a noisy version of an unknown clean prior:

```text
Pi_tilde[m, y] = s[m, y] * Pi_star[m, y] + (1 - s[m, y]) * epsilon[m, y]
```

where:

- `Pi_star[m, y]` is the true class-level concept prevalence,
- `epsilon[m, y]` is a corrupted or uninformative value,
- `s[m, y] in {0,1}` indicates whether the prior entry is reliable.

Since `s` is unknown in real settings, R-PCP estimates a soft reliability score:

```text
r[m, y] in [0,1]
```

Interpretation:

- `r[m, y] ~= 1`: trust this prior entry strongly.
- `r[m, y] ~= 0`: downweight this prior entry.

## 3.5 Identifiability Limitation

Class-level priors constrain only averages. They do not uniquely determine image-level concept labels.

For any function `h_m(x)` satisfying:

```text
E_x|y [h_m(x)] = 0 for all y,
```

the predictor

```text
c_hat'_m(x) = c_hat_m(x) + h_m(x)
```

can preserve the same class-level moment while changing per-image concept predictions.

Therefore, per-image concepts are not identifiable from class-level priors alone. Recoverability depends on:

- visual separability of the concept,
- prior separation across classes,
- architecture bias,
- auxiliary losses,
- external evidence or audit signals.

This limitation is central to the paper, not a footnote.

---

# 4. Method: Reliability-Weighted PCP

## 4.1 Base PCP Losses

The base model follows PCP-style training:

```text
L_base = L_cls + lambda_match * L_match + lambda_kl * L_KL + lambda_ent * L_ent
```

where:

- `L_cls`: class prediction loss,
- `L_match`: aligns predicted concepts with class priors for class discrimination,
- `L_KL`: aligns class-level predicted concept means with priors,
- `L_ent`: encourages sharper attention or concept selectivity.

The revised method changes the prior-matching term.

## 4.2 Reliability-Weighted Prior Loss

For each class `y`, compute:

```text
p_bar[m, y] = mean_{i: y_i = y} c_hat_m(x_i)
```

Use a reliability-weighted Bernoulli divergence:

```text
L_prior =
  sum_y sum_m r[m, y] * D_Bernoulli(Pi_tilde[m, y] || p_bar[m, y])
```

with:

```text
D_Bernoulli(a || b)
  = a log(a / b) + (1 - a) log((1 - a) / (1 - b))
```

If reproducing PCP exactly requires the original grouped one-sided KL, include both versions:

- `R-PCP-PCPKL`: reliability-weighted original PCP KL,
- `R-PCP-BernKL`: reliability-weighted full Bernoulli KL.

This avoids overclaiming that the original PCP loss is a complete Bernoulli KL if it is not.

## 4.3 Reliability Estimation

The original plan used only model-prior agreement:

```text
if p_bar[m, y] ~= Pi_tilde[m, y], then r[m, y] is high
```

That is risky because the model is trained to agree with the prior. The revised plan uses reliability evidence from at least one of the following settings.

### Evidence Mode A: Controlled Corruption

This is the primary publishable benchmark.

1. Compute clean priors from datasets that have concept annotations.
2. Corrupt known prior entries.
3. Train without using per-image concept labels.
4. Evaluate whether learned `r[m, y]` recovers the known corruption mask.

This directly tests reliability detection.

### Evidence Mode B: Multi-Source Priors

Collect several prior tables:

```text
Pi^(1), Pi^(2), ..., Pi^(S)
```

Sources may include:

- expert 1,
- expert 2,
- literature-derived priors,
- dataset-level estimates,
- LLM-generated priors.

Estimate prior-source disagreement:

```text
u[m, y] = variance_s Pi^(s)[m, y]
```

Initialize reliability lower when sources disagree:

```text
r_0[m, y] = exp(-alpha * u[m, y])
```

Then update reliability during training using both:

- source agreement,
- held-out model consistency,
- optional audit evidence.

### Evidence Mode C: Small Held-Out Concept Audit

Use a small labeled audit set, for example 5% to 10% of training images.

Important rule:

The audit set is not used to train the concept predictor directly. It is used only to calibrate or evaluate reliability.

For each `(m, y)`, compute audit prevalence:

```text
Pi_audit[m, y] = mean_{i in audit, y_i = y} c_i[m]
```

Then estimate reliability by comparing priors to audit prevalence:

```text
r_audit[m, y] = exp(-beta * abs(Pi_tilde[m, y] - Pi_audit[m, y]))
```

This creates a stronger clinical argument: a tiny annotation budget can audit many prior entries.

### Evidence Mode D: Multi-Rater Reliability

For LIDC-IDRI or similar datasets with multiple radiologist annotations, define reliability from inter-rater agreement.

Example:

```text
r_true[m, y] = agreement among raters for concept m in class y
```

Then test whether learned reliability correlates with this natural reliability signal.

## 4.4 Practical Reliability Score

Use a combined reliability score:

```text
r[m, y] = sigmoid(
    w0
  + w1 * agreement_score[m, y]
  - w2 * source_disagreement[m, y]
  - w3 * instability_score[m, y]
  - w4 * prior_model_residual[m, y]
)
```

where:

- `agreement_score`: optional multi-source or audit agreement,
- `source_disagreement`: variance across prior sources,
- `instability_score`: variance of `p_bar[m, y]` across seeds, augmentations, or folds,
- `prior_model_residual`: disagreement between held-out model means and prior.

If only controlled corruption is available, train with the unsupervised score and evaluate it against the known corruption mask.

Important wording:

Model-prior residual is an evidence term, not proof of reliability.

## 4.5 Cross-Fitting to Reduce Self-Confirmation

To reduce circularity:

1. Split the training set into folds `A` and `B`.
2. Train concept predictor on fold `A`.
3. Estimate class-level means on held-out fold `B`.
4. Update reliability using held-out fold `B`.
5. Swap folds and average reliability.

This does not fully solve identifiability, but it reduces direct training-set self-confirmation.

## 4.6 Final Objective

```text
L_total =
    L_cls
  + lambda_match * L_match
  + lambda_prior * L_prior
  + lambda_ent * L_ent
  + lambda_r * R(r)
```

where:

```text
L_prior =
  sum_y sum_m r[m, y] * D_Bernoulli(Pi_tilde[m, y] || p_bar[m, y])
```

and `R(r)` prevents trivial collapse:

```text
R(r) = - sum_m,y log Beta(r[m, y]; a0, b0)
```

Use different Beta priors for different experiments:

- `Beta(2,2)`: discourages extreme reliability without evidence.
- `Beta(5,2)`: optimistic trust in expert priors.
- `Beta(1,5)`: pessimistic setting for LLM-generated priors.

---

# 5. Algorithm

## 5.1 Training Schedule

### Phase 0: Baseline Warmup

Epochs `0` to `T_warmup`.

- Train PCP-style model with `r = 1`.
- Save model checkpoint.
- Compute initial class-level means.

### Phase 1: Reliability Estimation

Epochs `T_warmup` to `T_mid`.

- Estimate reliability every `E_freq` epochs.
- Use cross-fitted held-out means where possible.
- Smooth updates with exponential moving average:

```text
r_t = gamma * r_{t-1} + (1 - gamma) * r_new
```

### Phase 2: Reliability-Weighted Training

Epochs `T_mid` to `T_final`.

- Train with reliability-weighted prior loss.
- Continue reliability updates at lower frequency.
- Track reliability collapse and oscillation.

### Phase 3: Final Evaluation

- Freeze final model.
- Evaluate concept prediction on hidden concept labels.
- Evaluate class prediction.
- Evaluate reliability against corruption masks or audit signals.

## 5.2 Pseudocode

```python
for epoch in range(num_epochs):
    train_model_one_epoch(model, train_loader, reliability=r)

    if epoch >= warmup_epochs and epoch % em_freq == 0:
        p_bar_heldout = estimate_class_means_crossfit(model, train_loader)
        instability = estimate_seed_or_aug_instability(models_or_augments)
        r_new = reliability_update(
            priors=Pi_tilde,
            model_means=p_bar_heldout,
            source_disagreement=source_disagreement,
            instability=instability,
            audit_signal=optional_audit_signal,
        )
        r = ema(r, r_new)
```

---

# 6. Experimental Design

## 6.1 Datasets

Use datasets with concept annotations available for evaluation:

1. **PH2**
   - Dermoscopy.
   - Small binary dataset.
   - Useful for fast controlled experiments.

2. **WBCatt**
   - Hematology.
   - Larger dataset.
   - Better for stable concept-F1 and reliability analysis.

3. **Derm7pt**
   - Dermoscopy with seven-point checklist attributes.
   - Good external validation dataset.

4. **LIDC-IDRI**
   - Lung nodules.
   - Multi-radiologist attribute annotations.
   - Best for natural disagreement/reliability analysis.

## 6.2 Prior Construction

For datasets with concept labels, compute clean priors:

```text
Pi_true[m, y] = mean_{i: y_i = y} c_i[m]
```

For realistic priors, create:

- `Pi_expert`: manually specified or literature-derived priors,
- `Pi_llm`: generated priors,
- `Pi_dataset`: priors computed from training concept labels but used only to create benchmark priors, not per-image supervision.

## 6.3 Controlled Corruption Protocol

Create noisy priors:

```text
Pi_noisy[m, y] =
  (1 - alpha) * Pi_true[m, y] + alpha * noise[m, y]
```

Noise types:

1. **Uniform noise**

```text
noise[m, y] ~ Uniform(0, 1)
```

2. **Background collapse**

```text
noise[m, y] = mean_y Pi_true[m, y]
```

3. **Class swap**

```text
Pi_noisy[m, y] = Pi_true[m, y']
```

4. **Adversarial flip**

```text
Pi_noisy[m, y] = 1 - Pi_true[m, y]
```

5. **LLM-style moderate bias**

Move priors toward generic clinical expectations rather than dataset-specific prevalence.

Keep a known corruption mask:

```text
s_true[m, y] = 1 if clean
s_true[m, y] = 0 if corrupted
```

This mask is not used by R-PCP during training. It is used only for evaluation.

## 6.4 Prior-Separation Experiment

Measure prior separation:

```text
Delta = min_{y != y'} ||Pi[:, y] - Pi[:, y']||_2
```

Construct a sweep:

```text
Pi_alpha[:, y2] = (1 - alpha) * Pi_true[:, y2] + alpha * Pi_true[:, y1]
```

As `alpha -> 1`, the class prior signatures become indistinguishable.

Expected result:

- concept F1 degrades as `Delta` decreases,
- class prediction may remain high if the model uses non-concept shortcuts,
- reliability becomes harder to estimate when priors are not separable.

## 6.5 Main Baselines

Compare:

1. **Black-box classifier**
   - Class labels only.
   - No concept interpretability.

2. **Supervised CBM**
   - Uses per-image concept labels.
   - Upper bound for concept quality.

3. **PCP**
   - Class-level priors.
   - No reliability weighting.

4. **R-PCP unsupervised**
   - Reliability estimated without concept audit labels.

5. **R-PCP + multi-source priors**
   - Reliability initialized from source agreement.

6. **R-PCP + small audit**
   - Uses 5% or 10% concept audit labels only for reliability calibration.

7. **Oracle reliability**
   - Uses true corruption mask.
   - Upper bound for reliability weighting.

The most important comparison is:

```text
PCP with noisy priors
vs R-PCP with noisy priors
vs oracle-reliability PCP
```

## 6.6 Metrics

### Concept Prediction

- macro concept F1,
- per-concept F1,
- concept AUROC,
- concept calibration error.

### Classification

- macro class F1,
- accuracy,
- AUROC where applicable.

### Reliability

- AUROC for detecting corrupted prior entries,
- AUPRC for corrupted-prior detection,
- Spearman correlation between `r[m, y]` and true prior correctness,
- Spearman correlation between `r[m, y]` and per-concept/per-class F1,
- calibration of reliability scores.

### Robustness

Plot performance against:

- corruption strength `alpha`,
- corruption fraction,
- prior separation `Delta`,
- audit-label budget.

---

# 7. Headline Figures

## Figure 1: Method Overview

Show:

```text
image -> backbone -> concept predictor -> class prediction
                         |
                         v
             class-level concept means
                         |
prior table -> reliability module -> weighted prior loss
```

Emphasize that prior entries are weighted individually.

## Figure 2: Prior Corruption Robustness

X-axis:

```text
corruption strength alpha
```

Y-axis:

```text
macro concept F1
```

Curves:

- PCP,
- R-PCP,
- R-PCP + audit,
- oracle reliability.

Expected result:

R-PCP should degrade less than PCP and approach oracle when reliability evidence is strong.

## Figure 3: Reliability Recovers Corruption Mask

Show heatmaps:

- true corruption mask,
- learned reliability matrix,
- absolute prior error.

Report AUROC/AUPRC.

## Figure 4: Prior Separation and Identifiability

X-axis:

```text
Delta = min class-prior distance
```

Y-axis:

```text
concept F1
```

Expected:

Concept F1 drops as class prior signatures collapse.

## Figure 5: Audit Budget

X-axis:

```text
percentage of images with concept audit labels
```

Y-axis:

```text
reliability AUROC and concept F1
```

Expected:

Even small audit budgets improve reliability estimates.

---

# 8. Ablation Studies

Run these ablations:

1. PCP without reliability.
2. R-PCP with model-prior residual only.
3. R-PCP with cross-fitting.
4. R-PCP with source-disagreement initialization.
5. R-PCP with small audit calibration.
6. R-PCP with full Bernoulli KL vs original PCP KL.
7. R-PCP without entropy loss.
8. R-PCP without class-matching loss.
9. R-PCP with fixed reliability threshold.
10. R-PCP with learned continuous reliability.

Critical ablation:

```text
model-prior agreement only vs agreement + external evidence
```

If model-prior agreement only performs poorly, that supports the revised motivation.

---

# 9. Theory Plan

## 9.1 Proposition 1: Moment Non-Identifiability

State formally:

Class-level moment constraints do not identify per-image concept labels.

Proof:

Any zero-mean within-class perturbation preserves class-level priors but changes instance-level predictions.

Purpose:

This protects the paper from overclaiming.

## 9.2 Proposition 2: Prior Signature Separability

Let:

```text
Pi in [0,1]^(M x K)
```

If two classes have identical prior columns:

```text
Pi[:, y] = Pi[:, y']
```

then prior matching cannot distinguish the classes through concepts alone.

Define:

```text
Delta = min_{y != y'} ||Pi[:, y] - Pi[:, y']||_2
```

Expected recoverability worsens as `Delta -> 0`.

Purpose:

This motivates the prior-separation experiment.

## 9.3 Proposition 3: Reliability Helps Under Known Corruption

In a simplified setting where:

- the corruption mask is known or estimated with bounded error,
- prior corruption affects a subset of entries,
- concept means are estimated with finite-sample error,

show that reliability weighting reduces the contribution of corrupted prior entries to the moment-matching objective.

Do not claim a broad universal bound unless fully proven.

Safer statement:

```text
If r[m, y] is lower on corrupted entries than clean entries, the weighted prior objective has lower bias than the unweighted PCP objective.
```

This is easier to prove and aligns with the experiments.

## 9.4 Avoid Overclaiming Natarajan/Scott-Zhang Bound

The old plan proposed:

```text
Risk gap ~= O(1 / (Delta^2 * (1 - 2 eta)^2 * sqrt(N)))
```

This should not be used as a main theorem unless derived rigorously.

Better:

- cite noisy-label and LLP theory as motivation,
- present a simplified proposition,
- use empirical Delta and corruption sweeps as the main evidence.

---

# 10. Implementation Plan

## 10.1 Repository Structure

```text
rpcp/
  configs/
    ph2.yaml
    wbcatt.yaml
    derm7pt.yaml
    lidc.yaml
  rpcp/
    models/
      backbone.py
      concept_predictor.py
      rpcp.py
      reliability.py
    losses/
      prior.py
      matching.py
      entropy.py
      classification.py
      composite.py
    data/
      ph2.py
      wbcatt.py
      derm7pt.py
      lidc.py
      priors.py
      corruption.py
    evaluation/
      concept_metrics.py
      class_metrics.py
      reliability_metrics.py
      calibration.py
      prior_separation.py
    training/
      trainer.py
      crossfit.py
      schedules.py
    plotting/
      heatmaps.py
      robustness_curves.py
  scripts/
    train.py
    evaluate.py
    run_corruption_sweep.py
    run_delta_sweep.py
    run_audit_budget_sweep.py
    make_figures.py
  tests/
    test_prior_loss.py
    test_corruption.py
    test_reliability_metrics.py
    test_crossfit.py
```

## 10.2 Core Modules

### Prior Corruption

Implement:

```python
def compute_priors_from_annotations(dataset):
    ...

def corrupt_priors(priors, mode, alpha, fraction, seed):
    ...

def compute_corruption_mask(clean_priors, noisy_priors):
    ...
```

### Reliability Metrics

Implement:

```python
def reliability_auroc(r, corruption_mask):
    ...

def reliability_auprc(r, corruption_mask):
    ...

def reliability_spearman(r, prior_error):
    ...
```

### Cross-Fit Reliability

Implement:

```python
def estimate_class_means_crossfit(model_factory, dataset, folds):
    ...
```

### Prior Losses

Implement:

```python
def bernoulli_prior_kl(priors, predicted_means, reliability):
    ...

def original_pcp_kl(priors, predicted_means, reliability, concept_groups):
    ...
```

---

# 11. Milestones

## Week 1: Reproduce PCP Setup

- Implement or reuse PCP baseline.
- Run on PH2 first.
- Verify concept and class metrics.
- Compute clean prior table from hidden concept annotations.

Exit criterion:

PCP baseline runs end-to-end with clean and noisy priors.

## Week 2: Controlled Corruption Benchmark

- Implement prior corruption modes.
- Run PCP under increasing corruption.
- Produce first robustness curves.

Exit criterion:

Show that PCP degrades under noisy priors.

## Week 3: R-PCP Reliability Module

- Implement reliability-weighted prior loss.
- Implement cross-fit held-out mean estimation.
- Implement reliability AUROC/AUPRC.

Exit criterion:

R-PCP produces reliability heatmaps and does not collapse to all zeros or all ones.

## Week 4: Main PH2 and WBCatt Sweeps

- Run corruption sweeps.
- Compare PCP, R-PCP, R-PCP + audit, oracle.

Exit criterion:

R-PCP improves over PCP under at least one realistic corruption mode.

## Week 5: Prior-Separation Experiment

- Run Delta sweep.
- Plot concept F1 vs `Delta`.

Exit criterion:

Observed degradation supports identifiability argument.

## Week 6: Go/No-Go Decision

Go if:

- R-PCP improves concept F1 under noisy priors by at least 2 points in one dataset,
- reliability AUROC for corruption detection is above 0.70 in controlled corruption,
- R-PCP is closer to oracle reliability than PCP,
- Delta sweep shows meaningful degradation as prior signatures collapse.

No-go if:

- reliability scores do not recover corruption masks,
- R-PCP does not outperform PCP under corruption,
- performance gains appear only in clean-prior settings.

Fallback:

Pivot to **small-audit reliability calibration** or **multi-rater reliability modeling**.

## Week 7-8: External Validation

- Add Derm7pt or LIDC-IDRI.
- Run audit-budget experiments.
- Evaluate multi-rater disagreement if using LIDC-IDRI.

## Week 9-10: Paper Figures and Writing

- Finalize figures.
- Write method and experiments.
- Keep limitations prominent.

---

# 12. Go/No-Go Criteria

## Strong Go

The paper is strong if:

- R-PCP consistently beats PCP under corrupted priors.
- Reliability scores recover corrupted prior entries.
- Small audit or multi-source priors improve reliability.
- Oracle reliability shows meaningful headroom.
- Delta sweep supports the theory.

## Weak Go

The paper may still work if:

- unsupervised reliability is weak,
- but small audit calibration works very well.

Reframe as:

> Auditing class-level priors with minimal concept annotation for robust weakly supervised concept learning.

## No-Go

Do not continue with the original unsupervised reliability claim if:

- reliability mostly tracks model confidence rather than prior correctness,
- R-PCP helps classification but hurts concept F1,
- reliability cannot distinguish clean from corrupted priors in controlled tests.

---

# 13. Risk Register

## Risk 1: Reliability Is Self-Confirming

Problem:

The model is trained to match priors, then reliability is estimated from agreement with priors.

Mitigation:

- controlled corruption mask,
- cross-fitting,
- small audit set,
- multi-source prior disagreement,
- oracle comparison.

## Risk 2: Concept Labels Are Not Identifiable

Problem:

Class-level priors constrain only averages.

Mitigation:

- state non-identifiability,
- evaluate on hidden concept labels,
- run Delta sweep,
- avoid universal claims.

## Risk 3: R-PCP Ignores Too Many Priors

Problem:

Reliability collapses to zero.

Mitigation:

- Beta prior,
- lower-bound reliability clamp,
- monitor reliability histograms,
- oracle and threshold ablations.

## Risk 4: R-PCP Trusts Bad Priors

Problem:

Reliability remains high on corrupted entries.

Mitigation:

- stronger instability penalty,
- multi-source disagreement,
- audit calibration,
- corruption-specific analysis.

## Risk 5: Gains Are Only Classification Gains

Problem:

The model may improve class F1 while learning wrong concepts.

Mitigation:

- concept F1 is primary metric,
- concept calibration,
- reliability vs concept-F1 correlation,
- intervention tests if possible.

---

# 14. Revised Abstract Draft

Concept Bottleneck Models require per-image concept annotations, which are expensive in medical imaging. Recent prior-guided approaches reduce this burden by using class-level concept priors as weak supervision, but they assume these priors are reliable. This assumption is fragile: priors derived from experts, literature, small cohorts, or language models may be noisy or biased. We introduce Reliability-Weighted Prior-Guided Concept Prediction (R-PCP), a framework for robust concept learning from noisy class-level priors. R-PCP assigns a reliability score to each concept-class prior entry and downweights unreliable constraints during training. We evaluate R-PCP using controlled prior-corruption benchmarks on datasets with hidden concept annotations, measuring both concept prediction quality and the ability to recover corrupted prior entries. We further analyze identifiability through prior-separation experiments and show that small audit sets or multi-source priors substantially improve reliability estimation. Our results characterize when class-level priors support reliable concept learning and when external audit evidence is necessary.

---

# 15. Revised Paper Outline

## 1. Introduction

- CBMs need concept labels.
- PCP reduces annotation burden using class-level priors.
- But class-level priors can be wrong.
- Wrong priors can force wrong concepts.
- Need robustness and auditability.

## 2. Related Work

- Concept Bottleneck Models.
- Label-free and weakly supervised CBMs.
- PCP/class-level prior supervision.
- Posterior regularization.
- Learning from label proportions.
- Noisy-label learning.
- Reasoning shortcuts and information leakage.

## 3. Problem Setup

- Data.
- Priors.
- Noisy prior model.
- Reliability definition.
- Non-identifiability statement.

## 4. Method

- Base PCP.
- Reliability-weighted prior loss.
- Reliability estimation.
- Cross-fitting.
- Optional multi-source/audit extensions.

## 5. Experiments

- Datasets.
- Controlled prior corruption.
- Baselines.
- Metrics.
- Main results.
- Reliability recovery.
- Prior separation.
- Audit budget.
- Ablations.

## 6. Discussion

- What reliability means.
- When class-level priors are enough.
- When audit labels are necessary.
- Clinical implications.

## 7. Limitations

- No full identifiability.
- Reliability can be self-confirming without external evidence.
- Dataset concept labels may be imperfect.
- Medical priors vary by population.

## 8. Conclusion

- R-PCP improves robustness to noisy class-level priors.
- Reliability audit is necessary for trustworthy weakly supervised concepts.

---

# 16. Final Positioning

The strongest version of this paper is not:

```text
We solve concept learning without concept annotations.
```

The strongest version is:

```text
We show that prior-guided concept learning is fragile under noisy priors,
then propose and evaluate reliability weighting as a way to audit and reduce
that fragility.
```

That framing is more honest, more testable, and more defensible.
