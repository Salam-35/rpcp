# R-PCP: Reliability-Weighted Concept Learning from Noisy Class-Level Priors

Reference implementation of *Reliability-Weighted Prior-Guided Concept Prediction*
(R-PCP): concept-bottleneck training from **class-level concept priors only**, with a
reliability score per prior entry that detects and downweights entries the priors get wrong.

The claim this code is built to test is deliberately narrow:

> Reliability weighting improves robustness when class-level priors are noisy, and learned
> reliability scores can recover known corrupted prior entries under controlled and
> clinically motivated corruption settings.

It is **not** built to claim that class-level priors identify per-image concepts. That
non-identifiability is stated in the code (`rpcp/evaluation/prior_separation.py`,
`rpcp/functional.py`) and measured by the Δ-sweep.

---

## Install

```bash
python -m pip install -e ".[dev]"          # torch, numpy, pandas, matplotlib, pyyaml
python -m pip install -e ".[dev,excel]"    # + openpyxl, needed for the PH2 spreadsheet
```

Python ≥ 3.11. CPU is enough for the synthetic benchmark; the medical datasets want a GPU.

## Quick start (no data download required)

```bash
# one model
python scripts/train.py --config configs/synthetic.yaml --method r-pcp \
    -o priors.corruption.mode=class_swap priors.corruption.alpha=0.75

# Figure 2 + Figure 3: robustness curve and reliability heatmaps
python scripts/run_corruption_sweep.py --config configs/synthetic.yaml \
    --methods pcp r-pcp r-pcp-audit oracle \
    --mode class_swap --alphas 0 0.25 0.5 0.75 1.0 --seeds 0 1 2

# Figure 4: identifiability vs prior separation
python scripts/run_delta_sweep.py --config configs/synthetic.yaml --methods pcp r-pcp

# Figure 5: audit budget
python scripts/run_audit_budget_sweep.py --config configs/synthetic.yaml --budgets 0 0.02 0.05 0.1 0.2

# ablation table (plan section 8)
python scripts/run_ablations.py --config configs/synthetic.yaml --alpha 0.75

# rebuild every figure from saved artefacts
python scripts/make_figures.py --results-dir runs --output-dir figures
```

Every script takes `-o key=value` overrides against the YAML config
(`-o optim.epochs=5 reliability.ema_gamma=0.5 ...`), so no config file needs editing to run
a variant.

---

## What lives where

```
rpcp/
  config.py            typed experiment schema (YAML <-> dataclasses), all enums
  functional.py        prior-similarity logits (shared by models and losses)
  class_means.py       p_bar[m,y] estimator: differentiable batch + detached EMA history
  methods.py           named baselines/ablations as config overrides (plan 6.5, section 8)
  data/
    base.py            ConceptDataset, splits, loaders; batch dict contract
    manifest.py        CSV/XLSX-manifest dataset shared by all medical datasets
    synthetic.py       runnable benchmark with a known ground-truth prior
    ph2.py wbcatt.py derm7pt.py lidc.py    dataset-specific manifest builders
    priors.py          prior construction, audit prevalence, multi-source, rater agreement
    corruption.py      the five corruption modes + ground-truth mask s_true
  models/
    backbone.py        spatial feature extractors (torchvision optional)
    concept_predictor.py  attention / linear concept heads
    rpcp.py            backbone -> concepts -> class head
    reliability.py     r[m,y], evidence terms, Beta prior R(r), oracle reliability
  losses/
    prior.py           reliability-weighted Bernoulli KL and grouped PCP KL
    matching.py        L_match; classification.py: L_cls; entropy.py: L_ent
    composite.py       L_total
  training/
    schedules.py       phases 0-2, optimisers, EMA
    crossfit.py        held-out and cross-fitted class means, instability
    trainer.py         the loop, reliability updates, checkpointing, artefacts
  evaluation/
    concept_metrics.py class_metrics.py reliability_metrics.py calibration.py ranking.py
    prior_separation.py  Delta and the blend sweep
  plotting/            Figures 1-5
configs/               base + one per dataset
scripts/               train, evaluate, three sweeps, ablations, make_figures
tests/                 pytest suite (fast, CPU, no downloads)
```

## The objective, line by line

| Plan | Equation | Code |
|---|---|---|
| 3.2 | `Pi_hat[m,y] = E_{x|y}[c_hat_m(x)]` | `rpcp/class_means.py::ClassMeanEstimator` |
| 4.1 | `L_cls`, `L_match`, `L_ent` | `losses/classification.py`, `losses/matching.py`, `losses/entropy.py` |
| 4.2 | `L_prior = Σ_y Σ_m r[m,y] D(Π̃ ‖ p̄)` | `losses/prior.py::bernoulli_prior_kl` |
| 4.2 | grouped one-sided PCP KL (`R-PCP-PCPKL`) | `losses/prior.py::original_pcp_kl` |
| 4.3 A | controlled corruption + known mask | `data/corruption.py` |
| 4.3 B | `u = Var_s Π^(s)`, `r_0 = exp(-α u)` | `data/priors.py::source_disagreement`, `reliability_from_sources` |
| 4.3 C | `r_audit = exp(-β |Π̃ - Π_audit|)` | `data/priors.py::reliability_from_audit` |
| 4.3 D | inter-rater agreement | `data/priors.py::multi_rater_agreement` |
| 4.4 | `r = σ(w0 + w1·agree − w2·disagree − w3·instab − w4·residual)` | `models/reliability.py::ReliabilityModule.score` |
| 4.5 | cross-fitting | `training/crossfit.py` |
| 4.6 | `L_total`, `R(r) = −Σ log Beta(r; a0,b0)` | `losses/composite.py`, `models/reliability.py::beta_log_prior_penalty` |
| 5.1 | phases 0-3, EMA `r_t = γ r_{t-1} + (1−γ) r_new` | `training/schedules.py`, `training/trainer.py` |
| 6.3 | 5 corruption modes | `data/corruption.py::corruption_target` |
| 6.4 | `Δ = min_{y≠y'} ‖Π[:,y]−Π[:,y']‖₂`, blend sweep | `evaluation/prior_separation.py`, `data/priors.py::blend_prior_columns` |
| 6.6 | concept / class / reliability metrics | `evaluation/` |
| 9.1-9.2 | non-identifiability, separability | documented at `evaluation/prior_separation.py`, `functional.py` |

### Design decisions worth knowing

- **`r = 1` *is* PCP.** The baseline is the same code with `reliability.mode: none`, so it
  cannot drift away from the method it is compared against.
- **Normalised prior loss.** `L_prior` divides by `Σ r` by default
  (`loss.normalize_prior_by_reliability`), so lowering reliability cannot reduce the loss by
  itself — necessary once the reliability weights are learnable.
- **Reliability is estimated in Phase 1 but applied from Phase 2.** A noisy first estimate
  cannot wreck the warm-started model.
- **Residuals come from held-out data.** `reliability.use_crossfit: true` measures
  `|Π̃ − p̄|` on the validation split (never trained on). Set it to `false` to reproduce the
  self-confirming variant of ablation 2. Full A/B refitting is
  `training/crossfit.py::estimate_class_means_crossfit`.
- **Model selection never uses concept labels.** The default monitor is
  `val/class_macro_f1`; selecting on `val/concept_macro_f1` logs a warning because concept
  annotations are evaluation-only. `eval.monitor: last` keeps the final epoch.
- **Corruption masks are honest.** An entry selected for corruption but numerically
  unchanged counts as clean, so reliability is never punished for trusting it.

---

## Datasets

| Config | Dataset | Notes |
|---|---|---|
| `configs/synthetic.yaml` | in-memory | known `Π*`, runs in seconds, used by the tests |
| `configs/ph2.yaml` | PH2 | `rpcp.data.ph2.prepare_manifest(root)` converts the official XLSX |
| `configs/wbcatt.yaml` | WBCAtt | `prepare_manifest` merges the official train/val/test CSVs |
| `configs/derm7pt.yaml` | Derm7pt | `prepare_manifest` reads `meta/meta.csv` + official split indexes |
| `configs/lidc.yaml` | LIDC-IDRI | `prepare_manifest` needs `pylidc`; keeps per-rater columns |

All real datasets go through one manifest schema:

```csv
image,label,<concept_1>,...,<concept_M>[,split][,<concept>__rater0,...]
images/IMD003.bmp,melanoma,1,0,...,1,train,1,...
```

Concept columns are **evaluation-only**: they are used to build the benchmark prior table
(`priors.source: dataset`), to compute concept F1, and — for `r-pcp-audit` only — on the
small audit split. To supply an expert or LLM prior instead, set `priors.source: file` and
`priors.path` to a table with concepts as rows and classes as columns.

Adding a dataset means writing a `prepare_manifest` and registering a builder in
`rpcp/data/__init__.py::DATASET_REGISTRY`.

## Methods

`--method` (see `rpcp/methods.py`): `blackbox`, `supervised-cbm`, `pcp`, `r-pcp`,
`r-pcp-multisource`, `r-pcp-audit`, `r-pcp-multirater`, `oracle`.
The headline comparison is `pcp` vs `r-pcp` vs `oracle` under corruption; `supervised-cbm`
is the concept upper bound and the only method allowed to use per-image concept labels
(`loss.lambda_concept > 0`).

## Outputs

Each run writes to `runs/<name>-<tag>/`:

```
summary.json             config + all headline metrics
history.jsonl            per-epoch losses, val metrics, reliability stats
reliability.npy          final r  (M, K)
reliability_history.npy  r after every update (T, M, K) -> trajectory plot
prior_observed.npy prior_clean.npy corruption_mask.npy
checkpoint.pt
```

Sweeps additionally write `results.csv` (one row per run) next to their figures.

## Go / no-go instrumentation (plan 6, week 6)

The numbers the decision needs are columns in `results.csv`:

| Criterion | Column |
|---|---|
| R-PCP beats PCP under corruption by ≥ 2 points | `test/concept_macro_f1` per `method` |
| corruption detection AUROC > 0.70 | `reliability/auroc` |
| R-PCP closer to oracle than PCP | same column, `method == oracle` |
| Δ-sweep degradation | `prior/delta` vs `test/concept_macro_f1` |
| gains are not classification-only (Risk 5) | `test/class_macro_f1` vs `test/concept_macro_f1` |
| reliability separates clean from corrupt | `reliability/separation` |
| collapse / stuck-at-one (Risks 3-4) | `reliability/mean`, `reliability/std` in `history.jsonl` |

## Sanity run (what "working" looks like)

A 12-epoch toy run on the synthetic benchmark (600 train images, `simple_cnn`,
`adversarial_flip` on 40% of the prior entries, `alpha = 1`, single seed) gives:

| method | test concept F1 | reliability AUROC | mean r (clean) | mean r (corrupt) |
|---|---|---|---|---|
| *(no corruption)* | 0.730 | – | – | – |
| pcp | 0.433 | 0.50 | 1.00 | 1.00 |
| r-pcp (unsupervised) | 0.433 | 0.44 | 0.50 | 0.58 |
| r-pcp + multi-source | 0.432 | 0.64 | 0.92 | 0.88 |
| r-pcp + 10% audit | **0.445** | **1.00** | 0.92 | 0.09 |
| oracle reliability | 0.449 | 1.00 | 1.00 | 0.05 |

Read it as a plumbing check, not a result: the corruption really does break PCP
(0.73 → 0.43), audit-calibrated reliability recovers the corrupted entries perfectly and
tracks the oracle, and **unsupervised reliability sits at or below chance** — the
self-confirmation failure the plan anticipates (Risk 1, and the "weak go → pivot to
small-audit calibration" branch). Real conclusions need the full epoch budget, several
seeds and the medical datasets.

## Tests

```bash
python -m pytest -q            # ~20 s, CPU, no downloads
ruff check . && mypy rpcp      # optional
```

The suite covers the loss algebra (Bernoulli KL closed form, weighting, normalisation,
group partitions), every corruption mode and its mask, the ranking metrics against known
values, the reliability module (EMA, freezing, thresholding, Beta prior), class-mean
estimation and cross-fitting, and three end-to-end training runs.

## Limitations (kept prominent on purpose)

- Class-level priors constrain averages only; per-image concepts are **not** identifiable
  from them (Proposition 1). Concept F1 on hidden labels is the only honest check.
- Model-prior agreement is *evidence*, not proof. Without external evidence (audit,
  multi-source, multi-rater) reliability estimation is at risk of self-confirmation, which
  is what ablation 2 is for.
- Dataset concept annotations are themselves imperfect, and medical priors vary by
  population — a reliability score is a research instrument, not a clinical guarantee.
