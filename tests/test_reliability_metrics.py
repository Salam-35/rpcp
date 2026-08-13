"""Tests for reliability estimation and its audit metrics (plan 4.3-4.4, 6.6)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from rpcp.config import ReliabilityConfig, ReliabilityMode
from rpcp.data.priors import (
    PriorBundle,
    reliability_from_audit,
    reliability_from_sources,
    source_disagreement,
    synthesize_prior_sources,
)
from rpcp.evaluation.ranking import average_precision, roc_auc, spearman
from rpcp.evaluation.reliability_metrics import (
    detection_at_threshold,
    reliability_auprc,
    reliability_auroc,
    reliability_metrics,
)
from rpcp.models.reliability import (
    ReliabilityEvidence,
    ReliabilityModule,
    beta_log_prior_penalty,
    build_reliability_module,
    oracle_reliability,
)


# --------------------------------------------------------------------------- #
# Ranking primitives
# --------------------------------------------------------------------------- #
def test_roc_auc_perfect_and_random() -> None:
    labels = np.array([0, 0, 1, 1])
    assert roc_auc(labels, np.array([0.1, 0.2, 0.8, 0.9])) == pytest.approx(1.0)
    assert roc_auc(labels, np.array([0.9, 0.8, 0.2, 0.1])) == pytest.approx(0.0)
    assert roc_auc(labels, np.array([0.5, 0.5, 0.5, 0.5])) == pytest.approx(0.5)


def test_roc_auc_is_undefined_for_one_class() -> None:
    assert np.isnan(roc_auc(np.zeros(4), np.random.rand(4)))


def test_average_precision_perfect_ranking() -> None:
    assert average_precision(np.array([0, 1, 1]), np.array([0.1, 0.9, 0.8])) == pytest.approx(1.0)


def test_spearman_is_rank_based() -> None:
    x = np.array([1.0, 2.0, 3.0, 4.0])
    assert spearman(x, x**3) == pytest.approx(1.0)
    assert spearman(x, -(x**3)) == pytest.approx(-1.0)


# --------------------------------------------------------------------------- #
# Reliability audit metrics
# --------------------------------------------------------------------------- #
def test_perfect_reliability_detects_corruption() -> None:
    corruption = torch.tensor([[True, False], [False, True]])
    reliability = torch.where(corruption, 0.05, 0.95)
    assert reliability_auroc(reliability, corruption) == pytest.approx(1.0)
    assert reliability_auprc(reliability, corruption) == pytest.approx(1.0)


def test_inverted_reliability_scores_below_chance() -> None:
    corruption = torch.tensor([[True, False], [False, True]])
    reliability = torch.where(corruption, 0.95, 0.05)
    assert reliability_auroc(reliability, corruption) == pytest.approx(0.0)


def test_detection_triplet() -> None:
    corruption = torch.tensor([[True, True], [False, False]])
    reliability = torch.tensor([[0.1, 0.9], [0.8, 0.7]])
    scores = detection_at_threshold(reliability, corruption, threshold=0.5)
    assert scores["precision"] == pytest.approx(1.0)
    assert scores["recall"] == pytest.approx(0.5)


def test_reliability_metrics_bundle() -> None:
    corruption = torch.tensor([[True, False], [False, False]])
    reliability = torch.tensor([[0.1, 0.9], [0.85, 0.95]])
    prior_error = torch.tensor([[0.6, 0.0], [0.0, 0.0]])
    metrics = reliability_metrics(
        reliability, corruption_mask=corruption, prior_error=prior_error
    )
    assert metrics.auroc == pytest.approx(1.0)
    assert metrics.separation > 0
    assert metrics.spearman_prior_error < 0  # more error -> less trust
    assert "precision" in metrics.detection
    assert "reliability/auroc" in metrics.as_dict()


# --------------------------------------------------------------------------- #
# Evidence helpers
# --------------------------------------------------------------------------- #
def test_source_disagreement_and_initial_reliability() -> None:
    sources = torch.stack([torch.full((2, 2), 0.5), torch.full((2, 2), 0.5)])
    disagreement = source_disagreement(sources)
    assert torch.allclose(disagreement, torch.zeros(2, 2))
    assert torch.allclose(reliability_from_sources(disagreement, 8.0), torch.ones(2, 2))


def test_synthetic_sources_disagree_more_on_corrupted_entries() -> None:
    prior = torch.full((4, 3), 0.5)
    corrupted = torch.zeros(4, 3, dtype=torch.bool)
    corrupted[0] = True
    sources = synthesize_prior_sources(
        prior, n_sources=32, noise=0.05, corrupted_mask=corrupted, seed=0
    )
    disagreement = source_disagreement(sources)
    assert disagreement[corrupted].mean() > disagreement[~corrupted].mean()


def test_audit_reliability_decays_with_error() -> None:
    observed = torch.tensor([[0.9, 0.1]])
    audit = torch.tensor([[0.9, 0.9]])
    reliability = reliability_from_audit(observed, audit, beta=5.0)
    assert reliability[0, 0] == pytest.approx(1.0)
    assert reliability[0, 1] < 0.1


# --------------------------------------------------------------------------- #
# ReliabilityModule
# --------------------------------------------------------------------------- #
def test_beta_penalty_prefers_its_mode() -> None:
    centre = beta_log_prior_penalty(torch.full((2, 2), 0.5), 2.0, 2.0)
    extreme = beta_log_prior_penalty(torch.full((2, 2), 0.99), 2.0, 2.0)
    assert centre < extreme

    optimistic_high = beta_log_prior_penalty(torch.full((2, 2), 0.9), 5.0, 2.0)
    optimistic_low = beta_log_prior_penalty(torch.full((2, 2), 0.1), 5.0, 2.0)
    assert optimistic_high < optimistic_low  # Beta(5,2) trusts priors


def test_module_scores_high_residual_as_unreliable() -> None:
    config = ReliabilityConfig(mode=ReliabilityMode.UNSUPERVISED)
    module = ReliabilityModule((2, 2), config)
    evidence = ReliabilityEvidence(
        prior_model_residual=torch.tensor([[0.0, 0.5], [0.0, 0.0]])
    )
    module.update(evidence)
    reliability = module()
    assert reliability[0, 1] < reliability[0, 0]


def test_ema_smoothing_is_applied_after_the_first_update() -> None:
    config = ReliabilityConfig(mode=ReliabilityMode.UNSUPERVISED, ema_gamma=0.5)
    module = ReliabilityModule((1, 1), config)
    module.update(ReliabilityEvidence(prior_model_residual=torch.zeros(1, 1)))
    first = module().clone()
    module.update(ReliabilityEvidence(prior_model_residual=torch.ones(1, 1)))
    second = module()
    assert second < first  # moved towards the new (worse) evidence
    direct = module.score(ReliabilityEvidence(prior_model_residual=torch.ones(1, 1)))
    assert second > direct  # but only halfway, because gamma = 0.5


def test_frozen_module_ignores_updates() -> None:
    config = ReliabilityConfig(mode=ReliabilityMode.NONE)
    module = ReliabilityModule((2, 2), config, frozen=True)
    module.update(ReliabilityEvidence(prior_model_residual=torch.ones(2, 2)))
    assert torch.allclose(module(), torch.ones(2, 2))


def test_hard_threshold_binarises() -> None:
    config = ReliabilityConfig(
        mode=ReliabilityMode.UNSUPERVISED, hard_threshold=0.5, min_reliability=0.0
    )
    module = ReliabilityModule((1, 2), config, init=torch.tensor([[0.2, 0.8]]))
    assert torch.equal(module(), torch.tensor([[0.0, 1.0]]))


def test_oracle_reliability_matches_the_mask() -> None:
    clean_mask = torch.tensor([[True, False]])
    assert torch.equal(oracle_reliability(clean_mask), torch.tensor([[1.0, 0.0]]))


def test_build_reliability_module_requires_its_evidence() -> None:
    priors = PriorBundle(observed=torch.rand(3, 2))
    with pytest.raises(ValueError, match="multi_source"):
        build_reliability_module(ReliabilityConfig(mode=ReliabilityMode.MULTI_SOURCE), priors)
    with pytest.raises(ValueError, match="audit"):
        build_reliability_module(ReliabilityConfig(mode=ReliabilityMode.AUDIT), priors)
    with pytest.raises(ValueError, match="oracle"):
        build_reliability_module(ReliabilityConfig(mode=ReliabilityMode.ORACLE), priors)


def test_build_reliability_module_oracle_is_frozen_and_exact() -> None:
    clean_mask = torch.tensor([[True, False], [True, True]])
    priors = PriorBundle(observed=torch.rand(2, 2), clean_mask=clean_mask)
    module, evidence = build_reliability_module(
        ReliabilityConfig(mode=ReliabilityMode.ORACLE, min_reliability=0.0), priors
    )
    assert module.frozen
    assert evidence.is_empty()
    assert torch.equal(module(), clean_mask.float())
