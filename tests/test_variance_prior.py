"""Tests for the optional second-moment (variance) prior term."""

from __future__ import annotations

import torch

from rpcp.losses.variance_prior import BernoulliVariancePriorLoss


def test_zero_variance_batch_is_penalised_when_prior_variance_is_high() -> None:
    """A model that predicts the same value for every image in a class."""
    loss_fn = BernoulliVariancePriorLoss()
    # 4 samples of class 0, all with the *exact* same concept prediction: the
    # mean is matched perfectly (0.5), but the within-class variance is 0.
    concept_probs = torch.full((4, 1), 0.5)
    labels = torch.zeros(4, dtype=torch.long)
    means = torch.full((1, 1), 0.5)
    priors = torch.full((1, 1), 0.5)  # Bernoulli(0.5) implies variance 0.25
    loss = loss_fn(concept_probs, labels, priors, means)
    assert loss.item() > 0.05  # squared error against a variance target of 0.25


def test_matched_variance_gives_near_zero_loss() -> None:
    loss_fn = BernoulliVariancePriorLoss(min_count=2.0)
    # Half the class-0 samples at 0, half at 1: sample variance = 0.25, which
    # matches Bernoulli(0.5)'s variance exactly.
    concept_probs = torch.tensor([[0.0], [1.0], [0.0], [1.0]])
    labels = torch.zeros(4, dtype=torch.long)
    means = torch.full((1, 1), 0.5)
    priors = torch.full((1, 1), 0.5)
    loss = loss_fn(concept_probs, labels, priors, means)
    assert loss.item() < 1e-5


def test_small_classes_are_excluded_by_min_count() -> None:
    loss_fn = BernoulliVariancePriorLoss(min_count=2.0)
    # Only one sample of class 1 in the batch -> that column must not
    # contribute (a single point has a trivial, uninformative zero variance).
    concept_probs = torch.tensor([[0.0], [1.0], [0.9]])
    labels = torch.tensor([0, 0, 1])
    means = torch.tensor([[0.5, 0.9]])
    priors = torch.tensor([[0.5, 0.9]])
    loss = loss_fn(concept_probs, labels, priors, means)
    # Class 0 alone (variance 0.25 vs target 0.25) should give ~0 loss; if the
    # single-sample class-1 column leaked in it would not.
    assert loss.item() < 1e-5


def test_gradient_flows_to_concept_probs() -> None:
    loss_fn = BernoulliVariancePriorLoss()
    concept_probs = torch.rand(6, 2).clamp(0.05, 0.95).requires_grad_(True)
    labels = torch.tensor([0, 0, 0, 1, 1, 1])
    means = torch.rand(2, 2).clamp(0.05, 0.95)
    priors = torch.rand(2, 2).clamp(0.05, 0.95)
    loss = loss_fn(concept_probs, labels, priors, means)
    loss.backward()
    assert concept_probs.grad is not None
    assert torch.isfinite(concept_probs.grad).all()


def test_disabled_by_default_in_composite_objective() -> None:
    """`lambda_var=0.0` (the default) must not change the total loss."""
    from rpcp.config import LossConfig
    from rpcp.losses.composite import CompositeObjective

    class _Output:
        def __init__(self) -> None:
            self.concept_probs = torch.rand(4, 3).clamp(0.05, 0.95)
            self.attention = None
            self.class_logits = torch.randn(4, 2)

    config = LossConfig()
    assert config.lambda_var == 0.0
    objective = CompositeObjective(config, n_concepts=3, n_classes=2)
    labels = torch.tensor([0, 1, 0, 1])
    priors = torch.rand(3, 2).clamp(0.05, 0.95)
    breakdown = objective(_Output(), labels, priors)
    assert breakdown.variance is None
