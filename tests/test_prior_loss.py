"""Tests for the reliability-weighted prior losses (plan 4.2)."""

from __future__ import annotations

import math

import pytest
import torch

from rpcp.config import LossConfig, PriorLossType
from rpcp.losses.prior import (
    ReliabilityWeightedPriorLoss,
    bernoulli_kl,
    bernoulli_prior_kl,
    build_prior_loss,
    original_pcp_kl,
)


def test_bernoulli_kl_is_zero_at_the_optimum() -> None:
    priors = torch.tensor([[0.2, 0.8], [0.5, 0.9]])
    assert torch.allclose(bernoulli_kl(priors, priors), torch.zeros_like(priors), atol=1e-6)


def test_bernoulli_kl_matches_closed_form() -> None:
    a, b = torch.tensor(0.3), torch.tensor(0.7)
    expected = 0.3 * math.log(0.3 / 0.7) + 0.7 * math.log(0.7 / 0.3)
    assert bernoulli_kl(a, b).item() == pytest.approx(expected, abs=1e-6)


def test_bernoulli_kl_is_non_negative_and_asymmetric() -> None:
    a = torch.rand(50).clamp(0.01, 0.99)
    b = torch.rand(50).clamp(0.01, 0.99)
    assert torch.all(bernoulli_kl(a, b) >= -1e-7)
    assert not torch.allclose(bernoulli_kl(a, b), bernoulli_kl(b, a))


def test_reliability_zero_removes_an_entry_from_the_loss() -> None:
    priors = torch.tensor([[0.9, 0.1], [0.2, 0.8]])
    means = torch.tensor([[0.5, 0.5], [0.2, 0.8]])
    reliability = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    loss = bernoulli_prior_kl(priors, means, reliability, reduction="sum")
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_downweighting_a_corrupted_entry_lowers_the_objective() -> None:
    """The core claim of plan 9.3 in miniature."""
    clean = torch.tensor([[0.9, 0.1], [0.2, 0.8]])
    corrupted = clean.clone()
    corrupted[0, 0] = 0.05  # this entry is now wrong
    means = clean  # a model that learned the *true* prior

    unweighted = bernoulli_prior_kl(corrupted, means, None, reduction="sum")
    weights = torch.ones_like(clean)
    weights[0, 0] = 0.1
    weighted = bernoulli_prior_kl(corrupted, means, weights, reduction="sum")
    assert weighted < unweighted


def test_normalisation_prevents_trivial_collapse() -> None:
    """With normalize=True, scaling all weights down cannot reduce the loss."""
    priors = torch.tensor([[0.9, 0.1], [0.2, 0.8]])
    means = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
    full = bernoulli_prior_kl(priors, means, torch.ones_like(priors), normalize=True)
    scaled = bernoulli_prior_kl(priors, means, 0.1 * torch.ones_like(priors), normalize=True)
    assert full.item() == pytest.approx(scaled.item(), abs=1e-5)


def test_pcp_kl_is_zero_when_group_normalised_means_match() -> None:
    priors = torch.tensor([[0.6, 0.2], [0.3, 0.6], [0.1, 0.2]])
    means = priors * 0.5  # same shape after within-group renormalisation
    loss = original_pcp_kl(priors, means, None, [[0, 1, 2]], reduction="sum")
    assert loss.item() == pytest.approx(0.0, abs=1e-5)


def test_pcp_kl_requires_a_partition() -> None:
    priors = torch.rand(4, 2)
    with pytest.raises(ValueError, match="partition"):
        original_pcp_kl(priors, priors, None, [[0, 1]])


def test_shape_validation() -> None:
    with pytest.raises(ValueError, match="same shape"):
        bernoulli_prior_kl(torch.rand(3, 2), torch.rand(4, 2))
    with pytest.raises(ValueError, match="reliability"):
        bernoulli_prior_kl(torch.rand(3, 2), torch.rand(3, 2), torch.rand(2, 2))


@pytest.mark.parametrize("loss_type", list(PriorLossType))
def test_module_variants_are_differentiable(loss_type: PriorLossType) -> None:
    priors = torch.rand(5, 3).clamp(0.05, 0.95)
    means = torch.rand(5, 3).clamp(0.05, 0.95).requires_grad_(True)
    loss = ReliabilityWeightedPriorLoss(loss_type)(priors, means, torch.rand(5, 3))
    loss.backward()
    assert means.grad is not None
    assert torch.isfinite(means.grad).all()


def test_build_prior_loss_reads_the_config() -> None:
    loss = build_prior_loss(LossConfig(prior_loss=PriorLossType.PCP_KL, prior_reduction="sum"))
    assert loss.loss_type is PriorLossType.PCP_KL
    assert loss.reduction == "sum"


def test_gradient_pushes_means_towards_the_prior() -> None:
    priors = torch.full((2, 2), 0.8)
    means = torch.full((2, 2), 0.3).requires_grad_(True)
    bernoulli_prior_kl(priors, means, None, reduction="sum").backward()
    # Loss decreases as the mean increases towards 0.8 -> negative gradient.
    assert torch.all(means.grad < 0)
