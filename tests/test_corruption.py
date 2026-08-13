"""Tests for controlled prior corruption (plan 6.3)."""

from __future__ import annotations

import pytest
import torch

from rpcp.config import CorruptionMode
from rpcp.data.corruption import compute_corruption_mask, corrupt_priors, corruption_target
from rpcp.data.priors import blend_prior_columns, compute_priors_from_annotations
from rpcp.evaluation.prior_separation import prior_separation_delta


@pytest.fixture
def priors() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.rand(6, 3).clamp(0.05, 0.95)


def test_alpha_zero_is_a_no_op(priors: torch.Tensor) -> None:
    result = corrupt_priors(priors, CorruptionMode.UNIFORM, alpha=0.0, fraction=1.0)
    assert torch.allclose(result.priors, priors)
    assert bool(result.clean_mask.all())


def test_fraction_controls_the_number_of_corrupted_entries(priors: torch.Tensor) -> None:
    result = corrupt_priors(
        priors, CorruptionMode.ADVERSARIAL_FLIP, alpha=1.0, fraction=0.5, seed=1
    )
    n_entries = priors.numel()
    assert result.selected.sum().item() == pytest.approx(round(0.5 * n_entries), abs=1)
    assert result.corruption_mask.sum() <= result.selected.sum()


def test_adversarial_flip_inverts_selected_entries(priors: torch.Tensor) -> None:
    result = corrupt_priors(priors, CorruptionMode.ADVERSARIAL_FLIP, alpha=1.0, fraction=1.0)
    assert torch.allclose(result.priors, (1.0 - priors).clamp(1e-3, 1 - 1e-3), atol=1e-6)


def test_background_collapse_removes_class_information(priors: torch.Tensor) -> None:
    result = corrupt_priors(priors, CorruptionMode.BACKGROUND_COLLAPSE, alpha=1.0, fraction=1.0)
    # Every class column becomes the concept marginal -> zero separation.
    assert prior_separation_delta(result.priors) == pytest.approx(0.0, abs=1e-5)


def test_class_swap_uses_a_derangement(priors: torch.Tensor) -> None:
    generator = torch.Generator().manual_seed(3)
    target = corruption_target(priors, CorruptionMode.CLASS_SWAP, generator=generator)
    # Every column of the target must come from a *different* class column.
    for k in range(priors.shape[1]):
        assert not torch.allclose(target[:, k], priors[:, k])


def test_llm_bias_is_class_independent(priors: torch.Tensor) -> None:
    generator = torch.Generator().manual_seed(0)
    target = corruption_target(priors, CorruptionMode.LLM_BIAS, generator=generator)
    assert torch.allclose(target, target[:, :1].expand_as(target))


def test_alpha_interpolates_monotonically(priors: torch.Tensor) -> None:
    distances = [
        float(
            (corrupt_priors(priors, CorruptionMode.UNIFORM, alpha=a, fraction=1.0, seed=5).priors
             - priors).abs().mean()
        )
        for a in (0.2, 0.5, 0.9)
    ]
    assert distances[0] < distances[1] < distances[2]


def test_mask_is_reproducible_for_a_fixed_seed(priors: torch.Tensor) -> None:
    kwargs = {"mode": CorruptionMode.UNIFORM, "alpha": 0.7, "fraction": 0.4, "seed": 11}
    first = corrupt_priors(priors, **kwargs)
    second = corrupt_priors(priors, **kwargs)
    assert torch.equal(first.priors, second.priors)
    assert torch.equal(first.clean_mask, second.clean_mask)


def test_mask_matches_the_actual_difference(priors: torch.Tensor) -> None:
    result = corrupt_priors(priors, CorruptionMode.UNIFORM, alpha=0.8, fraction=0.6, seed=2)
    recomputed = compute_corruption_mask(priors, result.priors)
    assert torch.equal(recomputed, result.clean_mask)


def test_invalid_arguments_raise(priors: torch.Tensor) -> None:
    with pytest.raises(ValueError, match="alpha"):
        corrupt_priors(priors, CorruptionMode.UNIFORM, alpha=1.5)
    with pytest.raises(ValueError, match="fraction"):
        corrupt_priors(priors, CorruptionMode.UNIFORM, alpha=0.5, fraction=2.0)


def test_priors_from_annotations_recover_the_generating_probabilities() -> None:
    torch.manual_seed(0)
    n = 20000
    labels = torch.randint(0, 2, (n,))
    true = torch.tensor([[0.9, 0.1], [0.3, 0.7]])
    concepts = (torch.rand(n, 2) < true[:, labels].T).float()
    estimated = compute_priors_from_annotations(concepts.numpy(), labels.numpy(), n_classes=2)
    assert torch.allclose(estimated, true, atol=0.02)


def test_blending_columns_reduces_separation(priors: torch.Tensor) -> None:
    deltas = [
        prior_separation_delta(blend_prior_columns(priors, alpha, source=0, target=1))
        for alpha in (0.0, 0.5, 1.0)
    ]
    assert deltas[0] >= deltas[1] >= deltas[2]
    assert deltas[2] == pytest.approx(0.0, abs=1e-6)
