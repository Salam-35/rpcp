"""Tests for class-mean estimation and cross-fitting (plan 3.2 / 4.5)."""

from __future__ import annotations

import pytest
import torch

from rpcp.class_means import ClassMeanEstimator, batch_class_sums
from rpcp.config import ExperimentConfig
from rpcp.data import SplitBundle, make_dataloader
from rpcp.models.rpcp import RPCPModel, build_model
from rpcp.training.crossfit import (
    estimate_class_means,
    estimate_class_means_crossfit,
    estimate_instability,
    fold_class_means,
    model_evidence,
)


# --------------------------------------------------------------------------- #
# ClassMeanEstimator
# --------------------------------------------------------------------------- #
def test_batch_class_sums_are_exact() -> None:
    probs = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
    labels = torch.tensor([0, 0, 1])
    sums, counts = batch_class_sums(probs, labels, n_classes=2)
    assert torch.allclose(sums, torch.tensor([[1.0, 0.5], [1.0, 0.5]]))
    assert torch.allclose(counts, torch.tensor([2.0, 1.0]))


def test_class_means_without_history_equal_batch_means() -> None:
    estimator = ClassMeanEstimator(2, 2, momentum=0.0)
    probs = torch.tensor([[0.2, 0.8], [0.4, 0.6]])
    labels = torch.tensor([0, 0])
    estimate = estimator(probs, labels)
    assert torch.allclose(estimate.means[:, 0], torch.tensor([0.3, 0.7]), atol=1e-6)
    assert not bool(estimate.valid[1])  # class 1 has no support in this batch


def test_history_stabilises_the_estimate() -> None:
    estimator = ClassMeanEstimator(1, 1, momentum=0.9)
    labels = torch.zeros(4, dtype=torch.long)
    for _ in range(5):
        estimator(torch.full((4, 1), 0.8), labels)
    # A single deviant batch barely moves the running estimate.
    estimate = estimator(torch.full((4, 1), 0.0), labels)
    assert 0.4 < float(estimate.means[0, 0]) < 0.8


def test_gradients_flow_through_the_current_batch_only() -> None:
    estimator = ClassMeanEstimator(2, 2, momentum=0.9)
    labels = torch.tensor([0, 1])
    first = torch.rand(2, 2, requires_grad=True)
    estimator(first, labels).means.sum().backward()
    assert first.grad is not None

    second = torch.rand(2, 2, requires_grad=True)
    estimate = estimator(second, labels)
    estimate.means.sum().backward()
    assert torch.isfinite(second.grad).all()
    # History is detached, so no second gradient reaches `first`.
    assert first.grad.abs().sum() > 0


def test_reset_clears_history() -> None:
    estimator = ClassMeanEstimator(1, 1)
    estimator(torch.ones(2, 1), torch.zeros(2, dtype=torch.long))
    estimator.reset()
    assert float(estimator.history_count.sum()) == 0.0


# --------------------------------------------------------------------------- #
# Cross-fitting with a real (tiny) model
# --------------------------------------------------------------------------- #
@pytest.fixture
def tiny_model(tiny_config: ExperimentConfig, tiny_splits: SplitBundle) -> RPCPModel:
    return build_model(
        tiny_config,
        n_concepts=tiny_splits.n_concepts,
        n_classes=tiny_splits.n_classes,
        priors=torch.full((tiny_splits.n_concepts, tiny_splits.n_classes), 0.5),
    )


def test_estimate_class_means_shapes_and_support(
    tiny_model: RPCPModel, tiny_splits: SplitBundle
) -> None:
    loader = make_dataloader(tiny_splits.val, batch_size=16, shuffle=False)
    result = estimate_class_means(tiny_model, loader, n_classes=tiny_splits.n_classes)
    assert result.means.shape == (tiny_splits.n_concepts, tiny_splits.n_classes)
    assert float(result.counts.sum()) == len(tiny_splits.val)  # type: ignore[arg-type]
    assert bool(((result.means > 0) & (result.means < 1)).all())


def test_fold_means_and_instability(tiny_model: RPCPModel, tiny_splits: SplitBundle) -> None:
    folds = fold_class_means(
        tiny_model, tiny_splits.val, n_classes=tiny_splits.n_classes, n_folds=3, batch_size=8
    )
    assert folds.shape == (3, tiny_splits.n_concepts, tiny_splits.n_classes)
    instability = estimate_instability(folds)
    assert instability.shape == folds.shape[1:]
    assert bool((instability >= 0).all())


def test_instability_is_zero_for_identical_folds() -> None:
    folds = torch.rand(1, 3, 2).expand(4, 3, 2)
    assert torch.allclose(estimate_instability(folds), torch.zeros(3, 2))


def test_crossfit_requires_a_train_fn(tiny_splits: SplitBundle, tiny_model: RPCPModel) -> None:
    with pytest.raises(ValueError, match="train_fn"):
        estimate_class_means_crossfit(
            lambda: tiny_model, tiny_splits.train, folds=2, n_classes=tiny_splits.n_classes
        )


def test_crossfit_averages_over_folds(tiny_splits: SplitBundle, tiny_model: RPCPModel) -> None:
    means, per_fold = estimate_class_means_crossfit(
        lambda: tiny_model,
        tiny_splits.train,
        folds=2,
        train_fn=lambda model, _subset: model,  # no refit: we only test the plumbing
        n_classes=tiny_splits.n_classes,
        batch_size=16,
    )
    assert per_fold.shape[0] == 2
    assert torch.allclose(means, per_fold.mean(dim=0))


def test_model_evidence_is_the_absolute_residual() -> None:
    priors = torch.tensor([[0.9, 0.1]])
    held_out = torch.tensor([[0.4, 0.1]])
    evidence = model_evidence(priors, held_out)
    assert evidence.prior_model_residual is not None
    assert torch.allclose(evidence.prior_model_residual, torch.tensor([[0.5, 0.0]]), atol=1e-6)


def test_model_evidence_masks_unsupported_classes() -> None:
    priors = torch.tensor([[0.9, 0.1]])
    held_out = torch.tensor([[0.1, 0.9]])
    valid = torch.tensor([[True, False]])
    evidence = model_evidence(priors, held_out, valid_mask=valid)
    assert evidence.prior_model_residual is not None
    assert float(evidence.prior_model_residual[0, 1]) == 0.0
