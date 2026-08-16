"""Held-out and cross-fitted class means (plan 4.5).

Reliability estimated from ``|Pi_tilde - p_bar|`` on the *training* split is
self-confirming: the model was trained to make that residual small, so a
corrupted prior that the model successfully over-fitted looks reliable.  Two
remedies are implemented here:

``held_out``
    Estimate ``p_bar`` on datasets the current model never trained on (the
    validation split, or folds of the training split held out at estimation
    time).  Cheap; used by default.

``crossfit``
    Split the training set into folds, refit the model on fold ``A``, estimate
    class means on fold ``B``, swap, and average (plan 4.5, steps 1-5).  Costly
    but removes the *training* self-confirmation, not just the evaluation one.

Neither fixes non-identifiability (Proposition 1) -- they only remove the
circularity of scoring a prior with a model that was fitted to it.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from rpcp.data.base import ConceptBatch, TransformedSubset, make_dataloader
from rpcp.models.reliability import ReliabilityEvidence
from rpcp.models.rpcp import RPCPModel
from rpcp.utils.logging import get_logger

__all__ = [
    "ClassMeanResult",
    "estimate_class_means",
    "estimate_class_means_crossfit",
    "estimate_instability",
    "fold_class_means",
    "model_evidence",
]

logger = get_logger(__name__)

ModelFactory = Callable[[], RPCPModel]
TrainFn = Callable[[RPCPModel, Dataset[ConceptBatch]], RPCPModel]


@dataclass(slots=True)
class ClassMeanResult:
    """Class means plus the support behind them."""

    means: torch.Tensor  # (M, K)
    counts: torch.Tensor  # (K,)

    @property
    def valid(self) -> torch.Tensor:
        return self.counts > 0


@torch.no_grad()
def estimate_class_means(
    model: RPCPModel,
    loader: DataLoader[ConceptBatch],
    *,
    n_classes: int,
    device: torch.device | str = "cpu",
    priors: torch.Tensor | None = None,
    fill_value: float = 0.5,
) -> ClassMeanResult:
    """``p_bar[m, y]`` over a whole loader (no gradient, eval mode).

    Classes with no samples get ``fill_value`` so downstream residuals stay
    finite; use :attr:`ClassMeanResult.valid` to ignore them.
    """
    was_training = model.training
    model.eval()

    sums = torch.zeros(model.n_concepts, n_classes, device=device)
    counts = torch.zeros(n_classes, device=device)
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        probs = model(images, priors=priors).concept_probs
        one_hot = torch.zeros(labels.shape[0], n_classes, device=device, dtype=probs.dtype)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)
        sums += probs.t() @ one_hot
        counts += one_hot.sum(dim=0)

    model.train(was_training)
    means = torch.where(
        counts > 0,
        sums / counts.clamp_min(1.0).unsqueeze(0),
        torch.full_like(sums, fill_value),
    )
    return ClassMeanResult(means=means.cpu().clamp(1e-6, 1 - 1e-6), counts=counts.cpu())


def fold_class_means(
    model: RPCPModel,
    dataset: TransformedSubset,
    *,
    n_classes: int,
    n_folds: int = 2,
    batch_size: int = 64,
    device: torch.device | str = "cpu",
    seed: int = 0,
) -> torch.Tensor:
    """Class means computed independently on each of ``n_folds`` disjoint folds.

    Returns:
        ``(n_folds, M, K)`` tensor.  The spread across folds is the sampling
        component of the instability score.
    """
    rng = np.random.default_rng(seed)
    positions = rng.permutation(len(dataset))
    folds = np.array_split(positions, max(1, n_folds))

    means = []
    for fold in folds:
        if len(fold) == 0:
            continue
        loader = make_dataloader(
            dataset.subset(fold), batch_size=batch_size, shuffle=False, seed=seed
        )
        means.append(estimate_class_means(model, loader, n_classes=n_classes, device=device).means)
    return torch.stack(means)


def estimate_instability(fold_means: torch.Tensor) -> torch.Tensor:
    """Standard deviation of ``p_bar`` across folds/views -> ``(M, K)``.

    High instability means the class mean itself is not pinned down by the datasets,
    so the corresponding prior residual carries little information.
    """
    if fold_means.ndim != 3:
        raise ValueError(f"fold_means must be (F, M, K), got {tuple(fold_means.shape)}")
    if fold_means.shape[0] < 2:
        return torch.zeros(fold_means.shape[1:])
    return fold_means.std(dim=0, unbiased=False)


def estimate_class_means_crossfit(
    model_factory: ModelFactory,
    dataset: TransformedSubset,
    folds: int = 2,
    *,
    train_fn: TrainFn | None = None,
    n_classes: int,
    batch_size: int = 64,
    device: torch.device | str = "cpu",
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Full cross-fitting (plan 4.5).

    For each fold ``j``: train a *fresh* model on all other folds, estimate class
    means on fold ``j``.  Averaging over folds gives an out-of-fold estimate of
    ``p_bar`` in which no image contributed to the model that scored it.

    Args:
        model_factory: Returns a freshly initialised model.
        dataset: The training split.
        folds: Number of folds (2 reproduces the plan's A/B swap).
        train_fn: ``(model, subset) -> trained model``.  Required; the trainer
            passes a closure that runs the standard loop for a reduced number of
            epochs.
        n_classes: ``K``.
        batch_size / device / seed: Estimation settings.

    Returns:
        ``(means, per_fold_means)`` of shapes ``(M, K)`` and ``(F, M, K)``.
    """
    if train_fn is None:
        raise ValueError(
            "estimate_class_means_crossfit requires train_fn; use estimate_class_means on a "
            "held-out loader for the cheap variant."
        )
    rng = np.random.default_rng(seed)
    positions = rng.permutation(len(dataset))
    fold_positions: Sequence[np.ndarray] = np.array_split(positions, max(2, folds))

    per_fold: list[torch.Tensor] = []
    for index, held_out in enumerate(fold_positions):
        train_positions = np.concatenate(
            [f for j, f in enumerate(fold_positions) if j != index and len(f)]
        )
        logger.info(
            "Cross-fit fold %d/%d: train on %d, estimate on %d",
            index + 1,
            len(fold_positions),
            len(train_positions),
            len(held_out),
        )
        model = train_fn(model_factory(), dataset.subset(train_positions))
        loader = make_dataloader(
            dataset.subset(held_out), batch_size=batch_size, shuffle=False, seed=seed
        )
        per_fold.append(
            estimate_class_means(model, loader, n_classes=n_classes, device=device).means
        )

    stacked = torch.stack(per_fold)
    return stacked.mean(dim=0), stacked


def model_evidence(
    priors: torch.Tensor,
    held_out_means: torch.Tensor,
    *,
    instability: torch.Tensor | None = None,
    valid_mask: torch.Tensor | None = None,
) -> ReliabilityEvidence:
    """Turn held-out class means into the model-dependent evidence terms.

    ``prior_model_residual = |Pi_tilde - p_bar_heldout|`` is an *evidence* term,
    not proof of reliability: a prior can be wrong and still be matched by a
    model that latched onto a correlated shortcut.  Entries whose class has no
    held-out support are zeroed so they neither raise nor lower ``r``.
    """
    residual = (priors.cpu() - held_out_means.cpu()).abs()
    if valid_mask is not None:
        residual = residual * valid_mask.cpu().float()
    return ReliabilityEvidence(
        prior_model_residual=residual,
        instability=None if instability is None else instability.cpu(),
    )
