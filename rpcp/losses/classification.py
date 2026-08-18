"""``L_cls``: the class-prediction loss."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

__all__ = ["ClassificationLoss", "class_balanced_weights"]


def class_balanced_weights(
    class_counts: torch.Tensor,
    *,
    beta: float = 0.999,
    normalize: bool = True,
) -> torch.Tensor:
    """Effective-number class weights (Cui et al., 2019).

    Medical concept datasets are usually imbalanced (melanoma is rare); without
    reweighting, ``L_cls`` can dominate the concept terms for the majority class.

    Args:
        class_counts: ``(K,)`` number of training images per class.
        beta: Re-weighting strength; ``beta -> 0`` gives uniform weights.
        normalize: Rescale weights to mean 1.
    """
    counts = class_counts.clamp_min(1).float()
    effective = (1.0 - torch.pow(beta, counts)) / (1.0 - beta)
    weights = 1.0 / effective
    return weights / weights.mean() if normalize else weights


class ClassificationLoss(nn.Module):
    """Cross-entropy with optional label smoothing and class weights."""

    def __init__(
        self,
        *,
        label_smoothing: float = 0.0,
        class_weights: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.label_smoothing = label_smoothing
        # Registered as a buffer (not a plain attribute) so it moves with the
        # module under `.to(device)`; forward() always reads it back through
        # `getattr(self, "_class_weights", None)` rather than caching a second
        # reference, which previously went stale (stuck on the construction-
        # time device) the moment the module was moved.
        if class_weights is not None:
            self.register_buffer("_class_weights", class_weights.float())

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        weights = getattr(self, "_class_weights", None)
        return F.cross_entropy(
            logits,
            labels,
            weight=weights.to(logits.device) if weights is not None else None,
            label_smoothing=self.label_smoothing,
        )
