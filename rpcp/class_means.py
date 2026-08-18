"""Model-implied class-level concept means (plan 3.2 / 4.2).

.. math::
    \\hat\\Pi_\\theta[m, y] = \\mathbb{E}_{x|y}\\big[\\hat c_{\\theta,m}(x)\\big]
    \\quad\\text{estimated by}\\quad
    \\bar p[m, y] = \\mathrm{mean}_{i: y_i = y}\\, \\hat c_m(x_i)

A single mini-batch is a poor and high-variance estimator of a *class* mean,
especially for rare classes that may not appear in the batch at all.  We
therefore combine the current (differentiable) batch statistics with an
exponentially-weighted history of previous batches, detached from the graph::

    p_bar = (batch_sum + m * history_sum) / (batch_count + m * history_count)

Gradients flow only through ``batch_sum``, so the estimator is unbiased in the
sense that matters here: the model is pushed to change *this* batch's concepts
in the direction that fixes the class mean, while the target itself is stable.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

__all__ = ["ClassMeanEstimate", "ClassMeanEstimator", "batch_class_sums"]


def batch_class_sums(
    concept_probs: torch.Tensor,
    labels: torch.Tensor,
    n_classes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-class sums of concept probabilities.

    Args:
        concept_probs: ``(B, M)``.
        labels: ``(B,)`` class indices.
        n_classes: ``K``.

    Returns:
        ``(sums, counts)`` of shapes ``(M, K)`` and ``(K,)``.
    """
    if concept_probs.ndim != 2:
        raise ValueError(f"concept_probs must be (B, M), got {tuple(concept_probs.shape)}")
    one_hot = torch.zeros(
        labels.shape[0], n_classes, device=concept_probs.device, dtype=concept_probs.dtype
    )
    one_hot.scatter_(1, labels.view(-1, 1), 1.0)
    sums = concept_probs.t() @ one_hot  # (M, K)
    counts = one_hot.sum(dim=0)  # (K,)
    return sums, counts


@dataclass(slots=True)
class ClassMeanEstimate:
    """Result of :meth:`ClassMeanEstimator.forward`.

    Attributes:
        means: ``(M, K)`` class means, differentiable through the current batch.
        support: ``(K,)`` effective number of samples behind each column.
        valid: ``(K,)`` boolean mask, ``False`` for classes with no support.
    """

    means: torch.Tensor
    support: torch.Tensor
    valid: torch.Tensor

    def column_mask(self, n_concepts: int) -> torch.Tensor:
        """``(M, K)`` mask broadcasting :attr:`valid` over concepts."""
        return self.valid.view(1, -1).expand(n_concepts, -1).float()


class ClassMeanEstimator(nn.Module):
    """Streaming estimator of ``p_bar`` with an EMA history.

    Args:
        n_concepts: ``M``.
        n_classes: ``K``.
        momentum: Decay applied to the history each step (``0`` = batch only).
        min_count: Classes with less effective support than this are marked invalid.
        eps: Numerical floor for the denominator.
        gradient_rescale: See :meth:`forward`.
    """

    def __init__(
        self,
        n_concepts: int,
        n_classes: int,
        *,
        momentum: float = 0.9,
        min_count: float = 1.0,
        eps: float = 1e-6,
        gradient_rescale: bool = True,
    ) -> None:
        super().__init__()
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"momentum must be in [0, 1), got {momentum}")
        self.n_concepts = n_concepts
        self.n_classes = n_classes
        self.momentum = momentum
        self.min_count = min_count
        self.eps = eps
        self.gradient_rescale = gradient_rescale
        self.register_buffer("history_sum", torch.zeros(n_concepts, n_classes))
        self.register_buffer("history_count", torch.zeros(n_classes))

    # ------------------------------------------------------------------ #
    def forward(
        self,
        concept_probs: torch.Tensor,
        labels: torch.Tensor,
        *,
        update: bool = True,
    ) -> ClassMeanEstimate:
        """Compute ``p_bar`` for this batch's labels, blended with history.

        With ``gradient_rescale`` (the default), the *forward value* of
        ``means`` is exactly the momentum-blended estimate below, but the
        *gradient* it passes back to ``concept_probs`` is rescaled so that
        ``d(means)/d(sums)`` equals ``1/counts`` (a batch-only estimator)
        instead of ``1/total_count``. Without this, at steady state the
        history holds ``momentum/(1-momentum)`` times the batch count -- 9x at
        the default ``momentum=0.9`` -- so ``d(means)/d(sums) = 1/total_count``
        is diluted by that same factor, and ``L_prior``'s gradient through
        ``p_bar`` (hence its effective strength relative to ``lambda_prior``)
        silently shrinks as the history accumulates. The history still reduces
        *variance* of the target value exactly as before; only the step size
        of the gradient it produces is restored.
        """
        sums, counts = batch_class_sums(concept_probs, labels, self.n_classes)

        history_sum = self.history_sum.to(sums.device) * self.momentum
        history_count = self.history_count.to(counts.device) * self.momentum

        total_sum = sums + history_sum.detach()
        total_count = counts + history_count.detach()
        means = total_sum / total_count.clamp_min(self.eps).unsqueeze(0)

        if self.gradient_rescale:
            # Straight-through rescale: `means - means.detach()` is exactly
            # 0.0 in the forward pass (same tensor, values bit-identical), so
            # this changes nothing numerically here; the multiply only alters
            # the gradient that flows back through the subtraction.
            scale = (total_count / counts.clamp_min(self.eps)).detach()
            means = means.detach() + scale.unsqueeze(0) * (means - means.detach())

        means = means.clamp(self.eps, 1.0 - self.eps)

        if update:
            with torch.no_grad():
                self.history_sum.copy_(history_sum + sums.detach())
                self.history_count.copy_(history_count + counts.detach())

        valid = total_count >= self.min_count
        return ClassMeanEstimate(means=means, support=total_count.detach(), valid=valid)

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def reset(self) -> None:
        self.history_sum.zero_()
        self.history_count.zero_()

    @torch.no_grad()
    def current_means(self) -> torch.Tensor:
        """Class means implied by the history alone (no current batch)."""
        return (
            self.history_sum / self.history_count.clamp_min(self.eps).unsqueeze(0)
        ).clamp(self.eps, 1.0 - self.eps)

    def extra_repr(self) -> str:
        return f"M={self.n_concepts}, K={self.n_classes}, momentum={self.momentum}"
