"""Reliability-weighted prior-matching losses (plan 4.2 / 4.6).

Two variants are provided, exactly as the plan requires:

``R-PCP-BernKL``
    the full Bernoulli divergence

    .. math::
        D_{Bern}(a \\| b) = a\\log\\frac{a}{b} + (1-a)\\log\\frac{1-a}{1-b}

``R-PCP-PCPKL``
    the original PCP grouped, one-sided KL, in which prior and predicted class
    means are renormalised inside each concept group and only the
    ``a log(a/b)`` term is kept.

Both are summed with per-entry reliability weights:

.. math::
    L_{prior} = \\sum_y \\sum_m r[m, y]\\, D\\big(\\tilde\\Pi[m, y] \\,\\|\\, \\bar p[m, y]\\big)
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from rpcp.config import LossConfig, PriorLossType

__all__ = [
    "ReliabilityWeightedPriorLoss",
    "bernoulli_kl",
    "bernoulli_prior_kl",
    "build_prior_loss",
    "original_pcp_kl",
    "reduce_weighted",
]

EPS = 1e-6


def bernoulli_kl(a: torch.Tensor, b: torch.Tensor, *, eps: float = EPS) -> torch.Tensor:
    """Element-wise Bernoulli KL divergence ``D(a || b)``.

    Args:
        a: Target probabilities (the prior ``Pi_tilde``).
        b: Model probabilities (the class means ``p_bar``).
        eps: Clamping constant keeping both arguments inside ``(0, 1)``.

    Returns:
        Tensor of the same shape as ``a``, non-negative, zero iff ``a == b``.
    """
    a = a.clamp(eps, 1.0 - eps)
    b = b.clamp(eps, 1.0 - eps)
    return a * (a.log() - b.log()) + (1.0 - a) * ((1.0 - a).log() - (1.0 - b).log())


def reduce_weighted(
    values: torch.Tensor,
    weights: torch.Tensor | None,
    *,
    reduction: str = "mean",
    normalize: bool = True,
    eps: float = EPS,
) -> torch.Tensor:
    """Apply reliability weights and reduce.

    Args:
        values: Per-entry divergences ``(M, K)``.
        weights: Per-entry reliability ``r`` in ``[0, 1]``, or ``None`` for PCP.
        reduction: ``"sum"`` (the plan's literal objective), ``"mean"`` (the same
            objective rescaled by ``1/(MK)``, which keeps ``lambda_prior``
            comparable across datasets) or ``"none"``.
        normalize: When ``True`` divide by ``sum(r)`` instead of ``M*K`` so that
            *downweighting cannot by itself reduce the loss*.  This matters when
            the reliability weights are learnable.
        eps: Numerical floor for the normaliser.
    """
    weighted = values if weights is None else weights * values
    match reduction:
        case "none":
            return weighted
        case "sum":
            return weighted.sum()
        case "mean":
            if weights is None or not normalize:
                return weighted.mean()
            return weighted.sum() / weights.sum().clamp_min(eps)
        case _:
            raise ValueError(f"Unknown reduction '{reduction}'")


def bernoulli_prior_kl(
    priors: torch.Tensor,
    predicted_means: torch.Tensor,
    reliability: torch.Tensor | None = None,
    *,
    reduction: str = "mean",
    normalize: bool = True,
    eps: float = EPS,
) -> torch.Tensor:
    """Reliability-weighted full Bernoulli KL between priors and class means.

    Args:
        priors: ``(M, K)`` observed prior table ``Pi_tilde``.
        predicted_means: ``(M, K)`` model class means ``p_bar`` (differentiable).
        reliability: ``(M, K)`` weights ``r``; ``None`` reproduces PCP.
        reduction / normalize / eps: see :func:`reduce_weighted`.
    """
    _check_shapes(priors, predicted_means, reliability)
    divergence = bernoulli_kl(priors, predicted_means, eps=eps)
    return reduce_weighted(
        divergence, reliability, reduction=reduction, normalize=normalize, eps=eps
    )


def original_pcp_kl(
    priors: torch.Tensor,
    predicted_means: torch.Tensor,
    reliability: torch.Tensor | None = None,
    concept_groups: Sequence[Sequence[int]] | None = None,
    *,
    reduction: str = "mean",
    normalize: bool = True,
    eps: float = EPS,
) -> torch.Tensor:
    """Reliability-weighted version of the original PCP grouped one-sided KL.

    Inside every concept group the prior column and the predicted class means
    are renormalised to sum to one, and only the forward term is kept::

        L = sum_y sum_g sum_{m in g} r[m, y] * q[m, y] * log(q[m, y] / p[m, y])

    with ``q = Pi_tilde_g / sum(Pi_tilde_g)`` and ``p = p_bar_g / sum(p_bar_g)``.

    Args:
        priors: ``(M, K)`` prior table.
        predicted_means: ``(M, K)`` model class means.
        reliability: ``(M, K)`` weights, or ``None``.
        concept_groups: Partition of concept indices into mutually-exclusive
            groups (e.g. the colour attributes of PH2).  ``None`` treats all
            concepts as a single group, which is the PCP default.
        reduction / normalize / eps: see :func:`reduce_weighted`.
    """
    _check_shapes(priors, predicted_means, reliability)
    n_concepts = priors.shape[0]
    groups = _resolve_groups(concept_groups, n_concepts)

    terms = torch.zeros_like(priors)
    for group in groups:
        index = torch.as_tensor(list(group), dtype=torch.long, device=priors.device)
        prior_group = priors.index_select(0, index).clamp_min(eps)
        mean_group = predicted_means.index_select(0, index).clamp_min(eps)
        q = prior_group / prior_group.sum(dim=0, keepdim=True).clamp_min(eps)
        p = mean_group / mean_group.sum(dim=0, keepdim=True).clamp_min(eps)
        terms.index_copy_(0, index, q * (q.clamp_min(eps).log() - p.clamp_min(eps).log()))

    return reduce_weighted(terms, reliability, reduction=reduction, normalize=normalize, eps=eps)


def _resolve_groups(
    concept_groups: Sequence[Sequence[int]] | None,
    n_concepts: int,
) -> list[list[int]]:
    if concept_groups is None:
        return [list(range(n_concepts))]
    flat = [m for group in concept_groups for m in group]
    if sorted(flat) != list(range(n_concepts)):
        raise ValueError(
            f"concept_groups must partition all {n_concepts} concepts, got {concept_groups}"
        )
    return [list(group) for group in concept_groups]


def _check_shapes(
    priors: torch.Tensor,
    predicted_means: torch.Tensor,
    reliability: torch.Tensor | None,
) -> None:
    if priors.shape != predicted_means.shape:
        raise ValueError(
            f"priors {tuple(priors.shape)} and class means {tuple(predicted_means.shape)} "
            "must have the same shape (M, K)"
        )
    if reliability is not None and reliability.shape != priors.shape:
        raise ValueError(
            f"reliability {tuple(reliability.shape)} must match priors {tuple(priors.shape)}"
        )


class ReliabilityWeightedPriorLoss(nn.Module):
    """``L_prior`` of plan 4.6, selectable between the two KL variants.

    Args:
        loss_type: :class:`~rpcp.config.PriorLossType`.
        concept_groups: Only used by the grouped PCP KL.
        reduction / normalize / eps: see :func:`reduce_weighted`.
    """

    def __init__(
        self,
        loss_type: PriorLossType | str = PriorLossType.BERNOULLI,
        *,
        concept_groups: Sequence[Sequence[int]] | None = None,
        reduction: str = "mean",
        normalize: bool = True,
        eps: float = EPS,
    ) -> None:
        super().__init__()
        self.loss_type = PriorLossType(loss_type)
        self.concept_groups = concept_groups
        self.reduction = reduction
        self.normalize = normalize
        self.eps = eps

    def forward(
        self,
        priors: torch.Tensor,
        predicted_means: torch.Tensor,
        reliability: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.loss_type is PriorLossType.BERNOULLI:
            return bernoulli_prior_kl(
                priors,
                predicted_means,
                reliability,
                reduction=self.reduction,
                normalize=self.normalize,
                eps=self.eps,
            )
        return original_pcp_kl(
            priors,
            predicted_means,
            reliability,
            self.concept_groups,
            reduction=self.reduction,
            normalize=self.normalize,
            eps=self.eps,
        )

    def extra_repr(self) -> str:
        return f"loss_type={self.loss_type}, reduction={self.reduction}, normalize={self.normalize}"


def build_prior_loss(config: LossConfig) -> ReliabilityWeightedPriorLoss:
    return ReliabilityWeightedPriorLoss(
        config.prior_loss,
        concept_groups=config.concept_groups,
        reduction=config.prior_reduction,
        normalize=config.normalize_prior_by_reliability,
        eps=config.eps,
    )
