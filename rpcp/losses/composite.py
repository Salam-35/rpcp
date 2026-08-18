"""The full R-PCP objective (plan 4.6).

.. math::
    L_{total} = L_{cls}
        + \\lambda_{match} L_{match}
        + \\lambda_{prior} L_{prior}
        + \\lambda_{ent} L_{ent}
        + \\lambda_{r} R(r)

with

.. math::
    L_{prior} = \\sum_y \\sum_m r[m, y]\\,
        D_{Bern}\\big(\\tilde\\Pi[m, y] \\,\\|\\, \\bar p[m, y]\\big).

Setting ``r = 1`` everywhere recovers the PCP baseline, which is exactly how the
``reliability.mode: none`` configuration is implemented -- one code path, no
separate baseline implementation to drift out of sync.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
from torch import nn

from rpcp.class_means import ClassMeanEstimate, ClassMeanEstimator
from rpcp.config import LossConfig
from rpcp.losses.classification import ClassificationLoss
from rpcp.losses.entropy import EntropyLoss
from rpcp.losses.matching import PriorMatchingLoss
from rpcp.losses.prior import ReliabilityWeightedPriorLoss, build_prior_loss
from rpcp.losses.variance_prior import BernoulliVariancePriorLoss

if TYPE_CHECKING:  # avoids a models <-> losses import cycle
    from rpcp.models.rpcp import RPCPOutput

__all__ = ["CompositeObjective", "LossBreakdown"]


@dataclass(slots=True)
class LossBreakdown:
    """Per-term losses for one optimisation step."""

    total: torch.Tensor
    classification: torch.Tensor
    matching: torch.Tensor
    prior: torch.Tensor
    entropy: torch.Tensor
    reliability_penalty: torch.Tensor
    variance: torch.Tensor | None = None
    concept_supervision: torch.Tensor | None = None
    class_means: ClassMeanEstimate | None = None
    extras: dict[str, float] = field(default_factory=dict)

    def as_dict(self, prefix: str = "loss/") -> dict[str, float]:
        scalar = lambda t: float(t.detach())  # noqa: E731
        out = {
            f"{prefix}total": scalar(self.total),
            f"{prefix}cls": scalar(self.classification),
            f"{prefix}match": scalar(self.matching),
            f"{prefix}prior": scalar(self.prior),
            f"{prefix}ent": scalar(self.entropy),
            f"{prefix}reliability_penalty": scalar(self.reliability_penalty),
        }
        if self.variance is not None:
            out[f"{prefix}var"] = scalar(self.variance)
        if self.concept_supervision is not None:
            out[f"{prefix}concept_supervised"] = scalar(self.concept_supervision)
        out.update({f"{prefix}{k}": v for k, v in self.extras.items()})
        return out


class CompositeObjective(nn.Module):
    """Assembles ``L_cls``, ``L_match``, ``L_prior``, ``L_ent`` and ``R(r)``.

    Args:
        config: Loss configuration (lambdas, divergence variant, groups).
        n_concepts: ``M``.
        n_classes: ``K``.
        class_weights: Optional ``(K,)`` weights for ``L_cls``.
    """

    def __init__(
        self,
        config: LossConfig,
        *,
        n_concepts: int,
        n_classes: int,
        class_weights: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.classification = ClassificationLoss(
            label_smoothing=config.label_smoothing, class_weights=class_weights
        )
        self.matching = PriorMatchingLoss(
            temperature=config.match_temperature,
            label_smoothing=config.label_smoothing,
            reliability_weighted=config.match_reliability_weighted,
            eps=config.eps,
        )
        self.prior_loss: ReliabilityWeightedPriorLoss = build_prior_loss(config)
        self.entropy = EntropyLoss(
            on_attention=config.entropy_on_attention,
            on_concepts=config.entropy_on_concepts,
        )
        self.class_means = ClassMeanEstimator(
            n_concepts,
            n_classes,
            momentum=config.class_mean_momentum,
            eps=config.eps,
            gradient_rescale=config.class_mean_gradient_rescale,
        )
        self.variance_prior = BernoulliVariancePriorLoss(eps=config.eps)

    # ------------------------------------------------------------------ #
    def forward(
        self,
        output: RPCPOutput,
        labels: torch.Tensor,
        priors: torch.Tensor,
        *,
        reliability: torch.Tensor | None = None,
        reliability_penalty: torch.Tensor | None = None,
        repair_prior: torch.Tensor | None = None,
        update_class_means: bool = True,
        concepts: torch.Tensor | None = None,
        concept_mask: torch.Tensor | None = None,
    ) -> LossBreakdown:
        """Compute every loss term for one batch.

        Args:
            output: Model forward output.
            labels: ``(B,)`` class labels.
            priors: ``(M, K)`` observed prior table.
            reliability: ``(M, K)`` weights ``r``; ``None`` == PCP (``r = 1``).
            reliability_penalty: Pre-computed ``R(r)`` (see
                :meth:`rpcp.models.reliability.ReliabilityModule.penalty`).
            repair_prior: Optional replacement table for ``loss.prior_repair='audit'``.
            update_class_means: Whether to advance the class-mean history.
            concepts: ``(B, M)`` per-image concept labels -- used *only* when
                ``lambda_concept > 0`` (the supervised-CBM upper bound).
            concept_mask: ``(B,)`` which rows of ``concepts`` are valid.
        """
        device = output.concept_probs.device
        priors = priors.to(device)
        weights = None if reliability is None else reliability.to(device)

        estimate = self.class_means(
            output.concept_probs, labels, update=update_class_means
        )
        # Classes absent from both batch and history contribute nothing.
        column_mask = estimate.column_mask(priors.shape[0])
        weights = column_mask if weights is None else weights * column_mask
        prior_target = priors
        prior_weights = weights
        if self.config.prior_repair == "background":
            # Replace untrusted class-specific entries with the concept's
            # class-agnostic prevalence instead of merely downweighting them.
            r = torch.ones_like(priors) if reliability is None else reliability.to(device)
            rho = priors.mean(dim=1, keepdim=True)
            prior_target = r * priors + (1.0 - r) * rho
            prior_weights = column_mask
        elif self.config.prior_repair == "audit":
            # Replace untrusted entries with a class-specific prevalence estimated
            # on the held-out audit split.  With no reliability (PCP), r=1 makes
            # this a no-op so multi-method runs can share the same override.
            r = torch.ones_like(priors) if reliability is None else reliability.to(device)
            if repair_prior is None:
                if reliability is None:
                    audit_target = priors
                else:
                    raise ValueError(
                        "loss.prior_repair='audit' requires an audit prior; set "
                        "data.audit_fraction > 0 and use an audit-capable method"
                    )
            else:
                audit_target = repair_prior.to(device)
            if audit_target.shape != priors.shape:
                raise ValueError(
                    f"repair_prior {tuple(audit_target.shape)} must match priors "
                    f"{tuple(priors.shape)}"
                )
            prior_target = r * priors + (1.0 - r) * audit_target
            prior_weights = column_mask
        elif self.config.prior_repair != "none":
            raise ValueError(f"Unknown loss.prior_repair='{self.config.prior_repair}'")

        loss_cls = self.classification(output.class_logits, labels)
        loss_match = self.matching(output.concept_probs, priors, labels, reliability)
        loss_prior = self.prior_loss(prior_target, estimate.means, prior_weights)
        loss_var: torch.Tensor | None = None
        if self.config.lambda_var > 0:
            loss_var = self.variance_prior(
                output.concept_probs, labels, prior_target, estimate.means, weights=prior_weights
            )
        entropy_terms = self.entropy(output.concept_probs, output.attention)
        penalty = (
            torch.zeros((), device=device)
            if reliability_penalty is None
            else reliability_penalty.to(device)
        )

        loss_concept: torch.Tensor | None = None
        if self.config.lambda_concept > 0:
            loss_concept = self._concept_supervision(output, concepts, concept_mask)

        total = (
            self.config.lambda_cls * loss_cls
            + self.config.lambda_match * loss_match
            + self.config.lambda_prior * loss_prior
            + self.config.lambda_ent * entropy_terms.total
            + self.config.lambda_r * penalty
        )
        if loss_concept is not None:
            total = total + self.config.lambda_concept * loss_concept
        if loss_var is not None:
            total = total + self.config.lambda_var * loss_var

        extras: dict[str, float] = {}
        if entropy_terms.attention is not None:
            extras["ent_attention"] = float(entropy_terms.attention.detach())
        if entropy_terms.concept is not None:
            extras["ent_concept"] = float(entropy_terms.concept.detach())

        return LossBreakdown(
            total=total,
            classification=loss_cls,
            matching=loss_match,
            prior=loss_prior,
            entropy=entropy_terms.total,
            reliability_penalty=penalty,
            variance=loss_var,
            concept_supervision=loss_concept,
            class_means=estimate,
            extras=extras,
        )

    # ------------------------------------------------------------------ #
    @staticmethod
    def _concept_supervision(
        output: RPCPOutput,
        concepts: torch.Tensor | None,
        concept_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Masked BCE against per-image concept labels (supervised CBM only)."""
        device = output.concept_logits.device
        if concepts is None:
            raise ValueError(
                "lambda_concept > 0 but no per-image concepts were supplied; this term is "
                "reserved for the supervised-CBM baseline."
            )
        concepts = concepts.to(device).float()
        losses = nn.functional.binary_cross_entropy_with_logits(
            output.concept_logits, concepts, reduction="none"
        )
        if concept_mask is not None:
            mask = concept_mask.to(device).float().view(-1, 1)
            denominator = mask.sum().clamp_min(1.0) * losses.shape[1]
            return (losses * mask).sum() / denominator
        return losses.mean()

    def reset_class_means(self) -> None:
        self.class_means.reset()
