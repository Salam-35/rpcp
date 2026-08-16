"""Reliability estimation over prior entries (plan 4.3 - 4.4).

The reliability matrix ``r in [0, 1]^{M x K}`` says how much each prior entry
should be trusted:

.. math::
    r[m, y] = \\sigma\\big(w_0
        + w_1\\,\\mathrm{agreement}[m, y]
        - w_2\\,\\mathrm{source\\_disagreement}[m, y]
        - w_3\\,\\mathrm{instability}[m, y]
        - w_4\\,\\mathrm{prior\\_model\\_residual}[m, y]\\big)

and is smoothed across updates with an EMA

.. math:: r_t = \\gamma r_{t-1} + (1-\\gamma) r_{new}.

Two warnings are baked into the API, matching the plan's framing:

* ``prior_model_residual`` is *evidence*, not proof -- the model was trained to
  agree with the prior, so residuals must come from held-out / cross-fitted
  class means (see :mod:`rpcp.training.crossfit`);
* the Beta log-prior ``R(r)`` is always available to stop the trivial collapse
  ``r -> 0``.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

import torch
from torch import nn

from rpcp.config import ReliabilityConfig, ReliabilityMode
from rpcp.data.priors import (
    PriorBundle,
    reliability_from_audit,
    reliability_from_sources,
    source_disagreement,
)

__all__ = [
    "ReliabilityEvidence",
    "ReliabilityModule",
    "beta_log_prior_penalty",
    "build_reliability_module",
    "oracle_reliability",
]


@dataclass(slots=True)
class ReliabilityEvidence:
    """Evidence terms feeding :meth:`ReliabilityModule.update`.

    All fields are ``(M, K)`` tensors in a roughly ``[0, 1]`` range (variances
    excepted) or ``None`` when that evidence source is unavailable.

    Attributes:
        agreement: External agreement (audit prevalence match, multi-source
            agreement, or inter-rater agreement).  Higher = more trustworthy.
        source_disagreement: Variance of the prior across sources.
        instability: Variance/std of the model class means across seeds,
            augmentations or folds.
        prior_model_residual: ``|Pi_tilde - p_bar_heldout|`` from held-out datasets.
    """

    agreement: torch.Tensor | None = None
    source_disagreement: torch.Tensor | None = None
    instability: torch.Tensor | None = None
    prior_model_residual: torch.Tensor | None = None

    def to(self, device: torch.device | str) -> ReliabilityEvidence:
        return ReliabilityEvidence(
            **{
                f.name: (None if (v := getattr(self, f.name)) is None else v.to(device))
                for f in fields(self)
            }
        )

    def merge(self, other: ReliabilityEvidence) -> ReliabilityEvidence:
        """Fill in this object's missing fields from ``other``."""
        return ReliabilityEvidence(
            **{
                f.name: (
                    getattr(self, f.name)
                    if getattr(self, f.name) is not None
                    else getattr(other, f.name)
                )
                for f in fields(self)
            }
        )

    def as_dict(self) -> dict[str, torch.Tensor]:
        return {
            f.name: value
            for f in fields(self)
            if (value := getattr(self, f.name)) is not None
        }

    def is_empty(self) -> bool:
        return not self.as_dict()


def beta_log_prior_penalty(
    reliability: torch.Tensor,
    a0: float = 2.0,
    b0: float = 2.0,
    *,
    eps: float = 1e-4,
    reduction: str = "mean",
) -> torch.Tensor:
    """``R(r) = -sum_{m,y} log Beta(r[m, y]; a0, b0)`` (plan 4.6).

    Choose ``(a0, b0)`` to encode how much the priors deserve trust *a priori*:
    ``Beta(2, 2)`` discourages extreme reliabilities without evidence,
    ``Beta(5, 2)`` is optimistic (expert priors), ``Beta(1, 5)`` pessimistic
    (LLM-generated priors).
    """
    r = reliability.clamp(eps, 1.0 - eps)
    distribution = torch.distributions.Beta(
        torch.tensor(float(a0), device=r.device),
        torch.tensor(float(b0), device=r.device),
    )
    penalty = -distribution.log_prob(r)
    match reduction:
        case "sum":
            return penalty.sum()
        case "mean":
            return penalty.mean()
        case "none":
            return penalty
        case _:
            raise ValueError(f"Unknown reduction '{reduction}'")


def oracle_reliability(
    clean_mask: torch.Tensor,
    *,
    clean_value: float = 1.0,
    corrupt_value: float = 0.0,
) -> torch.Tensor:
    """Upper-bound reliability built from the ground-truth corruption mask."""
    return torch.where(
        clean_mask,
        torch.full_like(clean_mask, clean_value, dtype=torch.float32),
        torch.full_like(clean_mask, corrupt_value, dtype=torch.float32),
    )


class ReliabilityModule(nn.Module):
    """Holds and updates the reliability matrix ``r``.

    Args:
        shape: ``(M, K)``.
        config: :class:`~rpcp.config.ReliabilityConfig`.
        init: Optional initial reliability (e.g. ``r_0 = exp(-alpha u)``).
        frozen: When ``True`` (PCP baseline / oracle) ``update`` is a no-op.
    """

    def __init__(
        self,
        shape: tuple[int, int],
        config: ReliabilityConfig,
        *,
        init: torch.Tensor | None = None,
        frozen: bool = False,
    ) -> None:
        super().__init__()
        self.config = config
        self.frozen = frozen
        self.n_updates = 0

        initial = torch.ones(shape) if init is None else init.detach().clone().float()
        if initial.shape != shape:
            raise ValueError(f"init shape {tuple(initial.shape)} != {shape}")
        self.register_buffer("reliability", initial)

        weights = torch.tensor(
            [config.w0, config.w1, config.w2, config.w3, config.w4], dtype=torch.float32
        )
        if config.learnable_weights and not frozen:
            self.weights = nn.Parameter(weights)
        else:
            self.register_buffer("weights", weights)

    # ------------------------------------------------------------------ #
    @property
    def shape(self) -> tuple[int, int]:
        return tuple(self.reliability.shape)  # type: ignore[return-value]

    def forward(self) -> torch.Tensor:
        """Current reliability, clamped (and optionally hard-thresholded)."""
        r = self.reliability
        if self.config.hard_threshold is not None:
            r = (r >= self.config.hard_threshold).float()
        return r.clamp(self.config.min_reliability, self.config.max_reliability)

    # ------------------------------------------------------------------ #
    def score(self, evidence: ReliabilityEvidence) -> torch.Tensor:
        """Map evidence to a reliability score with the plan 4.4 formula.

        External agreement enters *centred* at
        :attr:`ReliabilityConfig.agreement_center`: agreement above the centre
        raises reliability, below it lowers reliability, and a missing agreement
        signal contributes exactly nothing.  Without centring, strong external
        evidence that an entry is wrong (``agreement -> 0``) would merely fail to
        raise ``r`` instead of pushing it down, and could be overridden by a
        self-confirming model residual.
        """
        zeros = torch.zeros(self.shape, device=self.reliability.device)
        agreement = (
            zeros
            if evidence.agreement is None
            else _get(evidence.agreement, zeros) - self.config.agreement_center
        )
        disagreement = _get(evidence.source_disagreement, zeros)
        instability = _get(evidence.instability, zeros)
        residual = _get(evidence.prior_model_residual, zeros)

        w0, w1, w2, w3, w4 = self.weights
        logits = (
            w0
            + w1 * agreement
            - w2 * disagreement
            - w3 * self.config.instability_scale * instability
            - w4 * self.config.residual_scale * residual
        )
        return torch.sigmoid(logits)

    @torch.no_grad()
    def update(self, evidence: ReliabilityEvidence) -> torch.Tensor:
        """EMA-update the stored reliability from new evidence.

        Returns the updated (raw, unclamped) reliability matrix.
        """
        if self.frozen or evidence.is_empty():
            return self.reliability
        new = self.score(evidence.to(self.reliability.device))
        gamma = self.config.ema_gamma if self.n_updates > 0 else 0.0
        self.reliability.copy_(gamma * self.reliability + (1.0 - gamma) * new)
        self.n_updates += 1
        return self.reliability

    @torch.no_grad()
    def set_reliability(self, value: torch.Tensor) -> None:
        self.reliability.copy_(value.to(self.reliability.device).float())

    def penalty(self) -> torch.Tensor:
        """``R(r)`` -- the Beta log-prior regulariser."""
        return beta_log_prior_penalty(
            self.forward(), self.config.beta_a0, self.config.beta_b0, reduction="mean"
        )

    # ------------------------------------------------------------------ #
    def stats(self) -> dict[str, float]:
        """Summary used to monitor collapse/oscillation (plan 5.2, Risk 3)."""
        r = self.forward()
        return {
            "reliability/mean": float(r.mean()),
            "reliability/std": float(r.std(unbiased=False)),
            "reliability/min": float(r.min()),
            "reliability/max": float(r.max()),
            "reliability/frac_below_0.5": float((r < 0.5).float().mean()),
        }

    def extra_repr(self) -> str:
        return f"shape={self.shape}, mode={self.config.mode}, frozen={self.frozen}"


def _get(value: torch.Tensor | None, default: torch.Tensor) -> torch.Tensor:
    return default if value is None else value.to(default.device)


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
def build_reliability_module(
    config: ReliabilityConfig,
    priors: PriorBundle,
) -> tuple[ReliabilityModule, ReliabilityEvidence]:
    """Create the reliability module and the *static* evidence for its mode.

    Static evidence is everything that does not depend on the model: source
    disagreement (Mode B), audit agreement (Mode C), rater agreement (Mode D).
    The model-dependent terms (residual, instability) are supplied per update by
    the trainer.

    Returns:
        ``(module, static_evidence)``.
    """
    shape = priors.shape
    mode = ReliabilityMode(config.mode)
    evidence = ReliabilityEvidence()
    init: torch.Tensor | None = None
    frozen = False

    match mode:
        case ReliabilityMode.NONE:
            init, frozen = torch.ones(shape), True

        case ReliabilityMode.UNSUPERVISED:
            init = torch.ones(shape)

        case ReliabilityMode.MULTI_SOURCE:
            if priors.sources is None:
                raise ValueError(
                    "reliability.mode='multi_source' requires prior sources; set "
                    "priors.multi_source_paths or priors.n_synthetic_sources"
                )
            disagreement = source_disagreement(priors.sources)
            evidence.source_disagreement = disagreement
            evidence.agreement = reliability_from_sources(disagreement, config.source_alpha)
            init = evidence.agreement.clone()

        case ReliabilityMode.AUDIT:
            if priors.audit is None:
                raise ValueError(
                    "reliability.mode='audit' requires an audit split; set datasets.audit_fraction > 0"
                )
            evidence.agreement = reliability_from_audit(
                priors.observed,
                priors.audit,
                config.audit_beta,
                tolerance=config.audit_tolerance,
            )
            init = evidence.agreement.clone()

        case ReliabilityMode.MULTI_RATER:
            if priors.rater_agreement is None:
                raise ValueError(
                    "reliability.mode='multi_rater' requires per-rater annotations "
                    "(e.g. LIDC-IDRI manifests with '<concept>__rater<k>' columns)"
                )
            evidence.agreement = priors.rater_agreement
            init = evidence.agreement.clone()

        case ReliabilityMode.ORACLE:
            if priors.clean_mask is None:
                raise ValueError("reliability.mode='oracle' requires a known corruption mask")
            init = oracle_reliability(
                priors.clean_mask,
                clean_value=config.oracle_clean_value,
                corrupt_value=config.oracle_corrupt_value,
            )
            frozen = True

        case _:  # pragma: no cover
            raise ValueError(f"Unknown reliability mode: {mode}")

    module = ReliabilityModule(shape, config, init=init, frozen=frozen)
    return module, evidence


def reliability_summary(module: ReliabilityModule, priors: PriorBundle) -> dict[str, Any]:
    """Convenience bundle of reliability stats plus the prior error, if known."""
    summary: dict[str, Any] = module.stats()
    error = priors.prior_error()
    if error is not None:
        summary["prior/mean_abs_error"] = float(error.mean())
    return summary
