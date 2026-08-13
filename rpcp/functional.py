"""Dependency-free tensor functions shared by models and losses.

Kept in one leaf module (imports nothing from ``rpcp``) so that
:mod:`rpcp.models` and :mod:`rpcp.losses` can both use it without an import
cycle.
"""

from __future__ import annotations

import torch

__all__ = ["EPS", "prior_similarity_logits"]

EPS = 1e-6


def prior_similarity_logits(
    concept_probs: torch.Tensor,
    priors: torch.Tensor,
    *,
    reliability: torch.Tensor | None = None,
    temperature: float = 1.0,
    eps: float = EPS,
) -> torch.Tensor:
    """Bernoulli log-likelihood of predicted concepts under each class prior.

    .. math::
        \\ell_k(x) = \\frac{1}{\\tau}\\sum_m w_{m,k}\\Big[
            \\hat c_m(x)\\log\\tilde\\Pi[m,k]
            + (1-\\hat c_m(x))\\log(1-\\tilde\\Pi[m,k])\\Big]

    Args:
        concept_probs: ``(B, M)`` predicted concept probabilities.
        priors: ``(M, K)`` prior table.
        reliability: ``(M, K)`` optional weights ``w``; ``None`` means ``w = 1``.
        temperature: Logit temperature ``tau``.
        eps: Clamping constant keeping the logs finite.

    Returns:
        ``(B, K)`` class logits.  Two classes with identical prior columns give
        identical logits for every input -- Proposition 2 of the plan.
    """
    if concept_probs.ndim != 2:
        raise ValueError(f"concept_probs must be (B, M), got {tuple(concept_probs.shape)}")
    if concept_probs.shape[1] != priors.shape[0]:
        raise ValueError(
            f"concept dim mismatch: concepts {tuple(concept_probs.shape)} vs "
            f"priors {tuple(priors.shape)}"
        )
    priors = priors.clamp(eps, 1.0 - eps)
    log_present = priors.log()
    log_absent = (1.0 - priors).log()
    if reliability is not None:
        log_present = reliability * log_present
        log_absent = reliability * log_absent
    return (concept_probs @ log_present + (1.0 - concept_probs) @ log_absent) / temperature
