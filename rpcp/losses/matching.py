"""``L_match``: prior-guided concept-to-class matching (plan 4.1).

PCP asks the *instance-level* concept vector to be discriminative with respect
to the class prior signatures.  We implement that as a parameter-free classifier
whose logits are the Bernoulli log-likelihood of the predicted concept vector
under each class's prior column (:func:`rpcp.functional.prior_similarity_logits`):

.. math::
    \\ell_k(x) = \\frac{1}{\\tau}\\sum_m w_{m,k}\\Big[
        \\hat c_m(x)\\log\\tilde\\Pi[m,k] + (1-\\hat c_m(x))\\log(1-\\tilde\\Pi[m,k])\\Big]

with ``w = 1`` for PCP and ``w = r`` when reliability weighting is enabled (a
prior entry we do not trust should not drive class discrimination either).
``L_match`` is the cross-entropy of these logits against the true class.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from rpcp.functional import EPS, prior_similarity_logits

__all__ = ["PriorMatchingLoss", "prior_similarity_logits"]


class PriorMatchingLoss(nn.Module):
    """Cross-entropy of the prior-similarity classifier (``L_match``).

    Args:
        temperature: Logit temperature ``tau``.
        label_smoothing: Cross-entropy label smoothing.
        reliability_weighted: Weight the per-concept log-likelihood terms by
            ``r[m, y]`` (ablation 8; off by default so that ``L_match`` stays
            identical to PCP).
        eps: Clamping constant.
    """

    def __init__(
        self,
        *,
        temperature: float = 1.0,
        label_smoothing: float = 0.0,
        reliability_weighted: bool = False,
        eps: float = EPS,
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.label_smoothing = label_smoothing
        self.reliability_weighted = reliability_weighted
        self.eps = eps

    def forward(
        self,
        concept_probs: torch.Tensor,
        priors: torch.Tensor,
        labels: torch.Tensor,
        reliability: torch.Tensor | None = None,
    ) -> torch.Tensor:
        logits = prior_similarity_logits(
            concept_probs,
            priors,
            reliability=reliability if self.reliability_weighted else None,
            temperature=self.temperature,
            eps=self.eps,
        )
        return F.cross_entropy(logits, labels, label_smoothing=self.label_smoothing)
