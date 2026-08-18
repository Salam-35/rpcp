"""Second-moment (variance) prior term -- an extension beyond plan 4.6.

``L_prior`` only constrains the class-conditional *first* moment
``E[c_m | y]``. A model can satisfy it exactly while predicting the *same*
constant probability for every image of class ``y`` -- zero within-class
variance -- which drives that concept's F1 to 0 for any concept whose true
label is not unanimous within the class, even though the mean is matched
perfectly. That is a distinct failure mode from the dead-concept and
identifiability-ceiling issues documented in
``concept-bottleneck-diagnosis.md``: a model minimising
``L_cls + L_match + L_prior + L_ent`` alone has no term that forbids
collapsing every image in a class to one point estimate, and the zero-
variance solution is often the easiest one to reach.

``L_var`` adds a second-moment target: treating each concept as
``c_m | y ~ Bernoulli(Pi[m, y])``, the target within-class variance is
``Pi[m, y] * (1 - Pi[m, y])``. Matching it does not resolve *which* image
should get 0 vs 1 -- Proposition 2 says that is fundamentally unidentified
from class-level priors alone -- but it does forbid the zero-variance
collapse, which is a strictly worse point on the loss surface than any
solution with correctly-sized spread. Combined with ``L_ent`` (which
penalises within-image indecision, not within-class collapse) and
``L_match`` (whose per-image coefficients at least make some spread cheaper
than none), this closes off the easiest way to satisfy ``L_prior`` while
destroying concept F1.

Disabled by default (``loss.lambda_var = 0.0``): this is an ablation-ready
addition, not a change to the paper's stated objective.
"""

from __future__ import annotations

import torch
from torch import nn

__all__ = ["BernoulliVariancePriorLoss"]


class BernoulliVariancePriorLoss(nn.Module):
    """``L_var``: matches within-class prediction variance to ``Pi(1-Pi)``.

    Args:
        eps: Numerical floor for divisions and the variance target.
        min_count: Classes with fewer than this many batch samples are
            excluded (a single-sample "class" has a trivial, uninformative
            zero variance that would otherwise be pushed toward
            ``Pi(1-Pi)`` for the wrong reason).
    """

    def __init__(self, *, eps: float = 1e-6, min_count: float = 2.0) -> None:
        super().__init__()
        self.eps = eps
        self.min_count = min_count

    def forward(
        self,
        concept_probs: torch.Tensor,
        labels: torch.Tensor,
        priors: torch.Tensor,
        means: torch.Tensor,
        weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute ``L_var`` for one batch.

        Args:
            concept_probs: ``(B, M)`` predicted concept probabilities.
            labels: ``(B,)`` class indices.
            priors: ``(M, K)`` target prior table (post prior-repair, so this
                term stays consistent with whatever ``L_prior`` is matching).
            means: ``(M, K)`` the *model's* class-conditional means for this
                batch (``estimate.means`` from :class:`ClassMeanEstimator`),
                so the variance is measured around the same target as
                ``L_prior`` uses, not a separate ad hoc estimate.
            weights: Optional ``(M, K)`` mask/weights, same convention as
                ``L_prior`` (e.g. reliability * column_mask).
        """
        device = concept_probs.device
        n_concepts, n_classes = means.shape
        one_hot = torch.zeros(labels.shape[0], n_classes, device=device, dtype=concept_probs.dtype)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)
        counts = one_hot.sum(dim=0)  # (K,)

        class_mean_per_sample = means.t()[labels]  # (B, M)
        squared_error = (concept_probs - class_mean_per_sample) ** 2  # (B, M)
        variance_sum = squared_error.t() @ one_hot  # (M, K)
        variance_hat = variance_sum / counts.clamp_min(self.eps).unsqueeze(0)

        target_variance = (priors * (1.0 - priors)).clamp_min(self.eps)

        valid = (counts >= self.min_count).view(1, -1).expand(n_concepts, -1).float()
        mask = valid if weights is None else valid * weights.to(device)

        diff2 = (variance_hat - target_variance) ** 2
        denominator = mask.sum().clamp_min(self.eps)
        return (diff2 * mask).sum() / denominator
