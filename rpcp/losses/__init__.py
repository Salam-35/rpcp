"""Loss terms of the R-PCP objective."""

from __future__ import annotations

from rpcp.losses.classification import ClassificationLoss, class_balanced_weights
from rpcp.losses.composite import CompositeObjective, LossBreakdown
from rpcp.losses.entropy import EntropyLoss, attention_entropy, binary_entropy
from rpcp.losses.matching import PriorMatchingLoss, prior_similarity_logits
from rpcp.losses.prior import (
    ReliabilityWeightedPriorLoss,
    bernoulli_kl,
    bernoulli_prior_kl,
    build_prior_loss,
    original_pcp_kl,
)

__all__ = [
    "ClassificationLoss",
    "CompositeObjective",
    "EntropyLoss",
    "LossBreakdown",
    "PriorMatchingLoss",
    "ReliabilityWeightedPriorLoss",
    "attention_entropy",
    "bernoulli_kl",
    "bernoulli_prior_kl",
    "binary_entropy",
    "build_prior_loss",
    "class_balanced_weights",
    "original_pcp_kl",
    "prior_similarity_logits",
]
