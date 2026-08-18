"""``L_ent``: attention sharpness and concept selectivity (plan 4.1).

Two complementary terms:

* **Attention entropy** -- the normalised Shannon entropy of each concept's
  spatial attention map.  Minimising it pushes concepts to look at a localised
  region instead of the whole image.
* **Concept (binary) entropy** -- the mean binary entropy of ``\\hat c_m(x)``.
  Minimising it pushes per-image concept predictions away from 0.5, i.e. makes
  the bottleneck decisive rather than hedging at the class prior.

Both are normalised to ``[0, 1]`` so that ``lambda_ent`` transfers across
datasets and image resolutions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

__all__ = ["EntropyLoss", "EntropyTerms", "attention_entropy", "binary_entropy"]

EPS = 1e-8


def binary_entropy(probs: torch.Tensor, *, eps: float = EPS) -> torch.Tensor:
    """Element-wise binary entropy in nats, normalised to ``[0, 1]``."""
    probs = probs.clamp(eps, 1.0 - eps)
    entropy = -(probs * probs.log() + (1.0 - probs) * (1.0 - probs).log())
    return entropy / math.log(2.0)


def attention_entropy(attention: torch.Tensor, *, eps: float = EPS) -> torch.Tensor:
    """Normalised entropy of per-concept spatial attention.

    Args:
        attention: ``(B, M, H, W)`` maps that sum to one over ``(H, W)``.
        eps: Numerical floor.

    Returns:
        ``(B, M)`` entropies divided by ``log(H*W)``, hence in ``[0, 1]``.
    """
    if attention.ndim != 4:
        raise ValueError(f"attention must be (B, M, H, W), got {tuple(attention.shape)}")
    flat = attention.flatten(2).clamp_min(eps)
    flat = flat / flat.sum(dim=-1, keepdim=True)
    entropy = -(flat * flat.log()).sum(dim=-1)
    # A 1x1 feature map has exactly one location: the entropy is 0 and so is
    # log(H*W).  Dividing gives 0/0 = NaN, which then propagates through
    # ``L_total.backward()`` and silently destroys every parameter in the model.
    # (Reachable via ``RPCPModel.forward``'s 1x1 fallback for pooled backbones,
    # and via SimpleCNN at image_size 16.)
    denominator = math.log(flat.shape[-1])
    if denominator <= 0.0:
        return torch.zeros_like(entropy)
    return entropy / denominator


@dataclass(slots=True)
class EntropyTerms:
    """Breakdown of the entropy penalty."""

    total: torch.Tensor
    attention: torch.Tensor | None
    concept: torch.Tensor | None


class EntropyLoss(nn.Module):
    """Weighted sum of attention entropy and concept binary entropy."""

    def __init__(
        self,
        *,
        on_attention: bool = True,
        on_concepts: bool = True,
        attention_weight: float = 1.0,
        concept_weight: float = 1.0,
        eps: float = EPS,
    ) -> None:
        super().__init__()
        self.on_attention = on_attention
        self.on_concepts = on_concepts
        self.attention_weight = attention_weight
        self.concept_weight = concept_weight
        self.eps = eps

    def forward(
        self,
        concept_probs: torch.Tensor,
        attention: torch.Tensor | None = None,
    ) -> EntropyTerms:
        device = concept_probs.device
        total = torch.zeros((), device=device)
        attention_term: torch.Tensor | None = None
        concept_term: torch.Tensor | None = None

        if self.on_attention and attention is not None:
            attention_term = attention_entropy(attention, eps=self.eps).mean()
            total = total + self.attention_weight * attention_term
        if self.on_concepts:
            concept_term = binary_entropy(concept_probs, eps=self.eps).mean()
            total = total + self.concept_weight * concept_term

        return EntropyTerms(total=total, attention=attention_term, concept=concept_term)
