"""Concept prediction heads.

Both heads implement

.. math:: \\hat c_\\theta(x) = \\sigma(g_\\theta(x)) \\in [0, 1]^M

(plan 3.2).  The attention head additionally exposes per-concept spatial
attention maps, which is what the entropy term ``L_ent`` sharpens.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

__all__ = ["ConceptHeadOutput", "AttentionConceptHead", "LinearConceptHead", "build_concept_head"]


@dataclass(slots=True)
class ConceptHeadOutput:
    """Output of a concept head.

    Attributes:
        logits: ``(B, M)`` pre-sigmoid concept scores ``g_theta(x)``.
        probs: ``(B, M)`` concept probabilities ``sigmoid(g_theta(x))``.
        attention: ``(B, M, H, W)`` spatial attention, or ``None``.
        pooled: ``(B, M, D)`` per-concept pooled features, or ``None``.
    """

    logits: torch.Tensor
    probs: torch.Tensor
    attention: torch.Tensor | None = None
    pooled: torch.Tensor | None = None


class LinearConceptHead(nn.Module):
    """Global-average-pool followed by an optional MLP and a linear concept layer."""

    def __init__(
        self,
        feature_dim: int,
        n_concepts: int,
        *,
        hidden_dim: int = 0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = feature_dim
        if hidden_dim > 0:
            layers += [nn.Linear(in_dim, hidden_dim), nn.ReLU(inplace=True)]
            in_dim = hidden_dim
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(in_dim, n_concepts))
        self.mlp = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> ConceptHeadOutput:
        pooled = features.mean(dim=(-2, -1))
        logits = self.mlp(pooled)
        return ConceptHeadOutput(logits=logits, probs=torch.sigmoid(logits))


class AttentionConceptHead(nn.Module):
    """Per-concept spatial attention pooling.

    For each concept ``m`` the head predicts an attention distribution
    ``a_m(x) in Delta^{HW}`` over feature locations, pools the feature map with
    it, and scores the pooled vector with a concept-specific weight::

        a_m      = softmax_hw(conv_m(F) / tau)
        z_m      = sum_hw a_m[h, w] * F[:, h, w]
        g_m(x)   = <w_m, z_m> + b_m

    Sharp attention (low entropy) means the concept is localised, which is what
    ``L_ent`` encourages.
    """

    def __init__(
        self,
        feature_dim: int,
        n_concepts: int,
        *,
        hidden_dim: int = 0,
        dropout: float = 0.0,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.n_concepts = n_concepts
        self.temperature = temperature
        self.project = (
            nn.Sequential(nn.Conv2d(feature_dim, hidden_dim, 1), nn.ReLU(inplace=True))
            if hidden_dim > 0
            else nn.Identity()
        )
        pooled_dim = hidden_dim if hidden_dim > 0 else feature_dim
        self.attention = nn.Conv2d(pooled_dim, n_concepts, kernel_size=1)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.weight = nn.Parameter(torch.empty(n_concepts, pooled_dim))
        self.bias = nn.Parameter(torch.zeros(n_concepts))
        nn.init.normal_(self.weight, std=pooled_dim**-0.5)

    def forward(self, features: torch.Tensor) -> ConceptHeadOutput:
        features = self.project(features)
        batch, channels, height, width = features.shape

        attention_logits = self.attention(features) / self.temperature  # (B, M, H, W)
        attention = torch.softmax(attention_logits.flatten(2), dim=-1)  # (B, M, HW)

        flat_features = features.flatten(2)  # (B, D, HW)
        pooled = torch.einsum("bmp,bdp->bmd", attention, flat_features)  # (B, M, D)
        pooled = self.dropout(pooled)

        logits = torch.einsum("bmd,md->bm", pooled, self.weight) + self.bias
        return ConceptHeadOutput(
            logits=logits,
            probs=torch.sigmoid(logits),
            attention=attention.view(batch, self.n_concepts, height, width),
            pooled=pooled,
        )


def build_concept_head(
    feature_dim: int,
    n_concepts: int,
    *,
    use_attention: bool = True,
    hidden_dim: int = 0,
    dropout: float = 0.0,
    temperature: float = 1.0,
) -> nn.Module:
    if use_attention:
        return AttentionConceptHead(
            feature_dim,
            n_concepts,
            hidden_dim=hidden_dim,
            dropout=dropout,
            temperature=temperature,
        )
    return LinearConceptHead(feature_dim, n_concepts, hidden_dim=hidden_dim, dropout=dropout)
