"""The R-PCP model: backbone -> concept bottleneck -> class prediction.

.. code-block:: text

    image -> backbone -> concept predictor -> class prediction
                             |
                             v
                 class-level concept means
                             |
    prior table -> reliability module -> weighted prior loss

This module owns the left-hand path (Figure 1 of the plan); the class-mean and
reliability machinery lives in :mod:`rpcp.training` and
:mod:`rpcp.models.reliability`.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import torch
from torch import nn

from rpcp.config import ClassHead, ExperimentConfig, ModelConfig
from rpcp.functional import prior_similarity_logits
from rpcp.models.backbone import Backbone, build_backbone
from rpcp.models.concept_predictor import ConceptHeadOutput, build_concept_head

__all__ = ["RPCPModel", "RPCPOutput", "build_model"]


@dataclass(slots=True)
class RPCPOutput:
    """Forward output.

    Attributes:
        concept_logits: ``(B, M)`` ``g_theta(x)``.
        concept_probs: ``(B, M)`` ``\\hat c_theta(x)``.
        class_logits: ``(B, K)``.
        attention: ``(B, M, H, W)`` or ``None``.
        features: ``(B, D, H, W)`` backbone features.
    """

    concept_logits: torch.Tensor
    concept_probs: torch.Tensor
    class_logits: torch.Tensor
    attention: torch.Tensor | None = None
    features: torch.Tensor | None = None

    def detach(self) -> RPCPOutput:
        maybe = lambda t: None if t is None else t.detach()  # noqa: E731
        return RPCPOutput(
            concept_logits=self.concept_logits.detach(),
            concept_probs=self.concept_probs.detach(),
            class_logits=self.class_logits.detach(),
            attention=maybe(self.attention),
            features=maybe(self.features),
        )


class RPCPModel(nn.Module):
    """Concept-bottleneck model trained from class labels + class-level priors.

    Args:
        backbone: Spatial feature extractor.
        n_concepts: ``M``.
        n_classes: ``K``.
        config: Model configuration.
        priors: ``(M, K)`` prior table, required by the ``prior``/``hybrid``
            class heads.  Stored as a buffer so it moves with ``.to(device)``
            and is saved in checkpoints.
        match_temperature: Temperature of the prior-similarity logits.
    """

    def __init__(
        self,
        backbone: Backbone,
        n_concepts: int,
        n_classes: int,
        config: ModelConfig,
        *,
        priors: torch.Tensor | None = None,
        match_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.config = config
        self.n_concepts = n_concepts
        self.n_classes = n_classes
        self.class_head_type = ClassHead(config.class_head)
        self.match_temperature = match_temperature

        self.backbone = backbone.module
        self.feature_dim = backbone.feature_dim
        self.concept_head = build_concept_head(
            self.feature_dim,
            n_concepts,
            use_attention=config.use_attention,
            hidden_dim=config.concept_hidden_dim,
            dropout=config.dropout,
            temperature=config.attention_temperature,
        )

        self.classifier = (
            nn.Linear(n_concepts, n_classes, bias=config.class_head_bias)
            if self.class_head_type in {ClassHead.LINEAR, ClassHead.HYBRID}
            else None
        )
        if self.class_head_type in {ClassHead.PRIOR, ClassHead.HYBRID}:
            if priors is None:
                raise ValueError(f"class_head='{self.class_head_type}' requires a prior table")
            self.register_buffer("priors", priors.detach().clone().float())
        else:
            self.priors = None  # type: ignore[assignment]

        if config.freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad_(False)

    # ------------------------------------------------------------------ #
    def forward(
        self,
        images: torch.Tensor,
        *,
        priors: torch.Tensor | None = None,
        reliability: torch.Tensor | None = None,
        return_features: bool = False,
    ) -> RPCPOutput:
        features = self.backbone(images)
        if features.ndim == 2:  # pooled backbone: fake a 1x1 spatial map
            features = features[..., None, None]
        head: ConceptHeadOutput = self.concept_head(features)
        class_logits = self.class_logits(head.probs, priors=priors, reliability=reliability)
        return RPCPOutput(
            concept_logits=head.logits,
            concept_probs=head.probs,
            class_logits=class_logits,
            attention=head.attention,
            features=features if return_features else None,
        )

    def class_logits(
        self,
        concept_probs: torch.Tensor,
        *,
        priors: torch.Tensor | None = None,
        reliability: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Class logits from the concept layer, per the configured head."""
        linear = None if self.classifier is None else self.classifier(concept_probs)
        if self.class_head_type is ClassHead.LINEAR:
            assert linear is not None
            return linear

        table = priors if priors is not None else getattr(self, "priors", None)
        if table is None:
            raise ValueError("prior table unavailable for the prior-similarity class head")
        prior_logits = prior_similarity_logits(
            concept_probs,
            table.to(concept_probs.device),
            reliability=reliability,
            temperature=self.match_temperature,
        )
        if self.class_head_type is ClassHead.PRIOR:
            return prior_logits
        assert linear is not None
        return 0.5 * (linear + prior_logits)

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def predict(self, images: torch.Tensor, **kwargs: object) -> RPCPOutput:
        was_training = self.training
        self.eval()
        output = self.forward(images, **kwargs)  # type: ignore[arg-type]
        self.train(was_training)
        return output.detach()

    def parameter_groups(
        self,
        lr: float,
        backbone_lr: float | None = None,
        weight_decay: float = 0.0,
    ) -> list[dict[str, object]]:
        """Optimiser groups with an optional lower backbone learning rate."""
        backbone_params = [p for p in self.backbone.parameters() if p.requires_grad]
        head_params = [
            p
            for name, p in self.named_parameters()
            if p.requires_grad and not name.startswith("backbone.")
        ]
        groups: list[dict[str, object]] = [
            {"params": head_params, "lr": lr, "weight_decay": weight_decay}
        ]
        if backbone_params:
            groups.append(
                {
                    "params": backbone_params,
                    "lr": backbone_lr if backbone_lr is not None else lr,
                    "weight_decay": weight_decay,
                }
            )
        return groups

    def trainable_parameters(self) -> Iterator[nn.Parameter]:
        return (p for p in self.parameters() if p.requires_grad)


def build_model(
    config: ExperimentConfig,
    *,
    n_concepts: int,
    n_classes: int,
    priors: torch.Tensor | None = None,
) -> RPCPModel:
    """Instantiate an :class:`RPCPModel` from an experiment config."""
    backbone = build_backbone(
        config.model.backbone,
        pretrained=config.model.pretrained,
        in_channels=config.model.in_channels,
    )
    return RPCPModel(
        backbone,
        n_concepts=n_concepts,
        n_classes=n_classes,
        config=config.model,
        priors=priors,
        match_temperature=config.loss.match_temperature,
    )
