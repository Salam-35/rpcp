"""Model components: backbone, concept predictor, R-PCP wrapper, reliability."""

from __future__ import annotations

from rpcp.models.backbone import Backbone, SimpleCNN, build_backbone
from rpcp.models.concept_predictor import (
    AttentionConceptHead,
    ConceptHeadOutput,
    LinearConceptHead,
    build_concept_head,
)
from rpcp.models.reliability import (
    ReliabilityEvidence,
    ReliabilityModule,
    beta_log_prior_penalty,
    build_reliability_module,
    oracle_reliability,
)
from rpcp.models.rpcp import RPCPModel, RPCPOutput, build_model

__all__ = [
    "AttentionConceptHead",
    "Backbone",
    "ConceptHeadOutput",
    "LinearConceptHead",
    "RPCPModel",
    "RPCPOutput",
    "ReliabilityEvidence",
    "ReliabilityModule",
    "SimpleCNN",
    "beta_log_prior_penalty",
    "build_backbone",
    "build_concept_head",
    "build_model",
    "build_reliability_module",
    "oracle_reliability",
]
