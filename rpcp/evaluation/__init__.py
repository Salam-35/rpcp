"""Evaluation: predictions, concept/class metrics, reliability audit."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from rpcp.data.priors import PriorBundle
from rpcp.evaluation.calibration import (
    CalibrationCurve,
    brier_score,
    expected_calibration_error,
    reliability_curve,
)
from rpcp.evaluation.class_metrics import ClassMetrics, class_metrics, confusion_matrix
from rpcp.evaluation.concept_metrics import ConceptMetrics, binary_f1, concept_metrics
from rpcp.evaluation.prior_separation import (
    SeparationReport,
    delta_sweep,
    prior_separation_delta,
    separation_report,
)
from rpcp.evaluation.ranking import average_precision, roc_auc, spearman
from rpcp.evaluation.reliability_metrics import (
    ReliabilityMetrics,
    reliability_auprc,
    reliability_auroc,
    reliability_metrics,
    reliability_spearman,
)
from rpcp.models.rpcp import RPCPModel

__all__ = [
    "CalibrationCurve",
    "ClassMetrics",
    "ConceptMetrics",
    "EvaluationResult",
    "Predictions",
    "ReliabilityMetrics",
    "SeparationReport",
    "average_precision",
    "binary_f1",
    "brier_score",
    "class_metrics",
    "collect_predictions",
    "concept_metrics",
    "confusion_matrix",
    "delta_sweep",
    "evaluate_reliability",
    "evaluate_split",
    "expected_calibration_error",
    "prior_separation_delta",
    "reliability_auprc",
    "reliability_auroc",
    "reliability_curve",
    "reliability_metrics",
    "reliability_spearman",
    "roc_auc",
    "separation_report",
    "spearman",
]


@dataclass(slots=True)
class Predictions:
    """Stacked model outputs and ground truth for one split."""

    concept_probs: np.ndarray  # (N, M)
    class_probs: np.ndarray  # (N, K)
    labels: np.ndarray  # (N,)
    concepts: np.ndarray | None  # (N, M) evaluation-only
    indices: np.ndarray  # (N,)

    @property
    def has_concepts(self) -> bool:
        return self.concepts is not None

    def class_means(self, n_classes: int) -> np.ndarray:
        """Empirical ``p_bar`` on this split: ``(M, K)``."""
        means = np.full((self.concept_probs.shape[1], n_classes), np.nan)
        for y in range(n_classes):
            mask = self.labels == y
            if mask.any():
                means[:, y] = self.concept_probs[mask].mean(axis=0)
        return means


@torch.no_grad()
def collect_predictions(
    model: RPCPModel,
    loader: DataLoader[dict[str, torch.Tensor]],
    *,
    device: torch.device | str = "cpu",
    priors: torch.Tensor | None = None,
    reliability: torch.Tensor | None = None,
) -> Predictions:
    """Run the model over a loader and stack concept/class predictions."""
    was_training = model.training
    model.eval()

    concept_probs: list[np.ndarray] = []
    class_probs: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    concepts: list[np.ndarray] = []
    indices: list[np.ndarray] = []
    any_concepts = False

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        output = model(images, priors=priors, reliability=reliability)
        concept_probs.append(output.concept_probs.detach().cpu().numpy())
        class_probs.append(torch.softmax(output.class_logits, dim=-1).detach().cpu().numpy())
        labels.append(batch["label"].numpy())
        indices.append(batch["index"].numpy())
        if bool(batch["has_concepts"].any()):
            any_concepts = True
            concepts.append(batch["concepts"].numpy())

    model.train(was_training)
    return Predictions(
        concept_probs=np.concatenate(concept_probs) if concept_probs else np.empty((0, 0)),
        class_probs=np.concatenate(class_probs) if class_probs else np.empty((0, 0)),
        labels=np.concatenate(labels) if labels else np.empty(0, dtype=np.int64),
        concepts=np.concatenate(concepts) if any_concepts else None,
        indices=np.concatenate(indices) if indices else np.empty(0, dtype=np.int64),
    )


@dataclass(slots=True)
class EvaluationResult:
    """Concept + class metrics for one split (plus the raw predictions)."""

    split: str
    classification: ClassMetrics
    concept: ConceptMetrics | None = None
    predictions: Predictions | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    def as_dict(self, prefix: str | None = None) -> dict[str, float]:
        prefix = f"{self.split}/" if prefix is None else prefix
        out = self.classification.as_dict(f"{prefix}class_")
        if self.concept is not None:
            out.update(self.concept.as_dict(f"{prefix}concept_"))
        return out

    @property
    def monitor_value(self) -> float:
        if self.concept is not None and np.isfinite(self.concept.macro_f1):
            return self.concept.macro_f1
        return self.classification.macro_f1


def evaluate_split(
    model: RPCPModel,
    loader: DataLoader[dict[str, torch.Tensor]],
    *,
    split: str,
    device: torch.device | str = "cpu",
    priors: torch.Tensor | None = None,
    reliability: torch.Tensor | None = None,
    n_classes: int | None = None,
    concept_threshold: float = 0.5,
    n_bins: int = 15,
    concept_names: list[str] | None = None,
    class_names: list[str] | None = None,
    keep_predictions: bool = False,
) -> EvaluationResult:
    """Evaluate one split end to end."""
    predictions = collect_predictions(
        model, loader, device=device, priors=priors, reliability=reliability
    )
    n_classes = int(n_classes or predictions.class_probs.shape[1])

    concept_result: ConceptMetrics | None = None
    if predictions.has_concepts:
        concept_result = concept_metrics(
            predictions.concept_probs,
            predictions.concepts,  # type: ignore[arg-type]
            labels=predictions.labels,
            n_classes=n_classes,
            threshold=concept_threshold,
            n_bins=n_bins,
            concept_names=concept_names,
        )

    return EvaluationResult(
        split=split,
        classification=class_metrics(
            predictions.class_probs, predictions.labels, class_names=class_names
        ),
        concept=concept_result,
        predictions=predictions if keep_predictions else None,
    )


def evaluate_reliability(
    reliability: torch.Tensor,
    priors: PriorBundle,
    *,
    concept_result: ConceptMetrics | None = None,
    threshold: float = 0.5,
    n_bins: int = 10,
) -> ReliabilityMetrics:
    """Audit ``r`` against the corruption mask, prior error and concept quality."""
    return reliability_metrics(
        reliability,
        corruption_mask=priors.corruption_mask,
        prior_error=priors.prior_error(),
        per_entry_concept_f1=(
            None if concept_result is None else concept_result.per_class_concept_f1
        ),
        rater_agreement=priors.rater_agreement,
        threshold=threshold,
        n_bins=n_bins,
    )
