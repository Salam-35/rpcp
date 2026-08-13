"""Concept-prediction metrics (plan 6.6).

Concept macro-F1 is the paper's *primary* metric: a method that improves class
accuracy while degrading concept F1 has learned a shortcut, not a concept
(plan 13, Risk 5).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from rpcp.evaluation.calibration import brier_score, expected_calibration_error
from rpcp.evaluation.ranking import roc_auc

__all__ = ["ConceptMetrics", "binary_f1", "concept_metrics"]


def binary_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """F1 for one binary concept; ``nan`` if the concept never occurs nor fires."""
    y_true = np.asarray(y_true).ravel() > 0.5
    y_pred = np.asarray(y_pred).ravel() > 0.5
    true_positives = float(np.sum(y_true & y_pred))
    predicted_positives = float(np.sum(y_pred))
    actual_positives = float(np.sum(y_true))
    if actual_positives == 0 and predicted_positives == 0:
        return float("nan")
    if true_positives == 0:
        return 0.0
    precision = true_positives / predicted_positives
    recall = true_positives / actual_positives
    return float(2 * precision * recall / (precision + recall))


@dataclass(slots=True)
class ConceptMetrics:
    """Concept-level evaluation results.

    Attributes:
        macro_f1: Mean of per-concept F1 over concepts with support.
        per_concept_f1: ``(M,)`` F1 per concept (``nan`` where undefined).
        macro_auroc: Mean per-concept AUROC.
        per_concept_auroc: ``(M,)`` AUROC per concept.
        ece: Expected calibration error over all (image, concept) pairs.
        brier: Brier score over all (image, concept) pairs.
        accuracy: Element-wise concept accuracy.
        per_class_concept_f1: ``(M, K)`` F1 computed within each class, used for
            the reliability-vs-concept-quality correlation of plan 6.6.
    """

    macro_f1: float
    per_concept_f1: np.ndarray
    macro_auroc: float
    per_concept_auroc: np.ndarray
    ece: float
    brier: float
    accuracy: float
    per_class_concept_f1: np.ndarray | None = None
    concept_names: list[str] = field(default_factory=list)

    def as_dict(self, prefix: str = "concept/") -> dict[str, float]:
        return {
            f"{prefix}macro_f1": self.macro_f1,
            f"{prefix}macro_auroc": self.macro_auroc,
            f"{prefix}ece": self.ece,
            f"{prefix}brier": self.brier,
            f"{prefix}accuracy": self.accuracy,
        }

    def per_concept_table(self) -> dict[str, dict[str, float]]:
        names = self.concept_names or [f"concept_{m}" for m in range(len(self.per_concept_f1))]
        return {
            name: {"f1": float(self.per_concept_f1[m]), "auroc": float(self.per_concept_auroc[m])}
            for m, name in enumerate(names)
        }


def concept_metrics(
    probabilities: np.ndarray,
    targets: np.ndarray,
    *,
    labels: np.ndarray | None = None,
    n_classes: int | None = None,
    threshold: float = 0.5,
    n_bins: int = 15,
    concept_names: list[str] | None = None,
) -> ConceptMetrics:
    """Compute all concept metrics.

    Args:
        probabilities: ``(N, M)`` predicted concept probabilities.
        targets: ``(N, M)`` binary ground-truth concepts (evaluation only).
        labels: ``(N,)`` class labels, needed for ``per_class_concept_f1``.
        n_classes: ``K`` (inferred from ``labels`` when omitted).
        threshold: Binarisation threshold for F1/accuracy.
        n_bins: Calibration bins.
        concept_names: Optional names for reporting.
    """
    probabilities = np.asarray(probabilities, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    if probabilities.shape != targets.shape:
        raise ValueError(
            f"probabilities {probabilities.shape} and targets {targets.shape} must match"
        )

    n_concepts = probabilities.shape[1]
    predictions = probabilities >= threshold

    f1_scores = np.array(
        [binary_f1(targets[:, m], predictions[:, m]) for m in range(n_concepts)]
    )
    auroc_scores = np.array(
        [roc_auc(targets[:, m], probabilities[:, m]) for m in range(n_concepts)]
    )

    per_class: np.ndarray | None = None
    if labels is not None:
        labels = np.asarray(labels).ravel()
        n_classes = int(n_classes or labels.max() + 1)
        per_class = np.full((n_concepts, n_classes), np.nan)
        for y in range(n_classes):
            mask = labels == y
            if not mask.any():
                continue
            for m in range(n_concepts):
                per_class[m, y] = binary_f1(targets[mask, m], predictions[mask, m])

    return ConceptMetrics(
        macro_f1=float(np.nanmean(f1_scores)) if np.isfinite(f1_scores).any() else float("nan"),
        per_concept_f1=f1_scores,
        macro_auroc=(
            float(np.nanmean(auroc_scores)) if np.isfinite(auroc_scores).any() else float("nan")
        ),
        per_concept_auroc=auroc_scores,
        ece=expected_calibration_error(probabilities.ravel(), targets.ravel(), n_bins=n_bins),
        brier=brier_score(probabilities.ravel(), targets.ravel()),
        accuracy=float(np.mean(predictions == (targets > 0.5))),
        per_class_concept_f1=per_class,
        concept_names=list(concept_names or []),
    )
