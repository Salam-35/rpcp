"""Class-prediction metrics (plan 6.6)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from rpcp.evaluation.concept_metrics import binary_f1
from rpcp.evaluation.ranking import roc_auc

__all__ = ["ClassMetrics", "class_metrics", "confusion_matrix"]


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    matrix = np.zeros((n_classes, n_classes), dtype=np.int64)
    for true, pred in zip(np.asarray(y_true).ravel(), np.asarray(y_pred).ravel(), strict=True):
        matrix[int(true), int(pred)] += 1
    return matrix


@dataclass(slots=True)
class ClassMetrics:
    accuracy: float
    macro_f1: float
    per_class_f1: np.ndarray
    macro_auroc: float
    balanced_accuracy: float
    confusion: np.ndarray
    class_names: list[str] = field(default_factory=list)

    def as_dict(self, prefix: str = "class/") -> dict[str, float]:
        return {
            f"{prefix}accuracy": self.accuracy,
            f"{prefix}macro_f1": self.macro_f1,
            f"{prefix}macro_auroc": self.macro_auroc,
            f"{prefix}balanced_accuracy": self.balanced_accuracy,
        }


def class_metrics(
    probabilities: np.ndarray,
    labels: np.ndarray,
    *,
    class_names: list[str] | None = None,
) -> ClassMetrics:
    """Accuracy, macro-F1, one-vs-rest macro AUROC and the confusion matrix.

    Args:
        probabilities: ``(N, K)`` class probabilities (softmax outputs).
        labels: ``(N,)`` integer labels.
        class_names: Optional names for reporting.
    """
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels).ravel().astype(np.int64)
    n_classes = probabilities.shape[1]
    predictions = probabilities.argmax(axis=1)

    per_class = np.array(
        [binary_f1(labels == k, predictions == k) for k in range(n_classes)]
    )
    aurocs = np.array(
        [roc_auc((labels == k).astype(float), probabilities[:, k]) for k in range(n_classes)]
    )
    recalls = [
        float(np.mean(predictions[labels == k] == k))
        for k in range(n_classes)
        if (labels == k).any()
    ]

    return ClassMetrics(
        accuracy=float(np.mean(predictions == labels)),
        macro_f1=float(np.nanmean(per_class)),
        per_class_f1=per_class,
        macro_auroc=float(np.nanmean(aurocs)) if np.isfinite(aurocs).any() else float("nan"),
        balanced_accuracy=float(np.mean(recalls)) if recalls else float("nan"),
        confusion=confusion_matrix(labels, predictions, n_classes),
        class_names=list(class_names or []),
    )
