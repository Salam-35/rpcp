"""Null and identifiability-ceiling baselines for concept macro-F1.

Concept macro-F1 on its own is not interpretable: 0.43 could mean "the model
is broken" or "the model is doing about as well as anything trained from
class-level priors alone possibly could." Two reference points turn it into
an interpretable number:

* :func:`null_concept_baseline` -- the best a model that ignores the image
  entirely can do: predict each concept's single global majority value for
  every example. Anything a trained model should comfortably beat.
* :func:`identifiability_ceiling` -- the best *any* model restricted to
  matching class-conditional first moments could ever do: give every image
  of class ``y`` the identical concept probability
  ``mean_{i: y_i=y} targets[i, m]`` (computed from the evaluation-only
  concept labels themselves, not the training prior, so it reflects the
  actual achievable ceiling on this data), thresholded the same way as the
  model. This is exactly the quantity the paper's identifiability analysis
  (``L_prior``/``L_match`` are both class-level-only) says a prior-supervised
  model cannot exceed without per-image concept supervision -- see
  ``concept-bottleneck-diagnosis.md``. A trained macro-F1 below the null
  floor indicates a bug; one at or above the ceiling indicates the model has
  found information ``L_prior``/``L_match`` alone should not have been able
  to give it (e.g. concept-label leakage) and is worth auditing for that.

Neither baseline requires training a model: both are computed directly from
the evaluation split's ground-truth ``(N, M)`` concept matrix and ``(N,)``
labels, so they cost nothing to report alongside every real run.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rpcp.evaluation.concept_metrics import binary_f1

__all__ = ["BaselineMetrics", "concept_baselines", "identifiability_ceiling", "null_concept_baseline"]


def null_concept_baseline(
    targets: np.ndarray,
    *,
    threshold: float = 0.5,
) -> tuple[float, np.ndarray]:
    """Macro-F1 of predicting each concept's global majority value everywhere.

    Args:
        targets: ``(N, M)`` binary ground-truth concepts.
        threshold: Majority-vote threshold (``>=`` counts as "concept present").

    Returns:
        ``(macro_f1, per_concept_f1)``.
    """
    targets = np.asarray(targets, dtype=np.float64)
    prevalence = targets.mean(axis=0)
    majority = np.broadcast_to(prevalence >= threshold, targets.shape).astype(np.float64)
    per_concept = np.array(
        [binary_f1(targets[:, m], majority[:, m]) for m in range(targets.shape[1])]
    )
    macro = float(np.nanmean(per_concept)) if np.isfinite(per_concept).any() else float("nan")
    return macro, per_concept


def identifiability_ceiling(
    targets: np.ndarray,
    labels: np.ndarray,
    *,
    n_classes: int | None = None,
    threshold: float = 0.5,
) -> tuple[float, np.ndarray]:
    """Macro-F1 achievable by predicting the true class-conditional mean for every image.

    This is the ceiling any model can reach using *only* ``E[c_m | y]`` -- the
    quantity ``L_prior`` actually constrains. Every image of class ``y`` is
    given the identical prediction ``mean_{i: y_i=y} targets[i, m]``,
    thresholded, exactly like a real model's probabilities would be.

    Args:
        targets: ``(N, M)`` binary ground-truth concepts.
        labels: ``(N,)`` integer class labels.
        n_classes: ``K`` (inferred from ``labels`` when omitted).
        threshold: Binarisation threshold, matching the one used for the
            trained model's concept metrics so the comparison is apples-to-apples.

    Returns:
        ``(macro_f1, per_concept_f1)``.
    """
    targets = np.asarray(targets, dtype=np.float64)
    labels = np.asarray(labels).ravel()
    n_classes = int(n_classes if n_classes is not None else labels.max() + 1)

    class_mean_prediction = np.empty_like(targets)
    for y in range(n_classes):
        mask = labels == y
        if not mask.any():
            continue
        class_mean_prediction[mask] = targets[mask].mean(axis=0, keepdims=True) >= threshold

    n_concepts = targets.shape[1]
    per_concept = np.array(
        [binary_f1(targets[:, m], class_mean_prediction[:, m]) for m in range(n_concepts)]
    )
    macro = float(np.nanmean(per_concept)) if np.isfinite(per_concept).any() else float("nan")
    return macro, per_concept


@dataclass(slots=True)
class BaselineMetrics:
    """Null floor and identifiability ceiling for one evaluation split.

    Attributes:
        null_macro_f1: See :func:`null_concept_baseline`.
        ceiling_macro_f1: See :func:`identifiability_ceiling`.
        null_per_concept_f1: ``(M,)``.
        ceiling_per_concept_f1: ``(M,)``.
    """

    null_macro_f1: float
    ceiling_macro_f1: float
    null_per_concept_f1: np.ndarray
    ceiling_per_concept_f1: np.ndarray

    def progress(self, model_macro_f1: float) -> float:
        """Where ``model_macro_f1`` sits between the null floor (0) and ceiling (1).

        Values outside ``[0, 1]`` are meaningful, not clamped: below 0 means the
        trained model is *worse* than ignoring the image, above 1 means it beat
        the class-conditional-mean ceiling (worth auditing for leakage; see
        module docstring). Returns ``nan`` if the floor and ceiling coincide.
        """
        span = self.ceiling_macro_f1 - self.null_macro_f1
        if not np.isfinite(span) or abs(span) < 1e-12:
            return float("nan")
        return float((model_macro_f1 - self.null_macro_f1) / span)

    def as_dict(self, prefix: str = "concept/") -> dict[str, float]:
        return {
            f"{prefix}null_macro_f1": self.null_macro_f1,
            f"{prefix}ceiling_macro_f1": self.ceiling_macro_f1,
        }


def concept_baselines(
    targets: np.ndarray,
    labels: np.ndarray,
    *,
    n_classes: int | None = None,
    threshold: float = 0.5,
) -> BaselineMetrics:
    """Compute both baselines at once (see module docstring)."""
    null_macro, null_per_concept = null_concept_baseline(targets, threshold=threshold)
    ceiling_macro, ceiling_per_concept = identifiability_ceiling(
        targets, labels, n_classes=n_classes, threshold=threshold
    )
    return BaselineMetrics(
        null_macro_f1=null_macro,
        ceiling_macro_f1=ceiling_macro,
        null_per_concept_f1=null_per_concept,
        ceiling_per_concept_f1=ceiling_per_concept,
    )
