"""Calibration metrics for concept and reliability probabilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["CalibrationCurve", "brier_score", "expected_calibration_error", "reliability_curve"]


@dataclass(slots=True)
class CalibrationCurve:
    """Binned calibration curve (the datasets behind a reliability diagram)."""

    bin_edges: np.ndarray
    bin_confidence: np.ndarray
    bin_accuracy: np.ndarray
    bin_count: np.ndarray

    @property
    def gap(self) -> np.ndarray:
        return np.abs(self.bin_accuracy - self.bin_confidence)


def reliability_curve(
    probabilities: np.ndarray,
    targets: np.ndarray,
    *,
    n_bins: int = 15,
) -> CalibrationCurve:
    """Equal-width binning of predicted probabilities against outcomes."""
    probabilities = np.asarray(probabilities, dtype=np.float64).ravel()
    targets = np.asarray(targets, dtype=np.float64).ravel()
    if probabilities.shape != targets.shape:
        raise ValueError("probabilities and targets must have the same shape")

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    indices = np.clip(np.digitize(probabilities, edges[1:-1], right=False), 0, n_bins - 1)

    confidence = np.full(n_bins, np.nan)
    accuracy = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins, dtype=np.int64)
    for b in range(n_bins):
        mask = indices == b
        counts[b] = int(mask.sum())
        if counts[b]:
            confidence[b] = probabilities[mask].mean()
            accuracy[b] = targets[mask].mean()
    return CalibrationCurve(edges, confidence, accuracy, counts)


def expected_calibration_error(
    probabilities: np.ndarray,
    targets: np.ndarray,
    *,
    n_bins: int = 15,
) -> float:
    """ECE: support-weighted mean gap between confidence and empirical rate.

    For concept predictors this is computed over *all* (image, concept) pairs,
    which is the quantity a clinician cares about: "when the model says 0.8, is
    the concept present 80% of the time?"
    """
    curve = reliability_curve(probabilities, targets, n_bins=n_bins)
    total = curve.bin_count.sum()
    if total == 0:
        return float("nan")
    valid = curve.bin_count > 0
    weights = curve.bin_count[valid] / total
    return float((weights * curve.gap[valid]).sum())


def brier_score(probabilities: np.ndarray, targets: np.ndarray) -> float:
    """Mean squared error between predicted probabilities and binary outcomes."""
    probabilities = np.asarray(probabilities, dtype=np.float64).ravel()
    targets = np.asarray(targets, dtype=np.float64).ravel()
    if probabilities.size == 0:
        return float("nan")
    return float(np.mean((probabilities - targets) ** 2))
