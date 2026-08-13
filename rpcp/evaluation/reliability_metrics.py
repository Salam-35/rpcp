"""Reliability-audit metrics (plan 6.6, "Reliability").

The central question of the paper is *not* whether the loss goes down but
whether the learned ``r[m, y]`` actually knows which prior entries are wrong.
These metrics answer that directly:

* corruption-detection AUROC / AUPRC against the known mask ``s_true``;
* Spearman correlation between ``r`` and the true prior error;
* Spearman correlation between ``r`` and per-(concept, class) concept F1;
* calibration of ``r`` read as ``P(entry is clean)``.

A high AUROC here with a low correlation to prior error is a warning sign that
``r`` is tracking something else (e.g. model confidence) -- exactly the
self-confirmation risk of plan 13, Risk 1.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from rpcp.evaluation.calibration import expected_calibration_error
from rpcp.evaluation.ranking import average_precision, roc_auc, spearman

__all__ = [
    "ReliabilityMetrics",
    "detection_at_threshold",
    "reliability_auprc",
    "reliability_auroc",
    "reliability_metrics",
    "reliability_spearman",
]


def _to_numpy(value: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _corruption_labels(mask: torch.Tensor | np.ndarray) -> np.ndarray:
    """``1`` where the entry is corrupted.

    Accepts either ``clean_mask`` semantics inverted by the caller or an already
    corruption-valued mask; the convention in this project is that the argument
    named ``corruption_mask`` is ``True`` for corrupted entries.
    """
    return (_to_numpy(mask).ravel() > 0.5).astype(np.float64)


def reliability_auroc(
    reliability: torch.Tensor | np.ndarray,
    corruption_mask: torch.Tensor | np.ndarray,
) -> float:
    """AUROC for detecting corrupted entries, scoring them by ``1 - r``."""
    scores = 1.0 - _to_numpy(reliability).ravel()
    return roc_auc(_corruption_labels(corruption_mask), scores)


def reliability_auprc(
    reliability: torch.Tensor | np.ndarray,
    corruption_mask: torch.Tensor | np.ndarray,
) -> float:
    """AUPRC for detecting corrupted entries (the rare-positive view)."""
    scores = 1.0 - _to_numpy(reliability).ravel()
    return average_precision(_corruption_labels(corruption_mask), scores)


def reliability_spearman(
    reliability: torch.Tensor | np.ndarray,
    quantity: torch.Tensor | np.ndarray,
) -> float:
    """Spearman correlation between ``r`` and any per-entry quantity.

    Against ``|Pi_tilde - Pi_star|`` a *negative* value is the desired outcome
    (more error -> less trust); against per-entry concept F1 a *positive* value
    is desired (more trust -> better concepts).
    """
    return spearman(_to_numpy(reliability).ravel(), _to_numpy(quantity).ravel())


def detection_at_threshold(
    reliability: torch.Tensor | np.ndarray,
    corruption_mask: torch.Tensor | np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Precision/recall/F1 of flagging entries with ``r < threshold``."""
    flagged = _to_numpy(reliability).ravel() < threshold
    corrupted = _corruption_labels(corruption_mask) > 0.5
    true_positives = float(np.sum(flagged & corrupted))
    precision = true_positives / max(float(flagged.sum()), 1e-9)
    recall = true_positives / max(float(corrupted.sum()), 1e-9)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "flagged_fraction": float(flagged.mean()),
    }


@dataclass(slots=True)
class ReliabilityMetrics:
    """Everything reported in the reliability-audit table."""

    auroc: float = float("nan")
    auprc: float = float("nan")
    spearman_prior_error: float = float("nan")
    spearman_concept_f1: float = float("nan")
    spearman_rater_agreement: float = float("nan")
    ece: float = float("nan")
    mean_reliability_clean: float = float("nan")
    mean_reliability_corrupt: float = float("nan")
    detection: dict[str, float] = field(default_factory=dict)

    @property
    def separation(self) -> float:
        """Mean ``r`` on clean entries minus mean ``r`` on corrupted entries."""
        return self.mean_reliability_clean - self.mean_reliability_corrupt

    def as_dict(self, prefix: str = "reliability/") -> dict[str, float]:
        out = {
            f"{prefix}auroc": self.auroc,
            f"{prefix}auprc": self.auprc,
            f"{prefix}spearman_prior_error": self.spearman_prior_error,
            f"{prefix}spearman_concept_f1": self.spearman_concept_f1,
            f"{prefix}spearman_rater_agreement": self.spearman_rater_agreement,
            f"{prefix}ece": self.ece,
            f"{prefix}mean_clean": self.mean_reliability_clean,
            f"{prefix}mean_corrupt": self.mean_reliability_corrupt,
            f"{prefix}separation": self.separation,
        }
        out.update({f"{prefix}detect_{k}": v for k, v in self.detection.items()})
        return out


def reliability_metrics(
    reliability: torch.Tensor | np.ndarray,
    *,
    corruption_mask: torch.Tensor | np.ndarray | None = None,
    prior_error: torch.Tensor | np.ndarray | None = None,
    per_entry_concept_f1: torch.Tensor | np.ndarray | None = None,
    rater_agreement: torch.Tensor | np.ndarray | None = None,
    threshold: float = 0.5,
    n_bins: int = 10,
) -> ReliabilityMetrics:
    """Audit the learned reliability matrix against every available reference.

    Args:
        reliability: ``(M, K)`` learned reliability.
        corruption_mask: ``(M, K)`` ``True`` where the prior entry is corrupted.
        prior_error: ``(M, K)`` ``|Pi_tilde - Pi_star|``.
        per_entry_concept_f1: ``(M, K)`` concept F1 within each class.
        rater_agreement: ``(M, K)`` natural reliability signal (Mode D).
        threshold: Operating point for the detection triplet.
        n_bins: Calibration bins.
    """
    values = _to_numpy(reliability).ravel()
    metrics = ReliabilityMetrics()

    if corruption_mask is not None:
        labels = _corruption_labels(corruption_mask)
        metrics.auroc = reliability_auroc(reliability, corruption_mask)
        metrics.auprc = reliability_auprc(reliability, corruption_mask)
        metrics.detection = detection_at_threshold(reliability, corruption_mask, threshold)
        clean = labels < 0.5
        metrics.mean_reliability_clean = (
            float(values[clean].mean()) if clean.any() else float("nan")
        )
        metrics.mean_reliability_corrupt = (
            float(values[~clean].mean()) if (~clean).any() else float("nan")
        )
        # r is read as P(entry is clean).
        metrics.ece = expected_calibration_error(values, 1.0 - labels, n_bins=n_bins)

    if prior_error is not None:
        metrics.spearman_prior_error = reliability_spearman(reliability, prior_error)
    if per_entry_concept_f1 is not None:
        metrics.spearman_concept_f1 = reliability_spearman(reliability, per_entry_concept_f1)
    if rater_agreement is not None:
        metrics.spearman_rater_agreement = reliability_spearman(reliability, rater_agreement)

    return metrics
