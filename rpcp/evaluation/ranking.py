"""Dependency-free ranking statistics (AUROC, AUPRC, Spearman).

Implemented in NumPy rather than pulled from scikit-learn so that the metric
definitions are visible and tie handling is explicit: reliability scores are
frequently tied (whole rows of a prior table can share a value), and different
tie conventions change reported AUROCs by a surprising amount.
"""

from __future__ import annotations

import numpy as np

__all__ = ["average_precision", "pearson", "rank_data", "roc_auc", "spearman"]


def _as_1d(values: np.ndarray | list[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).ravel()
    return array


def rank_data(values: np.ndarray) -> np.ndarray:
    """Average ranks (1-based), ties share their mean rank."""
    values = _as_1d(values)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(1, len(values) + 1, dtype=np.float64)

    sorted_values = values[order]
    start = 0
    for i in range(1, len(values) + 1):
        if i == len(values) or sorted_values[i] != sorted_values[start]:
            if i - start > 1:
                ranks[order[start:i]] = ranks[order[start:i]].mean()
            start = i
    return ranks


def roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Area under the ROC curve via the tie-corrected Mann-Whitney statistic.

    Returns ``nan`` when one of the two classes is absent (AUROC undefined).
    """
    y_true = _as_1d(y_true)
    y_score = _as_1d(y_score)
    if y_true.shape != y_score.shape:
        raise ValueError("y_true and y_score must have the same shape")

    positives = y_true > 0.5
    n_pos = int(positives.sum())
    n_neg = int(len(y_true) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    ranks = rank_data(y_score)
    return float((ranks[positives].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Area under the precision-recall curve (AUPRC), step-wise interpolation.

    Equivalent to ``sklearn.metrics.average_precision_score``.  Returns ``nan``
    when there are no positives.
    """
    y_true = _as_1d(y_true)
    y_score = _as_1d(y_score)
    n_pos = int((y_true > 0.5).sum())
    if n_pos == 0:
        return float("nan")

    order = np.argsort(-y_score, kind="mergesort")
    labels = (y_true[order] > 0.5).astype(np.float64)
    scores = y_score[order]

    true_positives = np.cumsum(labels)
    predicted = np.arange(1, len(labels) + 1, dtype=np.float64)
    precision = true_positives / predicted
    recall = true_positives / n_pos

    # Only count the last position of each group of tied scores.
    keep = np.ones(len(scores), dtype=bool)
    keep[:-1] = scores[1:] != scores[:-1]

    precision, recall = precision[keep], recall[keep]
    recall_delta = np.diff(np.concatenate([[0.0], recall]))
    return float((precision * recall_delta).sum())


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation; ``nan`` if either input is constant."""
    x, y = _as_1d(x), _as_1d(y)
    if len(x) < 2:
        return float("nan")
    x_centered, y_centered = x - x.mean(), y - y.mean()
    denominator = np.sqrt((x_centered**2).sum() * (y_centered**2).sum())
    if denominator < 1e-12:
        return float("nan")
    return float((x_centered * y_centered).sum() / denominator)


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation (Pearson correlation of average ranks)."""
    x, y = _as_1d(x), _as_1d(y)
    finite = np.isfinite(x) & np.isfinite(y)
    if finite.sum() < 2:
        return float("nan")
    return pearson(rank_data(x[finite]), rank_data(y[finite]))
