"""Figure 3 and friends: reliability heatmaps (plan section 7).

Panels: the true corruption mask, the learned reliability matrix, and the
absolute prior error.  If reliability works, panel 2 should look like the
inverse of panels 1 and 3.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from rpcp.evaluation.reliability_metrics import reliability_auprc, reliability_auroc
from rpcp.plotting.style import save_figure, use_paper_style

__all__ = [
    "plot_prior_table",
    "plot_reliability_evolution",
    "plot_reliability_heatmaps",
]


def _annotate(
    axis: Any,
    matrix: np.ndarray,
    *,
    fmt: str = "{:.2f}",
    threshold: float = 0.5,
) -> None:
    if matrix.size > 120:  # too dense to annotate legibly
        return
    for (row, col), value in np.ndenumerate(matrix):
        axis.text(
            col,
            row,
            fmt.format(value),
            ha="center",
            va="center",
            fontsize=7,
            color="white" if value > threshold else "black",
        )


def _set_labels(
    axis: Any,
    concept_names: Sequence[str] | None,
    class_names: Sequence[str] | None,
    shape: tuple[int, int],
) -> None:
    n_concepts, n_classes = shape
    axis.set_xticks(range(n_classes))
    axis.set_yticks(range(n_concepts))
    axis.set_xticklabels(
        list(class_names) if class_names else [f"y{k}" for k in range(n_classes)],
        rotation=45,
        ha="right",
    )
    axis.set_yticklabels(
        list(concept_names) if concept_names else [f"c{m}" for m in range(n_concepts)]
    )


def plot_reliability_heatmaps(
    reliability: np.ndarray,
    *,
    corruption_mask: np.ndarray | None = None,
    prior_error: np.ndarray | None = None,
    concept_names: Sequence[str] | None = None,
    class_names: Sequence[str] | None = None,
    title: str = "Figure 3: reliability recovers the corruption mask",
    path: str | Path | None = None,
) -> Any:
    """Side-by-side heatmaps of ``s_true``, ``r`` and ``|Pi_tilde - Pi_star|``.

    Args:
        reliability: ``(M, K)`` learned reliability.
        corruption_mask: ``(M, K)`` ``True``/1 where the entry is corrupted.
        prior_error: ``(M, K)`` absolute prior error.
        concept_names / class_names: Axis labels.
        title: Figure title; the detection AUROC/AUPRC are appended when the
            mask is available.
        path: If given, the figure is saved there.
    """
    use_paper_style()
    panels: list[tuple[str, np.ndarray, str, tuple[float, float]]] = []
    if corruption_mask is not None:
        panels.append(
            ("true corruption mask $1-s$", np.asarray(corruption_mask, dtype=float), "Reds", (0, 1))
        )
    panels.append(
        ("learned reliability $r$", np.asarray(reliability, dtype=float), "viridis", (0, 1))
    )
    if prior_error is not None:
        error = np.asarray(prior_error, dtype=float)
        vmax = max(0.5, float(error.max()))
        panels.append((r"prior error $|\tilde\Pi-\Pi^\star|$", error, "magma", (0, vmax)))

    figure, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 4.0), squeeze=False)
    for axis, (panel_title, matrix, cmap, (vmin, vmax)) in zip(axes[0], panels, strict=True):
        image = axis.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        axis.set_title(panel_title)
        _set_labels(axis, concept_names, class_names, matrix.shape)
        _annotate(axis, matrix, threshold=(vmin + vmax) / 2)
        axis.grid(False)
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

    if corruption_mask is not None:
        auroc = reliability_auroc(np.asarray(reliability), np.asarray(corruption_mask))
        auprc = reliability_auprc(np.asarray(reliability), np.asarray(corruption_mask))
        title = f"{title}\ndetection AUROC={auroc:.3f}, AUPRC={auprc:.3f}"
    figure.suptitle(title)
    figure.tight_layout()

    if path is not None:
        save_figure(figure, path, close=False)
    return figure


def plot_prior_table(
    prior: np.ndarray,
    *,
    concept_names: Sequence[str] | None = None,
    class_names: Sequence[str] | None = None,
    title: str = "Class-level concept prior",
    path: str | Path | None = None,
) -> Any:
    """Heatmap of a single prior table."""
    use_paper_style()
    figure, axis = plt.subplots(figsize=(4.5, 4.0))
    image = axis.imshow(
        np.asarray(prior, dtype=float), cmap="viridis", vmin=0, vmax=1, aspect="auto"
    )
    axis.set_title(title)
    _set_labels(axis, concept_names, class_names, np.asarray(prior).shape)
    _annotate(axis, np.asarray(prior, dtype=float))
    axis.grid(False)
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    if path is not None:
        save_figure(figure, path, close=False)
    return figure


def plot_reliability_evolution(
    history: np.ndarray,
    *,
    corruption_mask: np.ndarray | None = None,
    title: str = "Reliability trajectory",
    path: str | Path | None = None,
) -> Any:
    """Mean reliability of clean vs corrupted entries over reliability updates.

    This is the diagnostic for Risk 3 (collapse to zero) and Risk 4 (staying
    high on corrupted entries) of the plan's risk register.
    """
    use_paper_style()
    history = np.asarray(history, dtype=float)  # (T, M, K)
    steps = np.arange(history.shape[0])

    figure, axis = plt.subplots(figsize=(5.0, 3.4))
    if corruption_mask is None:
        axis.plot(steps, history.reshape(len(steps), -1).mean(axis=1), label="mean $r$")
    else:
        mask = np.asarray(corruption_mask, dtype=bool).ravel()
        flat = history.reshape(len(steps), -1)
        axis.plot(steps, flat[:, ~mask].mean(axis=1), label="clean entries")
        axis.plot(steps, flat[:, mask].mean(axis=1), label="corrupted entries")
        axis.fill_between(
            steps,
            flat[:, ~mask].mean(axis=1),
            flat[:, mask].mean(axis=1),
            alpha=0.15,
            label="separation",
        )
    axis.set_xlabel("reliability update")
    axis.set_ylabel("mean reliability $r$")
    axis.set_ylim(0, 1)
    axis.set_title(title)
    axis.legend()
    figure.tight_layout()
    if path is not None:
        save_figure(figure, path, close=False)
    return figure
