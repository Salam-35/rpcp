"""Figures 2, 4 and 5: robustness, identifiability and audit-budget curves."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from rpcp.plotting.style import METHOD_COLOURS, METHOD_ORDER, save_figure, use_paper_style

__all__ = [
    "Curve",
    "plot_audit_budget",
    "plot_corruption_robustness",
    "plot_delta_curve",
    "plot_curves",
]


@dataclass(slots=True)
class Curve:
    """One method's curve, optionally with error bars over seeds."""

    x: Sequence[float]
    y: Sequence[float]
    yerr: Sequence[float] | None = None
    label: str | None = None

    def as_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        order = np.argsort(np.asarray(self.x, dtype=float))
        x = np.asarray(self.x, dtype=float)[order]
        y = np.asarray(self.y, dtype=float)[order]
        yerr = None if self.yerr is None else np.asarray(self.yerr, dtype=float)[order]
        return x, y, yerr


def plot_curves(
    curves: Mapping[str, Curve],
    *,
    xlabel: str,
    ylabel: str,
    title: str = "",
    path: str | Path | None = None,
    figsize: tuple[float, float] = (5.2, 3.6),
    ylim: tuple[float, float] | None = None,
    annotate_last: bool = False,
) -> Any:
    """Generic multi-method curve plot with consistent colours and error bars."""
    use_paper_style()
    figure, axis = plt.subplots(figsize=figsize)

    ordered = sorted(
        curves.items(),
        key=lambda item: METHOD_ORDER.index(item[0]) if item[0] in METHOD_ORDER else 99,
    )
    for name, curve in ordered:
        x, y, yerr = curve.as_arrays()
        colour = METHOD_COLOURS.get(name)
        label = curve.label or name
        if yerr is not None:
            axis.errorbar(x, y, yerr=yerr, label=label, color=colour, marker="o", capsize=3)
            axis.fill_between(x, y - yerr, y + yerr, color=colour, alpha=0.12)
        else:
            axis.plot(x, y, label=label, color=colour, marker="o")
        if annotate_last and len(x):
            axis.annotate(
                f"{y[-1]:.2f}",
                (x[-1], y[-1]),
                textcoords="offset points",
                xytext=(4, 2),
                fontsize=8,
                color=colour,
            )

    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    if title:
        axis.set_title(title)
    if ylim is not None:
        axis.set_ylim(*ylim)
    axis.legend(frameon=False)
    figure.tight_layout()
    if path is not None:
        save_figure(figure, path, close=False)
    return figure


def plot_corruption_robustness(
    curves: Mapping[str, Curve],
    *,
    metric: str = "macro concept F1",
    corruption_mode: str = "",
    path: str | Path | None = None,
) -> Any:
    """Figure 2: metric vs corruption strength ``alpha``.

    Expected reading: R-PCP degrades more slowly than PCP and approaches the
    oracle-reliability curve when the reliability evidence is strong.
    """
    suffix = f" ({corruption_mode})" if corruption_mode else ""
    return plot_curves(
        curves,
        xlabel=r"corruption strength $\alpha$",
        ylabel=metric,
        title=f"Figure 2: robustness to noisy priors{suffix}",
        path=path,
    )


def plot_delta_curve(
    curves: Mapping[str, Curve],
    *,
    metric: str = "macro concept F1",
    path: str | Path | None = None,
    show_class_metric: Mapping[str, Curve] | None = None,
) -> Any:
    """Figure 4: metric vs prior separation ``Delta``.

    When ``show_class_metric`` is supplied it is drawn with dashed lines on a
    twin axis: the point of the figure is that class F1 can stay high (via
    non-concept shortcuts) while concept F1 collapses as ``Delta -> 0``.
    """
    use_paper_style()
    figure, axis = plt.subplots(figsize=(5.4, 3.6))

    for name, curve in curves.items():
        x, y, yerr = curve.as_arrays()
        colour = METHOD_COLOURS.get(name)
        axis.errorbar(
            x, y, yerr=yerr, label=curve.label or name, color=colour, marker="o", capsize=3
        )

    axis.set_xlabel(r"prior separation $\Delta=\min_{y\neq y'}\|\Pi[:,y]-\Pi[:,y']\|_2$")
    axis.set_ylabel(metric)
    axis.set_title("Figure 4: prior separation and identifiability")

    if show_class_metric:
        twin = axis.twinx()
        twin.grid(False)
        for name, curve in show_class_metric.items():
            x, y, _ = curve.as_arrays()
            twin.plot(
                x,
                y,
                linestyle="--",
                marker="s",
                alpha=0.7,
                color=METHOD_COLOURS.get(name),
                label=f"{curve.label or name} (class F1)",
            )
        twin.set_ylabel("macro class F1")
        handles, labels = axis.get_legend_handles_labels()
        twin_handles, twin_labels = twin.get_legend_handles_labels()
        axis.legend(handles + twin_handles, labels + twin_labels, frameon=False, fontsize=8)
    else:
        axis.legend(frameon=False)

    figure.tight_layout()
    if path is not None:
        save_figure(figure, path, close=False)
    return figure


def plot_audit_budget(
    concept_f1: Curve,
    reliability_auroc: Curve,
    *,
    path: str | Path | None = None,
) -> Any:
    """Figure 5: concept F1 and reliability AUROC vs audit-label budget."""
    use_paper_style()
    figure, axis = plt.subplots(figsize=(5.4, 3.6))

    x, y, yerr = concept_f1.as_arrays()
    axis.errorbar(
        x, y, yerr=yerr, color=METHOD_COLOURS["r-pcp-audit"], marker="o", capsize=3,
        label=concept_f1.label or "concept macro F1",
    )
    axis.set_xlabel("% of training images with concept audit labels")
    axis.set_ylabel("macro concept F1")

    twin = axis.twinx()
    twin.grid(False)
    x2, y2, yerr2 = reliability_auroc.as_arrays()
    twin.errorbar(
        x2, y2, yerr=yerr2, color=METHOD_COLOURS["oracle"], marker="s", linestyle="--", capsize=3,
        label=reliability_auroc.label or "reliability AUROC",
    )
    twin.set_ylabel("reliability AUROC")
    twin.axhline(0.7, color="grey", linewidth=1, linestyle=":", alpha=0.8)

    handles, labels = axis.get_legend_handles_labels()
    twin_handles, twin_labels = twin.get_legend_handles_labels()
    axis.legend(handles + twin_handles, labels + twin_labels, frameon=False, loc="lower right")
    axis.set_title("Figure 5: a small audit budget buys reliability")

    figure.tight_layout()
    if path is not None:
        save_figure(figure, path, close=False)
    return figure
