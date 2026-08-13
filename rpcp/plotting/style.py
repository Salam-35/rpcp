"""Shared matplotlib styling for the paper figures."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

__all__ = ["METHOD_COLOURS", "METHOD_ORDER", "save_figure", "use_paper_style"]

#: Consistent colours across Figures 2, 4 and 5.
METHOD_COLOURS: dict[str, str] = {
    "pcp": "#4C72B0",
    "r-pcp": "#DD8452",
    "r-pcp-audit": "#55A868",
    "r-pcp-multisource": "#8172B3",
    "oracle": "#937860",
    "supervised-cbm": "#64B5CD",
    "blackbox": "#8C8C8C",
}

METHOD_ORDER: tuple[str, ...] = (
    "blackbox",
    "supervised-cbm",
    "pcp",
    "r-pcp",
    "r-pcp-multisource",
    "r-pcp-audit",
    "oracle",
)


def use_paper_style(font_scale: float = 1.0) -> None:
    """Apply a clean, publication-ready rcParams set."""
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "font.size": 10 * font_scale,
            "axes.titlesize": 11 * font_scale,
            "axes.labelsize": 10 * font_scale,
            "legend.fontsize": 9 * font_scale,
            "xtick.labelsize": 9 * font_scale,
            "ytick.labelsize": 9 * font_scale,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
            "lines.linewidth": 1.8,
            "lines.markersize": 5,
            "figure.autolayout": False,
        }
    )


def save_figure(figure: Any, path: str | Path, *, close: bool = True) -> Path:
    """Save to ``path`` (creating parents) and optionally close the figure."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path)
    if close:
        plt.close(figure)
    return path
