"""Paper figures (plan section 7)."""

from __future__ import annotations

from rpcp.plotting.heatmaps import (
    plot_prior_table,
    plot_reliability_evolution,
    plot_reliability_heatmaps,
)
from rpcp.plotting.overview import plot_method_overview
from rpcp.plotting.robustness_curves import (
    Curve,
    plot_audit_budget,
    plot_corruption_robustness,
    plot_curves,
    plot_delta_curve,
)
from rpcp.plotting.style import METHOD_COLOURS, save_figure, use_paper_style

__all__ = [
    "METHOD_COLOURS",
    "Curve",
    "plot_audit_budget",
    "plot_corruption_robustness",
    "plot_curves",
    "plot_delta_curve",
    "plot_method_overview",
    "plot_prior_table",
    "plot_reliability_evolution",
    "plot_reliability_heatmaps",
    "save_figure",
    "use_paper_style",
]
