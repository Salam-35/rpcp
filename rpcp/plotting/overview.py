"""Figure 1: the method overview diagram (plan section 7).

Drawn with matplotlib so the figure regenerates with the code rather than
drifting away from it in a separate vector-graphics file.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from rpcp.plotting.style import save_figure, use_paper_style

__all__ = ["plot_method_overview"]

_BOX_STYLE = {"boxstyle": "round,pad=0.3", "linewidth": 1.2}


@dataclass(slots=True)
class _Box:
    """A labelled rectangle with named anchor points."""

    x: float
    y: float
    w: float
    h: float

    def anchor(self, side: str) -> tuple[float, float]:
        match side:
            case "left":
                return self.x, self.y + self.h / 2
            case "right":
                return self.x + self.w, self.y + self.h / 2
            case "top":
                return self.x + self.w / 2, self.y + self.h
            case "bottom":
                return self.x + self.w / 2, self.y
            case "topleft":
                return self.x, self.y + self.h
            case "bottomleft":
                return self.x, self.y
            case _:
                raise ValueError(f"Unknown anchor '{side}'")


def _box(axis: Any, box: _Box, text: str, colour: str, *, fontsize: float = 9) -> _Box:
    axis.add_patch(
        FancyBboxPatch(
            (box.x, box.y), box.w, box.h, facecolor=colour, edgecolor="#333333", **_BOX_STYLE
        )
    )
    axis.text(
        box.x + box.w / 2,
        box.y + box.h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
    )
    return box


def _arrow(axis: Any, start: _Box, start_side: str, end: _Box, end_side: str) -> None:
    axis.add_patch(
        FancyArrowPatch(
            start.anchor(start_side),
            end.anchor(end_side),
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=1.2,
            color="#333333",
            shrinkA=2,
            shrinkB=2,
        )
    )


def plot_method_overview(path: str | Path | None = None) -> Any:
    """Render the R-PCP data/loss flow, emphasising per-entry prior weighting."""
    use_paper_style()
    figure, axis = plt.subplots(figsize=(9.5, 4.6))
    axis.set_xlim(0, 100)
    axis.set_ylim(0, 48)
    axis.axis("off")
    axis.grid(False)

    image = _box(axis, _Box(2, 33, 13, 9), "image\n$x$", "#EAF2FB")
    backbone = _box(axis, _Box(18, 33, 15, 9), "backbone\n$f_\\theta$", "#D7E7F7")
    concepts = _box(axis, _Box(36, 33, 19, 9), "concept predictor\n$\\hat c_\\theta(x)$", "#C3DBF2")
    classes = _box(axis, _Box(58, 33, 17, 9), "class prediction\n$L_{cls}$", "#B0CFED")

    means = _box(axis, _Box(36, 19, 19, 9), "class-level means\n$\\bar p[m,y]$", "#F6E7CE")
    evidence = _box(
        axis,
        _Box(60, 17, 34, 13),
        "evidence:  multi-source priors  •  audit split\n"
        "instability across folds  •  held-out residual",
        "#EFEFEF",
        fontsize=8.5,
    )

    prior = _box(axis, _Box(2, 3, 17, 9), "prior table\n$\\tilde\\Pi[m,y]$", "#F3D9D9")
    reliability = _box(axis, _Box(22, 3, 20, 9), "reliability module\n$r[m,y]$", "#E8D6F0")
    loss = _box(
        axis,
        _Box(46, 3, 30, 9),
        "weighted prior loss\n$\\sum_{m,y} r[m,y]\\,D(\\tilde\\Pi\\,\\|\\,\\bar p)$",
        "#DDEEDD",
    )

    for start, start_side, end, end_side in [
        (image, "right", backbone, "left"),
        (backbone, "right", concepts, "left"),
        (concepts, "right", classes, "left"),
        (concepts, "bottom", means, "top"),
        (means, "bottom", loss, "top"),
        (prior, "right", reliability, "left"),
        (reliability, "right", loss, "left"),
        (evidence, "bottomleft", reliability, "top"),
    ]:
        _arrow(axis, start, start_side, end, end_side)

    axis.text(
        50,
        45.5,
        "Figure 1: prior entries are weighted individually by their estimated reliability",
        ha="center",
        fontsize=11,
    )
    axis.text(
        50,
        0.2,
        "class-level priors constrain averages only: per-image concepts are not identifiable "
        "from $\\tilde\\Pi$ alone (Proposition 1)",
        ha="center",
        fontsize=8,
        style="italic",
        color="#555555",
    )

    figure.tight_layout()
    if path is not None:
        save_figure(figure, path, close=False)
    return figure
