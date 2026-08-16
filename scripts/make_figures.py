#!/usr/bin/env python3
"""Regenerate every paper figure from sweep artefacts (plan section 7).

Figure 1 is drawn from code; Figures 2-5 are rebuilt from the ``results.csv``
files written by the sweep scripts, and the reliability heatmaps/trajectories
from the ``.npy`` files inside each run directory.

Example::

    python scripts/make_figures.py --results-dir runs --output-dir figures
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rpcp.plotting import (  # noqa: E402
    plot_audit_budget,
    plot_corruption_robustness,
    plot_delta_curve,
    plot_method_overview,
    plot_reliability_evolution,
    plot_reliability_heatmaps,
)
from rpcp.plotting.robustness_curves import Curve  # noqa: E402
from rpcp.utils.logging import configure_logging, get_logger  # noqa: E402

logger = get_logger("make-figures")


def _curves(frame: pd.DataFrame, x_column: str, metric: str) -> dict[str, Curve]:
    curves: dict[str, Curve] = {}
    for method, group in frame.groupby("method"):
        aggregated = group.groupby(x_column)[metric].agg(["mean", "std"])
        curves[str(method)] = Curve(
            x=aggregated.index.tolist(),
            y=aggregated["mean"].tolist(),
            yerr=aggregated["std"].fillna(0.0).tolist(),
            label=str(method),
        )
    return curves


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("runs"))
    parser.add_argument("--output-dir", type=Path, default=Path("figures"))
    parser.add_argument("--metric", default="test/concept_macro_f1")
    args = parser.parse_args(argv)
    configure_logging()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Figure 1 --------------------------------------------------------- #
    path = args.output_dir / "figure1_method_overview.png"
    plot_method_overview(path=path)
    logger.info("Wrote %s", path)

    # ---- Figures 2 and 3 -------------------------------------------------- #
    for sweep_dir in sorted(args.results_dir.glob("corruption_*")):
        results = sweep_dir / "results.csv"
        if not results.exists():
            continue
        frame = pd.read_csv(results)
        mode = sweep_dir.name.replace("corruption_", "")
        path = args.output_dir / f"figure2_corruption_{mode}.png"
        plot_corruption_robustness(
            _curves(frame, "priors.corruption.alpha", args.metric),
            metric=args.metric.split("/")[-1].replace("_", " "),
            corruption_mode=mode,
            path=path,
        )
        logger.info("Wrote %s", path)

        worst = frame[frame["priors.corruption.alpha"] == frame["priors.corruption.alpha"].max()]
        for _, row in worst.iterrows():
            run_dir = Path(row["run_dir"])
            reliability = run_dir / "reliability.npy"
            mask = run_dir / "corruption_mask.npy"
            if not (reliability.exists() and mask.exists()):
                continue
            clean, observed = run_dir / "prior_clean.npy", run_dir / "prior_observed.npy"
            error = (
                np.abs(np.load(observed) - np.load(clean))
                if clean.exists() and observed.exists()
                else None
            )
            path = args.output_dir / f"figure3_{mode}_{row['method']}.png"
            plot_reliability_heatmaps(
                np.load(reliability),
                corruption_mask=np.load(mask),
                prior_error=error,
                title=f"Figure 3: {row['method']} under {mode}",
                path=path,
            )
            logger.info("Wrote %s", path)

            history = run_dir / "reliability_history.npy"
            if history.exists():
                path = args.output_dir / f"figure3b_trajectory_{mode}_{row['method']}.png"
                plot_reliability_evolution(
                    np.load(history),
                    corruption_mask=np.load(mask),
                    title=f"Reliability trajectory: {row['method']} under {mode}",
                    path=path,
                )
                logger.info("Wrote %s", path)

    # ---- Figure 4 --------------------------------------------------------- #
    delta_results = args.results_dir / "delta_sweep" / "results.csv"
    if delta_results.exists():
        frame = pd.read_csv(delta_results)
        path = args.output_dir / "figure4_prior_separation.png"
        plot_delta_curve(
            _curves(frame, "prior/delta", args.metric),
            show_class_metric=_curves(frame, "prior/delta", "test/class_macro_f1"),
            path=path,
        )
        logger.info("Wrote %s", path)

    # ---- Figure 5 --------------------------------------------------------- #
    audit_results = args.results_dir / "audit_budget" / "results.csv"
    if audit_results.exists():
        frame = pd.read_csv(audit_results)
        aggregated = frame.groupby("datasets.audit_fraction").agg(
            concept_mean=(args.metric, "mean"),
            concept_std=(args.metric, "std"),
            auroc_mean=("reliability/auroc", "mean"),
            auroc_std=("reliability/auroc", "std"),
        )
        percentages = [100 * value for value in aggregated.index.tolist()]
        path = args.output_dir / "figure5_audit_budget.png"
        plot_audit_budget(
            Curve(
                x=percentages,
                y=aggregated["concept_mean"].tolist(),
                yerr=aggregated["concept_std"].fillna(0.0).tolist(),
            ),
            Curve(
                x=percentages,
                y=aggregated["auroc_mean"].tolist(),
                yerr=aggregated["auroc_std"].fillna(0.0).tolist(),
            ),
            path=path,
        )
        logger.info("Wrote %s", path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
