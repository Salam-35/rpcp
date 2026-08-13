#!/usr/bin/env python3
"""Corruption sweep (plan 6.3 / Figure 2 / Figure 3).

Trains every requested method under increasing prior-corruption strength and
writes ``results.csv`` plus the robustness curve and reliability heatmaps.

Example::

    python scripts/run_corruption_sweep.py --config configs/synthetic.yaml \\
        --methods pcp r-pcp oracle --mode class_swap --alphas 0 0.25 0.5 0.75 1.0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rpcp.plotting import plot_corruption_robustness, plot_reliability_heatmaps  # noqa: E402
from rpcp.plotting.robustness_curves import Curve  # noqa: E402
from rpcp.utils.logging import get_logger  # noqa: E402
from scripts._common import SweepRunner, add_common_args, build_config  # noqa: E402

logger = get_logger("corruption-sweep")


def main(argv: list[str] | None = None) -> int:
    parser = add_common_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--methods", nargs="+", default=["pcp", "r-pcp", "oracle"])
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--mode", default="class_swap")
    parser.add_argument("--fraction", type=float, default=0.3)
    parser.add_argument(
        "--metric", default="test/concept_macro_f1", help="Column plotted on the y-axis"
    )
    args = parser.parse_args(argv)

    base = build_config(args)
    output_dir = Path(base.output_dir) / f"corruption_{args.mode}"

    runner = SweepRunner(
        base=base,
        output_dir=output_dir,
        methods=args.methods,
        seeds=args.seeds,
    )
    frame = runner.run(
        "priors.corruption.alpha",
        args.alphas,
        extra_overrides={
            "priors.corruption.mode": args.mode,
            "priors.corruption.fraction": args.fraction,
        },
    )
    logger.info("Sweep complete: %d runs -> %s", len(frame), runner.save())

    # ---- Figure 2 -------------------------------------------------------- #
    curves: dict[str, Curve] = {}
    for method, group in frame.groupby("method"):
        aggregated = group.groupby("priors.corruption.alpha")[args.metric].agg(["mean", "std"])
        curves[str(method)] = Curve(
            x=aggregated.index.tolist(),
            y=aggregated["mean"].tolist(),
            yerr=aggregated["std"].fillna(0.0).tolist(),
            label=str(method),
        )
    figure2 = output_dir / "figure2_corruption_robustness.png"
    plot_corruption_robustness(
        curves,
        metric=args.metric.split("/")[-1].replace("_", " "),
        corruption_mode=args.mode,
        path=figure2,
    )
    logger.info("Wrote %s", figure2)

    # ---- Figure 3: heatmaps for the strongest corruption ----------------- #
    strongest = frame[frame["priors.corruption.alpha"] == max(args.alphas)]
    for method in args.methods:
        rows = strongest[strongest["method"] == method]
        if rows.empty:
            continue
        run_dir = Path(rows.iloc[0]["run_dir"])
        mask_path, reliability_path = run_dir / "corruption_mask.npy", run_dir / "reliability.npy"
        if not (mask_path.exists() and reliability_path.exists()):
            continue
        clean = run_dir / "prior_clean.npy"
        observed = run_dir / "prior_observed.npy"
        error = (
            np.abs(np.load(observed) - np.load(clean))
            if clean.exists() and observed.exists()
            else None
        )
        path = output_dir / f"figure3_reliability_{method}.png"
        plot_reliability_heatmaps(
            np.load(reliability_path),
            corruption_mask=np.load(mask_path),
            prior_error=error,
            title=f"Figure 3: {method}, {args.mode} (alpha={max(args.alphas)})",
            path=path,
        )
        logger.info("Wrote %s", path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
