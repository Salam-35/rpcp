#!/usr/bin/env python3
"""Prior-separation sweep (plan 6.4 / Figure 4).

Blends one class prior column into another (``alpha -> 1`` makes the two class
signatures identical) and reports concept F1 against the resulting separation
``Delta``.  Class F1 is plotted on a twin axis: if it stays high while concept
F1 collapses, the model is classifying through a non-concept shortcut.

Example::

    python scripts/run_delta_sweep.py --config configs/synthetic.yaml \\
        --methods pcp r-pcp --blend-alphas 0 0.25 0.5 0.75 0.95
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rpcp.plotting import plot_delta_curve  # noqa: E402
from rpcp.plotting.robustness_curves import Curve  # noqa: E402
from rpcp.utils.logging import get_logger  # noqa: E402
from scripts._common import SweepRunner, add_common_args, build_config  # noqa: E402

logger = get_logger("delta-sweep")


def main(argv: list[str] | None = None) -> int:
    parser = add_common_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--methods", nargs="+", default=["pcp", "r-pcp"])
    parser.add_argument(
        "--blend-alphas", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 0.95]
    )
    parser.add_argument("--source", type=int, default=0)
    parser.add_argument("--target", type=int, default=1)
    args = parser.parse_args(argv)

    base = build_config(args)
    output_dir = Path(base.output_dir) / "delta_sweep"

    runner = SweepRunner(base=base, output_dir=output_dir, methods=args.methods, seeds=args.seeds)
    frame = runner.run(
        "priors.blend_alpha",
        args.blend_alphas,
        extra_overrides={
            "priors.blend_source": args.source,
            "priors.blend_target": args.target,
        },
    )
    logger.info("Sweep complete: %d runs -> %s", len(frame), runner.save())

    concept_curves: dict[str, Curve] = {}
    class_curves: dict[str, Curve] = {}
    for method, group in frame.groupby("method"):
        # x-axis is the *measured* Delta of the prior actually used for training.
        aggregated = group.groupby("prior/delta").agg(
            concept_mean=("test/concept_macro_f1", "mean"),
            concept_std=("test/concept_macro_f1", "std"),
            class_mean=("test/class_macro_f1", "mean"),
        )
        concept_curves[str(method)] = Curve(
            x=aggregated.index.tolist(),
            y=aggregated["concept_mean"].tolist(),
            yerr=aggregated["concept_std"].fillna(0.0).tolist(),
            label=str(method),
        )
        class_curves[str(method)] = Curve(
            x=aggregated.index.tolist(), y=aggregated["class_mean"].tolist(), label=str(method)
        )

    path = output_dir / "figure4_prior_separation.png"
    plot_delta_curve(concept_curves, show_class_metric=class_curves, path=path)
    logger.info("Wrote %s", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
