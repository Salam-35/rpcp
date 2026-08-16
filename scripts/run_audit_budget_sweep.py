#!/usr/bin/env python3
"""Audit-budget sweep (plan 4.3 Evidence Mode C / Figure 5).

Varies the fraction of training images carrying concept annotations that are
used *only* to calibrate reliability (never to train the concept predictor) and
reports both concept F1 and reliability-detection AUROC.

Example::

    python scripts/run_audit_budget_sweep.py --config configs/synthetic.yaml \\
        --budgets 0.0 0.02 0.05 0.1 0.2 --alpha 0.75 --mode class_swap
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rpcp.plotting import plot_audit_budget  # noqa: E402
from rpcp.plotting.robustness_curves import Curve  # noqa: E402
from rpcp.utils.logging import get_logger  # noqa: E402
from scripts._common import SweepRunner, add_common_args, build_config  # noqa: E402

logger = get_logger("audit-sweep")


def main(argv: list[str] | None = None) -> int:
    parser = add_common_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--budgets", type=float, nargs="+", default=[0.0, 0.02, 0.05, 0.1, 0.2])
    parser.add_argument("--alpha", type=float, default=0.75)
    parser.add_argument("--mode", default="class_swap")
    parser.add_argument("--fraction", type=float, default=0.3)
    args = parser.parse_args(argv)

    base = build_config(args)
    output_dir = Path(base.output_dir) / "audit_budget"

    # A zero budget cannot use audit evidence: fall back to unsupervised R-PCP.
    def per_value(value: float) -> dict[str, object]:
        return {"reliability.mode": "audit" if value > 0 else "unsupervised"}

    runner = SweepRunner(
        base=base, output_dir=output_dir, methods=["r-pcp-audit"], seeds=args.seeds
    )
    frame = runner.run(
        "datasets.audit_fraction",
        args.budgets,
        extra_overrides={
            "priors.corruption.mode": args.mode,
            "priors.corruption.alpha": args.alpha,
            "priors.corruption.fraction": args.fraction,
        },
        per_value_overrides=per_value,
    )
    logger.info("Sweep complete: %d runs -> %s", len(frame), runner.save())

    aggregated = frame.groupby("datasets.audit_fraction").agg(
        concept_mean=("test/concept_macro_f1", "mean"),
        concept_std=("test/concept_macro_f1", "std"),
        auroc_mean=("reliability/auroc", "mean"),
        auroc_std=("reliability/auroc", "std"),
    )
    percentages = [100 * value for value in aggregated.index.tolist()]

    path = output_dir / "figure5_audit_budget.png"
    plot_audit_budget(
        Curve(
            x=percentages,
            y=aggregated["concept_mean"].tolist(),
            yerr=aggregated["concept_std"].fillna(0.0).tolist(),
            label="concept macro F1",
        ),
        Curve(
            x=percentages,
            y=aggregated["auroc_mean"].tolist(),
            yerr=aggregated["auroc_std"].fillna(0.0).tolist(),
            label="reliability AUROC",
        ),
        path=path,
    )
    logger.info("Wrote %s", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
