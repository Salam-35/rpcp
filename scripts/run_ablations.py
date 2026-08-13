#!/usr/bin/env python3
"""Ablation table (plan section 8).

Runs every ablation at a fixed corruption setting and writes a tidy CSV plus a
markdown table ready to paste into the paper.

The critical row pair is ``2-residual-only`` vs ``4/5`` (external evidence): if
model-prior agreement alone performs poorly, that supports the revised
motivation of the paper.

Example::

    python scripts/run_ablations.py --config configs/synthetic.yaml \\
        --alpha 0.75 --mode class_swap --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rpcp.methods import ABLATIONS  # noqa: E402
from rpcp.utils.logging import get_logger  # noqa: E402
from scripts._common import SweepRunner, add_common_args, build_config  # noqa: E402

logger = get_logger("ablations")

REPORT_COLUMNS = [
    "test/concept_macro_f1",
    "test/concept_macro_auroc",
    "test/class_macro_f1",
    "reliability/auroc",
    "reliability/auprc",
    "reliability/separation",
]


def main(argv: list[str] | None = None) -> int:
    parser = add_common_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--ablations", nargs="+", default=sorted(ABLATIONS))
    parser.add_argument("--alpha", type=float, default=0.75)
    parser.add_argument("--mode", default="class_swap")
    parser.add_argument("--fraction", type=float, default=0.3)
    args = parser.parse_args(argv)

    base = build_config(args)
    output_dir = Path(base.output_dir) / "ablations"

    runner = SweepRunner(
        base=base, output_dir=output_dir, methods=args.ablations, seeds=args.seeds
    )
    frame = runner.run(
        "priors.corruption.alpha",
        [args.alpha],
        extra_overrides={
            "priors.corruption.mode": args.mode,
            "priors.corruption.fraction": args.fraction,
        },
    )

    columns = [c for c in REPORT_COLUMNS if c in frame.columns]
    table = frame.groupby("method")[columns].agg(["mean", "std"]).round(4)
    table.to_csv(output_dir / "ablation_table.csv")
    (output_dir / "ablation_table.md").write_text(table.to_markdown(), encoding="utf-8")
    logger.info("Wrote %s", output_dir / "ablation_table.md")
    print(table.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
