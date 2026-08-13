#!/usr/bin/env python3
"""Train a single R-PCP (or baseline) model.

Examples::

    python scripts/train.py --config configs/synthetic.yaml --method pcp
    python scripts/train.py --config configs/ph2.yaml --method r-pcp \\
        -o priors.corruption.mode=class_swap priors.corruption.alpha=0.7
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rpcp.methods import METHODS, apply_method  # noqa: E402
from rpcp.training import run_experiment  # noqa: E402
from rpcp.utils.logging import get_logger  # noqa: E402
from scripts._common import add_common_args, build_config  # noqa: E402

logger = get_logger("train")


def main(argv: list[str] | None = None) -> int:
    parser = add_common_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--method", default="pcp", choices=sorted(METHODS))
    args = parser.parse_args(argv)

    base = build_config(args)
    summaries = []
    for seed in args.seeds:
        config = apply_method(
            base,
            args.method,
            extra={"seed": seed},
            tag=f"{args.method}_seed{seed}",
        )
        result = run_experiment(config)
        summary = result.summary()
        summaries.append({"method": args.method, "seed": seed, **summary})
        logger.info(
            "%s (seed %d): test concept F1=%.4f | test class F1=%.4f | reliability AUROC=%.4f",
            args.method,
            seed,
            summary.get("test/concept_macro_f1", float("nan")),
            summary.get("test/class_macro_f1", float("nan")),
            summary.get("reliability/auroc", float("nan")),
        )

    print(json.dumps(summaries, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
