#!/usr/bin/env python3
"""Train one or more R-PCP methods.

Examples::

    python scripts/train.py --config configs/synthetic.yaml --method pcp
    python scripts/train.py --config configs/synthetic.yaml --method pcp,r-pcp,r-pcp-audit
    python scripts/train.py --config configs/synthetic.yaml --method '[pcp,r-pcp,r-pcp-audit]'
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


def parse_methods(values: list[str]) -> list[str]:
    """Parse one or more method names from CLI tokens.

    Accepted forms:

    - ``--method pcp``
    - ``--method pcp,r-pcp``
    - ``--method '[pcp,r-pcp]'``
    - ``--method pcp r-pcp``
    """
    methods: list[str] = []
    for value in values:
        cleaned = value.strip().lstrip("[").rstrip("]")
        for part in cleaned.split(","):
            method = part.strip().lstrip("[").rstrip("]")
            if method:
                methods.append(method)
    return methods


def main(argv: list[str] | None = None) -> int:
    parser = add_common_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument(
        "--method",
        nargs="+",
        default=["pcp"],
        help=(
            "Method name(s). Accepts one name, comma-separated names, bracketed names "
            "like '[pcp,r-pcp]', or space-separated names."
        ),
    )
    args = parser.parse_args(argv)
    methods = parse_methods(args.method)
    unknown = [method for method in methods if method not in METHODS]
    if unknown:
        parser.error(f"unknown method(s): {unknown}. Known: {sorted(METHODS)}")

    base = build_config(args)
    summaries: dict[str, dict[str, dict[str, object]]] = {}
    for method in methods:
        summaries[method] = {}
        for seed in args.seeds:
            config = apply_method(
                base,
                method,
                extra={"seed": seed},
                tag=f"{method}_seed{seed}",
            )
            result = run_experiment(config)
            summary = result.summary()
            summaries[method][str(seed)] = {"method": method, "seed": seed, **summary}
            logger.info(
                (
                    "%s (seed %d): test concept F1=%.4f | test class F1=%.4f | "
                    "reliability AUROC=%.4f"
                ),
                method,
                seed,
                summary.get("test/concept_macro_f1", float("nan")),
                summary.get("test/class_macro_f1", float("nan")),
                summary.get("reliability/auroc", float("nan")),
            )

    print(json.dumps(summaries, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
