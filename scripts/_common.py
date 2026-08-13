"""Shared CLI plumbing for the ``scripts/`` entry points."""

from __future__ import annotations

import argparse
import itertools
import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from rpcp.config import ExperimentConfig, load_config
from rpcp.methods import apply_method
from rpcp.training import TrainingResult, run_experiment
from rpcp.utils.logging import configure_logging, get_logger

__all__ = [
    "SweepRunner",
    "add_common_args",
    "build_config",
    "parse_overrides",
    "results_to_frame",
]

logger = get_logger(__name__)


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--config", type=Path, default=Path("configs/synthetic.yaml"))
    parser.add_argument(
        "--override",
        "-o",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="Dotted config overrides, e.g. -o optim.epochs=5 priors.corruption.alpha=0.6",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--seeds", type=int, nargs="*", default=[0])
    parser.add_argument("--log-level", default="INFO")
    return parser


def parse_overrides(pairs: Sequence[str]) -> dict[str, Any]:
    """Parse ``key=value`` strings, decoding values as YAML scalars."""
    overrides: dict[str, Any] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Override '{pair}' is not of the form key=value")
        key, raw = pair.split("=", 1)
        overrides[key.strip()] = yaml.safe_load(raw)
    return overrides


def build_config(args: argparse.Namespace) -> ExperimentConfig:
    configure_logging(getattr(logging, str(args.log_level).upper(), logging.INFO))
    overrides = parse_overrides(args.override)
    if args.output_dir is not None:
        overrides["output_dir"] = str(args.output_dir)
    return load_config(args.config, overrides=overrides)


def results_to_frame(rows: Iterable[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(list(rows))
    return frame.sort_values(list(frame.columns[:3])) if len(frame) else frame


@dataclass(slots=True)
class SweepRunner:
    """Runs ``methods x sweep-values x seeds`` and collects one row per run.

    Args:
        base: Base experiment config.
        output_dir: Where ``results.csv`` and per-run artefacts are written.
        methods: Method names from :mod:`rpcp.methods`.
        seeds: Random seeds; results are averaged over them in the figures.
    """

    base: ExperimentConfig
    output_dir: Path
    methods: Sequence[str]
    seeds: Sequence[int] = (0,)
    rows: list[dict[str, Any]] = field(default_factory=list)

    def run(
        self,
        sweep_key: str,
        sweep_values: Sequence[Any],
        *,
        extra_overrides: dict[str, Any] | None = None,
        per_value_overrides: Any = None,
    ) -> pd.DataFrame:
        """Execute the grid and return the results frame.

        Args:
            sweep_key: Dotted config key varied along the sweep (e.g.
                ``priors.corruption.alpha``).
            sweep_values: Values for that key.
            extra_overrides: Applied to every run.
            per_value_overrides: Optional callable ``value -> dict`` of extra
                overrides for that sweep value.
        """
        total = len(self.methods) * len(sweep_values) * len(self.seeds)
        for index, (method, value, seed) in enumerate(
            itertools.product(self.methods, sweep_values, self.seeds), start=1
        ):
            overrides: dict[str, Any] = {sweep_key: value, "seed": seed}
            overrides.update(extra_overrides or {})
            if per_value_overrides is not None:
                overrides.update(per_value_overrides(value))
            tag = f"{method}/{sweep_key.split('.')[-1]}={value}/seed={seed}"
            overrides["output_dir"] = str(self.output_dir / "runs")
            config = apply_method(self.base, method, extra=overrides, tag=tag.replace("/", "_"))

            logger.info("[%d/%d] %s", index, total, tag)
            result = run_experiment(config)
            self.rows.append(self._row(method, sweep_key, value, seed, result))
            self.save()
        return results_to_frame(self.rows)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _row(
        method: str,
        sweep_key: str,
        value: Any,
        seed: int,
        result: TrainingResult,
    ) -> dict[str, Any]:
        row: dict[str, Any] = {
            "method": method,
            sweep_key: value,
            "seed": seed,
            "run_dir": str(result.run_dir),
        }
        row.update(result.summary())
        return row

    def save(self) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / "results.csv"
        results_to_frame(self.rows).to_csv(path, index=False)
        return path
