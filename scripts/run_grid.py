#!/usr/bin/env python3
"""Generic experiment grid runner.

Runs ``methods x seeds x grid-values`` through the normal training entrypoint
and writes combined result tables next to the per-run artefacts.

The preferred workflow is to edit a YAML spec file and run this script without
arguments::

    python scripts/run_grid.py

By default it reads ``configs/grid.yaml``. A different spec can be passed with
``--spec``.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rpcp.methods import METHODS, apply_method  # noqa: E402
from rpcp.training import run_experiment  # noqa: E402
from rpcp.utils.io import save_json, to_dict  # noqa: E402
from rpcp.utils.logging import get_logger  # noqa: E402
from scripts._common import add_common_args, build_config, results_to_frame  # noqa: E402

logger = get_logger("grid")
DEFAULT_SPEC = Path("configs/grid.yaml")


def parse_grid(entries: list[str]) -> list[tuple[str, list[Any]]]:
    grid: list[tuple[str, list[Any]]] = []
    for entry in entries:
        if "=" not in entry:
            raise ValueError(
                f"Grid entry '{entry}' is not of the form key=[v1,v2,...] or key=value"
            )
        key, raw = entry.split("=", 1)
        value = yaml.safe_load(raw)
        values = value if isinstance(value, list) else [value]
        if not values:
            raise ValueError(f"Grid entry '{entry}' produced an empty value list")
        grid.append((key.strip(), values))
    return grid


def load_spec(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise TypeError(f"Grid spec at {path} must be a mapping, got {type(payload)}")
    return payload


def spec_to_argv(spec: dict[str, Any]) -> list[str]:
    argv: list[str] = []
    if config := spec.get("config"):
        argv += ["--config", str(config)]
    if output_dir := spec.get("output_dir"):
        argv += ["--output-dir", str(output_dir)]
    if seeds := spec.get("seeds"):
        argv += ["--seeds", *[str(seed) for seed in seeds]]
    if log_level := spec.get("log_level"):
        argv += ["--log-level", str(log_level)]
    if overrides := spec.get("override"):
        argv += ["-o", *[f"{key}={render_cli_value(value)}" for key, value in overrides.items()]]
    return argv


def render_cli_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value)


def make_combo_tag(index: int, method: str, seed: int, overrides: dict[str, Any]) -> str:
    parts = [f"{method}", f"seed{seed}", f"run{index:03d}"]
    for key, value in overrides.items():
        leaf = key.split(".")[-1]
        rendered = json.dumps(value, default=str).strip('"').replace("/", "-")
        rendered = rendered.replace(" ", "").replace("[", "").replace("]", "")
        rendered = rendered.replace(",", "_").replace(":", "-")
        parts.append(f"{leaf}={rendered}")
    return "_".join(parts)


def main(argv: list[str] | None = None) -> int:
    parser = add_common_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument(
        "--spec",
        type=Path,
        default=DEFAULT_SPEC,
        help="YAML grid spec file. Defaults to configs/grid.yaml",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=sorted(METHODS),
        help="Methods to run, e.g. pcp r-pcp-audit oracle-anchor",
    )
    parser.add_argument(
        "--grid",
        nargs="+",
        default=[],
        metavar="KEY=[V1,V2,...]",
        help="Grid dimensions, e.g. loss.lambda_audit_concept=[0.5,1.0,2.0]",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Subdirectory name under output_dir for this grid run",
    )
    args = parser.parse_args(argv)

    spec = load_spec(args.spec)
    spec_argv = spec_to_argv(spec)

    # `argv is None` means "read sys.argv"; `argv or []` would silently drop
    # every command-line flag on the second parse, so resolve it explicitly.
    cli_argv = sys.argv[1:] if argv is None else list(argv)
    merged = parser.parse_args([*spec_argv, *cli_argv])
    args = merged

    methods = args.methods or spec.get("methods")
    if not methods:
        raise ValueError(
            f"No methods specified. Set 'methods' in {args.spec} or pass --methods on the CLI."
        )
    grid_entries = args.grid or [
        f"{key}={json.dumps(values)}"
        for key, values in (spec.get("grid") or {}).items()
    ]
    name = args.name or spec.get("name") or "grid"

    base = build_config(args)
    grid = parse_grid(grid_entries)
    output_dir = Path(base.output_dir) / name
    runs_dir = output_dir / "runs"
    output_dir.mkdir(parents=True, exist_ok=True)

    dimensions = [key for key, _ in grid]
    value_lists = [values for _, values in grid]
    combinations = list(itertools.product(*value_lists)) if value_lists else [()]
    total = len(methods) * len(args.seeds) * len(combinations)

    rows: list[dict[str, Any]] = []
    detailed_rows: list[dict[str, Any]] = []
    index = 0
    for method, seed, combo in itertools.product(methods, args.seeds, combinations):
        index += 1
        combo_overrides = {key: value for key, value in zip(dimensions, combo, strict=True)}
        overrides = {
            **combo_overrides,
            "seed": seed,
            "output_dir": str(runs_dir),
        }
        tag = make_combo_tag(index, method, seed, combo_overrides)
        logger.info("[%d/%d] %s | %s", index, total, method, combo_overrides)
        config = apply_method(base, method, extra=overrides, tag=tag)
        result = run_experiment(config)

        summary = result.summary()
        row: dict[str, Any] = {
            "method": method,
            "seed": seed,
            "run_dir": str(result.run_dir),
            **combo_overrides,
            **summary,
        }
        rows.append(row)
        detailed_rows.append(
            {
                "method": method,
                "seed": seed,
                "run_dir": str(result.run_dir),
                "grid_overrides": combo_overrides,
                "config": to_dict(config),
                "summary": summary,
            }
        )

        results_to_frame(rows).to_csv(output_dir / "results.csv", index=False)
        (output_dir / "results.jsonl").write_text(
            "".join(json.dumps(item, default=str) + "\n" for item in detailed_rows),
            encoding="utf-8",
        )

    save_json(
        {
            "base_config": to_dict(base),
            "methods": methods,
            "seeds": args.seeds,
            "grid": [{key: values} for key, values in grid],
            "n_runs": len(rows),
            "spec_path": str(args.spec),
            "spec": spec,
        },
        output_dir / "grid_spec.json",
    )
    logger.info("Grid complete: %d runs -> %s", len(rows), output_dir)
    print(results_to_frame(rows).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
