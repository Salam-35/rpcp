#!/usr/bin/env python3
"""Evaluate a saved checkpoint: concept metrics, class metrics, reliability audit.

Example::

    python scripts/evaluate.py --run-dir runs/rpcp-synthetic-r-pcp_seed0 --split test
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rpcp.config import ExperimentConfig  # noqa: E402
from rpcp.data import build_dataloaders, build_prior_bundle, build_splits  # noqa: E402
from rpcp.evaluation import evaluate_reliability, evaluate_split  # noqa: E402
from rpcp.evaluation.prior_separation import separation_report  # noqa: E402
from rpcp.models.rpcp import build_model  # noqa: E402
from rpcp.utils.io import from_dict, save_json  # noqa: E402
from rpcp.utils.logging import configure_logging, get_logger  # noqa: E402

logger = get_logger("evaluate")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--split", default="test", choices=["train_eval", "val", "test"])
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    configure_logging()

    checkpoint = torch.load(args.run_dir / "checkpoint.pt", map_location="cpu", weights_only=False)
    config: ExperimentConfig = from_dict(ExperimentConfig, checkpoint["config"])
    device = torch.device(args.device or config.resolved_device())

    splits = build_splits(config.data)
    priors = build_prior_bundle(config, splits)
    loaders = build_dataloaders(splits, config.data, seed=config.seed)

    model = build_model(
        config,
        n_concepts=splits.n_concepts,
        n_classes=splits.n_classes,
        priors=priors.observed,
    )
    model.load_state_dict(checkpoint["model"])
    model.to(device)

    result = evaluate_split(
        model,
        loaders[args.split],  # type: ignore[arg-type]
        split=args.split,
        device=device,
        priors=priors.observed.to(device),
        n_classes=splits.n_classes,
        concept_threshold=config.eval.concept_threshold,
        n_bins=config.eval.n_calibration_bins,
        concept_names=splits.concept_names,
        class_names=splits.class_names,
    )

    reliability = torch.as_tensor(checkpoint.get("reliability", np.ones(priors.shape)))
    audit = evaluate_reliability(reliability, priors, concept_result=result.concept)

    payload = {
        **result.as_dict(),
        **audit.as_dict(),
        "prior/delta": separation_report(priors.observed).delta,
        "per_concept": None if result.concept is None else result.concept.per_concept_table(),
    }
    output = args.output or args.run_dir / f"eval_{args.split}.json"
    save_json(payload, output)
    print(json.dumps(payload, indent=2, default=float))
    logger.info("Wrote %s", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
