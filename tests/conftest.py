"""Shared pytest fixtures: a tiny synthetic experiment that runs in seconds."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rpcp.config import ExperimentConfig, load_config  # noqa: E402
from rpcp.data import SplitBundle, build_prior_bundle, build_splits  # noqa: E402
from rpcp.data.priors import PriorBundle  # noqa: E402
from rpcp.utils.seeding import seed_everything  # noqa: E402

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


@pytest.fixture(autouse=True)
def _deterministic() -> None:
    seed_everything(0)
    torch.set_num_threads(1)


@pytest.fixture
def tiny_config(tmp_path: Path) -> ExperimentConfig:
    """A synthetic config small enough to train inside a unit test."""
    return load_config(
        CONFIG_DIR / "synthetic.yaml",
        overrides={
            "output_dir": str(tmp_path),
            "optim.epochs": 3,
            "reliability.warmup_epochs": 1,
            "reliability.mid_epochs": 2,
            "data.batch_size": 32,
            "data.synthetic.n_train": 120,
            "data.synthetic.n_val": 60,
            "data.synthetic.n_test": 60,
            "data.synthetic.n_concepts": 6,
            "data.synthetic.n_classes": 3,
            "data.synthetic.image_size": 16,
            "eval.save_checkpoints": False,
        },
    )


@pytest.fixture
def tiny_splits(tiny_config: ExperimentConfig) -> SplitBundle:
    return build_splits(tiny_config.data)


@pytest.fixture
def tiny_priors(tiny_config: ExperimentConfig, tiny_splits: SplitBundle) -> PriorBundle:
    return build_prior_bundle(tiny_config, tiny_splits)
