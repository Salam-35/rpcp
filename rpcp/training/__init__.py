"""Training loop, phase schedules, class-mean estimation and cross-fitting."""

from __future__ import annotations

from rpcp.class_means import ClassMeanEstimate, ClassMeanEstimator
from rpcp.config import ExperimentConfig
from rpcp.data import build_prior_bundle, build_splits
from rpcp.training.crossfit import (
    estimate_class_means,
    estimate_class_means_crossfit,
    estimate_instability,
    fold_class_means,
)
from rpcp.training.schedules import Phase, PhaseSchedule, build_optimizer, build_scheduler
from rpcp.training.trainer import RPCPTrainer, TrainerState, TrainingResult
from rpcp.utils.logging import get_logger
from rpcp.utils.seeding import seed_everything

__all__ = [
    "ClassMeanEstimate",
    "ClassMeanEstimator",
    "Phase",
    "PhaseSchedule",
    "RPCPTrainer",
    "TrainerState",
    "TrainingResult",
    "build_optimizer",
    "build_scheduler",
    "estimate_class_means",
    "estimate_class_means_crossfit",
    "estimate_instability",
    "fold_class_means",
    "run_experiment",
]

logger = get_logger(__name__)


def run_experiment(config: ExperimentConfig) -> TrainingResult:
    """Build datasets, priors and model from a config, then train and evaluate.

    This is the single entry point shared by ``scripts/train.py`` and every
    sweep script, so a sweep cell is guaranteed to run the same code path as a
    single run.
    """
    seed_everything(config.seed)
    splits = build_splits(config.data)
    priors = build_prior_bundle(config, splits)
    trainer = RPCPTrainer(config, splits, priors)
    return trainer.fit()
