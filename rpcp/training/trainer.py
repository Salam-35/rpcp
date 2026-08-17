"""The R-PCP training loop (plan 5.1 / 5.2).

.. code-block:: python

    for epoch in range(num_epochs):
        train_model_one_epoch(model, train_loader, reliability=r)

        if epoch >= warmup_epochs and epoch % em_freq == 0:
            p_bar_heldout = estimate_class_means_crossfit(model, train_loader)
            instability   = estimate_seed_or_aug_instability(models_or_augments)
            r_new = reliability_update(...)
            r = ema(r, r_new)

The class differs from the pseudocode in exactly two respects, both deliberate:
reliability is *estimated* during Phase 1 but only *applied* from Phase 2 (so a
noisy first estimate cannot wreck the warm-started model), and the evidence is
computed on held-out datasets by default (plan 4.5).
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from rpcp.config import ExperimentConfig, ReliabilityMode
from rpcp.data import SplitBundle, build_dataloaders
from rpcp.data.base import ConceptBatch, TransformedSubset
from rpcp.data.priors import PriorBundle
from rpcp.evaluation import EvaluationResult, evaluate_reliability, evaluate_split
from rpcp.evaluation.prior_separation import separation_report
from rpcp.evaluation.reliability_metrics import ReliabilityMetrics
from rpcp.losses.classification import class_balanced_weights
from rpcp.losses.composite import CompositeObjective
from rpcp.models.reliability import ReliabilityEvidence, ReliabilityModule, build_reliability_module
from rpcp.models.rpcp import RPCPModel, build_model
from rpcp.training.crossfit import (
    estimate_class_means,
    estimate_instability,
    fold_class_means,
    model_evidence,
)
from rpcp.training.schedules import Phase, PhaseSchedule, build_optimizer, build_scheduler
from rpcp.utils.io import save_json, to_dict
from rpcp.utils.logging import MetricHistory, get_logger
from rpcp.utils.seeding import seed_everything

__all__ = ["RPCPTrainer", "TrainerState", "TrainingResult"]

logger = get_logger(__name__)


@dataclass(slots=True)
class TrainerState:
    epoch: int = 0
    best_score: float = -float("inf")
    best_metric: float = float("nan")
    best_epoch: int = -1
    best_state: dict[str, torch.Tensor] | None = None
    reliability_history: list[np.ndarray] = field(default_factory=list)
    stopped_early: bool = False


@dataclass(slots=True)
class TrainingResult:
    """Everything a sweep needs from one run."""

    config: ExperimentConfig
    val: EvaluationResult
    test: EvaluationResult
    reliability: ReliabilityMetrics
    reliability_matrix: np.ndarray
    history: MetricHistory
    prior_delta: float
    run_dir: Path

    def summary(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        out.update(self.val.as_dict("val/"))
        out.update(self.test.as_dict("test/"))
        out.update(self.reliability.as_dict())
        out["prior/delta"] = self.prior_delta
        out["thresholds"] = {
            "concept": self.config.eval.concept_threshold,
            "reliability_detection": self.config.eval.concept_threshold,
            "reliability_hard": self.config.reliability.hard_threshold,
            "reliability_min": self.config.reliability.min_reliability,
            "reliability_max": self.config.reliability.max_reliability,
            "n_calibration_bins": self.config.eval.n_calibration_bins,
        }
        return out


class RPCPTrainer:
    """Trains one R-PCP (or PCP) model end to end.

    Args:
        config: Full experiment configuration.
        splits: Train/val/test/audit splits.
        priors: Prior bundle (observed table + evidence + benchmark masks).
        model: Optional pre-built model (defaults to ``build_model(config)``).
        device: Overrides ``config.device``.
    """

    def __init__(
        self,
        config: ExperimentConfig,
        splits: SplitBundle,
        priors: PriorBundle,
        *,
        model: RPCPModel | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        self.config = config
        self.splits = splits
        self.device = torch.device(device or config.resolved_device())
        seed_everything(config.seed)

        self.priors = priors.to("cpu")
        self.prior_table = self.priors.observed.to(self.device)
        self.audit_prior = None if self.priors.audit is None else self.priors.audit.to(self.device)

        self.model = (
            model
            if model is not None
            else build_model(
                config,
                n_concepts=splits.n_concepts,
                n_classes=splits.n_classes,
                priors=self.priors.observed,
            )
        ).to(self.device)

        counts = torch.as_tensor(
            np.bincount(
                np.asarray(splits.train.labels), minlength=splits.n_classes
            )
        )
        self.objective = CompositeObjective(
            config.loss,
            n_concepts=splits.n_concepts,
            n_classes=splits.n_classes,
            class_weights=class_balanced_weights(counts),
        ).to(self.device)

        self.reliability_module: ReliabilityModule
        self.static_evidence: ReliabilityEvidence
        self.reliability_module, self.static_evidence = build_reliability_module(
            config.reliability, self.priors
        )
        self.reliability_module = self.reliability_module.to(self.device)

        self.loaders: dict[str, DataLoader[ConceptBatch]] = build_dataloaders(  # type: ignore[assignment]
            splits, config.data, seed=config.seed
        )
        self.optimizer = build_optimizer(self.model, config.optim)
        self.scheduler = build_scheduler(self.optimizer, config.optim)
        self.schedule = PhaseSchedule.from_configs(config.optim, config.reliability)

        self.state = TrainerState()
        self.history = MetricHistory()
        self.run_dir = config.run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Trainer ready | device=%s | M=%d K=%d | reliability=%s | prior Delta=%.3f",
            self.device,
            splits.n_concepts,
            splits.n_classes,
            config.reliability.mode,
            separation_report(self.priors.observed).delta,
        )

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #
    def fit(self) -> TrainingResult:
        """Run all phases and return the final evaluation."""
        for epoch in range(self.config.optim.epochs):
            self.state.epoch = epoch
            started = time.perf_counter()

            train_metrics = self.train_one_epoch(epoch)

            reliability_metrics: dict[str, float] = {}
            if self.schedule.should_update_reliability(epoch):
                reliability_metrics = self.update_reliability(epoch)

            val_result = self.evaluate("val")
            record = {
                "epoch": epoch,
                "phase": str(self.schedule.phase(epoch)),
                "lr": self.optimizer.param_groups[0]["lr"],
                "seconds": time.perf_counter() - started,
                **train_metrics,
                **val_result.as_dict("val/"),
                **reliability_metrics,
                **self.reliability_module.stats(),
            }
            self.history.records.append(record)
            logger.info(
                "epoch %3d [%s] loss=%.4f val/concept_f1=%.4f val/class_f1=%.4f r_mean=%.3f",
                epoch,
                self.schedule.phase(epoch),
                record.get("train/loss_total", float("nan")),
                record.get("val/concept_macro_f1", float("nan")),
                record.get("val/class_macro_f1", float("nan")),
                record.get("reliability/mean", float("nan")),
            )

            self._track_best(val_result, epoch)
            if self.scheduler is not None:
                self.scheduler.step()
            if self._should_stop(epoch):
                self.state.stopped_early = True
                logger.info("Early stopping at epoch %d", epoch)
                break

        return self.finalize()

    def train_one_epoch(self, epoch: int) -> dict[str, float]:
        """One pass over the training datasets (Phase 0/1: ``r = 1``; Phase 2: weighted)."""
        self.model.train()
        use_reliability = self.schedule.use_reliability(epoch)
        reliability = self.reliability_module() if use_reliability else None
        penalty = (
            self.reliability_module.penalty()
            if use_reliability and self.config.loss.lambda_r > 0
            else None
        )

        totals: dict[str, float] = {}
        n_batches = 0
        scaler = torch.amp.GradScaler(
            device=self.device.type, enabled=self.config.optim.amp and self.device.type == "cuda"
        )

        for batch in self.loaders["train"]:
            images = batch["image"].to(self.device, non_blocking=True)
            labels = batch["label"].to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=self.device.type,
                enabled=self.config.optim.amp and self.device.type == "cuda",
            ):
                output = self.model(
                    images,
                    priors=self.prior_table,
                    reliability=(
                        reliability if self.config.loss.match_reliability_weighted else None
                    ),
                )
                losses = self.objective(
                    output,
                    labels,
                    self.prior_table,
                    reliability=reliability,
                    reliability_penalty=penalty,
                    repair_prior=self.audit_prior,
                    concepts=(
                        batch["concepts"].to(self.device)
                        if self.config.loss.lambda_concept > 0
                        else None
                    ),
                    concept_mask=(
                        batch["has_concepts"].to(self.device)
                        if self.config.loss.lambda_concept > 0
                        else None
                    ),
                )

            scaler.scale(losses.total).backward()
            if self.config.optim.grad_clip is not None:
                scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    list(self.model.parameters()), self.config.optim.grad_clip
                )
            scaler.step(self.optimizer)
            scaler.update()

            for key, value in losses.as_dict("train/loss_").items():
                totals[key] = totals.get(key, 0.0) + value
            n_batches += 1

        return {key: value / max(1, n_batches) for key, value in totals.items()}

    # ------------------------------------------------------------------ #
    # Reliability
    # ------------------------------------------------------------------ #
    def collect_evidence(self) -> ReliabilityEvidence:
        """Gather static + model-dependent evidence for the current model."""
        evidence = ReliabilityEvidence()
        mode = ReliabilityMode(self.config.reliability.mode)
        if mode in {ReliabilityMode.NONE, ReliabilityMode.ORACLE}:
            return evidence

        # Held-out class means: the validation split is never trained on, so the
        # residual is not a restatement of the training objective (plan 4.5).
        holdout_key = "val" if self.config.reliability.use_crossfit else "train_eval"
        holdout = estimate_class_means(
            self.model,
            self.loaders[holdout_key],
            n_classes=self.splits.n_classes,
            device=self.device,
            priors=self.prior_table,
        )

        instability = None
        if self.config.reliability.n_folds > 1:
            split_name = "val" if holdout_key == "val" else "train"
            split: TransformedSubset = getattr(self.splits, split_name)
            fold_means = fold_class_means(
                self.model,
                split,
                n_classes=self.splits.n_classes,
                n_folds=self.config.reliability.n_folds,
                batch_size=self.config.data.eval_batch_size,
                device=self.device,
                seed=self.config.seed + self.state.epoch,
            )
            instability = estimate_instability(fold_means)

        valid = holdout.valid.view(1, -1).expand(self.splits.n_concepts, -1)
        evidence = model_evidence(
            self.priors.observed,
            holdout.means,
            instability=instability,
            valid_mask=valid,
        )
        return evidence.merge(self.static_evidence)

    def update_reliability(self, epoch: int) -> dict[str, float]:
        """Recompute evidence and EMA-update ``r`` (plan 5.1, Phase 1/2)."""
        evidence = self.collect_evidence()
        if evidence.is_empty():
            return {}
        self.reliability_module.update(evidence)
        matrix = self.reliability_module().detach().cpu().numpy()
        self.state.reliability_history.append(matrix.copy())

        metrics: dict[str, float] = {"reliability/updated_epoch": float(epoch)}
        if self.priors.corruption_mask is not None:
            audit = evaluate_reliability(
                self.reliability_module().detach().cpu(),
                self.priors,
                threshold=self.config.eval.concept_threshold,
            )
            metrics.update(audit.as_dict())
        for name, value in evidence.as_dict().items():
            metrics[f"evidence/{name}_mean"] = float(value.mean())
        return metrics

    # ------------------------------------------------------------------ #
    # Evaluation
    # ------------------------------------------------------------------ #
    def evaluate(self, split: str, *, keep_predictions: bool = False) -> EvaluationResult:
        return evaluate_split(
            self.model,
            self.loaders[split],
            split=split,
            device=self.device,
            priors=self.prior_table,
            n_classes=self.splits.n_classes,
            concept_threshold=self.config.eval.concept_threshold,
            n_bins=self.config.eval.n_calibration_bins,
            concept_names=self.splits.concept_names,
            class_names=self.splits.class_names,
            keep_predictions=keep_predictions,
        )

    def finalize(self) -> TrainingResult:
        """Phase 3: restore the best model, evaluate, audit reliability, save."""
        if self.state.best_state is not None:
            self.model.load_state_dict(self.state.best_state)
            logger.info("Restored best checkpoint from epoch %d", self.state.best_epoch)

        val_result = self.evaluate("val")
        test_result = self.evaluate("test", keep_predictions=True)
        reliability_matrix = self.reliability_module().detach().cpu()
        audit = evaluate_reliability(
            reliability_matrix,
            self.priors,
            concept_result=test_result.concept,
            threshold=self.config.eval.concept_threshold,
        )

        result = TrainingResult(
            config=self.config,
            val=val_result,
            test=test_result,
            reliability=audit,
            reliability_matrix=reliability_matrix.numpy(),
            history=self.history,
            prior_delta=separation_report(self.priors.observed).delta,
            run_dir=self.run_dir,
        )
        self._save(result)
        return result

    # ------------------------------------------------------------------ #
    # Bookkeeping
    # ------------------------------------------------------------------ #
    def _track_best(self, result: EvaluationResult, epoch: int) -> None:
        """Checkpoint selection.

        ``eval.monitor == "last"`` keeps the final epoch, which is the safest
        choice when no label-only validation signal is trusted.  Selecting on a
        concept metric would leak the evaluation-only concept annotations into
        training and is flagged loudly here.
        """
        monitor = self.config.eval.monitor
        if monitor in {"last", "none", ""}:
            self.state.best_epoch = epoch
            self.state.best_state = copy.deepcopy(self.model.state_dict())
            return

        metrics = result.as_dict("val/")
        if "concept" in monitor and self.config.loss.lambda_concept == 0:
            logger.warning(
                "eval.monitor='%s' selects checkpoints with concept labels, which are supposed "
                "to be evaluation-only; report this as an upper bound, not as R-PCP.",
                monitor,
            )
        value = metrics.get(monitor, result.classification.macro_f1)
        if not np.isfinite(value):
            return

        sign = 1.0 if self.config.eval.monitor_mode == "max" else -1.0
        score = sign * value
        if score > self.state.best_score:
            self.state.best_score = score
            self.state.best_metric = value
            self.state.best_epoch = epoch
            self.state.best_state = copy.deepcopy(self.model.state_dict())

    def _should_stop(self, epoch: int) -> bool:
        patience = self.config.optim.early_stopping_patience
        if patience is None or self.state.best_epoch < 0:
            return False
        # Never stop before reliability weighting has had a chance to act.
        if self.schedule.phase(epoch) is not Phase.WEIGHTED:
            return False
        return epoch - self.state.best_epoch >= patience

    def _save(self, result: TrainingResult) -> None:
        self.history.to_jsonl(self.run_dir / "history.jsonl")
        save_json(
            {
                "config": to_dict(self.config),
                "summary": result.summary(),
                "best_epoch": self.state.best_epoch,
                "stopped_early": self.state.stopped_early,
                "split_sizes": self.splits.sizes(),
            },
            self.run_dir / "summary.json",
        )
        np.save(self.run_dir / "reliability.npy", result.reliability_matrix)
        np.save(self.run_dir / "prior_observed.npy", self.priors.observed.numpy())
        if self.priors.clean is not None:
            np.save(self.run_dir / "prior_clean.npy", self.priors.clean.numpy())
        if self.priors.clean_mask is not None:
            np.save(self.run_dir / "corruption_mask.npy", (~self.priors.clean_mask).numpy())
        if self.config.eval.save_reliability_history and self.state.reliability_history:
            np.save(
                self.run_dir / "reliability_history.npy",
                np.stack(self.state.reliability_history),
            )
        if self.config.eval.save_checkpoints:
            torch.save(
                {
                    "model": self.model.state_dict(),
                    "reliability": result.reliability_matrix,
                    "config": to_dict(self.config),
                    "epoch": self.state.best_epoch,
                },
                self.run_dir / "checkpoint.pt",
            )
        logger.info("Saved run artefacts to %s", self.run_dir)

    # ------------------------------------------------------------------ #
    def predict(self, split: str = "test") -> Any:
        return self.evaluate(split, keep_predictions=True).predictions
