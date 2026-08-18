"""Training phases, optimisers and LR schedules (plan 5.1).

Phases
------
======================  ==================================================
Phase 0 ``warmup``      PCP-style training with ``r = 1``; class means warm up.
Phase 1 ``estimate``    Reliability is *estimated* every ``update_every``
                        epochs but the prior loss is still unweighted.
Phase 2 ``weighted``    Reliability-weighted prior loss; updates continue at a
                        lower frequency.
Phase 3 ``final``       Handled outside the loop: freeze, evaluate, audit.
======================  ==================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from rpcp.config import OptimConfig, ReliabilityConfig
from rpcp.utils.logging import get_logger

__all__ = ["Phase", "PhaseSchedule", "build_optimizer", "build_scheduler", "ema_update"]

logger = get_logger(__name__)


class Phase(StrEnum):
    WARMUP = "warmup"
    ESTIMATE = "estimate"
    WEIGHTED = "weighted"


@dataclass(slots=True)
class PhaseSchedule:
    """Maps epoch index -> training phase and reliability actions.

    Args:
        warmup_epochs: ``T_warmup``; end of Phase 0.
        mid_epochs: ``T_mid``; start of reliability-weighted training.
        total_epochs: ``T_final``.
        update_every: ``E_freq`` during Phase 1.
        late_update_every: Update frequency during Phase 2 (defaults to
            ``2 * update_every``, i.e. "lower frequency" as the plan requires).
    """

    warmup_epochs: int = 5
    mid_epochs: int = 10
    total_epochs: int = 30
    update_every: int = 1
    late_update_every: int | None = None

    def __post_init__(self) -> None:
        if self.mid_epochs < self.warmup_epochs:
            raise ValueError("mid_epochs must be >= warmup_epochs")
        if self.total_epochs > 0 and self.mid_epochs >= self.total_epochs:
            # epoch indices run 0..total_epochs-1, and phase(e) only returns
            # WEIGHTED once e >= mid_epochs -- so mid_epochs == total_epochs
            # (permitted by the old `<=` check here) means the WEIGHTED phase
            # never runs. The trainer would then quietly train as plain PCP
            # while its config, checkpoints and summary.json all say R-PCP.
            raise ValueError(
                "mid_epochs must be < total_epochs, or the WEIGHTED phase never "
                f"runs (mid_epochs={self.mid_epochs}, total_epochs={self.total_epochs})"
            )

    # ------------------------------------------------------------------ #
    def phase(self, epoch: int) -> Phase:
        if epoch < self.warmup_epochs:
            return Phase.WARMUP
        if epoch < self.mid_epochs:
            return Phase.ESTIMATE
        return Phase.WEIGHTED

    def should_update_reliability(self, epoch: int) -> bool:
        """Whether to recompute evidence and EMA-update ``r`` at this epoch."""
        match self.phase(epoch):
            case Phase.WARMUP:
                return False
            case Phase.ESTIMATE:
                return (epoch - self.warmup_epochs) % max(1, self.update_every) == 0
            case Phase.WEIGHTED:
                every = self.late_update_every or 2 * max(1, self.update_every)
                return (epoch - self.mid_epochs) % max(1, every) == 0

    def use_reliability(self, epoch: int) -> bool:
        """Whether the prior loss is reliability-weighted at this epoch."""
        return self.phase(epoch) is Phase.WEIGHTED

    @classmethod
    def from_configs(cls, optim: OptimConfig, reliability: ReliabilityConfig) -> PhaseSchedule:
        """Build a schedule that fits ``optim.epochs``, without ever silently
        eliminating the WEIGHTED phase.

        ``reliability.warmup_epochs``/``mid_epochs`` are configured
        independently of ``optim.epochs`` (so the same reliability schedule can
        be reused across a quick-iteration config and a full run). The old
        ``min(reliability.mid_epochs, optim.epochs)`` clamp could set
        ``mid_epochs == total_epochs``, which leaves zero WEIGHTED epochs: the
        run would train as plain PCP -- ``use_reliability`` is ``False`` for
        every epoch -- while still reporting itself as R-PCP and writing a
        ``reliability.npy`` / ``reliability/auroc`` that never influenced
        training. Clamping to ``optim.epochs - 1`` guarantees at least the
        final epoch is WEIGHTED whenever ``optim.epochs >= 1``.
        """
        total = optim.epochs
        mid = min(reliability.mid_epochs, max(0, total - 1))
        warmup = min(reliability.warmup_epochs, mid)
        if mid != reliability.mid_epochs or warmup != reliability.warmup_epochs:
            logger.warning(
                "reliability.{warmup_epochs=%d, mid_epochs=%d} do not fit "
                "optim.epochs=%d; clamped to {warmup_epochs=%d, mid_epochs=%d} "
                "so at least the last epoch is reliability-weighted. Increase "
                "optim.epochs or lower reliability.mid_epochs to configure this "
                "explicitly instead of relying on the clamp.",
                reliability.warmup_epochs,
                reliability.mid_epochs,
                total,
                warmup,
                mid,
            )
        return cls(
            warmup_epochs=warmup,
            mid_epochs=mid,
            total_epochs=total,
            update_every=reliability.update_every,
        )


def ema_update(previous: torch.Tensor, new: torch.Tensor, gamma: float) -> torch.Tensor:
    """``r_t = gamma * r_{t-1} + (1 - gamma) * r_new`` (plan 5.1, Phase 1)."""
    if not 0.0 <= gamma <= 1.0:
        raise ValueError(f"gamma must be in [0, 1], got {gamma}")
    return gamma * previous + (1.0 - gamma) * new


def build_optimizer(model: nn.Module, config: OptimConfig) -> Optimizer:
    """AdamW / Adam / SGD over parameter groups (backbone may use its own LR)."""
    if hasattr(model, "parameter_groups"):
        groups = model.parameter_groups(  # type: ignore[operator]
            config.lr, config.backbone_lr, config.weight_decay
        )
    else:
        groups = [
            {
                "params": [p for p in model.parameters() if p.requires_grad],
                "lr": config.lr,
                "weight_decay": config.weight_decay,
            }
        ]

    match config.optimizer.lower():
        case "adamw":
            return torch.optim.AdamW(groups, lr=config.lr, weight_decay=config.weight_decay)
        case "adam":
            return torch.optim.Adam(groups, lr=config.lr, weight_decay=config.weight_decay)
        case "sgd":
            return torch.optim.SGD(
                groups,
                lr=config.lr,
                momentum=config.momentum,
                weight_decay=config.weight_decay,
                nesterov=True,
            )
        case name:
            raise ValueError(f"Unknown optimizer '{name}'")


def build_scheduler(optimizer: Optimizer, config: OptimConfig) -> LRScheduler | None:
    match config.scheduler.lower():
        case "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
        case "step":
            return torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=config.step_size, gamma=config.gamma
            )
        case "none":
            return None
        case name:
            raise ValueError(f"Unknown scheduler '{name}'")
