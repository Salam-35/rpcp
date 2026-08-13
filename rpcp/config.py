"""Typed configuration schema for R-PCP experiments.

Every experiment is fully described by :class:`ExperimentConfig`, which is
serialised to/from YAML (see ``configs/``).  Keeping the schema typed means a
config typo fails at load time instead of silently changing an experiment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from rpcp.utils.io import deep_update, from_dict, load_yaml, to_dict

__all__ = [
    "ClassHead",
    "CorruptionConfig",
    "CorruptionMode",
    "DataConfig",
    "EvalConfig",
    "ExperimentConfig",
    "LossConfig",
    "ModelConfig",
    "OptimConfig",
    "PriorConfig",
    "PriorLossType",
    "ReliabilityConfig",
    "ReliabilityMode",
    "SyntheticConfig",
    "load_config",
]


# --------------------------------------------------------------------------- #
# Enums
# --------------------------------------------------------------------------- #
class CorruptionMode(StrEnum):
    """Prior corruption modes of plan section 6.3."""

    NONE = "none"
    UNIFORM = "uniform"
    BACKGROUND_COLLAPSE = "background_collapse"
    CLASS_SWAP = "class_swap"
    ADVERSARIAL_FLIP = "adversarial_flip"
    LLM_BIAS = "llm_bias"


class ReliabilityMode(StrEnum):
    """Reliability evidence modes of plan section 4.3."""

    NONE = "none"  # PCP baseline: r == 1 everywhere
    UNSUPERVISED = "unsupervised"  # Evidence Mode A (model residual + instability)
    MULTI_SOURCE = "multi_source"  # Evidence Mode B
    AUDIT = "audit"  # Evidence Mode C
    MULTI_RATER = "multi_rater"  # Evidence Mode D
    ORACLE = "oracle"  # upper bound: true corruption mask


class PriorLossType(StrEnum):
    """Which prior-matching divergence to use (plan section 4.2)."""

    BERNOULLI = "bernoulli"  # R-PCP-BernKL
    PCP_KL = "pcp_kl"  # R-PCP-PCPKL (grouped one-sided KL)


class ClassHead(StrEnum):
    """How class logits are produced from the concept layer."""

    LINEAR = "linear"  # standard concept bottleneck
    PRIOR = "prior"  # prior-similarity classifier only
    HYBRID = "hybrid"  # linear head + prior-similarity head (averaged logits)


# --------------------------------------------------------------------------- #
# Sub-configs
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class SyntheticConfig:
    """Synthetic concept-image generator (used for smoke tests and unit tests)."""

    n_train: int = 1200
    n_val: int = 300
    n_test: int = 300
    n_classes: int = 3
    n_concepts: int = 8
    image_size: int = 32
    concept_visibility: float = 0.85
    pixel_noise: float = 0.15
    prior_sharpness: float = 0.75
    class_shortcut: float = 0.0  # >0 lets the model classify without concepts
    seed: int = 0


@dataclass(slots=True)
class DataConfig:
    name: str = "synthetic"
    root: str | None = None
    manifest: str | None = None
    image_size: int = 224
    batch_size: int = 32
    eval_batch_size: int = 64
    num_workers: int = 0
    pin_memory: bool = False
    augment: bool = True
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    audit_fraction: float = 0.0
    split_seed: int = 0
    synthetic: SyntheticConfig = field(default_factory=SyntheticConfig)


@dataclass(slots=True)
class ModelConfig:
    backbone: str = "resnet18"
    pretrained: bool = False
    in_channels: int = 3
    use_attention: bool = True
    concept_hidden_dim: int = 0
    dropout: float = 0.0
    class_head: ClassHead = ClassHead.LINEAR
    class_head_bias: bool = True
    freeze_backbone: bool = False
    attention_temperature: float = 1.0


@dataclass(slots=True)
class CorruptionConfig:
    mode: CorruptionMode = CorruptionMode.NONE
    alpha: float = 0.0
    fraction: float = 0.3
    seed: int = 0
    llm_bias_strength: float = 0.6
    tolerance: float = 1e-3


@dataclass(slots=True)
class PriorConfig:
    """How the (possibly corrupted) prior table is built."""

    source: str = "dataset"  # dataset | file | multi_source
    path: str | None = None
    multi_source_paths: list[str] = field(default_factory=list)
    n_synthetic_sources: int = 0
    synthetic_source_noise: float = 0.08
    laplace_smoothing: float = 1e-3
    clip_min: float = 1e-3
    clip_max: float = 1 - 1e-3
    corruption: CorruptionConfig = field(default_factory=CorruptionConfig)
    # Delta-sweep (plan 6.4): blend prior column `blend_target` into `blend_source`.
    blend_alpha: float = 0.0
    blend_source: int = 0
    blend_target: int = 1


@dataclass(slots=True)
class ReliabilityConfig:
    mode: ReliabilityMode = ReliabilityMode.NONE
    # r = sigmoid(w0 + w1*agree - w2*source_disagree - w3*instability - w4*residual)
    #
    # External agreement (w1) is weighted ABOVE the model-prior residual (w4) on
    # purpose: the residual is the one term the model can talk itself into
    # (plan 13, Risk 1), so it must not be able to override audit, multi-source
    # or multi-rater evidence.
    w0: float = 1.0
    w1: float = 6.0
    w2: float = 4.0
    w3: float = 3.0
    w4: float = 3.0
    learnable_weights: bool = False
    ema_gamma: float = 0.7
    update_every: int = 1
    warmup_epochs: int = 5
    mid_epochs: int = 10
    min_reliability: float = 0.05
    max_reliability: float = 1.0
    hard_threshold: float | None = None  # if set, binarise r at this value
    beta_a0: float = 2.0
    beta_b0: float = 2.0
    lambda_r: float = 0.0
    audit_beta: float = 5.0
    #: Absolute prior/audit gap tolerated before an entry is called unreliable;
    #: absorbs the sampling noise of a small audit split.
    audit_tolerance: float = 0.05
    #: Agreement enters the score centred: values above this push r up, below it down.
    agreement_center: float = 0.5
    source_alpha: float = 8.0
    residual_scale: float = 1.0
    instability_scale: float = 1.0
    use_crossfit: bool = True
    n_folds: int = 2
    n_instability_views: int = 2
    oracle_clean_value: float = 1.0
    oracle_corrupt_value: float = 0.0


@dataclass(slots=True)
class LossConfig:
    lambda_cls: float = 1.0
    lambda_match: float = 0.5
    lambda_prior: float = 1.0
    lambda_ent: float = 0.05
    lambda_r: float = 0.01
    #: Per-image concept supervision. Non-zero ONLY for the supervised-CBM upper
    #: bound of plan 6.5; every other method must leave this at 0 so that no
    #: per-image concept label touches training.
    lambda_concept: float = 0.0
    prior_loss: PriorLossType = PriorLossType.BERNOULLI
    prior_reduction: str = "mean"  # "sum" is the plan's literal objective
    normalize_prior_by_reliability: bool = True
    concept_groups: list[list[int]] | None = None
    match_temperature: float = 1.0
    match_reliability_weighted: bool = False
    entropy_on_attention: bool = True
    entropy_on_concepts: bool = True
    label_smoothing: float = 0.0
    class_mean_momentum: float = 0.9
    eps: float = 1e-6


@dataclass(slots=True)
class OptimConfig:
    optimizer: str = "adamw"
    lr: float = 1e-3
    backbone_lr: float | None = None
    weight_decay: float = 1e-4
    momentum: float = 0.9
    epochs: int = 30
    scheduler: str = "cosine"  # cosine | step | none
    step_size: int = 10
    gamma: float = 0.1
    grad_clip: float | None = 5.0
    amp: bool = False
    early_stopping_patience: int | None = None


@dataclass(slots=True)
class EvalConfig:
    concept_threshold: float = 0.5
    n_calibration_bins: int = 15
    #: Checkpoint-selection metric.  It must NOT depend on per-image concept
    #: labels: those are evaluation-only, so selecting on ``val/concept_macro_f1``
    #: leaks concept supervision into training.  Use ``"last"`` to keep the final
    #: epoch, or a concept metric only for deliberate upper-bound analyses.
    monitor: str = "val/class_macro_f1"
    monitor_mode: str = "max"
    save_checkpoints: bool = True
    save_reliability_history: bool = True


@dataclass(slots=True)
class ExperimentConfig:
    name: str = "rpcp"
    output_dir: str = "runs"
    seed: int = 0
    device: str = "auto"
    tag: str = ""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    priors: PriorConfig = field(default_factory=PriorConfig)
    reliability: ReliabilityConfig = field(default_factory=ReliabilityConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    # ------------------------------------------------------------------ #
    @property
    def run_dir(self) -> Path:
        suffix = f"-{self.tag}" if self.tag else ""
        return Path(self.output_dir) / f"{self.name}{suffix}"

    def resolved_device(self) -> str:
        if self.device != "auto":
            return self.device
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"

    def to_dict(self) -> dict[str, Any]:
        return to_dict(self)

    def replace(self, **overrides: Any) -> ExperimentConfig:
        """Return a new config with (possibly nested, dotted) overrides applied.

        Example::

            cfg.replace(**{"priors.corruption.alpha": 0.5, "name": "sweep"})
        """
        payload = to_dict(self)
        for key, value in overrides.items():
            node = payload
            *parents, leaf = key.split(".")
            for part in parents:
                node = node[part]
            node[leaf] = value
        return from_dict(ExperimentConfig, payload)


def load_config(
    path: str | Path | None = None,
    *,
    overrides: dict[str, Any] | None = None,
) -> ExperimentConfig:
    """Load an :class:`ExperimentConfig` from YAML, applying optional overrides.

    ``overrides`` accepts either a nested mapping or dotted keys
    (``{"optim.lr": 1e-4}``).
    """
    payload: dict[str, Any] = load_yaml(path) if path is not None else {}
    if base := payload.pop("_base_", None):
        base_path = (Path(path).parent / base) if path else Path(base)
        payload = deep_update(load_yaml(base_path), payload)
    if overrides:
        nested: dict[str, Any] = {}
        for key, value in overrides.items():
            node = nested
            *parents, leaf = key.split(".")
            for part in parents:
                node = node.setdefault(part, {})
            node[leaf] = value
        payload = deep_update(payload, nested)
    return from_dict(ExperimentConfig, payload)
