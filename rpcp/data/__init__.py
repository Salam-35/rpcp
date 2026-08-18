"""Datasets, prior tables and prior corruption.

The two entry points used by everything else are :func:`build_splits` (images)
and :func:`build_prior_bundle` (the ``Pi`` tables plus every reliability
evidence signal available for the chosen mode).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch

from rpcp.config import DataConfig, ExperimentConfig, PriorConfig, ReliabilityMode
from rpcp.data.base import (
    ConceptBatch,
    ConceptDataset,
    SplitBundle,
    TransformedSubset,
    collate_concepts,
    make_dataloader,
    stratified_split,
)
from rpcp.data.corruption import CorruptionResult, corrupt_priors
from rpcp.data.manifest import ManifestConceptDataset
from rpcp.data.priors import (
    PriorBundle,
    audit_prevalence,
    blend_prior_columns,
    compute_priors_from_annotations,
    load_prior_table,
    multi_rater_agreement,
    synthesize_prior_sources,
)
from rpcp.data.synthetic import SyntheticConceptDataset, make_synthetic_world
from rpcp.data.transforms import build_transform
from rpcp.utils.logging import get_logger

__all__ = [
    "ConceptBatch",
    "ConceptDataset",
    "CorruptionResult",
    "DATASET_REGISTRY",
    "ManifestConceptDataset",
    "PriorBundle",
    "SplitBundle",
    "SyntheticConceptDataset",
    "TransformedSubset",
    "build_dataloaders",
    "build_prior_bundle",
    "build_splits",
    "collate_concepts",
    "compute_priors_from_annotations",
    "corrupt_priors",
    "make_dataloader",
]

logger = get_logger(__name__)

DatasetBuilder = Callable[[DataConfig], ConceptDataset]


def _build_synthetic(config: DataConfig) -> ConceptDataset:
    """Synthetic splits are concatenated into one dataset with fixed splits."""
    world = make_synthetic_world(config.synthetic)
    total = config.synthetic.n_train + config.synthetic.n_val + config.synthetic.n_test
    dataset = SyntheticConceptDataset(
        world,
        n_samples=total,
        seed=config.synthetic.seed + 1,
        class_shortcut=config.synthetic.class_shortcut,
    )
    return dataset


def _build_ph2(config: DataConfig) -> ConceptDataset:
    from rpcp.data.ph2 import build_ph2

    return build_ph2(_require_root(config), manifest=config.manifest)


def _build_wbcatt(config: DataConfig) -> ConceptDataset:
    from rpcp.data.wbcatt import build_wbcatt

    return build_wbcatt(_require_root(config), manifest=config.manifest)


def _build_derm7pt(config: DataConfig) -> ConceptDataset:
    from rpcp.data.derm7pt import build_derm7pt

    return build_derm7pt(_require_root(config), manifest=config.manifest)


def _build_lidc(config: DataConfig) -> ConceptDataset:
    from rpcp.data.lidc import build_lidc

    return build_lidc(_require_root(config), manifest=config.manifest)


DATASET_REGISTRY: dict[str, DatasetBuilder] = {
    "synthetic": _build_synthetic,
    "ph2": _build_ph2,
    "wbcatt": _build_wbcatt,
    "derm7pt": _build_derm7pt,
    "lidc": _build_lidc,
}


def _require_root(config: DataConfig) -> Path:
    if config.root is None:
        raise ValueError(f"datasets.root must be set for dataset '{config.name}'")
    return Path(config.root)


# --------------------------------------------------------------------------- #
# Splits
# --------------------------------------------------------------------------- #
def build_splits(config: DataConfig) -> SplitBundle:
    """Build train/val/test (+ optional audit) splits for the configured dataset.

    The audit split (Evidence Mode C) is carved out of the *training* pool, so
    the number of images the concept predictor sees shrinks accordingly: the
    audit budget is a real annotation budget, not free extra datasets.
    """
    if config.name not in DATASET_REGISTRY:
        raise KeyError(f"Unknown dataset '{config.name}'. Known: {sorted(DATASET_REGISTRY)}")
    dataset = DATASET_REGISTRY[config.name](config)

    predefined = (
        dataset.predefined_splits() if isinstance(dataset, ManifestConceptDataset) else None
    )
    if predefined and {"train", "test"} <= set(predefined):
        indices = {
            "train": predefined["train"],
            "val": predefined.get("val", predefined.get("valid", np.array([], dtype=np.int64))),
            "test": predefined["test"],
        }
        if len(indices["val"]) == 0:  # carve a val split out of train
            sub = stratified_split(
                dataset.labels[indices["train"]],
                {"val": config.val_fraction},
                seed=config.split_seed,
            )
            indices["val"] = indices["train"][sub["val"]]
            indices["train"] = indices["train"][sub["train"]]
        logger.info("Using predefined splits from manifest for '%s'", config.name)
    else:
        fractions = {"val": config.val_fraction, "test": config.test_fraction}
        if isinstance(dataset, SyntheticConceptDataset):
            # Honour the explicit synthetic split sizes instead of the generic fractions.
            synthetic = config.synthetic
            total = synthetic.n_train + synthetic.n_val + synthetic.n_test
            fractions = {"val": synthetic.n_val / total, "test": synthetic.n_test / total}
        indices = stratified_split(dataset.labels, fractions, seed=config.split_seed)

    audit_indices: np.ndarray | None = None
    if config.audit_fraction > 0:
        train_labels = dataset.labels[indices["train"]]
        # Fraction is expressed w.r.t. the whole dataset; convert to a train fraction.
        train_fraction = min(
            0.9, config.audit_fraction * len(dataset.labels) / max(1, len(train_labels))
        )
        sub = stratified_split(train_labels, {"audit": train_fraction}, seed=config.split_seed + 1)
        audit_indices = indices["train"][sub["audit"]]
        indices["train"] = indices["train"][sub["train"]]

    train_transform = build_transform(config.image_size, train=True, augment=config.augment)
    eval_transform = build_transform(config.image_size, train=False)

    bundle = SplitBundle(
        train=TransformedSubset(dataset, indices["train"], train_transform),
        train_eval=TransformedSubset(dataset, indices["train"], eval_transform),
        val=TransformedSubset(dataset, indices["val"], eval_transform),
        test=TransformedSubset(dataset, indices["test"], eval_transform),
        audit=(
            None
            if audit_indices is None
            else TransformedSubset(dataset, audit_indices, eval_transform)
        ),
        concept_names=dataset.concept_names,
        class_names=dataset.class_names,
        source=dataset,
    )
    logger.info("Split sizes for '%s': %s", config.name, bundle.sizes())
    return bundle


def build_dataloaders(
    splits: SplitBundle,
    config: DataConfig,
    *,
    seed: int = 0,
) -> dict[str, object]:
    """Standard loaders: shuffled train, deterministic val/test/audit."""
    loaders: dict[str, object] = {
        "train": make_dataloader(
            splits.train,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            seed=seed,
        ),
        "train_eval": make_dataloader(
            # Same images/order as `train`, deterministic transform: used to
            # measure the model (held-out class-mean residual when
            # `reliability.use_crossfit=False`), never to optimise it.
            splits.train_eval,
            batch_size=config.eval_batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            seed=seed,
        ),
    }
    for name in ("val", "test"):
        loaders[name] = make_dataloader(
            getattr(splits, name),
            batch_size=config.eval_batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            seed=seed,
        )
    if splits.audit is not None:
        loaders["audit"] = make_dataloader(
            splits.audit,
            batch_size=config.eval_batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            seed=seed,
        )
    return loaders


# --------------------------------------------------------------------------- #
# Priors
# --------------------------------------------------------------------------- #
def build_prior_bundle(
    config: ExperimentConfig,
    splits: SplitBundle,
) -> PriorBundle:
    """Assemble the prior table plus every available reliability evidence source.

    Pipeline (plan 6.2 -> 6.4 -> 6.3):

    1. build the clean prior ``Pi_star`` from training concept annotations, or
       load an expert/LLM table from disk;
    2. optionally collapse two class signatures together (Delta sweep);
    3. corrupt a fraction of the entries and record ``s_true``;
    4. attach multi-source / audit / multi-rater evidence.
    """
    prior_cfg: PriorConfig = config.priors
    clean = _base_prior(prior_cfg, splits)

    if prior_cfg.blend_alpha > 0:
        clean = blend_prior_columns(
            clean,
            prior_cfg.blend_alpha,
            source=prior_cfg.blend_source,
            target=prior_cfg.blend_target,
        )

    corruption = corrupt_priors(
        clean,
        mode=prior_cfg.corruption.mode,
        alpha=prior_cfg.corruption.alpha,
        fraction=prior_cfg.corruption.fraction,
        seed=prior_cfg.corruption.seed,
        llm_bias_strength=prior_cfg.corruption.llm_bias_strength,
        tolerance=prior_cfg.corruption.tolerance,
        clip=(prior_cfg.clip_min, prior_cfg.clip_max),
    )
    logger.info(
        "Prior table: mode=%s alpha=%.2f fraction=%.2f -> %.1f%% of entries corrupted",
        prior_cfg.corruption.mode,
        prior_cfg.corruption.alpha,
        prior_cfg.corruption.fraction,
        100 * corruption.corrupted_fraction,
    )

    bundle = PriorBundle(
        observed=corruption.priors,
        clean=clean,
        clean_mask=corruption.clean_mask,
        concept_names=list(splits.concept_names),
        class_names=list(splits.class_names),
    )

    bundle.sources = _prior_sources(prior_cfg, bundle, corruption)
    bundle.audit, bundle.audit_support = _audit_prior(config, splits)
    bundle.rater_agreement = _rater_agreement(splits)
    return bundle


def _base_prior(prior_cfg: PriorConfig, splits: SplitBundle) -> torch.Tensor:
    if prior_cfg.source == "file":
        if prior_cfg.path is None:
            raise ValueError("priors.path must be set when priors.source == 'file'")
        table, _, _ = load_prior_table(
            prior_cfg.path,
            concept_names=splits.concept_names,
            class_names=splits.class_names,
        )
        return table.clamp(prior_cfg.clip_min, prior_cfg.clip_max)

    train = splits.train
    concepts = getattr(train, "concepts", None)
    if concepts is None:
        raise ValueError(
            "priors.source == 'dataset' requires concept annotations on the training split; "
            "supply an expert prior table with priors.source == 'file' instead."
        )
    return compute_priors_from_annotations(
        concepts,
        train.labels,
        n_classes=splits.n_classes,
        smoothing=prior_cfg.laplace_smoothing,
        clip_min=prior_cfg.clip_min,
        clip_max=prior_cfg.clip_max,
    )


def _prior_sources(
    prior_cfg: PriorConfig,
    bundle: PriorBundle,
    corruption: CorruptionResult,
) -> torch.Tensor | None:
    if prior_cfg.multi_source_paths:
        tables = [
            load_prior_table(
                path,
                concept_names=bundle.concept_names,
                class_names=bundle.class_names,
            )[0]
            for path in prior_cfg.multi_source_paths
        ]
        return torch.stack(tables)
    if prior_cfg.n_synthetic_sources > 0:
        return synthesize_prior_sources(
            bundle.observed,
            n_sources=prior_cfg.n_synthetic_sources,
            noise=prior_cfg.synthetic_source_noise,
            corrupted_mask=corruption.corruption_mask,
            seed=prior_cfg.corruption.seed + 7,
        )
    return None


def _audit_prior(
    config: ExperimentConfig, splits: SplitBundle
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Audit-split prevalence and its per-``(concept, class)`` support mask.

    The support mask matters: with a small audit split (e.g. 10% of PH2's 40
    melanoma cases), a rare class can easily draw zero audit examples, and
    :func:`audit_prevalence` then falls back to a 0.5 placeholder for that
    column. Discarding the mask -- as this used to do -- makes
    :func:`reliability_from_audit` read that placeholder as "the prior and the
    audit strongly disagree" and flag every entry in the column as unreliable,
    from nothing but sampling noise.
    """
    if splits.audit is None:
        return None, None
    if config.reliability.mode not in {ReliabilityMode.AUDIT, ReliabilityMode.ORACLE}:
        logger.info("Audit split present but reliability.mode=%s", config.reliability.mode)
    concepts = getattr(splits.audit, "concepts", None)
    if concepts is None:
        return None, None
    prevalence, support = audit_prevalence(
        concepts,
        splits.audit.labels,
        n_classes=splits.n_classes,
        smoothing=config.priors.laplace_smoothing,
    )
    return prevalence, support


def _rater_agreement(splits: SplitBundle) -> torch.Tensor | None:
    source = splits.source
    votes = getattr(source, "rater_votes", None)
    if votes is None:
        return None
    return multi_rater_agreement(votes, source.labels, n_classes=source.n_classes)
