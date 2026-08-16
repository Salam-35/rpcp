"""Class-level concept prior tables.

A prior table is a tensor ``Pi`` of shape ``(M, K)`` with

.. math::  \\tilde\\Pi[m, y] \\approx P(c_m = 1 \\mid y).

This module builds prior tables from concept annotations (plan 6.2), loads
expert/LLM tables from disk, derives multi-source disagreement (Evidence Mode B),
audit prevalence (Evidence Mode C) and multi-rater agreement (Evidence Mode D),
and implements the class-signature blending used by the Delta sweep (plan 6.4).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from rpcp.data.base import ConceptDataset, read_table

__all__ = [
    "PriorBundle",
    "audit_prevalence",
    "blend_prior_columns",
    "compute_priors_from_annotations",
    "load_prior_table",
    "multi_rater_agreement",
    "reliability_from_audit",
    "reliability_from_sources",
    "save_prior_table",
    "source_disagreement",
    "synthesize_prior_sources",
]


@dataclass(slots=True)
class PriorBundle:
    """Everything the trainer needs to know about the prior table.

    Attributes:
        observed: ``(M, K)`` the (possibly corrupted) prior actually used for
            supervision -- ``Pi_tilde`` in the plan.
        clean: ``(M, K)`` the uncorrupted reference prior ``Pi_star`` when known
            (benchmark setting only; never used for training).
        clean_mask: ``(M, K)`` boolean, ``True`` where the observed entry is
            uncorrupted (``s_true`` of plan 3.4/6.3).  Evaluation only.
        sources: ``(S, M, K)`` optional stack of prior sources (Evidence Mode B).
        audit: ``(M, K)`` optional prevalence estimated on the audit split
            (Evidence Mode C).
        rater_agreement: ``(M, K)`` optional inter-rater agreement (Mode D).
    """

    observed: torch.Tensor
    clean: torch.Tensor | None = None
    clean_mask: torch.Tensor | None = None
    sources: torch.Tensor | None = None
    audit: torch.Tensor | None = None
    rater_agreement: torch.Tensor | None = None
    concept_names: list[str] = field(default_factory=list)
    class_names: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.observed.ndim != 2:
            raise ValueError(f"observed prior must be (M, K), got {tuple(self.observed.shape)}")
        for name in ("clean", "audit", "rater_agreement"):
            value = getattr(self, name)
            if value is not None and value.shape != self.observed.shape:
                raise ValueError(f"{name} shape {tuple(value.shape)} != {tuple(self.shape)}")

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(self.observed.shape)  # type: ignore[return-value]

    @property
    def n_concepts(self) -> int:
        return int(self.observed.shape[0])

    @property
    def n_classes(self) -> int:
        return int(self.observed.shape[1])

    @property
    def corruption_mask(self) -> torch.Tensor | None:
        """``True`` where the entry *is corrupted* (complement of ``clean_mask``)."""
        return None if self.clean_mask is None else ~self.clean_mask

    def prior_error(self) -> torch.Tensor | None:
        """``|Pi_tilde - Pi_star|`` when the clean prior is known."""
        return None if self.clean is None else (self.observed - self.clean).abs()

    def to(self, device: torch.device | str) -> PriorBundle:
        move = lambda t: None if t is None else t.to(device)  # noqa: E731
        return PriorBundle(
            observed=self.observed.to(device),
            clean=move(self.clean),
            clean_mask=move(self.clean_mask),
            sources=move(self.sources),
            audit=move(self.audit),
            rater_agreement=move(self.rater_agreement),
            concept_names=list(self.concept_names),
            class_names=list(self.class_names),
        )


# --------------------------------------------------------------------------- #
# Construction
# --------------------------------------------------------------------------- #
def compute_priors_from_annotations(
    concepts: np.ndarray | torch.Tensor | ConceptDataset,
    labels: np.ndarray | torch.Tensor | None = None,
    *,
    n_classes: int | None = None,
    smoothing: float = 1e-3,
    clip_min: float = 1e-3,
    clip_max: float = 1 - 1e-3,
) -> torch.Tensor:
    """Class-conditional concept prevalence (plan 6.2).

    .. math:: \\Pi_{true}[m, y] = \\frac{1}{|\\{i: y_i = y\\}|}\\sum_{i: y_i=y} c_i[m]

    Args:
        concepts: ``(N, M)`` binary concept matrix, or a :class:`ConceptDataset`
            carrying one.
        labels: ``(N,)`` class labels (ignored if a dataset is passed).
        n_classes: Number of classes ``K``; inferred when omitted.
        smoothing: Laplace smoothing added to numerator/denominator, which keeps
            rare classes away from exact 0/1.
        clip_min / clip_max: Final clipping range, so log-domain losses stay finite.

    Returns:
        ``(M, K)`` float tensor.
    """
    if isinstance(concepts, ConceptDataset):
        dataset = concepts
        if dataset.concepts is None:
            raise ValueError(f"{type(dataset).__name__} has no concept annotations")
        concept_array = np.asarray(dataset.concepts, dtype=np.float64)
        label_array = np.asarray(dataset.labels, dtype=np.int64)
        n_classes = n_classes or dataset.n_classes
    else:
        if labels is None:
            raise ValueError("labels are required when concepts is an array")
        concept_array = np.asarray(
            concepts.cpu() if isinstance(concepts, torch.Tensor) else concepts, dtype=np.float64
        )
        label_array = np.asarray(
            labels.cpu() if isinstance(labels, torch.Tensor) else labels, dtype=np.int64
        )

    n_classes = int(n_classes or label_array.max() + 1)
    n_concepts = concept_array.shape[1]
    priors = np.full((n_concepts, n_classes), np.nan, dtype=np.float64)

    for y in range(n_classes):
        rows = concept_array[label_array == y]
        if rows.size == 0:
            priors[:, y] = 0.5  # uninformative fallback for empty classes
            continue
        priors[:, y] = (rows.sum(axis=0) + smoothing) / (len(rows) + 2.0 * smoothing)

    return torch.from_numpy(np.clip(priors, clip_min, clip_max)).float()


def audit_prevalence(
    concepts: np.ndarray | torch.Tensor,
    labels: np.ndarray | torch.Tensor,
    *,
    n_classes: int,
    smoothing: float = 1e-3,
    min_count: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prevalence on a small labelled audit split (Evidence Mode C, plan 4.3).

    .. math:: \\Pi_{audit}[m, y] = \\mathrm{mean}_{i \\in audit,\\; y_i = y} c_i[m]

    Returns:
        ``(prevalence, support_mask)`` where ``support_mask[m, y]`` is ``True``
        when class ``y`` had at least ``min_count`` audit images (the prevalence
        of unsupported classes is set to 0.5 and should be ignored).
    """
    labels_np = np.asarray(labels.cpu() if isinstance(labels, torch.Tensor) else labels)
    counts = np.bincount(labels_np.astype(np.int64), minlength=n_classes)
    prevalence = compute_priors_from_annotations(
        concepts, labels_np, n_classes=n_classes, smoothing=smoothing
    )
    support = torch.from_numpy(counts >= min_count)
    mask = support.unsqueeze(0).expand_as(prevalence).clone()
    prevalence = torch.where(mask, prevalence, torch.full_like(prevalence, 0.5))
    return prevalence, mask


def multi_rater_agreement(
    rater_concepts: np.ndarray | torch.Tensor,
    labels: np.ndarray | torch.Tensor,
    *,
    n_classes: int,
) -> torch.Tensor:
    """Inter-rater agreement per (concept, class) -- Evidence Mode D (plan 4.3).

    Agreement is the mean pairwise concordance of binary rater votes::

        r_true[m, y] = mean_{i: y_i = y} mean_{r != r'} 1[c_i^{(r)}[m] == c_i^{(r')}[m]]

    Args:
        rater_concepts: ``(N, R, M)`` binary votes; NaN entries are ignored.
        labels: ``(N,)`` class labels.
        n_classes: Number of classes.

    Returns:
        ``(M, K)`` agreement in ``[0, 1]``.
    """
    votes = np.asarray(
        rater_concepts.cpu() if isinstance(rater_concepts, torch.Tensor) else rater_concepts,
        dtype=np.float64,
    )
    labels_np = np.asarray(labels.cpu() if isinstance(labels, torch.Tensor) else labels)
    if votes.ndim != 3:
        raise ValueError(f"rater_concepts must be (N, R, M), got {votes.shape}")

    n_concepts = votes.shape[2]
    agreement = np.full((n_concepts, n_classes), 0.5)
    for y in range(n_classes):
        block = votes[labels_np == y]  # (n_y, R, M)
        if block.size == 0:
            continue
        # Fraction voting positive per (sample, concept), NaN-aware.
        with np.errstate(invalid="ignore"):
            p = np.nanmean(block, axis=1)  # (n_y, M)
        # Pairwise concordance for binary votes: p^2 + (1-p)^2 (chance-adjusted below).
        concordance = p**2 + (1.0 - p) ** 2
        agreement[:, y] = np.nanmean(concordance, axis=0)
    return torch.from_numpy(agreement).float()


# --------------------------------------------------------------------------- #
# Multi-source priors (Evidence Mode B)
# --------------------------------------------------------------------------- #
def source_disagreement(sources: torch.Tensor, *, unbiased: bool = False) -> torch.Tensor:
    """``u[m, y] = Var_s Pi^{(s)}[m, y]`` across ``S`` prior sources."""
    if sources.ndim != 3:
        raise ValueError(f"sources must be (S, M, K), got {tuple(sources.shape)}")
    if sources.shape[0] < 2:
        return torch.zeros(sources.shape[1:], dtype=sources.dtype, device=sources.device)
    return sources.var(dim=0, unbiased=unbiased)


def reliability_from_sources(disagreement: torch.Tensor, alpha: float = 8.0) -> torch.Tensor:
    """``r_0[m, y] = exp(-alpha * u[m, y])`` (plan 4.3, Evidence Mode B)."""
    return torch.exp(-alpha * disagreement.clamp_min(0.0))


def reliability_from_audit(
    observed: torch.Tensor,
    audit: torch.Tensor,
    beta: float = 5.0,
    *,
    tolerance: float = 0.0,
    support_mask: torch.Tensor | None = None,
    default: float = 0.5,
) -> torch.Tensor:
    """``r_audit[m, y] = exp(-beta * |Pi_tilde[m, y] - Pi_audit[m, y]|)`` (Mode C).

    ``tolerance`` subtracts an allowance for the sampling noise of a small audit
    split before the exponential, so that a 5%-of-training audit does not flag
    every entry merely because its prevalence estimate is noisy.
    """
    deviation = ((observed - audit).abs() - tolerance).clamp_min(0.0)
    reliability = torch.exp(-beta * deviation)
    if support_mask is not None:
        reliability = torch.where(support_mask, reliability, torch.full_like(reliability, default))
    return reliability


def synthesize_prior_sources(
    prior: torch.Tensor,
    *,
    n_sources: int,
    noise: float = 0.08,
    corrupted_mask: torch.Tensor | None = None,
    corrupted_noise_scale: float = 3.0,
    seed: int = 0,
) -> torch.Tensor:
    """Simulate ``S`` noisy prior sources around ``prior``.

    Used to study Evidence Mode B without collecting several real expert tables.
    Entries flagged by ``corrupted_mask`` (``True`` = corrupted) receive larger
    across-source noise, which is exactly the signal that mode is meant to
    exploit: independent sources disagree where the table is unreliable.
    """
    generator = torch.Generator(device="cpu").manual_seed(seed)
    scale = torch.full_like(prior, noise)
    if corrupted_mask is not None:
        scale = torch.where(corrupted_mask, scale * corrupted_noise_scale, scale)
    samples = prior.unsqueeze(0) + scale.unsqueeze(0) * torch.randn(
        (n_sources, *prior.shape), generator=generator
    )
    return samples.clamp(1e-3, 1 - 1e-3)


# --------------------------------------------------------------------------- #
# Delta sweep helper (plan 6.4)
# --------------------------------------------------------------------------- #
def blend_prior_columns(
    prior: torch.Tensor,
    alpha: float,
    *,
    source: int = 0,
    target: int = 1,
) -> torch.Tensor:
    """Collapse two class signatures together.

    .. math:: \\Pi_\\alpha[:, y_2] = (1-\\alpha)\\,\\Pi[:, y_2] + \\alpha\\,\\Pi[:, y_1]

    with ``y_2 = target`` and ``y_1 = source``.  ``alpha -> 1`` makes the two
    class prior signatures indistinguishable.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    blended = prior.clone()
    blended[:, target] = (1.0 - alpha) * prior[:, target] + alpha * prior[:, source]
    return blended


# --------------------------------------------------------------------------- #
# I/O
# --------------------------------------------------------------------------- #
def load_prior_table(
    path: str | Path,
    *,
    concept_names: Sequence[str] | None = None,
    class_names: Sequence[str] | None = None,
) -> tuple[torch.Tensor, list[str], list[str]]:
    """Load a prior table stored as a wide table (rows = concepts, cols = classes).

    The first column is interpreted as the concept name.  Rows/columns are
    reordered to match ``concept_names`` / ``class_names`` when provided, and a
    missing entry raises rather than silently defaulting.
    """
    frame = read_table(path)
    frame = frame.set_index(frame.columns[0])
    if concept_names is not None:
        missing = [c for c in concept_names if c not in frame.index]
        if missing:
            raise KeyError(f"Prior table {path} is missing concepts: {missing}")
        frame = frame.loc[list(concept_names)]
    if class_names is not None:
        missing = [c for c in class_names if c not in frame.columns]
        if missing:
            raise KeyError(f"Prior table {path} is missing classes: {missing}")
        frame = frame[list(class_names)]
    table = torch.tensor(frame.to_numpy(dtype=float), dtype=torch.float32)
    return table, [str(i) for i in frame.index], [str(c) for c in frame.columns]


def save_prior_table(
    prior: torch.Tensor,
    path: str | Path,
    *,
    concept_names: Sequence[str] | None = None,
    class_names: Sequence[str] | None = None,
) -> None:
    import pandas as pd

    concept_names = list(concept_names or [f"concept_{m}" for m in range(prior.shape[0])])
    class_names = list(class_names or [f"class_{k}" for k in range(prior.shape[1])])
    frame = pd.DataFrame(prior.detach().cpu().numpy(), index=concept_names, columns=class_names)
    frame.index.name = "concept"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        frame.to_excel(path)
    else:
        frame.to_csv(path)
