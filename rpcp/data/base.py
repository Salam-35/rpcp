"""Core dataset abstractions.

Every dataset in this project yields the same batch dictionary::

    {
        "image":    FloatTensor (B, C, H, W),
        "label":    LongTensor  (B,)            # y_i
        "concepts": FloatTensor (B, M)          # c_i, evaluation only
        "has_concepts": BoolTensor (B,)         # False when c_i is unavailable
        "index":    LongTensor  (B,)            # position in the parent dataset
    }

The concept vector is *never* consumed by the main training objective; it is
carried through the pipeline so that evaluation (and the optional audit split of
Evidence Mode C) can use it.  ``RPCPTrainer`` enforces this separation.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from rpcp.utils.seeding import worker_init_fn

__all__ = [
    "ConceptBatch",
    "ConceptDataset",
    "SplitBundle",
    "TransformedSubset",
    "collate_concepts",
    "make_dataloader",
    "read_table",
    "stratified_split",
    "subset_view",
]

ConceptBatch = dict[str, torch.Tensor]
Transform = Callable[[torch.Tensor], torch.Tensor]


class ConceptDataset(Dataset[ConceptBatch]):
    """Base class for image datasets with (evaluation-only) concept labels.

    Args:
        labels: Integer class label per sample, shape ``(N,)``.
        concepts: Binary concept matrix, shape ``(N, M)``; ``None`` when the
            dataset has no concept annotations at all.
        concept_names: Length-``M`` list of concept names.
        class_names: Length-``K`` list of class names.
        transform: Optional image transform applied to the ``(C, H, W)`` tensor.
    """

    def __init__(
        self,
        *,
        labels: Sequence[int] | np.ndarray,
        concepts: np.ndarray | None,
        concept_names: Sequence[str],
        class_names: Sequence[str],
        transform: Transform | None = None,
    ) -> None:
        self.labels = np.asarray(labels, dtype=np.int64)
        self.concepts = None if concepts is None else np.asarray(concepts, dtype=np.float32)
        self.concept_names = list(concept_names)
        self.class_names = list(class_names)
        self.transform = transform

        if self.concepts is not None and self.concepts.shape != (
            len(self.labels),
            len(self.concept_names),
        ):
            raise ValueError(
                f"concepts has shape {self.concepts.shape}, expected "
                f"{(len(self.labels), len(self.concept_names))}"
            )

    # -- required interface ------------------------------------------------ #
    def load_image(self, index: int) -> torch.Tensor:
        """Return the raw ``(C, H, W)`` float image for ``index``."""
        raise NotImplementedError

    # -- dunder ------------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> ConceptBatch:
        image = self.load_image(index)
        if self.transform is not None:
            image = self.transform(image)
        return self.build_sample(index, image)

    def build_sample(self, index: int, image: torch.Tensor) -> ConceptBatch:
        """Assemble the batch dictionary for an already-loaded image."""
        if self.concepts is None:
            concepts = torch.zeros(self.n_concepts, dtype=torch.float32)
            has_concepts = torch.zeros((), dtype=torch.bool)
        else:
            concepts = torch.from_numpy(self.concepts[index])
            has_concepts = torch.ones((), dtype=torch.bool)
        return {
            "image": image,
            "label": torch.tensor(int(self.labels[index]), dtype=torch.long),
            "concepts": concepts,
            "has_concepts": has_concepts,
            "index": torch.tensor(index, dtype=torch.long),
        }

    # -- convenience ------------------------------------------------------- #
    @property
    def n_classes(self) -> int:
        return len(self.class_names)

    @property
    def n_concepts(self) -> int:
        return len(self.concept_names)

    def class_counts(self) -> np.ndarray:
        return np.bincount(self.labels, minlength=self.n_classes)

    def describe(self) -> str:
        counts = ", ".join(
            f"{name}={count}"
            for name, count in zip(self.class_names, self.class_counts(), strict=False)
        )
        return (
            f"{type(self).__name__}(n={len(self)}, K={self.n_classes}, M={self.n_concepts}, "
            f"concepts={'yes' if self.concepts is not None else 'no'}, {counts})"
        )


@dataclass(slots=True)
class SplitBundle:
    """Train / val / test (+ optional reliability-audit) splits of one dataset.

    ``train_eval`` is the *same images and indices* as ``train``, but with the
    deterministic evaluation transform instead of the training augmentation.
    Anything that reads model outputs off the training set for measurement
    rather than optimisation -- held-out class means when
    ``reliability.use_crossfit=False``, fold/instability estimation, ad hoc
    diagnostics -- should use it instead of ``train``. Reusing ``train``
    directly for measurement makes the "held-out" residual partly a measurement
    of random crops and flips instead of the model, and makes cross-fitted
    instability estimates non-reproducible between runs.
    """

    train: Dataset[ConceptBatch]
    train_eval: Dataset[ConceptBatch]
    val: Dataset[ConceptBatch]
    test: Dataset[ConceptBatch]
    audit: Dataset[ConceptBatch] | None
    concept_names: list[str]
    class_names: list[str]
    source: ConceptDataset

    @property
    def n_classes(self) -> int:
        return len(self.class_names)

    @property
    def n_concepts(self) -> int:
        return len(self.concept_names)

    def sizes(self) -> dict[str, int]:
        return {
            "train": len(self.train),  # type: ignore[arg-type]
            "val": len(self.val),  # type: ignore[arg-type]
            "test": len(self.test),  # type: ignore[arg-type]
            "audit": 0 if self.audit is None else len(self.audit),  # type: ignore[arg-type]
        }


class TransformedSubset(Dataset[ConceptBatch]):
    """A split of a :class:`ConceptDataset` with its own image transform.

    Splits share one underlying dataset object (so images are loaded/generated
    once) but need different pipelines: augmentation on train, deterministic
    resizing everywhere else.
    """

    def __init__(
        self,
        dataset: ConceptDataset,
        indices: Sequence[int],
        transform: Transform | None = None,
    ) -> None:
        self.dataset = dataset
        self.indices = np.asarray(indices, dtype=np.int64)
        self.transform = transform

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, position: int) -> ConceptBatch:
        index = int(self.indices[position])
        image = self.dataset.load_image(index)
        if self.transform is not None:
            image = self.transform(image)
        return self.dataset.build_sample(index, image)

    # -- convenience ------------------------------------------------------- #
    @property
    def labels(self) -> np.ndarray:
        return self.dataset.labels[self.indices]

    @property
    def concepts(self) -> np.ndarray | None:
        return None if self.dataset.concepts is None else self.dataset.concepts[self.indices]

    @property
    def concept_names(self) -> list[str]:
        return self.dataset.concept_names

    @property
    def class_names(self) -> list[str]:
        return self.dataset.class_names

    @property
    def n_classes(self) -> int:
        return self.dataset.n_classes

    @property
    def n_concepts(self) -> int:
        return self.dataset.n_concepts

    def subset(self, positions: Sequence[int]) -> TransformedSubset:
        """A further split of this split (used by cross-fitting)."""
        return TransformedSubset(self.dataset, self.indices[np.asarray(positions)], self.transform)


def subset_view(dataset: ConceptDataset, indices: Sequence[int]) -> Subset[ConceptBatch]:
    """Plain :class:`torch.utils.datasets.Subset` that keeps dataset metadata."""
    subset = Subset(dataset, list(indices))
    subset.concept_names = dataset.concept_names  # type: ignore[attr-defined]
    subset.class_names = dataset.class_names  # type: ignore[attr-defined]
    return subset


def stratified_split(
    labels: np.ndarray,
    fractions: dict[str, float],
    *,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Class-stratified split of indices.

    Args:
        labels: ``(N,)`` integer class labels.
        fractions: Mapping ``split -> fraction``.  Fractions must sum to <= 1;
            the remainder goes to the split named ``"train"`` (created if absent).
        seed: RNG seed.

    Returns:
        Mapping ``split -> index array``.
    """
    total = sum(fractions.values())
    if total > 1.0 + 1e-8:
        raise ValueError(f"Split fractions sum to {total} > 1")
    rng = np.random.default_rng(seed)
    names = [n for n in fractions if n != "train"]
    out: dict[str, list[int]] = {name: [] for name in [*names, "train"]}

    for cls in np.unique(labels):
        idx = np.flatnonzero(labels == cls)
        rng.shuffle(idx)
        start = 0
        for name in names:
            take = int(round(fractions[name] * len(idx)))
            # Guarantee at least one sample per non-empty requested split.
            if fractions[name] > 0 and take == 0 and len(idx) - start > 1:
                take = 1
            out[name].extend(idx[start : start + take].tolist())
            start += take
        out["train"].extend(idx[start:].tolist())

    return {name: np.asarray(sorted(v), dtype=np.int64) for name, v in out.items()}


def collate_concepts(samples: list[ConceptBatch]) -> ConceptBatch:
    """Default collate; explicit so batch keys stay stable."""
    return {key: torch.stack([s[key] for s in samples]) for key in samples[0]}


def make_dataloader(
    dataset: Dataset[ConceptBatch],
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False,
    seed: int = 0,
) -> DataLoader[ConceptBatch]:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=collate_concepts,
        worker_init_fn=worker_init_fn if num_workers > 0 else None,
        generator=generator,
        persistent_workers=num_workers > 0,
    )


def read_table(path: str | Path, **kwargs: Any) -> Any:
    """Read a CSV/TSV/XLSX manifest with pandas (imported lazily)."""
    import pandas as pd

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    match path.suffix.lower():
        case ".csv":
            return pd.read_csv(path, **kwargs)
        case ".tsv" | ".txt":
            return pd.read_csv(path, sep="\t", **kwargs)
        case ".xlsx" | ".xls":
            return pd.read_excel(path, **kwargs)
        case suffix:
            raise ValueError(f"Unsupported manifest format: {suffix}")
