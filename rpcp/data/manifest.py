"""Manifest-driven image dataset shared by all real medical datasets.

Every real dataset in this project is described by a *manifest* table (CSV/TSV/
XLSX) with one row per image::

    image,label,<concept_1>,<concept_2>,...,<concept_M>
    IMD003.bmp,nevus,0,1,...,0

``image`` is a path relative to ``root``; ``label`` is a class name (or index);
the remaining named columns are binary concept annotations used for *evaluation
only*.  Optional ``split`` column overrides the automatic stratified split, and
optional ``<concept>__rater<k>`` columns carry per-rater votes (Evidence Mode D).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

from rpcp.data.base import ConceptDataset, read_table

__all__ = ["ManifestConceptDataset"]


class ManifestConceptDataset(ConceptDataset):
    """Image dataset backed by a manifest table.

    Args:
        root: Directory containing the images.
        manifest: Path to the manifest table.
        concept_columns: Concept columns to use; defaults to every column that
            is not ``image``/``label``/``split`` and is not a rater column.
        class_names: Explicit class ordering; inferred (sorted) when omitted.
        image_column / label_column / split_column: Column names.
        binarize_threshold: Concepts above this value are treated as present
            (allows ordinal annotations such as LIDC 1-5 scales after scaling).
        grayscale: Load images as single-channel.
        transform: Image transform applied after loading.
    """

    def __init__(
        self,
        root: str | Path,
        manifest: str | Path,
        *,
        concept_columns: Sequence[str] | None = None,
        class_names: Sequence[str] | None = None,
        image_column: str = "image",
        label_column: str = "label",
        split_column: str = "split",
        rater_suffix: str = "__rater",
        binarize_threshold: float = 0.5,
        grayscale: bool = False,
        transform: object | None = None,
    ) -> None:
        self.root = Path(root)
        frame = read_table(manifest)
        if image_column not in frame or label_column not in frame:
            raise KeyError(
                f"Manifest {manifest} must contain '{image_column}' and '{label_column}' columns; "
                f"found {list(frame.columns)}"
            )

        reserved = {image_column, label_column, split_column}
        rater_columns = [c for c in frame.columns if rater_suffix in str(c)]
        if concept_columns is None:
            concept_columns = [
                c for c in frame.columns if c not in reserved and c not in rater_columns
            ]
        missing = [c for c in concept_columns if c not in frame.columns]
        if missing:
            raise KeyError(f"Manifest {manifest} is missing concept columns: {missing}")

        raw_labels = frame[label_column]
        if class_names is None:
            class_names = (
                [str(v) for v in sorted(raw_labels.unique())]
                if raw_labels.dtype == object
                else [str(v) for v in sorted(raw_labels.unique())]
            )
        lookup = {name: idx for idx, name in enumerate(class_names)}
        labels = np.array([lookup[str(v)] for v in raw_labels], dtype=np.int64)

        concepts = frame[list(concept_columns)].to_numpy(dtype=np.float32)
        concepts = (concepts >= binarize_threshold).astype(np.float32)

        super().__init__(
            labels=labels,
            concepts=concepts,
            concept_names=[str(c) for c in concept_columns],
            class_names=[str(c) for c in class_names],
            transform=transform,  # type: ignore[arg-type]
        )

        self.image_paths = [self.root / str(p) for p in frame[image_column]]
        self.grayscale = grayscale
        self.split_labels = (
            frame[split_column].astype(str).to_numpy() if split_column in frame else None
        )
        self.rater_votes = self._collect_rater_votes(frame, concept_columns, rater_suffix)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _collect_rater_votes(
        frame: object,
        concept_columns: Sequence[str],
        rater_suffix: str,
    ) -> np.ndarray | None:
        """Stack ``<concept>__rater<k>`` columns into an ``(N, R, M)`` array."""
        import pandas as pd

        assert isinstance(frame, pd.DataFrame)
        raters: dict[str, dict[str, str]] = {}
        for column in frame.columns:
            name = str(column)
            if rater_suffix not in name:
                continue
            concept, rater = name.split(rater_suffix)
            raters.setdefault(rater, {})[concept] = name
        if not raters:
            return None

        votes = np.full((len(frame), len(raters), len(concept_columns)), np.nan, dtype=np.float32)
        for r_idx, rater in enumerate(sorted(raters)):
            for m_idx, concept in enumerate(concept_columns):
                column = raters[rater].get(str(concept))
                if column is not None:
                    votes[:, r_idx, m_idx] = frame[column].to_numpy(dtype=np.float32)
        return votes

    # ------------------------------------------------------------------ #
    def load_image(self, index: int) -> torch.Tensor:
        from PIL import Image

        path = self.image_paths[index]
        if not path.exists():
            path = _resolve_case_insensitive(path) or path
        if not path.exists():
            raise FileNotFoundError(f"Image referenced by the manifest is missing: {path}")
        with Image.open(path) as handle:
            image = handle.convert("L" if self.grayscale else "RGB")
            array = np.asarray(image, dtype=np.float32) / 255.0
        array = array[None, ...] if array.ndim == 2 else array.transpose(2, 0, 1)
        return torch.from_numpy(np.ascontiguousarray(array))

    def predefined_splits(self) -> dict[str, np.ndarray] | None:
        """Index arrays from the manifest ``split`` column, when present."""
        if self.split_labels is None:
            return None
        return {
            name: np.flatnonzero(self.split_labels == name)
            for name in np.unique(self.split_labels)
        }


def _resolve_case_insensitive(path: Path) -> Path | None:
    """Resolve a path whose manifest casing differs from the filesystem casing.

    Derm7pt metadata contains paths such as ``FCl/Fcl068.jpg`` while some
    extracted archives contain ``FCL/...``.  That works accidentally on
    case-insensitive macOS volumes but fails on Linux/Kaggle.
    """
    current = Path(path.anchor) if path.is_absolute() else Path(".")
    parts = path.parts[1:] if path.is_absolute() else path.parts
    for part in parts:
        candidate = current / part
        if candidate.exists():
            current = candidate
            continue
        if not current.exists() or not current.is_dir():
            return None
        part_lower = part.lower()
        matches = [child for child in current.iterdir() if child.name.lower() == part_lower]
        if not matches:
            return None
        current = matches[0]
    return current
