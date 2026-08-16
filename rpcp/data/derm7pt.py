"""Derm7pt: the seven-point checklist dermoscopy dataset (plan 6.1, dataset 3).

Used as the *external validation* dataset: priors fitted or audited on PH2 can
be transferred here to test whether reliability estimates survive a population
shift.

Expected layout::

    <root>/
        images/...
        meta/meta.csv
        meta/train_indexes.csv, meta/valid_indexes.csv, meta/test_indexes.csv
"""

from __future__ import annotations

from pathlib import Path

from rpcp.data.manifest import ManifestConceptDataset
from rpcp.utils.logging import get_logger

__all__ = ["DERM7PT_CLASSES", "DERM7PT_CONCEPTS", "build_derm7pt", "prepare_manifest"]

logger = get_logger(__name__)

DERM7PT_CLASSES: tuple[str, ...] = ("nevus", "melanoma")

#: Seven-point checklist criteria, binarised as "abnormal vs absent/typical".
DERM7PT_CONCEPTS: tuple[str, ...] = (
    "atypical_pigment_network",
    "blue_whitish_veil",
    "atypical_vascular_structures",
    "irregular_streaks",
    "irregular_pigmentation",
    "irregular_dots_and_globules",
    "regression_structures",
)

#: Raw meta.csv column -> values that count as *present/abnormal*.
_CONCEPT_RULES: dict[str, tuple[str, tuple[str, ...]]] = {
    "atypical_pigment_network": ("pigment_network", ("atypical",)),
    "blue_whitish_veil": ("blue_whitish_veil", ("present",)),
    "atypical_vascular_structures": (
        "vascular_structures",
        (
            "arborizing",
            "comma",
            "hairpin",
            "within regression",
            "wreath",
            "dotted",
            "linear irregular",
        ),
    ),
    "irregular_streaks": ("streaks", ("irregular",)),
    "irregular_pigmentation": (
        "pigmentation",
        ("localized irregular", "diffuse irregular"),
    ),
    "irregular_dots_and_globules": ("dots_and_globules", ("irregular",)),
    "regression_structures": (
        "regression_structures",
        ("blue areas", "white areas", "combinations"),
    ),
}

_MELANOMA_KEYS = ("melanoma", "basal cell carcinoma")


def prepare_manifest(
    root: str | Path,
    *,
    meta_csv: str | Path | None = None,
    image_column: str = "derm",
    output: str | Path | None = None,
    use_official_splits: bool = True,
) -> Path:
    """Convert ``meta/meta.csv`` into the project manifest schema."""
    import pandas as pd

    root = Path(root)
    meta_csv = Path(meta_csv or root / "meta" / "meta.csv")
    meta = pd.read_csv(meta_csv)
    meta.columns = [str(c).strip().lower() for c in meta.columns]

    records = pd.DataFrame(
        {
            "image": "images/" + meta[image_column].astype(str),
            "label": [
                "melanoma" if any(k in str(d).lower() for k in _MELANOMA_KEYS) else "nevus"
                for d in meta["diagnosis"]
            ],
        }
    )
    for concept, (column, positives) in _CONCEPT_RULES.items():
        if column not in meta:
            logger.warning("Derm7pt column '%s' missing; concept '%s' set to 0", column, concept)
            records[concept] = 0
            continue
        values = meta[column].astype(str).str.strip().str.lower()
        records[concept] = values.isin([p.lower() for p in positives]).astype(int)

    if use_official_splits:
        split = pd.Series(["train"] * len(meta), index=meta.index)
        for name, filename in (
            ("val", "valid_indexes.csv"),
            ("test", "test_indexes.csv"),
        ):
            path = root / "meta" / filename
            if path.exists():
                indexes = pd.read_csv(path).iloc[:, 0].to_numpy()
                split.iloc[indexes] = name
        records["split"] = split.to_numpy()

    output = Path(output or root / "manifest.csv")
    records.to_csv(output, index=False)
    logger.info("Wrote Derm7pt manifest with %d rows to %s", len(records), output)
    return output


def build_derm7pt(
    root: str | Path,
    *,
    manifest: str | Path | None = None,
    transform: object | None = None,
) -> ManifestConceptDataset:
    root = Path(root)
    manifest = Path(manifest or root / "manifest.csv")
    if not manifest.exists():
        prepare_manifest(root, output=manifest)
    dataset = ManifestConceptDataset(
        root=root,
        manifest=manifest,
        concept_columns=DERM7PT_CONCEPTS,
        class_names=DERM7PT_CLASSES,
        transform=transform,
    )
    logger.info("Derm7pt: %s", dataset.describe())
    return dataset
