"""WBCAtt: white-blood-cell morphological attributes (plan 6.1, dataset 2).

WBCAtt augments the PBC blood-cell dataset with 11 morphological attributes per
cell, which makes it the largest and most stable concept benchmark used here.

Expected layout::

    <root>/
        images/...                      # PBC images referenced by the CSVs
        pbc_attr_v1_train.csv
        pbc_attr_v1_val.csv
        pbc_attr_v1_test.csv

The official CSVs carry a ``path`` column and categorical attribute columns; the
loader binarises them with :datasets:`ATTRIBUTE_POSITIVE_VALUES` and merges the
official splits into a single dataset with a ``split`` column preserved.
"""

from __future__ import annotations

from pathlib import Path

from rpcp.data.manifest import ManifestConceptDataset
from rpcp.utils.logging import get_logger

__all__ = ["WBCATT_CLASSES", "WBCATT_CONCEPTS", "build_wbcatt", "prepare_manifest"]

logger = get_logger(__name__)

WBCATT_CLASSES: tuple[str, ...] = (
    "basophil",
    "eosinophil",
    "lymphocyte",
    "monocyte",
    "neutrophil",
)

WBCATT_CONCEPTS: tuple[str, ...] = (
    "cell_size_big",
    "cell_shape_irregular",
    "nucleus_shape_segmented_multilobed",
    "nuclear_cytoplasmic_ratio_high",
    "chromatin_density_densely_clumped",
    "cytoplasm_vacuole_yes",
    "cytoplasm_texture_clear",
    "cytoplasm_colour_light_blue",
    "granule_type_coarse",
    "granule_colour_pink",
    "granularity_yes",
)

#: Raw column -> value that counts as the positive class for our binary concept.
ATTRIBUTE_POSITIVE_VALUES: dict[str, tuple[str, str]] = {
    "cell_size_big": ("cell_size", "big"),
    "cell_shape_irregular": ("cell_shape", "irregular"),
    "nucleus_shape_segmented_multilobed": ("nucleus_shape", "segmented-multilobed"),
    "nuclear_cytoplasmic_ratio_high": ("nuclear_cytoplasmic_ratio", "high"),
    "chromatin_density_densely_clumped": ("chromatin_density", "densely clumped"),
    "cytoplasm_vacuole_yes": ("cytoplasm_vacuole", "yes"),
    "cytoplasm_texture_clear": ("cytoplasm_texture", "clear"),
    "cytoplasm_colour_light_blue": ("cytoplasm_colour", "light blue"),
    "granule_type_coarse": ("granule_type", "coarse"),
    "granule_colour_pink": ("granule_colour", "pink"),
    "granularity_yes": ("granularity", "yes"),
}


def prepare_manifest(
    root: str | Path,
    *,
    csv_names: tuple[str, ...] = (
        "pbc_attr_v1_train.csv",
        "pbc_attr_v1_val.csv",
        "pbc_attr_v1_test.csv",
    ),
    path_column: str = "path",
    label_column: str = "label",
    output: str | Path | None = None,
) -> Path:
    """Merge the official WBCAtt CSVs into the project manifest schema."""
    import pandas as pd

    root = Path(root)
    frames = []
    for name in csv_names:
        path = root / name
        if not path.exists():
            logger.warning("WBCAtt split file missing, skipping: %s", path)
            continue
        frame = pd.read_csv(path)
        frame["split"] = name.replace("pbc_attr_v1_", "").replace(".csv", "")
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No WBCAtt CSVs found under {root}")

    merged = pd.concat(frames, ignore_index=True)
    merged.columns = [str(c).strip().lower().replace(" ", "_") for c in merged.columns]

    records = pd.DataFrame(
        {
            "image": merged[path_column].astype(str),
            "label": merged[label_column].astype(str).str.lower(),
            "split": merged["split"],
        }
    )
    for concept, (column, positive) in ATTRIBUTE_POSITIVE_VALUES.items():
        if column not in merged:
            logger.warning("WBCAtt attribute column '%s' missing; concept set to 0", column)
            records[concept] = 0
            continue
        records[concept] = (
            merged[column].astype(str).str.strip().str.lower() == positive
        ).astype(int)

    output = Path(output or root / "manifest.csv")
    records.to_csv(output, index=False)
    logger.info("Wrote WBCAtt manifest with %d rows to %s", len(records), output)
    return output


def build_wbcatt(
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
        concept_columns=WBCATT_CONCEPTS,
        class_names=WBCATT_CLASSES,
        transform=transform,
    )
    logger.info("WBCAtt: %s", dataset.describe())
    return dataset
