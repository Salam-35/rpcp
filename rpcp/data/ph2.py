"""PH2 dermoscopy dataset (plan 6.1, dataset 1).

PH2 ships 200 dermoscopic images plus a spreadsheet of dermoscopic criteria.
It is tiny, which makes it the right place to iterate on corruption sweeps.

Expected layout::

    <root>/
        images/IMD003.bmp ...
        PH2_dataset.xlsx          # official annotation spreadsheet
        manifest.csv              # produced by `prepare_manifest`

Because the official spreadsheet has a merged multi-row header that changes
between distributions, :func:`prepare_manifest` is deliberately tolerant and
prints the mapping it inferred -- verify it once per download.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from rpcp.data.manifest import ManifestConceptDataset
from rpcp.utils.logging import get_logger

__all__ = ["PH2_CLASSES", "PH2_CONCEPTS", "build_ph2", "prepare_manifest"]

logger = get_logger(__name__)

PH2_CLASSES: tuple[str, ...] = ("common_nevus", "atypical_nevus", "melanoma")

#: Binary dermoscopic criteria used as concepts.
PH2_CONCEPTS: tuple[str, ...] = (
    "asymmetry",
    "atypical_pigment_network",
    "atypical_dots_globules",
    "streaks",
    "regression_areas",
    "blue_whitish_veil",
    "colour_white",
    "colour_red",
    "colour_light_brown",
    "colour_dark_brown",
    "colour_blue_gray",
    "colour_black",
)

_CLINICAL_DIAGNOSIS_TO_CLASS = {"0": "common_nevus", "1": "atypical_nevus", "2": "melanoma"}


def _normalise(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def prepare_manifest(
    root: str | Path,
    *,
    spreadsheet: str | Path | None = None,
    image_dir: str = "images",
    image_suffix: str = ".bmp",
    output: str | Path | None = None,
    binary_classes: bool = False,
) -> Path:
    """Convert the official PH2 spreadsheet into the project manifest schema.

    Args:
        root: PH2 dataset root.
        spreadsheet: Path to ``PH2_dataset.xlsx`` (defaults to ``root``).
        image_dir: Sub-directory of ``root`` holding the images.
        image_suffix: Image file extension.
        output: Manifest destination (defaults to ``root/manifest.csv``).
        binary_classes: Collapse the two nevus classes into ``nevus``.

    Returns:
        Path to the written manifest.
    """
    import pandas as pd

    root = Path(root)
    spreadsheet = Path(spreadsheet or root / "PH2_dataset.xlsx")
    frame = pd.read_excel(spreadsheet, header=None)

    # Locate the header row: the first row mentioning the image-name column.
    header_row = next(
        i
        for i in range(len(frame))
        if any("image" in _normalise(v) and "name" in _normalise(v) for v in frame.iloc[i].values)
    )
    frame = pd.read_excel(spreadsheet, header=header_row)
    frame.columns = [_normalise(c) for c in frame.columns]
    frame = frame.dropna(how="all")

    name_col = next(c for c in frame.columns if "image" in c)
    diag_col = next(c for c in frame.columns if "clinical" in c or "diagnosis" in c)

    records: list[dict[str, object]] = []
    for _, row in frame.iterrows():
        image_name = str(row[name_col]).strip()
        if not image_name or image_name.lower() == "nan":
            continue
        diagnosis = str(int(float(row[diag_col])))
        label = _CLINICAL_DIAGNOSIS_TO_CLASS.get(diagnosis, "unknown")
        if binary_classes:
            label = "melanoma" if label == "melanoma" else "nevus"
        record: dict[str, object] = {
            "image": f"{image_dir}/{image_name}{image_suffix}",
            "label": label,
        }
        for concept in PH2_CONCEPTS:
            record[concept] = _extract_concept(row, concept)
        records.append(record)

    manifest = pd.DataFrame.from_records(records)
    output = Path(output or root / "manifest.csv")
    manifest.to_csv(output, index=False)
    logger.info("Wrote PH2 manifest with %d rows to %s", len(manifest), output)
    return output


def _extract_concept(row: object, concept: str) -> int:
    """Best-effort binarisation of one PH2 criterion from a spreadsheet row."""
    import pandas as pd

    assert isinstance(row, pd.Series)
    key = concept.replace("colour_", "")
    candidates = [c for c in row.index if key.split("_")[-1] in c or key in c]
    for column in candidates:
        value = row[column]
        if pd.isna(value):
            continue
        text = str(value).strip().lower()
        if text in {"x", "yes", "present", "a", "at"}:
            return 1
        if text in {"", "no", "absent", "t", "typical"}:
            return 0
        try:
            return int(float(text) > 0)
        except ValueError:
            continue
    return 0


def build_ph2(
    root: str | Path,
    *,
    manifest: str | Path | None = None,
    transform: object | None = None,
    binary_classes: bool = False,
) -> ManifestConceptDataset:
    """Instantiate the PH2 dataset from a prepared manifest."""
    root = Path(root)
    manifest = Path(manifest or root / "manifest.csv")
    if not manifest.exists():
        raise FileNotFoundError(
            f"PH2 manifest {manifest} not found. Run "
            "`python -c \"from rpcp.datasets.ph2 import prepare_manifest; "
            f"prepare_manifest('{root}')\"`"
        )
    classes = ("nevus", "melanoma") if binary_classes else PH2_CLASSES
    dataset = ManifestConceptDataset(
        root=root,
        manifest=manifest,
        concept_columns=PH2_CONCEPTS,
        class_names=classes,
        transform=transform,
    )
    logger.info("PH2: %s", dataset.describe())
    return dataset


def concept_support(dataset: ManifestConceptDataset) -> np.ndarray:
    """Number of positive annotations per concept (useful for sanity checks)."""
    assert dataset.concepts is not None
    return dataset.concepts.sum(axis=0)
