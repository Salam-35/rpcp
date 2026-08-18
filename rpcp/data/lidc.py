"""LIDC-IDRI lung nodules with multi-radiologist attributes (plan 6.1, dataset 4).

LIDC is the only dataset here with *natural* annotation disagreement: up to four
radiologists rate each nodule on eight ordinal attributes.  That disagreement is
the reliability signal of Evidence Mode D (plan 4.3): we can ask whether learned
``r[m, y]`` correlates with how much the radiologists actually agreed.

Expected layout (produced by :func:`prepare_manifest` from ``pylidc``)::

    <root>/
        patches/LIDC-IDRI-0001_n0.png ...
        manifest.csv     # image,label,<concept>,<concept>__rater0,...
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from rpcp.data.manifest import ManifestConceptDataset
from rpcp.utils.logging import get_logger

__all__ = [
    "LIDC_CLASSES",
    "LIDC_CONCEPTS",
    "build_lidc",
    "prepare_manifest",
    "rater_agreement_matrix",
]

logger = get_logger(__name__)

LIDC_CLASSES: tuple[str, ...] = ("benign", "malignant")

#: Ordinal attributes (1-5), binarised at ``>= 4`` -- "clearly present".
LIDC_CONCEPTS: tuple[str, ...] = (
    "subtlety",
    "sphericity",
    "margin",
    "lobulation",
    "spiculation",
    "texture",
    "calcification",
    "internal_structure",
)

ORDINAL_POSITIVE_THRESHOLD = 4.0
MALIGNANCY_THRESHOLD = 3.0

#: pylidc.Annotation attribute names for each concept. Most LIDC_CONCEPTS
#: entries are already single lowercase words so `concept.replace("_", "")`
#: happens to match, but "internal_structure" maps to the camelCase
#: `internalStructure` on the pylidc Annotation object -- the naive
#: `.replace("_", "")` gives "internalstructure", which does not exist, so
#: `getattr(a, attribute, np.nan)` silently returned NaN for every rater on
#: every nodule, making internal_structure a dead (always-0) concept by
#: construction. Spelled out explicitly here so a future renamed/added
#: concept fails loudly (KeyError) instead of silently going dead again.
_PYLIDC_ATTRIBUTE_NAMES: dict[str, str] = {
    "subtlety": "subtlety",
    "sphericity": "sphericity",
    "margin": "margin",
    "lobulation": "lobulation",
    "spiculation": "spiculation",
    "texture": "texture",
    "calcification": "calcification",
    "internal_structure": "internalStructure",
}


def prepare_manifest(
    root: str | Path,
    *,
    patch_dir: str = "patches",
    output: str | Path | None = None,
    max_nodules: int | None = None,
) -> Path:
    """Extract nodule patches + per-rater attributes using ``pylidc``.

    ``pylidc`` must be configured (``~/.pylidcrc``) to point at the DICOM tree.
    Nodules whose median malignancy equals 3 ("indeterminate") are dropped, which
    is the standard LIDC binary protocol.
    """
    import pandas as pd
    import pylidc as pl
    from PIL import Image

    root = Path(root)
    (root / patch_dir).mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []

    scans = pl.query(pl.Scan)
    for scan_idx, scan in enumerate(scans):
        for nodule_idx, nodule_annotations in enumerate(scan.cluster_annotations()):
            if max_nodules is not None and len(records) >= max_nodules:
                break
            malignancy = np.median([a.malignancy for a in nodule_annotations])
            if malignancy == MALIGNANCY_THRESHOLD:
                continue

            reference = nodule_annotations[0]
            volume = reference.scan.to_volume()
            bbox = reference.bbox()
            centre = bbox[2].start + (bbox[2].stop - bbox[2].start) // 2
            patch = volume[bbox[0], bbox[1], centre]
            # `ndarray.ptp()` was removed in NumPy 2.0; use the `np.ptp` function.
            patch = (patch - patch.min()) / max(float(np.ptp(patch)), 1e-6)
            name = f"{scan.patient_id}_n{nodule_idx}.png"
            Image.fromarray((patch * 255).astype(np.uint8)).save(root / patch_dir / name)

            record: dict[str, object] = {
                "image": f"{patch_dir}/{name}",
                "label": "malignant" if malignancy > MALIGNANCY_THRESHOLD else "benign",
            }
            for concept in LIDC_CONCEPTS:
                attribute = _PYLIDC_ATTRIBUTE_NAMES[concept]
                scores = [
                    float(getattr(a, attribute, np.nan)) for a in nodule_annotations
                ]
                record[concept] = int(np.nanmedian(scores) >= ORDINAL_POSITIVE_THRESHOLD)
                for r_idx, score in enumerate(scores[:4]):
                    record[f"{concept}__rater{r_idx}"] = int(score >= ORDINAL_POSITIVE_THRESHOLD)
            records.append(record)
        logger.info("Processed scan %d (%d nodules so far)", scan_idx, len(records))

    manifest = pd.DataFrame.from_records(records)
    output = Path(output or root / "manifest.csv")
    manifest.to_csv(output, index=False)
    logger.info("Wrote LIDC manifest with %d nodules to %s", len(manifest), output)
    return output


def build_lidc(
    root: str | Path,
    *,
    manifest: str | Path | None = None,
    transform: object | None = None,
) -> ManifestConceptDataset:
    root = Path(root)
    manifest = Path(manifest or root / "manifest.csv")
    if not manifest.exists():
        raise FileNotFoundError(
            f"LIDC manifest {manifest} not found. Build it once with "
            f"`from rpcp.data.lidc import prepare_manifest; prepare_manifest('{root}')`."
        )
    dataset = ManifestConceptDataset(
        root=root,
        manifest=manifest,
        concept_columns=LIDC_CONCEPTS,
        class_names=LIDC_CLASSES,
        grayscale=True,
        transform=transform,
    )
    logger.info("LIDC: %s", dataset.describe())
    return dataset


def rater_agreement_matrix(dataset: ManifestConceptDataset) -> torch.Tensor | None:
    """``(M, K)`` inter-rater agreement, or ``None`` if the manifest has no raters."""
    from rpcp.data.priors import multi_rater_agreement

    if dataset.rater_votes is None:
        return None
    return multi_rater_agreement(
        dataset.rater_votes, dataset.labels, n_classes=dataset.n_classes
    )
