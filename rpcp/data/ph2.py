"""PH2 dermoscopy dataset (plan 6.1, dataset 1).

PH2 ships 200 dermoscopic images plus the official annotation table in two
equivalent forms (``PH2_dataset.xlsx`` and ``PH2_dataset.txt``).  The images are
distributed one directory per case::

    <root>/
        PH2 Dataset images/
            IMD003/
                IMD003_Dermoscopic_Image/IMD003.bmp
                IMD003_lesion/IMD003_lesion.bmp          # lesion segmentation
                IMD003_roi/IMD003_R1_Label4.bmp          # per-colour ROI masks
        PH2_dataset.xlsx
        PH2_dataset.txt
        manifest.csv                                     # produced here

Annotation semantics (from the ``Legends`` block of the official spreadsheet)::

    Asymmetry            0 = fully symmetric, 1 = symmetric in 1 axis,
                         2 = fully asymmetric
    Pigment Network      T  = typical,  AT = atypical
    Dots/Globules        A  = absent,   T  = typical,  AT = atypical
    Streaks              A  = absent,   P  = present
    Regression Areas     A  = absent,   P  = present
    Blue-Whitish Veil    A  = absent,   P  = present
    Colors               1 = white, 2 = red, 3 = light brown, 4 = dark brown,
                         5 = blue-gray, 6 = black

Every value is mapped through an explicit table below.  Substring matching on
column names is deliberately avoided: ``A`` means *absent* for four criteria and
*atypical* for none of them, and the earlier heuristic version of this module
inverted three concepts outright because it treated ``"a"`` as positive.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rpcp.data.manifest import ManifestConceptDataset
from rpcp.utils.logging import get_logger

__all__ = [
    "PH2_CLASSES",
    "PH2_CONCEPTS",
    "PH2_CONCEPT_GROUPS",
    "PH2_BINARY_CLASSES",
    "PH2Annotations",
    "annotation_report",
    "build_ph2",
    "concept_support",
    "prepare_manifest",
    "read_annotations",
]

logger = get_logger(__name__)

PH2_CLASSES: tuple[str, ...] = ("common_nevus", "atypical_nevus", "melanoma")
PH2_BINARY_CLASSES: tuple[str, ...] = ("nevus", "melanoma")

#: Binary dermoscopic criteria used as concepts, in manifest column order.
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

#: The six colour attributes form one group; the six structural criteria are
#: *independent* binary observations and must each be their own group.  The
#: grouped PCP KL (:func:`rpcp.losses.prior.original_pcp_kl`) renormalises
#: within a group, which is only meaningful for a mutually-exclusive set.
PH2_CONCEPT_GROUPS: list[list[int]] = [[0], [1], [2], [3], [4], [5], [6, 7, 8, 9, 10, 11]]

#: Colour code -> concept name, per the spreadsheet legend.
_COLOUR_CODES: dict[int, str] = {
    1: "colour_white",
    2: "colour_red",
    3: "colour_light_brown",
    4: "colour_dark_brown",
    5: "colour_blue_gray",
    6: "colour_black",
}

#: Spreadsheet header fragment -> (concept name, {raw value -> binary}).
#: ``None`` in a value map means "raise": an unexpected code is a data error,
#: not something to silently default to zero.
_CRITERIA: tuple[tuple[str, str, dict[str, int]], ...] = (
    ("asymmetry", "asymmetry", {"0": 0, "1": 1, "2": 1}),
    ("pigment_network", "atypical_pigment_network", {"T": 0, "AT": 1, "A": 0}),
    ("dots_globules", "atypical_dots_globules", {"A": 0, "T": 0, "AT": 1}),
    ("streaks", "streaks", {"A": 0, "P": 1}),
    ("regression_areas", "regression_areas", {"A": 0, "P": 1}),
    ("blue_whitish_veil", "blue_whitish_veil", {"A": 0, "P": 1}),
)

#: Ordinal asymmetry expansion used when ``ordinal_asymmetry=True``.
_ASYMMETRY_LEVELS: dict[str, tuple[int, int]] = {"0": (0, 0), "1": (1, 0), "2": (1, 1)}

_CLASS_COLUMNS: tuple[tuple[str, str], ...] = (
    ("common_nevus", "common_nevus"),
    ("atypical_nevus", "atypical_nevus"),
    ("melanoma", "melanoma"),
)


def _normalise(name: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


# --------------------------------------------------------------------------- #
# Reading the official annotation table
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class PH2Annotations:
    """Parsed PH2 annotations.

    Attributes:
        names: ``(N,)`` case identifiers, e.g. ``IMD003``.
        labels: ``(N,)`` class names drawn from :data:`PH2_CLASSES`.
        concepts: ``(N, M)`` binary concept matrix.
        concept_names: Length-``M`` concept names matching ``concepts`` columns.
        source: Which file the annotations were read from.
    """

    names: list[str]
    labels: list[str]
    concepts: np.ndarray
    concept_names: list[str]
    source: Path

    def __len__(self) -> int:
        return len(self.names)


def read_annotations(
    root: str | Path,
    *,
    spreadsheet: str | Path | None = None,
    ordinal_asymmetry: bool = False,
) -> PH2Annotations:
    """Parse the official PH2 annotation table.

    Prefers ``PH2_dataset.xlsx`` (in which the colour attributes are already
    expanded into six columns) and falls back to ``PH2_dataset.txt`` (in which
    they are a single space-separated set of codes).  Both produce identical
    output; the txt path exists because some mirrors ship only that file.

    Args:
        root: PH2 dataset root.
        spreadsheet: Explicit path to the annotation file, overriding discovery.
        ordinal_asymmetry: Emit ``asymmetry_any`` / ``asymmetry_full`` instead of
            a single ``asymmetry`` concept, preserving the 0/1/2 scale.

    Returns:
        A :class:`PH2Annotations` with 200 rows.
    """
    root = Path(root)
    path = Path(spreadsheet) if spreadsheet is not None else _find_annotation_file(root)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return _read_xlsx(path, ordinal_asymmetry=ordinal_asymmetry)
    return _read_txt(path, ordinal_asymmetry=ordinal_asymmetry)


def _find_annotation_file(root: Path) -> Path:
    for pattern in ("PH2_dataset.xlsx", "PH2 Dataset.xlsx", "*.xlsx", "PH2_dataset.txt", "*.txt"):
        matches = sorted(p for p in root.glob(pattern) if not p.name.startswith("."))
        matches = [p for p in matches if _normalise(p.stem) not in {"readme"}]
        if matches:
            return matches[0]
    raise FileNotFoundError(
        f"No PH2 annotation table (PH2_dataset.xlsx / PH2_dataset.txt) found under {root}"
    )


def _concept_names(ordinal_asymmetry: bool) -> list[str]:
    if not ordinal_asymmetry:
        return list(PH2_CONCEPTS)
    return ["asymmetry_any", "asymmetry_full", *PH2_CONCEPTS[1:]]


def _read_xlsx(path: Path, *, ordinal_asymmetry: bool) -> PH2Annotations:
    import pandas as pd

    raw = pd.read_excel(path, header=None)
    header_row = _locate_header_row(raw)
    frame = pd.read_excel(path, header=header_row)
    frame.columns = [_normalise(c) for c in frame.columns]

    name_col = _require_column(frame.columns, "image_name", "name")
    frame = frame[frame[name_col].astype(str).str.strip().str.match(r"^IMD\d+$", na=False)]
    if frame.empty:
        raise ValueError(f"No IMD* rows found in {path}")

    labels = _labels_from_onehot(frame, path)
    concept_names = _concept_names(ordinal_asymmetry)
    concepts = np.zeros((len(frame), len(concept_names)), dtype=np.float32)
    index = {name: i for i, name in enumerate(concept_names)}

    for fragment, concept, value_map in _CRITERIA:
        column = _require_column(frame.columns, fragment)
        values = frame[column].map(_clean_code)
        if concept == "asymmetry" and ordinal_asymmetry:
            levels = values.map(lambda v: _lookup(v, _ASYMMETRY_LEVELS, column))
            concepts[:, index["asymmetry_any"]] = [lv[0] for lv in levels]
            concepts[:, index["asymmetry_full"]] = [lv[1] for lv in levels]
            continue
        concepts[:, index[concept]] = [_lookup(v, value_map, column) for v in values]

    # Colour columns are pre-expanded in the spreadsheet and marked with 'X'.
    # Match on the exact normalised colour name ('light_brown', 'dark_brown'):
    # a substring match on 'brown' aliases the two together.
    for concept in _COLOUR_CODES.values():
        column = _require_column(frame.columns, concept.removeprefix("colour_"))
        concepts[:, index[concept]] = frame[column].map(_is_marked).to_numpy(dtype=np.float32)

    return PH2Annotations(
        names=[str(v).strip() for v in frame[name_col]],
        labels=labels,
        concepts=concepts,
        concept_names=concept_names,
        source=path,
    )


def _read_txt(path: Path, *, ordinal_asymmetry: bool) -> PH2Annotations:
    """Parse the pipe-delimited ``PH2_dataset.txt``.

    Empty cells must be preserved: ``Histological Diagnosis`` is blank for 159 of
    the 200 rows, so any split that drops empty fields silently shifts every
    later column by one.
    """
    lines = [
        line.rstrip("\n")
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
        if line.strip().startswith("|")
    ]
    if not lines:
        raise ValueError(f"{path} contains no pipe-delimited rows")

    header = _split_pipe_row(lines[0])
    columns = [_normalise(c) for c in header]
    rows = [_split_pipe_row(line) for line in lines[1:]]
    rows = [r for r in rows if r and re.match(r"^IMD\d+$", r[0].strip())]
    width = len(columns)
    rows = [(r + [""] * width)[:width] for r in rows]

    def column_values(fragment: str) -> list[str]:
        idx = _require_index(columns, fragment)
        return [_clean_code(r[idx]) for r in rows]

    name_idx = _require_index(columns, "name")
    names = [r[name_idx].strip() for r in rows]

    clinical = column_values("clinical_diagnosis")
    label_map = {"0": "common_nevus", "1": "atypical_nevus", "2": "melanoma"}
    labels = [_lookup_str(v, label_map, "clinical_diagnosis") for v in clinical]

    concept_names = _concept_names(ordinal_asymmetry)
    concepts = np.zeros((len(rows), len(concept_names)), dtype=np.float32)
    index = {name: i for i, name in enumerate(concept_names)}

    for fragment, concept, value_map in _CRITERIA:
        values = column_values(fragment)
        if concept == "asymmetry" and ordinal_asymmetry:
            levels = [_lookup(v, _ASYMMETRY_LEVELS, fragment) for v in values]
            concepts[:, index["asymmetry_any"]] = [lv[0] for lv in levels]
            concepts[:, index["asymmetry_full"]] = [lv[1] for lv in levels]
            continue
        concepts[:, index[concept]] = [_lookup(v, value_map, fragment) for v in values]

    colour_idx = _require_index(columns, "colors", "colours")
    for row_i, row in enumerate(rows):
        for token in str(row[colour_idx]).split():
            code = int(token)
            if code not in _COLOUR_CODES:
                raise ValueError(f"Unknown PH2 colour code {code!r} in row {names[row_i]}")
            concepts[row_i, index[_COLOUR_CODES[code]]] = 1.0

    return PH2Annotations(
        names=names,
        labels=labels,
        concepts=concepts,
        concept_names=concept_names,
        source=path,
    )


def _split_pipe_row(line: str) -> list[str]:
    """Split one ``|| a || b | c ||`` row, preserving empty cells."""
    stripped = line.strip()
    stripped = re.sub(r"^\|+", "", stripped)
    stripped = re.sub(r"\|+$", "", stripped)
    return [cell.strip() for cell in re.split(r"\|\|?", stripped)]


def _locate_header_row(raw: object) -> int:
    import pandas as pd

    assert isinstance(raw, pd.DataFrame)
    for i in range(len(raw)):
        cells = {_normalise(v) for v in raw.iloc[i].to_numpy()}
        if any("image" in c and "name" in c for c in cells):
            return i
        if "name" in cells and any("asymmetry" in c for c in cells):
            return i
    raise ValueError("Could not locate the PH2 header row (no 'Image Name' cell found)")


def _require_column(columns: Iterable[str], *fragments: str) -> str:
    return list(columns)[_require_index(list(columns), *fragments)]


def _require_index(columns: Sequence[str], *fragments: str) -> int:
    for fragment in fragments:
        for i, column in enumerate(columns):
            if column == fragment or column.startswith(f"{fragment}_"):
                return i
        for i, column in enumerate(columns):
            if fragment in column:
                return i
    raise KeyError(
        f"PH2 annotation table has no column matching {fragments}; columns are {list(columns)}"
    )


def _clean_code(value: object) -> str:
    import pandas as pd

    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return ""
    text = str(value).strip()
    if text.lower() in {"nan", "none"}:
        return ""
    # Excel stores the asymmetry scale as a float.
    if re.fullmatch(r"\d+\.0", text):
        text = text[:-2]
    return text.upper()


def _is_marked(value: object) -> float:
    return 1.0 if _clean_code(value) in {"X", "1", "YES", "TRUE", "P"} else 0.0


def _lookup(value: str, mapping: dict[str, object], column: str) -> object:
    if value not in mapping:
        raise ValueError(
            f"Unexpected PH2 value {value!r} in column {column!r}; "
            f"the legend allows {sorted(mapping)}"
        )
    return mapping[value]


def _lookup_str(value: str, mapping: dict[str, str], column: str) -> str:
    return str(_lookup(value, dict(mapping), column))


def _labels_from_onehot(frame: object, path: Path) -> list[str]:
    """Read the ``Common Nevus / Atypical Nevus / Melanoma`` one-hot triple."""
    import pandas as pd

    assert isinstance(frame, pd.DataFrame)
    marks = np.stack(
        [
            frame[_require_column(frame.columns, col)].map(_is_marked).to_numpy()
            for _, col in _CLASS_COLUMNS
        ],
        axis=1,
    )
    counts = marks.sum(axis=1)
    if not np.all(counts == 1):
        bad = int((counts != 1).sum())
        raise ValueError(
            f"{path}: {bad} row(s) do not have exactly one clinical-diagnosis mark "
            "(expected one 'X' among Common Nevus / Atypical Nevus / Melanoma)"
        )
    return [PH2_CLASSES[i] for i in marks.argmax(axis=1)]


# --------------------------------------------------------------------------- #
# Image discovery
# --------------------------------------------------------------------------- #
def _image_relpath(root: Path, name: str) -> str:
    """Locate one case's dermoscopic image, relative to ``root``.

    Handles the official per-case directory layout as well as the flattened
    ``images/IMD003.bmp`` layout that some re-distributions use.
    """
    candidates = [
        Path("PH2 Dataset images") / name / f"{name}_Dermoscopic_Image" / f"{name}.bmp",
        Path("PH2Dataset") / "PH2 Dataset images" / name / f"{name}_Dermoscopic_Image" / f"{name}.bmp",
        Path("images") / f"{name}.bmp",
        Path("images") / f"{name}.jpg",
        Path("images") / f"{name}.png",
    ]
    for candidate in candidates:
        if (root / candidate).exists():
            return candidate.as_posix()

    matches = sorted(
        p
        for p in root.rglob(f"{name}.*")
        if p.suffix.lower() in {".bmp", ".jpg", ".jpeg", ".png"} and not p.name.startswith(".")
    )
    if not matches:
        raise FileNotFoundError(f"No image found for PH2 case {name} under {root}")
    return matches[0].relative_to(root).as_posix()


def _mask_relpath(root: Path, name: str, kind: str) -> str:
    """Relative path of the lesion mask, or ``""`` when absent."""
    candidate = Path("PH2 Dataset images") / name / f"{name}_{kind}" / f"{name}_{kind}.bmp"
    if (root / candidate).exists():
        return candidate.as_posix()
    matches = sorted(
        p
        for p in root.rglob(f"{name}_{kind}*.bmp")
        if not p.name.startswith(".")
    )
    return matches[0].relative_to(root).as_posix() if matches else ""


def _roi_relpaths(root: Path, name: str) -> dict[str, str]:
    """Per-colour ROI masks, keyed by ``<concept>__mask``.

    PH2 ships ``<case>_roi/<case>_R<k>_Label<code>.bmp`` where ``code`` indexes
    :data:`_COLOUR_CODES`.  These give free *spatial* ground truth for the colour
    concepts and are recorded in the manifest so downstream code can use them
    (e.g. to supervise the concept attention maps).  A case may contain several
    regions for one colour; all of them are recorded, ``;``-separated.

    Every colour key is always present (empty when absent) so the manifest has a
    deterministic column set regardless of which case is read first.
    """
    directory = root / "PH2 Dataset images" / name / f"{name}_roi"
    found: dict[str, list[str]] = {concept: [] for concept in _COLOUR_CODES.values()}
    if directory.is_dir():
        for path in sorted(directory.glob(f"{name}_R*_Label*.bmp")):
            if path.name.startswith("."):
                continue
            match = re.search(r"Label(\d+)", path.name)
            if match is None:
                continue
            concept = _COLOUR_CODES.get(int(match.group(1)))
            if concept is not None:
                found[concept].append(path.relative_to(root).as_posix())
    return {f"{concept}__mask": ";".join(paths) for concept, paths in found.items()}


# --------------------------------------------------------------------------- #
# Manifest
# --------------------------------------------------------------------------- #
def prepare_manifest(
    root: str | Path,
    *,
    spreadsheet: str | Path | None = None,
    output: str | Path | None = None,
    binary_classes: bool = False,
    ordinal_asymmetry: bool = False,
    include_masks: bool = True,
    split_seed: int | None = None,
) -> Path:
    """Convert the official PH2 annotation table into the project manifest schema.

    Args:
        root: PH2 dataset root.
        spreadsheet: Explicit annotation file; discovered under ``root`` if omitted.
        output: Manifest destination (defaults to ``root/manifest.csv``).
        binary_classes: Collapse the two nevus classes into ``nevus``.
        ordinal_asymmetry: Split ``asymmetry`` into ``asymmetry_any`` and
            ``asymmetry_full`` instead of thresholding the 0/1/2 scale at 1.
        include_masks: Record lesion and per-colour ROI mask paths as extra
            ``*__mask`` columns (ignored by the concept loader, available to
            anything that wants spatial supervision).
        split_seed: When set, write a stratified ``split`` column with this seed
            so PH2 runs are reproducible across machines.  ``None`` leaves the
            split to :func:`rpcp.data.build_splits`.

    Returns:
        Path to the written manifest.
    """
    import pandas as pd

    root = Path(root)
    annotations = read_annotations(
        root, spreadsheet=spreadsheet, ordinal_asymmetry=ordinal_asymmetry
    )

    records: list[dict[str, object]] = []
    for i, name in enumerate(annotations.names):
        label = annotations.labels[i]
        if binary_classes:
            label = "melanoma" if label == "melanoma" else "nevus"
        record: dict[str, object] = {
            "image": _image_relpath(root, name),
            "label": label,
            "case": name,
        }
        for m, concept in enumerate(annotations.concept_names):
            record[concept] = int(annotations.concepts[i, m])
        if include_masks:
            record["lesion__mask"] = _mask_relpath(root, name, "lesion")
            record.update(_roi_relpaths(root, name))
        records.append(record)

    columns = ["image", "label", "case", *annotations.concept_names]
    if include_masks:
        columns += ["lesion__mask", *(f"{c}__mask" for c in _COLOUR_CODES.values())]
    manifest = pd.DataFrame.from_records(records, columns=columns)
    for column in manifest.columns:
        if column.endswith("__mask"):
            manifest[column] = manifest[column].fillna("")

    if split_seed is not None:
        from rpcp.data.base import stratified_split

        classes = sorted(set(manifest["label"]))
        codes = np.array([classes.index(v) for v in manifest["label"]], dtype=np.int64)
        indices = stratified_split(codes, {"val": 0.2, "test": 0.2}, seed=split_seed)
        split = np.empty(len(manifest), dtype=object)
        for name, idx in indices.items():
            split[idx] = name
        manifest["split"] = split

    output = Path(output or root / "manifest.csv")
    manifest.to_csv(output, index=False)
    logger.info(
        "Wrote PH2 manifest with %d rows and %d concepts to %s (source: %s)",
        len(manifest),
        len(annotations.concept_names),
        output,
        annotations.source.name,
    )
    for line in annotation_report(annotations).splitlines():
        logger.info("%s", line)
    return output


def annotation_report(annotations: PH2Annotations) -> str:
    """Human-readable prevalence table, with a warning for degenerate concepts.

    A concept whose class-conditional prevalence is constant carries no
    class-level signal at all: ``L_prior`` and ``L_match`` are both independent
    of it, so no prior-supervised method can recover it and its AUROC is chance
    by construction.  Catching that here is cheaper than discovering it in a
    results table.
    """
    labels = np.asarray(annotations.labels)
    classes = [c for c in PH2_CLASSES if c in set(labels)] or sorted(set(labels))
    concepts = annotations.concepts

    header = f"{'concept':26} {'overall':>8}" + "".join(f"{c[:14]:>16}" for c in classes) + "   range"
    lines = [header, "-" * len(header)]
    degenerate: list[str] = []
    for m, name in enumerate(annotations.concept_names):
        per_class = [
            float(concepts[labels == c, m].mean()) if (labels == c).any() else float("nan")
            for c in classes
        ]
        spread = float(np.nanmax(per_class) - np.nanmin(per_class))
        lines.append(
            f"{name:26} {concepts[:, m].mean():8.3f}"
            + "".join(f"{v:16.3f}" for v in per_class)
            + f"   {spread:6.3f}"
        )
        if spread < 1e-6 or concepts[:, m].mean() in (0.0, 1.0):
            degenerate.append(name)
    if degenerate:
        lines.append("")
        lines.append(f"WARNING: {len(degenerate)} concept(s) carry no class-level signal: {degenerate}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
def build_ph2(
    root: str | Path,
    *,
    manifest: str | Path | None = None,
    transform: object | None = None,
    binary_classes: bool = False,
    ordinal_asymmetry: bool = False,
    rebuild: bool = False,
) -> ManifestConceptDataset:
    """Instantiate the PH2 dataset, preparing the manifest on first use."""
    root = Path(root)
    manifest = Path(manifest or root / "manifest.csv")
    if rebuild or not manifest.exists():
        prepare_manifest(
            root,
            output=manifest,
            binary_classes=binary_classes,
            ordinal_asymmetry=ordinal_asymmetry,
        )

    concept_columns = _concept_names(ordinal_asymmetry)
    classes = PH2_BINARY_CLASSES if binary_classes else PH2_CLASSES
    dataset = ManifestConceptDataset(
        root=root,
        manifest=manifest,
        concept_columns=concept_columns,
        class_names=classes,
        transform=transform,
    )
    logger.info("PH2: %s", dataset.describe())
    return dataset


def concept_support(dataset: ManifestConceptDataset) -> np.ndarray:
    """Number of positive annotations per concept (useful for sanity checks)."""
    assert dataset.concepts is not None
    return dataset.concepts.sum(axis=0)
