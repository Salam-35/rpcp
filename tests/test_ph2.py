"""PH2 annotation-parsing tests.

These pin the *semantics* of the official PH2 codes.  The previous heuristic
parser matched column names by substring and treated the literal ``"A"`` as a
positive value, which inverted ``streaks``, ``regression_areas`` and
``blue_whitish_veil`` outright (``A`` means *absent*, not *atypical*) and marked
``atypical_dots_globules`` positive for all 87 absent cases.  A single assertion
on the value map would have caught it, so here it is.

The fixtures synthesise both official file formats, so the tests run without the
200-image download.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from rpcp.data.ph2 import (
    PH2_CLASSES,
    PH2_CONCEPTS,
    annotation_report,
    build_ph2,
    prepare_manifest,
    read_annotations,
)

# case, clinical class, asymmetry, pigment network, dots/globules, streaks,
# regression areas, blue-whitish veil, colour codes
CASES: tuple[tuple[str, str, str, str, str, str, str, str, tuple[int, ...]], ...] = (
    ("IMD003", "common_nevus", "0", "T", "A", "A", "A", "A", (4,)),
    ("IMD035", "common_nevus", "2", "T", "A", "A", "A", "A", (2, 3)),
    ("IMD146", "atypical_nevus", "1", "T", "A", "P", "A", "A", (2, 3, 4)),
    ("IMD405", "melanoma", "2", "AT", "AT", "P", "P", "P", (1, 4, 5, 6)),
)

_STRUCTURAL = ["asymmetry", "pigment_network", "dots_globules", "streaks",
               "regression_areas", "blue_whitish_veil"]


def _write_xlsx(root: Path) -> Path:
    import pandas as pd

    rows = [[None] * 17 for _ in range(13)]
    rows[11][2] = "Clinical Diagnosis"
    rows[11][11] = "Colors"
    rows[12] = [
        "Image Name", "Histological Diagnosis", "Common Nevus", "Atypical Nevus", "Melanoma",
        "Asymmetry\n(0/1/2)", "Pigment Network\n(AT/T)", "Dots/Globules\n(A/AT/T)",
        "Streaks\n(A/P)", "Regression Areas\n(A/P)", "Blue-Whitish Veil\n(A/P)",
        "White", "Red", "Light-Brown", "Dark-Brown", "Blue-Gray", "Black",
    ]
    for name, label, *criteria in CASES:
        colours = criteria[-1]
        row: list[object] = [name, None, None, None, None, *criteria[:-1]]
        row[2 + PH2_CLASSES.index(label)] = "X"
        row += ["X" if code in colours else None for code in range(1, 7)]
        rows.append(row)

    path = root / "PH2_dataset.xlsx"
    pd.DataFrame(rows).to_excel(path, header=False, index=False)
    return path


def _write_txt(root: Path) -> Path:
    header = (
        "||   Name || Histological Diagnosis || Clinical Diagnosis || Asymmetry | "
        "Pigment Network | Dots/Globules | Streaks | Regression Areas | "
        "Blue-Whitish Veil ||           Colors ||"
    )
    lines = [header]
    for name, label, *criteria in CASES:
        colours = "  ".join(str(c) for c in criteria[-1])
        middle = " | ".join(str(v) for v in criteria[:-1])
        lines.append(f"|| {name} ||  || {PH2_CLASSES.index(label)} || {middle} || {colours} ||")
    path = root / "PH2_dataset.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _write_images(root: Path) -> None:
    for name, *_ in CASES:
        case = root / "PH2 Dataset images" / name
        (case / f"{name}_Dermoscopic_Image").mkdir(parents=True, exist_ok=True)
        (case / f"{name}_Dermoscopic_Image" / f"{name}.bmp").touch()
        (case / f"{name}_lesion").mkdir(parents=True, exist_ok=True)
        (case / f"{name}_lesion" / f"{name}_lesion.bmp").touch()


@pytest.fixture
def ph2_root(tmp_path: Path) -> Path:
    _write_xlsx(tmp_path)
    _write_txt(tmp_path)
    _write_images(tmp_path)
    return tmp_path


# --------------------------------------------------------------------------- #
def test_absent_codes_are_negative(ph2_root: Path) -> None:
    """``A`` means *absent*; it must never be read as a positive concept."""
    annotations = read_annotations(ph2_root)
    index = {name: i for i, name in enumerate(annotations.concept_names)}
    imd003 = annotations.names.index("IMD003")

    for concept in ("atypical_dots_globules", "streaks", "regression_areas", "blue_whitish_veil"):
        assert annotations.concepts[imd003, index[concept]] == 0.0, (
            f"IMD003 has '{concept}' coded 'A' (absent) and must be negative"
        )


def test_present_and_atypical_codes_are_positive(ph2_root: Path) -> None:
    annotations = read_annotations(ph2_root)
    index = {name: i for i, name in enumerate(annotations.concept_names)}
    melanoma = annotations.names.index("IMD405")

    for concept in (
        "asymmetry",
        "atypical_pigment_network",
        "atypical_dots_globules",
        "streaks",
        "regression_areas",
        "blue_whitish_veil",
    ):
        assert annotations.concepts[melanoma, index[concept]] == 1.0, concept

    # 'P' on IMD146 streaks, 'A' elsewhere.
    partial = annotations.names.index("IMD146")
    assert annotations.concepts[partial, index["streaks"]] == 1.0
    assert annotations.concepts[partial, index["regression_areas"]] == 0.0


def test_typical_is_not_atypical(ph2_root: Path) -> None:
    """``T`` (typical) is a *negative* observation for the 'atypical_*' concepts."""
    annotations = read_annotations(ph2_root)
    index = {name: i for i, name in enumerate(annotations.concept_names)}
    for case in ("IMD003", "IMD035", "IMD146"):
        row = annotations.names.index(case)
        assert annotations.concepts[row, index["atypical_pigment_network"]] == 0.0, case


def test_colour_codes_map_to_the_right_concepts(ph2_root: Path) -> None:
    annotations = read_annotations(ph2_root)
    index = {name: i for i, name in enumerate(annotations.concept_names)}
    row = annotations.names.index("IMD405")
    expected = {
        "colour_white": 1.0,      # code 1
        "colour_red": 0.0,        # code 2 absent
        "colour_light_brown": 0.0,
        "colour_dark_brown": 1.0,  # code 4
        "colour_blue_gray": 1.0,   # code 5
        "colour_black": 1.0,       # code 6
    }
    for concept, value in expected.items():
        assert annotations.concepts[row, index[concept]] == value, concept

    # light-brown and dark-brown must not alias each other (they share 'brown').
    imd003 = annotations.names.index("IMD003")
    assert annotations.concepts[imd003, index["colour_dark_brown"]] == 1.0
    assert annotations.concepts[imd003, index["colour_light_brown"]] == 0.0


def test_asymmetry_scale(ph2_root: Path) -> None:
    plain = read_annotations(ph2_root)
    index = {n: i for i, n in enumerate(plain.concept_names)}
    assert plain.concepts[plain.names.index("IMD003"), index["asymmetry"]] == 0.0  # level 0
    assert plain.concepts[plain.names.index("IMD146"), index["asymmetry"]] == 1.0  # level 1
    assert plain.concepts[plain.names.index("IMD035"), index["asymmetry"]] == 1.0  # level 2

    ordinal = read_annotations(ph2_root, ordinal_asymmetry=True)
    index = {n: i for i, n in enumerate(ordinal.concept_names)}
    partial, full = ordinal.names.index("IMD146"), ordinal.names.index("IMD035")
    assert ordinal.concepts[partial, index["asymmetry_any"]] == 1.0
    assert ordinal.concepts[partial, index["asymmetry_full"]] == 0.0
    assert ordinal.concepts[full, index["asymmetry_full"]] == 1.0


def test_labels_come_from_the_clinical_diagnosis_one_hot(ph2_root: Path) -> None:
    annotations = read_annotations(ph2_root)
    assert annotations.labels == [label for _, label, *_ in CASES]


def test_xlsx_and_txt_agree(ph2_root: Path) -> None:
    """The two official formats must parse identically."""
    from_xlsx = read_annotations(ph2_root, spreadsheet=ph2_root / "PH2_dataset.xlsx")
    from_txt = read_annotations(ph2_root, spreadsheet=ph2_root / "PH2_dataset.txt")
    assert from_xlsx.names == from_txt.names
    assert from_xlsx.labels == from_txt.labels
    assert np.array_equal(from_xlsx.concepts, from_txt.concepts)


def test_txt_blank_histology_does_not_shift_columns(ph2_root: Path) -> None:
    """159/200 real rows have an empty Histological Diagnosis cell.

    A split that drops empty fields shifts every later column by one, silently
    reading the pigment-network code as the asymmetry score.
    """
    annotations = read_annotations(ph2_root, spreadsheet=ph2_root / "PH2_dataset.txt")
    index = {n: i for i, n in enumerate(annotations.concept_names)}
    assert set(np.unique(annotations.concepts[:, index["asymmetry"]])) <= {0.0, 1.0}


def test_unknown_code_raises(tmp_path: Path) -> None:
    """A code outside the legend is a data error, not a silent zero."""
    import pandas as pd

    _write_xlsx(tmp_path)
    frame = pd.read_excel(tmp_path / "PH2_dataset.xlsx", header=None)
    frame.iat[13, 8] = "Q"  # Streaks column, first data row
    frame.to_excel(tmp_path / "PH2_dataset.xlsx", header=False, index=False)
    with pytest.raises(ValueError, match="Unexpected PH2 value"):
        read_annotations(tmp_path, spreadsheet=tmp_path / "PH2_dataset.xlsx")


def test_manifest_roundtrip(ph2_root: Path) -> None:
    manifest = prepare_manifest(ph2_root)
    assert manifest.exists()
    dataset = build_ph2(ph2_root, manifest=manifest)
    assert dataset.concept_names == list(PH2_CONCEPTS)
    assert len(dataset) == len(CASES)
    assert dataset.class_names == list(PH2_CLASSES)

    # Rebuilding must be byte-identical (deterministic column order).
    again = prepare_manifest(ph2_root, output=ph2_root / "again.csv")
    assert manifest.read_text() == again.read_text()


def test_binary_classes(ph2_root: Path) -> None:
    manifest = prepare_manifest(ph2_root, output=ph2_root / "bin.csv", binary_classes=True)
    dataset = build_ph2(ph2_root, manifest=manifest, binary_classes=True)
    assert dataset.class_names == ["nevus", "melanoma"]
    assert int((dataset.labels == 1).sum()) == 1


def test_report_lists_every_concept(ph2_root: Path) -> None:
    report = annotation_report(read_annotations(ph2_root))
    for concept in PH2_CONCEPTS:
        assert concept in report
    for class_name in PH2_CLASSES:
        assert class_name[:14] in report


def test_report_flags_constant_concepts() -> None:
    """A concept with no class-level variation cannot be learned from priors.

    Both ``L_prior`` and ``L_match`` are independent of such a concept's
    per-image value, so its AUROC is chance by construction; the report must say
    so rather than let it quietly drag the macro average down.  On the real PH2
    table ``colour_red`` is exactly this case: prevalence 0.05 in all three
    classes, hence class-conditional range 0.
    """
    from rpcp.data.ph2 import PH2Annotations

    labels = ["common_nevus"] * 2 + ["atypical_nevus"] * 2 + ["melanoma"] * 2
    concepts = np.zeros((6, 3), dtype=np.float32)
    concepts[:, 0] = [0, 1, 0, 1, 0, 1]  # prevalence 0.5 in every class -> no signal
    concepts[:, 1] = [0, 0, 1, 1, 1, 1]  # genuine class signal
    concepts[:, 2] = 1.0  # constant
    annotations = PH2Annotations(
        names=list("abcdef"),
        labels=labels,
        concepts=concepts,
        concept_names=["flat", "informative", "always_on"],
        source=Path("synthetic"),
    )

    report = annotation_report(annotations)
    warning = report.splitlines()[-1]
    assert "WARNING" in warning
    assert "flat" in warning and "always_on" in warning
    assert "'informative'" not in warning
