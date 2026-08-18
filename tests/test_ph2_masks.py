"""Tests for PH2 colour-mask loading (spatial supervision hook)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from rpcp.data.ph2 import load_colour_mask


def _write_mask(root: Path, relpath: str, region: np.ndarray) -> None:
    from PIL import Image

    path = root / relpath
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((region.astype(np.uint8)) * 255, mode="L").save(path)


def test_empty_cell_returns_none(tmp_path: Path) -> None:
    assert load_colour_mask(tmp_path, "") is None
    assert load_colour_mask(tmp_path, "   ") is None


def test_loads_a_single_region(tmp_path: Path) -> None:
    region = np.zeros((8, 8), dtype=bool)
    region[2:5, 2:5] = True
    _write_mask(tmp_path, "IMD003_roi/IMD003_R1_Label4.bmp", region)

    mask = load_colour_mask(tmp_path, "IMD003_roi/IMD003_R1_Label4.bmp")
    assert mask is not None
    assert mask.shape == (8, 8)
    assert bool(mask[3, 3] == 1.0)
    assert bool(mask[0, 0] == 0.0)


def test_unions_multiple_regions(tmp_path: Path) -> None:
    region_a = np.zeros((8, 8), dtype=bool)
    region_a[0:2, 0:2] = True
    region_b = np.zeros((8, 8), dtype=bool)
    region_b[6:8, 6:8] = True
    _write_mask(tmp_path, "IMD003_roi/IMD003_R1_Label4.bmp", region_a)
    _write_mask(tmp_path, "IMD003_roi/IMD003_R2_Label4.bmp", region_b)

    cell = "IMD003_roi/IMD003_R1_Label4.bmp;IMD003_roi/IMD003_R2_Label4.bmp"
    mask = load_colour_mask(tmp_path, cell)
    assert mask is not None
    assert bool(mask[0, 0] == 1.0)
    assert bool(mask[7, 7] == 1.0)
    assert bool(mask[4, 4] == 0.0)
    assert float(mask.sum()) == pytest.approx(8.0)  # 2x2 + 2x2


def test_resizes_when_size_given(tmp_path: Path) -> None:
    region = np.ones((8, 8), dtype=bool)
    _write_mask(tmp_path, "IMD003_roi/IMD003_R1_Label4.bmp", region)
    mask = load_colour_mask(
        tmp_path, "IMD003_roi/IMD003_R1_Label4.bmp", size=(4, 4)
    )
    assert mask is not None
    assert mask.shape == (4, 4)
    assert bool((mask == 1.0).all())
