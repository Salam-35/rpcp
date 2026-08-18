"""Tests for the null-floor / identifiability-ceiling baselines."""

from __future__ import annotations

import numpy as np
import pytest

from rpcp.evaluation.baselines import (
    BaselineMetrics,
    concept_baselines,
    identifiability_ceiling,
    null_concept_baseline,
)


def test_null_baseline_is_the_global_majority_vote() -> None:
    # Concept is present in 3/4 rows overall -> majority baseline predicts 1
    # everywhere: TP=3, FP=1, FN=0 -> precision=0.75, recall=1 -> F1=6/7.
    targets = np.array([[1.0], [1.0], [1.0], [0.0]])
    macro, per_concept = null_concept_baseline(targets)
    assert macro == pytest.approx(6 / 7, abs=1e-6)
    assert per_concept[0] == pytest.approx(6 / 7, abs=1e-6)


def test_ceiling_is_perfect_when_concepts_are_class_deterministic() -> None:
    # Every class-0 image has concept=0, every class-1 image has concept=1:
    # the class-conditional mean IS the per-image ground truth, so the
    # ceiling must be 1.0.
    targets = np.array([[0.0], [0.0], [1.0], [1.0]])
    labels = np.array([0, 0, 1, 1])
    macro, per_concept = identifiability_ceiling(targets, labels, n_classes=2)
    assert macro == 1.0
    assert per_concept[0] == 1.0


def test_ceiling_is_below_one_when_concepts_vary_within_a_class() -> None:
    # Class 0 has a 50/50 split on the concept -> no class-only model can
    # separate those two images, so F1 for that concept must be < 1.
    targets = np.array([[1.0], [0.0], [1.0], [1.0]])
    labels = np.array([0, 0, 1, 1])
    macro, _ = identifiability_ceiling(targets, labels, n_classes=2)
    assert macro < 1.0


def test_ceiling_is_at_least_the_null_floor_on_typical_data() -> None:
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 3, size=200)
    # Concepts correlated with class, so there is real class-level signal.
    base = (labels % 3 == 0).astype(np.float64)
    noise = rng.random(200) < 0.1
    targets = np.abs(base - noise.astype(np.float64))[:, None]
    baselines = concept_baselines(targets, labels, n_classes=3)
    assert baselines.ceiling_macro_f1 >= baselines.null_macro_f1 - 1e-9


def test_progress_places_a_mid_score_between_zero_and_one() -> None:
    metrics = BaselineMetrics(
        null_macro_f1=0.3,
        ceiling_macro_f1=0.8,
        null_per_concept_f1=np.array([0.3]),
        ceiling_per_concept_f1=np.array([0.8]),
    )
    assert metrics.progress(0.55) == pytest.approx(0.5, abs=1e-6)


def test_progress_handles_a_degenerate_zero_span() -> None:
    metrics = BaselineMetrics(
        null_macro_f1=0.5,
        ceiling_macro_f1=0.5,
        null_per_concept_f1=np.array([0.5]),
        ceiling_per_concept_f1=np.array([0.5]),
    )
    assert np.isnan(metrics.progress(0.6))
