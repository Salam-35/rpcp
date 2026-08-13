"""End-to-end smoke tests: data -> priors -> training -> evaluation -> figures."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from rpcp.config import ExperimentConfig
from rpcp.data import SplitBundle, build_prior_bundle, build_splits
from rpcp.data.priors import PriorBundle
from rpcp.evaluation import evaluate_reliability
from rpcp.methods import METHODS, apply_method
from rpcp.models.rpcp import build_model
from rpcp.plotting import (
    plot_corruption_robustness,
    plot_method_overview,
    plot_reliability_heatmaps,
)
from rpcp.plotting.robustness_curves import Curve
from rpcp.training import RPCPTrainer, run_experiment


# --------------------------------------------------------------------------- #
# Data + priors
# --------------------------------------------------------------------------- #
def test_splits_are_disjoint_and_stratified(tiny_splits: SplitBundle) -> None:
    train = set(np.asarray(tiny_splits.train.indices).tolist())  # type: ignore[attr-defined]
    val = set(np.asarray(tiny_splits.val.indices).tolist())  # type: ignore[attr-defined]
    test = set(np.asarray(tiny_splits.test.indices).tolist())  # type: ignore[attr-defined]
    assert not (train & val) and not (train & test) and not (val & test)
    assert len(np.unique(tiny_splits.train.labels)) == tiny_splits.n_classes  # type: ignore[attr-defined]


def test_batches_have_the_documented_shape(tiny_splits: SplitBundle) -> None:
    sample = tiny_splits.train[0]
    assert set(sample) == {"image", "label", "concepts", "has_concepts", "index"}
    assert sample["image"].ndim == 3
    assert sample["concepts"].shape == (tiny_splits.n_concepts,)


def test_prior_bundle_tracks_corruption(tiny_config: ExperimentConfig) -> None:
    config = tiny_config.replace(
        **{
            "priors.corruption.mode": "adversarial_flip",
            "priors.corruption.alpha": 1.0,
            "priors.corruption.fraction": 0.5,
        }
    )
    splits = build_splits(config.data)
    priors = build_prior_bundle(config, splits)
    assert priors.observed.shape == (splits.n_concepts, splits.n_classes)
    assert priors.corruption_mask is not None
    assert 0.2 < float(priors.corruption_mask.float().mean()) < 0.8
    error = priors.prior_error()
    assert error is not None and float(error.max()) > 0.1


def test_synthetic_priors_match_the_generative_prior(
    tiny_splits: SplitBundle, tiny_priors: PriorBundle
) -> None:
    world = tiny_splits.source.world  # type: ignore[attr-defined]
    assert torch.allclose(tiny_priors.observed, world.prior, atol=0.25)


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def test_model_forward_shapes(tiny_config: ExperimentConfig, tiny_splits: SplitBundle) -> None:
    model = build_model(
        tiny_config, n_concepts=tiny_splits.n_concepts, n_classes=tiny_splits.n_classes
    )
    images = torch.stack([tiny_splits.train[i]["image"] for i in range(4)])
    output = model(images)
    assert output.concept_probs.shape == (4, tiny_splits.n_concepts)
    assert output.class_logits.shape == (4, tiny_splits.n_classes)
    assert output.attention is not None
    # Attention is a distribution over locations.
    assert torch.allclose(
        output.attention.flatten(2).sum(-1), torch.ones(4, tiny_splits.n_concepts), atol=1e-4
    )


def test_prior_class_head_needs_a_prior_table(
    tiny_config: ExperimentConfig, tiny_splits: SplitBundle
) -> None:
    config = tiny_config.replace(**{"model.class_head": "prior"})
    with pytest.raises(ValueError, match="prior table"):
        build_model(config, n_concepts=tiny_splits.n_concepts, n_classes=tiny_splits.n_classes)


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("method", ["pcp", "r-pcp", "oracle"])
def test_training_runs_end_to_end(tiny_config: ExperimentConfig, method: str) -> None:
    config = apply_method(
        tiny_config,
        method,
        extra={
            "priors.corruption.mode": "class_swap",
            "priors.corruption.alpha": 1.0,
            "priors.corruption.fraction": 0.4,
        },
    )
    result = run_experiment(config)
    summary = result.summary()

    assert np.isfinite(summary["test/class_macro_f1"])
    assert np.isfinite(summary["test/concept_macro_f1"])
    assert result.reliability_matrix.shape == (
        config.data.synthetic.n_concepts,
        config.data.synthetic.n_classes,
    )
    assert (result.run_dir / "history.jsonl").exists()
    assert (result.run_dir / "summary.json").exists()
    if method == "pcp":
        assert np.allclose(result.reliability_matrix, 1.0)
    if method == "oracle":
        assert np.isfinite(summary["reliability/auroc"])


def test_reliability_moves_away_from_one_for_rpcp(tiny_config: ExperimentConfig) -> None:
    config = apply_method(
        tiny_config,
        "r-pcp",
        extra={
            "priors.corruption.mode": "adversarial_flip",
            "priors.corruption.alpha": 1.0,
            "priors.corruption.fraction": 0.5,
            "optim.epochs": 4,
            "reliability.warmup_epochs": 1,
            "reliability.mid_epochs": 2,
        },
    )
    result = run_experiment(config)
    matrix = result.reliability_matrix
    # Neither collapsed to zero nor stuck at one (plan Risk 3 / Risk 4).
    assert 0.0 < matrix.mean() < 1.0
    assert matrix.std() > 0.0


def test_audit_split_is_removed_from_training(tiny_config: ExperimentConfig) -> None:
    config = tiny_config.replace(**{"data.audit_fraction": 0.2})
    splits = build_splits(config.data)
    assert splits.audit is not None
    audit = set(np.asarray(splits.audit.indices).tolist())  # type: ignore[attr-defined]
    train = set(np.asarray(splits.train.indices).tolist())  # type: ignore[attr-defined]
    assert not (audit & train)
    priors = build_prior_bundle(config, splits)
    assert priors.audit is not None


def test_supervised_cbm_uses_concept_labels(tiny_config: ExperimentConfig) -> None:
    config = apply_method(tiny_config, "supervised-cbm")
    assert config.loss.lambda_concept > 0
    result = run_experiment(config)
    assert np.isfinite(result.summary()["test/concept_macro_f1"])


def test_trainer_phases_gate_reliability(
    tiny_config: ExperimentConfig, tiny_splits: SplitBundle, tiny_priors: PriorBundle
) -> None:
    config = apply_method(tiny_config, "r-pcp")
    trainer = RPCPTrainer(config, tiny_splits, tiny_priors)
    assert not trainer.schedule.use_reliability(0)
    assert trainer.schedule.use_reliability(config.optim.epochs - 1)
    assert not trainer.schedule.should_update_reliability(0)


def test_every_method_preset_builds(tiny_config: ExperimentConfig) -> None:
    for method in METHODS:
        extra: dict[str, object] = {}
        if method == "oracle":
            extra = {"priors.corruption.mode": "uniform", "priors.corruption.alpha": 0.5}
        if method == "r-pcp-multirater":
            continue  # requires a multi-rater dataset
        config = apply_method(tiny_config, method, extra=extra)
        assert isinstance(config, ExperimentConfig)


# --------------------------------------------------------------------------- #
# Evaluation + plotting
# --------------------------------------------------------------------------- #
def test_reliability_audit_on_a_finished_run(tiny_config: ExperimentConfig) -> None:
    config = apply_method(
        tiny_config,
        "oracle",
        extra={"priors.corruption.mode": "uniform", "priors.corruption.alpha": 1.0},
    )
    splits = build_splits(config.data)
    priors = build_prior_bundle(config, splits)
    audit = evaluate_reliability(torch.as_tensor(priors.clean_mask, dtype=torch.float32), priors)
    assert audit.auroc == pytest.approx(1.0)


def test_figures_render(tmp_path: Path, tiny_priors: PriorBundle) -> None:
    overview = tmp_path / "figure1.png"
    plot_method_overview(path=overview)
    assert overview.exists()

    heatmap = tmp_path / "figure3.png"
    plot_reliability_heatmaps(
        np.random.rand(*tiny_priors.shape),
        corruption_mask=np.random.rand(*tiny_priors.shape) > 0.5,
        prior_error=np.random.rand(*tiny_priors.shape) * 0.5,
        path=heatmap,
    )
    assert heatmap.exists()

    curve = tmp_path / "figure2.png"
    plot_corruption_robustness(
        {
            "pcp": Curve(x=[0, 0.5, 1.0], y=[0.7, 0.5, 0.3]),
            "r-pcp": Curve(x=[0, 0.5, 1.0], y=[0.7, 0.62, 0.5], yerr=[0.01, 0.02, 0.03]),
        },
        path=curve,
    )
    assert curve.exists()
