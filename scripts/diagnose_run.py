#!/usr/bin/env python3
"""Deep diagnostics for a saved R-PCP run.

The script writes tables that make failure modes inspectable:

* prior/reliability/effective-target entries, including corruption labels;
* per-concept and per-class concept means/residuals;
* per-sample predictions, probabilities and confidences;
* feature-space separability summaries from pooled backbone features;
* class-head metrics with and without reliability passed at evaluation time.

Example::

    python scripts/diagnose_run.py --run-dir runs/rpcp-derm7pt-r-pcp-audit_seed0
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rpcp.config import ExperimentConfig  # noqa: E402
from rpcp.data import build_dataloaders, build_prior_bundle, build_splits  # noqa: E402
from rpcp.data.priors import PriorBundle  # noqa: E402
from rpcp.evaluation.calibration import brier_score  # noqa: E402
from rpcp.evaluation.class_metrics import class_metrics  # noqa: E402
from rpcp.evaluation.concept_metrics import binary_f1, concept_metrics  # noqa: E402
from rpcp.evaluation.ranking import roc_auc  # noqa: E402
from rpcp.evaluation.reliability_metrics import reliability_metrics  # noqa: E402
from rpcp.models.rpcp import build_model  # noqa: E402
from rpcp.utils.io import from_dict, save_json  # noqa: E402
from rpcp.utils.logging import configure_logging, get_logger  # noqa: E402

logger = get_logger("diagnose")


@dataclass(slots=True)
class DebugPredictions:
    indices: np.ndarray
    labels: np.ndarray
    concepts: np.ndarray | None
    concept_probs: np.ndarray
    concept_logits: np.ndarray
    class_probs: np.ndarray
    class_probs_with_reliability: np.ndarray | None
    features: np.ndarray


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--split", default="test", choices=["train_eval", "val", "test", "audit"])
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args(argv)
    configure_logging()

    checkpoint_path = args.run_dir / "checkpoint.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"{checkpoint_path} is missing. Re-run training with eval.save_checkpoints=true."
        )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config: ExperimentConfig = from_dict(ExperimentConfig, checkpoint["config"])
    device = torch.device(args.device or config.resolved_device())

    splits = build_splits(config.data)
    priors = build_prior_bundle(config, splits)
    loaders = build_dataloaders(splits, config.data, seed=config.seed)
    if args.split not in loaders:
        raise KeyError(f"Split '{args.split}' is unavailable; have {sorted(loaders)}")

    model = build_model(
        config,
        n_concepts=splits.n_concepts,
        n_classes=splits.n_classes,
        priors=priors.observed,
    )
    model.load_state_dict(checkpoint["model"])
    model.to(device)

    reliability = torch.as_tensor(checkpoint.get("reliability", np.ones(priors.shape))).float()
    effective_prior = build_effective_prior(config, priors, reliability)
    debug = collect_debug_predictions(
        model,
        loaders[args.split],  # type: ignore[arg-type]
        device=device,
        priors=priors.observed.to(device),
        reliability=reliability.to(device),
    )

    output_dir = args.output_dir or args.run_dir / f"diagnostics_{args.split}"
    output_dir.mkdir(parents=True, exist_ok=True)

    write_prior_table(output_dir, priors, reliability, effective_prior, debug, splits.concept_names, splits.class_names, config.eval.concept_threshold)
    write_concept_tables(output_dir, debug, splits.concept_names, splits.class_names, config.eval.concept_threshold)
    write_sample_tables(output_dir, debug, splits.concept_names, splits.class_names, config.eval.concept_threshold, max_samples=args.max_samples)
    feature_summary = write_feature_tables(output_dir, debug, splits.concept_names, splits.class_names)

    summary = build_summary(
        config=config,
        priors=priors,
        reliability=reliability,
        effective_prior=effective_prior,
        debug=debug,
        concept_names=splits.concept_names,
        class_names=splits.class_names,
        threshold=config.eval.concept_threshold,
        feature_summary=feature_summary,
    )
    save_json(summary, output_dir / "summary.json")
    print(json.dumps(summary, indent=2, default=float))
    logger.info("Wrote diagnostics to %s", output_dir)
    return 0


def build_effective_prior(
    config: ExperimentConfig,
    priors: PriorBundle,
    reliability: torch.Tensor,
) -> torch.Tensor:
    observed = priors.observed.float()
    r = reliability.float().clamp(config.reliability.min_reliability, config.reliability.max_reliability)
    match config.loss.prior_repair:
        case "none":
            return observed
        case "background":
            rho = observed.mean(dim=1, keepdim=True)
            return r * observed + (1.0 - r) * rho
        case "audit":
            if priors.audit is None:
                return observed
            return r * observed + (1.0 - r) * priors.audit.float()
        case other:
            raise ValueError(f"Unknown loss.prior_repair='{other}'")


@torch.no_grad()
def collect_debug_predictions(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader[dict[str, torch.Tensor]],
    *,
    device: torch.device,
    priors: torch.Tensor,
    reliability: torch.Tensor,
) -> DebugPredictions:
    was_training = model.training
    model.eval()

    indices: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    concepts: list[np.ndarray] = []
    concept_probs: list[np.ndarray] = []
    concept_logits: list[np.ndarray] = []
    class_probs: list[np.ndarray] = []
    class_probs_with_reliability: list[np.ndarray] = []
    features: list[np.ndarray] = []
    any_concepts = False

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        output = model(images, priors=priors, reliability=None, return_features=True)
        reliable_output = model(images, priors=priors, reliability=reliability, return_features=False)
        pooled = output.features.mean(dim=(-2, -1)) if output.features is not None else torch.empty(0)

        indices.append(batch["index"].cpu().numpy())
        labels.append(batch["label"].cpu().numpy())
        concept_probs.append(output.concept_probs.detach().cpu().numpy())
        concept_logits.append(output.concept_logits.detach().cpu().numpy())
        class_probs.append(torch.softmax(output.class_logits, dim=-1).detach().cpu().numpy())
        class_probs_with_reliability.append(
            torch.softmax(reliable_output.class_logits, dim=-1).detach().cpu().numpy()
        )
        features.append(pooled.detach().cpu().numpy())
        if bool(batch["has_concepts"].any()):
            any_concepts = True
            concepts.append(batch["concepts"].cpu().numpy())

    model.train(was_training)
    return DebugPredictions(
        indices=np.concatenate(indices),
        labels=np.concatenate(labels),
        concepts=np.concatenate(concepts) if any_concepts else None,
        concept_probs=np.concatenate(concept_probs),
        concept_logits=np.concatenate(concept_logits),
        class_probs=np.concatenate(class_probs),
        class_probs_with_reliability=np.concatenate(class_probs_with_reliability),
        features=np.concatenate(features),
    )


def write_prior_table(
    output_dir: Path,
    priors: PriorBundle,
    reliability: torch.Tensor,
    effective_prior: torch.Tensor,
    debug: DebugPredictions,
    concept_names: list[str],
    class_names: list[str],
    threshold: float,
) -> None:
    pred_means = split_class_means(debug.concept_probs, debug.labels, len(class_names))
    true_means = None if debug.concepts is None else split_class_means(debug.concepts, debug.labels, len(class_names))
    concept_f1_by_class = per_class_concept_f1(debug, len(class_names), threshold)

    observed = priors.observed.numpy()
    clean = None if priors.clean is None else priors.clean.numpy()
    audit = None if priors.audit is None else priors.audit.numpy()
    corrupt = None if priors.corruption_mask is None else priors.corruption_mask.numpy()
    rows: list[dict[str, Any]] = []
    for m, concept in enumerate(concept_names):
        rho = float(observed[m].mean())
        for y, class_name in enumerate(class_names):
            clean_value = np.nan if clean is None else float(clean[m, y])
            observed_value = float(observed[m, y])
            effective_value = float(effective_prior[m, y])
            predicted_value = float(pred_means[m, y])
            rows.append(
                {
                    "concept": concept,
                    "class": class_name,
                    "m": m,
                    "y": y,
                    "prior_observed": observed_value,
                    "prior_clean": clean_value,
                    "prior_audit": np.nan if audit is None else float(audit[m, y]),
                    "prior_background_rho": rho,
                    "prior_effective": effective_value,
                    "reliability": float(reliability[m, y]),
                    "is_corrupted": np.nan if corrupt is None else bool(corrupt[m, y]),
                    "prior_error_abs": (
                        np.nan if clean is None else abs(observed_value - clean_value)
                    ),
                    "predicted_class_mean": predicted_value,
                    "true_class_mean": (
                        np.nan if true_means is None else float(true_means[m, y])
                    ),
                    "residual_observed_abs": abs(predicted_value - observed_value),
                    "residual_effective_abs": abs(predicted_value - effective_value),
                    "per_class_concept_f1": concept_f1_by_class[m, y],
                }
            )
    pd.DataFrame(rows).to_csv(output_dir / "prior_reliability_entries.csv", index=False)


def write_concept_tables(
    output_dir: Path,
    debug: DebugPredictions,
    concept_names: list[str],
    class_names: list[str],
    threshold: float,
) -> None:
    if debug.concepts is None:
        return
    metrics = concept_metrics(
        debug.concept_probs,
        debug.concepts,
        labels=debug.labels,
        n_classes=len(class_names),
        threshold=threshold,
        concept_names=concept_names,
    )
    rows: list[dict[str, Any]] = []
    for m, concept in enumerate(concept_names):
        y_true = debug.concepts[:, m] > 0.5
        y_prob = debug.concept_probs[:, m]
        y_pred = y_prob >= threshold
        rows.append(
            {
                "concept": concept,
                "m": m,
                "f1": float(metrics.per_concept_f1[m]),
                "auroc": float(metrics.per_concept_auroc[m]),
                "true_prevalence": float(y_true.mean()),
                "predicted_prevalence": float(y_pred.mean()),
                "mean_probability": float(y_prob.mean()),
                "mean_probability_positive": (
                    float(y_prob[y_true].mean()) if y_true.any() else np.nan
                ),
                "mean_probability_negative": (
                    float(y_prob[~y_true].mean()) if (~y_true).any() else np.nan
                ),
                "brier": brier_score(y_prob, y_true.astype(float)),
            }
        )
    pd.DataFrame(rows).to_csv(output_dir / "concept_summary.csv", index=False)


def write_sample_tables(
    output_dir: Path,
    debug: DebugPredictions,
    concept_names: list[str],
    class_names: list[str],
    threshold: float,
    *,
    max_samples: int | None,
) -> None:
    class_pred = debug.class_probs.argmax(axis=1)
    class_conf = debug.class_probs.max(axis=1)
    reliable_probs = debug.class_probs_with_reliability
    reliable_pred = None if reliable_probs is None else reliable_probs.argmax(axis=1)
    reliable_conf = None if reliable_probs is None else reliable_probs.max(axis=1)

    rows: list[dict[str, Any]] = []
    n = len(debug.labels) if max_samples is None else min(max_samples, len(debug.labels))
    for i in range(n):
        row: dict[str, Any] = {
            "index": int(debug.indices[i]),
            "true_class": class_names[int(debug.labels[i])],
            "pred_class": class_names[int(class_pred[i])],
            "class_confidence": float(class_conf[i]),
            "class_correct": bool(class_pred[i] == debug.labels[i]),
        }
        for y, class_name in enumerate(class_names):
            row[f"class_prob/{class_name}"] = float(debug.class_probs[i, y])
            if reliable_probs is not None:
                row[f"class_prob_with_reliability/{class_name}"] = float(reliable_probs[i, y])
        if reliable_pred is not None and reliable_conf is not None:
            row["pred_class_with_reliability"] = class_names[int(reliable_pred[i])]
            row["class_confidence_with_reliability"] = float(reliable_conf[i])
            row["class_prediction_changed_by_reliability"] = bool(reliable_pred[i] != class_pred[i])
        for m, concept in enumerate(concept_names):
            prob = float(debug.concept_probs[i, m])
            pred = prob >= threshold
            row[f"concept_prob/{concept}"] = prob
            row[f"concept_logit/{concept}"] = float(debug.concept_logits[i, m])
            row[f"concept_pred/{concept}"] = bool(pred)
            if debug.concepts is not None:
                true = bool(debug.concepts[i, m] > 0.5)
                row[f"concept_true/{concept}"] = true
                row[f"concept_correct/{concept}"] = bool(pred == true)
        rows.append(row)
    pd.DataFrame(rows).to_csv(output_dir / "sample_predictions.csv", index=False)


def write_feature_tables(
    output_dir: Path,
    debug: DebugPredictions,
    concept_names: list[str],
    class_names: list[str],
) -> dict[str, float]:
    features = l2_normalize(debug.features)
    class_rows: list[dict[str, Any]] = []
    class_centroids = []
    for y, class_name in enumerate(class_names):
        mask = debug.labels == y
        centroid = features[mask].mean(axis=0) if mask.any() else np.full(features.shape[1], np.nan)
        class_centroids.append(centroid)
        class_rows.append(
            {
                "class": class_name,
                "y": y,
                "n": int(mask.sum()),
                "mean_feature_norm": float(np.linalg.norm(debug.features[mask], axis=1).mean()) if mask.any() else np.nan,
            }
        )
    class_centroids_arr = l2_normalize(np.vstack(class_centroids))
    centroid_scores = features @ class_centroids_arr.T
    nearest_centroid = centroid_scores.argmax(axis=1)
    nearest_class_accuracy = float(np.mean(nearest_centroid == debug.labels))
    pd.DataFrame(class_rows).to_csv(output_dir / "feature_class_summary.csv", index=False)

    concept_rows: list[dict[str, Any]] = []
    if debug.concepts is not None:
        for m, concept in enumerate(concept_names):
            positive = debug.concepts[:, m] > 0.5
            negative = ~positive
            row: dict[str, Any] = {
                "concept": concept,
                "m": m,
                "n_positive": int(positive.sum()),
                "n_negative": int(negative.sum()),
            }
            if positive.any() and negative.any():
                pos_centroid = l2_normalize(features[positive].mean(axis=0, keepdims=True))[0]
                neg_centroid = l2_normalize(features[negative].mean(axis=0, keepdims=True))[0]
                score = features @ pos_centroid - features @ neg_centroid
                row.update(
                    {
                        "centroid_cosine_gap": float(pos_centroid @ neg_centroid),
                        "feature_centroid_auroc": roc_auc(positive.astype(float), score),
                    }
                )
            else:
                row.update({"centroid_cosine_gap": np.nan, "feature_centroid_auroc": np.nan})
            concept_rows.append(row)
    pd.DataFrame(concept_rows).to_csv(output_dir / "feature_concept_summary.csv", index=False)
    return {"nearest_class_centroid_accuracy": nearest_class_accuracy}


def build_summary(
    *,
    config: ExperimentConfig,
    priors: PriorBundle,
    reliability: torch.Tensor,
    effective_prior: torch.Tensor,
    debug: DebugPredictions,
    concept_names: list[str],
    class_names: list[str],
    threshold: float,
    feature_summary: dict[str, float],
) -> dict[str, Any]:
    class_no_r = class_metrics(debug.class_probs, debug.labels, class_names=class_names)
    class_with_r = (
        None
        if debug.class_probs_with_reliability is None
        else class_metrics(debug.class_probs_with_reliability, debug.labels, class_names=class_names)
    )
    concept_result = (
        None
        if debug.concepts is None
        else concept_metrics(
            debug.concept_probs,
            debug.concepts,
            labels=debug.labels,
            n_classes=len(class_names),
            threshold=threshold,
            concept_names=concept_names,
        )
    )
    rel = reliability_metrics(
        reliability,
        corruption_mask=priors.corruption_mask,
        prior_error=priors.prior_error(),
        per_entry_concept_f1=None if concept_result is None else concept_result.per_class_concept_f1,
        threshold=config.eval.concept_threshold,
    )
    return {
        "run": {
            "name": config.name,
            "tag": config.tag,
            "method_like_reliability_mode": config.reliability.mode,
            "class_head": str(config.model.class_head),
            "prior_repair": config.loss.prior_repair,
            "concept_threshold": threshold,
        },
        "class_metrics_no_reliability_at_eval": class_no_r.as_dict("class/"),
        "class_metrics_with_reliability_at_eval": (
            None if class_with_r is None else class_with_r.as_dict("class/")
        ),
        "class_confusion_no_reliability": class_no_r.confusion.tolist(),
        "class_confusion_with_reliability": (
            None if class_with_r is None else class_with_r.confusion.tolist()
        ),
        "concept_metrics": None if concept_result is None else concept_result.as_dict("concept/"),
        "reliability_metrics": rel.as_dict("reliability/"),
        "prior_error_mean_observed": (
            None if priors.prior_error() is None else float(priors.prior_error().mean())
        ),
        "prior_error_mean_effective": (
            None
            if priors.clean is None
            else float((effective_prior - priors.clean).abs().mean())
        ),
        "feature_summary": feature_summary,
    }


def split_class_means(values: np.ndarray, labels: np.ndarray, n_classes: int) -> np.ndarray:
    means = np.full((values.shape[1], n_classes), np.nan)
    for y in range(n_classes):
        mask = labels == y
        if mask.any():
            means[:, y] = values[mask].mean(axis=0)
    return means


def per_class_concept_f1(
    debug: DebugPredictions,
    n_classes: int,
    threshold: float,
) -> np.ndarray:
    n_concepts = debug.concept_probs.shape[1]
    out = np.full((n_concepts, n_classes), np.nan)
    if debug.concepts is None:
        return out
    predictions = debug.concept_probs >= threshold
    for y in range(n_classes):
        mask = debug.labels == y
        if not mask.any():
            continue
        for m in range(n_concepts):
            out[m, y] = binary_f1(debug.concepts[mask, m], predictions[mask, m])
    return out


def l2_normalize(values: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, eps)


if __name__ == "__main__":
    raise SystemExit(main())
