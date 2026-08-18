"""Named methods and ablations (plan 6.5 and section 8).

Every entry is a set of dotted config overrides applied on top of a dataset
config, so all baselines share one training implementation and differ only in
configuration -- there is no separate "PCP code path" that can silently drift.
"""

from __future__ import annotations

from typing import Any

from rpcp.config import ExperimentConfig

__all__ = ["ABLATIONS", "METHODS", "apply_method", "method_names"]

#: Main baselines of plan 6.5.
METHODS: dict[str, dict[str, Any]] = {
    # 1. Black-box classifier: class labels only, concept layer unsupervised.
    "blackbox": {
        "loss.lambda_match": 0.0,
        "loss.lambda_prior": 0.0,
        "loss.lambda_ent": 0.0,
        "loss.lambda_concept": 0.0,
        "reliability.mode": "none",
        "model.class_head": "linear",
    },
    # 2. Supervised CBM: per-image concept labels (upper bound for concept quality).
    "supervised-cbm": {
        "loss.lambda_concept": 1.0,
        "loss.lambda_match": 0.0,
        "loss.lambda_prior": 0.0,
        "reliability.mode": "none",
        "model.class_head": "linear",
    },
    # 3. PCP: class-level priors, no reliability weighting (r == 1).
    "pcp": {
        "reliability.mode": "none",
        "loss.lambda_concept": 0.0,
    },
    # 4. R-PCP, unsupervised reliability (held-out residual + instability).
    "r-pcp": {
        "reliability.mode": "unsupervised",
        "reliability.use_crossfit": True,
        "loss.lambda_concept": 0.0,
    },
    # 5. R-PCP + multi-source priors (Evidence Mode B).
    "r-pcp-multisource": {
        "reliability.mode": "multi_source",
        "priors.n_synthetic_sources": 5,
        "loss.lambda_concept": 0.0,
    },
    # 6. Recommended audited R-PCP: audit reliability + prior repair +
    # image-level audit concept anchors.
    #
    # Two things this preset used to get wrong (both fixed here):
    #
    # - `audit_concept_weighting: "inverse_reliability"` weighted the audit
    #   anchor -- BCE against real per-image concept labels, the *only* loss
    #   term in the whole objective that is a joint function of (image,
    #   concept) -- by `1 - r`.  Per-image labels are equally correct
    #   regardless of how much the class-level prior is trusted, so this
    #   suppressed the anchor by up to ~38x on exactly the concepts whose
    #   prior looked reliable, leaving most concepts never anchored at all.
    #   Default is now "none"; the old behaviour survives as ablation
    #   `11-inverse-reliability-anchor` below.
    # - `lambda_match: 0.0` + `class_head: "linear"` removed the only other
    #   route from the concept layer to the prior table: with matching off
    #   and a free `Linear(M, K)` head, the model can reach any class
    #   accuracy through an arbitrary reparameterisation of the concept
    #   space, and Proposition 2 (two classes with identical prior columns
    #   get identical logits) is never actually exercised. `class_head:
    #   "prior"` is parameter-free, so classification is forced entirely
    #   through the prior-similarity path, and `lambda_match` is left at the
    #   dataset default instead of zeroed.
    "r-pcp-audit": {
        "reliability.mode": "audit",
        "data.audit_fraction": 0.1,
        "loss.lambda_concept": 0.0,
        "loss.lambda_audit_concept": 1.0,
        "loss.audit_concept_weighting": "none",
        "loss.prior_repair": "audit",
        "loss.match_reliability_weighted": True,
        "model.class_head": "prior",
    },
    # Plain audit calibration only. Kept as an ablation because it tends to know
    # which priors are wrong without translating that into better concept/class learning.
    "r-pcp-audit-plain": {
        "reliability.mode": "audit",
        "data.audit_fraction": 0.1,
        "loss.lambda_concept": 0.0,
    },
    # Backwards-compatible explicit name for the anchored audited variant.
    "r-pcp-audit-anchor": {
        "reliability.mode": "audit",
        "data.audit_fraction": 0.1,
        "loss.lambda_concept": 0.0,
        "loss.lambda_audit_concept": 1.0,
        "loss.audit_concept_weighting": "none",
        "loss.prior_repair": "audit",
        "loss.match_reliability_weighted": True,
        "model.class_head": "prior",
    },
    # 7. Oracle reliability: uses the true corruption mask (upper bound).
    "oracle": {
        "reliability.mode": "oracle",
        "loss.lambda_concept": 0.0,
    },
    # Oracle reliability plus the same audit anchor, for the upper-bound variant.
    "oracle-anchor": {
        "reliability.mode": "oracle",
        "data.audit_fraction": 0.1,
        "loss.lambda_concept": 0.0,
        "loss.lambda_audit_concept": 1.0,
        "loss.audit_concept_weighting": "none",
        "loss.prior_repair": "audit",
        "loss.match_reliability_weighted": True,
        "model.class_head": "prior",
    },
    # Evidence Mode D, only meaningful on multi-rater datasets (LIDC-IDRI).
    "r-pcp-multirater": {
        "reliability.mode": "multi_rater",
        "loss.lambda_concept": 0.0,
    },
}

#: Ablations of plan section 8, expressed relative to ``r-pcp``.
ABLATIONS: dict[str, dict[str, Any]] = {
    "1-pcp-no-reliability": METHODS["pcp"],
    "2-residual-only": {
        **METHODS["r-pcp"],
        "reliability.w3": 0.0,  # no instability term
        "reliability.use_crossfit": False,  # residual measured on training datasets
    },
    "3-crossfit": {**METHODS["r-pcp"], "reliability.use_crossfit": True},
    "4-source-disagreement-init": METHODS["r-pcp-multisource"],
    "5-audit-calibration": METHODS["r-pcp-audit-plain"],
    "6-pcp-kl-variant": {**METHODS["r-pcp"], "loss.prior_loss": "pcp_kl"},
    "7-no-entropy": {**METHODS["r-pcp"], "loss.lambda_ent": 0.0},
    "8-no-matching": {**METHODS["r-pcp"], "loss.lambda_match": 0.0},
    "9-hard-threshold": {**METHODS["r-pcp"], "reliability.hard_threshold": 0.5},
    "10-continuous-reliability": {**METHODS["r-pcp"], "reliability.hard_threshold": None},
    # Reproduces the pre-fix `r-pcp-audit` behaviour, for a direct before/after
    # comparison: weighting the audit anchor by `1 - r` suppresses the only
    # (image, concept) term in the objective on exactly the entries that look
    # trustworthy, which is the opposite of what an annotation-budget anchor
    # should do.
    "11-inverse-reliability-anchor": {
        **METHODS["r-pcp-audit"],
        "loss.audit_concept_weighting": "inverse_reliability",
    },
    # `L_prior` is satisfiable by predicting the identical probability for
    # every image in a class (zero within-class variance), which drives
    # concept F1 to 0 regardless of how well the class mean is matched --
    # see `rpcp.losses.variance_prior`. This ablation turns on the
    # second-moment term to test whether that collapse is contributing to a
    # low concept macro-F1 independently of the identifiability ceiling.
    "12-variance-anchor": {**METHODS["r-pcp"], "loss.lambda_var": 0.3},
}


def method_names() -> list[str]:
    return list(METHODS)


def apply_method(
    config: ExperimentConfig,
    method: str,
    *,
    extra: dict[str, Any] | None = None,
    tag: str | None = None,
) -> ExperimentConfig:
    """Return a copy of ``config`` configured as ``method``.

    Args:
        config: Base (dataset) configuration.
        method: Key of :datasets:`METHODS` or :datasets:`ABLATIONS`.
        extra: Additional dotted overrides applied after the preset.
        tag: Run tag; defaults to the method name.

    Raises:
        KeyError: If the method is unknown.
    """
    presets = METHODS.get(method) or ABLATIONS.get(method)
    if presets is None:
        raise KeyError(
            f"Unknown method '{method}'. Known: {sorted(set(METHODS) | set(ABLATIONS))}"
        )
    overrides: dict[str, Any] = {**presets, **(extra or {})}
    overrides.setdefault("tag", tag or method)
    return config.replace(**overrides)
