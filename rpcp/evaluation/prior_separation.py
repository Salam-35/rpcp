"""Prior-signature separation ``Delta`` (plan 6.4 and Proposition 2).

.. math:: \\Delta = \\min_{y \\neq y'} \\lVert \\Pi[:, y] - \\Pi[:, y'] \\rVert_2

If two classes share a prior column, prior matching cannot distinguish them
through concepts: any concept assignment consistent with one class is equally
consistent with the other.  ``Delta`` quantifies how far a prior table is from
that degenerate case, and is the x-axis of Figure 4.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

__all__ = [
    "SeparationReport",
    "delta_sweep",
    "nearest_class_pair",
    "pairwise_prior_distances",
    "prior_separation_delta",
    "separation_report",
]


def pairwise_prior_distances(prior: torch.Tensor, *, p: float = 2.0) -> torch.Tensor:
    """``(K, K)`` matrix of distances between class prior columns."""
    if prior.ndim != 2:
        raise ValueError(f"prior must be (M, K), got {tuple(prior.shape)}")
    columns = prior.t()  # (K, M)
    return torch.cdist(columns.unsqueeze(0), columns.unsqueeze(0), p=p).squeeze(0)


def prior_separation_delta(prior: torch.Tensor, *, p: float = 2.0) -> float:
    """Minimum distance between any two distinct class prior signatures."""
    distances = pairwise_prior_distances(prior, p=p)
    n_classes = distances.shape[0]
    if n_classes < 2:
        return float("inf")
    mask = ~torch.eye(n_classes, dtype=torch.bool, device=distances.device)
    return float(distances[mask].min())


def nearest_class_pair(prior: torch.Tensor, *, p: float = 2.0) -> tuple[int, int]:
    """Indices of the two closest class signatures."""
    distances = pairwise_prior_distances(prior, p=p)
    n_classes = distances.shape[0]
    distances = distances + torch.eye(n_classes, device=distances.device) * float("inf")
    flat = int(distances.argmin())
    return divmod(flat, n_classes)


@dataclass(slots=True)
class SeparationReport:
    """Summary of how identifiable a prior table is."""

    delta: float
    mean_distance: float
    nearest_pair: tuple[int, int]
    distances: np.ndarray
    per_concept_range: np.ndarray = field(default_factory=lambda: np.empty(0))

    def as_dict(self, prefix: str = "prior/") -> dict[str, float]:
        return {
            f"{prefix}delta": self.delta,
            f"{prefix}mean_class_distance": self.mean_distance,
            f"{prefix}nearest_pair_0": float(self.nearest_pair[0]),
            f"{prefix}nearest_pair_1": float(self.nearest_pair[1]),
            f"{prefix}mean_concept_range": float(np.mean(self.per_concept_range))
            if self.per_concept_range.size
            else float("nan"),
        }


def separation_report(prior: torch.Tensor, *, p: float = 2.0) -> SeparationReport:
    """Delta, mean pairwise distance, closest pair and per-concept dynamic range.

    ``per_concept_range[m] = max_y Pi[m, y] - min_y Pi[m, y]`` tells you which
    concepts carry class information at all: a concept with range ~0 is
    unidentifiable from class-level priors no matter how good the model is.
    """
    distances = pairwise_prior_distances(prior, p=p)
    n_classes = distances.shape[0]
    off_diagonal = (
        distances[~torch.eye(n_classes, dtype=torch.bool, device=distances.device)]
        if n_classes > 1
        else torch.tensor([float("inf")])
    )
    concept_range = (prior.max(dim=1).values - prior.min(dim=1).values).detach().cpu().numpy()
    return SeparationReport(
        delta=float(off_diagonal.min()),
        mean_distance=float(off_diagonal.mean()),
        nearest_pair=nearest_class_pair(prior, p=p) if n_classes > 1 else (0, 0),
        distances=distances.detach().cpu().numpy(),
        per_concept_range=concept_range,
    )


def delta_sweep(
    prior: torch.Tensor,
    alphas: list[float],
    *,
    source: int = 0,
    target: int = 1,
) -> list[tuple[float, float, torch.Tensor]]:
    """Blend two class signatures together and report the resulting ``Delta``.

    Returns:
        List of ``(alpha, delta, blended_prior)`` triples, ordered as ``alphas``.
    """
    from rpcp.data.priors import blend_prior_columns

    out: list[tuple[float, float, torch.Tensor]] = []
    for alpha in alphas:
        blended = blend_prior_columns(prior, alpha, source=source, target=target)
        out.append((alpha, prior_separation_delta(blended), blended))
    return out
