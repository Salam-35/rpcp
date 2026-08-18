"""Synthetic concept-image dataset.

The generator makes the *whole* R-PCP pipeline runnable (and unit-testable)
without downloading medical datasets, while preserving the properties the method
depends on:

* per-image concepts ``c_i`` are sampled from a known class-conditional prior
  ``Pi_star[:, y]`` -- so the clean prior table is known exactly;
* each concept is rendered as a visible patch, so concepts are *learnable from
  pixels* (otherwise no method could recover them);
* an optional class-only shortcut (a global tint) lets the model reach high
  class accuracy while getting concepts wrong -- the failure mode tracked in
  plan section 13, Risk 5.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from rpcp.config import SyntheticConfig
from rpcp.data.base import ConceptDataset

__all__ = ["SyntheticConceptDataset", "SyntheticWorld", "make_synthetic_world"]

_PALETTE = np.array(
    [
        [1.0, 0.2, 0.2],
        [0.2, 1.0, 0.2],
        [0.2, 0.4, 1.0],
        [1.0, 0.9, 0.2],
        [0.9, 0.3, 0.9],
        [0.2, 0.9, 0.9],
        [1.0, 0.6, 0.1],
        [0.6, 0.6, 1.0],
    ],
    dtype=np.float32,
)


@dataclass(slots=True)
class SyntheticWorld:
    """The ground-truth generative parameters of a synthetic benchmark."""

    prior: torch.Tensor  # (M, K) true class-conditional concept prevalence
    concept_names: list[str]
    class_names: list[str]
    config: SyntheticConfig


def make_synthetic_world(config: SyntheticConfig) -> SyntheticWorld:
    """Draw a well-separated, per-concept identifiable ground-truth prior table.

    Two conditions must both hold, and drawing ``bits`` uniformly at random only
    guarantees the first:

    * class signatures are pairwise distinct (``Delta > 0``), so no two classes
      are confusable through the prior;
    * every concept's row varies across classes (``per_concept_range[m] > 0``),
      so every concept actually carries class-level information.

    A row that is constant (e.g. all-1) makes ``Pi[m, :]`` independent of the
    class, and both ``L_prior`` and ``L_match`` are then independent of
    concept ``m``'s per-image value -- no prior-supervised method can recover
    it and its AUROC is chance *by construction*, regardless of training.  The
    original column-only repair left this uncaught: at the shipped defaults
    (``M=8, K=3, seed=0``) 5 of 8 concepts were constant, and in general a
    fraction ``2/2**K`` of concepts are dead in expectation.

    Raises:
        ValueError: If ``n_concepts`` cannot even in principle produce
            ``n_classes`` distinct binary signatures (``2**n_concepts < n_classes``).
        RuntimeError: If repair does not converge (should not happen whenever
            the above precondition holds; kept as a hard guard rather than a
            silent best-effort table).
    """
    from rpcp.evaluation.prior_separation import separation_report

    if 2**config.n_concepts < config.n_classes:
        raise ValueError(
            f"synthetic.n_concepts={config.n_concepts} cannot produce "
            f"{config.n_classes} distinct class signatures (need 2**n_concepts >= "
            "n_classes); increase n_concepts or decrease n_classes."
        )

    rng = np.random.default_rng(config.seed)
    high = 0.5 + 0.5 * config.prior_sharpness
    low = 1.0 - high

    bits = _draw_identifiable_bits(rng, config.n_concepts, config.n_classes)
    prior = np.where(bits == 1, high, low).astype(np.float32)
    prior_tensor = torch.from_numpy(prior)

    # Defensive re-check: cheap, and guards against a future edit to the repair
    # loop silently reintroducing an unidentifiable table.
    report = separation_report(prior_tensor)
    if report.delta <= 0.0 or report.per_concept_range.min() <= 0.0:
        raise RuntimeError(  # pragma: no cover -- _draw_identifiable_bits guarantees this
            "make_synthetic_world produced an unidentifiable prior table "
            f"(delta={report.delta}, min_concept_range={report.per_concept_range.min()})"
        )

    return SyntheticWorld(
        prior=prior_tensor,
        concept_names=[f"concept_{m:02d}" for m in range(config.n_concepts)],
        class_names=[f"class_{k}" for k in range(config.n_classes)],
        config=config,
    )


def _draw_identifiable_bits(
    rng: np.random.Generator,
    n_concepts: int,
    n_classes: int,
    *,
    max_outer_attempts: int = 200,
) -> np.ndarray:
    """Draw an ``(M, K)`` bit matrix with distinct columns and non-constant rows.

    Alternates fixing column collisions (two classes sharing a signature) and
    row collisions (a concept with no class-level signal), flipping a
    RNG-chosen bit each time rather than a fixed index, so the two repairs
    cannot lock into a cycle.  If one draw fails to converge, the whole matrix
    is redrawn fresh (consuming more of ``rng``, still deterministic for a
    given seed) rather than continuing to patch a bad starting point.
    """
    max_inner_rounds = n_concepts + n_classes + 8
    for _ in range(max_outer_attempts):
        bits = rng.integers(0, 2, size=(n_concepts, n_classes))
        for _ in range(max_inner_rounds):
            changed = False
            for y in range(1, n_classes):
                for y_prev in range(y):
                    if np.array_equal(bits[:, y], bits[:, y_prev]):
                        bits[rng.integers(0, n_concepts), y] ^= 1
                        changed = True
            dead = np.flatnonzero(bits.min(axis=1) == bits.max(axis=1))
            for m in dead:
                bits[m, rng.integers(0, n_classes)] ^= 1
                changed = True
            if not changed:
                return bits
        # This draw's basin didn't converge in time; start over from a fresh draw.
    raise RuntimeError(
        f"could not draw an identifiable {n_concepts}x{n_classes} prior table "
        f"after {max_outer_attempts} attempts"
    )


class SyntheticConceptDataset(ConceptDataset):
    """In-memory dataset of concept-rendered images.

    Args:
        world: Ground-truth generative parameters (shared across splits).
        n_samples: Number of images to draw.
        seed: Split-specific RNG seed.
        class_shortcut: Strength of a class-only tint (0 disables the shortcut).
    """

    def __init__(
        self,
        world: SyntheticWorld,
        *,
        n_samples: int,
        seed: int,
        class_shortcut: float = 0.0,
        transform: object | None = None,
    ) -> None:
        cfg = world.config
        rng = np.random.default_rng(seed)

        prior = world.prior.numpy()
        labels = rng.integers(0, cfg.n_classes, size=n_samples)
        concepts = (rng.random((n_samples, cfg.n_concepts)) < prior[:, labels].T).astype(np.float32)

        images = self._render(
            concepts=concepts,
            labels=labels,
            image_size=cfg.image_size,
            visibility=cfg.concept_visibility,
            pixel_noise=cfg.pixel_noise,
            class_shortcut=class_shortcut,
            n_classes=cfg.n_classes,
            rng=rng,
        )

        super().__init__(
            labels=labels,
            concepts=concepts,
            concept_names=world.concept_names,
            class_names=world.class_names,
            transform=transform,  # type: ignore[arg-type]
        )
        self.images = images
        self.world = world

    # ------------------------------------------------------------------ #
    @staticmethod
    def _render(
        *,
        concepts: np.ndarray,
        labels: np.ndarray,
        image_size: int,
        visibility: float,
        pixel_noise: float,
        class_shortcut: float,
        n_classes: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        n_samples, n_concepts = concepts.shape
        grid = math.ceil(math.sqrt(n_concepts))
        cell = max(2, image_size // grid)
        images = rng.normal(0.15, pixel_noise, size=(n_samples, 3, image_size, image_size))

        # Rendering noise: a concept is drawn only `visibility` of the time.
        drawn = concepts * (rng.random(concepts.shape) < visibility)
        for m in range(n_concepts):
            row, col = divmod(m, grid)
            y0, x0 = row * cell, col * cell
            y1, x1 = min(y0 + cell - 1, image_size), min(x0 + cell - 1, image_size)
            if y1 <= y0 or x1 <= x0:
                continue
            colour = _PALETTE[m % len(_PALETTE)].reshape(3, 1, 1)
            active = drawn[:, m].reshape(-1, 1, 1, 1)
            images[:, :, y0:y1, x0:x1] += active * colour

        if class_shortcut > 0.0:
            tint = np.linspace(-1.0, 1.0, n_classes)[labels].reshape(-1, 1, 1, 1)
            images += class_shortcut * tint

        return np.clip(images, 0.0, 1.0).astype(np.float32)

    # ------------------------------------------------------------------ #
    def load_image(self, index: int) -> torch.Tensor:
        return torch.from_numpy(self.images[index])
