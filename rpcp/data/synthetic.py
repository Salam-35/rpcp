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
    """Draw a well-separated ground-truth prior table ``Pi_star``."""
    rng = np.random.default_rng(config.seed)
    high = 0.5 + 0.5 * config.prior_sharpness
    low = 1.0 - high

    bits = rng.integers(0, 2, size=(config.n_concepts, config.n_classes))
    # Force distinct class signatures: if two columns coincide, flip one bit.
    for y in range(1, config.n_classes):
        for y_prev in range(y):
            if np.array_equal(bits[:, y], bits[:, y_prev]):
                bits[y % config.n_concepts, y] ^= 1
    prior = np.where(bits == 1, high, low).astype(np.float32)

    return SyntheticWorld(
        prior=torch.from_numpy(prior),
        concept_names=[f"concept_{m:02d}" for m in range(config.n_concepts)],
        class_names=[f"class_{k}" for k in range(config.n_classes)],
        config=config,
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
