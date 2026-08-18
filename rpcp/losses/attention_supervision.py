"""Optional spatial supervision for concept attention maps (PH2 extension).

``L_prior``/``L_match``/``L_cls`` are all class-level-only, so nothing in the
default R-PCP objective tells the model *where* in the image a concept lives
-- only that its overall probability should be near a class-level prior. PH2
ships free per-colour ROI masks (``<colour>__mask`` manifest columns, recorded
by ``rpcp.data.ph2.prepare_manifest`` / loaded by ``rpcp.data.ph2.load_colour_mask``);
when available they give real, per-image, spatial ground truth for exactly
the concepts (colour attributes) whose location the model would otherwise
have to guess.

This module is intentionally a standalone, opt-in utility rather than a
change to the core batch/trainer contract: wiring it end to end additionally
requires (a) the PH2 dataloader to yield a resized mask tensor + validity
mask per batch, which only PH2 (of the four datasets here) can provide, and
(b) the model's concept attention-map resolution, which varies by backbone
and input size. Both are dataset/model-specific enough that folding them into
``ConceptBatch``/``CompositeObjective`` unconditionally would touch every
dataset's batch contract to serve one of them. Call :func:`attention_mask_loss`
directly from a PH2-specific training loop or extend ``ConceptBatch`` with an
optional ``masks``/``mask_valid`` pair to wire it into the main trainer.
"""

from __future__ import annotations

import torch

__all__ = ["attention_mask_loss"]


def attention_mask_loss(
    attention: torch.Tensor,
    masks: torch.Tensor,
    valid: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Soft-Dice loss between concept attention maps and ground-truth ROI masks.

    Args:
        attention: ``(B, M, H, W)`` non-negative per-concept spatial weights
            (need not sum to 1 over ``H, W``; a soft-Dice loss is invariant to
            an overall positive rescaling of ``attention`` per ``(b, m)``, so
            it does not matter whether the model's attention is a softmax, a
            sigmoid, or a raw non-negative activation).
        masks: ``(B, M, H, W)`` binary ROI targets, already resized to the
            attention resolution (see ``rpcp.data.ph2.load_colour_mask``'s
            ``size`` argument -- resize with nearest-neighbour to keep them
            binary).
        valid: ``(B, M)`` boolean/float mask, ``True``/``1`` where
            ``masks[b, m]`` is a real annotation. PH2 does not ship negative
            masks, so a missing annotation must be excluded here, not treated
            as an implicit all-zero target.
        eps: Numerical floor (also acts as Dice's Laplace smoothing).

    Returns:
        Scalar loss: ``1 - Dice`` averaged over valid ``(b, m)`` pairs, in
        ``[0, 2]`` (0 = perfect overlap, 1 = chance, up to 2 for a
        maximally-anticorrelated arrangement).
    """
    if attention.shape != masks.shape:
        raise ValueError(
            f"attention {tuple(attention.shape)} must match masks {tuple(masks.shape)}"
        )
    batch, n_concepts = attention.shape[:2]
    flat_attention = attention.reshape(batch, n_concepts, -1).clamp_min(0.0)
    flat_masks = masks.reshape(batch, n_concepts, -1).float()

    intersection = (flat_attention * flat_masks).sum(dim=-1)
    denominator = flat_attention.sum(dim=-1) + flat_masks.sum(dim=-1)
    dice = (2.0 * intersection + eps) / (denominator + eps)
    per_pair_loss = 1.0 - dice  # (B, M)

    valid = valid.to(attention.device).float()
    total_valid = valid.sum().clamp_min(eps)
    return (per_pair_loss * valid).sum() / total_valid
