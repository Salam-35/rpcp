"""Tests for the optional PH2 attention-mask supervision loss."""

from __future__ import annotations

import torch

from rpcp.losses.attention_supervision import attention_mask_loss


def test_perfect_overlap_gives_near_zero_loss() -> None:
    attention = torch.zeros(1, 1, 4, 4)
    attention[0, 0, :2, :2] = 1.0
    masks = attention.clone()
    valid = torch.ones(1, 1)
    loss = attention_mask_loss(attention, masks, valid)
    assert loss.item() < 1e-5


def test_disjoint_regions_give_loss_near_one() -> None:
    attention = torch.zeros(1, 1, 4, 4)
    attention[0, 0, :2, :2] = 1.0
    masks = torch.zeros(1, 1, 4, 4)
    masks[0, 0, 2:, 2:] = 1.0
    valid = torch.ones(1, 1)
    loss = attention_mask_loss(attention, masks, valid)
    assert loss.item() > 0.9


def test_invalid_entries_are_excluded() -> None:
    # One valid pair with perfect overlap, one invalid pair with total
    # mismatch -- if the invalid entry leaked in, the loss would not be ~0.
    attention = torch.zeros(1, 2, 4, 4)
    attention[0, 0, :2, :2] = 1.0
    attention[0, 1, :2, :2] = 1.0
    masks = torch.zeros(1, 2, 4, 4)
    masks[0, 0, :2, :2] = 1.0  # matches
    masks[0, 1, 2:, 2:] = 1.0  # disjoint, but excluded via `valid`
    valid = torch.tensor([[1.0, 0.0]])
    loss = attention_mask_loss(attention, masks, valid)
    assert loss.item() < 1e-5


def test_shape_mismatch_raises() -> None:
    import pytest

    with pytest.raises(ValueError, match="must match"):
        attention_mask_loss(torch.zeros(1, 1, 4, 4), torch.zeros(1, 1, 3, 3), torch.ones(1, 1))


def test_gradient_flows_to_attention() -> None:
    attention = torch.rand(2, 3, 5, 5, requires_grad=True)
    masks = (torch.rand(2, 3, 5, 5) > 0.5).float()
    valid = torch.ones(2, 3)
    loss = attention_mask_loss(attention, masks, valid)
    loss.backward()
    assert attention.grad is not None
    assert torch.isfinite(attention.grad).all()
