"""Deterministic seeding helpers."""

from __future__ import annotations

import os
import random
from collections.abc import Iterator
from contextlib import contextmanager

import numpy as np
import torch

__all__ = ["seed_everything", "temporary_seed", "worker_init_fn"]


def seed_everything(seed: int, *, deterministic: bool = True) -> None:
    """Seed python, numpy and torch RNGs.

    Args:
        seed: Base seed.
        deterministic: If ``True``, request deterministic cuDNN kernels.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


@contextmanager
def temporary_seed(seed: int) -> Iterator[None]:
    """Temporarily seed numpy/torch/random, restoring previous state on exit."""
    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    try:
        seed_everything(seed, deterministic=False)
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.random.set_rng_state(torch_state)


def worker_init_fn(worker_id: int) -> None:
    """Give every dataloader worker a distinct but reproducible seed."""
    base = torch.initial_seed() % 2**31
    np.random.seed(base + worker_id)
    random.seed(base + worker_id)
