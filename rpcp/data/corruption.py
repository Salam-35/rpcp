"""Controlled prior corruption (plan 6.3).

All modes share one contract::

    Pi_noisy[m, y] = (1 - alpha) * Pi_true[m, y] + alpha * target[m, y]   if (m, y) selected
    Pi_noisy[m, y] = Pi_true[m, y]                                        otherwise

where ``target`` is mode-specific and a random ``fraction`` of the ``M x K``
entries is selected.  Writing modes 3 and 4 ("class swap", "adversarial flip")
in the same blended form makes ``alpha`` mean the same thing on every curve of
Figure 2; ``alpha = 1`` reproduces the direct assignment given in the plan.

The ground-truth corruption mask ``s_true`` is returned for evaluation only and
must never be handed to the training objective (except for the deliberate
``oracle`` reliability baseline).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from rpcp.config import CorruptionMode

__all__ = [
    "CorruptionResult",
    "compute_corruption_mask",
    "corrupt_priors",
    "corruption_target",
    "select_entries",
]


@dataclass(slots=True)
class CorruptionResult:
    """Output of :func:`corrupt_priors`.

    Attributes:
        priors: ``(M, K)`` corrupted prior table ``Pi_tilde``.
        clean_mask: ``(M, K)`` boolean, ``True`` where the entry is unchanged
            (``s_true = 1``).
        selected: ``(M, K)`` boolean, ``True`` where corruption was *attempted*.
        target: ``(M, K)`` the corruption target that was blended in.
    """

    priors: torch.Tensor
    clean_mask: torch.Tensor
    selected: torch.Tensor
    target: torch.Tensor

    @property
    def corruption_mask(self) -> torch.Tensor:
        """``True`` where the entry is corrupted (``s_true = 0``)."""
        return ~self.clean_mask

    @property
    def corrupted_fraction(self) -> float:
        return float(self.corruption_mask.float().mean())


def select_entries(
    shape: tuple[int, int],
    fraction: float,
    *,
    generator: torch.Generator,
) -> torch.Tensor:
    """Uniformly select ``round(fraction * M * K)`` entries without replacement."""
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"fraction must be in [0, 1], got {fraction}")
    n_entries = shape[0] * shape[1]
    n_selected = int(round(fraction * n_entries))
    mask = torch.zeros(n_entries, dtype=torch.bool)
    if n_selected > 0:
        idx = torch.randperm(n_entries, generator=generator)[:n_selected]
        mask[idx] = True
    return mask.view(shape)


def _class_swap_permutation(n_classes: int, generator: torch.Generator) -> torch.Tensor:
    """A derangement of the classes (no class maps to itself) when possible."""
    if n_classes < 2:
        return torch.zeros(1, dtype=torch.long)
    for _ in range(100):
        perm = torch.randperm(n_classes, generator=generator)
        if bool((perm != torch.arange(n_classes)).all()):
            return perm
    return torch.roll(torch.arange(n_classes), shifts=1)


def corruption_target(
    priors: torch.Tensor,
    mode: CorruptionMode | str,
    *,
    generator: torch.Generator,
    llm_bias_strength: float = 0.6,
) -> torch.Tensor:
    """Mode-specific corruption target (the value blended in with weight ``alpha``).

    * ``uniform``             -- ``noise[m, y] ~ U(0, 1)``
    * ``background_collapse`` -- ``noise[m, y] = mean_y Pi_true[m, y]`` (class signal removed)
    * ``class_swap``          -- ``Pi_true[m, y']`` for a fixed derangement ``y -> y'``
    * ``adversarial_flip``    -- ``1 - Pi_true[m, y]``
    * ``llm_bias``            -- a class-independent "textbook" value: concepts that are
      on average present become generically present, the rest generically absent.
      This mimics an LLM that knows the disease vocabulary but not the cohort prevalence.
    """
    mode = CorruptionMode(mode)
    match mode:
        case CorruptionMode.NONE:
            return priors.clone()
        case CorruptionMode.UNIFORM:
            return torch.rand(priors.shape, generator=generator, dtype=priors.dtype)
        case CorruptionMode.BACKGROUND_COLLAPSE:
            return priors.mean(dim=1, keepdim=True).expand_as(priors).clone()
        case CorruptionMode.CLASS_SWAP:
            perm = _class_swap_permutation(priors.shape[1], generator)
            return priors[:, perm].clone()
        case CorruptionMode.ADVERSARIAL_FLIP:
            return 1.0 - priors
        case CorruptionMode.LLM_BIAS:
            marginal = priors.mean(dim=1, keepdim=True)
            polarity = torch.where(marginal >= 0.5, 1.0, -1.0)
            generic = 0.5 + polarity * 0.5 * float(llm_bias_strength)
            return generic.expand_as(priors).clone()
        case _:  # pragma: no cover - exhaustive over the enum
            raise ValueError(f"Unknown corruption mode: {mode}")


def corrupt_priors(
    priors: torch.Tensor,
    mode: CorruptionMode | str = CorruptionMode.NONE,
    alpha: float = 0.0,
    fraction: float = 1.0,
    *,
    seed: int = 0,
    llm_bias_strength: float = 0.6,
    tolerance: float = 1e-3,
    clip: tuple[float, float] = (1e-3, 1 - 1e-3),
) -> CorruptionResult:
    """Corrupt a prior table and return the ground-truth corruption mask.

    Args:
        priors: ``(M, K)`` clean prior table.
        mode: One of :class:`~rpcp.config.CorruptionMode`.
        alpha: Corruption strength in ``[0, 1]``.
        fraction: Fraction of ``(m, y)`` entries eligible for corruption.
        seed: RNG seed (entry selection, uniform noise, class permutation).
        llm_bias_strength: Only used by ``llm_bias``.
        tolerance: An entry counts as corrupted when it moved by more than this.
        clip: Final clipping range so log-domain losses stay finite.

    Returns:
        :class:`CorruptionResult`.
    """
    if priors.ndim != 2:
        raise ValueError(f"priors must be (M, K), got {tuple(priors.shape)}")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")

    mode = CorruptionMode(mode)
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    priors = priors.detach().cpu().float()

    if mode is CorruptionMode.NONE or alpha == 0.0:
        clean_mask = torch.ones_like(priors, dtype=torch.bool)
        return CorruptionResult(
            priors=priors.clone(),
            clean_mask=clean_mask,
            selected=torch.zeros_like(clean_mask),
            target=priors.clone(),
        )

    selected = select_entries(tuple(priors.shape), fraction, generator=generator)  # type: ignore[arg-type]
    target = corruption_target(
        priors, mode, generator=generator, llm_bias_strength=llm_bias_strength
    )
    blended = (1.0 - alpha) * priors + alpha * target
    noisy = torch.where(selected, blended, priors).clamp(*clip)

    return CorruptionResult(
        priors=noisy,
        clean_mask=compute_corruption_mask(priors, noisy, tolerance=tolerance),
        selected=selected,
        target=target,
    )


def compute_corruption_mask(
    clean_priors: torch.Tensor,
    noisy_priors: torch.Tensor,
    *,
    tolerance: float = 1e-3,
) -> torch.Tensor:
    """``s_true``: ``True`` where the observed prior still matches the clean one.

    Entries that were *selected* for corruption but happen to be numerically
    unchanged (e.g. a class-swap between two classes with identical prevalence)
    are counted as clean -- reliability estimation cannot and should not be
    penalised for trusting them.
    """
    if clean_priors.shape != noisy_priors.shape:
        raise ValueError("clean and noisy prior tables must have the same shape")
    return (clean_priors - noisy_priors).abs() <= tolerance
