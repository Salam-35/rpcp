"""Minimal tensor image transforms (no torchvision dependency required)."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F

__all__ = [
    "Compose",
    "Normalize",
    "RandomFlip",
    "RandomResizedCropLike",
    "Resize",
    "build_transform",
]

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

Transform = Callable[[torch.Tensor], torch.Tensor]


@dataclass(slots=True)
class Compose:
    transforms: Sequence[Transform]

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        for transform in self.transforms:
            image = transform(image)
        return image


@dataclass(slots=True)
class Resize:
    size: int
    mode: str = "bilinear"

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        if image.shape[-2:] == (self.size, self.size):
            return image
        return F.interpolate(
            image.unsqueeze(0),
            size=(self.size, self.size),
            mode=self.mode,
            align_corners=False if self.mode in {"bilinear", "bicubic"} else None,
        ).squeeze(0)


@dataclass(slots=True)
class RandomFlip:
    p_horizontal: float = 0.5
    p_vertical: float = 0.0

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        if torch.rand(()) < self.p_horizontal:
            image = torch.flip(image, dims=(-1,))
        if torch.rand(()) < self.p_vertical:
            image = torch.flip(image, dims=(-2,))
        return image


@dataclass(slots=True)
class RandomResizedCropLike:
    """Random crop of a random scale, resized back to ``size``."""

    size: int
    min_scale: float = 0.8

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        _, height, width = image.shape
        scale = float(torch.empty(()).uniform_(self.min_scale, 1.0))
        crop_h, crop_w = max(1, int(height * scale)), max(1, int(width * scale))
        top = int(torch.randint(0, height - crop_h + 1, ()))
        left = int(torch.randint(0, width - crop_w + 1, ()))
        cropped = image[:, top : top + crop_h, left : left + crop_w]
        return Resize(self.size)(cropped)


@dataclass(slots=True)
class Normalize:
    mean: Sequence[float] = IMAGENET_MEAN
    std: Sequence[float] = IMAGENET_STD

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor(self.mean, dtype=image.dtype, device=image.device).view(-1, 1, 1)
        std = torch.tensor(self.std, dtype=image.dtype, device=image.device).view(-1, 1, 1)
        if mean.shape[0] != image.shape[0]:  # e.g. grayscale
            mean, std = mean.mean(0, keepdim=True), std.mean(0, keepdim=True)
        return (image - mean) / std


def build_transform(
    image_size: int,
    *,
    train: bool,
    augment: bool = True,
    normalize: bool = True,
) -> Compose:
    """Standard train/eval pipelines used by every real dataset loader."""
    stages: list[Transform] = [Resize(image_size)]
    if train and augment:
        stages = [RandomResizedCropLike(image_size), RandomFlip(0.5, 0.5)]
    if normalize:
        stages.append(Normalize())
    return Compose(stages)
