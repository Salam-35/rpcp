"""Image backbones returning *spatial* feature maps.

Concept attention needs feature maps, not pooled vectors, so every backbone
here returns ``(B, D, H', W')``.  torchvision is optional: when it is missing
(or ``backbone == "simple_cnn"``) a small built-in CNN is used, which is also
what the synthetic smoke tests run on.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from rpcp.utils.logging import get_logger

__all__ = ["Backbone", "SimpleCNN", "build_backbone"]

logger = get_logger(__name__)


@dataclass(slots=True)
class Backbone:
    """A feature extractor plus its output channel count."""

    module: nn.Module
    feature_dim: int

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        return self.module(images)


class SimpleCNN(nn.Module):
    """Small 4-block CNN used as a dependency-free default backbone."""

    def __init__(self, in_channels: int = 3, widths: tuple[int, ...] = (32, 64, 128, 128)) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        channels = in_channels
        for width in widths:
            layers += [
                nn.Conv2d(channels, width, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(width),
                nn.ReLU(inplace=True),
                nn.Conv2d(width, width, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(width),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ]
            channels = width
        self.features = nn.Sequential(*layers)
        self.out_channels = channels

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.features(images)


def _adapt_first_conv(module: nn.Module, in_channels: int) -> None:
    """Re-shape the stem convolution for non-RGB inputs (e.g. LIDC grayscale)."""
    if in_channels == 3:
        return
    for name, child in module.named_children():
        if isinstance(child, nn.Conv2d):
            new_conv = nn.Conv2d(
                in_channels,
                child.out_channels,
                kernel_size=child.kernel_size,  # type: ignore[arg-type]
                stride=child.stride,  # type: ignore[arg-type]
                padding=child.padding,  # type: ignore[arg-type]
                bias=child.bias is not None,
            )
            with torch.no_grad():
                weight = child.weight.mean(dim=1, keepdim=True).repeat(1, in_channels, 1, 1)
                new_conv.weight.copy_(weight * 3.0 / in_channels)
            setattr(module, name, new_conv)
            return
        _adapt_first_conv(child, in_channels)
        return


def build_backbone(
    name: str = "resnet18",
    *,
    pretrained: bool = False,
    in_channels: int = 3,
) -> Backbone:
    """Instantiate a spatial feature extractor.

    Args:
        name: ``simple_cnn`` or any ``torchvision.models`` ResNet/DenseNet name.
        pretrained: Load ImageNet weights (requires torchvision + network access).
        in_channels: Number of input channels.

    Returns:
        :class:`Backbone` producing ``(B, D, H', W')`` features.
    """
    if name == "simple_cnn":
        module = SimpleCNN(in_channels=in_channels)
        return Backbone(module=module, feature_dim=module.out_channels)

    try:
        from torchvision import models as tv_models
    except ImportError as exc:  # pragma: no cover - exercised only without torchvision
        # Do NOT silently substitute SimpleCNN here: the caller asked for a
        # named torchvision backbone (e.g. "resnet18", possibly pretrained),
        # and config.model.backbone / summary.json would go on reporting that
        # name even though a randomly-initialised, architecturally different
        # network was actually trained. That's a silent correctness bug, not
        # a graceful degradation -- a run's "backbone: resnet18, pretrained:
        # true" must mean what it says. Fail loudly instead; the caller can
        # explicitly opt into the dependency-free default with
        # ``model.backbone: simple_cnn``.
        raise ImportError(
            f"torchvision is required for backbone '{name}' but is not installed. "
            "Install torchvision, or set `model.backbone: simple_cnn` to explicitly "
            "opt into the dependency-free default instead of silently substituting it."
        ) from exc

    if not hasattr(tv_models, name):
        raise ValueError(f"Unknown backbone '{name}'")
    weights = "DEFAULT" if pretrained else None
    network = getattr(tv_models, name)(weights=weights)

    if hasattr(network, "fc"):  # ResNet family
        feature_dim = network.fc.in_features
        network.fc = nn.Identity()
        network.avgpool = nn.Identity()
        stem = nn.Sequential(
            network.conv1,
            network.bn1,
            network.relu,
            network.maxpool,
            network.layer1,
            network.layer2,
            network.layer3,
            network.layer4,
        )
    elif hasattr(network, "features"):  # DenseNet / VGG / EfficientNet family
        stem = network.features
        feature_dim = _infer_feature_dim(stem, in_channels=3)
    else:  # pragma: no cover
        raise ValueError(f"Backbone '{name}' has an unsupported structure")

    _adapt_first_conv(stem, in_channels)
    return Backbone(module=stem, feature_dim=feature_dim)


@torch.no_grad()
def _infer_feature_dim(module: nn.Module, *, in_channels: int, size: int = 64) -> int:
    was_training = module.training
    module.eval()
    output = module(torch.zeros(1, in_channels, size, size))
    module.train(was_training)
    return int(output.shape[1])
