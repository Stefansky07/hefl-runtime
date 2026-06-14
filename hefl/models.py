from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch
from torch import nn


class TinyCNN(nn.Module):
    def __init__(self, image_size: Tuple[int, int, int] = (3, 32, 32), num_classes: int = 10) -> None:
        super().__init__()
        in_channels = int(image_size[0])
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, *image_size)
            flat_features = int(self.features(dummy).reshape(1, -1).shape[1])
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_features, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class TinyMLP(nn.Module):
    def __init__(self, image_size: Tuple[int, int, int] = (3, 32, 32), num_classes: int = 10) -> None:
        super().__init__()
        in_features = int(image_size[0] * image_size[1] * image_size[2])
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def get_model(name: str, num_classes: int = 10, image_size: Tuple[int, int, int] = (3, 32, 32)) -> nn.Module:
    model_name = name.lower()
    if model_name in {"tiny_cnn", "tinycnn", "small_cnn"}:
        return TinyCNN(image_size=image_size, num_classes=num_classes)
    if model_name in {"tiny_mlp", "tinymlp", "mlp"}:
        return TinyMLP(image_size=image_size, num_classes=num_classes)
    if model_name in {"resnet18", "resnet_18"}:
        from torchvision.models import resnet18

        model = resnet18(weights=None, num_classes=num_classes)
        if image_size[0] != 3:
            model.conv1 = nn.Conv2d(image_size[0], 64, kernel_size=7, stride=2, padding=3, bias=False)
        return model
    raise ValueError(f"Unknown model: {name}")


def parameter_count(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def state_parameter_profile(model: nn.Module) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    offset = 0
    for name, tensor in model.state_dict().items():
        numel = int(tensor.numel())
        rows.append(
            {
                "name": name,
                "shape": list(tensor.shape),
                "numel": numel,
                "dtype": str(tensor.dtype),
                "offset": offset,
            }
        )
        offset += numel
    return rows
