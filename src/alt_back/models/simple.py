from __future__ import annotations

from typing import Sequence

import torch
from torch import nn


class SimpleFashionCNN(nn.Module):
    """A lightweight CNN baseline; keeps structure configurable for experimentation."""

    def __init__(
        self,
        in_channels: int = 1,
        conv_channels: Sequence[int] | None = None,
        kernel_size: int = 3,
        fc_hidden: int = 128,
        num_classes: int = 10,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        conv_channels = list(conv_channels or [32, 64])

        layers: list[nn.Module] = []
        input_channels = in_channels
        for idx, out_channels in enumerate(conv_channels):
            layers.append(nn.Conv2d(input_channels, out_channels, kernel_size=kernel_size, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            if idx < len(conv_channels) - 1:
                layers.append(nn.MaxPool2d(2))
            input_channels = out_channels

        self.features = nn.Sequential(*layers)

        example_input = torch.zeros(1, in_channels, 28, 28)
        with torch.no_grad():
            feature_dim = self.features(example_input).view(1, -1).shape[1]

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, fc_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)
