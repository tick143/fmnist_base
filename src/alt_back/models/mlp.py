from __future__ import annotations

from typing import Sequence

import torch
from torch import nn


class FashionMLP(nn.Module):
    """Configurable multilayer perceptron for Fashion-MNIST."""

    def __init__(
        self,
        in_features: int = 28 * 28,
        hidden_layers: Sequence[int] | None = None,
        dropout: float = 0.1,
        num_classes: int = 10,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        hidden_layers = list(hidden_layers or [512, 256])
        activation_cls = getattr(nn, activation.capitalize(), nn.ReLU)

        layers: list[nn.Module] = []
        prev_features = in_features
        for idx, hidden in enumerate(hidden_layers):
            layers.append(nn.Linear(prev_features, hidden))
            layers.append(activation_cls())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_features = hidden

        layers.append(nn.Linear(prev_features, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        return self.network(x)

