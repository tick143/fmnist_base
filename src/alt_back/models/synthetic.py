from __future__ import annotations

from typing import Sequence

import torch
from torch import nn


class SyntheticDenseNetwork(nn.Module):
    """Lightweight feedforward network for the synthetic threshold dataset."""

    def __init__(
        self,
        input_neurons: int = 5,
        hidden_layers: Sequence[int] | None = None,
        output_neurons: int = 2,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        hidden_layers = list(hidden_layers or [12, 8])

        try:
            activation_cls = getattr(nn, activation.capitalize())
        except AttributeError as exc:
            raise ValueError(f"Unsupported activation: {activation}") from exc
        if not issubclass(activation_cls, nn.Module):
            raise TypeError(f"Activation must be an nn.Module subclass, received {activation_cls}")
        self.activation = activation_cls()

        self.input_neurons = input_neurons
        self.hidden_layer_sizes = hidden_layers
        self.output_neurons = output_neurons

        self.hidden_layers = nn.ModuleList()
        in_features = input_neurons
        for size in hidden_layers:
            self.hidden_layers.append(nn.Linear(in_features, size))
            in_features = size

        self.decoder = nn.Linear(in_features, output_neurons)

        self.last_hidden_preacts: list[torch.Tensor] = []
        self.last_hidden_spikes: list[torch.Tensor] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)

        activ = x
        self.last_hidden_preacts = []
        self.last_hidden_spikes = []

        for layer in self.hidden_layers:
            preact = layer(activ)
            self.last_hidden_preacts.append(preact)
            activ = self.activation(preact)
            self.last_hidden_spikes.append(torch.sigmoid(preact))

        return self.decoder(activ)
