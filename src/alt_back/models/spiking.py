from __future__ import annotations

from typing import Sequence

import torch
from torch import nn


class SpikingActivation(nn.Module):
    """Soft spike-rate activation using a temperature-controlled sigmoid."""

    def __init__(self, threshold: float = 0.0, temperature: float = 1.0) -> None:
        super().__init__()
        self.threshold = threshold
        self.temperature = temperature

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        centered = (x - self.threshold) / max(self.temperature, 1e-6)
        return torch.sigmoid(centered)


class TinySpikingNetwork(nn.Module):
    """Minimal spiking network with an arbitrary stack of hidden layers."""

    def __init__(
        self,
        input_neurons: int = 5,
        hidden_layers: Sequence[int] | None = None,
        hidden_neurons: int | None = None,
        output_neurons: int = 2,
        spike_threshold: float = 0.0,
        spike_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        hidden_layers = list(hidden_layers) if hidden_layers is not None else None
        if hidden_layers is None:
            size = hidden_neurons if hidden_neurons is not None else 10
            hidden_layers = [size]
        if not hidden_layers:
            msg = "hidden_layers must contain at least one layer."
            raise ValueError(msg)

        self.input_neurons = input_neurons
        self.hidden_layer_sizes = hidden_layers
        self.output_neurons = output_neurons

        self.hidden_layers = nn.ModuleList()
        in_features = input_neurons
        for size in hidden_layers:
            self.hidden_layers.append(nn.Linear(in_features, size))
            in_features = size

        self.spike = SpikingActivation(threshold=spike_threshold, temperature=spike_temperature)
        self.decoder = nn.Linear(in_features, output_neurons)

        self.last_hidden_preacts: list[torch.Tensor] = []
        self.last_hidden_spikes: list[torch.Tensor] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        # Flatten images/sequences to feature vectors if needed
        if x.dim() > 2:
            x = x.view(x.size(0), -1)

        activ = x
        self.last_hidden_preacts = []
        self.last_hidden_spikes = []

        for layer in self.hidden_layers:
            preact = layer(activ)
            self.last_hidden_preacts.append(preact)
            activ = self.spike(preact)
            self.last_hidden_spikes.append(activ)

        return self.decoder(activ)
