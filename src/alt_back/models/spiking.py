from __future__ import annotations

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
    """Minimal two-layer spiking network leveraged for interactive demos."""

    def __init__(
        self,
        input_neurons: int = 5,
        hidden_neurons: int = 10,
        output_neurons: int = 2,
        spike_threshold: float = 0.0,
        spike_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.input_neurons = input_neurons
        self.hidden_neurons = hidden_neurons
        self.output_neurons = output_neurons

        self.encoder = nn.Linear(input_neurons, hidden_neurons)
        self.spike = SpikingActivation(threshold=spike_threshold, temperature=spike_temperature)
        self.decoder = nn.Linear(hidden_neurons, output_neurons)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        spikes = self.spike(self.encoder(x))
        return self.decoder(spikes)
