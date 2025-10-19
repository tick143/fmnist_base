from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import torch

from ..utils.activations import ActivationRecord


@dataclass(slots=True)
class BatchContext:
    """Rich context describing a single optimisation step."""

    epoch: int
    batch_idx: int
    model: torch.nn.Module
    inputs: torch.Tensor
    targets: torch.Tensor
    outputs: torch.Tensor
    loss: torch.Tensor
    device: torch.device
    activations: Dict[str, ActivationRecord] = field(default_factory=dict)
    extras: Dict[str, float] = field(default_factory=dict)
