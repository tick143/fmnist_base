from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch

from ..training.context import BatchContext


class BackwardStrategy(ABC):
    """Interface for custom gradient computation or credit assignment rules."""

    def __init__(self, **kwargs: Any) -> None:
        self.extra_config = kwargs

    @abstractmethod
    def backward(self, context: BatchContext) -> None:
        """Populate gradients on model parameters given the loss and batch context."""

    def zero_grad(self, model: torch.nn.Module) -> None:
        """Clear existing gradients before the backward step."""
        for param in model.parameters():
            if param.grad is not None:
                param.grad = None
