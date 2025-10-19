from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterable

import torch

from ..training.context import BatchContext


class OptimizerStrategy(ABC):
    """Interface for parameter updates decoupled from backward pass."""

    def __init__(self, **kwargs: Any) -> None:
        self.extra_config = kwargs

    @abstractmethod
    def setup(self, model: torch.nn.Module) -> None:
        """Initialize any stateful components that need model parameters."""

    def zero_grad(self, model: torch.nn.Module) -> None:
        """Clear gradient buffers before a backward call."""
        for parameter in self.parameters(model):
            if parameter.grad is not None:
                parameter.grad = None

    @abstractmethod
    def step(self, context: BatchContext) -> None:
        """Apply parameter updates based on current gradients."""

    def parameters(self, model: torch.nn.Module) -> Iterable[torch.nn.Parameter]:
        return model.parameters()

    def state_dict(self) -> dict[str, Any]:
        return {}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        raise NotImplementedError("Stateful optimizers should override load_state_dict")
