from __future__ import annotations

from typing import Any, Callable

import torch

from .base import OptimizerStrategy
from ..training.context import BatchContext
from ..utils.imports import import_from_string


class TorchOptimizerStrategy(OptimizerStrategy):
    """Wrap standard torch.optim optimizers while keeping config-driven selection."""

    def __init__(self, optimizer_class: str, **kwargs: Any) -> None:
        super().__init__(optimizer_class=optimizer_class, **kwargs)
        optimizer_factory: Callable[..., torch.optim.Optimizer] = import_from_string(optimizer_class)
        self.optimizer_cls = optimizer_factory
        self.optimizer_kwargs = kwargs
        self._optimizer: torch.optim.Optimizer | None = None

    def setup(self, model: torch.nn.Module) -> None:
        self._optimizer = self.optimizer_cls(model.parameters(), **self.optimizer_kwargs)

    def zero_grad(self, model: torch.nn.Module) -> None:
        if self._optimizer is None:
            return
        self._optimizer.zero_grad(set_to_none=True)

    def step(self, context: BatchContext) -> None:
        if self._optimizer is None:
            msg = "setup(model) must be called before step() for TorchOptimizerStrategy"
            raise RuntimeError(msg)
        self._optimizer.step()

    def state_dict(self) -> dict[str, Any]:
        if self._optimizer is None:
            return {}
        return self._optimizer.state_dict()

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if self._optimizer is None:
            msg = "setup(model) must be called before load_state_dict()"
            raise RuntimeError(msg)
        self._optimizer.load_state_dict(state)
