from __future__ import annotations

from typing import Any

import torch

from .base import OptimizerStrategy
from ..training.context import BatchContext


class NullOptimizerStrategy(OptimizerStrategy):
    """Optimizer stub for strategies that mutate parameters directly."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    def setup(self, model: torch.nn.Module) -> None:  # noqa: D401 - intentionally empty
        return

    def zero_grad(self, model: torch.nn.Module) -> None:
        return

    def step(self, context: BatchContext) -> None:
        return
