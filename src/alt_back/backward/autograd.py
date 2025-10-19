from __future__ import annotations

from typing import Any

from .base import BackwardStrategy
from ..training.context import BatchContext


class AutogradBackwardStrategy(BackwardStrategy):
    """Default strategy that delegates to torch.autograd."""

    def __init__(self, retain_graph: bool = False, **kwargs: Any) -> None:
        super().__init__(retain_graph=retain_graph, **kwargs)
        self.retain_graph = retain_graph

    def backward(self, context: BatchContext) -> None:
        context.loss.backward(retain_graph=self.retain_graph)
