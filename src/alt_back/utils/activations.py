from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn


@dataclass
class ActivationRecord:
    module: nn.Module
    inputs: tuple[torch.Tensor, ...]
    output: torch.Tensor


class ActivationRecorder:
    """Capture forward activations for selected modules."""

    def __init__(
        self,
        model: nn.Module,
        predicate: Callable[[str, nn.Module], bool] | None = None,
    ) -> None:
        self.model = model
        self.predicate = predicate or self._default_predicate
        self.handles: list[torch.utils.hooks.RemovableHandle] = []
        self._records: dict[str, ActivationRecord] = {}
        self._register()

    def _default_predicate(self, name: str, module: nn.Module) -> bool:
        return isinstance(module, (nn.Linear, nn.Conv2d, nn.BatchNorm1d, nn.BatchNorm2d))

    def _register(self) -> None:
        for name, module in self.model.named_modules():
            if name and self.predicate(name, module):
                handle = module.register_forward_hook(self._make_hook(name))
                self.handles.append(handle)

    def _make_hook(self, name: str) -> Callable[[nn.Module, tuple[torch.Tensor, ...], torch.Tensor], None]:
        def hook(module: nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            self._records[name] = ActivationRecord(
                module=module,
                inputs=tuple(inp.detach() for inp in inputs),
                output=output.detach(),
            )

        return hook

    def clear(self) -> None:
        self._records.clear()

    @property
    def records(self) -> dict[str, ActivationRecord]:
        return self._records

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.clear()

    def __enter__(self) -> "ActivationRecorder":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
