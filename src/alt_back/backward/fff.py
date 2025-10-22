from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import torch
import torch.nn.functional as F

from .base import BackwardStrategy
from ..training.context import BatchContext


@dataclass
class _Perturbation:
    param: torch.nn.Parameter
    delta: torch.Tensor


class FFFBackwardStrategy(BackwardStrategy):
    """Forward-Forward-Fallback parameter search via random perturbations.

    For each batch we:
      1. Evaluate the current loss.
      2. Randomly shift a subset of weights with varying magnitudes.
      3. Keep the shift if the loss improves.
      4. Otherwise, apply the opposite shift and keep it if that helped.
      5. If neither shift helps, we revert to the original weights.

    Only `adjustment_decay` is exposed as a hyper-parameter. Other constants are
    intentionally kept internal to reduce configuration surface area.
    """

    def __init__(self, adjustment_decay: float = 0.99, **_: float) -> None:
        super().__init__(adjustment_decay=adjustment_decay)
        if adjustment_decay <= 0:
            raise ValueError("adjustment_decay must be positive.")
        self.adjustment_decay = float(adjustment_decay)
        self._initial_rate = 0.1
        self._min_rate = 1e-5
        self._perturb_fraction = 0.05

    def backward(self, context: BatchContext) -> None:
        model = context.model
        if not any(param.requires_grad for param in model.parameters()):
            return

        with torch.no_grad():
            base_loss = self._evaluate_loss(model, context.inputs, context.targets)
            rate = max(self._min_rate, self._initial_rate * (self.adjustment_decay ** context.epoch))

            originals = [param.data.clone() for param in model.parameters() if param.requires_grad]
            perturbations = self._generate_perturbations(model.parameters(), rate)

            self._apply_perturbations(perturbations, direction=1.0)
            forward_loss = self._evaluate_loss(model, context.inputs, context.targets)

            if forward_loss < base_loss:
                final_loss = forward_loss
                accepted_direction = 1.0
            else:
                self._restore_originals(model.parameters(), originals)
                self._apply_perturbations(perturbations, direction=-1.0)
                backward_loss = self._evaluate_loss(model, context.inputs, context.targets)

                if backward_loss < base_loss:
                    final_loss = backward_loss
                    accepted_direction = -1.0
                else:
                    self._restore_originals(model.parameters(), originals)
                    final_loss = base_loss
                    accepted_direction = 0.0

            context.extras["fff/base_loss"] = float(base_loss)
            context.extras["fff/final_loss"] = float(final_loss)
            context.extras["fff/loss_delta"] = float(base_loss - final_loss)
            context.extras["fff/rate"] = float(rate)
            context.extras["fff/direction"] = float(accepted_direction)

    def _evaluate_loss(self, model: torch.nn.Module, inputs: torch.Tensor, targets: torch.Tensor) -> float:
        was_training = model.training
        model.eval()
        with torch.no_grad():
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, targets).item()
        if was_training:
            model.train()
        return loss

    def _generate_perturbations(self, params: Iterable[torch.nn.Parameter], rate: float) -> List[_Perturbation]:
        perturbations: List[_Perturbation] = []
        for param in params:
            if not param.requires_grad:
                continue
            numel = param.data.numel()
            if numel == 0:
                continue

            sample_size = max(1, int(numel * self._perturb_fraction))
            flat_noise = torch.zeros(numel, device=param.device)
            indices = torch.randperm(numel, device=param.device)[:sample_size]
            flat_noise[indices] = torch.randn(sample_size, device=param.device)
            rms = param.data.pow(2).mean().sqrt().clamp_min(1e-8)
            fan_in = float(param.data.shape[1]) if param.data.ndim >= 2 else 1.0
            scale = 0.5 * rms + 0.5 / fan_in ** 0.5
            delta = (rate * scale) * flat_noise.view_as(param.data)
            perturbations.append(_Perturbation(param=param, delta=delta))
        return perturbations

    def _apply_perturbations(self, perturbations: Iterable[_Perturbation], direction: float) -> None:
        for perturbation in perturbations:
            perturbation.param.add_(direction * perturbation.delta)

    def _restore_originals(self, params: Iterable[torch.nn.Parameter], originals: List[torch.Tensor]) -> None:
        for param, original in zip(
            (param for param in params if param.requires_grad),
            originals,
        ):
            param.data.copy_(original)
