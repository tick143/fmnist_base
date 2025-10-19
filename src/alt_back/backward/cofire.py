from __future__ import annotations

from typing import Any

import torch

from .base import BackwardStrategy
from ..training.context import BatchContext


class CoFireBackwardStrategy(BackwardStrategy):
    """Entropy-reduction learning rule encouraging neurons to co-fire."""

    def __init__(
        self,
        lr: float = 1e-3,
        clamp_value: float = 5.0,
        clamp_penalty: float = 0.01,
        entropy_weight: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(lr=lr, clamp_value=clamp_value, clamp_penalty=clamp_penalty, entropy_weight=entropy_weight, **kwargs)
        self.lr = lr
        self.clamp_value = clamp_value
        self.clamp_penalty = clamp_penalty
        self.entropy_weight = entropy_weight
        self.eps = 1e-8

    def backward(self, context: BatchContext) -> None:
        with torch.no_grad():
            clamp_events = 0.0
            total_units = 0.0
            entropy_accumulator = []

            for name, record in context.activations.items():
                module = record.module
                if not hasattr(module, "weight"):
                    continue

                inputs = record.inputs[0]
                outputs = record.output

                clamped_output = outputs.clamp(-self.clamp_value, self.clamp_value)
                clamp_mask = (outputs.abs() >= self.clamp_value).float()
                clamp_events += clamp_mask.sum().item()
                total_units += clamp_mask.numel()

                weight = module.weight
                input_flat = self._flatten_input(inputs)
                output_flat = self._flatten_output(clamped_output)

                input_mag = input_flat.abs().mean(dim=0)
                output_mag = output_flat.abs().mean(dim=0)

                correlation = torch.outer(output_mag, input_mag)
                prob = correlation / (correlation.sum() + self.eps)
                entropy = -(prob * torch.log(prob + self.eps)).sum()
                entropy_accumulator.append(entropy.item())

                desired = torch.zeros_like(prob)
                top_indices = prob.argmax(dim=1, keepdim=True)
                desired.scatter_(1, top_indices, 1.0)

                grad = (prob - desired) * self.entropy_weight

                if weight.dim() == 2:
                    module.weight.grad = (grad * self.lr).clone()
                elif weight.dim() == 4:
                    grad = grad.unsqueeze(-1).unsqueeze(-1)
                    module.weight.grad = (grad.expand_as(weight) * self.lr).clone()
                else:
                    continue

                if module.bias is not None:
                    bias_grad = output_mag - output_mag.mean()
                    module.bias.grad = bias_grad * self.entropy_weight * self.lr

                module.weight.grad.add_(self.clamp_penalty * (weight - weight.clamp(-self.clamp_value, self.clamp_value)))

            if total_units > 0:
                context.extras["clamp_ratio"] = clamp_events / total_units
            if entropy_accumulator:
                context.extras["cofire_entropy"] = sum(entropy_accumulator) / len(entropy_accumulator)

    def _flatten_input(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.dim() > 2:
            return tensor.flatten(start_dim=2).mean(dim=2)
        return tensor

    def _flatten_output(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.dim() > 2:
            return tensor.flatten(start_dim=2).mean(dim=2)
        return tensor
