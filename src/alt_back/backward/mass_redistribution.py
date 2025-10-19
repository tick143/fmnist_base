from __future__ import annotations

from typing import Any, Callable, Tuple

import torch
import torch.nn.functional as F

from .base import BackwardStrategy
from ..training.context import BatchContext


class MassRedistributionBackwardStrategy(BackwardStrategy):
    """Biologically inspired rule that redistributes synaptic mass without gradients."""

    def __init__(
        self,
        base_lr: float = 0.1,
        output_lr: float | None = None,
        correct_scale: float = 0.3,
        incorrect_scale: float = 1.0,
        max_signal: float = 3.0,
        redistribution_rate: float = 0.05,
        focus_power: float = 1.5,
        temperature: float = 1.0,
        min_signal: float = 0.5,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            base_lr=base_lr,
            output_lr=output_lr,
            correct_scale=correct_scale,
            incorrect_scale=incorrect_scale,
            max_signal=max_signal,
            redistribution_rate=redistribution_rate,
            focus_power=focus_power,
            temperature=temperature,
            min_signal=min_signal,
            **kwargs,
        )
        self.base_lr = base_lr
        self.output_lr = output_lr if output_lr is not None else base_lr
        self.correct_scale = correct_scale
        self.incorrect_scale = incorrect_scale
        self.max_signal = max_signal
        self.min_signal = min_signal
        self.redistribution_rate = redistribution_rate
        self.focus_power = focus_power
        self.temperature = temperature
        self.eps = 1e-8

    def backward(self, context: BatchContext) -> None:
        with torch.no_grad():
            predictions = context.outputs.argmax(dim=1)
            correct_mask = predictions.eq(context.targets)
            scale = torch.where(
                correct_mask,
                torch.full((correct_mask.size(0),), self.correct_scale, device=context.outputs.device, dtype=context.outputs.dtype),
                torch.full((correct_mask.size(0),), self.incorrect_scale, device=context.outputs.device, dtype=context.outputs.dtype),
            )

            mean_scale = scale.mean().item() if scale.numel() > 0 else 0.0
            context.extras["mean_step_scale"] = mean_scale

            module_count = 0
            total_signal = 0.0

            for name, record in context.activations.items():
                module = record.module
                if not hasattr(module, "weight"):
                    continue

                weight = module.weight
                if weight.dim() not in (2, 4):
                    continue

                flat_weight, reshape_fn = self._flatten_weight(weight)

                inputs, outputs = self._prepare_activations(module, record.inputs[0], record.output)

                if inputs.numel() == 0 or outputs.numel() == 0:
                    continue

                sample_scale = self._expand_scale(scale, outputs.size(0), record.output.shape)
                scaled_outputs = outputs * sample_scale.unsqueeze(1)
                coactivity = torch.matmul(scaled_outputs.T, inputs)

                if self._is_output_layer(module, record.output, context.outputs):
                    self._update_output_layer(module, record.inputs[0], context)

                focused = torch.tanh(coactivity / (self.temperature + 1e-6))
                signed_step = torch.sign(coactivity) * focused.abs().pow(self.focus_power)

                lr = self.base_lr
                new_weight = flat_weight + lr * signed_step

                row_l1 = new_weight.abs().sum(dim=1, keepdim=True)
                over = (row_l1 > self.max_signal).float()
                shrink = (row_l1 - self.max_signal) / (row_l1 + self.eps)
                new_weight = new_weight - over * shrink * new_weight
                new_weight = torch.nan_to_num(new_weight, nan=0.0, posinf=self.max_signal, neginf=-self.max_signal)

                module.weight.data.copy_(reshape_fn(new_weight))

                if module.weight.grad is not None:
                    module.weight.grad = None
                if module.bias is not None and module.bias.grad is not None:
                    module.bias.grad = None

                context.extras.setdefault("mass_entropy", 0.0)
                abs_alloc = new_weight.abs() / (new_weight.abs().sum(dim=1, keepdim=True) + self.eps)
                entropy = -(abs_alloc * torch.log(abs_alloc + self.eps)).sum(dim=1).mean().item()
                context.extras["mass_entropy"] += entropy
                total_signal += row_l1.mean().item()

                module_count += 1

            if module_count > 0 and "mass_entropy" in context.extras:
                context.extras["mass_entropy"] /= module_count
                context.extras["modules_tracked"] = float(module_count)
                context.extras["avg_signal"] = total_signal / module_count

    def _prepare_activations(
        self,
        module: torch.nn.Module,
        inputs: torch.Tensor,
        outputs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if inputs.dim() == 2 and outputs.dim() == 2:
            return inputs, outputs

        if inputs.dim() == 4 and outputs.dim() == 4:
            batch_size = inputs.size(0)
            unfold = F.unfold(
                inputs,
                kernel_size=module.kernel_size,
                dilation=module.dilation,
                padding=module.padding,
                stride=module.stride,
            )
            patches = unfold.transpose(1, 2).reshape(-1, unfold.size(1))
            out_flat = outputs.view(batch_size, outputs.size(1), -1).transpose(1, 2).reshape(-1, outputs.size(1))
            return patches, out_flat

        raise ValueError("Unsupported activation shapes for mass redistribution strategy")

    def _expand_scale(self, scale: torch.Tensor, expanded_size: int, original_output_shape: torch.Size) -> torch.Tensor:
        if expanded_size == scale.size(0):
            return scale

        batch_size = scale.size(0)
        if len(original_output_shape) == 4:
            spatial_locs = original_output_shape[2] * original_output_shape[3]
            return scale.repeat_interleave(spatial_locs)

        raise ValueError("Cannot expand scale for the given output shape")

    def _flatten_weight(self, weight: torch.Tensor) -> tuple[torch.Tensor, Callable[[torch.Tensor], torch.Tensor]]:
        if weight.dim() == 2:
            return weight.view(weight.size(0), -1), lambda w: w.view_as(weight)
        if weight.dim() == 4:
            shape = weight.shape
            return weight.view(shape[0], -1), lambda w: w.view(shape)
        raise ValueError("Unsupported weight dimensionality for mass redistribution strategy")

    def _is_output_layer(
        self,
        module: torch.nn.Module,
        hook_output: torch.Tensor,
        model_output: torch.Tensor,
    ) -> bool:
        if not isinstance(module, torch.nn.Linear):
            return False
        return hook_output.shape == model_output.shape

    def _update_output_layer(
        self,
        module: torch.nn.Module,
        inputs: torch.Tensor,
        context: BatchContext,
    ) -> None:
        with torch.no_grad():
            probs = F.softmax(context.outputs.detach(), dim=1)
            targets = torch.nn.functional.one_hot(context.targets, num_classes=probs.size(1)).to(probs.dtype)
            error = targets - probs
            delta = (error.T @ inputs) / inputs.size(0)
            module.weight.data.add_(self.output_lr * delta)
            if module.bias is not None:
                module.bias.data.add_(self.output_lr * error.mean(dim=0))
