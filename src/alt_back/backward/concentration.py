from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import torch

from .base import BackwardStrategy
from ..training.context import BatchContext


@dataclass
class _ModuleState:
    baseline: torch.Tensor


class ConcentrationGradientBackwardStrategy(BackwardStrategy):
    """Energy-driven concentration gradients without backpropagation."""

    def __init__(
        self,
        push_rate: float = 0.3,
        suppress_rate: float = 0.05,
        step_scale: float = 0.02,
        energy_slope: float = 1.4,
        energy_momentum: float = 0.5,
        concentration_momentum: float = 0.85,
        loss_tolerance: float = 1e-5,
        weight_clamp: float | None = 7.0,
        direction_mode: str = "outputs_minus_inputs",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.push_rate = float(push_rate)
        self.suppress_rate = float(suppress_rate)
        self.step_scale = float(step_scale)
        self.energy_slope = float(energy_slope)
        self.energy_momentum = float(max(min(energy_momentum, 1.0), 0.0))
        self.concentration_momentum = float(max(min(concentration_momentum, 1.0), 0.0))
        self.loss_tolerance = float(max(loss_tolerance, 0.0))
        self.weight_clamp = float(weight_clamp) if weight_clamp is not None else None
        allowed_modes = {"outputs_minus_inputs", "inputs_minus_outputs", "weight_sign", "raw_mean"}
        direction_mode = direction_mode.lower()
        if direction_mode not in allowed_modes:
            msg = f"Unsupported direction_mode '{direction_mode}'. Expected one of {sorted(allowed_modes)}."
            raise ValueError(msg)
        self.direction_mode = direction_mode

        self.prev_loss: float | None = None
        self.running_energy: float = 0.0
        self._module_states: Dict[int, _ModuleState] = {}
        self.eps = 1e-8

    def backward(self, context: BatchContext) -> None:
        with torch.no_grad():
            batch_loss = float(context.loss.detach().item())
            energy_total = 0.0
            module_payload: list[dict[str, Any]] = []

            for name, record in context.activations.items():
                module = record.module
                weight = getattr(module, "weight", None)
                if weight is None:
                    continue
                if weight.dim() not in {2, 4}:
                    continue

                inputs = self._flatten_inputs(record.inputs[0])
                outputs = self._flatten_outputs(record.output)
                if inputs is None or outputs is None:
                    continue
                if inputs.size(0) == 0 or outputs.size(0) == 0:
                    continue

                if outputs.size(0) != inputs.size(0):
                    # Batch misalignment, skip safely.
                    continue

                module_energy = self._sequential_energy(outputs)
                energy_total += module_energy

                delta_matrix = outputs.unsqueeze(2) - inputs.unsqueeze(1)
                delta_abs = delta_matrix.abs()
                neuron_delta = delta_abs.mean(dim=(0, 2))
                if neuron_delta.numel() == 0:
                    continue

                mean_delta = neuron_delta.mean()
                contrast = neuron_delta - mean_delta
                direction = self._compute_direction(delta_matrix, weight)

                module_key = id(module)
                state = self._module_states.get(module_key)
                if state is None:
                    baseline = neuron_delta.detach()
                    self._module_states[module_key] = _ModuleState(baseline=baseline.cpu())
                else:
                    baseline = state.baseline.to(weight.device, dtype=weight.dtype)
                    updated = (
                        self.concentration_momentum * baseline
                        + (1.0 - self.concentration_momentum) * neuron_delta
                    )
                    state.baseline = updated.detach().cpu()

                module_payload.append(
                    {
                        "module": module,
                        "weight": weight,
                        "bias": getattr(module, "bias", None),
                        "direction": direction,
                        "contrast": contrast,
                        "neuron_delta": neuron_delta,
                        "module_energy": module_energy,
                    }
                )

            self.running_energy = (
                self.energy_momentum * self.running_energy + (1.0 - self.energy_momentum) * energy_total
            )

            energy_factor = max(self.running_energy * self.energy_slope, 1.0)
            punishment = max(batch_loss * energy_factor, batch_loss)

            context.extras["energy"] = self.running_energy
            context.extras["energy_factor"] = energy_factor
            context.extras["punishment"] = punishment

            improved = self.prev_loss is not None and (batch_loss + self.loss_tolerance) < self.prev_loss

            for payload in module_payload:
                weight = payload["weight"]
                direction = payload["direction"]
                contrast = payload["contrast"]
                neuron_delta = payload["neuron_delta"]
                module_energy = payload["module_energy"]
                bias = payload["bias"]

                # Normalised push / suppression signals
                push_signal = contrast.clamp_min(0.0)
                suppress_signal = (-contrast).clamp_min(0.0)

                if improved:
                    push_norm = self._normalise(push_signal)
                    suppress_norm = self._normalise(suppress_signal)
                else:
                    push_norm = torch.zeros_like(push_signal)
                    suppress_norm = self._normalise(neuron_delta)

                row_delta = self.push_rate * push_norm - self.suppress_rate * suppress_norm
                base_scale = self.step_scale * punishment / (1.0 + module_energy)

                update_matrix = row_delta.unsqueeze(1) * direction
                update_matrix = torch.clamp(update_matrix, -1.0, 1.0)  # keep concentration shifts bounded
                update_matrix = base_scale * update_matrix

                if weight.dim() == 2:
                    weight.data.add_(update_matrix)
                else:
                    expanded = update_matrix.unsqueeze(-1).unsqueeze(-1).expand_as(weight)
                    weight.data.add_(expanded)

                if self.weight_clamp is not None:
                    weight.data.clamp_(-self.weight_clamp, self.weight_clamp)

                if bias is not None:
                    bias_update = row_delta * base_scale
                    bias_update = torch.clamp(bias_update, -1.0, 1.0)
                    bias.data.add_(bias_update.to(bias.device, dtype=bias.dtype))

            self.prev_loss = batch_loss

    def _flatten_inputs(self, tensor: torch.Tensor) -> torch.Tensor | None:
        if tensor is None:
            return None
        if tensor.dim() == 2:
            return tensor.detach()
        if tensor.dim() == 4:
            b, c, h, w = tensor.shape
            return tensor.view(b, c, -1).mean(dim=2).detach()
        if tensor.dim() == 3:
            b, c, l = tensor.shape
            return tensor.view(b, c, -1).mean(dim=2).detach()
        return tensor.view(tensor.size(0), -1).detach()

    def _flatten_outputs(self, tensor: torch.Tensor) -> torch.Tensor | None:
        if tensor is None:
            return None
        if tensor.dim() == 2:
            return tensor.detach()
        if tensor.dim() == 4:
            b, c, h, w = tensor.shape
            return tensor.view(b, c, -1).mean(dim=2).detach()
        if tensor.dim() == 3:
            b, c, l = tensor.shape
            return tensor.view(b, c, -1).mean(dim=2).detach()
        return tensor.view(tensor.size(0), -1).detach()

    def _sequential_energy(self, outputs: torch.Tensor) -> float:
        if outputs.size(1) < 2:
            return 0.0
        diffs = outputs[:, 1:] - outputs[:, :-1]
        return float(diffs.abs().mean().item())

    def _normalise(self, tensor: torch.Tensor) -> torch.Tensor:
        max_val = tensor.abs().max()
        if max_val <= self.eps:
            return torch.zeros_like(tensor)
        return tensor / (max_val + self.eps)

    def _compute_direction(self, delta_matrix: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        mean_delta = delta_matrix.mean(dim=0)
        if self.direction_mode == "outputs_minus_inputs":
            base = mean_delta
        elif self.direction_mode == "inputs_minus_outputs":
            base = -mean_delta
        elif self.direction_mode == "raw_mean":
            base = mean_delta
            if base.device != weight.device:
                base = base.to(weight.device)
            if base.dtype != weight.dtype:
                base = base.to(dtype=weight.dtype)
            if base.abs().max().item() <= self.eps:
                base = weight.detach()
            return base.to(weight.device, dtype=weight.dtype)
        else:  # weight_sign
            base = weight.detach()

        direction = torch.sign(base)
        direction[direction == 0] = 1.0
        if (direction.abs() <= self.eps).all():
            direction = torch.sign(weight.detach())
        if direction.dim() != 2:
            direction = direction.reshape(mean_delta.shape)
        return direction.to(weight.device, dtype=weight.dtype)
