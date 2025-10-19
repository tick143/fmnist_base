from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from .base import BackwardStrategy
from ..training.context import BatchContext


class MassRedistributionBackwardStrategy(BackwardStrategy):
    """Reward-modulated neurotransmitter flooding that reallocates synaptic mass without gradients."""

    def __init__(
        self,
        release_rate: float = 0.25,
        reward_gain: float = 0.6,
        base_release: float = 0.1,
        decay: float = 0.05,
        temperature: float = 1.0,
        efficiency_bonus: float = 0.4,
        column_competition: float = 0.3,
        noise_std: float = 0.01,
        mass_budget: float = 5.0,
        signed_weights: bool = True,
        enable_target_bonus: bool = True,
        target_gain: float = 0.5,
        affinity_strength: float = 0.1,
        affinity_decay: float = 0.99,
        affinity_temperature: float = 1.5,
        sign_consistency_strength: float = 0.2,
        sign_consistency_momentum: float = 0.9,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.release_rate = float(release_rate)
        self.reward_gain = float(reward_gain)
        self.base_release = float(base_release)
        self.decay = float(decay)
        self.temperature = float(max(temperature, 1e-3))
        self.efficiency_bonus = float(efficiency_bonus)
        self.column_competition = float(column_competition)
        self.noise_std = float(max(noise_std, 0.0))
        self.mass_budget = float(mass_budget)
        self.signed_weights = bool(signed_weights)
        self.enable_target_bonus = bool(enable_target_bonus)
        self.target_gain = float(target_gain)
        self.affinity_strength = float(max(affinity_strength, 0.0))
        self.affinity_decay = float(max(min(affinity_decay, 1.0), 0.0))
        self.affinity_temperature = float(max(affinity_temperature, 1e-3))
        self.sign_consistency_strength = float(max(sign_consistency_strength, 0.0))
        self.sign_consistency_momentum = float(max(min(sign_consistency_momentum, 1.0), 0.0))

        self._mass_buffers: dict[int, torch.Tensor] = {}
        self._affinity_buffers: dict[int, torch.Tensor] = {}
        self._sign_buffers: dict[int, torch.Tensor] = {}
        self.eps = 1e-8

    def backward(self, context: BatchContext) -> None:
        with torch.no_grad():
            predictions = context.outputs.argmax(dim=1)
            correct_mask = predictions.eq(context.targets)
            accuracy = correct_mask.float().mean().item() if correct_mask.numel() > 0 else 0.0
            reward = 2.0 * accuracy - 1.0
            release = max(self.base_release + self.reward_gain * reward, 0.0)

            context.extras["reward"] = reward
            context.extras["release"] = release

            total_entropy = 0.0
            total_signal = 0.0
            total_column_error = 0.0
            module_count = 0
            prev_neuron_sign: torch.Tensor | None = None

            for name, record in context.activations.items():
                module = record.module
                if not hasattr(module, "weight"):
                    continue

                weight = module.weight
                if weight is None or weight.dim() != 2:
                    continue

                inputs = record.inputs[0]
                outputs = record.output
                if inputs.dim() != 2 or outputs.dim() != 2:
                    continue

                spikes = torch.sigmoid(outputs)
                if spikes.numel() == 0 or inputs.numel() == 0:
                    continue

                module_key = id(module)
                mass = self._initialise_mass(module_key, weight).to(weight.device, dtype=weight.dtype)
                affinity = self._initialise_affinity(module_key, weight, mass).to(weight.device, dtype=weight.dtype)
                sign_pref = self._initialise_sign(module_key, weight).to(weight.device, dtype=weight.dtype)

                # Activity-driven signals
                coactivity = torch.relu(torch.matmul(spikes.T, inputs))
                pattern = F.softmax(coactivity / self.temperature, dim=0)

                spike_counts = spikes.sum(dim=0)
                if spike_counts.max().item() <= self.eps:
                    efficiency = torch.ones_like(spike_counts)
                else:
                    efficiency = 1.0 - (spike_counts / (spike_counts.max() + self.eps))
                efficiency = 1.0 + self.efficiency_bonus * efficiency

                column_pressure = self.column_competition * mass.sum(dim=0, keepdim=True)

                alloc_signal = pattern * efficiency.unsqueeze(1)
                alloc_signal = alloc_signal - column_pressure
                if release > 0.0:
                    alloc_signal = alloc_signal * (1.0 + release)
                if self.affinity_strength > 0.0:
                    alloc_signal = alloc_signal + self.affinity_strength * affinity
                if (
                    self.sign_consistency_strength > 0.0
                    and prev_neuron_sign is not None
                    and prev_neuron_sign.numel() == weight.size(1)
                ):
                    column_sign = prev_neuron_sign.to(weight.device, dtype=weight.dtype).unsqueeze(0)
                    alloc_signal = alloc_signal + self.sign_consistency_strength * column_sign

                if self.enable_target_bonus and reward > 0.0 and mass.size(0) == context.outputs.size(1):
                    bonus = torch.zeros_like(alloc_signal)
                    for cls_idx in range(context.outputs.size(1)):
                        class_mask = context.targets == cls_idx
                        if class_mask.any():
                            class_inputs = inputs[class_mask]
                            signal = class_inputs.sum(dim=0).abs()
                            if signal.sum().item() > self.eps:
                                signal = signal / (signal.sum() + self.eps)
                                bonus[cls_idx] = signal
                    alloc_signal = alloc_signal + self.target_gain * reward * bonus

                if self.noise_std > 0.0:
                    alloc_signal = alloc_signal + self.noise_std * torch.rand_like(alloc_signal)

                alloc_signal = alloc_signal.clamp_min(self.eps)
                alloc_signal = alloc_signal / alloc_signal.sum(dim=0, keepdim=True).clamp_min(self.eps)

                mass = (1.0 - self.release_rate) * mass + self.release_rate * alloc_signal

                decay_factor = 1.0 - self.decay * (spike_counts / (spikes.size(0) + self.eps))
                decay_factor = decay_factor.clamp(min=0.0).unsqueeze(1)
                mass = mass * decay_factor

                mass = mass.clamp_min(self.eps)
                mass = mass / mass.sum(dim=0, keepdim=True).clamp_min(self.eps)

                self._mass_buffers[module_key] = mass.detach().cpu()

                if self.signed_weights:
                    sign_unit = torch.sign(sign_pref)
                    sign_unit[sign_unit == 0] = 1.0
                    new_weight = self.mass_budget * mass * sign_unit
                else:
                    new_weight = self.mass_budget * mass
                module.weight.data.copy_(new_weight)

                if self.affinity_strength > 0.0:
                    updated_affinity = self.affinity_decay * affinity + (1.0 - self.affinity_decay) * mass
                    updated_affinity = F.softmax(updated_affinity / self.affinity_temperature, dim=0)
                    self._affinity_buffers[module_key] = updated_affinity.detach().cpu()

                if self.signed_weights:
                    observed_sign = torch.sign(new_weight + self.eps)
                    updated_sign = (
                        self.sign_consistency_momentum * sign_pref
                        + (1.0 - self.sign_consistency_momentum) * observed_sign
                    )
                    updated_sign = torch.tanh(updated_sign)
                    self._sign_buffers[module_key] = updated_sign.detach().cpu()
                    row_sign = updated_sign.mean(dim=1)
                else:
                    row_sign = torch.ones(weight.size(0), device=weight.device, dtype=weight.dtype)

                if module.bias is not None:
                    module.bias.data.mul_(1.0 - self.decay)

                module_entropy = -(mass * torch.log(mass + self.eps)).sum(dim=1).mean().item()
                total_entropy += module_entropy
                total_signal += new_weight.abs().sum().item() / max(new_weight.size(0), 1)
                column_mass = new_weight.abs().sum(dim=0)
                total_column_error += (column_mass - self.mass_budget).abs().mean().item()

                if prev_neuron_sign is None or prev_neuron_sign.numel() != row_sign.numel():
                    prev_neuron_sign = row_sign.detach().cpu()
                else:
                    blended_sign = (
                        self.sign_consistency_momentum * prev_neuron_sign.to(row_sign.device)
                        + (1.0 - self.sign_consistency_momentum) * row_sign
                    )
                    prev_neuron_sign = blended_sign.detach().cpu()

                module_count += 1

            if module_count > 0:
                context.extras["mass_entropy"] = total_entropy / module_count
                context.extras["avg_signal"] = total_signal / module_count
                context.extras["modules_tracked"] = float(module_count)
                context.extras["column_budget_error"] = total_column_error / module_count

    def _initialise_mass(self, module_key: int, weight: torch.Tensor) -> torch.Tensor:
        if module_key in self._mass_buffers and self._mass_buffers[module_key].shape == weight.shape:
            return self._mass_buffers[module_key]

        abs_weight = weight.detach().cpu().abs()
        if abs_weight.sum().item() <= self.eps:
            abs_weight = torch.rand_like(abs_weight)
        mass = abs_weight.clamp_min(self.eps)
        column_totals = mass.sum(dim=0, keepdim=True)
        column_totals[column_totals <= self.eps] = 1.0
        mass = mass / column_totals
        self._mass_buffers[module_key] = mass
        return mass

    def _initialise_affinity(self, module_key: int, weight: torch.Tensor, mass: torch.Tensor) -> torch.Tensor:
        if module_key in self._affinity_buffers and self._affinity_buffers[module_key].shape == weight.shape:
            return self._affinity_buffers[module_key]

        mass_cpu = mass.detach().cpu()
        prior = torch.rand_like(mass_cpu)
        prior = prior.clamp_min(self.eps)
        prior = prior / prior.sum(dim=0, keepdim=True).clamp_min(self.eps)
        combined = 0.5 * prior + 0.5 * mass_cpu
        combined = combined / combined.sum(dim=0, keepdim=True).clamp_min(self.eps)
        self._affinity_buffers[module_key] = combined
        return combined

    def _initialise_sign(self, module_key: int, weight: torch.Tensor) -> torch.Tensor:
        if module_key in self._sign_buffers and self._sign_buffers[module_key].shape == weight.shape:
            return self._sign_buffers[module_key]

        if weight.numel() == 0:
            sign = torch.ones_like(weight)
        else:
            sign = torch.sign(weight.detach().cpu())
            sign[sign == 0] = 1.0
        self._sign_buffers[module_key] = sign
        return sign
