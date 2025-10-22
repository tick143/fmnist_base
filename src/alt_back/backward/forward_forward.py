from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict

import torch
import torch.nn.functional as F

from .base import BackwardStrategy
from ..training.context import BatchContext
from ..utils.activations import ActivationRecorder


@dataclass
class _FFLayerState:
    threshold: float


class ForwardForwardBackwardStrategy(BackwardStrategy):
    """Hinton-style Forward-Forward weight updates without backpropagation."""

    def __init__(
        self,
        step_size: float = 1e-3,
        threshold: float = 2.0,
        threshold_ema: float = 0.99,
        noise_std: float = 0.2,
        gate_temperature: float = 1.0,
        scale_by_fanin: bool = False,
        clamp: float | None = 5.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.step_size = float(step_size)
        self.threshold = float(threshold)
        self.threshold_ema = float(threshold_ema)
        self.noise_std = float(max(noise_std, 0.0))
        self.gate_temperature = float(max(gate_temperature, 1e-6))
        self.scale_by_fanin = bool(scale_by_fanin)
        self.clamp = float(clamp) if clamp is not None else None
        self._states: Dict[int, _FFLayerState] = {}

    def backward(self, context: BatchContext) -> None:
        model = context.model
        device = context.device

        with torch.no_grad():
            pos_records = context.activations

            neg_inputs = context.inputs.detach().clone()
            if neg_inputs.numel() > 0:
                perm = torch.randperm(neg_inputs.size(0), device=device)
                neg_inputs = neg_inputs[perm]
            if self.noise_std > 0:
                neg_inputs = neg_inputs + self.noise_std * torch.randn_like(neg_inputs)

            original_mode = model.training
            try:
                model.train(original_mode)
                with ActivationRecorder(model) as neg_recorder:
                    model(neg_inputs)
                    neg_records = neg_recorder.records
            finally:
                model.train(original_mode)

            last_pos_good: float | None = None
            last_neg_good: float | None = None

            for name, pos_record in pos_records.items():
                module = pos_record.module
                weight = getattr(module, "weight", None)
                if weight is None or weight.dim() != 2:
                    continue

                neg_record = neg_records.get(name)
                if neg_record is None:
                    continue

                pos_in = pos_record.inputs[0].detach()
                neg_in = neg_record.inputs[0].detach()

                pos_pre = pos_record.output.detach()
                neg_pre = neg_record.output.detach()

                pos_act = F.relu(pos_pre)
                neg_act = F.relu(neg_pre)

                layer_state = self._states.get(id(module))
                if layer_state is None:
                    layer_state = _FFLayerState(threshold=self.threshold)
                    self._states[id(module)] = layer_state

                pos_good = pos_act.pow(2).sum(dim=1)
                neg_good = neg_act.pow(2).sum(dim=1)

                pos_scale = torch.sigmoid((pos_good - layer_state.threshold) / self.gate_temperature)
                neg_scale = torch.sigmoid((neg_good - layer_state.threshold) / self.gate_temperature)

                pos_outer = pos_act.unsqueeze(2) * pos_in.unsqueeze(1)
                neg_outer = neg_act.unsqueeze(2) * neg_in.unsqueeze(1)

                pos_outer = pos_outer * pos_scale.view(-1, 1, 1)
                neg_outer = neg_outer * neg_scale.view(-1, 1, 1)

                delta_weight = pos_outer.mean(dim=0) - neg_outer.mean(dim=0)
                step_scale = self.step_size
                if self.scale_by_fanin and weight.dim() > 1:
                    step_scale = step_scale / math.sqrt(float(weight.size(1)))
                weight.data.add_(step_scale * delta_weight)

                if module.bias is not None:
                    pos_bias = (pos_act * pos_scale.view(-1, 1)).mean(dim=0)
                    neg_bias = (neg_act * neg_scale.view(-1, 1)).mean(dim=0)
                    module.bias.data.add_(step_scale * (pos_bias - neg_bias))

                if self.clamp is not None:
                    weight.data.clamp_(-self.clamp, self.clamp)
                    if module.bias is not None:
                        module.bias.data.clamp_(-self.clamp, self.clamp)

                pos_mean = pos_good.mean().item()
                neg_mean = neg_good.mean().item()
                layer_state.threshold = float(
                    self.threshold_ema * layer_state.threshold
                    + (1.0 - self.threshold_ema) * 0.5 * (pos_mean + neg_mean)
                )
                last_pos_good = float(pos_mean)
                last_neg_good = float(neg_mean)

        if last_pos_good is not None:
            context.extras["ff_pos_good"] = last_pos_good
        else:
            context.extras.pop("ff_pos_good", None)

        if last_neg_good is not None:
            context.extras["ff_neg_good"] = last_neg_good
        else:
            context.extras.pop("ff_neg_good", None)
