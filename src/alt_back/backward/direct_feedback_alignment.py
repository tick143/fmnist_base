from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import torch
import torch.nn.functional as F

from .base import BackwardStrategy
from ..training.context import BatchContext


@dataclass
class _LayerState:
    feedback: torch.Tensor


class DirectFeedbackAlignmentBackwardStrategy(BackwardStrategy):
    """Direct Feedback Alignment (DFA) for feed-forward networks without backpropagation."""

    def __init__(
        self,
        feedback_scale: float = 1.0,
        retain_graph: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.feedback_scale = float(feedback_scale)
        self.retain_graph = bool(retain_graph)
        self._layer_states: Dict[int, _LayerState] = {}

    def backward(self, context: BatchContext) -> None:
        model = context.model
        outputs = context.outputs.detach()
        targets = context.targets.detach()
        batch_size = outputs.size(0)
        num_classes = outputs.size(1)

        with torch.no_grad():
            probs = F.softmax(outputs, dim=1)
            one_hot = F.one_hot(targets, num_classes=num_classes).to(dtype=probs.dtype)
            output_error = probs - one_hot  # (B, C)

        # Clear existing gradients
        for param in model.parameters():
            if param.grad is not None:
                param.grad = None

        # Traverse recorded activations in reverse to align with model layers
        records = list(context.activations.items())
        linear_records = [rec for rec in records if hasattr(rec[1].module, "weight") and rec[1].output.dim() == 2]

        if not linear_records:
            return

        downstream_error = output_error
        output_layer = linear_records[-1][1].module

        for name, record in reversed(linear_records):
            module = record.module
            weight = module.weight
            bias = module.bias
            inputs = record.inputs[0].detach()
            preact = record.output.detach()

            if module is output_layer:
                local_error = downstream_error
            else:
                state = self._layer_states.get(id(module))
                if state is None or state.feedback.device != weight.device or state.feedback.size(0) != weight.size(0) or state.feedback.size(1) != num_classes:
                    feedback = torch.randn(weight.size(0), num_classes, device=weight.device, dtype=weight.dtype)
                    feedback = feedback * (self.feedback_scale / math.sqrt(weight.size(0)))
                    state = _LayerState(feedback=feedback)
                    self._layer_states[id(module)] = state
                feedback = state.feedback
                local_error = downstream_error @ feedback

                # Activation derivative (assume ReLU-ish)
                local_error = local_error * (preact > 0).to(local_error.dtype)

            grad_weight = local_error.t() @ inputs
            grad_weight = grad_weight / batch_size
            weight.grad = grad_weight

            if bias is not None:
                bias.grad = local_error.mean(dim=0)

            downstream_error = local_error

        # No autodiff graph, but allow optimizers to step
        if self.retain_graph:
            context.loss.backward(retain_graph=True)
