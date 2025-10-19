from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from ..utils.activations import ActivationRecord


def compute_prediction_metrics(logits: torch.Tensor) -> dict[str, float]:
    probabilities = F.softmax(logits, dim=1)
    log_probs = torch.log(probabilities + 1e-8)
    entropy = -(probabilities * log_probs).sum(dim=1).mean().item()

    return {
        "entropy": entropy,
    }


def compute_activation_entropies(activations: Dict[str, ActivationRecord]) -> Tuple[dict[str, float], float]:
    layer_entropies: dict[str, float] = {}
    entropies = []
    for name, record in activations.items():
        output = record.output
        if output.ndim > 2:
            flat = output.flatten(start_dim=1).abs()
        else:
            flat = output.abs()

        probs = flat / (flat.sum(dim=1, keepdim=True) + 1e-8)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean()
        entropy_value = entropy.item()
        layer_entropies[name] = entropy_value
        entropies.append(entropy)

    network_entropy = torch.stack(entropies).mean().item() if entropies else 0.0
    return layer_entropies, network_entropy
