from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

import torch
from torch import nn

from ..backward.mass_redistribution import MassRedistributionBackwardStrategy
from ..data.synthetic import SyntheticDatasetConfig, create_dataloaders
from ..models.spiking import TinySpikingNetwork
from ..optim.torch_optimizer import TorchOptimizerStrategy
from ..training.context import BatchContext
from ..utils.activations import ActivationRecorder


@dataclass
class TrainerConfig:
    """Bundle configuration for the interactive tiny spiking trainer."""

    dataset: SyntheticDatasetConfig = field(default_factory=SyntheticDatasetConfig)
    device: str = "cpu"
    base_lr: float = 0.1
    output_lr: float = 0.1
    correct_scale: float = 0.4
    incorrect_scale: float = 1.0
    max_signal: float = 3.0
    redistribution_rate: float = 0.05
    focus_power: float = 1.5
    temperature: float = 1.0
    min_signal: float = 0.5


class TinySpikingTrainer:
    """Perform batched updates with mass redistribution and expose step snapshots."""

    def __init__(self, config: TrainerConfig | None = None) -> None:
        self.config = config or TrainerConfig()
        self.device = torch.device(self.config.device)
        self.loss_fn = nn.CrossEntropyLoss()

        self.model = TinySpikingNetwork(
            input_neurons=self.config.dataset.num_features,
            hidden_neurons=10,
            output_neurons=2,
        ).to(self.device)

        self.backward_strategy = MassRedistributionBackwardStrategy(
            base_lr=self.config.base_lr,
            output_lr=self.config.output_lr,
            correct_scale=self.config.correct_scale,
            incorrect_scale=self.config.incorrect_scale,
            max_signal=self.config.max_signal,
            redistribution_rate=self.config.redistribution_rate,
            focus_power=self.config.focus_power,
            temperature=self.config.temperature,
            min_signal=self.config.min_signal,
        )
        self.optimizer_strategy = TorchOptimizerStrategy("torch.optim.SGD", lr=0.0)
        self.optimizer_strategy.setup(self.model)

        self.global_step = 0
        self._build_data()

    def _build_data(self) -> None:
        self.train_loader, self.test_loader = create_dataloaders(self.config.dataset)
        self._train_iter = iter(self.train_loader)

    def reset(self, seed: int | None = None) -> None:
        """Reset data streams and model parameters."""
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            self.config.dataset.seed = seed

        self.model.apply(self._init_weights)
        self._build_data()
        self.global_step = 0

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _snapshot_weights(self) -> Dict[str, Dict[str, list[list[float]] | list[float]]]:
        snapshot: Dict[str, Dict[str, list[list[float]] | list[float]]] = {}
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                snapshot[name] = {
                    "weight": module.weight.detach().cpu().tolist(),
                    "bias": module.bias.detach().cpu().tolist() if module.bias is not None else [],
                }
        return snapshot

    def _compute_deltas(
        self,
        before: Dict[str, Dict[str, list[list[float]] | list[float]]],
        after: Dict[str, Dict[str, list[list[float]] | list[float]]],
    ) -> Dict[str, Dict[str, list[list[float]] | list[float]]]:
        deltas: Dict[str, Dict[str, list[list[float]] | list[float]]] = {}
        for layer, weights in after.items():
            prev = before.get(layer)
            if prev is None:
                continue
            weight_before = torch.tensor(prev["weight"])
            weight_after = torch.tensor(weights["weight"])
            bias_before = torch.tensor(prev["bias"])
            bias_after = torch.tensor(weights["bias"])
            deltas[layer] = {
                "weight": (weight_after - weight_before).tolist(),
                "bias": (bias_after - bias_before).tolist(),
            }
        return deltas

    def step(self) -> Dict[str, Any]:
        """Perform a single optimisation step and return a rich snapshot for visualisation."""
        self.model.train()
        try:
            batch = next(self._train_iter)
        except StopIteration:
            self._train_iter = iter(self.train_loader)
            batch = next(self._train_iter)

        inputs, targets = batch
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)

        before = self._snapshot_weights()
        self.optimizer_strategy.zero_grad(model=self.model)
        self.backward_strategy.zero_grad(model=self.model)

        with ActivationRecorder(self.model) as recorder:
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, targets)
            context = BatchContext(
                epoch=0,
                batch_idx=self.global_step,
                model=self.model,
                inputs=inputs,
                targets=targets,
                outputs=outputs,
                loss=loss,
                device=self.device,
                activations=dict(recorder.records),
            )
            self.backward_strategy.backward(context)
            self.optimizer_strategy.step(context)

        after = self._snapshot_weights()
        deltas = self._compute_deltas(before, after)

        preds = outputs.argmax(dim=1)
        accuracy = (preds == targets).float().mean().item()
        extras = {key: float(value) for key, value in context.extras.items()}

        hidden_linear = context.activations["encoder"].output
        spike_rates = self.model.spike(hidden_linear.detach()).cpu()

        result = {
            "step": self.global_step,
            "loss": float(loss.item()),
            "batch_accuracy": float(accuracy * 100.0),
            "predictions": preds.detach().cpu().tolist(),
            "targets": targets.detach().cpu().tolist(),
            "inputs": inputs.detach().cpu().tolist(),
            "logits": outputs.detach().cpu().tolist(),
            "hidden_preact": hidden_linear.detach().cpu().tolist(),
            "hidden_spike_rates": spike_rates.tolist(),
            "weights": after,
            "weight_deltas": deltas,
            "extras": extras,
        }

        self.global_step += 1
        return result

    def evaluate(self) -> Dict[str, float]:
        """Compute loss and accuracy on the held-out test dataset."""
        self.model.eval()
        loss_total = 0.0
        correct = 0
        count = 0

        with torch.no_grad():
            for inputs, targets in self.test_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)
                loss_total += loss.item() * targets.size(0)
                predictions = outputs.argmax(dim=1)
                correct += (predictions == targets).sum().item()
                count += targets.size(0)

        loss_avg = loss_total / max(count, 1)
        accuracy = correct / max(count, 1) * 100.0
        return {"loss": float(loss_avg), "accuracy": float(accuracy)}

    def topology(self) -> Dict[str, Any]:
        """Describe the network layout for client visualisations."""
        return {
            "input_neurons": self.model.input_neurons,
            "hidden_neurons": self.model.hidden_neurons,
            "output_neurons": self.model.output_neurons,
        }
