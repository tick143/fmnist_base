from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..backward.base import BackwardStrategy
from ..optim.base import OptimizerStrategy
from ..config import WandbConfig
from ..utils.activations import ActivationRecorder
from ..utils.logging import WandbLogger
from .metrics import compute_activation_entropies, compute_prediction_metrics
from .context import BatchContext


@dataclass
class EpochStats:
    loss: float
    accuracy: float


def run_training_batch(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    backward_strategy: BackwardStrategy,
    optimizer_strategy: OptimizerStrategy,
    device: torch.device,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    epoch: int,
    batch_idx: int,
) -> Tuple[BatchContext, torch.Tensor, torch.Tensor]:
    inputs = inputs.to(device)
    targets = targets.to(device)

    optimizer_strategy.zero_grad(model=model)
    backward_strategy.zero_grad(model=model)

    with ActivationRecorder(model) as recorder:
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)

        context = BatchContext(
            epoch=epoch,
            batch_idx=batch_idx,
            model=model,
            inputs=inputs,
            targets=targets,
            outputs=outputs,
            loss=loss,
            device=device,
            activations=dict(recorder.records),
        )

        backward_strategy.backward(context)
        optimizer_strategy.step(context)

    return context, outputs, loss


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    backward_strategy: BackwardStrategy,
    optimizer_strategy: OptimizerStrategy,
    device: torch.device,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    log_interval: int,
    epoch: int,
    logger: WandbLogger | None = None,
    wandb_cfg: WandbConfig | None = None,
    global_step: int = 0,
) -> Tuple[EpochStats, int]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(dataloader):
        context, outputs, loss = run_training_batch(
            model=model,
            inputs=inputs,
            targets=targets,
            backward_strategy=backward_strategy,
            optimizer_strategy=optimizer_strategy,
            device=device,
            loss_fn=loss_fn,
            epoch=epoch,
            batch_idx=batch_idx,
        )

        batch_size = context.inputs.size(0)
        running_loss += loss.item() * batch_size
        preds = outputs.argmax(dim=1)
        correct += preds.eq(context.targets).sum().item()
        total += context.targets.size(0)

        if logger is not None and logger.enabled:
            probs = F.softmax(outputs.detach(), dim=1)
            pred_metrics = compute_prediction_metrics(outputs.detach())
            layer_entropies, network_entropy = compute_activation_entropies(context.activations)

            log_payload = {
                "train/loss": loss.item(),
                "train/accuracy": preds.eq(context.targets).float().mean().item() * 100,
                "train/prediction_entropy": pred_metrics["entropy"],
                "train/network_entropy": network_entropy,
            }

            if wandb_cfg is None or wandb_cfg.log_logits:
                log_payload["train/logits/mean"] = outputs.detach().mean().item()
                log_payload["train/logits/std"] = outputs.detach().std().item()

            if wandb_cfg is None or wandb_cfg.log_probabilities:
                log_payload["train/probabilities/mean"] = probs.mean().item()
                log_payload["train/probabilities/std"] = probs.std().item()

            if wandb_cfg is None or wandb_cfg.log_entropies:
                for layer_name, entropy_value in layer_entropies.items():
                    sanitized = layer_name.replace(".", "/")
                    log_payload[f"train/layer_entropy/{sanitized}"] = entropy_value

            for extra_key, extra_value in context.extras.items():
                log_payload[f"train/{extra_key}"] = extra_value

            logger.log(log_payload, step=global_step)

        global_step += 1

        if log_interval and (batch_idx + 1) % log_interval == 0:
            avg_loss = running_loss / total
            accuracy = correct / total * 100
            print(f"[train] batch {batch_idx + 1}/{len(dataloader)} loss={avg_loss:.4f} acc={accuracy:.2f}%")

    avg_loss = running_loss / total
    accuracy = correct / total * 100
    return EpochStats(loss=avg_loss, accuracy=accuracy), global_step


def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> EpochStats:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            running_loss += loss.item() * inputs.size(0)
            preds = outputs.argmax(dim=1)
            correct += preds.eq(targets).sum().item()
            total += targets.size(0)

    avg_loss = running_loss / total
    accuracy = correct / total * 100
    print(f"[evaluate] test_loss={avg_loss:.4f} acc={accuracy:.2f}%")
    return EpochStats(loss=avg_loss, accuracy=accuracy)
