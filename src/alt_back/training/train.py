from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Tuple

import torch
import torch.nn.functional as F

from ..config import (
    ComponentConfig,
    DatasetConfig as FashionDatasetConfig,
    TrainingConfig as FashionTrainingConfig,
)
from ..data.synthetic import create_dataloaders
from ..training.context import BatchContext
from ..training.loop import EpochStats, evaluate
from ..training.metrics import compute_prediction_metrics, compute_activation_entropies
from ..utils.activations import ActivationRecorder
from ..utils.logging import WandbLogger
from ..utils.imports import import_from_string
from ..visualization.trainer import trainer_config_from_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Fashion-MNIST with configurable learning rules.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to the YAML config file.",
    )
    return parser.parse_args()


def instantiate_component(component_cfg: ComponentConfig, **extra_kwargs: Any) -> Any:
    component_cls = import_from_string(component_cfg.target)
    kwargs = dict(component_cfg.params)
    kwargs.update(extra_kwargs)
    return component_cls(**kwargs)


def resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)


def train(config_path: str) -> tuple[list[EpochStats], list[EpochStats]]:
    config = trainer_config_from_yaml(config_path)
    device = resolve_device(config.device)

    if config.dataset.seed is not None:
        torch.manual_seed(config.dataset.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.dataset.seed)

    train_loader, test_loader = _resolve_dataloaders(config)
    print(
        f"[train] dataset_target={getattr(config, 'dataset_target', None)} | batches(train)={len(train_loader)} | batches(test)={len(test_loader)}"
    )

    model_component = ComponentConfig(target=config.model_target, params=config.model_params)
    model = instantiate_component(model_component)
    model.to(device)

    def _init_weights(module: torch.nn.Module) -> None:
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    model.apply(_init_weights)

    backward_component = ComponentConfig(target=config.backward_target, params=config.backward_params)
    backward_strategy: BackwardStrategy = instantiate_component(backward_component)

    optimizer_component = ComponentConfig(target=config.optimizer_target, params=config.optimizer_params)
    optimizer_strategy: OptimizerStrategy = instantiate_component(optimizer_component)
    optimizer_strategy.setup(model=model)

    loss_fn = torch.nn.CrossEntropyLoss()
    train_history: list[EpochStats] = []
    eval_history: list[EpochStats] = []
    wandb_logger = WandbLogger(config)
    # wandb_logger.watch(model)
    global_step = 0

    try:
        for epoch in range(1, config.epochs + 1):
            print(f"[train] starting epoch {epoch}/{config.epochs}")
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                if batch_idx == 0:
                    print(
                        f"[train] epoch {epoch} batch {batch_idx + 1}: inputs={inputs.shape} targets={targets.shape}"
                    )

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

                running_loss += loss.item() * inputs.size(0)
                preds = outputs.argmax(dim=1)
                correct += preds.eq(targets).sum().item()
                total += targets.size(0)

                if (global_step + 1) % config.log_interval == 0:
                    train_stats = EpochStats(loss=running_loss / total, accuracy=correct / total * 100)
                    eval_stats = evaluate(
                        model=model,
                        dataloader=test_loader,
                        device=device,
                        loss_fn=loss_fn,
                    )
                    print(
                        f"[step {global_step + 1}] train_loss={train_stats.loss:.4f} train_acc={train_stats.accuracy:.2f}% | "
                        f"test_loss={eval_stats.loss:.4f} test_acc={eval_stats.accuracy:.2f}%"
                    )
                    wandb_logger.log(
                        {
                            "train/epoch_loss": train_stats.loss,
                            "train/epoch_accuracy": train_stats.accuracy,
                            "eval/loss": eval_stats.loss,
                            "eval/accuracy": eval_stats.accuracy,
                        },
                        step=global_step,
                    )

                    # Detailed logging
                    probs = F.softmax(outputs.detach(), dim=1)
                    pred_metrics = compute_prediction_metrics(outputs.detach())
                    layer_entropies, network_entropy = compute_activation_entropies(context.activations)

                    log_payload = {
                        "train/loss": loss.item(),
                        "train/accuracy": preds.eq(targets).float().mean().item() * 100,
                        "train/prediction_entropy": pred_metrics["entropy"],
                        "train/network_entropy": network_entropy,
                    }

                    if config.logging.get("log_logits", True):
                        log_payload["train/logits/mean"] = outputs.detach().mean().item()
                        log_payload["train/logits/std"] = outputs.detach().std().item()

                    if config.logging.get("log_probabilities", True):
                        log_payload["train/probabilities/mean"] = probs.mean().item()
                        log_payload["train/probabilities/std"] = probs.std().item()

                    if config.logging.get("log_entropies", True):
                        for layer_name, entropy_value in layer_entropies.items():
                            sanitized = layer_name.replace(".", "/")
                            log_payload[f"train/layer_entropy/{sanitized}"] = entropy_value

                    for extra_key, extra_value in context.extras.items():
                        log_payload[f"train/{extra_key}"] = extra_value

                    wandb_logger.log(log_payload, step=global_step)

                    train_history.append(train_stats)
                    eval_history.append(eval_stats)

                global_step += 1

    finally:
        wandb_logger.finish()

    return train_history, eval_history


def _resolve_dataloaders(config) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    target = (getattr(config, "dataset_target", None) or "").strip()
    if not target:
        print("[train] using default synthetic dataloaders")
        return create_dataloaders(config.dataset)

    print(f"[train] attempting dataset_target='{target}'")
    try:
        loader_fn = import_from_string(target)
    except Exception:
        print(f"[train] failed to import {target}, falling back to synthetic dataloader")
        return create_dataloaders(config.dataset)

    raw_params = dict(getattr(config, "dataset_raw", {}) or {})

    # Special-case Fashion-MNIST helper which expects dataset/training configs
    if target.endswith("alt_back.data.fashion.dataloaders"):
        print(f"[train] building Fashion-MNIST dataloaders with params={raw_params}")
        root = raw_params.get("root", "./data")
        download = bool(raw_params.get("download", True))
        augmentations = raw_params.get("augmentations", {})
        dataset_cfg = FashionDatasetConfig(root=root, download=download, augmentations=augmentations)

        batch_size = getattr(config.dataset, "batch_size", 64)
        num_workers = getattr(config.dataset, "num_workers", 0)
        seed = getattr(config.dataset, "seed", None)
        training_cfg = FashionTrainingConfig(
            epochs=config.epochs,
            batch_size=batch_size,
            log_interval=config.log_interval,
            device=config.device,
            seed=seed,
            num_workers=num_workers,
        )
        print(
            f"[train] resolved Fashion configs: batch_size={training_cfg.batch_size}, num_workers={training_cfg.num_workers}, root={dataset_cfg.root}"
        )
        return loader_fn(dataset_cfg, training_cfg)

    try:
        result = loader_fn(**raw_params)
        if isinstance(result, tuple) and len(result) == 2:
            print("[train] custom dataloader returned loaders successfully")
            return result
    except Exception:
        print("[train] custom dataloader invocation failed, falling back to synthetic")
        pass

    return create_dataloaders(config.dataset)


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    print(f"Loaded config from {config_path}")
    train(args.config)


if __name__ == "__main__":
    main()
