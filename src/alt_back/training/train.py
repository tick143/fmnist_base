from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from ..config import Config, ComponentConfig, load_config
from ..data import fashion
from ..training.loop import EpochStats, evaluate, train_one_epoch
from ..utils.logging import WandbLogger
from ..utils.imports import import_from_string


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


def train(config: Config) -> tuple[list[EpochStats], list[EpochStats]]:
    device = resolve_device(config.training.device)
    if config.training.seed is not None:
        torch.manual_seed(config.training.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.training.seed)

    if config.dataset.target:
        dataloader_factory = import_from_string(config.dataset.target)
        factory_args = dict(config.dataset.params)
        factory_args.setdefault("batch_size", config.training.batch_size)
        factory_args.setdefault("num_workers", config.training.num_workers)
        factory_args.setdefault("seed", config.training.seed or 0)
        train_loader, test_loader = dataloader_factory(factory_args)
    else:
        train_loader, test_loader = fashion.dataloaders(config.dataset, config.training)

    model = instantiate_component(config.model)
    model.to(device)

    def _init_weights(module: torch.nn.Module) -> None:
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    model.apply(_init_weights)

    backward_strategy = instantiate_component(config.backward)
    optimizer_strategy = instantiate_component(config.optimizer)
    optimizer_strategy.setup(model=model)

    loss_fn = torch.nn.CrossEntropyLoss()
    train_history: list[EpochStats] = []
    eval_history: list[EpochStats] = []
    wandb_logger = WandbLogger(config)
    wandb_logger.watch(model)
    global_step = 0

    try:
        for epoch in range(1, config.training.epochs + 1):
            print(f"\nEpoch {epoch}/{config.training.epochs} -> device={device}")
            train_stats, global_step = train_one_epoch(
                model=model,
                dataloader=train_loader,
                backward_strategy=backward_strategy,
                optimizer_strategy=optimizer_strategy,
                device=device,
                loss_fn=loss_fn,
                log_interval=config.training.log_interval,
                epoch=epoch,
                logger=wandb_logger,
                wandb_cfg=config.logging,
                global_step=global_step,
            )
            eval_stats = evaluate(
                model=model,
                dataloader=test_loader,
                device=device,
                loss_fn=loss_fn,
            )

            print(
                f"[epoch {epoch}] train_loss={train_stats.loss:.4f} train_acc={train_stats.accuracy:.2f}% | "
                f"test_loss={eval_stats.loss:.4f} test_acc={eval_stats.accuracy:.2f}%"
            )

            if wandb_logger.enabled:
                wandb_logger.log(
                    {
                        "epoch": epoch,
                        "train/epoch_loss": train_stats.loss,
                        "train/epoch_accuracy": train_stats.accuracy,
                        "eval/loss": eval_stats.loss,
                        "eval/accuracy": eval_stats.accuracy,
                    },
                    step=global_step,
                )

            train_history.append(train_stats)
            eval_history.append(eval_stats)
    finally:
        wandb_logger.finish()
    return train_history, eval_history


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    config_path = Path(args.config).resolve()
    print(f"Loaded config from {config_path}")
    train(config)


if __name__ == "__main__":
    main()
