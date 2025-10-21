from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import torch

from ..training.components import build_components
from ..training.loop import EpochStats, evaluate, train_one_epoch
from ..utils.logging import WandbLogger
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


def resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)


def train(config_path: str) -> Tuple[list[EpochStats], list[EpochStats]]:
    config = trainer_config_from_yaml(config_path)
    device = resolve_device(config.device)

    if config.dataset.seed is not None:
        torch.manual_seed(config.dataset.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.dataset.seed)

    model, backward_strategy, optimizer_strategy, loss_fn, train_loader, test_loader = build_components(config, device)

    print(
        f"[train] dataset_target={getattr(config, 'dataset_target', None)} | batches(train)={len(train_loader)} | batches(test)={len(test_loader)}"
    )

    wandb_logger = WandbLogger(config)
    global_step = 0
    train_history: list[EpochStats] = []
    eval_history: list[EpochStats] = []

    try:
        for epoch in range(1, config.epochs + 1):
            print(f"[train] starting epoch {epoch}/{config.epochs}")
            epoch_stats, global_step = train_one_epoch(
                model=model,
                dataloader=train_loader,
                backward_strategy=backward_strategy,
                optimizer_strategy=optimizer_strategy,
                device=device,
                loss_fn=loss_fn,
                log_interval=config.log_interval,
                epoch=epoch,
                logger=wandb_logger,
                wandb_cfg=wandb_logger.config if hasattr(wandb_logger, "config") else None,
                global_step=global_step,
            )
            train_history.append(epoch_stats)

            eval_stats = evaluate(
                model=model,
                dataloader=test_loader,
                device=device,
                loss_fn=loss_fn,
            )
            eval_history.append(eval_stats)

            if wandb_logger.enabled:
                wandb_logger.log(
                    {
                        "train/epoch_loss": epoch_stats.loss,
                        "train/epoch_accuracy": epoch_stats.accuracy,
                        "eval/loss": eval_stats.loss,
                        "eval/accuracy": eval_stats.accuracy,
                        "epoch": epoch,
                    },
                    step=global_step,
                )

    finally:
        wandb_logger.finish()

    return train_history, eval_history


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    print(f"Loaded config from {config_path}")
    train(args.config)


if __name__ == "__main__":
    main()
