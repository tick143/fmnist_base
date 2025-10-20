from dataclasses import asdict, fields
from typing import Dict, Any

import torch
import wandb

from ..config import Config, WandbConfig
from ..visualization.trainer import TrainerConfig


class WandbLogger:
    """Log metrics and watch models with Weights & Biases."""

    def __init__(self, config: Config | TrainerConfig) -> None:
        if isinstance(config, TrainerConfig):
            logging_data = config.logging
            self.config = WandbConfig()
            for key, value in logging_data.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
            flat_config = asdict(config)
        else:
            self.config = config.logging
            flat_config = self._flatten_config(config)

        self.enabled = self.config.enabled
        self.run = None
        if self.enabled:
            self.run = wandb.init(
                project=self.config.project,
                name=self.config.run_name,
                tags=self.config.tags,
                config=flat_config,
            )

    def log(self, data: Dict[str, Any], step: int | None = None) -> None:
        if not self.enabled or self.run is None:
            return
        import wandb

        wandb.log(data, step=step)

    def watch(self, model: torch.nn.Module) -> None:
        if not self.enabled or self.run is None:
            return
        self.run.watch(model, log=self.config.watch_log, log_freq=self.config.watch_log_freq)

    def finish(self) -> None:
        if not self.enabled or self.run is None:
            return
        import wandb

        wandb.finish()
