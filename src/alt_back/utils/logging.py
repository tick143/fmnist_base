from dataclasses import asdict, fields
from typing import Dict, Any

import torch
import wandb

from ..config import Config, WandbConfig


class WandbLogger:
    """Log metrics and watch models with Weights & Biases."""

    def __init__(self, config: Config | object) -> None:
        logging_section = getattr(config, "logging", None)

        if isinstance(logging_section, dict):
            self.config = WandbConfig()
            for key, value in logging_section.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
            flat_config = asdict(config)
        else:
            assert isinstance(config, Config)
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
