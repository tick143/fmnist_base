from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict

from ..config import Config


class WandbLogger:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.enabled = config.logging.enabled
        self.run = None
        self._watch_requested = config.logging.watch_model

        if self.enabled:
            try:
                import wandb
            except ModuleNotFoundError as exc:
                raise RuntimeError("wandb logging requested but wandb is not installed") from exc

            run_config: Dict[str, Any] = {
                "training": asdict(config.training),
                "dataset": asdict(config.dataset),
                "model": asdict(config.model),
                "backward": asdict(config.backward),
                "optimizer": asdict(config.optimizer),
            }
            tags = config.logging.tags or []
            self.run = wandb.init(
                project=config.logging.project,
                entity=config.logging.entity,
                name=config.logging.run_name,
                config=run_config,
                tags=tags,
            )

    def log(self, data: Dict[str, Any], step: int | None = None) -> None:
        if not self.enabled or self.run is None:
            return
        import wandb

        wandb.log(data, step=step)

    def watch(self, model: Any) -> None:
        if not self.enabled or self.run is None or not self._watch_requested:
            return
        import wandb

        wandb.watch(model, log="all", log_freq=100)
        self._watch_requested = False

    def finish(self) -> None:
        if not self.enabled or self.run is None:
            return
        import wandb

        wandb.finish()
