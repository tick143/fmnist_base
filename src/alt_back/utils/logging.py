from dataclasses import asdict, fields
from importlib import import_module
from typing import Dict, Any
import warnings

import torch

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
        self._wandb = None
        if self.enabled:
            try:
                with warnings.catch_warnings():
                    try:
                        from pydantic._internal._generate_schema import UnsupportedFieldAttributeWarning  # type: ignore

                        warnings.filterwarnings("ignore", category=UnsupportedFieldAttributeWarning)
                    except Exception:  # pragma: no cover - defensive fallback if internals change
                        warnings.filterwarnings(
                            "ignore",
                            message="The 'repr' attribute with value False was provided to the `Field\(\)` function",
                        )
                        warnings.filterwarnings(
                            "ignore",
                            message="The 'frozen' attribute with value True was provided to the `Field\(\)` function",
                        )
                    module = import_module("wandb")
                self._wandb = module
            except Exception:  # pragma: no cover - wandb not available or import failed
                warnings.warn("wandb is not available; disabling Weights & Biases logging.")
                self.enabled = False
                self.config.enabled = False
                return

            self.run = self._wandb.init(
                project=self.config.project,
                name=self.config.run_name,
                tags=self.config.tags,
                config=flat_config,
            )

    def log(self, data: Dict[str, Any], step: int | None = None) -> None:
        if not self.enabled or self.run is None or self._wandb is None:
            return
        self._wandb.log(data, step=step)

    def watch(self, model: torch.nn.Module) -> None:
        if not self.enabled or self.run is None or self._wandb is None:
            return
        self.run.watch(model, log=self.config.watch_log, log_freq=self.config.watch_log_freq)

    def finish(self) -> None:
        if not self.enabled or self.run is None or self._wandb is None:
            return
        self._wandb.finish()
