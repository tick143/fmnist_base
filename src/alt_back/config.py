from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml


@dataclass
class ComponentConfig:
    target: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class DatasetConfig:
    root: str = "./data"
    download: bool = True
    augmentations: dict[str, Any] = field(default_factory=dict)
    target: str | None = None
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingConfig:
    epochs: int = 5
    batch_size: int = 64
    log_interval: int = 100
    device: str = "auto"
    seed: int | None = 42
    num_workers: int = 2


@dataclass
class WandbConfig:
    enabled: bool = False
    project: str = "alt-back"
    entity: str | None = None
    run_name: str | None = None
    tags: list[str] = field(default_factory=list)
    log_logits: bool = True
    log_probabilities: bool = True
    log_entropies: bool = True
    watch_model: bool = False


@dataclass
class Config:
    training: TrainingConfig = field(default_factory=TrainingConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ComponentConfig = field(default_factory=lambda: ComponentConfig(target="alt_back.models.simple.SimpleFashionCNN"))
    backward: ComponentConfig = field(default_factory=lambda: ComponentConfig(target="alt_back.backward.autograd.AutogradBackwardStrategy"))
    optimizer: ComponentConfig = field(
        default_factory=lambda: ComponentConfig(
            target="alt_back.optim.torch_optimizer.TorchOptimizerStrategy",
            params={"optimizer_class": "torch.optim.Adam", "lr": 1e-3},
        )
    )
    logging: WandbConfig = field(default_factory=WandbConfig)


def _merge_dicts(base: dict[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overrides.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), Mapping):
            merged[key] = _merge_dicts(base[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path) -> Config:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file '{config_path}' not found.")

    with config_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}

    base = Config()
    return Config(
        training=_dataclass_from_mapping(base.training, data.get("training", {})),
        dataset=_dataclass_from_mapping(base.dataset, data.get("dataset", {})),
        model=_dataclass_from_mapping(base.model, data.get("model", {})),
        backward=_dataclass_from_mapping(base.backward, data.get("backward", {})),
        optimizer=_dataclass_from_mapping(base.optimizer, data.get("optimizer", {})),
        logging=_dataclass_from_mapping(base.logging, data.get("logging", {})),
    )


def _dataclass_from_mapping(dc: Any, overrides: Mapping[str, Any]) -> Any:
    if not overrides:
        return dc
    merged = _merge_dicts(dc.__dict__, overrides)
    return dc.__class__(**merged)
