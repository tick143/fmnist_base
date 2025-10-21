from __future__ import annotations

from typing import Any, Tuple

import torch
from torch.utils.data import DataLoader

from ..config import ComponentConfig, DatasetConfig as FashionDatasetConfig, TrainingConfig as FashionTrainingConfig
from ..data.synthetic import create_dataloaders
from ..utils.imports import import_from_string


def instantiate_component(component_cfg: ComponentConfig, **extra_kwargs: Any) -> Any:
    component_cls = import_from_string(component_cfg.target)
    kwargs = dict(component_cfg.params)
    kwargs.update(extra_kwargs)
    return component_cls(**kwargs)


def initialise_linear_layers(model: torch.nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)


def resolve_dataloaders(config: Any) -> Tuple[DataLoader, DataLoader]:
    target = str(getattr(config, "dataset_target", "") or "").strip()
    if not target:
        return create_dataloaders(config.dataset)

    try:
        loader_fn = import_from_string(target)
    except Exception:
        return create_dataloaders(config.dataset)

    raw_params = dict(getattr(config, "dataset_raw", {}) or {})

    if target.endswith("alt_back.data.fashion.dataloaders"):
        root = raw_params.get("root", "./data")
        download = bool(raw_params.get("download", True))
        augmentations = raw_params.get("augmentations", {})
        dataset_cfg = FashionDatasetConfig(root=root, download=download, augmentations=augmentations)

        batch_size = getattr(config.dataset, "batch_size", 64)
        num_workers = getattr(config.dataset, "num_workers", 0)
        seed = getattr(config.dataset, "seed", None)
        training_cfg = FashionTrainingConfig(
            epochs=getattr(config, "epochs", 1),
            batch_size=batch_size,
            log_interval=getattr(config, "log_interval", 100),
            device=getattr(config, "device", "cpu"),
            seed=seed,
            num_workers=num_workers,
        )
        return loader_fn(dataset_cfg, training_cfg)

    try:
        result = loader_fn(**raw_params)
        if isinstance(result, tuple) and len(result) == 2:
            return result
    except Exception:
        pass

    return create_dataloaders(config.dataset)


def compose_model_params(config: Any) -> dict[str, Any]:
    params = dict(getattr(config, "model_params", {}) or {})
    defaults = {
        "input_neurons": getattr(config, "dataset", None).num_features if getattr(config, "dataset", None) else None,
        "hidden_layers": getattr(config, "hidden_layers", None),
        "output_neurons": getattr(config, "output_neurons", None),
        "spike_threshold": getattr(config, "spike_threshold", None),
        "spike_temperature": getattr(config, "spike_temperature", None),
    }
    for key, value in defaults.items():
        if value is not None and key not in params:
            params[key] = value
    return params


def compose_backward_params(config: Any) -> dict[str, Any]:
    params = dict(getattr(config, "backward_params", {}) or {})
    possible_keys = (
        "release_rate",
        "reward_gain",
        "base_release",
        "decay",
        "temperature",
        "efficiency_bonus",
        "column_competition",
        "noise_std",
        "mass_budget",
        "signed_weights",
        "use_target_bonus",
        "target_gain",
        "affinity_strength",
        "affinity_decay",
        "affinity_temperature",
        "sign_consistency_strength",
        "sign_consistency_momentum",
        "push_rate",
        "suppress_rate",
        "step_scale",
        "energy_slope",
        "energy_momentum",
        "concentration_momentum",
        "loss_tolerance",
        "weight_clamp",
        "direction_mode",
    )
    for key in possible_keys:
        if key in params:
            continue
        if hasattr(config, key):
            params[key] = getattr(config, key)

    alias_map = {
        "use_target_bonus": "enable_target_bonus",
    }
    for source, target_key in alias_map.items():
        if target_key in params:
            continue
        if hasattr(config, source):
            params[target_key] = getattr(config, source)

    return params


def build_components(config: Any, device: torch.device):
    train_loader, test_loader = resolve_dataloaders(config)

    model_cfg = ComponentConfig(
        target=getattr(config, "model_target", "alt_back.models.spiking.TinySpikingNetwork"),
        params=compose_model_params(config),
    )
    model = instantiate_component(model_cfg)
    model.to(device)
    initialise_linear_layers(model)

    backward_cfg = ComponentConfig(
        target=getattr(config, "backward_target", "alt_back.backward.mass_redistribution.MassRedistributionBackwardStrategy"),
        params=compose_backward_params(config),
    )
    backward_strategy = instantiate_component(backward_cfg)

    optimizer_cfg = ComponentConfig(
        target=getattr(config, "optimizer_target", "alt_back.optim.null_optimizer.NullOptimizerStrategy"),
        params=dict(getattr(config, "optimizer_params", {}) or {}),
    )
    optimizer_strategy = instantiate_component(optimizer_cfg)
    optimizer_strategy.setup(model=model)

    loss_fn = torch.nn.CrossEntropyLoss()

    return model, backward_strategy, optimizer_strategy, loss_fn, train_loader, test_loader
