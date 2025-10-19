from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Tuple

import torch
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class SyntheticDatasetConfig:
    """Configuration for the thresholded synthetic dataset."""

    num_features: int = 5
    threshold: float = 1.0
    num_train: int = 2048
    num_test: int = 1024
    seed: int = 7
    batch_size: int = 32
    num_workers: int = 0
    shuffle: bool = True
    feature_min: float = -1.0
    feature_max: float = 2.0
    noise_std: float = 0.2


def _generate_balanced_split(
    num_samples: int,
    num_features: int,
    threshold: float,
    generator: torch.Generator,
    feature_min: float,
    feature_max: float,
    noise_std: float,
) -> TensorDataset:
    """Create a perfectly balanced dataset subject to a sum threshold rule."""
    half = num_samples // 2
    remainder = num_samples - half * 2

    if feature_max <= feature_min:
        msg = f"feature_max ({feature_max}) must be greater than feature_min ({feature_min})."
        raise ValueError(msg)
    positives = _sample_with_condition(
        required=half + remainder,
        num_features=num_features,
        condition=lambda batch: batch.sum(dim=1) > threshold,
        generator=generator,
        feature_min=feature_min,
        feature_max=feature_max,
        noise_std=noise_std,
    )
    negatives = _sample_with_condition(
        required=half,
        num_features=num_features,
        condition=lambda batch: batch.sum(dim=1) <= threshold,
        generator=generator,
        feature_min=feature_min,
        feature_max=feature_max,
        noise_std=noise_std,
    )

    data = torch.cat([positives[: half + remainder], negatives[:half]], dim=0)
    targets = torch.cat(
        [
            torch.ones(half + remainder, dtype=torch.long),
            torch.zeros(half, dtype=torch.long),
        ],
        dim=0,
    )

    perm = torch.randperm(num_samples, generator=generator)
    data = data[perm]
    targets = targets[perm]
    return TensorDataset(data, targets)


def _sample_with_condition(
    required: int,
    num_features: int,
    condition: Callable[[torch.Tensor], torch.Tensor],
    generator: torch.Generator,
    feature_min: float,
    feature_max: float,
    noise_std: float,
    max_iters: int = 20,
) -> torch.Tensor:
    """Sample uniformly until enough items satisfy the predicate."""
    batch_size = max(32, required * 2)
    collected = []
    total = 0
    iterations = 0

    while total < required and iterations < max_iters:
        base = torch.rand(batch_size, num_features, generator=generator)
        batch = base * (feature_max - feature_min) + feature_min
        if noise_std > 0:
            batch = batch + torch.randn(batch_size, num_features, generator=generator) * noise_std
        mask = condition(batch)
        if mask.any():
            selected = batch[mask]
            collected.append(selected)
            total += selected.size(0)
        iterations += 1

    if total < required:
        # Fallback to rejection sampling one by one to guarantee completion.
        while total < required:
            base = torch.rand(1, num_features, generator=generator)
            sample = base * (feature_max - feature_min) + feature_min
            if noise_std > 0:
                sample = sample + torch.randn(1, num_features, generator=generator) * noise_std
            if condition(sample):
                collected.append(sample)
                total += 1

    return torch.cat(collected, dim=0)


def create_dataloaders(
    cfg: SyntheticDatasetConfig | dict | None = None,
    **overrides: object,
) -> Tuple[DataLoader, DataLoader]:
    """Construct train and test dataloaders for the synthetic dataset."""
    if cfg is None:
        cfg = SyntheticDatasetConfig(**overrides)
    elif isinstance(cfg, dict):
        merged = {**cfg, **overrides}
        cfg = SyntheticDatasetConfig(**merged)
    elif overrides:
        cfg = replace(cfg, **overrides)

    generator = torch.Generator().manual_seed(cfg.seed)
    train_dataset = _generate_balanced_split(
        num_samples=cfg.num_train,
        num_features=cfg.num_features,
        threshold=cfg.threshold,
        generator=generator,
        feature_min=cfg.feature_min,
        feature_max=cfg.feature_max,
        noise_std=cfg.noise_std,
    )
    test_dataset = _generate_balanced_split(
        num_samples=cfg.num_test,
        num_features=cfg.num_features,
        threshold=cfg.threshold,
        generator=generator,
        feature_min=cfg.feature_min,
        feature_max=cfg.feature_max,
        noise_std=cfg.noise_std,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=cfg.shuffle,
        num_workers=cfg.num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
    )
    return train_loader, test_loader
