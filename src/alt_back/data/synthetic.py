from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Tuple

import torch
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class SyntheticDatasetConfig:
    """Configuration for the thresholded synthetic dataset."""

    num_features: int = 5
    threshold: float = 2.5
    num_train: int = 1024
    num_test: int = 512
    seed: int = 0
    batch_size: int = 32
    num_workers: int = 0
    shuffle: bool = True


def _generate_balanced_split(
    num_samples: int,
    num_features: int,
    threshold: float,
    generator: torch.Generator,
) -> TensorDataset:
    """Create a perfectly balanced dataset subject to a sum threshold rule."""
    half = num_samples // 2
    remainder = num_samples - half * 2

    positives = _sample_with_condition(
        required=half + remainder,
        num_features=num_features,
        condition=lambda batch: batch.sum(dim=1) > threshold,
        generator=generator,
    )
    negatives = _sample_with_condition(
        required=half,
        num_features=num_features,
        condition=lambda batch: batch.sum(dim=1) <= threshold,
        generator=generator,
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
    max_iters: int = 10,
) -> torch.Tensor:
    """Sample uniformly until enough items satisfy the predicate."""
    batch_size = max(32, required * 2)
    collected = []
    total = 0
    iterations = 0

    while total < required and iterations < max_iters:
        batch = torch.rand(batch_size, num_features, generator=generator)
        mask = condition(batch)
        if mask.any():
            selected = batch[mask]
            collected.append(selected)
            total += selected.size(0)
        iterations += 1

    if total < required:
        # Fallback to rejection sampling one by one to guarantee completion.
        while total < required:
            sample = torch.rand(1, num_features, generator=generator)
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
    )
    test_dataset = _generate_balanced_split(
        num_samples=cfg.num_test,
        num_features=cfg.num_features,
        threshold=cfg.threshold,
        generator=generator,
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
