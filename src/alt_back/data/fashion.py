from __future__ import annotations

from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from ..config import DatasetConfig, TrainingConfig


def build_transforms(augmentations: dict | None = None) -> tuple[transforms.Compose, transforms.Compose]:
    augmentations = augmentations or {}

    train_transforms: list[object] = [transforms.ToTensor()]
    test_transforms: list[object] = [transforms.ToTensor()]

    if augmentations.get("random_rotation"):
        degrees = float(augmentations["random_rotation"])
        train_transforms.append(transforms.RandomRotation(degrees))

    if augmentations.get("random_horizontal_flip"):
        probability = float(augmentations["random_horizontal_flip"])
        train_transforms.append(transforms.RandomHorizontalFlip(probability))

    normalize_cfg = augmentations.get("normalize", {"mean": [0.5], "std": [0.5]})
    mean = normalize_cfg.get("mean", [0.5])
    std = normalize_cfg.get("std", [0.5])
    normalization = transforms.Normalize(mean=mean, std=std)

    train_transforms.append(normalization)
    test_transforms.append(normalization)

    train_transform = transforms.Compose(train_transforms)
    test_transform = transforms.Compose(test_transforms)
    return train_transform, test_transform


def dataloaders(
    dataset_cfg: DatasetConfig,
    training_cfg: TrainingConfig,
) -> tuple[DataLoader, DataLoader]:
    root = Path(dataset_cfg.root)
    root.mkdir(parents=True, exist_ok=True)

    train_transform, test_transform = build_transforms(dataset_cfg.augmentations)

    train_dataset = datasets.FashionMNIST(
        root=str(root),
        train=True,
        transform=train_transform,
        download=dataset_cfg.download,
    )

    test_dataset = datasets.FashionMNIST(
        root=str(root),
        train=False,
        transform=test_transform,
        download=dataset_cfg.download,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=training_cfg.batch_size,
        shuffle=True,
        num_workers=training_cfg.num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=training_cfg.batch_size,
        shuffle=False,
        num_workers=training_cfg.num_workers,
        pin_memory=True,
    )

    return train_loader, test_loader
