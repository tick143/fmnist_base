from __future__ import annotations

import argparse
import itertools
import math
from copy import deepcopy
from typing import Any, Dict, Iterable, Iterator, Sequence, Tuple

from ..config import Config, load_config

TARGET_TRAINING_STEPS = 1500

# Sweep ranges are centred on the baseline values specified in configs/tiny_spiking.yaml.
# Adjust as needed by editing this mapping or by filtering via --limit.
RAW_BACKWARD_SWEEP_GRID: Dict[str, Sequence[Any] | Any] = {
    "release_rate": (0.15, 0.5, 0.75),
    "reward_gain": (0, 1, 2.5),
    "base_release": (0.0, 0.05),
    "decay": (0.05, 0.15),
    "temperature": (1.0),
    "efficiency_bonus": (0.1, 0.2, 0.4),
    "column_competition": (0.01),
    "noise_std": (0, 0.1),
    "mass_budget": (5.0, 10.0),
    "signed_weights": (True),
    "enable_target_bonus": (True),
    "target_gain": (0.3, 0.7),
    "affinity_strength": (0.05, 0.2),
    "affinity_decay": (0, 0.99),
    "affinity_temperature": (1.5),
}

def _normalise_values(values: Sequence[Any] | Any) -> Tuple[Any, ...]:
    if isinstance(values, (list, tuple)):
        result = tuple(values)
    elif hasattr(values, "__iter__") and not isinstance(values, (str, bytes, dict)):
        try:
            result = tuple(values)
        except TypeError:
            result = (values,)
    else:
        result = (values,)

    if not result:
        msg = "Sweep parameter lists cannot be empty."
        raise ValueError(msg)
    return result


BACKWARD_SWEEP_GRID: Dict[str, Tuple[Any, ...]] = {
    key: _normalise_values(values) for key, values in RAW_BACKWARD_SWEEP_GRID.items()
}

HYPERPARAM_KEYS: Sequence[str] = tuple(BACKWARD_SWEEP_GRID.keys())


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a full-matrix hyperparameter sweep for the tiny spiking demo.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/tiny_spiking.yaml",
        help="Base YAML config to clone for each sweep member.",
    )
    parser.add_argument(
        "--project",
        type=str,
        default=None,
        help="Optional Weights & Biases project override (defaults to config value).",
    )
    parser.add_argument(
        "--entity",
        type=str,
        default=None,
        help="Optional Weights & Biases entity override (defaults to config value).",
    )
    parser.add_argument(
        "--run-prefix",
        type=str,
        default="tiny-spiking",
        help="Prefix applied to generated WandB run names.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of sweep members to launch (applies to both dry-runs and executions).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List planned sweep members without launching training jobs.",
    )
    return parser.parse_args()


def _format_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        formatted = f"{value:.3f}".rstrip("0").rstrip(".")
        return formatted if formatted else "0"
    return str(value)


def _compute_epoch_budget(config: Config, target_steps: int) -> int:
    batch_size = config.training.batch_size
    dataset_params = config.dataset.params or {}
    num_train = dataset_params.get("num_train")

    if not isinstance(num_train, int) or num_train <= 0 or batch_size <= 0:
        return max(config.training.epochs, target_steps)

    steps_per_epoch = math.ceil(num_train / batch_size)
    if steps_per_epoch <= 0:
        return max(config.training.epochs, target_steps)

    return max(config.training.epochs, math.ceil(target_steps / steps_per_epoch))


def _iter_hyperparam_grid() -> Iterator[Dict[str, Any]]:
    values = (BACKWARD_SWEEP_GRID[key] for key in HYPERPARAM_KEYS)
    for combo in itertools.product(*values):
        yield dict(zip(HYPERPARAM_KEYS, combo, strict=False))


def _apply_hyperparams(base_config: Config, hyperparams: Dict[str, Any], project: str | None, entity: str | None, run_name: str) -> Config:
    config = deepcopy(base_config)

    config.logging.enabled = True
    if project is not None:
        config.logging.project = project
    if entity is not None:
        config.logging.entity = entity
    config.logging.run_name = run_name

    tags = list(config.logging.tags)
    tags.extend(["tiny-spiking", "mass-redistribution", "sweep"])
    config.logging.tags = list(dict.fromkeys(tags))

    config.training.epochs = _compute_epoch_budget(config, TARGET_TRAINING_STEPS)
    config.backward.params.update(hyperparams)
    return config


def _iter_with_limit(limit: int | None) -> Iterable[Dict[str, Any]]:
    iterator = _iter_hyperparam_grid()
    if limit is None:
        return iterator
    return itertools.islice(iterator, limit)


def _preview_hyperparams(limit: int | None) -> None:
    total_members = math.prod(len(BACKWARD_SWEEP_GRID[key]) for key in HYPERPARAM_KEYS)
    preview_budget = limit or min(total_members, 50)
    print(f"Full grid size: {total_members} sweep members")
    print(f"Previewing first {preview_budget} combinations:")

    for idx, hyperparams in enumerate(_iter_with_limit(preview_budget), start=1):
        formatted = ", ".join(f"{key}={_format_value(hyperparams[key])}" for key in HYPERPARAM_KEYS)
        print(f"  [{idx:04d}] {formatted}")

    if preview_budget < total_members:
        print("  ... (use --limit to preview or run a subset)")


def main() -> None:
    args = _parse_args()
    base_config = load_config(args.config)

    if args.dry_run:
        _preview_hyperparams(args.limit)
        return

    from ..training.train import train

    total_members = math.prod(len(BACKWARD_SWEEP_GRID[key]) for key in HYPERPARAM_KEYS)
    planned_total = total_members if args.limit is None else min(args.limit, total_members)

    if planned_total == 0:
        print("No sweep members to launch (limit=0).")
        return

    index_width = max(4, len(str(planned_total)))
    print(f"Launching {planned_total} sweep members (subset of {total_members} total combinations).")

    for idx, hyperparams in enumerate(_iter_with_limit(args.limit), start=1):
        run_name = f"{args.run_prefix}-{idx:0{index_width}d}"
        config = _apply_hyperparams(base_config, hyperparams, args.project, args.entity, run_name)

        snapshot = ", ".join(f"{key}={_format_value(hyperparams[key])}" for key in HYPERPARAM_KEYS)
        print(f"\n[{idx}/{planned_total}] run_name={run_name}")
        print(f"  hyperparams: {snapshot}")

        train(config)


if __name__ == "__main__":
    main()
