from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Tuple

import torch

from ..config import ComponentConfig, load_config
from ..data import fashion
from ..data.synthetic import create_dataloaders
from ..training.context import BatchContext
from ..training.loop import EpochStats, evaluate
from ..utils.activations import ActivationRecorder
from ..utils.logging import WandbLogger
from ..utils.imports import import_from_string
from ..visualization.trainer import trainer_config_from_yaml
from ..models.spiking import TinySpikingNetwork
from ..models.synthetic import SyntheticDenseNetwork
from ..backward.mass_redistribution import MassRedistributionBackwardStrategy
from ..backward.concentration import ConcentrationGradientBackwardStrategy
from ..optim.null_optimizer import NullOptimizerStrategy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Fashion-MNIST with configurable learning rules.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to the YAML config file.",
    )
    return parser.parse_args()


def instantiate_component(component_cfg: ComponentConfig, **extra_kwargs: Any) -> Any:
    component_cls = import_from_string(component_cfg.target)
    kwargs = dict(component_cfg.params)
    kwargs.update(extra_kwargs)
    return component_cls(**kwargs)


def resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)


from ..visualization.trainer import trainer_config_from_yaml

def train(config_path: str) -> tuple[list[EpochStats], list[EpochStats]]:
    config = trainer_config_from_yaml(config_path)
    device = resolve_device(config.device)

    if config.dataset.seed is not None:
        torch.manual_seed(config.dataset.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.dataset.seed)

    train_loader, test_loader = create_dataloaders(config.dataset)

    if config.model_type == "spiking":
        model = TinySpikingNetwork(
            input_neurons=config.dataset.num_features,
            hidden_layers=config.hidden_layers,
            output_neurons=2,  # Assuming binary classification for synthetic data
            spike_threshold=config.spike_threshold,
            spike_temperature=config.spike_temperature,
        )
    else:
        model = SyntheticDenseNetwork(
            input_neurons=config.dataset.num_features,
            hidden_layers=config.hidden_layers,
            output_neurons=2,  # Assuming binary classification for synthetic data
        )
    model.to(device)

    def _init_weights(module: torch.nn.Module) -> None:
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    model.apply(_init_weights)

    if config.learning_rule == "mass_redistribution":
        backward_strategy = MassRedistributionBackwardStrategy(
            release_rate=config.release_rate,
            reward_gain=config.reward_gain,
            base_release=config.base_release,
            decay=config.decay,
            temperature=config.temperature,
            efficiency_bonus=config.efficiency_bonus,
            column_competition=config.column_competition,
            noise_std=config.noise_std,
            mass_budget=config.mass_budget,
            signed_weights=config.signed_weights,
            enable_target_bonus=config.use_target_bonus,
            target_gain=config.target_gain,
            affinity_strength=config.affinity_strength,
            affinity_decay=config.affinity_decay,
            affinity_temperature=config.affinity_temperature,
            sign_consistency_strength=config.sign_consistency_strength,
            sign_consistency_momentum=config.sign_consistency_momentum,
        )
    else:
        backward_strategy = ConcentrationGradientBackwardStrategy(
            push_rate=config.push_rate,
            suppress_rate=config.suppress_rate,
            step_scale=config.step_scale,
            energy_slope=config.energy_slope,
            energy_momentum=config.energy_momentum,
            concentration_momentum=config.concentration_momentum,
            loss_tolerance=config.loss_tolerance,
            weight_clamp=config.weight_clamp,
            direction_mode=config.direction_mode,
        )

    optimizer_strategy = NullOptimizerStrategy()
    optimizer_strategy.setup(model=model)

    loss_fn = torch.nn.CrossEntropyLoss()
    train_history: list[EpochStats] = []
    eval_history: list[EpochStats] = []
    wandb_logger = WandbLogger(config)
    # wandb_logger.watch(model)
    global_step = 0

    try:
        for epoch in range(1, config.epochs + 1):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            # Create a new backward strategy for each epoch to reset its state
            if config.learning_rule == "mass_redistribution":
                backward_strategy = MassRedistributionBackwardStrategy(
                    release_rate=config.release_rate,
                    reward_gain=config.reward_gain,
                    base_release=config.base_release,
                    decay=config.decay,
                    temperature=config.temperature,
                    efficiency_bonus=config.efficiency_bonus,
                    column_competition=config.column_competition,
                    noise_std=config.noise_std,
                    mass_budget=config.mass_budget,
                    signed_weights=config.signed_weights,
                    enable_target_bonus=config.use_target_bonus,
                    target_gain=config.target_gain,
                    affinity_strength=config.affinity_strength,
                    affinity_decay=config.affinity_decay,
                    affinity_temperature=config.affinity_temperature,
                    sign_consistency_strength=config.sign_consistency_strength,
                    sign_consistency_momentum=config.sign_consistency_momentum,
                )
            else:
                backward_strategy = ConcentrationGradientBackwardStrategy(
                    push_rate=config.push_rate,
                    suppress_rate=config.suppress_rate,
                    step_scale=config.step_scale,
                    energy_slope=config.energy_slope,
                    energy_momentum=config.energy_momentum,
                    concentration_momentum=config.concentration_momentum,
                    loss_tolerance=config.loss_tolerance,
                    weight_clamp=config.weight_clamp,
                    direction_mode=config.direction_mode,
                )

            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(device), targets.to(device)

                optimizer_strategy.zero_grad(model=model)
                backward_strategy.zero_grad(model=model)

                with ActivationRecorder(model) as recorder:
                    outputs = model(inputs)
                    loss = loss_fn(outputs, targets)

                    context = BatchContext(
                        epoch=epoch,
                        batch_idx=batch_idx,
                        model=model,
                        inputs=inputs,
                        targets=targets,
                        outputs=outputs,
                        loss=loss,
                        device=device,
                        activations=dict(recorder.records),
                    )

                    backward_strategy.backward(context)
                    optimizer_strategy.step(context)

                running_loss += loss.item() * inputs.size(0)
                preds = outputs.argmax(dim=1)
                correct += preds.eq(targets).sum().item()
                total += targets.size(0)

                if (global_step + 1) % config.log_interval == 0:
                    train_stats = EpochStats(loss=running_loss / total, accuracy=correct / total * 100)
                    eval_stats = evaluate(
                        model=model,
                        dataloader=test_loader,
                        device=device,
                        loss_fn=loss_fn,
                    )
                    print(
                        f"[step {global_step + 1}] train_loss={train_stats.loss:.4f} train_acc={train_stats.accuracy:.2f}% | "
                        f"test_loss={eval_stats.loss:.4f} test_acc={eval_stats.accuracy:.2f}%"
                    )
                    wandb_logger.log(
                        {
                            "train/epoch_loss": train_stats.loss,
                            "train/epoch_accuracy": train_stats.accuracy,
                            "eval/loss": eval_stats.loss,
                            "eval/accuracy": eval_stats.accuracy,
                        },
                        step=global_step,
                    )
                    train_history.append(train_stats)
                    eval_history.append(eval_stats)

                global_step += 1

    finally:
        wandb_logger.finish()

    return train_history, eval_history


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    print(f"Loaded config from {config_path}")
    train(args.config)


if __name__ == "__main__":
    main()
