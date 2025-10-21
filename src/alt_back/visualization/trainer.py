from __future__ import annotations

from dataclasses import dataclass, field, replace, fields, asdict
from typing import Any, Dict, Iterable

import torch
import yaml

from ..data.synthetic import SyntheticDatasetConfig
from ..training.components import build_components
from ..training.loop import run_training_batch


LEARNING_RULE_LABELS: Dict[str, str] = {
    "mass_redistribution": "Mass Redistribution",
    "concentration": "Concentration Gradient",
}

MODEL_TYPE_LABELS: Dict[str, str] = {
    "spiking": "Spiking (Tiny Network)",
    "dense": "Dense (Feedforward)",
}

CONCENTRATION_DIRECTION_LABELS: Dict[str, str] = {
    "outputs_minus_inputs": "Outputs - Inputs",
    "inputs_minus_outputs": "Inputs - Outputs (flip)",
    "weight_sign": "Weight Sign",
    "raw_mean": "Raw Mean (no sign)",
}


def _option_list(labels: Dict[str, str]) -> list[dict[str, str]]:
    return [{"key": key, "label": value} for key, value in labels.items()]


@dataclass
class TrainerConfig:
    """Bundle configuration for the interactive tiny spiking trainer."""

    epochs: int = 1
    log_interval: int = 10
    dataset: SyntheticDatasetConfig = field(
        default_factory=lambda: SyntheticDatasetConfig(
            num_features=5,
            threshold=1.0,
            num_train=4096,
            num_test=1024,
            seed=13,
            batch_size=64,
            feature_min=-1.5,
            feature_max=2.5,
            noise_std=0.0,
        )
    )
    device: str = "cpu"
    model_type: str = "spiking"
    learning_rule: str = "mass_redistribution"
    # If provided, overrides the default synthetic dataset loader.
    # Example: "alt_back.data.fashion.dataloaders"
    dataset_target: str | None = None
    # Raw params from YAML dataset.params for custom loaders
    dataset_raw: dict = field(default_factory=dict)
    release_rate: float = 0.25
    reward_gain: float = 0.6
    base_release: float = 0.1
    decay: float = 0.05
    temperature: float = 1.0
    efficiency_bonus: float = 0.4
    column_competition: float = 0.3
    noise_std: float = 0.01
    mass_budget: float = 5.0
    spike_threshold: float = 0.1
    spike_temperature: float = 0.3
    hidden_layers: list[int] = field(default_factory=lambda: [12, 12, 8])
    output_neurons: int = 2
    signed_weights: bool = True
    use_target_bonus: bool = True
    target_gain: float = 0.5
    affinity_strength: float = 0.1
    affinity_decay: float = 0.99
    affinity_temperature: float = 1.5
    sign_consistency_strength: float = 0.2
    sign_consistency_momentum: float = 0.9
    push_rate: float = 0.3
    suppress_rate: float = 0.1
    step_scale: float = 0.02
    energy_slope: float = 1.4
    energy_momentum: float = 0.5
    concentration_momentum: float = 0.85
    loss_tolerance: float = 1e-5
    weight_clamp: float | None = 6.5
    direction_mode: str = "outputs_minus_inputs"
    snapshot_interval: int = 1
    evaluate_interval: int = 1
    logging: dict = field(default_factory=dict)
    model_target: str = "alt_back.models.spiking.TinySpikingNetwork"
    model_params: dict[str, Any] = field(default_factory=dict)
    backward_target: str = "alt_back.backward.mass_redistribution.MassRedistributionBackwardStrategy"
    backward_params: dict[str, Any] = field(default_factory=dict)
    optimizer_target: str = "alt_back.optim.null_optimizer.NullOptimizerStrategy"
    optimizer_params: dict[str, Any] = field(default_factory=dict)


class TinySpikingTrainer:
    """Perform batched updates with mass redistribution and expose step snapshots."""

    def __init__(self, config: TrainerConfig | None = None) -> None:
        self.config = config or TrainerConfig()
        self.device = torch.device(self.config.device)

        self.config.model_type = (self.config.model_type or "spiking").lower()
        self.model_type = self._normalise_choice(self.config.model_type, MODEL_TYPE_LABELS.keys(), default="spiking")

        self.config.direction_mode = self._normalise_choice(
            getattr(self.config, "direction_mode", "outputs_minus_inputs"),
            CONCENTRATION_DIRECTION_LABELS.keys(),
            default="outputs_minus_inputs",
        )
        self.direction_mode = self.config.direction_mode

        initial_rule = (self.config.learning_rule or "mass_redistribution").lower()
        self.learning_rule = self._normalise_choice(initial_rule, LEARNING_RULE_LABELS.keys(), default="mass_redistribution")

        self._synchronise_component_targets()
        (
            self.model,
            self.backward_strategy,
            self.optimizer_strategy,
            self.loss_fn,
            self.train_loader,
            self.test_loader,
        ) = build_components(self.config, self.device)
        self._train_iter = iter(self.train_loader)

        self.global_step = 0
        self.snapshot_interval = max(int(self.config.snapshot_interval), 1)
        self.evaluate_interval = max(int(self.config.evaluate_interval), 1)
        self._last_visuals: dict[str, Any] = self._empty_visuals()
        self._last_eval: dict[str, float] | None = None

    def options(self) -> Dict[str, list[dict[str, str]]]:
        return {
            "learning_rules": _option_list(LEARNING_RULE_LABELS),
            "model_types": _option_list(MODEL_TYPE_LABELS),
            "direction_modes": _option_list(CONCENTRATION_DIRECTION_LABELS),
        }

    @staticmethod
    def _normalise_choice(value: str, valid: Iterable[str], default: str) -> str:
        normalized = (value or default).lower()
        if normalized not in valid:
            return default
        return normalized

    def _synchronise_component_targets(self) -> None:
        if self.model_type == "dense":
            self.config.model_target = "alt_back.models.synthetic.SyntheticDenseNetwork"
        else:
            self.config.model_target = "alt_back.models.spiking.TinySpikingNetwork"

        rule = self._normalise_choice(self.learning_rule, LEARNING_RULE_LABELS.keys(), default="mass_redistribution")
        self.learning_rule = rule
        self.config.learning_rule = rule
        self.config.direction_mode = self._normalise_choice(
            getattr(self.config, "direction_mode", self.direction_mode),
            CONCENTRATION_DIRECTION_LABELS.keys(),
            default="outputs_minus_inputs",
        )
        self.direction_mode = self.config.direction_mode
        if rule == "concentration":
            self.config.backward_target = "alt_back.backward.concentration.ConcentrationGradientBackwardStrategy"
        else:
            self.config.backward_target = "alt_back.backward.mass_redistribution.MassRedistributionBackwardStrategy"

    def _rebuild_components(self) -> None:
        self._synchronise_component_targets()
        (
            self.model,
            self.backward_strategy,
            self.optimizer_strategy,
            self.loss_fn,
            self.train_loader,
            self.test_loader,
        ) = build_components(self.config, self.device)
        self._train_iter = iter(self.train_loader)

    def reset(self, seed: int | None = None) -> None:
        """Reset data streams and model parameters."""
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            self.config.dataset.seed = seed

        self._rebuild_components()
        self.global_step = 0
        self._last_visuals = self._empty_visuals()
        self._last_eval = None

    def _empty_visuals(self) -> Dict[str, Any]:
        return {
            "inputs": [],
            "targets": [],
            "predictions": [],
            "logits": [],
            "hidden_preact": [],
            "hidden_spike_rates": [],
            "weights": {},
            "weight_deltas": {},
        }

    def _snapshot_weights(self) -> Dict[str, Dict[str, list[list[float]] | list[float]]]:
        snapshot: Dict[str, Dict[str, list[list[float]] | list[float]]] = {}
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                snapshot[name] = {
                    "weight": module.weight.detach().cpu().tolist(),
                    "bias": module.bias.detach().cpu().tolist() if module.bias is not None else [],
                }
        return snapshot

    def _compute_deltas(
        self,
        before: Dict[str, Dict[str, list[list[float]] | list[float]]],
        after: Dict[str, Dict[str, list[list[float]] | list[float]]],
    ) -> Dict[str, Dict[str, list[list[float]] | list[float]]]:
        deltas: Dict[str, Dict[str, list[list[float]] | list[float]]] = {}
        for layer, weights in after.items():
            prev = before.get(layer)
            if prev is None:
                continue
            weight_before = torch.tensor(prev["weight"])
            weight_after = torch.tensor(weights["weight"])
            bias_before = torch.tensor(prev["bias"])
            bias_after = torch.tensor(weights["bias"])
            deltas[layer] = {
                "weight": (weight_after - weight_before).tolist(),
                "bias": (bias_after - bias_before).tolist(),
            }
        return deltas

    def step(self) -> Dict[str, Any]:
        """Perform a single optimisation step and return a rich snapshot for visualisation."""
        self.model.train()
        try:
            batch = next(self._train_iter)
        except StopIteration:
            self._train_iter = iter(self.train_loader)
            batch = next(self._train_iter)

        capture_snapshot = (self.global_step % self.snapshot_interval) == 0
        before = self._snapshot_weights() if capture_snapshot else None
        inputs, targets = batch
        context, outputs, loss = run_training_batch(
            model=self.model,
            inputs=inputs,
            targets=targets,
            backward_strategy=self.backward_strategy,
            optimizer_strategy=self.optimizer_strategy,
            device=self.device,
            loss_fn=self.loss_fn,
            epoch=0,
            batch_idx=self.global_step,
        )

        preds = outputs.argmax(dim=1)
        accuracy = (preds == context.targets).float().mean().item()
        extras = {key: float(value) for key, value in context.extras.items()}

        if capture_snapshot:
            after = self._snapshot_weights()
            deltas = self._compute_deltas(before, after) if before is not None else {}
            hidden_preacts = [tensor.detach().cpu().tolist() for tensor in self.model.last_hidden_preacts]
            spike_rates = [tensor.detach().cpu().tolist() for tensor in self.model.last_hidden_spikes]
            inputs_list = context.inputs.detach().cpu().tolist()
            targets_list = context.targets.detach().cpu().tolist()
            preds_list = preds.detach().cpu().tolist()
            logits_list = outputs.detach().cpu().tolist()

            visuals = {
                "inputs": inputs_list,
                "targets": targets_list,
                "predictions": preds_list,
                "logits": logits_list,
                "hidden_preact": hidden_preacts,
                "hidden_spike_rates": spike_rates,
                "weights": after,
                "weight_deltas": deltas,
            }
            self._last_visuals = visuals
        else:
            visuals = self._last_visuals

        should_evaluate = (self.global_step % self.evaluate_interval) == 0
        if should_evaluate:
            evaluation = self.evaluate()
            self._last_eval = evaluation
        else:
            evaluation = self._last_eval

        result = {
            "step": self.global_step,
            "loss": float(loss.item()),
            "batch_accuracy": float(accuracy * 100.0),
            "predictions": visuals["predictions"],
            "targets": visuals["targets"],
            "inputs": visuals["inputs"],
            "logits": visuals["logits"],
            "hidden_preact": visuals["hidden_preact"],
            "hidden_spike_rates": visuals["hidden_spike_rates"],
            "weights": visuals["weights"],
            "weight_deltas": visuals["weight_deltas"],
            "extras": extras,
            "snapshot_captured": capture_snapshot,
            "eval": evaluation,
            "evaluation_captured": should_evaluate,
        }

        self.global_step += 1
        return result

    def evaluate(self) -> Dict[str, float]:
        """Compute loss and accuracy on the held-out test dataset."""
        self.model.eval()
        loss_total = 0.0
        correct = 0
        count = 0

        with torch.no_grad():
            for inputs, targets in self.test_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)
                loss_total += loss.item() * targets.size(0)
                predictions = outputs.argmax(dim=1)
                correct += (predictions == targets).sum().item()
                count += targets.size(0)

        loss_avg = loss_total / max(count, 1)
        accuracy = correct / max(count, 1) * 100.0
        print(f"[evaluate] test_loss={loss_avg:.4f} acc={accuracy:.2f}%")
        return {"loss": float(loss_avg), "accuracy": float(accuracy)}

    def topology(self) -> Dict[str, Any]:
        """Describe the network layout for client visualisations."""
        hidden_layers = list(self.model.hidden_layer_sizes)
        layer_sizes = [self.model.input_neurons, *hidden_layers, self.model.output_neurons]
        layer_labels = ["Input"] + [f"Hidden {idx}" for idx in range(len(hidden_layers))] + ["Output"]

        connections: list[Dict[str, Any]] = []
        for idx in range(len(hidden_layers)):
            connections.append(
                {
                    "name": f"hidden_layers.{idx}",
                    "from": idx,
                    "to": idx + 1,
                    "label": f"{layer_labels[idx]} → {layer_labels[idx + 1]}",
                }
            )

        connections.append(
            {
                "name": "decoder",
                "from": len(layer_sizes) - 2,
                "to": len(layer_sizes) - 1,
                "label": f"{layer_labels[-2]} → {layer_labels[-1]}",
            }
        )

        return {
            "layer_sizes": layer_sizes,
            "layer_labels": layer_labels,
            "connections": connections,
            "hidden_layers": hidden_layers,
            "input_neurons": self.model.input_neurons,
            "output_neurons": self.model.output_neurons,
        }


def _filter_updates(instance: Any, updates: Dict[str, Any]) -> Dict[str, Any]:
    allowed = {f.name for f in fields(instance)}
    return {key: value for key, value in updates.items() if key in allowed}


def _ensure_int_list(value: Any) -> list[int]:
    if isinstance(value, list | tuple):
        result = [int(v) for v in value]
    elif isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
        if not parts:
            msg = "hidden_layers string must contain at least one integer"
            raise ValueError(msg)
        result = [int(part) for part in parts]
    elif isinstance(value, int):
        result = [int(value)]
    else:
        msg = f"Cannot coerce {value!r} into a list of integers"
        raise TypeError(msg)

    if not result:
        msg = "hidden_layers must contain at least one entry"
        raise ValueError(msg)
    return result


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "on"}:
            return True
        if lowered in {"false", "0", "no", "off"}:
            return False
    return bool(value)


def _coerce_positive_int(value: Any, minimum: int = 1) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive conversion
        raise ValueError(f"Expected an integer value, received {value!r}") from exc
    return max(parsed, minimum)


def trainer_config_from_dict(
    data: Dict[str, Any],
    base: TrainerConfig | None = None,
) -> TrainerConfig:
    """Merge a payload of user-provided values into a TrainerConfig."""
    config = base or TrainerConfig()
    raw_updates = {k: v for k, v in data.items() if k != "dataset"}

    if "hidden_layers" in raw_updates:
        raw_updates["hidden_layers"] = _ensure_int_list(raw_updates["hidden_layers"])
    if "use_target_bonus" in raw_updates:
        raw_updates["use_target_bonus"] = _coerce_bool(raw_updates["use_target_bonus"])
    if "signed_weights" in raw_updates:
        raw_updates["signed_weights"] = _coerce_bool(raw_updates["signed_weights"])
    if "snapshot_interval" in raw_updates:
        raw_updates["snapshot_interval"] = _coerce_positive_int(raw_updates["snapshot_interval"], minimum=1)
    if "evaluate_interval" in raw_updates:
        raw_updates["evaluate_interval"] = _coerce_positive_int(raw_updates["evaluate_interval"], minimum=1)
    if "learning_rule" in raw_updates:
        rule = str(raw_updates["learning_rule"]).strip().lower()
        if rule not in LEARNING_RULE_LABELS:
            msg = f"Unsupported learning rule: {raw_updates['learning_rule']!r}"
            raise ValueError(msg)
        raw_updates["learning_rule"] = rule
    if "model_type" in raw_updates:
        model = str(raw_updates["model_type"]).strip().lower()
        if model not in MODEL_TYPE_LABELS:
            msg = f"Unsupported model type: {raw_updates['model_type']!r}"
            raise ValueError(msg)
        raw_updates["model_type"] = model
    if "direction_mode" in raw_updates:
        direction = str(raw_updates["direction_mode"]).strip().lower()
        if direction not in CONCENTRATION_DIRECTION_LABELS:
            msg = f"Unsupported direction mode: {raw_updates['direction_mode']!r}"
            raise ValueError(msg)
        raw_updates["direction_mode"] = direction

    top_level_updates = _filter_updates(config, raw_updates)
    if top_level_updates:
        config = replace(config, **top_level_updates)

    dataset_payload = data.get("dataset")
    if dataset_payload:
        filtered_dataset = _filter_updates(config.dataset, dataset_payload)
        config = replace(config, dataset=replace(config.dataset, **filtered_dataset))

    return config


def trainer_config_to_dict(config: TrainerConfig) -> Dict[str, Any]:
    """Serialise the trainer config for JSON responses."""
    return asdict(config)


def trainer_config_from_yaml(path: str, base: TrainerConfig | None = None) -> TrainerConfig:
    """Load a trainer configuration from a YAML file compatible with training configs."""
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    payload: Dict[str, Any] = {}
    training_section = raw.get("training", {})
    dataset_section = raw.get("dataset", {})
    model_section = raw.get("model", {})
    backward_section = raw.get("backward", {})

    dataset_params = dataset_section.get("params", {})
    dataset_payload = {
        key: value
        for key, value in dataset_params.items()
        if key
        in {
            "num_features",
            "threshold",
            "num_train",
            "num_test",
            "seed",
            "batch_size",
            "num_workers",
            "shuffle",
            "feature_min",
            "feature_max",
            "noise_std",
        }
    }

    if training_section.get("batch_size") is not None:
        dataset_payload.setdefault("batch_size", training_section["batch_size"])
    if training_section.get("num_workers") is not None:
        dataset_payload.setdefault("num_workers", training_section["num_workers"])
    if training_section.get("seed") is not None:
        dataset_payload.setdefault("seed", training_section["seed"])

    # Capture explicit dataset target/params so both trainer and CLI can hook custom loaders
    dtarget = dataset_section.get("target")
    if dtarget:
        payload["dataset_target"] = dtarget
        payload["dataset_raw"] = dict(dataset_params)

    if dataset_payload:
        payload["dataset"] = dataset_payload

    model_params = model_section.get("params", {})
    if "hidden_layers" in model_params:
        payload["hidden_layers"] = _ensure_int_list(model_params["hidden_layers"])
    elif "hidden_neurons" in model_params:
        payload["hidden_layers"] = _ensure_int_list(model_params["hidden_neurons"])

    model_target = model_section.get("target")
    if model_target:
        payload["model_target"] = model_target
    if model_params:
        payload["model_params"] = dict(model_params)

    target_model = model_section.get("target") or ""
    target_lower = str(target_model).lower()
    if target_lower:
        if "spiking" in target_lower:
            payload["model_type"] = "spiking"
        else:
            payload["model_type"] = "dense"

    for key in ("spike_threshold", "spike_temperature", "input_neurons", "output_neurons"):
        if key in model_params:
            if key == "input_neurons":
                payload.setdefault("dataset", {})["num_features"] = model_params[key]
            elif key == "output_neurons":
                payload["output_neurons"] = int(model_params[key])
            else:
                payload[key] = model_params[key]

    backward_params = backward_section.get("params", {})
    backward_target = backward_section.get("target", "") or ""

    if backward_target:
        payload["backward_target"] = backward_target
    if backward_params:
        payload["backward_params"] = dict(backward_params)

    if backward_target.endswith("ConcentrationGradientBackwardStrategy"):
        payload["learning_rule"] = "concentration"
        for key in (
            "push_rate",
            "suppress_rate",
            "step_scale",
            "energy_slope",
            "energy_momentum",
            "concentration_momentum",
            "loss_tolerance",
            "weight_clamp",
            "direction_mode",
        ):
            if key in backward_params:
                payload[key] = backward_params[key]
    else:
        if backward_target:
            payload["learning_rule"] = "mass_redistribution"
        for key in (
            "release_rate",
            "reward_gain",
            "base_release",
            "decay",
            "temperature",
            "efficiency_bonus",
            "column_competition",
            "noise_std",
            "mass_budget",
            "target_gain",
            "affinity_strength",
            "affinity_decay",
            "affinity_temperature",
            "sign_consistency_strength",
            "sign_consistency_momentum",
        ):
            if key in backward_params:
                payload[key] = backward_params[key]

    if "signed_weights" in backward_params:
        payload["signed_weights"] = _coerce_bool(backward_params["signed_weights"])

    if "enable_target_bonus" in backward_params:
        payload["use_target_bonus"] = _coerce_bool(backward_params["enable_target_bonus"])
    elif "use_target_bonus" in backward_params:
        payload["use_target_bonus"] = _coerce_bool(backward_params["use_target_bonus"])

    if training_section.get("device"):
        payload["device"] = training_section["device"]
    if training_section.get("epochs"):
        payload["epochs"] = training_section["epochs"]
    if training_section.get("log_interval"):
        payload["log_interval"] = training_section["log_interval"]

    logging_section = raw.get("logging", {})
    if "snapshot_interval" in logging_section:
        payload["snapshot_interval"] = logging_section["snapshot_interval"]
    if "evaluate_interval" in logging_section:
        payload["evaluate_interval"] = logging_section["evaluate_interval"]

    if logging_section:
        payload["logging"] = logging_section

    optimizer_section = raw.get("optimizer", {})
    optimizer_target = optimizer_section.get("target")
    optimizer_params = optimizer_section.get("params", {})
    if optimizer_target:
        payload["optimizer_target"] = optimizer_target
    if optimizer_params:
        payload["optimizer_params"] = dict(optimizer_params)

    return trainer_config_from_dict(payload, base=base)
