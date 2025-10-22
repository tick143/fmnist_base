from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Dict

import torch

from .base import BackwardStrategy
from ..training.context import BatchContext


@dataclass
class MCFConfig:
    eta: float = 0.02
    mu: float = 0.9
    sigma: float = 5.0
    inhib: float = 0.2
    B: float = 1.0
    tau: float = 0.98


class MinimalCreditField:
    """Minimal directional credit assignment using relative updates and reward budgets."""

    def __init__(self, model: torch.nn.Module, cfg: MCFConfig) -> None:
        self.cfg = cfg
        self.model = model
        self.momentum: Dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if param.requires_grad and param.dim() >= 2:
                self.momentum[name] = torch.zeros_like(param.data)
        self.loss_ema: float | None = None
        self.last_reward: float = 0.0

    @torch.no_grad()
    def step(self, reward: float | None = None, loss: torch.Tensor | None = None) -> None:
        if reward is None and loss is None:
            raise ValueError("Provide reward or loss to MinimalCreditField.step")

        weight_layers = [
            (name, param)
            for name, param in self.model.named_parameters()
            if param.requires_grad and param.dim() == 2
        ]
        if not weight_layers:
            return

        if reward is None:
            L = float(loss.detach().item())
            if self.loss_ema is None:
                self.loss_ema = L
            improvement = (self.loss_ema - L) / (abs(self.loss_ema) + 1e-8)
            self.loss_ema = self.cfg.tau * self.loss_ema + (1.0 - self.cfg.tau) * L
            R = float(math.tanh(3.0 * improvement))
        else:
            R = float(max(-1.0, min(1.0, reward)))

        self.last_reward = R
        budget = self.cfg.B * abs(R)
        if budget <= 1e-12:
            return

        epic_layer_name, epic_weight = random.choice(weight_layers)
        out_dim = epic_weight.shape[0]
        epic_neuron = random.randrange(out_dim) if out_dim > 1 else 0

        if out_dim > 1:
            idx = torch.arange(out_dim, device=epic_weight.device)
            gauss = torch.exp(-(idx - epic_neuron) ** 2 / (2 * (self.cfg.sigma ** 2 + 1e-8)))
            gauss = gauss / (gauss.sum() + 1e-8)
            gauss = gauss.view(out_dim, 1)
        else:
            gauss = torch.ones((1, 1), device=epic_weight.device)

        allocations: Dict[str, tuple[torch.Tensor, float]] = {}
        total_mass = 0.0
        for name, param in weight_layers:
            abs_weight = param.data.abs()
            alloc = abs_weight.clone()
            if name == epic_layer_name:
                alloc = alloc * gauss
            if self.cfg.inhib > 0.0:
                row_mean = alloc.mean(dim=1, keepdim=True)
                alloc = alloc - self.cfg.inhib * row_mean
                alloc.clamp_(min=0.0)
            mass = float(alloc.sum().item())
            allocations[name] = (alloc, mass)
            total_mass += mass

        if total_mass <= 1e-12:
            return

        scale = budget / total_mass
        sign_R = 1.0 if R >= 0 else -1.0

        for name, param in weight_layers:
            alloc, _ = allocations[name]
            rel_step = self.cfg.eta * param.data.abs()
            step_mag = rel_step * (alloc * scale)

            velocity = self.momentum[name]
            velocity_sign = torch.sign(velocity)
            weight_sign = torch.where(param.data >= 0, torch.ones_like(param.data), -torch.ones_like(param.data))
            direction = torch.where(velocity_sign != 0, velocity_sign, weight_sign)

            velocity.mul_(self.cfg.mu).add_((1.0 - self.cfg.mu) * (direction * step_mag * sign_R))
            self.momentum[name] = velocity

            param.add_(torch.sign(velocity) * step_mag * sign_R)


class MinimalCreditFieldBackwardStrategy(BackwardStrategy):
    """Wrap MinimalCreditField as a BackwardStrategy for the training loop."""

    def __init__(
        self,
        eta: float = 0.02,
        mu: float = 0.9,
        sigma: float = 5.0,
        inhib: float = 0.2,
        reward_budget: float = 1.0,
        loss_tau: float = 0.98,
        reward_mode: str = "loss",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.cfg = MCFConfig(eta=eta, mu=mu, sigma=sigma, inhib=inhib, B=reward_budget, tau=loss_tau)
        reward_mode = reward_mode.lower()
        if reward_mode not in {"loss", "accuracy"}:
            raise ValueError("reward_mode must be 'loss' or 'accuracy'")
        self.reward_mode = reward_mode
        self._engine: MinimalCreditField | None = None
        self._model_ref: torch.nn.Module | None = None

    def _ensure_engine(self, model: torch.nn.Module) -> None:
        if self._engine is None or self._model_ref is not model:
            self._engine = MinimalCreditField(model, self.cfg)
            self._model_ref = model

    def backward(self, context: BatchContext) -> None:
        self._ensure_engine(context.model)
        assert self._engine is not None  # for type checkers

        reward = None
        loss = None
        if self.reward_mode == "accuracy":
            outputs = context.outputs.detach()
            if outputs.numel() > 0:
                preds = outputs.argmax(dim=1)
                accuracy = preds.eq(context.targets).float().mean().item() if preds.numel() else 0.0
                reward = 2.0 * accuracy - 1.0
            else:
                reward = 0.0
        else:
            loss = context.loss

        self._engine.step(reward=reward, loss=loss)
        context.extras["mcf_reward"] = float(self._engine.last_reward)
