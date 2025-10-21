from __future__ import annotations

from typing import Any, List, Tuple

import torch
from torch import nn

from .base import BackwardStrategy
from ..training.context import BatchContext


def _normalise(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    m = v.abs().max()
    if isinstance(m, torch.Tensor):
        m_val = float(m.item())
    else:
        m_val = float(m)
    return torch.zeros_like(v) if m_val <= eps else v / (m + eps)


def _wta_spikes(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    idx = x.argmax(dim=1)
    out = torch.zeros_like(x)
    out[torch.arange(x.size(0), device=x.device), idx] = 1.0
    return out, idx


class CgWtaBackwardStrategy(BackwardStrategy):
    """Winner-Take-All CG-style local update, gradient-free.

    Applies a reward-modulated, batch-statistic update per linear layer:
    - Convert linear preactivations to one-hot WTA spikes per sample
    - Compute contrast signal from input/output batch moments
    - Direction from reward-weighted co-firing
    - Push/suppress rows with lateral inhibition and energy gating

    Expects Linear modules and uses recorded inputs (post-activation from previous layer)
    and recorded outputs (current layer preactivation).
    """

    def __init__(
        self,
        step_scale: float = 0.02,
        push_rate: float = 0.52,
        suppress_rate: float = 0.12,
        energy_momentum: float = 0.965,
        energy_slope: float = 1.4,
        lateral_inhibition: float = 0.03,
        background: float = 0.025,
        weight_clamp: float | None = 6.5,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.step_scale = float(step_scale)
        self.push_rate = float(push_rate)
        self.suppress_rate = float(suppress_rate)
        self.energy_momentum = float(max(min(energy_momentum, 1.0), 0.0))
        self.energy_slope = float(energy_slope)
        self.lateral_inhibition = float(max(lateral_inhibition, 0.0))
        self.background = float(max(min(background, 1.0), 0.0))
        self.weight_clamp = float(weight_clamp) if weight_clamp is not None else None

        self.running_energy: float = 0.0
        self.eps: float = 1e-8

    def backward(self, context: BatchContext) -> None:
        with torch.no_grad():
            # Prepare per-sample reward from head WTA correctness
            linear_records: List[Tuple[str, nn.Linear, torch.Tensor, torch.Tensor]] = []
            for name, rec in context.activations.items():
                module = rec.module
                if isinstance(module, nn.Linear) and module.weight is not None and module.weight.dim() == 2:
                    I_f = rec.inputs[0]
                    O_pre = rec.output
                    if I_f.dim() != 2 or O_pre.dim() != 2:
                        continue
                    if I_f.size(0) != O_pre.size(0):
                        continue
                    linear_records.append((name, module, I_f, O_pre))

            if not linear_records:
                return

            # Head reward signal from last linear layer vs targets
            _, last_module, _, last_preact = linear_records[-1]
            last_out_wta, winners = _wta_spikes(last_preact)
            r = torch.where(
                winners == context.targets,
                torch.tensor(1.0, device=last_preact.device),
                torch.tensor(-1.0, device=last_preact.device),
            )

            wrong_rate = float((r < 0).float().mean().item())

            # Per-layer energy (std of neuron_delta normalised), with EMA
            total_energy = 0.0

            for _, module, I_f, O_pre in linear_records:
                W = module.weight.data
                b = module.bias.data if module.bias is not None else None

                # Winner-take-all spikes for current layer preactivations
                O_f, _ = _wta_spikes(O_pre)

                B = I_f.size(0)
                # batch moments
                m_i = I_f.mean(dim=0)
                m_o = O_f.mean(dim=0)
                Eoi = (O_f.T @ I_f) / max(B, 1)  # (out x in)
                Ei_bar = m_i.mean()
                neuron_delta = m_o + Ei_bar - 2.0 * Eoi.mean(dim=1)  # (out,)
                contrast = neuron_delta - neuron_delta.mean()

                module_energy = float(neuron_delta.float().std(unbiased=False).item())
                norm = float(neuron_delta.float().mean().abs().item()) + 1e-6
                module_energy = max(module_energy / norm, 1e-6)
                total_energy += module_energy

                # reward-weighted co-firing direction
                O_r = O_f * r.unsqueeze(1)
                E_r = (O_r.T @ I_f) / max(B, 1)
                D = torch.sign(E_r - E_r.mean())
                D[D == 0] = 1.0
                D = D.to(W.dtype)

                if wrong_rate < 0.5:
                    push_norm = _normalise(torch.clamp(contrast, min=0.0))
                    suppress_norm = _normalise(torch.clamp(-contrast, min=0.0))
                else:
                    push_norm = torch.zeros_like(contrast)
                    suppress_norm = _normalise(neuron_delta)

                row_delta = self.push_rate * push_norm - self.suppress_rate * suppress_norm

                # lateral inhibition across output rows
                if W.size(0) >= 2 and self.lateral_inhibition > 0:
                    mean_other = (row_delta.sum() - row_delta) / max(W.size(0) - 1, 1)
                    row_delta = row_delta - self.lateral_inhibition * mean_other

                # event-driven mask: only reinforce co-firing pairs; keep a tiny background floor
                co_mask = (Eoi > 0).to(W.dtype)
                mask = self.background + (1 - self.background) * co_mask

                # will scale below after energy gate is updated
                update_unnorm = (row_delta.unsqueeze(1) * D) * mask

                # apply update later after energy gate computation
                # stash on module for a moment
                module._cg_cache = (update_unnorm, row_delta, module_energy)  # type: ignore[attr-defined]

            # Update energy gate
            self.running_energy = self.energy_momentum * self.running_energy + (1.0 - self.energy_momentum) * (
                total_energy / max(len(linear_records), 1)
            )
            energy_factor = max(self.running_energy * self.energy_slope, 1.0)
            base_scale = (self.step_scale * max(wrong_rate * energy_factor, 1e-4))

            context.extras["cg_wrong_rate"] = wrong_rate
            context.extras["cg_energy"] = self.running_energy
            context.extras["cg_energy_factor"] = energy_factor

            # Apply cached per-layer updates with final scaling and bias
            for _, module, _, _ in linear_records:
                W = module.weight.data
                b = module.bias.data if module.bias is not None else None
                update_unnorm, row_delta, module_energy = module._cg_cache  # type: ignore[attr-defined]
                scale = base_scale / (1.0 + module_energy)
                update = torch.clamp(update_unnorm * scale, -1.0, 1.0)

                W.add_(update)
                if self.weight_clamp is not None:
                    W.clamp_(-self.weight_clamp, self.weight_clamp)

                if b is not None:
                    b.add_(torch.clamp(row_delta * scale, -1.0, 1.0))

                # cleanup
                delattr(module, "_cg_cache")

