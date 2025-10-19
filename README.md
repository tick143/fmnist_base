## alt-back

Experimental playground for Fashion-MNIST with configurable "backwards" and optimisation strategies. The project is designed so you can swap out gradient calculation and weight update logic via configuration without touching the core training loop.

### Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) for dependency and virtualenv management

### Getting started

```bash
uv venv
source .venv/bin/activate
uv sync
```

Then launch a default training run:

```bash
alt-back-train --config configs/default.yaml
```

To try the alternative co-firing rule with the MLP baseline and Weights & Biases logging:

```bash
alt-back-train --config configs/cofire.yaml
```

To experiment with the loss-free mass redistribution strategy that reallocates synaptic magnitude without backpropagation:

```bash
alt-back-train --config configs/mass_redistribution.yaml
```

### Tiny spiking playground

For a lightweight, synthetic demo that highlights how the mass-redistribution rule manipulates weights step-by-step, launch the interactive visualiser:

```bash
alt-back-viz --port 8000
```

This spins up a FastAPI server with a spiking network trained on a balanced synthetic dataset (`y = 1` when `sum(x) > 1.0`). Inputs are sampled from a widened range (default `[-1.5, 2.5]`) with injected noise so the early synapses stay lively. The learning rule is now “neurotransmitter redistribution”: after every batch we release a reward-dependent pool of transmitter, push it through co-activity/efficiency signals, and redistribute each neuron’s incoming mass without gradients. Persistent affinities, column competition, and sign-consistency keep pathways from collapsing. You can tweak the dataset params (`feature_min`, `feature_max`, `noise_std`, `threshold`), reshape the hidden stack (`hidden_layers` accepts an arbitrary comma-separated list), or adjust the redistribution knobs (release rate, reward gain, decay, affinity/sign strengths, column competition, mass budget, noise) from the UI. Flip the target bonus toggle off to stay fully reward-agnostic. Edit `configs/tiny_spiking.yaml`, reload it straight from the UI, or adjust everything live in the browser panel. The web UI exposes single-step and auto-stepping controls, live topology visualisation (orange/blue edges encode positive/negative weight strength), per-synapse deltas, spike-rate heatmaps, and running evaluation metrics.

### Configuration

All pluggable components are configured in YAML:

- `model.target`: dotted import path for the network module/class
- `backward.target`: class implementing `BackwardStrategy`
- `optimizer.target`: class implementing `OptimizerStrategy`

Each section is accompanied by `params` that are forwarded to the target class constructor. Define your own module under `src/alt_back/...` (or any importable package) and point the config at it to experiment with alternative update rules or biologically inspired models.

For gradient-free approaches, use `alt_back.optim.null_optimizer.NullOptimizerStrategy` so the backward rule can mutate parameters directly.

#### Logging

Enable Weights & Biases by setting `logging.enabled: true`. The trainer will stream batch metrics such as logits/probabilities statistics, prediction entropy, per-layer entropy, network-wide entropy, and any custom signals exposed by backward rules (e.g. clamp ratios in the co-fire strategy).

#### Extending strategies

Custom backwards/optimiser implementations receive a rich `BatchContext` object containing model, activations, targets, and metadata (epoch, batch index, device). Implementations can ignore unused fields while still having everything required for local or layer-wise learning rules.

### Next steps

- Implement custom `BackwardStrategy` classes to explore non-gradient or local learning rules
- Add additional optimiser strategies (e.g. Hebbian-style, equilibrium propagation)
- Extend the model zoo with modules that emulate growth or developmental processes
- Experiment with the provided `CoFireBackwardStrategy` that minimises layer entropy and encourages sequential neurons to co-activate
- Explore the `MassRedistributionBackwardStrategy` for backprop-free, magnitude-constrained plasticity dynamics
