"""Model factories for fashion experiments."""

from .simple import SimpleFashionCNN
from .mlp import FashionMLP
from .spiking import TinySpikingNetwork
from .synthetic import SyntheticDenseNetwork

__all__ = ["SimpleFashionCNN", "FashionMLP", "TinySpikingNetwork", "SyntheticDenseNetwork"]
