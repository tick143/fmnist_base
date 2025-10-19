"""Model factories for fashion experiments."""

from .simple import SimpleFashionCNN
from .mlp import FashionMLP
from .spiking import TinySpikingNetwork

__all__ = ["SimpleFashionCNN", "FashionMLP", "TinySpikingNetwork"]
