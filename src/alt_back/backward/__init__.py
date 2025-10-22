"""Credit assignment strategies."""

from .autograd import AutogradBackwardStrategy
from .base import BackwardStrategy
from .cofire import CoFireBackwardStrategy
from .concentration import ConcentrationGradientBackwardStrategy
from .mass_redistribution import MassRedistributionBackwardStrategy
from .cg_wta import CgWtaBackwardStrategy
from .minimal_credit_field import MinimalCreditFieldBackwardStrategy
from .direct_feedback_alignment import DirectFeedbackAlignmentBackwardStrategy
from .forward_forward import ForwardForwardBackwardStrategy
from .fff import FFFBackwardStrategy

__all__ = [
    "BackwardStrategy",
    "AutogradBackwardStrategy",
    "CoFireBackwardStrategy",
    "MassRedistributionBackwardStrategy",
    "ConcentrationGradientBackwardStrategy",
    "MinimalCreditFieldBackwardStrategy",
    "DirectFeedbackAlignmentBackwardStrategy",
    "ForwardForwardBackwardStrategy",
    "CgWtaBackwardStrategy",
    "FFFBackwardStrategy",
]
