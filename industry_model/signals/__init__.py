"""
Signal Generation Module for Industry Model

Each signal outputs a pd.Series indexed by sector with scores in [-1, +1].
"""

from .base import BaseSignal
from .quadrant import QuadrantSignal
from .yield_curve import YieldCurveSignal
from .credit import CreditSignal
from .high_freq import HighFreqSignal
from .housing import HousingSignal
from .consumer import ConsumerSignal
from .dollar import DollarSignal
from .liquidity import LiquiditySignal
from .volatility import VolatilitySignal
from .gemini import GeminiSignal
from .labor import LaborSignal
from .energy import EnergySignal

__all__ = [
    "BaseSignal",
    "QuadrantSignal",
    "YieldCurveSignal",
    "CreditSignal",
    "HighFreqSignal",
    "HousingSignal",
    "ConsumerSignal",
    "DollarSignal",
    "LiquiditySignal",
    "VolatilitySignal",
    "GeminiSignal",
    "LaborSignal",
    "EnergySignal",
]

# All available signals for easy iteration
ALL_SIGNALS = [
    QuadrantSignal,
    YieldCurveSignal,
    CreditSignal,
    HighFreqSignal,
    HousingSignal,
    ConsumerSignal,
    DollarSignal,
    LiquiditySignal,
    VolatilitySignal,
    GeminiSignal,
    LaborSignal,
    EnergySignal,
]
