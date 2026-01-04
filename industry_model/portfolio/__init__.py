"""
Portfolio Construction Module for Industry Model

Combines signals and generates portfolio weights.
"""

from .combiner import SignalCombiner
from .weights import PortfolioWeights

__all__ = ["SignalCombiner", "PortfolioWeights"]
