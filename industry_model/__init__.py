"""
Industry ETF Relative Performance Model

A heuristic-based quantitative model for predicting relative sector ETF performance
using macroeconomic indicators.

Usage:
    python -m industry_model.run --signals      # Generate current signals
    python -m industry_model.run --weights      # Generate portfolio weights
    python -m industry_model.run --backtest     # Run full backtest
    python -m industry_model.run --walk-forward # Walk-forward analysis
"""

from .config import ModelConfig, SECTOR_ETFS, SECTOR_NAMES

__version__ = "0.1.0"

__all__ = [
    "ModelConfig",
    "SECTOR_ETFS",
    "SECTOR_NAMES",
]
