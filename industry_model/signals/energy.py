"""
Energy Cycle Signal - Based on Energy Sector Dynamics

Signal Group 12: Monitors energy-specific indicators.
Dollar inverse correlation → XLE
Energy benefits from inflation → XLE during Quad 2/4
"""

from datetime import date
from typing import Optional

import pandas as pd

from .base import BaseSignal
from ..config import SECTOR_ETFS, ModelConfig
from ..data.macro_loader import MacroFeatureLoader


class EnergySignal(BaseSignal):
    """
    Energy cycle signal based on inflation and dollar dynamics.

    Logic:
    - Rising inflation → XLE (inflation hedge)
    - Weakening dollar → XLE (dollar-denominated commodities)
    - Dollar correlation → XLE inverse to dollar
    """

    name: str = "energy"
    description: str = "Energy cycle signal"

    def __init__(self, config: Optional[ModelConfig] = None):
        super().__init__(config)
        self._macro_loader = MacroFeatureLoader(config)

    def generate(self, as_of_date: date) -> pd.Series:
        """Generate sector scores based on energy cycle dynamics."""
        # Energy signal based on inflation and dollar
        inflation_momentum = self._macro_loader.get_inflation_momentum(as_of_date)
        dollar = self._macro_loader.get_dollar_index(as_of_date)

        signal_components = []

        # Inflation component (rising inflation helps energy)
        if not inflation_momentum.empty:
            inflation_val = inflation_momentum.iloc[-1]
            if not pd.isna(inflation_val):
                # Normalize: 2% inflation momentum is significant
                inflation_signal = inflation_val / 0.02
                inflation_signal = max(min(inflation_signal, 1.0), -1.0)
                signal_components.append(inflation_signal)

        # Dollar component (weak dollar helps energy)
        if not dollar.empty and len(dollar) > 12:
            dollar_roc = dollar.pct_change(periods=12).iloc[-1]
            if not pd.isna(dollar_roc):
                # Inverse: weak dollar is positive for energy
                dollar_signal = -dollar_roc / 0.05
                dollar_signal = max(min(dollar_signal, 1.0), -1.0)
                signal_components.append(dollar_signal)

        if not signal_components:
            return self._empty_scores()

        # Average the components
        energy_signal = sum(signal_components) / len(signal_components)

        # Apply primarily to energy, with inverse to other sectors
        scores = self._empty_scores()

        # XLE gets the direct signal
        scores["XLE"] = energy_signal

        # XLB also benefits from commodity strength
        scores["XLB"] = energy_signal * 0.5

        # Defensives get slight inverse
        scores["XLU"] = -energy_signal * 0.2
        scores["XLP"] = -energy_signal * 0.2

        scores.name = self.name
        return self._normalize_scores(scores)

    def get_energy_info(self, as_of_date: date) -> dict:
        """Get energy cycle information."""
        inflation_momentum = self._macro_loader.get_inflation_momentum(as_of_date)
        dollar = self._macro_loader.get_dollar_index(as_of_date)

        info = {
            "inflation_momentum": None,
            "dollar_change": None,
            "energy_outlook": "neutral",
        }

        if not inflation_momentum.empty:
            info["inflation_momentum"] = inflation_momentum.iloc[-1]

        if not dollar.empty and len(dollar) > 12:
            info["dollar_change"] = dollar.pct_change(periods=12).iloc[-1]

        # Determine outlook
        inf_positive = info["inflation_momentum"] and info["inflation_momentum"] > 0
        dollar_weak = info["dollar_change"] and info["dollar_change"] < 0

        if inf_positive and dollar_weak:
            info["energy_outlook"] = "bullish"
        elif inf_positive or dollar_weak:
            info["energy_outlook"] = "slightly_bullish"
        elif not inf_positive and not dollar_weak:
            info["energy_outlook"] = "bearish"
        else:
            info["energy_outlook"] = "neutral"

        return info
