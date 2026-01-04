"""
Volatility Regime Signal - Based on VIX Level

Signal Group 9: Monitors volatility regime for beta positioning.
VIX < 15: Low vol, favor beta (XLK, XLY, XLF)
VIX > 25: High vol, favor low-beta (XLU, XLP, XLV)
"""

from datetime import date
from typing import Optional

import pandas as pd

from .base import BaseSignal
from ..config import SECTOR_ETFS, VIX_LOW_SECTORS, VIX_HIGH_SECTORS, ModelConfig
from ..data.macro_loader import MacroFeatureLoader


class VolatilitySignal(BaseSignal):
    """
    Volatility regime signal based on VIX level.

    Logic:
    - VIX < 15: Low vol, favor beta (XLK, XLY, XLF)
    - VIX 15-25: Normal, neutral
    - VIX > 25: High vol, favor low-beta (XLU, XLP, XLV)
    """

    name: str = "volatility"
    description: str = "Volatility regime signal"

    # VIX thresholds
    VIX_LOW = 15.0
    VIX_HIGH = 25.0

    # Sector beta characteristics
    HIGH_BETA_SECTORS = {
        "XLK": 1.0,     # Tech: high beta
        "XLY": 0.9,     # Discretionary: high beta
        "XLF": 0.7,     # Financials: high beta
        "XLC": 0.6,     # Communication: moderate-high
        "XLI": 0.5,     # Industrials: moderate
        "XLB": 0.4,     # Materials: moderate
        "XLE": 0.3,     # Energy: idiosyncratic
        "XLRE": 0.0,    # REITs: moderate
        "XLV": -0.5,    # Healthcare: low beta
        "XLP": -0.7,    # Staples: low beta
        "XLU": -0.8,    # Utilities: lowest beta
    }

    def __init__(self, config: Optional[ModelConfig] = None):
        super().__init__(config)
        self._macro_loader = MacroFeatureLoader(config)

    def generate(self, as_of_date: date) -> pd.Series:
        """Generate sector scores based on volatility regime."""
        vix = self._macro_loader.get_vix(as_of_date)

        if vix.empty:
            return self._empty_scores()

        current_vix = vix.iloc[-1]

        if pd.isna(current_vix):
            return self._empty_scores()

        # Determine regime and signal strength
        if current_vix < self.VIX_LOW:
            # Low vol: favor high beta
            signal_strength = 1.0 - (current_vix / self.VIX_LOW)  # Higher when VIX lower
            signal_strength = min(signal_strength, 1.0)
        elif current_vix > self.VIX_HIGH:
            # High vol: favor low beta
            excess_vix = current_vix - self.VIX_HIGH
            signal_strength = -min(excess_vix / 10.0, 1.0)  # Negative for high VIX
        else:
            # Normal: neutral
            signal_strength = 0.0

        # Apply to sectors based on beta
        scores = self._empty_scores()
        for sector in SECTOR_ETFS:
            beta_sensitivity = self.HIGH_BETA_SECTORS.get(sector, 0.0)
            scores[sector] = signal_strength * beta_sensitivity

        scores.name = self.name
        return self._normalize_scores(scores)

    def get_volatility_info(self, as_of_date: date) -> dict:
        """Get volatility regime information."""
        vix = self._macro_loader.get_vix(as_of_date)

        if vix.empty:
            return {"vix": None, "regime": "unknown"}

        current_vix = vix.iloc[-1]

        if pd.isna(current_vix):
            return {"vix": None, "regime": "unknown"}

        if current_vix < self.VIX_LOW:
            regime = "low"
        elif current_vix > self.VIX_HIGH:
            regime = "high"
        else:
            regime = "normal"

        return {
            "vix": current_vix,
            "regime": regime,
        }
