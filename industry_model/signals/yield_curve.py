"""
Yield Curve Signal - Based on 10Y-2Y Spread Changes

Signal Group 2: Monitors yield curve steepening/flattening.
Steepening benefits financials, flattening benefits utilities/REITs.
"""

from datetime import date
from typing import Optional

import pandas as pd

from .base import BaseSignal
from ..config import SECTOR_ETFS, YIELD_CURVE_SENSITIVITY, ModelConfig
from ..data.macro_loader import MacroFeatureLoader


class YieldCurveSignal(BaseSignal):
    """
    Yield curve slope signal.

    Logic:
    - Steepening (T10Y2Y rising): XLF benefits (net interest margin widens)
    - Flattening/Inverting: XLU, XLP, XLV benefit (defensive, long-duration)
    """

    name: str = "yield_curve"
    description: str = "Yield curve steepening/flattening signal"

    def __init__(self, config: Optional[ModelConfig] = None):
        super().__init__(config)
        self._macro_loader = MacroFeatureLoader(config)

    def generate(self, as_of_date: date) -> pd.Series:
        """Generate sector scores based on yield curve dynamics."""
        # Get yield curve spread (T10Y2Y)
        spread = self._macro_loader.get_yield_curve_spread(as_of_date)

        if spread.empty or len(spread) < self.config.yield_curve_ma_periods + 1:
            return self._empty_scores()

        # Calculate change in spread over lookback period
        spread_change = spread.diff(self.config.yield_curve_ma_periods).iloc[-1]

        if pd.isna(spread_change):
            return self._empty_scores()

        # Normalize change to a signal strength (-1 to +1)
        # Typical spread changes are ~50bp, so scale by 50
        signal_strength = spread_change / 0.50
        signal_strength = max(min(signal_strength, 1.0), -1.0)

        # Apply sector sensitivity
        scores = self._empty_scores()
        for sector in SECTOR_ETFS:
            sensitivity = YIELD_CURVE_SENSITIVITY.get(sector, 0.0)
            scores[sector] = signal_strength * sensitivity

        scores.name = self.name
        return self._normalize_scores(scores)

    def get_curve_info(self, as_of_date: date) -> dict:
        """Get yield curve information."""
        spread = self._macro_loader.get_yield_curve_spread(as_of_date)

        if spread.empty:
            return {"spread": None, "change": None, "direction": "unknown"}

        current_spread = spread.iloc[-1]

        if len(spread) > self.config.yield_curve_ma_periods:
            spread_change = spread.diff(self.config.yield_curve_ma_periods).iloc[-1]
        else:
            spread_change = None

        return {
            "spread": current_spread,
            "change": spread_change,
            "direction": "steepening" if spread_change and spread_change > 0 else "flattening",
            "inverted": current_spread < 0 if not pd.isna(current_spread) else None,
        }
