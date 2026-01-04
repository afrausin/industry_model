"""
Housing Cycle Signal - Based on Housing Starts, Permits, and Mortgage Rates

Signal Group 5: Monitors housing cycle for sector rotation.
Rising housing activity → XLB, XLI, XLRE
Falling mortgage rates → XLRE, XLF
"""

from datetime import date
from typing import Optional

import pandas as pd

from .base import BaseSignal
from ..config import SECTOR_ETFS, HOUSING_SENSITIVITY, ModelConfig
from ..data.macro_loader import MacroFeatureLoader


class HousingSignal(BaseSignal):
    """
    Housing cycle signal based on starts, permits, and mortgage rates.

    Logic:
    - Rising housing starts/permits → XLB (materials), XLI (industrials)
    - Falling mortgage rates → XLRE (REITs), XLF (mortgage originators)
    - Rising home prices → XLY (home improvement)
    """

    name: str = "housing"
    description: str = "Housing cycle signal"

    def __init__(self, config: Optional[ModelConfig] = None):
        super().__init__(config)
        self._macro_loader = MacroFeatureLoader(config)

    def generate(self, as_of_date: date) -> pd.Series:
        """Generate sector scores based on housing cycle."""
        housing_starts = self._macro_loader.get_housing_starts(as_of_date)

        if housing_starts.empty or len(housing_starts) < 5:
            return self._empty_scores()

        # Calculate housing momentum (3-month rate of change)
        housing_roc = housing_starts.pct_change(periods=12).iloc[-1]  # 12 weeks ~ 3 months

        if pd.isna(housing_roc):
            return self._empty_scores()

        # Normalize: 10% change in housing is significant
        signal_strength = housing_roc / 0.10
        signal_strength = max(min(signal_strength, 1.0), -1.0)

        # Apply sector sensitivity
        scores = self._empty_scores()
        for sector in SECTOR_ETFS:
            sensitivity = HOUSING_SENSITIVITY.get(sector, 0.0)
            scores[sector] = signal_strength * sensitivity

        scores.name = self.name
        return self._normalize_scores(scores)

    def get_housing_info(self, as_of_date: date) -> dict:
        """Get housing cycle information."""
        housing_starts = self._macro_loader.get_housing_starts(as_of_date)

        if housing_starts.empty:
            return {"starts": None, "change": None, "direction": "unknown"}

        current = housing_starts.iloc[-1]
        change = None
        if len(housing_starts) > 12:
            change = housing_starts.pct_change(periods=12).iloc[-1]

        return {
            "starts": current,
            "change": change,
            "direction": "improving" if change and change > 0 else "deteriorating",
        }
