"""
Dollar & Trade Signal - Based on Trade-Weighted Dollar Index

Signal Group 7: Monitors dollar strength for sector rotation.
Weakening dollar → XLI, XLK, XLB (exporters)
Strengthening dollar → XLP, XLU (domestic defensives)
"""

from datetime import date
from typing import Optional

import pandas as pd

from .base import BaseSignal
from ..config import SECTOR_ETFS, DOLLAR_SENSITIVITY, ModelConfig
from ..data.macro_loader import MacroFeatureLoader


class DollarSignal(BaseSignal):
    """
    Dollar strength signal based on trade-weighted dollar index.

    Logic:
    - Weakening dollar → XLI, XLK, XLB (exporters benefit)
    - Strengthening dollar → XLP, XLU (domestic defensives)
    """

    name: str = "dollar"
    description: str = "Dollar strength signal"

    def __init__(self, config: Optional[ModelConfig] = None):
        super().__init__(config)
        self._macro_loader = MacroFeatureLoader(config)

    def generate(self, as_of_date: date) -> pd.Series:
        """Generate sector scores based on dollar dynamics."""
        dollar = self._macro_loader.get_dollar_index(as_of_date)

        if dollar.empty or len(dollar) < 13:
            return self._empty_scores()

        # Calculate dollar momentum (12-week rate of change)
        dollar_roc = dollar.pct_change(periods=12).iloc[-1]

        if pd.isna(dollar_roc):
            return self._empty_scores()

        # Normalize: 5% dollar move is significant
        # NEGATIVE because weak dollar helps exporters
        signal_strength = -dollar_roc / 0.05
        signal_strength = max(min(signal_strength, 1.0), -1.0)

        # Apply sector sensitivity
        scores = self._empty_scores()
        for sector in SECTOR_ETFS:
            sensitivity = DOLLAR_SENSITIVITY.get(sector, 0.0)
            scores[sector] = signal_strength * sensitivity

        scores.name = self.name
        return self._normalize_scores(scores)

    def get_dollar_info(self, as_of_date: date) -> dict:
        """Get dollar information."""
        dollar = self._macro_loader.get_dollar_index(as_of_date)

        if dollar.empty:
            return {"level": None, "change": None, "direction": "unknown"}

        current = dollar.iloc[-1]
        change = None
        if len(dollar) > 12:
            change = dollar.pct_change(periods=12).iloc[-1]

        return {
            "level": current,
            "change": change,
            "direction": "strengthening" if change and change > 0 else "weakening",
        }
