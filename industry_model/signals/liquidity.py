"""
Liquidity Signal - Based on M2, Fed Balance Sheet, and Bank Reserves

Signal Group 8: Monitors monetary liquidity for risk appetite.
Fed balance sheet expanding (QE) → XLK, XLY (risk-on)
Fed balance sheet contracting (QT) → XLU, XLP (defensive)
"""

from datetime import date
from typing import Optional

import pandas as pd

from .base import BaseSignal
from ..config import SECTOR_ETFS, CYCLICAL_SECTORS, DEFENSIVE_SECTORS, ModelConfig
from ..data.macro_loader import MacroFeatureLoader


class LiquiditySignal(BaseSignal):
    """
    Liquidity signal based on Fed balance sheet and money supply.

    Logic:
    - Fed balance sheet expanding (QE) → XLK, XLY (risk-on)
    - Fed balance sheet contracting (QT) → XLU, XLP (defensive)
    - Rising M2 growth → inflationary, favor XLE, XLB
    """

    name: str = "liquidity"
    description: str = "Monetary liquidity signal"

    # Sector sensitivity to liquidity
    LIQUIDITY_SENSITIVITY = {
        "XLK": 1.0,     # Tech: max benefit from liquidity
        "XLY": 0.8,     # Discretionary: risk-on
        "XLF": 0.5,     # Financials: mixed
        "XLC": 0.4,     # Communication: growth sensitive
        "XLI": 0.3,     # Industrials: some benefit
        "XLE": 0.2,     # Energy: inflation hedge
        "XLB": 0.3,     # Materials: inflation hedge
        "XLRE": 0.1,    # REITs: rates matter more
        "XLV": -0.3,    # Healthcare: defensive
        "XLP": -0.5,    # Staples: defensive
        "XLU": -0.6,    # Utilities: defensive
    }

    def __init__(self, config: Optional[ModelConfig] = None):
        super().__init__(config)
        self._macro_loader = MacroFeatureLoader(config)

    def generate(self, as_of_date: date) -> pd.Series:
        """Generate sector scores based on liquidity conditions."""
        fed_bs = self._macro_loader.get_fed_balance_sheet(as_of_date)

        if fed_bs.empty or len(fed_bs) < 13:
            return self._empty_scores()

        # Calculate Fed balance sheet growth (12-week rate of change)
        bs_growth = fed_bs.pct_change(periods=12).iloc[-1]

        if pd.isna(bs_growth):
            return self._empty_scores()

        # Normalize: 5% balance sheet change is significant
        signal_strength = bs_growth / 0.05
        signal_strength = max(min(signal_strength, 1.0), -1.0)

        # Apply sector sensitivity
        scores = self._empty_scores()
        for sector in SECTOR_ETFS:
            sensitivity = self.LIQUIDITY_SENSITIVITY.get(sector, 0.0)
            scores[sector] = signal_strength * sensitivity

        scores.name = self.name
        return self._normalize_scores(scores)

    def get_liquidity_info(self, as_of_date: date) -> dict:
        """Get liquidity information."""
        fed_bs = self._macro_loader.get_fed_balance_sheet(as_of_date)

        if fed_bs.empty:
            return {"balance_sheet": None, "growth": None, "regime": "unknown"}

        current = fed_bs.iloc[-1]
        growth = None
        if len(fed_bs) > 12:
            growth = fed_bs.pct_change(periods=12).iloc[-1]

        regime = "unknown"
        if growth is not None:
            if growth > 0.02:
                regime = "QE"
            elif growth < -0.02:
                regime = "QT"
            else:
                regime = "neutral"

        return {
            "balance_sheet": current,
            "growth": growth,
            "regime": regime,
        }
