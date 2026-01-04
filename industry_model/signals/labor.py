"""
Labor Market Signal - Based on Payrolls, Unemployment, and Job Openings

Signal Group 11: Monitors labor market for economic activity.
Rising payrolls + falling unemployment → cyclicals (XLK, XLY, XLI)
Rising claims + falling job openings → defensives (XLU, XLP)
"""

from datetime import date
from typing import Optional

import pandas as pd

from .base import BaseSignal
from ..config import SECTOR_ETFS, CYCLICAL_SECTORS, DEFENSIVE_SECTORS, ModelConfig
from ..data.macro_loader import MacroFeatureLoader


class LaborSignal(BaseSignal):
    """
    Labor market signal based on payrolls, unemployment, and job openings.

    Logic:
    - Rising payrolls + falling unemployment → cyclicals (XLK, XLY, XLI)
    - Rising claims + falling job openings → defensives (XLU, XLP)
    - Labor participation changes → consumer strength signal
    """

    name: str = "labor"
    description: str = "Labor market signal"

    # Sector sensitivity to labor market strength
    LABOR_SENSITIVITY = {
        "XLK": 0.6,     # Tech: moderate positive
        "XLY": 0.8,     # Discretionary: strong positive (consumer income)
        "XLI": 0.7,     # Industrials: strong positive
        "XLF": 0.5,     # Financials: moderate positive
        "XLC": 0.4,     # Communication: moderate
        "XLB": 0.5,     # Materials: moderate
        "XLE": 0.2,     # Energy: slight positive
        "XLRE": 0.1,    # REITs: slight
        "XLV": -0.3,    # Healthcare: defensive
        "XLP": -0.5,    # Staples: defensive
        "XLU": -0.6,    # Utilities: defensive
    }

    def __init__(self, config: Optional[ModelConfig] = None):
        super().__init__(config)
        self._macro_loader = MacroFeatureLoader(config)

    def generate(self, as_of_date: date) -> pd.Series:
        """Generate sector scores based on labor market conditions."""
        claims = self._macro_loader.get_initial_claims(as_of_date)

        if claims.empty or len(claims) < 5:
            return self._empty_scores()

        # Calculate claims momentum (falling claims is positive)
        claims_change = claims.diff(4).iloc[-1]  # 4-week change

        if pd.isna(claims_change):
            return self._empty_scores()

        # Normalize: 50k change in claims is significant
        # NEGATIVE because falling claims is good
        signal_strength = -claims_change / 50000.0
        signal_strength = max(min(signal_strength, 1.0), -1.0)

        # Apply sector sensitivity
        scores = self._empty_scores()
        for sector in SECTOR_ETFS:
            sensitivity = self.LABOR_SENSITIVITY.get(sector, 0.0)
            scores[sector] = signal_strength * sensitivity

        scores.name = self.name
        return self._normalize_scores(scores)

    def get_labor_info(self, as_of_date: date) -> dict:
        """Get labor market information."""
        claims = self._macro_loader.get_initial_claims(as_of_date)

        if claims.empty:
            return {"claims": None, "change": None, "direction": "unknown"}

        current = claims.iloc[-1]
        change = None
        if len(claims) > 4:
            change = claims.diff(4).iloc[-1]

        return {
            "claims": current,
            "change": change,
            "direction": "improving" if change and change < 0 else "deteriorating",
        }
