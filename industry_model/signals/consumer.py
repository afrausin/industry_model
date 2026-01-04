"""
Consumer Health Signal - Based on Sentiment, Retail Sales, and Credit

Signal Group 6: Monitors consumer health for discretionary vs staples.
Rising sentiment/sales → XLY (discretionary)
Falling sentiment → XLP (staples)
"""

from datetime import date
from typing import Optional

import pandas as pd

from .base import BaseSignal
from ..config import SECTOR_ETFS, ModelConfig
from ..data.macro_loader import MacroFeatureLoader


class ConsumerSignal(BaseSignal):
    """
    Consumer health signal based on sentiment, retail sales, and credit.

    Logic:
    - Rising consumer sentiment → XLY (discretionary)
    - Falling sentiment → XLP (staples)
    - Rising retail sales → XLY
    - Rising consumer credit → XLF, XLY
    """

    name: str = "consumer"
    description: str = "Consumer health signal"

    # Sector sensitivity to consumer strength
    CONSUMER_SENSITIVITY = {
        "XLY": 1.0,     # Consumer discretionary: max positive
        "XLF": 0.4,     # Financials: consumer credit
        "XLC": 0.3,     # Communication: consumer spending
        "XLK": 0.2,     # Tech: consumer tech
        "XLI": 0.1,     # Industrials: slight positive
        "XLE": 0.0,     # Energy: neutral
        "XLB": 0.0,     # Materials: neutral
        "XLRE": 0.0,    # REITs: neutral
        "XLV": -0.3,    # Healthcare: defensive
        "XLP": -0.7,    # Staples: inverse relationship
        "XLU": -0.5,    # Utilities: defensive
    }

    def __init__(self, config: Optional[ModelConfig] = None):
        super().__init__(config)
        self._macro_loader = MacroFeatureLoader(config)

    def generate(self, as_of_date: date) -> pd.Series:
        """Generate sector scores based on consumer health."""
        sentiment = self._macro_loader.get_consumer_sentiment(as_of_date)

        if sentiment.empty or len(sentiment) < 5:
            return self._empty_scores()

        # Calculate sentiment momentum
        sentiment_change = sentiment.diff(12).iloc[-1]  # 12-week change

        if pd.isna(sentiment_change):
            return self._empty_scores()

        # Normalize: 10-point sentiment change is significant
        signal_strength = sentiment_change / 10.0
        signal_strength = max(min(signal_strength, 1.0), -1.0)

        # Apply sector sensitivity
        scores = self._empty_scores()
        for sector in SECTOR_ETFS:
            sensitivity = self.CONSUMER_SENSITIVITY.get(sector, 0.0)
            scores[sector] = signal_strength * sensitivity

        scores.name = self.name
        return self._normalize_scores(scores)

    def get_consumer_info(self, as_of_date: date) -> dict:
        """Get consumer health information."""
        sentiment = self._macro_loader.get_consumer_sentiment(as_of_date)

        if sentiment.empty:
            return {"sentiment": None, "change": None, "direction": "unknown"}

        current = sentiment.iloc[-1]
        change = None
        if len(sentiment) > 12:
            change = sentiment.diff(12).iloc[-1]

        return {
            "sentiment": current,
            "change": change,
            "direction": "improving" if change and change > 0 else "deteriorating",
        }
