"""
Gemini AI Signal - Based on AI-Derived Quadrant Probabilities and Recommendations

Signal Group 10: Uses Gemini AI features for signal generation.
- QuadrantProbabilities: AI-derived probability for each quadrant
- DocumentSummary: Growth/inflation assessments from Fed docs
- PortfolioRecommendations: AI-suggested sector overweights/underweights
"""

from datetime import date
from typing import Optional

import pandas as pd

from .base import BaseSignal
from ..config import SECTOR_ETFS, QUADRANT_SECTOR_SCORES, ModelConfig
from ..data.gemini_loader import GeminiLoader


class GeminiSignal(BaseSignal):
    """
    Gemini AI-derived signal using quadrant probabilities and sector recommendations.

    Logic:
    - Use QuadrantProbabilities.most_likely() → Apply quadrant sector preferences
    - Use PortfolioRecommendations.sector_overweights/underweights → Direct signal
    - Blend AI quadrant view with explicit recommendations
    """

    name: str = "gemini"
    description: str = "Gemini AI-derived signal"

    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        api_key: Optional[str] = None,
    ):
        super().__init__(config)
        self._gemini_loader = GeminiLoader(api_key=api_key, config=config)

    def generate(self, as_of_date: date) -> pd.Series:
        """Generate sector scores based on Gemini AI features."""
        # Get AI-derived sector scores (handles quadrant probs + recommendations)
        scores = self._gemini_loader.get_gemini_sector_scores(as_of_date)

        if scores.empty:
            return self._empty_scores()

        scores.name = self.name
        return self._normalize_scores(scores)

    def get_gemini_info(self, as_of_date: date) -> dict:
        """Get Gemini AI feature information."""
        probs = self._gemini_loader.get_quadrant_probabilities(as_of_date)
        recs = self._gemini_loader.get_sector_recommendations(as_of_date)
        tone = self._gemini_loader.get_tone_shift_signal(as_of_date)

        return {
            "quadrant_probs": probs,
            "sector_recs": recs,
            "tone_shift": tone,
        }
