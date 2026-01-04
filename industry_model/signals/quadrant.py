"""
Quadrant Signal - Macro Regime Based on Growth and Inflation Momentum

Signal Group 1: Uses INDPRO (growth) and CPIAUCSL (inflation) rate of change
to determine Hedgeye-style macro quadrant.
"""

from datetime import date
from typing import Optional

import pandas as pd

from .base import BaseSignal
from ..config import SECTOR_ETFS, QUADRANT_SECTOR_SCORES, ModelConfig
from ..data.macro_loader import MacroFeatureLoader


class QuadrantSignal(BaseSignal):
    """
    Macro quadrant signal based on growth and inflation momentum.

    Quadrants:
    - Quad 1: Growth Rising, Inflation Falling (Goldilocks)
    - Quad 2: Growth Rising, Inflation Rising (Inflationary Boom)
    - Quad 3: Growth Falling, Inflation Falling (Deflationary Slowdown)
    - Quad 4: Growth Falling, Inflation Rising (Stagflation)
    """

    name: str = "quadrant"
    description: str = "Macro quadrant based on growth/inflation momentum"

    def __init__(self, config: Optional[ModelConfig] = None):
        super().__init__(config)
        self._macro_loader = MacroFeatureLoader(config)

    def generate(self, as_of_date: date) -> pd.Series:
        """Generate sector scores based on current macro quadrant."""
        # Get growth momentum (Industrial Production YoY change)
        growth_momentum = self._macro_loader.get_growth_momentum(as_of_date)

        # Get inflation momentum (CPI YoY change)
        inflation_momentum = self._macro_loader.get_inflation_momentum(as_of_date)

        if growth_momentum.empty or inflation_momentum.empty:
            return self._empty_scores()

        # Get latest values
        latest_growth = growth_momentum.iloc[-1] if len(growth_momentum) > 0 else 0
        latest_inflation = inflation_momentum.iloc[-1] if len(inflation_momentum) > 0 else 0

        # Determine quadrant
        quadrant = self._determine_quadrant(latest_growth, latest_inflation)

        # Get sector scores for this quadrant
        quadrant_scores = QUADRANT_SECTOR_SCORES.get(quadrant, {})

        # Build output series
        scores = self._empty_scores()
        for sector in SECTOR_ETFS:
            scores[sector] = quadrant_scores.get(sector, 0.0)

        scores.name = self.name
        return scores

    def _determine_quadrant(self, growth: float, inflation: float) -> int:
        """
        Determine macro quadrant from growth and inflation direction.

        Uses rate of change - positive means accelerating.
        """
        if growth > 0 and inflation <= 0:
            return 1  # Goldilocks: Growth up, Inflation down
        elif growth > 0 and inflation > 0:
            return 2  # Inflationary Boom: Both up
        elif growth <= 0 and inflation <= 0:
            return 3  # Deflationary Slowdown: Both down
        else:
            return 4  # Stagflation: Growth down, Inflation up

    def get_quadrant_info(self, as_of_date: date) -> dict:
        """
        Get detailed quadrant information.

        Returns:
            Dict with quadrant number, growth, inflation values.
        """
        growth_momentum = self._macro_loader.get_growth_momentum(as_of_date)
        inflation_momentum = self._macro_loader.get_inflation_momentum(as_of_date)

        if growth_momentum.empty or inflation_momentum.empty:
            return {"quadrant": 0, "growth": None, "inflation": None}

        latest_growth = growth_momentum.iloc[-1] if len(growth_momentum) > 0 else 0
        latest_inflation = inflation_momentum.iloc[-1] if len(inflation_momentum) > 0 else 0

        quadrant = self._determine_quadrant(latest_growth, latest_inflation)

        return {
            "quadrant": quadrant,
            "growth": latest_growth,
            "inflation": latest_inflation,
            "growth_direction": "rising" if latest_growth > 0 else "falling",
            "inflation_direction": "rising" if latest_inflation > 0 else "falling",
        }
