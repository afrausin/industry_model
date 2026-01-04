"""
Abstract Base Class for Signals

All signals output a pd.Series indexed by sector with scores in [-1, +1].
"""

from abc import ABC, abstractmethod
from datetime import date
from typing import Optional

import pandas as pd

from ..config import SECTOR_ETFS, ModelConfig


class BaseSignal(ABC):
    """
    Abstract base class for sector rotation signals.

    Each signal takes macroeconomic data and outputs sector scores.
    Positive scores indicate overweight, negative scores indicate underweight.
    """

    name: str = "base"
    description: str = "Base signal class"

    def __init__(self, config: Optional[ModelConfig] = None):
        """
        Initialize signal.

        Args:
            config: Model configuration.
        """
        self.config = config or ModelConfig()

    @abstractmethod
    def generate(self, as_of_date: date) -> pd.Series:
        """
        Generate sector scores for a given date.

        Args:
            as_of_date: Point-in-time date for signal generation.

        Returns:
            Series indexed by sector ETF symbol with scores in [-1, +1].
        """
        pass

    def _empty_scores(self) -> pd.Series:
        """Return neutral (zero) scores for all sectors."""
        return pd.Series(0.0, index=SECTOR_ETFS, name=self.name)

    def _normalize_scores(self, scores: pd.Series) -> pd.Series:
        """
        Normalize scores to [-1, +1] range.

        Args:
            scores: Raw scores.

        Returns:
            Normalized scores clamped to [-1, +1].
        """
        if scores.empty:
            return self._empty_scores()

        # Clamp to [-1, +1]
        normalized = scores.clip(lower=-1.0, upper=1.0)
        normalized.name = self.name

        return normalized

    def _apply_sector_mapping(
        self,
        mapping: dict,
        condition_value: float,
        threshold: float = 0.0,
    ) -> pd.Series:
        """
        Apply sector scores based on a condition value.

        Args:
            mapping: Dict mapping sector -> score when condition is positive.
            condition_value: The condition to check (e.g., momentum value).
            threshold: Threshold for determining direction.

        Returns:
            Sector scores.
        """
        scores = self._empty_scores()

        if condition_value > threshold:
            # Apply positive mapping
            for sector, score in mapping.items():
                if sector in scores.index:
                    scores[sector] = score
        elif condition_value < -threshold:
            # Apply inverse mapping
            for sector, score in mapping.items():
                if sector in scores.index:
                    scores[sector] = -score

        return self._normalize_scores(scores)

    def generate_historical(
        self,
        start_date: date,
        end_date: date,
        frequency: str = "W-FRI",
    ) -> pd.DataFrame:
        """
        Generate signals for a date range.

        Args:
            start_date: Start date.
            end_date: End date.
            frequency: Resampling frequency.

        Returns:
            DataFrame with dates as index and sectors as columns.
        """
        # Generate date range
        dates = pd.date_range(start=start_date, end=end_date, freq=frequency)

        results = []
        for dt in dates:
            try:
                signal = self.generate(dt.date())
                signal.name = dt
                results.append(signal)
            except Exception as e:
                print(f"Warning: Could not generate signal for {dt.date()}: {e}")
                signal = self._empty_scores()
                signal.name = dt
                results.append(signal)

        if not results:
            return pd.DataFrame()

        df = pd.DataFrame(results)
        df.index.name = "date"

        return df
