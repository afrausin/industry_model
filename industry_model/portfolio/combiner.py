"""
Signal Combiner for Industry Model

Aggregates signals from all 12 signal groups into composite sector scores.
Uses equal-weighted average (NOT optimization) to avoid overfitting.
"""

from datetime import date
from typing import List, Optional, Dict, Type

import pandas as pd

from ..config import SECTOR_ETFS, ModelConfig
from ..signals.base import BaseSignal
from ..signals import ALL_SIGNALS


class SignalCombiner:
    """
    Combine multiple signals into composite sector scores.

    Approach: Equal-weighted average (NOT optimization to avoid overfitting)
    - Each signal group contributes 1/12 weight
    - Combined score per sector = mean of all signal scores
    - Rank sectors 1-11 based on combined scores
    """

    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        signal_classes: Optional[List[Type[BaseSignal]]] = None,
        gemini_api_key: Optional[str] = None,
    ):
        """
        Initialize signal combiner.

        Args:
            config: Model configuration.
            signal_classes: List of signal classes to use. Defaults to ALL_SIGNALS.
            gemini_api_key: API key for Gemini signal.
        """
        self.config = config or ModelConfig()
        self.gemini_api_key = gemini_api_key

        # Initialize signals
        signal_classes = signal_classes or ALL_SIGNALS
        self.signals: List[BaseSignal] = []

        for signal_cls in signal_classes:
            try:
                if signal_cls.__name__ == "GeminiSignal":
                    signal = signal_cls(config=self.config, api_key=gemini_api_key)
                else:
                    signal = signal_cls(config=self.config)
                self.signals.append(signal)
            except Exception as e:
                print(f"Warning: Could not initialize {signal_cls.__name__}: {e}")

    def generate_combined_scores(
        self,
        as_of_date: date,
        return_individual: bool = False,
    ) -> pd.Series:
        """
        Generate combined sector scores from all signals.

        Args:
            as_of_date: Point-in-time date.
            return_individual: If True, return DataFrame with all individual signals.

        Returns:
            Series of combined sector scores indexed by sector.
        """
        all_scores = {}

        for signal in self.signals:
            try:
                scores = signal.generate(as_of_date)
                if not scores.empty:
                    all_scores[signal.name] = scores
            except Exception as e:
                print(f"Warning: {signal.name} failed: {e}")

        if not all_scores:
            # Return neutral scores
            return pd.Series(0.0, index=SECTOR_ETFS, name="combined_score")

        # Create DataFrame of all signal scores
        scores_df = pd.DataFrame(all_scores)

        # Equal-weighted average
        combined = scores_df.mean(axis=1)
        combined.name = "combined_score"

        if return_individual:
            scores_df["combined_score"] = combined
            return scores_df

        return combined

    def generate_rankings(
        self,
        as_of_date: date,
    ) -> pd.Series:
        """
        Generate sector rankings from combined scores.

        Args:
            as_of_date: Point-in-time date.

        Returns:
            Series of sector rankings (1 = best, 11 = worst).
        """
        scores = self.generate_combined_scores(as_of_date)

        # Rank in descending order (highest score = rank 1)
        rankings = scores.rank(ascending=False).astype(int)
        rankings.name = "ranking"

        return rankings

    def generate_historical_scores(
        self,
        start_date: date,
        end_date: date,
        frequency: str = "W-FRI",
    ) -> pd.DataFrame:
        """
        Generate combined scores for a date range.

        Args:
            start_date: Start date.
            end_date: End date.
            frequency: Resampling frequency.

        Returns:
            DataFrame with dates as index and sectors as columns.
        """
        dates = pd.date_range(start=start_date, end=end_date, freq=frequency)

        results = []
        for dt in dates:
            try:
                scores = self.generate_combined_scores(dt.date())
                scores.name = dt
                results.append(scores)
            except Exception as e:
                print(f"Warning: Could not generate scores for {dt.date()}: {e}")
                scores = pd.Series(0.0, index=SECTOR_ETFS)
                scores.name = dt
                results.append(scores)

        if not results:
            return pd.DataFrame()

        df = pd.DataFrame(results)
        df.index.name = "date"

        return df

    def get_signal_breakdown(
        self,
        as_of_date: date,
    ) -> pd.DataFrame:
        """
        Get breakdown of all individual signal scores.

        Args:
            as_of_date: Point-in-time date.

        Returns:
            DataFrame with sectors as index and signals as columns.
        """
        return self.generate_combined_scores(as_of_date, return_individual=True)

    def get_active_signals(self) -> List[str]:
        """Return names of active signals."""
        return [s.name for s in self.signals]
