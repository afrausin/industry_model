"""
Portfolio Weights for Industry Model

Converts sector scores/rankings into portfolio weights with constraints.
"""

from datetime import date
from typing import Optional

import pandas as pd
import numpy as np

from ..config import SECTOR_ETFS, ModelConfig
from .combiner import SignalCombiner


class PortfolioWeights:
    """
    Convert sector scores into portfolio weights.

    Approach: Rank-based tilting with constraints
    - Higher combined score = higher weight
    - Constraints: 5% min, 20% max per sector
    - Benchmark: Equal-weight (9.09% each sector)
    """

    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        combiner: Optional[SignalCombiner] = None,
        gemini_api_key: Optional[str] = None,
    ):
        """
        Initialize portfolio weights calculator.

        Args:
            config: Model configuration.
            combiner: Signal combiner. If None, creates new one.
            gemini_api_key: API key for Gemini signal.
        """
        self.config = config or ModelConfig()
        self.combiner = combiner or SignalCombiner(
            config=self.config,
            gemini_api_key=gemini_api_key,
        )
        self.n_sectors = len(SECTOR_ETFS)
        self.benchmark_weight = 1.0 / self.n_sectors

    def scores_to_weights(
        self,
        scores: pd.Series,
        method: str = "rank_tilt",
    ) -> pd.Series:
        """
        Convert sector scores to portfolio weights.

        Args:
            scores: Sector scores in [-1, +1].
            method: Weighting method - "rank_tilt" or "score_proportional".

        Returns:
            Series of portfolio weights summing to 1.
        """
        if scores.empty:
            return self._benchmark_weights()

        if method == "rank_tilt":
            weights = self._rank_tilt_weights(scores)
        elif method == "score_proportional":
            weights = self._score_proportional_weights(scores)
        else:
            raise ValueError(f"Unknown method: {method}")

        # Apply constraints
        weights = self._apply_constraints(weights)

        return weights

    def _rank_tilt_weights(self, scores: pd.Series) -> pd.Series:
        """
        Calculate weights based on score rankings.

        Uses linear tilting from benchmark based on rank position.
        """
        # Rank sectors (1 = best)
        rankings = scores.rank(ascending=False)

        # Calculate tilt from benchmark
        # Rank 1 gets max tilt, Rank 11 gets min tilt
        # Linear scale from +0.05 to -0.05 relative to benchmark
        max_tilt = self.config.max_sector_weight - self.benchmark_weight
        min_tilt = self.config.min_sector_weight - self.benchmark_weight

        # Linear interpolation based on rank
        tilts = rankings.apply(
            lambda r: max_tilt - (max_tilt - min_tilt) * (r - 1) / (self.n_sectors - 1)
        )

        # Apply tilts to benchmark
        weights = self.benchmark_weight + tilts

        # Normalize to sum to 1
        weights = weights / weights.sum()
        weights.name = "weight"

        return weights

    def _score_proportional_weights(self, scores: pd.Series) -> pd.Series:
        """
        Calculate weights proportional to scores.

        Shifts scores to positive range and normalizes.
        """
        # Shift to positive (scores are in [-1, +1], shift to [0, 2])
        shifted = scores + 1.0

        # Prevent divide by zero
        if shifted.sum() == 0:
            return self._benchmark_weights()

        # Normalize
        weights = shifted / shifted.sum()
        weights.name = "weight"

        return weights

    def _apply_constraints(self, weights: pd.Series) -> pd.Series:
        """
        Apply position constraints to weights.

        Min: 5%, Max: 20%
        """
        min_weight = self.config.min_sector_weight
        max_weight = self.config.max_sector_weight

        # Iterative constraint application
        for _ in range(10):  # Max iterations
            # Clip weights
            clipped = weights.clip(lower=min_weight, upper=max_weight)

            # Calculate excess/deficit
            total = clipped.sum()

            if abs(total - 1.0) < 0.0001:
                break

            # Redistribute
            if total > 1.0:
                # Reduce weights proportionally for unconstrained sectors
                excess = total - 1.0
                unconstrained = clipped[(clipped > min_weight) & (clipped < max_weight)]
                if not unconstrained.empty:
                    reduction = excess / len(unconstrained)
                    clipped[unconstrained.index] -= reduction
            else:
                # Increase weights proportionally for unconstrained sectors
                deficit = 1.0 - total
                unconstrained = clipped[(clipped > min_weight) & (clipped < max_weight)]
                if not unconstrained.empty:
                    addition = deficit / len(unconstrained)
                    clipped[unconstrained.index] += addition

            weights = clipped

        # Final normalization
        weights = weights / weights.sum()
        weights.name = "weight"

        return weights

    def _benchmark_weights(self) -> pd.Series:
        """Return equal-weight benchmark."""
        return pd.Series(
            self.benchmark_weight,
            index=SECTOR_ETFS,
            name="weight",
        )

    def generate_weights(
        self,
        as_of_date: date,
        method: str = "rank_tilt",
    ) -> pd.Series:
        """
        Generate portfolio weights for a given date.

        Args:
            as_of_date: Point-in-time date.
            method: Weighting method.

        Returns:
            Series of portfolio weights.
        """
        scores = self.combiner.generate_combined_scores(as_of_date)
        return self.scores_to_weights(scores, method=method)

    def generate_historical_weights(
        self,
        start_date: date,
        end_date: date,
        frequency: str = "W-FRI",
        method: str = "rank_tilt",
    ) -> pd.DataFrame:
        """
        Generate portfolio weights for a date range.

        Args:
            start_date: Start date.
            end_date: End date.
            frequency: Resampling frequency.
            method: Weighting method.

        Returns:
            DataFrame with dates as index and sectors as columns.
        """
        scores_df = self.combiner.generate_historical_scores(
            start_date, end_date, frequency
        )

        if scores_df.empty:
            return pd.DataFrame()

        weights_list = []
        for dt in scores_df.index:
            scores = scores_df.loc[dt]
            weights = self.scores_to_weights(scores, method=method)
            weights.name = dt
            weights_list.append(weights)

        weights_df = pd.DataFrame(weights_list)
        weights_df.index.name = "date"

        return weights_df

    def get_active_weights(
        self,
        weights: pd.Series,
    ) -> pd.Series:
        """
        Calculate active weights (deviation from benchmark).

        Args:
            weights: Portfolio weights.

        Returns:
            Series of active weights (positive = overweight).
        """
        active = weights - self.benchmark_weight
        active.name = "active_weight"
        return active

    def calculate_turnover(
        self,
        weights_df: pd.DataFrame,
    ) -> pd.Series:
        """
        Calculate portfolio turnover over time.

        Args:
            weights_df: Historical weights DataFrame.

        Returns:
            Series of turnover values (sum of absolute weight changes).
        """
        if weights_df.empty:
            return pd.Series()

        # Calculate absolute changes
        turnover = weights_df.diff().abs().sum(axis=1) / 2  # Divide by 2 for one-way
        turnover.name = "turnover"

        return turnover
