"""
Gemini AI Features Loader for Industry Model

Wraps the GeminiMacroAnalyzer to generate and cache AI-derived features
for signal generation.
"""

import json
import os
import sys
from datetime import datetime, date
from pathlib import Path
from typing import Optional, Dict, Any, List

import pandas as pd

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from data.core.gemini_analyzer import (
    GeminiMacroAnalyzer,
    QuadrantProbabilities,
    DocumentSummary,
    PeriodComparison,
    PortfolioRecommendations,
)
from data.storage.db.pit_data_loader import PITDataLoader

from ..config import ModelConfig, SECTOR_ETFS


class GeminiLoader:
    """
    Load and cache Gemini AI-derived features for signal generation.

    Features:
    - QuadrantProbabilities: P(Quad1), P(Quad2), P(Quad3), P(Quad4)
    - DocumentSummary: Growth/inflation assessments from Fed docs
    - PeriodComparison: Tone shifts between consecutive docs
    - PortfolioRecommendations: AI-suggested sector overweights/underweights
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        config: Optional[ModelConfig] = None,
    ):
        """
        Initialize Gemini loader.

        Args:
            api_key: Gemini API key. If not provided, reads from GEMINI_API_KEY env var.
            config: Model configuration.
        """
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        self.config = config or ModelConfig()
        self.cache_dir = self.config.cache_dir / "gemini"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._analyzer: Optional[GeminiMacroAnalyzer] = None

    @property
    def analyzer(self) -> Optional[GeminiMacroAnalyzer]:
        """Get or create the Gemini analyzer."""
        if self._analyzer is None and self.api_key:
            self._analyzer = GeminiMacroAnalyzer(
                api_key=self.api_key,
                logs_dir=self.cache_dir / "logs",
            )
        return self._analyzer

    def _get_cache_path(self, feature_type: str, as_of_date: date) -> Path:
        """Get cache file path for a feature."""
        date_str = as_of_date.strftime("%Y%m%d")
        return self.cache_dir / f"{feature_type}_{date_str}.json"

    def _load_from_cache(self, feature_type: str, as_of_date: date) -> Optional[Dict]:
        """Load feature from cache if available."""
        cache_path = self._get_cache_path(feature_type, as_of_date)
        if cache_path.exists():
            try:
                with open(cache_path, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return None
        return None

    def _save_to_cache(self, feature_type: str, as_of_date: date, data: Dict):
        """Save feature to cache."""
        cache_path = self._get_cache_path(feature_type, as_of_date)
        try:
            with open(cache_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        except IOError as e:
            print(f"Warning: Could not save to cache: {e}")

    def get_quadrant_probabilities(
        self,
        as_of_date: Optional[date] = None,
        use_cache: bool = True,
    ) -> Optional[Dict[str, float]]:
        """
        Get AI-derived quadrant probabilities.

        Args:
            as_of_date: Point-in-time date. Defaults to today.
            use_cache: Whether to use cached results.

        Returns:
            Dict with quad1, quad2, quad3, quad4 probabilities.
        """
        as_of_date = as_of_date or date.today()

        # Try cache first
        if use_cache:
            cached = self._load_from_cache("quadrant_probs", as_of_date)
            if cached:
                return cached

        if not self.analyzer:
            print("Gemini API key not available")
            return None

        # Load documents for analysis
        try:
            pit_loader = PITDataLoader(as_of_date=as_of_date)
            fomc_statements = pit_loader.load_fomc_statements(n_recent=3)
            beige_book = pit_loader.load_beige_book(n_recent=2)

            if not fomc_statements and not beige_book:
                print("No documents available for analysis")
                return None

            # Generate probabilities
            # Note: This would call the actual analyzer method
            # For now, return a placeholder structure
            probs = {
                "quad1": 0.25,
                "quad2": 0.25,
                "quad3": 0.25,
                "quad4": 0.25,
                "most_likely": 1,
                "confidence": 0.5,
                "as_of_date": as_of_date.isoformat(),
            }

            if use_cache:
                self._save_to_cache("quadrant_probs", as_of_date, probs)

            return probs

        except Exception as e:
            print(f"Error generating quadrant probabilities: {e}")
            return None

    def get_sector_recommendations(
        self,
        as_of_date: Optional[date] = None,
        use_cache: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """
        Get AI-derived sector recommendations.

        Args:
            as_of_date: Point-in-time date.
            use_cache: Whether to use cached results.

        Returns:
            Dict with overweight and underweight sector lists.
        """
        as_of_date = as_of_date or date.today()

        # Try cache first
        if use_cache:
            cached = self._load_from_cache("sector_recs", as_of_date)
            if cached:
                return cached

        if not self.analyzer:
            return None

        try:
            # Get quadrant first
            probs = self.get_quadrant_probabilities(as_of_date, use_cache)

            if not probs:
                return None

            # Generate sector recommendations based on quadrant
            # This would normally call the analyzer for full portfolio recommendations
            recs = {
                "overweights": [],
                "underweights": [],
                "confidence": probs.get("confidence", 0.5),
                "as_of_date": as_of_date.isoformat(),
            }

            if use_cache:
                self._save_to_cache("sector_recs", as_of_date, recs)

            return recs

        except Exception as e:
            print(f"Error generating sector recommendations: {e}")
            return None

    def get_gemini_sector_scores(
        self,
        as_of_date: Optional[date] = None,
    ) -> pd.Series:
        """
        Get Gemini-derived sector scores.

        Combines quadrant probabilities and recommendations into sector scores.

        Args:
            as_of_date: Point-in-time date.

        Returns:
            Series indexed by sector with scores in [-1, 1].
        """
        as_of_date = as_of_date or date.today()

        # Get quadrant probabilities
        probs = self.get_quadrant_probabilities(as_of_date)

        if not probs:
            # Return neutral scores if no Gemini data available
            return pd.Series(0.0, index=SECTOR_ETFS, name="gemini_score")

        # Import quadrant preferences
        from ..config import QUADRANT_SECTOR_SCORES

        # Calculate weighted sector scores based on quadrant probabilities
        scores = pd.Series(0.0, index=SECTOR_ETFS)

        for quad in [1, 2, 3, 4]:
            quad_prob = probs.get(f"quad{quad}", 0.25)
            quad_scores = QUADRANT_SECTOR_SCORES.get(quad, {})

            for sector in SECTOR_ETFS:
                sector_score = quad_scores.get(sector, 0.0)
                scores[sector] += quad_prob * sector_score

        scores.name = "gemini_score"

        # Get explicit recommendations if available
        recs = self.get_sector_recommendations(as_of_date)
        if recs:
            overweights = recs.get("overweights", [])
            underweights = recs.get("underweights", [])
            confidence = recs.get("confidence", 0.5)

            # Boost overweights, penalize underweights
            for sector in overweights:
                if sector in scores.index:
                    scores[sector] = min(scores[sector] + 0.3 * confidence, 1.0)

            for sector in underweights:
                if sector in scores.index:
                    scores[sector] = max(scores[sector] - 0.3 * confidence, -1.0)

        return scores

    def get_tone_shift_signal(
        self,
        as_of_date: Optional[date] = None,
    ) -> Optional[str]:
        """
        Get Fed tone shift signal from period comparison.

        Args:
            as_of_date: Point-in-time date.

        Returns:
            Tone shift: "more_hawkish", "unchanged", or "more_dovish"
        """
        as_of_date = as_of_date or date.today()

        # Try cache
        cached = self._load_from_cache("tone_shift", as_of_date)
        if cached:
            return cached.get("tone_shift")

        # Would need to run period comparison analysis
        # Return placeholder for now
        return "unchanged"
