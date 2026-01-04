"""
Macro Feature Loader for Industry Model

Wraps the existing PITDataLoader to provide PIT-compliant macro features
aligned to weekly frequency for signal generation.
"""

import sys
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Union

import pandas as pd
import numpy as np

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from data.storage.db.pit_data_loader import PITDataLoader
from data.storage.db.pit_database import PITDatabase

from ..config import ModelConfig


class MacroFeatureLoader:
    """
    Load macro features from PIT database with proper temporal alignment.

    Features:
    - Wraps PITDataLoader for PIT-compliant queries
    - Aligns monthly/quarterly data to weekly frequency
    - Calculates rates of change for growth/inflation signals
    - Ensures no look-ahead bias
    """

    def __init__(self, config: Optional[ModelConfig] = None):
        """
        Initialize macro feature loader.

        Args:
            config: Model configuration.
        """
        self.config = config or ModelConfig()
        self._db = PITDatabase()
        self._cache: Dict[str, pd.DataFrame] = {}

    def load_series_as_of(
        self,
        series_id: str,
        as_of_date: Union[str, date, datetime],
    ) -> pd.DataFrame:
        """
        Load a single FRED series with PIT filtering.

        Args:
            series_id: FRED series ID (e.g., "CPIAUCSL")
            as_of_date: Point-in-time date. Only data available on this date is returned.

        Returns:
            DataFrame indexed by observation date with 'value' column.
        """
        if isinstance(as_of_date, str):
            as_of_date = datetime.strptime(as_of_date, "%Y-%m-%d").date()
        elif isinstance(as_of_date, datetime):
            as_of_date = as_of_date.date()

        loader = PITDataLoader(as_of_date=as_of_date)
        df = loader.load_fred_series(series_id)

        return df

    def load_all_features(
        self,
        as_of_date: Union[str, date, datetime],
        series_list: Optional[List[str]] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Load all FRED series for signal generation.

        Args:
            as_of_date: Point-in-time date.
            series_list: List of series to load. Defaults to all configured series.

        Returns:
            Dict mapping series_id to DataFrame.
        """
        series_list = series_list or self.config.all_fred_series

        loader = PITDataLoader(as_of_date=as_of_date)
        result = {}

        for series_id in series_list:
            try:
                df = loader.load_fred_series(series_id)
                if not df.empty:
                    result[series_id] = df
            except Exception as e:
                print(f"Warning: Could not load {series_id}: {e}")

        return result

    def get_weekly_features(
        self,
        as_of_date: Union[str, date, datetime],
        series_list: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Get all features aligned to weekly frequency.

        Uses forward-fill to propagate monthly/quarterly values to weekly dates.

        Args:
            as_of_date: Point-in-time date.
            series_list: List of series to load.

        Returns:
            DataFrame indexed by weekly date with columns as series.
        """
        features = self.load_all_features(as_of_date, series_list)

        if not features:
            return pd.DataFrame()

        # Combine all series
        combined = {}
        for series_id, df in features.items():
            if "value" in df.columns:
                combined[series_id] = df["value"]

        if not combined:
            return pd.DataFrame()

        # Create wide DataFrame
        wide = pd.DataFrame(combined)

        # Resample to weekly (Friday close) using forward-fill
        weekly = wide.resample("W-FRI").last().ffill()

        return weekly

    def calculate_rate_of_change(
        self,
        series: pd.Series,
        periods: int = 12,
        annualize: bool = False,
    ) -> pd.Series:
        """
        Calculate rate of change (percent change) over specified periods.

        Args:
            series: Time series data.
            periods: Number of periods for change calculation.
            annualize: If True, annualize the change.

        Returns:
            Rate of change series.
        """
        roc = series.pct_change(periods=periods)

        if annualize:
            # Assume weekly data, annualize
            roc = roc * (52 / periods)

        return roc

    def get_growth_momentum(
        self,
        as_of_date: Union[str, date, datetime],
        periods: int = None,
    ) -> pd.Series:
        """
        Calculate composite growth momentum indicator.

        Uses INDPRO as primary growth proxy.

        Args:
            as_of_date: Point-in-time date.
            periods: Periods for rate of change.

        Returns:
            Series of growth momentum values.
        """
        periods = periods or self.config.growth_roc_periods

        features = self.get_weekly_features(
            as_of_date,
            series_list=self.config.growth_series
        )

        if features.empty or "INDPRO" not in features.columns:
            return pd.Series()

        growth_roc = self.calculate_rate_of_change(features["INDPRO"], periods)
        growth_roc.name = "growth_momentum"

        return growth_roc

    def get_inflation_momentum(
        self,
        as_of_date: Union[str, date, datetime],
        periods: int = None,
    ) -> pd.Series:
        """
        Calculate composite inflation momentum indicator.

        Uses CPIAUCSL as primary inflation proxy.

        Args:
            as_of_date: Point-in-time date.
            periods: Periods for rate of change.

        Returns:
            Series of inflation momentum values.
        """
        periods = periods or self.config.inflation_roc_periods

        features = self.get_weekly_features(
            as_of_date,
            series_list=self.config.inflation_series
        )

        if features.empty or "CPIAUCSL" not in features.columns:
            return pd.Series()

        inflation_roc = self.calculate_rate_of_change(features["CPIAUCSL"], periods)
        inflation_roc.name = "inflation_momentum"

        return inflation_roc

    def get_yield_curve_spread(
        self,
        as_of_date: Union[str, date, datetime],
    ) -> pd.Series:
        """
        Get yield curve spread (10Y - 2Y).

        Args:
            as_of_date: Point-in-time date.

        Returns:
            Series of yield curve spread values.
        """
        features = self.get_weekly_features(
            as_of_date,
            series_list=["T10Y2Y"]
        )

        if features.empty or "T10Y2Y" not in features.columns:
            # Try to calculate from raw yields
            features = self.get_weekly_features(
                as_of_date,
                series_list=["DGS10", "DGS2"]
            )
            if "DGS10" in features.columns and "DGS2" in features.columns:
                spread = features["DGS10"] - features["DGS2"]
                spread.name = "yield_curve_spread"
                return spread
            return pd.Series()

        spread = features["T10Y2Y"]
        spread.name = "yield_curve_spread"

        return spread

    def get_credit_spread(
        self,
        as_of_date: Union[str, date, datetime],
    ) -> pd.Series:
        """
        Get high-yield credit spread.

        Args:
            as_of_date: Point-in-time date.

        Returns:
            Series of credit spread values (BAMLH0A0HYM2).
        """
        features = self.get_weekly_features(
            as_of_date,
            series_list=["BAMLH0A0HYM2"]
        )

        if features.empty or "BAMLH0A0HYM2" not in features.columns:
            return pd.Series()

        spread = features["BAMLH0A0HYM2"]
        spread.name = "credit_spread"

        return spread

    def get_financial_conditions(
        self,
        as_of_date: Union[str, date, datetime],
    ) -> pd.Series:
        """
        Get Chicago Fed National Financial Conditions Index.

        Args:
            as_of_date: Point-in-time date.

        Returns:
            Series of NFCI values. Negative = loose, Positive = tight.
        """
        features = self.get_weekly_features(
            as_of_date,
            series_list=["NFCI"]
        )

        if features.empty or "NFCI" not in features.columns:
            return pd.Series()

        nfci = features["NFCI"]
        nfci.name = "financial_conditions"

        return nfci

    def get_vix(
        self,
        as_of_date: Union[str, date, datetime],
    ) -> pd.Series:
        """
        Get VIX volatility index.

        Args:
            as_of_date: Point-in-time date.

        Returns:
            Series of VIX values.
        """
        features = self.get_weekly_features(
            as_of_date,
            series_list=["VIXCLS"]
        )

        if features.empty or "VIXCLS" not in features.columns:
            return pd.Series()

        vix = features["VIXCLS"]
        vix.name = "vix"

        return vix

    def get_dollar_index(
        self,
        as_of_date: Union[str, date, datetime],
    ) -> pd.Series:
        """
        Get trade-weighted dollar index.

        Args:
            as_of_date: Point-in-time date.

        Returns:
            Series of dollar index values.
        """
        features = self.get_weekly_features(
            as_of_date,
            series_list=["DTWEXBGS"]
        )

        if features.empty or "DTWEXBGS" not in features.columns:
            return pd.Series()

        dollar = features["DTWEXBGS"]
        dollar.name = "dollar_index"

        return dollar

    def get_housing_starts(
        self,
        as_of_date: Union[str, date, datetime],
    ) -> pd.Series:
        """
        Get housing starts.

        Args:
            as_of_date: Point-in-time date.

        Returns:
            Series of housing starts values.
        """
        features = self.get_weekly_features(
            as_of_date,
            series_list=["HOUST"]
        )

        if features.empty or "HOUST" not in features.columns:
            return pd.Series()

        houst = features["HOUST"]
        houst.name = "housing_starts"

        return houst

    def get_consumer_sentiment(
        self,
        as_of_date: Union[str, date, datetime],
    ) -> pd.Series:
        """
        Get University of Michigan Consumer Sentiment.

        Args:
            as_of_date: Point-in-time date.

        Returns:
            Series of consumer sentiment values.
        """
        features = self.get_weekly_features(
            as_of_date,
            series_list=["UMCSENT"]
        )

        if features.empty or "UMCSENT" not in features.columns:
            return pd.Series()

        sent = features["UMCSENT"]
        sent.name = "consumer_sentiment"

        return sent

    def get_fed_balance_sheet(
        self,
        as_of_date: Union[str, date, datetime],
    ) -> pd.Series:
        """
        Get Fed balance sheet total assets.

        Args:
            as_of_date: Point-in-time date.

        Returns:
            Series of Fed balance sheet values.
        """
        features = self.get_weekly_features(
            as_of_date,
            series_list=["WALCL"]
        )

        if features.empty or "WALCL" not in features.columns:
            return pd.Series()

        walcl = features["WALCL"]
        walcl.name = "fed_balance_sheet"

        return walcl

    def get_initial_claims(
        self,
        as_of_date: Union[str, date, datetime],
    ) -> pd.Series:
        """
        Get initial jobless claims.

        Args:
            as_of_date: Point-in-time date.

        Returns:
            Series of initial claims values.
        """
        features = self.get_weekly_features(
            as_of_date,
            series_list=["ICSA"]
        )

        if features.empty or "ICSA" not in features.columns:
            return pd.Series()

        icsa = features["ICSA"]
        icsa.name = "initial_claims"

        return icsa

    def determine_quadrant(
        self,
        as_of_date: Union[str, date, datetime],
    ) -> int:
        """
        Determine current macro quadrant based on growth and inflation momentum.

        Args:
            as_of_date: Point-in-time date.

        Returns:
            Quadrant number (1-4).
        """
        growth = self.get_growth_momentum(as_of_date)
        inflation = self.get_inflation_momentum(as_of_date)

        if growth.empty or inflation.empty:
            return 0  # Unknown

        # Get latest values
        latest_growth = growth.iloc[-1] if len(growth) > 0 else 0
        latest_inflation = inflation.iloc[-1] if len(inflation) > 0 else 0

        # Determine quadrant
        if latest_growth > 0 and latest_inflation <= 0:
            return 1  # Growth up, Inflation down
        elif latest_growth > 0 and latest_inflation > 0:
            return 2  # Growth up, Inflation up
        elif latest_growth <= 0 and latest_inflation <= 0:
            return 3  # Growth down, Inflation down
        else:
            return 4  # Growth down, Inflation up
