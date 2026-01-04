"""
ETF Data Loader for Industry Model

Fetches historical sector ETF prices from FMP API with local caching.
"""

import json
import os
import hashlib
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any

import pandas as pd
import requests

from ..config import SECTOR_ETFS, ModelConfig


class ETFDataLoader:
    """
    Load historical sector ETF prices from FMP API.

    Features:
    - Fetches daily OHLCV data
    - Caches locally to avoid repeated API calls
    - Calculates weekly returns
    - Computes benchmark (equal-weight) returns
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        config: Optional[ModelConfig] = None,
    ):
        """
        Initialize ETF data loader.

        Args:
            api_key: FMP API key. If not provided, reads from FMP_API_KEY env var.
            config: Model configuration.
        """
        self.api_key = api_key or os.environ.get("FMP_API_KEY")
        if not self.api_key:
            raise ValueError("FMP_API_KEY not found. Set environment variable or pass api_key.")

        self.config = config or ModelConfig()
        self.base_url = self.config.fmp_base_url
        self._cache: Dict[str, pd.DataFrame] = {}

    def _get_cache_path(self, symbol: str) -> Path:
        """Get cache file path for a symbol."""
        return self.config.cache_dir / f"etf_{symbol.lower()}_daily.parquet"

    def _fetch_from_api(self, symbol: str) -> pd.DataFrame:
        """
        Fetch historical prices from FMP API.

        Uses stable/historical-price-eod/full endpoint.
        """
        url = f"{self.base_url}/historical-price-eod/full"
        params = {
            "symbol": symbol,
            "apikey": self.api_key,
        }

        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as e:
            print(f"Error fetching {symbol}: {e}")
            return pd.DataFrame()

        if not data:
            print(f"No data returned for {symbol}")
            return pd.DataFrame()

        # Handle both list format and dict with "historical" key
        if isinstance(data, list):
            historical_data = data
        elif isinstance(data, dict) and "historical" in data:
            historical_data = data["historical"]
        else:
            print(f"Unexpected data format for {symbol}")
            return pd.DataFrame()

        # Parse historical data
        records = []
        for item in historical_data:
            records.append({
                "date": pd.to_datetime(item["date"]),
                "open": item["open"],
                "high": item["high"],
                "low": item["low"],
                "close": item["close"],
                "adj_close": item.get("adjClose", item["close"]),
                "volume": item["volume"],
                "symbol": symbol,
            })

        df = pd.DataFrame(records)
        if not df.empty:
            df = df.sort_values("date").set_index("date")

        return df

    def load_etf_prices(
        self,
        symbols: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        use_cache: bool = True,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """
        Load historical prices for sector ETFs.

        Args:
            symbols: List of ETF symbols. Defaults to all sector ETFs.
            start_date: Start date (YYYY-MM-DD). Defaults to earliest available.
            end_date: End date (YYYY-MM-DD). Defaults to today.
            use_cache: Whether to use cached data.
            force_refresh: Force refresh from API even if cache exists.

        Returns:
            DataFrame with MultiIndex (date, symbol) and columns:
            [open, high, low, close, adj_close, volume]
        """
        symbols = symbols or SECTOR_ETFS
        all_data = []

        for symbol in symbols:
            cache_path = self._get_cache_path(symbol)

            # Try cache first
            if use_cache and not force_refresh and cache_path.exists():
                df = pd.read_parquet(cache_path)
                print(f"Loaded {symbol} from cache ({len(df)} rows)")
            else:
                # Fetch from API
                print(f"Fetching {symbol} from FMP API...")
                df = self._fetch_from_api(symbol)

                if not df.empty and use_cache:
                    # Save to cache
                    df.to_parquet(cache_path)
                    print(f"Cached {symbol} ({len(df)} rows)")

            if not df.empty:
                all_data.append(df)

        if not all_data:
            return pd.DataFrame()

        # Combine all ETF data
        combined = pd.concat(all_data)

        # Filter by date range
        if start_date:
            combined = combined[combined.index >= pd.to_datetime(start_date)]
        if end_date:
            combined = combined[combined.index <= pd.to_datetime(end_date)]

        return combined

    def get_prices_wide(
        self,
        price_col: str = "adj_close",
        **kwargs
    ) -> pd.DataFrame:
        """
        Get prices in wide format (date x symbols).

        Args:
            price_col: Column to use for prices.
            **kwargs: Passed to load_etf_prices.

        Returns:
            DataFrame indexed by date with columns as symbols.
        """
        df = self.load_etf_prices(**kwargs)
        if df.empty:
            return pd.DataFrame()

        return df.reset_index().pivot(
            index="date",
            columns="symbol",
            values=price_col
        )

    def calculate_returns(
        self,
        prices: Optional[pd.DataFrame] = None,
        frequency: str = "W",
        log_returns: bool = False,
        **kwargs
    ) -> pd.DataFrame:
        """
        Calculate returns at specified frequency.

        Args:
            prices: Wide-format prices DataFrame. If None, loads from API.
            frequency: Return frequency - 'D' (daily), 'W' (weekly), 'M' (monthly).
            log_returns: If True, calculate log returns instead of simple returns.
            **kwargs: Passed to get_prices_wide if prices is None.

        Returns:
            DataFrame of returns indexed by date with columns as symbols.
        """
        if prices is None:
            prices = self.get_prices_wide(**kwargs)

        if prices.empty:
            return pd.DataFrame()

        # Resample to frequency if needed
        if frequency != "D":
            prices = prices.resample(frequency).last()

        # Calculate returns
        if log_returns:
            import numpy as np
            returns = np.log(prices / prices.shift(1))
        else:
            returns = prices.pct_change()

        return returns.dropna(how="all")

    def calculate_benchmark_returns(
        self,
        returns: Optional[pd.DataFrame] = None,
        **kwargs
    ) -> pd.Series:
        """
        Calculate equal-weight benchmark returns.

        Args:
            returns: Returns DataFrame. If None, calculates from prices.
            **kwargs: Passed to calculate_returns if returns is None.

        Returns:
            Series of equal-weight benchmark returns.
        """
        if returns is None:
            returns = self.calculate_returns(**kwargs)

        if returns.empty:
            return pd.Series()

        # Equal weight across all sectors
        benchmark = returns.mean(axis=1)
        benchmark.name = "benchmark"

        return benchmark

    def calculate_relative_returns(
        self,
        returns: Optional[pd.DataFrame] = None,
        benchmark: Optional[pd.Series] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Calculate returns relative to benchmark.

        Args:
            returns: Returns DataFrame. If None, calculates from prices.
            benchmark: Benchmark returns. If None, uses equal-weight.
            **kwargs: Passed to calculate_returns if returns is None.

        Returns:
            DataFrame of excess returns (sector return - benchmark return).
        """
        if returns is None:
            returns = self.calculate_returns(**kwargs)

        if benchmark is None:
            benchmark = self.calculate_benchmark_returns(returns=returns)

        if returns.empty or benchmark.empty:
            return pd.DataFrame()

        # Subtract benchmark from each sector
        relative = returns.subtract(benchmark, axis=0)

        return relative

    def get_data_range(self, **kwargs) -> tuple:
        """
        Get the date range of available data.

        Returns:
            Tuple of (start_date, end_date, n_weeks)
        """
        prices = self.get_prices_wide(**kwargs)

        if prices.empty:
            return None, None, 0

        start = prices.index.min()
        end = prices.index.max()
        n_weeks = len(prices.resample("W").last().dropna())

        return start, end, n_weeks


def load_sector_prices(
    api_key: Optional[str] = None,
    frequency: str = "W",
    **kwargs
) -> tuple:
    """
    Convenience function to load sector ETF data.

    Args:
        api_key: FMP API key.
        frequency: Return frequency.
        **kwargs: Passed to ETFDataLoader methods.

    Returns:
        Tuple of (prices, returns, relative_returns, benchmark)
    """
    loader = ETFDataLoader(api_key=api_key)

    prices = loader.get_prices_wide(**kwargs)
    returns = loader.calculate_returns(prices=prices, frequency=frequency)
    benchmark = loader.calculate_benchmark_returns(returns=returns)
    relative = loader.calculate_relative_returns(returns=returns, benchmark=benchmark)

    return prices, returns, relative, benchmark
