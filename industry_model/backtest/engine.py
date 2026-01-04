"""
Backtest Engine for Industry Model

VectorBT-based backtesting with transaction costs and slippage.
"""

from datetime import date
from typing import Optional, Tuple, Dict, Any

import pandas as pd
import numpy as np

try:
    import vectorbt as vbt
    HAS_VECTORBT = True
except ImportError:
    HAS_VECTORBT = False
    print("Warning: vectorbt not installed. Install with: pip install vectorbt")

from ..config import SECTOR_ETFS, ModelConfig
from ..data.etf_loader import ETFDataLoader
from ..portfolio.weights import PortfolioWeights


class BacktestEngine:
    """
    Backtest engine using VectorBT.

    Features:
    - Weekly rebalancing
    - Transaction costs (10 bps)
    - Slippage (5 bps)
    - Benchmark comparison (equal-weight)
    """

    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        etf_loader: Optional[ETFDataLoader] = None,
        portfolio_weights: Optional[PortfolioWeights] = None,
        fmp_api_key: Optional[str] = None,
        gemini_api_key: Optional[str] = None,
    ):
        """
        Initialize backtest engine.

        Args:
            config: Model configuration.
            etf_loader: ETF data loader.
            portfolio_weights: Portfolio weights generator.
            fmp_api_key: FMP API key for ETF data.
            gemini_api_key: Gemini API key for AI signals.
        """
        if not HAS_VECTORBT:
            raise ImportError("vectorbt is required for backtesting")

        self.config = config or ModelConfig()

        # Initialize ETF loader
        self.etf_loader = etf_loader or ETFDataLoader(
            api_key=fmp_api_key,
            config=self.config,
        )

        # Initialize portfolio weights
        self.portfolio_weights = portfolio_weights or PortfolioWeights(
            config=self.config,
            gemini_api_key=gemini_api_key,
        )

        # Cache for data
        self._prices: Optional[pd.DataFrame] = None
        self._returns: Optional[pd.DataFrame] = None

    def load_data(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        force_refresh: bool = False,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load ETF price data.

        Args:
            start_date: Start date (YYYY-MM-DD).
            end_date: End date (YYYY-MM-DD).
            force_refresh: Force refresh from API.

        Returns:
            Tuple of (prices_df, returns_df).
        """
        if self._prices is None or force_refresh:
            self._prices = self.etf_loader.get_prices_wide(
                start_date=start_date,
                end_date=end_date,
                use_cache=True,
                force_refresh=force_refresh,
            )

            # Calculate weekly returns
            self._returns = self.etf_loader.calculate_returns(
                prices=self._prices,
                frequency="W",
            )

        return self._prices, self._returns

    def run_backtest(
        self,
        start_date: str,
        end_date: str,
        initial_capital: float = 100000.0,
        method: str = "rank_tilt",
    ) -> Dict[str, Any]:
        """
        Run backtest for the strategy.

        Args:
            start_date: Start date (YYYY-MM-DD).
            end_date: End date (YYYY-MM-DD).
            initial_capital: Starting capital.
            method: Portfolio weighting method.

        Returns:
            Dict with backtest results.
        """
        # Load data
        prices, returns = self.load_data(start_date=start_date, end_date=end_date)

        if prices.empty or returns.empty:
            raise ValueError("No data available for backtesting")

        # Generate weights for all dates
        weights_df = self.portfolio_weights.generate_historical_weights(
            start_date=pd.Timestamp(start_date).date(),
            end_date=pd.Timestamp(end_date).date(),
            frequency="W-FRI",
            method=method,
        )

        if weights_df.empty:
            raise ValueError("No weights generated")

        # Align weights with returns
        weights_df, returns_aligned = self._align_data(weights_df, returns)

        # Calculate strategy returns
        strategy_returns = self._calculate_portfolio_returns(
            weights_df, returns_aligned
        )

        # Calculate benchmark returns (equal-weight)
        benchmark_returns = returns_aligned.mean(axis=1)

        # Calculate metrics
        strategy_metrics = self._calculate_metrics(strategy_returns, "strategy")
        benchmark_metrics = self._calculate_metrics(benchmark_returns, "benchmark")

        # Calculate relative metrics
        excess_returns = strategy_returns - benchmark_returns
        relative_metrics = self._calculate_metrics(excess_returns, "relative")

        # Calculate turnover
        turnover = self.portfolio_weights.calculate_turnover(weights_df)

        return {
            "strategy_returns": strategy_returns,
            "benchmark_returns": benchmark_returns,
            "excess_returns": excess_returns,
            "weights": weights_df,
            "turnover": turnover,
            "strategy_metrics": strategy_metrics,
            "benchmark_metrics": benchmark_metrics,
            "relative_metrics": relative_metrics,
            "start_date": start_date,
            "end_date": end_date,
        }

    def _align_data(
        self,
        weights: pd.DataFrame,
        returns: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Align weights and returns to common dates."""
        # Ensure both have DatetimeIndex
        if not isinstance(weights.index, pd.DatetimeIndex):
            weights.index = pd.to_datetime(weights.index)
        if not isinstance(returns.index, pd.DatetimeIndex):
            returns.index = pd.to_datetime(returns.index)

        # Get common dates
        common_dates = weights.index.intersection(returns.index)

        if len(common_dates) == 0:
            # Try forward fill alignment
            weights_reindexed = weights.reindex(returns.index, method="ffill")
            common_dates = weights_reindexed.dropna(how="all").index.intersection(returns.index)

            if len(common_dates) == 0:
                # Try backward fill as fallback
                weights_reindexed = weights.reindex(returns.index, method="bfill")
                common_dates = weights_reindexed.dropna(how="all").index.intersection(returns.index)

            if len(common_dates) == 0:
                # Last resort: use merge_asof approach
                weights_reset = weights.reset_index()
                returns_reset = returns.reset_index()

                weights_reset.columns = ["date"] + list(weights.columns)
                returns_reset.columns = ["date"] + list(returns.columns)

                merged = pd.merge_asof(
                    returns_reset.sort_values("date"),
                    weights_reset.sort_values("date"),
                    on="date",
                    direction="backward",
                )

                if merged.empty or merged[weights.columns].isna().all().all():
                    raise ValueError("No overlapping dates between weights and returns")

                # Extract aligned data
                merged = merged.set_index("date").dropna()
                weight_cols = [c for c in merged.columns if c.endswith("_y") or c in weights.columns]
                return_cols = [c for c in merged.columns if c.endswith("_x") or c in returns.columns]

                # Simplified return
                return weights.reindex(merged.index, method="ffill").dropna(), returns.loc[merged.index]

            return weights_reindexed.loc[common_dates], returns.loc[common_dates]

        return weights.loc[common_dates], returns.loc[common_dates]

    def _calculate_portfolio_returns(
        self,
        weights: pd.DataFrame,
        returns: pd.DataFrame,
    ) -> pd.Series:
        """
        Calculate portfolio returns with transaction costs.

        Args:
            weights: Portfolio weights.
            returns: Asset returns.

        Returns:
            Series of portfolio returns.
        """
        # Get common columns
        common_cols = weights.columns.intersection(returns.columns)
        weights = weights[common_cols]
        returns = returns[common_cols]

        # Calculate raw portfolio returns
        portfolio_returns = (weights.shift(1) * returns).sum(axis=1)

        # Calculate transaction costs
        weight_changes = weights.diff().abs().sum(axis=1) / 2  # One-way turnover
        total_cost_bps = self.config.transaction_cost_bps + self.config.slippage_bps
        transaction_costs = weight_changes * total_cost_bps / 10000

        # Net returns
        net_returns = portfolio_returns - transaction_costs

        net_returns.name = "portfolio_return"
        return net_returns.dropna()

    def _calculate_metrics(
        self,
        returns: pd.Series,
        name: str,
    ) -> Dict[str, float]:
        """
        Calculate performance metrics.

        Args:
            returns: Return series.
            name: Metric name prefix.

        Returns:
            Dict of performance metrics.
        """
        if returns.empty:
            return {}

        # Annualize (assuming weekly returns)
        ann_factor = 52

        # Basic metrics
        total_return = (1 + returns).prod() - 1
        ann_return = (1 + total_return) ** (ann_factor / len(returns)) - 1
        ann_vol = returns.std() * np.sqrt(ann_factor)
        sharpe = ann_return / ann_vol if ann_vol > 0 else 0

        # Drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()

        # Hit rate
        hit_rate = (returns > 0).mean()

        return {
            f"{name}_total_return": total_return,
            f"{name}_ann_return": ann_return,
            f"{name}_ann_vol": ann_vol,
            f"{name}_sharpe": sharpe,
            f"{name}_max_drawdown": max_drawdown,
            f"{name}_hit_rate": hit_rate,
            f"{name}_n_periods": len(returns),
        }

    def calculate_information_ratio(
        self,
        strategy_returns: pd.Series,
        benchmark_returns: pd.Series,
    ) -> float:
        """
        Calculate Information Ratio.

        Args:
            strategy_returns: Strategy return series.
            benchmark_returns: Benchmark return series.

        Returns:
            Information Ratio.
        """
        excess = strategy_returns - benchmark_returns
        ann_factor = 52

        ann_excess = excess.mean() * ann_factor
        tracking_error = excess.std() * np.sqrt(ann_factor)

        if tracking_error > 0:
            return ann_excess / tracking_error
        return 0.0

    def print_results(self, results: Dict[str, Any]) -> None:
        """Print backtest results summary."""
        print("\n" + "=" * 60)
        print("BACKTEST RESULTS")
        print("=" * 60)
        print(f"Period: {results['start_date']} to {results['end_date']}")
        print(f"N Periods: {results['strategy_metrics'].get('strategy_n_periods', 0)}")

        print("\n--- Strategy ---")
        for key, value in results["strategy_metrics"].items():
            if "return" in key or "vol" in key:
                print(f"{key}: {value:.2%}")
            elif "sharpe" in key or "ratio" in key:
                print(f"{key}: {value:.2f}")
            else:
                print(f"{key}: {value:.4f}")

        print("\n--- Benchmark (Equal-Weight) ---")
        for key, value in results["benchmark_metrics"].items():
            if "return" in key or "vol" in key:
                print(f"{key}: {value:.2%}")
            elif "sharpe" in key or "ratio" in key:
                print(f"{key}: {value:.2f}")
            else:
                print(f"{key}: {value:.4f}")

        # Information Ratio
        ir = self.calculate_information_ratio(
            results["strategy_returns"],
            results["benchmark_returns"],
        )
        print(f"\n--- Relative ---")
        print(f"Information Ratio: {ir:.2f}")

        # Turnover
        avg_turnover = results["turnover"].mean()
        print(f"Avg Weekly Turnover: {avg_turnover:.2%}")

        print("=" * 60)
