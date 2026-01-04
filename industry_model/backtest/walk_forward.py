"""
Walk-Forward Analysis for Industry Model

Rolling in-sample/out-of-sample validation with dynamic window sizing.
"""

from datetime import date, timedelta
from typing import List, Dict, Any, Optional, Tuple

import pandas as pd
import numpy as np

from ..config import ModelConfig
from .engine import BacktestEngine


class WalkForwardAnalyzer:
    """
    Walk-forward validation with dynamic window sizing.

    Dynamic window sizing based on available data:
    - Determine optimal IS/OOS ratio: IS = 80%, OOS = 20%
    - Minimum window size: 26 weeks (6 months)
    - Maximum IS period: 156 weeks (3 years)
    """

    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        backtest_engine: Optional[BacktestEngine] = None,
        fmp_api_key: Optional[str] = None,
        gemini_api_key: Optional[str] = None,
    ):
        """
        Initialize walk-forward analyzer.

        Args:
            config: Model configuration.
            backtest_engine: Backtest engine.
            fmp_api_key: FMP API key.
            gemini_api_key: Gemini API key.
        """
        self.config = config or ModelConfig()
        self.backtest_engine = backtest_engine or BacktestEngine(
            config=self.config,
            fmp_api_key=fmp_api_key,
            gemini_api_key=gemini_api_key,
        )

    def determine_windows(
        self,
        start_date: date,
        end_date: date,
        n_windows: int = 5,
    ) -> List[Dict[str, date]]:
        """
        Determine walk-forward windows based on available data.

        Args:
            start_date: Start of data.
            end_date: End of data.
            n_windows: Target number of windows.

        Returns:
            List of dicts with is_start, is_end, oos_start, oos_end.
        """
        total_days = (end_date - start_date).days
        total_weeks = total_days // 7

        if total_weeks < self.config.min_window_weeks * 2:
            raise ValueError(
                f"Not enough data for walk-forward. "
                f"Need at least {self.config.min_window_weeks * 2} weeks, "
                f"have {total_weeks} weeks."
            )

        # Calculate window size
        window_size = total_weeks // n_windows
        window_size = max(window_size, self.config.min_window_weeks)

        # Calculate IS/OOS split
        is_weeks = int(window_size * self.config.is_oos_ratio)
        is_weeks = min(is_weeks, self.config.max_is_weeks)
        oos_weeks = window_size - is_weeks

        if oos_weeks < 4:
            oos_weeks = 4
            is_weeks = window_size - oos_weeks

        windows = []
        current_start = start_date

        while current_start + timedelta(weeks=is_weeks + oos_weeks) <= end_date:
            is_start = current_start
            is_end = is_start + timedelta(weeks=is_weeks)
            oos_start = is_end
            oos_end = oos_start + timedelta(weeks=oos_weeks)

            # Ensure we don't exceed end_date
            if oos_end > end_date:
                oos_end = end_date
                if (oos_end - oos_start).days < 7 * 4:  # Min 4 weeks OOS
                    break

            windows.append({
                "is_start": is_start,
                "is_end": is_end,
                "oos_start": oos_start,
                "oos_end": oos_end,
            })

            # Move to next window (non-overlapping)
            current_start = oos_end

        return windows

    def run_walk_forward(
        self,
        start_date: str,
        end_date: str,
        n_windows: int = 5,
        holdout_pct: float = 0.20,
    ) -> Dict[str, Any]:
        """
        Run walk-forward analysis.

        Args:
            start_date: Start date (YYYY-MM-DD).
            end_date: End date (YYYY-MM-DD).
            n_windows: Target number of walk-forward windows.
            holdout_pct: Percentage of data to hold out for final OOS.

        Returns:
            Dict with walk-forward results.
        """
        start = pd.Timestamp(start_date).date()
        end = pd.Timestamp(end_date).date()

        # Reserve final holdout
        total_days = (end - start).days
        holdout_days = int(total_days * holdout_pct)
        wf_end = end - timedelta(days=holdout_days)
        holdout_start = wf_end

        # Determine windows
        windows = self.determine_windows(start, wf_end, n_windows)

        if not windows:
            raise ValueError("Could not create walk-forward windows")

        # Run backtest for each window
        is_results = []
        oos_results = []

        for i, window in enumerate(windows):
            print(f"\nWindow {i + 1}/{len(windows)}:")
            print(f"  IS: {window['is_start']} to {window['is_end']}")
            print(f"  OOS: {window['oos_start']} to {window['oos_end']}")

            # In-sample backtest
            try:
                is_result = self.backtest_engine.run_backtest(
                    start_date=window["is_start"].strftime("%Y-%m-%d"),
                    end_date=window["is_end"].strftime("%Y-%m-%d"),
                )
                is_results.append(is_result)
            except Exception as e:
                print(f"  IS backtest failed: {e}")
                is_results.append(None)

            # Out-of-sample backtest
            try:
                oos_result = self.backtest_engine.run_backtest(
                    start_date=window["oos_start"].strftime("%Y-%m-%d"),
                    end_date=window["oos_end"].strftime("%Y-%m-%d"),
                )
                oos_results.append(oos_result)
            except Exception as e:
                print(f"  OOS backtest failed: {e}")
                oos_results.append(None)

        # Calculate Walk-Forward Efficiency
        wfe = self._calculate_wfe(is_results, oos_results)

        # Run final holdout test
        holdout_result = None
        if holdout_pct > 0:
            print(f"\nFinal Holdout: {holdout_start} to {end}")
            try:
                holdout_result = self.backtest_engine.run_backtest(
                    start_date=holdout_start.strftime("%Y-%m-%d"),
                    end_date=end.strftime("%Y-%m-%d"),
                )
            except Exception as e:
                print(f"  Holdout backtest failed: {e}")

        return {
            "windows": windows,
            "is_results": is_results,
            "oos_results": oos_results,
            "wfe": wfe,
            "holdout_result": holdout_result,
            "holdout_start": holdout_start,
            "holdout_end": end,
            "n_windows": len(windows),
        }

    def _calculate_wfe(
        self,
        is_results: List[Optional[Dict]],
        oos_results: List[Optional[Dict]],
    ) -> float:
        """
        Calculate Walk-Forward Efficiency.

        WFE = OOS Sharpe / IS Sharpe
        Should be > 60% for a robust strategy.

        Args:
            is_results: In-sample backtest results.
            oos_results: Out-of-sample backtest results.

        Returns:
            Walk-Forward Efficiency (0-1 range, can exceed 1).
        """
        is_sharpes = []
        oos_sharpes = []

        for is_result, oos_result in zip(is_results, oos_results):
            if is_result and oos_result:
                is_sharpe = is_result["strategy_metrics"].get("strategy_sharpe", 0)
                oos_sharpe = oos_result["strategy_metrics"].get("strategy_sharpe", 0)

                if is_sharpe > 0:  # Only count positive IS Sharpe
                    is_sharpes.append(is_sharpe)
                    oos_sharpes.append(oos_sharpe)

        if not is_sharpes:
            return 0.0

        avg_is_sharpe = np.mean(is_sharpes)
        avg_oos_sharpe = np.mean(oos_sharpes)

        if avg_is_sharpe <= 0:
            return 0.0

        return avg_oos_sharpe / avg_is_sharpe

    def print_walk_forward_report(self, results: Dict[str, Any]) -> None:
        """Print walk-forward analysis report."""
        print("\n" + "=" * 70)
        print("WALK-FORWARD ANALYSIS REPORT")
        print("=" * 70)

        print(f"\nNumber of Windows: {results['n_windows']}")

        # Window-by-window results
        print("\n--- Window Results ---")
        print(f"{'Window':<8} {'IS Sharpe':<12} {'OOS Sharpe':<12} {'WFE':<10}")
        print("-" * 42)

        for i, (is_result, oos_result) in enumerate(
            zip(results["is_results"], results["oos_results"])
        ):
            is_sharpe = (
                is_result["strategy_metrics"].get("strategy_sharpe", 0)
                if is_result else 0
            )
            oos_sharpe = (
                oos_result["strategy_metrics"].get("strategy_sharpe", 0)
                if oos_result else 0
            )
            window_wfe = oos_sharpe / is_sharpe if is_sharpe > 0 else 0

            print(f"{i + 1:<8} {is_sharpe:<12.2f} {oos_sharpe:<12.2f} {window_wfe:<10.2%}")

        # Overall WFE
        print(f"\n--- Overall ---")
        print(f"Walk-Forward Efficiency: {results['wfe']:.2%}")

        # Check against threshold
        wfe_threshold = self.config.min_wfe
        if results["wfe"] >= wfe_threshold:
            print(f"✓ WFE meets threshold ({wfe_threshold:.0%})")
        else:
            print(f"✗ WFE below threshold ({wfe_threshold:.0%})")

        # Holdout results
        if results["holdout_result"]:
            print(f"\n--- Final Holdout ({results['holdout_start']} to {results['holdout_end']}) ---")
            holdout = results["holdout_result"]

            sharpe = holdout["strategy_metrics"].get("strategy_sharpe", 0)
            ret = holdout["strategy_metrics"].get("strategy_ann_return", 0)
            dd = holdout["strategy_metrics"].get("strategy_max_drawdown", 0)

            print(f"Sharpe Ratio: {sharpe:.2f}")
            print(f"Annual Return: {ret:.2%}")
            print(f"Max Drawdown: {dd:.2%}")

            # Check against thresholds
            if sharpe >= self.config.min_sharpe:
                print(f"✓ Sharpe meets threshold ({self.config.min_sharpe:.1f})")
            else:
                print(f"✗ Sharpe below threshold ({self.config.min_sharpe:.1f})")

            if abs(dd) <= self.config.max_drawdown_limit:
                print(f"✓ Drawdown within limit ({self.config.max_drawdown_limit:.0%})")
            else:
                print(f"✗ Drawdown exceeds limit ({self.config.max_drawdown_limit:.0%})")

        print("=" * 70)
