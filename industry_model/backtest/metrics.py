"""
Performance Metrics for Industry Model

Calculates various performance metrics per Quant Constitution requirements.
"""

from typing import Dict, Optional, Tuple

import pandas as pd
import numpy as np


def calculate_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 52,
) -> float:
    """
    Calculate annualized Sharpe Ratio.

    Args:
        returns: Return series.
        risk_free_rate: Annual risk-free rate.
        periods_per_year: Number of periods per year (52 for weekly).

    Returns:
        Annualized Sharpe Ratio.
    """
    if returns.empty or returns.std() == 0:
        return 0.0

    excess_returns = returns - risk_free_rate / periods_per_year
    ann_return = excess_returns.mean() * periods_per_year
    ann_vol = returns.std() * np.sqrt(periods_per_year)

    return ann_return / ann_vol if ann_vol > 0 else 0.0


def calculate_sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 52,
) -> float:
    """
    Calculate annualized Sortino Ratio.

    Uses downside deviation instead of standard deviation.

    Args:
        returns: Return series.
        risk_free_rate: Annual risk-free rate.
        periods_per_year: Number of periods per year.

    Returns:
        Annualized Sortino Ratio.
    """
    if returns.empty:
        return 0.0

    excess_returns = returns - risk_free_rate / periods_per_year
    ann_return = excess_returns.mean() * periods_per_year

    # Downside deviation
    downside_returns = returns[returns < 0]
    if len(downside_returns) == 0:
        return np.inf if ann_return > 0 else 0.0

    downside_dev = downside_returns.std() * np.sqrt(periods_per_year)

    return ann_return / downside_dev if downside_dev > 0 else 0.0


def calculate_max_drawdown(returns: pd.Series) -> Tuple[float, pd.Timestamp, pd.Timestamp]:
    """
    Calculate maximum drawdown and dates.

    Args:
        returns: Return series.

    Returns:
        Tuple of (max_drawdown, peak_date, trough_date).
    """
    if returns.empty:
        return 0.0, None, None

    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max

    max_dd = drawdown.min()
    trough_idx = drawdown.idxmin()

    # Find peak before trough
    peak_idx = cumulative[:trough_idx].idxmax()

    return max_dd, peak_idx, trough_idx


def calculate_calmar_ratio(
    returns: pd.Series,
    periods_per_year: int = 52,
) -> float:
    """
    Calculate Calmar Ratio (annual return / max drawdown).

    Args:
        returns: Return series.
        periods_per_year: Number of periods per year.

    Returns:
        Calmar Ratio.
    """
    if returns.empty:
        return 0.0

    ann_return = returns.mean() * periods_per_year
    max_dd, _, _ = calculate_max_drawdown(returns)

    if max_dd == 0:
        return np.inf if ann_return > 0 else 0.0

    return ann_return / abs(max_dd)


def calculate_information_ratio(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    periods_per_year: int = 52,
) -> float:
    """
    Calculate Information Ratio.

    Args:
        strategy_returns: Strategy return series.
        benchmark_returns: Benchmark return series.
        periods_per_year: Number of periods per year.

    Returns:
        Information Ratio.
    """
    if strategy_returns.empty or benchmark_returns.empty:
        return 0.0

    # Align series
    common_idx = strategy_returns.index.intersection(benchmark_returns.index)
    strategy = strategy_returns.loc[common_idx]
    benchmark = benchmark_returns.loc[common_idx]

    excess = strategy - benchmark
    tracking_error = excess.std() * np.sqrt(periods_per_year)
    active_return = excess.mean() * periods_per_year

    return active_return / tracking_error if tracking_error > 0 else 0.0


def calculate_hit_rate(returns: pd.Series) -> float:
    """
    Calculate hit rate (percentage of positive periods).

    Args:
        returns: Return series.

    Returns:
        Hit rate (0-1).
    """
    if returns.empty:
        return 0.0

    return (returns > 0).mean()


def calculate_win_loss_ratio(returns: pd.Series) -> float:
    """
    Calculate win/loss ratio (average win / average loss).

    Args:
        returns: Return series.

    Returns:
        Win/Loss ratio.
    """
    if returns.empty:
        return 0.0

    wins = returns[returns > 0]
    losses = returns[returns < 0]

    if len(wins) == 0 or len(losses) == 0:
        return np.inf if len(wins) > 0 else 0.0

    avg_win = wins.mean()
    avg_loss = abs(losses.mean())

    return avg_win / avg_loss if avg_loss > 0 else np.inf


def calculate_profit_factor(returns: pd.Series) -> float:
    """
    Calculate profit factor (sum of wins / sum of losses).

    Args:
        returns: Return series.

    Returns:
        Profit factor.
    """
    if returns.empty:
        return 0.0

    wins = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())

    return wins / losses if losses > 0 else np.inf


def calculate_all_metrics(
    strategy_returns: pd.Series,
    benchmark_returns: Optional[pd.Series] = None,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 52,
) -> Dict[str, float]:
    """
    Calculate all performance metrics.

    Args:
        strategy_returns: Strategy return series.
        benchmark_returns: Benchmark return series (optional).
        risk_free_rate: Annual risk-free rate.
        periods_per_year: Number of periods per year.

    Returns:
        Dict of all metrics.
    """
    metrics = {}

    # Basic return metrics
    total_return = (1 + strategy_returns).prod() - 1
    ann_return = strategy_returns.mean() * periods_per_year
    ann_vol = strategy_returns.std() * np.sqrt(periods_per_year)

    metrics["total_return"] = total_return
    metrics["annual_return"] = ann_return
    metrics["annual_volatility"] = ann_vol

    # Risk-adjusted metrics
    metrics["sharpe_ratio"] = calculate_sharpe_ratio(
        strategy_returns, risk_free_rate, periods_per_year
    )
    metrics["sortino_ratio"] = calculate_sortino_ratio(
        strategy_returns, risk_free_rate, periods_per_year
    )

    # Drawdown metrics
    max_dd, peak, trough = calculate_max_drawdown(strategy_returns)
    metrics["max_drawdown"] = max_dd
    metrics["max_dd_peak"] = peak
    metrics["max_dd_trough"] = trough
    metrics["calmar_ratio"] = calculate_calmar_ratio(
        strategy_returns, periods_per_year
    )

    # Win/loss metrics
    metrics["hit_rate"] = calculate_hit_rate(strategy_returns)
    metrics["win_loss_ratio"] = calculate_win_loss_ratio(strategy_returns)
    metrics["profit_factor"] = calculate_profit_factor(strategy_returns)

    # Benchmark-relative metrics
    if benchmark_returns is not None and not benchmark_returns.empty:
        metrics["information_ratio"] = calculate_information_ratio(
            strategy_returns, benchmark_returns, periods_per_year
        )

        # Excess metrics
        common_idx = strategy_returns.index.intersection(benchmark_returns.index)
        excess = strategy_returns.loc[common_idx] - benchmark_returns.loc[common_idx]
        metrics["excess_return"] = excess.mean() * periods_per_year
        metrics["tracking_error"] = excess.std() * np.sqrt(periods_per_year)

    # Count metrics
    metrics["n_periods"] = len(strategy_returns)
    metrics["n_positive"] = (strategy_returns > 0).sum()
    metrics["n_negative"] = (strategy_returns < 0).sum()

    return metrics


def check_risk_limits(
    metrics: Dict[str, float],
    max_drawdown_limit: float = 0.20,
    min_sharpe: float = 1.0,
    min_wfe: float = 0.60,
    wfe: Optional[float] = None,
) -> Dict[str, Tuple[bool, str]]:
    """
    Check if metrics meet risk limits per Quant Constitution.

    Args:
        metrics: Performance metrics dict.
        max_drawdown_limit: Maximum allowed drawdown.
        min_sharpe: Minimum required Sharpe ratio.
        min_wfe: Minimum Walk-Forward Efficiency.
        wfe: Walk-Forward Efficiency (if calculated).

    Returns:
        Dict of (passed, message) for each limit.
    """
    results = {}

    # Max Drawdown
    max_dd = abs(metrics.get("max_drawdown", 0))
    passed = max_dd <= max_drawdown_limit
    results["max_drawdown"] = (
        passed,
        f"Max DD: {max_dd:.1%} (limit: {max_drawdown_limit:.0%})",
    )

    # Sharpe Ratio
    sharpe = metrics.get("sharpe_ratio", 0)
    passed = sharpe >= min_sharpe
    results["sharpe_ratio"] = (
        passed,
        f"Sharpe: {sharpe:.2f} (min: {min_sharpe:.1f})",
    )

    # Walk-Forward Efficiency
    if wfe is not None:
        passed = wfe >= min_wfe
        results["wfe"] = (
            passed,
            f"WFE: {wfe:.1%} (min: {min_wfe:.0%})",
        )

    return results


def print_metrics_report(
    metrics: Dict[str, float],
    title: str = "Performance Metrics",
) -> None:
    """Print formatted metrics report."""
    print("\n" + "=" * 50)
    print(title)
    print("=" * 50)

    # Returns
    print("\n--- Returns ---")
    print(f"Total Return: {metrics.get('total_return', 0):.2%}")
    print(f"Annual Return: {metrics.get('annual_return', 0):.2%}")
    print(f"Annual Volatility: {metrics.get('annual_volatility', 0):.2%}")

    # Risk-Adjusted
    print("\n--- Risk-Adjusted ---")
    print(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
    print(f"Sortino Ratio: {metrics.get('sortino_ratio', 0):.2f}")
    print(f"Calmar Ratio: {metrics.get('calmar_ratio', 0):.2f}")

    # Drawdown
    print("\n--- Drawdown ---")
    print(f"Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")

    # Win/Loss
    print("\n--- Win/Loss ---")
    print(f"Hit Rate: {metrics.get('hit_rate', 0):.1%}")
    print(f"Win/Loss Ratio: {metrics.get('win_loss_ratio', 0):.2f}")
    print(f"Profit Factor: {metrics.get('profit_factor', 0):.2f}")

    # Benchmark-relative (if available)
    if "information_ratio" in metrics:
        print("\n--- Relative to Benchmark ---")
        print(f"Information Ratio: {metrics.get('information_ratio', 0):.2f}")
        print(f"Excess Return: {metrics.get('excess_return', 0):.2%}")
        print(f"Tracking Error: {metrics.get('tracking_error', 0):.2%}")

    print("=" * 50)
