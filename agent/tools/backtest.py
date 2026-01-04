"""
Backtest Tools
==============

Tools for backtesting heuristic strategies.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Callable
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .analysis import get_factor_data, _compute_simple_signal

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "agent" / "output"


def run_backtest(
    factor: str,
    signal_func: Optional[Callable] = None,
    signal_weights: Optional[Dict[str, float]] = None,
    threshold: float = 0.0,
    start_date: str = "2016-01-01",
    train_window: int = 504,
    hold_period: int = 0,
    target_type: str = "premium",
    benchmark: str = "SPY",
    oos_start_date: Optional[str] = None,
) -> Dict:
    """
    Run a backtest for a factor timing strategy.

    Args:
        factor: Factor ETF symbol (VLUE, SIZE, etc.)
        signal_func: Optional custom signal function (features -> signal Series)
        signal_weights: Dict of feature weights for default signal
        threshold: Signal threshold for taking positions
        start_date: Start date
        train_window: Number of days for training/warmup (ignored if oos_start_date is set)
        hold_period: Minimum holding period in days
        target_type: "premium" (factor - benchmark) or "ratio" (factor/benchmark relative returns)
        benchmark: Benchmark symbol for ratio calculation (default: SPY)
        oos_start_date: OOS period start date in YYYY-MM-DD format (overrides train_window)

    Returns:
        Dict with backtest results
    """
    features, factor_ret, premium = get_factor_data(factor, start_date, benchmark=benchmark)

    # For ratio mode, compute log returns of the ratio
    if target_type == "ratio":
        # Ratio returns = factor returns - benchmark returns (same as premium for daily)
        # But for interpretation: long signal = long factor, short benchmark
        #                        short signal = short factor, long benchmark
        target_ret = premium  # Already factor - benchmark
    else:
        target_ret = premium
    
    # Compute signal
    if signal_func is not None:
        signal = signal_func(features)
    elif signal_weights is not None:
        signal = _compute_weighted_signal(features, signal_weights)
    else:
        signal = _compute_simple_signal(features, factor)
    
    # Create positions
    if threshold > 0 and hold_period > 0:
        positions = _threshold_positions(signal, threshold, hold_period)
    elif threshold > 0:
        positions = pd.Series(0.0, index=signal.index)
        positions[signal > threshold] = 1.0
        positions[signal < -threshold] = -1.0
    else:
        positions = np.sign(signal)
    
    # Lag positions by 1 day (no lookahead)
    positions = positions.shift(1).fillna(0)

    # Determine OOS start
    if oos_start_date:
        # Find the index position of the OOS start date
        oos_date = pd.Timestamp(oos_start_date)
        mask = positions.index >= oos_date
        if mask.any():
            oos_start = mask.argmax()
        else:
            oos_start = len(positions) - 1
    else:
        oos_start = train_window

    # Only use OOS period
    positions_oos = positions.iloc[oos_start:]
    target_oos = target_ret.iloc[oos_start:]

    # Strategy returns
    # For ratio mode: position * (factor_ret - benchmark_ret)
    #   Long (+1): profit when factor > benchmark
    #   Short (-1): profit when factor < benchmark
    strategy_ret = positions_oos * target_oos
    strategy_ret = strategy_ret.dropna()
    benchmark_ret = target_oos.loc[strategy_ret.index]
    
    # Calculate metrics
    strat_m = _calc_metrics(strategy_ret)
    bench_m = _calc_metrics(benchmark_ret)
    
    # Additional stats
    time_in_market = (positions_oos != 0).mean()
    long_pct = (positions_oos > 0).mean()
    short_pct = (positions_oos < 0).mean()
    
    # Hit rate
    signal_dir = np.sign(signal.shift(1).loc[target_oos.index])
    actual_dir = np.sign(target_oos)
    hit_rate = (signal_dir == actual_dir).mean()

    # Equity curves
    equity_curve = (1 + strategy_ret).cumprod() - 1  # Cumulative return
    benchmark_curve = (1 + benchmark_ret).cumprod() - 1

    return {
        "factor": factor,
        "benchmark": benchmark,
        "target_type": target_type,
        "period": f"{strategy_ret.index[0].date()} to {strategy_ret.index[-1].date()}",
        "days": len(strategy_ret),
        "strategy": strat_m,
        "benchmark_metrics": bench_m,
        "hit_rate": float(hit_rate),
        "time_in_market": float(time_in_market),
        "long_pct": float(long_pct),
        "short_pct": float(short_pct),
        "threshold": threshold,
        "hold_period": hold_period,
        "equity_curve": equity_curve,
        "benchmark_curve": benchmark_curve,
    }


def _compute_weighted_signal(features: pd.DataFrame, weights: Dict[str, float]) -> pd.Series:
    """
    Compute weighted signal from features with support for complex rules.

    Supports:
    - Simple features: "XLY_ma_ratio"
    - Interactions: "feat1_X_feat2" (multiply two features)
    - Squared: "feat1_SQ" (square a feature)
    - Conditional AND: "feat1_AND_feat2" (both must be positive)
    - Conditional OR: "feat1_OR_feat2" (either positive)
    - Threshold GT: "feat1_GT_0.5" (feature > 0.5 threshold)
    - Threshold LT: "feat1_LT_-0.5" (feature < threshold)
    - Inverse: "feat1_INV" (flip sign)
    - Absolute: "feat1_ABS"
    - Sign only: "feat1_SIGN" (-1, 0, or 1)
    - Conditional: "IF_feat1_THEN_feat2" (use feat2 when feat1 > 0)
    - Max/Min: "MAX_feat1_feat2", "MIN_feat1_feat2"
    """
    signal = pd.Series(0.0, index=features.index)
    total_weight = 0.0

    def normalize_feature(feat: pd.Series) -> pd.Series:
        """Normalize feature to z-score."""
        feat_norm = (feat - feat.rolling(252).mean()) / feat.rolling(252).std()
        return feat_norm.clip(-3, 3) / 3

    def get_feature(name: str) -> pd.Series:
        """Get a feature by name, return zeros if not found."""
        if name in features.columns:
            return normalize_feature(features[name])
        return pd.Series(0.0, index=features.index)

    for feat_name, weight in weights.items():
        feat_value = None

        try:
            # Check for interaction terms: feat1_X_feat2
            if "_X_" in feat_name and not feat_name.startswith("IF_"):
                parts = feat_name.split("_X_")
                if len(parts) == 2:
                    f1 = get_feature(parts[0])
                    f2 = get_feature(parts[1])
                    feat_value = f1 * f2

            # Check for squared terms: feat1_SQ
            elif feat_name.endswith("_SQ"):
                base_feat = feat_name[:-3]
                f1 = get_feature(base_feat)
                feat_value = f1 ** 2

            # Check for AND condition: feat1_AND_feat2 (both positive)
            elif "_AND_" in feat_name:
                parts = feat_name.split("_AND_")
                if len(parts) == 2:
                    f1 = get_feature(parts[0])
                    f2 = get_feature(parts[1])
                    # Both must be positive for signal to be positive
                    feat_value = ((f1 > 0) & (f2 > 0)).astype(float) * (f1 + f2) / 2

            # Check for OR condition: feat1_OR_feat2 (either positive)
            elif "_OR_" in feat_name:
                parts = feat_name.split("_OR_")
                if len(parts) == 2:
                    f1 = get_feature(parts[0])
                    f2 = get_feature(parts[1])
                    # Either positive for signal
                    feat_value = ((f1 > 0) | (f2 > 0)).astype(float) * pd.concat([f1, f2], axis=1).max(axis=1)

            # Check for greater than threshold: feat1_GT_0.5
            elif "_GT_" in feat_name:
                parts = feat_name.split("_GT_")
                if len(parts) == 2:
                    f1 = get_feature(parts[0])
                    try:
                        threshold_val = float(parts[1])
                        feat_value = (f1 > threshold_val).astype(float)
                    except ValueError:
                        feat_value = f1

            # Check for less than threshold: feat1_LT_-0.5
            elif "_LT_" in feat_name:
                parts = feat_name.split("_LT_")
                if len(parts) == 2:
                    f1 = get_feature(parts[0])
                    try:
                        threshold_val = float(parts[1])
                        feat_value = (f1 < threshold_val).astype(float)
                    except ValueError:
                        feat_value = f1

            # Check for inverse: feat1_INV (flip sign)
            elif feat_name.endswith("_INV"):
                base_feat = feat_name[:-4]
                f1 = get_feature(base_feat)
                feat_value = -f1

            # Check for absolute value: feat1_ABS
            elif feat_name.endswith("_ABS"):
                base_feat = feat_name[:-4]
                f1 = get_feature(base_feat)
                feat_value = f1.abs()

            # Check for sign only: feat1_SIGN (just direction, -1, 0, or 1)
            elif feat_name.endswith("_SIGN"):
                base_feat = feat_name[:-5]
                f1 = get_feature(base_feat)
                feat_value = np.sign(f1)

            # Check for IF condition: IF_feat1_THEN_feat2 (use feat2 only when feat1 > 0)
            elif feat_name.startswith("IF_") and "_THEN_" in feat_name:
                parts = feat_name[3:].split("_THEN_")
                if len(parts) == 2:
                    condition_feat = get_feature(parts[0])
                    value_feat = get_feature(parts[1])
                    feat_value = (condition_feat > 0).astype(float) * value_feat

            # Check for IFNOT condition: IFNOT_feat1_THEN_feat2 (use feat2 when feat1 <= 0)
            elif feat_name.startswith("IFNOT_") and "_THEN_" in feat_name:
                parts = feat_name[6:].split("_THEN_")
                if len(parts) == 2:
                    condition_feat = get_feature(parts[0])
                    value_feat = get_feature(parts[1])
                    feat_value = (condition_feat <= 0).astype(float) * value_feat

            # Check for MAX of two features: MAX_feat1_AND_feat2
            elif feat_name.startswith("MAX_") and "_AND_" in feat_name:
                parts = feat_name[4:].split("_AND_")
                if len(parts) == 2:
                    f1 = get_feature(parts[0])
                    f2 = get_feature(parts[1])
                    feat_value = pd.concat([f1, f2], axis=1).max(axis=1)

            # Check for MIN of two features: MIN_feat1_AND_feat2
            elif feat_name.startswith("MIN_") and "_AND_" in feat_name:
                parts = feat_name[4:].split("_AND_")
                if len(parts) == 2:
                    f1 = get_feature(parts[0])
                    f2 = get_feature(parts[1])
                    feat_value = pd.concat([f1, f2], axis=1).min(axis=1)

            # Simple feature lookup
            elif feat_name in features.columns:
                feat_value = get_feature(feat_name)

        except Exception:
            # If any parsing fails, try simple lookup
            if feat_name in features.columns:
                feat_value = get_feature(feat_name)

        # Add to signal if we computed a value
        if feat_value is not None:
            signal += weight * feat_value.fillna(0)
            total_weight += abs(weight)

    # Normalize by total weight
    if total_weight > 0:
        signal = signal / total_weight

    return signal


def _threshold_positions(
    signal: pd.Series,
    threshold: float,
    hold_period: int,
) -> pd.Series:
    """Create positions with threshold crossings and holding periods."""
    position = pd.Series(0.0, index=signal.index)
    current_pos = 0.0
    hold_counter = 0
    
    for i, (date, sig) in enumerate(signal.items()):
        if pd.isna(sig):
            position.iloc[i] = current_pos
            continue
        
        if hold_counter > 0:
            position.iloc[i] = current_pos
            hold_counter -= 1
        else:
            if sig > threshold:
                current_pos = 1.0
                hold_counter = hold_period
            elif sig < -threshold:
                current_pos = -1.0
                hold_counter = hold_period
            else:
                current_pos = 0.0
            position.iloc[i] = current_pos
    
    return position


def _calc_metrics(rets: pd.Series) -> Dict:
    """Calculate performance metrics."""
    ann_ret = rets.mean() * 252
    ann_vol = rets.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    cum = (1 + rets).cumprod()
    peak = cum.expanding().max()
    max_dd = ((cum - peak) / peak).min()
    total_ret = (1 + rets).prod() - 1
    
    # Calmar ratio
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0
    
    return {
        "annual_return": float(ann_ret),
        "annual_vol": float(ann_vol),
        "sharpe": float(sharpe),
        "max_drawdown": float(max_dd),
        "total_return": float(total_ret),
        "calmar": float(calmar),
    }


def run_parameter_sweep(
    factor: str,
    thresholds: List[float] = [0.0, 0.1, 0.2, 0.3, 0.5],
    hold_periods: List[int] = [0, 5, 10, 21],
    start_date: str = "2016-01-01",
) -> pd.DataFrame:
    """
    Run parameter sweep for threshold and holding period.
    
    Returns DataFrame with results for each combination.
    """
    results = []
    
    for threshold in thresholds:
        for hold_period in hold_periods:
            try:
                res = run_backtest(
                    factor=factor,
                    threshold=threshold,
                    hold_period=hold_period,
                    start_date=start_date,
                )
                results.append({
                    "threshold": threshold,
                    "hold_period": hold_period,
                    "sharpe": res["strategy"]["sharpe"],
                    "annual_return": res["strategy"]["annual_return"],
                    "max_drawdown": res["strategy"]["max_drawdown"],
                    "hit_rate": res["hit_rate"],
                    "time_in_market": res["time_in_market"],
                })
            except Exception as e:
                print(f"Error with threshold={threshold}, hold={hold_period}: {e}")
                continue
    
    return pd.DataFrame(results)


def compare_strategies(
    factor: str,
    strategies: Dict[str, Dict[str, float]],
    start_date: str = "2016-01-01",
    save_plot: bool = True,
) -> Dict:
    """
    Compare multiple signal weight configurations.
    
    Args:
        factor: Factor ETF symbol
        strategies: Dict mapping strategy name to weight dict
        start_date: Start date
        save_plot: Whether to save comparison plot
        
    Returns:
        Dict with comparison results
    """
    features, factor_ret, premium = get_factor_data(factor, start_date)
    
    results = {}
    equity_curves = {}
    
    for name, weights in strategies.items():
        signal = _compute_weighted_signal(features, weights)
        positions = np.sign(signal).shift(1).fillna(0)
        
        # OOS only
        oos_start = 504
        positions_oos = positions.iloc[oos_start:]
        premium_oos = premium.iloc[oos_start:]
        
        strategy_ret = (positions_oos * premium_oos).dropna()
        
        results[name] = _calc_metrics(strategy_ret)
        equity_curves[name] = (1 + strategy_ret).cumprod()
    
    # Add benchmark
    benchmark_ret = premium.iloc[504:].dropna()
    results["benchmark"] = _calc_metrics(benchmark_ret)
    equity_curves["benchmark"] = (1 + benchmark_ret).cumprod()
    
    # Create comparison plot
    if save_plot:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # Equity curves
        ax1 = axes[0]
        for name, curve in equity_curves.items():
            style = "--" if name == "benchmark" else "-"
            alpha = 0.6 if name == "benchmark" else 0.9
            ax1.plot(curve.index, curve.values * 100, label=f"{name} (SR: {results[name]['sharpe']:.2f})", 
                    linestyle=style, alpha=alpha, linewidth=2)
        ax1.axhline(100, color="gray", linestyle="--", alpha=0.5)
        ax1.set_ylabel("Cumulative Return (indexed to 100)")
        ax1.set_title(f"Strategy Comparison - {factor}", fontsize=12, fontweight="bold")
        ax1.legend(loc="upper left")
        ax1.grid(True, alpha=0.3)
        
        # Metrics bar chart
        ax2 = axes[1]
        names = list(results.keys())
        sharpes = [results[n]["sharpe"] for n in names]
        colors = ["#2ecc71" if s > 0 else "#e74c3c" for s in sharpes]
        
        bars = ax2.bar(names, sharpes, color=colors, alpha=0.8)
        ax2.axhline(0, color="black", linewidth=0.5)
        ax2.set_ylabel("Sharpe Ratio")
        ax2.set_title("Sharpe Ratio Comparison", fontsize=11)
        ax2.grid(True, alpha=0.3, axis="y")
        
        # Add value labels
        for bar, sharpe in zip(bars, sharpes):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                    f"{sharpe:.2f}", ha="center", va="bottom", fontsize=10)
        
        plt.tight_layout()
        plot_path = OUTPUT_DIR / f"strategy_comparison_{factor}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()
        
        print(f"Comparison plot saved to: {plot_path}")
    
    return {
        "metrics": results,
        "best_strategy": max(results.keys(), key=lambda x: results[x]["sharpe"] if x != "benchmark" else -999),
    }

