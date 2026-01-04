"""
Optimization Tools
==================

Tools for optimizing heuristic signal weights and combinations.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings
import json

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy import stats

from .analysis import get_factor_data, analyze_feature_correlations
from .backtest import run_backtest, _compute_weighted_signal, _calc_metrics

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "agent" / "output"


def generate_new_signal(
    factor: str,
    top_n_features: int = 5,
    use_ic_weights: bool = True,
    start_date: str = "2016-01-01",
) -> Dict:
    """
    Generate a new signal based on top correlated features.
    
    Args:
        factor: Factor ETF symbol
        top_n_features: Number of top features to include
        use_ic_weights: Weight features by their IC if True
        start_date: Start date
        
    Returns:
        Dict with signal weights and backtest results
    """
    # Get top correlated features
    correlations = analyze_feature_correlations(factor, start_date=start_date, top_n=top_n_features)
    
    # Create weight dict
    weights = {}
    for _, row in correlations.iterrows():
        feat = row["feature"]
        ic = row["ic"]
        
        if use_ic_weights:
            # Weight by absolute IC, apply sign
            weight = abs(ic) * (1 if ic > 0 else -1)
        else:
            # Equal weight with sign
            weight = 1.0 if ic > 0 else -1.0
        
        weights[feat] = weight
    
    # Normalize weights to sum to 1
    total = sum(abs(w) for w in weights.values())
    weights = {k: v / total for k, v in weights.items()}
    
    # Run backtest
    result = run_backtest(factor, signal_weights=weights, start_date=start_date)
    
    return {
        "weights": weights,
        "correlations": correlations.to_dict("records"),
        "backtest": result,
    }


def test_signal_combination(
    factor: str,
    feature_weights: Dict[str, float],
    threshold: float = 0.0,
    hold_period: int = 0,
    start_date: str = "2016-01-01",
) -> Dict:
    """
    Test a specific feature weight combination.
    
    Args:
        factor: Factor ETF symbol
        feature_weights: Dict mapping feature name to weight
        threshold: Signal threshold for positions
        hold_period: Minimum holding period
        start_date: Start date
        
    Returns:
        Backtest results
    """
    return run_backtest(
        factor=factor,
        signal_weights=feature_weights,
        threshold=threshold,
        hold_period=hold_period,
        start_date=start_date,
    )


def optimize_weights(
    factor: str,
    features_to_use: List[str],
    objective: str = "sharpe",
    start_date: str = "2016-01-01",
    n_trials: int = 100,
) -> Dict:
    """
    Optimize feature weights to maximize objective.
    
    WARNING: This uses in-sample optimization and may overfit.
    Use primarily for research/understanding, not production.
    
    Args:
        factor: Factor ETF symbol
        features_to_use: List of features to optimize
        objective: "sharpe", "return", or "calmar"
        start_date: Start date
        n_trials: Number of random restarts
        
    Returns:
        Dict with optimal weights and performance
    """
    features_df, _, premium = get_factor_data(factor, start_date)
    
    # Filter to available features
    available = [f for f in features_to_use if f in features_df.columns]
    if not available:
        return {"error": "No valid features found"}
    
    n_features = len(available)
    
    # Objective function
    def neg_objective(weights: np.ndarray) -> float:
        weight_dict = {f: w for f, w in zip(available, weights)}
        signal = _compute_weighted_signal(features_df, weight_dict)
        positions = np.sign(signal).shift(1).fillna(0)
        
        # OOS period
        oos_start = 504
        positions_oos = positions.iloc[oos_start:]
        premium_oos = premium.iloc[oos_start:]
        
        strategy_ret = (positions_oos * premium_oos).dropna()
        if len(strategy_ret) < 100:
            return 999.0
        
        metrics = _calc_metrics(strategy_ret)
        
        if objective == "sharpe":
            return -metrics["sharpe"]
        elif objective == "return":
            return -metrics["annual_return"]
        elif objective == "calmar":
            return -metrics["calmar"]
        else:
            return -metrics["sharpe"]
    
    # Run optimization with multiple random starts
    best_result = None
    best_obj = float("inf")
    
    for trial in range(n_trials):
        # Random initial weights
        x0 = np.random.randn(n_features)
        x0 = x0 / np.sum(np.abs(x0))  # Normalize
        
        try:
            result = minimize(
                neg_objective,
                x0,
                method="L-BFGS-B",
                bounds=[(-1, 1)] * n_features,
                options={"maxiter": 100},
            )
            
            if result.fun < best_obj:
                best_obj = result.fun
                best_result = result
        except Exception:
            continue
    
    if best_result is None:
        return {"error": "Optimization failed"}
    
    # Get final weights
    opt_weights = {f: float(w) for f, w in zip(available, best_result.x)}
    
    # Normalize
    total = sum(abs(w) for w in opt_weights.values())
    if total > 0:
        opt_weights = {k: v / total for k, v in opt_weights.items()}
    
    # Run final backtest
    final_result = run_backtest(factor, signal_weights=opt_weights, start_date=start_date)
    
    return {
        "optimal_weights": opt_weights,
        "objective": objective,
        "objective_value": -best_obj,
        "backtest": final_result,
        "warning": "Optimized in-sample - may be overfit. Validate out-of-sample.",
    }


def suggest_improvements(
    factor: str,
    current_sharpe: float,
    start_date: str = "2016-01-01",
) -> Dict:
    """
    Analyze current strategy and suggest improvements.
    
    Returns specific, actionable suggestions.
    """
    # Get correlations
    correlations = analyze_feature_correlations(factor, start_date=start_date, top_n=30)
    
    # Identify top predictors that might not be in current model
    top_predictors = correlations[correlations["significant"]].head(10)
    
    # Generate new signal and compare
    new_signal_result = generate_new_signal(factor, top_n_features=5, start_date=start_date)
    new_sharpe = new_signal_result["backtest"]["strategy"]["sharpe"]
    
    suggestions = []
    
    if new_sharpe > current_sharpe + 0.1:
        suggestions.append({
            "type": "feature_selection",
            "description": f"Use IC-weighted top features instead of fixed weights",
            "expected_improvement": f"+{new_sharpe - current_sharpe:.2f} Sharpe",
            "weights": new_signal_result["weights"],
        })
    
    # Check for underutilized features
    for _, row in top_predictors.iterrows():
        feat = row["feature"]
        ic = row["ic"]
        if abs(ic) > 0.1:
            suggestions.append({
                "type": "add_feature",
                "feature": feat,
                "ic": ic,
                "description": f"Consider adding {feat} (IC={ic:.3f})",
            })
    
    # Parameter sweep
    from .backtest import run_parameter_sweep
    sweep_results = run_parameter_sweep(factor, start_date=start_date)
    best_params = sweep_results.loc[sweep_results["sharpe"].idxmax()]
    
    if best_params["sharpe"] > current_sharpe + 0.05:
        suggestions.append({
            "type": "parameter_tuning",
            "description": f"Optimal threshold={best_params['threshold']:.1f}, hold={int(best_params['hold_period'])}d",
            "expected_sharpe": best_params["sharpe"],
        })
    
    return {
        "current_sharpe": current_sharpe,
        "suggestions": suggestions,
        "top_predictors": top_predictors.to_dict("records"),
    }

