"""
Feature Engineering Tools
==========================

Tools for creating and testing new derived features.
"""

import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Any
import warnings

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).parent.parent.parent
DB_PATH = PROJECT_ROOT / "storage" / "database" / "pit_macro.db"


def get_db_connection() -> sqlite3.Connection:
    """Get database connection."""
    return sqlite3.connect(DB_PATH)


def create_derived_feature(
    base_feature: str,
    transformation: str,
    window: int = 21,
    feature_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Create a new derived feature from an existing one.
    
    Args:
        base_feature: Name of existing feature (e.g., 'DGS10', 'SPY', 'VIX')
        transformation: Type of transformation:
            - 'ma_ratio': Price / Moving Average - 1
            - 'roc': Rate of change (% change over window)
            - 'zscore': Rolling z-score
            - 'volatility': Rolling standard deviation
            - 'momentum': Returns over window
            - 'diff': First difference
            - 'log_diff': Log difference (log returns)
            - 'ema': Exponential moving average
            - 'rsi': Relative strength index
            - 'rank': Rolling percentile rank
        window: Window size for rolling calculations (default: 21)
        feature_name: Custom name for the new feature
        
    Returns:
        Dict with feature stats and sample values
    """
    conn = get_db_connection()
    
    # Try to find the base feature
    # Check if it's a market symbol
    query = "SELECT price_date as date, close as value FROM market_prices WHERE symbol = ? ORDER BY price_date"
    df = pd.read_sql_query(query, conn, params=[base_feature])
    
    if df.empty:
        # Try economic series
        query = "SELECT observation_date as date, value FROM observations WHERE series_id = ? ORDER BY observation_date"
        df = pd.read_sql_query(query, conn, params=[base_feature])
    
    conn.close()
    
    if df.empty:
        return {"error": f"Feature '{base_feature}' not found in database"}
    
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date').sort_index()
    values = df['value']
    
    # Apply transformation
    if transformation == 'ma_ratio':
        ma = values.rolling(window).mean()
        derived = (values / ma) - 1
        name = feature_name or f"{base_feature}_ma_ratio_{window}d"
        
    elif transformation == 'roc':
        derived = values.pct_change(window)
        name = feature_name or f"{base_feature}_roc_{window}d"
        
    elif transformation == 'zscore':
        rolling_mean = values.rolling(window).mean()
        rolling_std = values.rolling(window).std()
        derived = (values - rolling_mean) / rolling_std
        name = feature_name or f"{base_feature}_zscore_{window}d"
        
    elif transformation == 'volatility':
        derived = values.pct_change().rolling(window).std() * np.sqrt(252)
        name = feature_name or f"{base_feature}_vol_{window}d"
        
    elif transformation == 'momentum':
        derived = values / values.shift(window) - 1
        name = feature_name or f"{base_feature}_mom_{window}d"
        
    elif transformation == 'diff':
        derived = values.diff(window)
        name = feature_name or f"{base_feature}_diff_{window}d"
        
    elif transformation == 'log_diff':
        derived = np.log(values).diff(window)
        name = feature_name or f"{base_feature}_logret_{window}d"
        
    elif transformation == 'ema':
        derived = values.ewm(span=window).mean()
        name = feature_name or f"{base_feature}_ema_{window}d"
        
    elif transformation == 'rsi':
        delta = values.diff()
        gain = delta.where(delta > 0, 0).rolling(window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
        rs = gain / loss
        derived = 100 - (100 / (1 + rs))
        name = feature_name or f"{base_feature}_rsi_{window}d"
        
    elif transformation == 'rank':
        derived = values.rolling(window).apply(
            lambda x: stats.percentileofscore(x, x.iloc[-1]) / 100,
            raw=False
        )
        name = feature_name or f"{base_feature}_rank_{window}d"
        
    else:
        return {"error": f"Unknown transformation: {transformation}. Options: ma_ratio, roc, zscore, volatility, momentum, diff, log_diff, ema, rsi, rank"}
    
    # Calculate stats
    derived = derived.dropna()
    
    return {
        "feature_name": name,
        "base_feature": base_feature,
        "transformation": transformation,
        "window": window,
        "stats": {
            "count": len(derived),
            "mean": round(derived.mean(), 6),
            "std": round(derived.std(), 6),
            "min": round(derived.min(), 6),
            "max": round(derived.max(), 6),
        },
        "sample_values": {str(k): v for k, v in derived.tail(5).round(6).to_dict().items()},
        "date_range": f"{derived.index.min().date()} to {derived.index.max().date()}",
    }


def combine_features(
    features: Dict[str, float],
    operation: str = "weighted_sum",
) -> Dict[str, Any]:
    """
    Combine multiple features into a single composite feature.
    
    Args:
        features: Dict of feature_name -> weight
        operation: How to combine:
            - 'weighted_sum': Sum of (feature * weight)
            - 'product': Product of features
            - 'mean': Simple average
            - 'zscore_sum': Sum of z-scored features
            
    Returns:
        Dict with combined feature stats
    """
    conn = get_db_connection()
    
    all_data = {}
    
    for feature_name, weight in features.items():
        # Parse feature name to get base and transformation
        # Try market prices first
        query = "SELECT price_date as date, close as value FROM market_prices WHERE symbol = ? ORDER BY price_date"
        df = pd.read_sql_query(query, conn, params=[feature_name])
        
        if df.empty:
            # Try economic series
            query = "SELECT observation_date as date, value FROM observations WHERE series_id = ? ORDER BY observation_date"
            df = pd.read_sql_query(query, conn, params=[feature_name])
        
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
            all_data[feature_name] = df['value']
    
    conn.close()
    
    if not all_data:
        return {"error": "No features found"}
    
    # Combine into DataFrame
    combined_df = pd.DataFrame(all_data)
    combined_df = combined_df.ffill().dropna()
    
    if operation == 'weighted_sum':
        result = pd.Series(0.0, index=combined_df.index)
        for feat, weight in features.items():
            if feat in combined_df.columns:
                # Z-score each feature first
                zscore = (combined_df[feat] - combined_df[feat].rolling(252).mean()) / combined_df[feat].rolling(252).std()
                result += weight * zscore.fillna(0)
                
    elif operation == 'product':
        result = pd.Series(1.0, index=combined_df.index)
        for feat in features:
            if feat in combined_df.columns:
                result *= combined_df[feat]
                
    elif operation == 'mean':
        result = combined_df.mean(axis=1)
        
    elif operation == 'zscore_sum':
        zscored = (combined_df - combined_df.rolling(252).mean()) / combined_df.rolling(252).std()
        result = zscored.sum(axis=1)
        
    else:
        return {"error": f"Unknown operation: {operation}"}
    
    result = result.dropna()
    
    return {
        "operation": operation,
        "features_used": list(features.keys()),
        "weights": features,
        "stats": {
            "count": len(result),
            "mean": round(result.mean(), 6),
            "std": round(result.std(), 6),
            "min": round(result.min(), 6),
            "max": round(result.max(), 6),
        },
        "sample_values": {str(k): v for k, v in result.tail(5).round(6).to_dict().items()},
    }


def test_feature_predictive_power(
    feature_spec: Dict[str, Any],
    target_factor: str = "VLUE",
    forward_days: List[int] = [5, 10, 21, 63],
) -> Dict[str, Any]:
    """
    Test a new feature's predictive power for a factor.
    
    Args:
        feature_spec: Dict with either:
            - {"base": "VIX", "transform": "zscore", "window": 21} for derived feature
            - {"features": {"VIX": 0.5, "DGS10": -0.3}} for combined feature
        target_factor: Factor to predict (VLUE, SIZE, QUAL, etc.)
        forward_days: Forward return periods to test
        
    Returns:
        Dict with IC values and statistics
    """
    conn = get_db_connection()
    
    # Load factor returns
    query = """
        SELECT price_date as date, close
        FROM market_prices 
        WHERE symbol = ?
        ORDER BY price_date
    """
    factor_df = pd.read_sql_query(query, conn, params=[target_factor])
    factor_df['date'] = pd.to_datetime(factor_df['date'])
    factor_df = factor_df.set_index('date').sort_index()
    
    # Build the feature
    if "base" in feature_spec:
        # Derived feature
        result = create_derived_feature(
            base_feature=feature_spec["base"],
            transformation=feature_spec.get("transform", "zscore"),
            window=feature_spec.get("window", 21),
        )
        if "error" in result:
            return result
        
        # Rebuild the feature series for testing
        base = feature_spec["base"]
        query = "SELECT price_date as date, close as value FROM market_prices WHERE symbol = ? ORDER BY price_date"
        df = pd.read_sql_query(query, conn, params=[base])
        if df.empty:
            query = "SELECT observation_date as date, value FROM observations WHERE series_id = ? ORDER BY observation_date"
            df = pd.read_sql_query(query, conn, params=[base])
        
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()
        
        trans = feature_spec.get("transform", "zscore")
        window = feature_spec.get("window", 21)
        
        if trans == "zscore":
            feature = (df['value'] - df['value'].rolling(window).mean()) / df['value'].rolling(window).std()
        elif trans == "roc":
            feature = df['value'].pct_change(window)
        elif trans == "ma_ratio":
            feature = df['value'] / df['value'].rolling(window).mean() - 1
        elif trans == "volatility":
            feature = df['value'].pct_change().rolling(window).std() * np.sqrt(252)
        else:
            feature = df['value']
            
        feature_name = result["feature_name"]
        
    elif "features" in feature_spec:
        # Combined feature
        feature = pd.Series(0.0)
        for feat, weight in feature_spec["features"].items():
            query = "SELECT price_date as date, close as value FROM market_prices WHERE symbol = ? ORDER BY price_date"
            df = pd.read_sql_query(query, conn, params=[feat])
            if df.empty:
                query = "SELECT observation_date as date, value FROM observations WHERE series_id = ? ORDER BY observation_date"
                df = pd.read_sql_query(query, conn, params=[feat])
            
            if not df.empty:
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date').sort_index()
                zscore = (df['value'] - df['value'].rolling(252).mean()) / df['value'].rolling(252).std()
                if len(feature) == 1:
                    feature = weight * zscore
                else:
                    feature = feature.add(weight * zscore, fill_value=0)
        
        feature_name = "combined_" + "_".join(feature_spec["features"].keys())
    else:
        conn.close()
        return {"error": "Invalid feature_spec. Need 'base' or 'features' key"}
    
    conn.close()
    
    # Align with factor
    combined = pd.DataFrame({
        'feature': feature,
        'factor': factor_df['close']
    }).dropna()
    
    # Calculate forward returns
    results = {
        "feature_name": feature_name,
        "target_factor": target_factor,
        "correlations": {},
    }
    
    for days in forward_days:
        fwd_ret = combined['factor'].pct_change(days).shift(-days)
        valid = pd.DataFrame({'feature': combined['feature'], 'fwd_ret': fwd_ret}).dropna()
        
        if len(valid) > 30:
            ic, pvalue = stats.spearmanr(valid['feature'], valid['fwd_ret'])
            results["correlations"][f"{days}d"] = {
                "ic": round(ic, 4),
                "pvalue": round(pvalue, 4),
                "significant": pvalue < 0.05,
                "observations": len(valid),
            }
    
    # Overall assessment
    avg_ic = np.mean([v["ic"] for v in results["correlations"].values()])
    significant_count = sum(1 for v in results["correlations"].values() if v["significant"])
    
    results["summary"] = {
        "avg_ic": round(avg_ic, 4),
        "significant_horizons": significant_count,
        "total_horizons": len(forward_days),
        "recommendation": "strong" if abs(avg_ic) > 0.05 and significant_count >= 2 else 
                         "moderate" if abs(avg_ic) > 0.02 else "weak"
    }
    
    return results


def get_existing_derived_features(start_date: str = "2016-01-01") -> Dict[str, Any]:
    """
    Get all existing derived features using the same logic as analysis.py _build_features().
    This gives Claude access to the same 69 derived features that Gemini uses.
    """
    from .analysis import get_factor_data
    
    # Use VLUE as a reference factor to get the full feature set
    features, _, _ = get_factor_data("VLUE", start_date)
    
    # Group features by category
    categorized = {
        "market_momentum": [col for col in features.columns if any(x in col for x in ["_mom_", "spy_mom"])],
        "market_volatility": [col for col in features.columns if any(x in col for x in ["_vol_", "spy_vol"])], 
        "rates_features": [col for col in features.columns if any(x in col for x in ["rate_", "yield_", "fed_funds", "mortgage_"])],
        "volatility_risk": [col for col in features.columns if any(x in col for x in ["vix", "hy_spread", "nfci"])],
        "sector_features": [col for col in features.columns if any(x in col.split("_")[0] for x in ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLY", "HYG", "IEF", "IWM", "SHY", "TLT"])],
        "macro_features": [col for col in features.columns if any(x in col for x in ["unrate", "sentiment"])],
        "fed_features": [col for col in features.columns if col.startswith("fed_")],
    }
    
    # Get current values for all features
    latest_values = {}
    for col in features.columns:
        val = features[col].dropna().iloc[-1] if len(features[col].dropna()) > 0 else None
        latest_values[col] = float(val) if val is not None else None
    
    return {
        "total_features": len(features.columns),
        "feature_categories": categorized,
        "all_features": features.columns.tolist(),
        "latest_values": latest_values,
        "date_range": f"{features.index.min().date()} to {features.index.max().date()}",
        "data_points": len(features),
    }


def create_enhanced_features(start_date: str = "2016-01-01") -> Dict[str, Any]:
    """
    Create enhanced features that are missing from the current 69-feature set.
    These are high-value features identified from quant research.
    """
    conn = get_db_connection()
    results = []
    
    try:
        # 1. VIX Term Structure Features
        vix_symbols = ["^VIX", "^VIX1D", "^VIX3M", "^VIX6M"]
        vix_query = f"""
            SELECT symbol, price_date, close
            FROM market_prices 
            WHERE symbol IN ({','.join(['?' for _ in vix_symbols])})
              AND price_date >= ?
            ORDER BY price_date
        """
        vix_df = pd.read_sql_query(vix_query, conn, params=vix_symbols + [start_date])
        if not vix_df.empty:
            vix_df['price_date'] = pd.to_datetime(vix_df['price_date'])
            vix_pivot = vix_df.pivot(index='price_date', columns='symbol', values='close')
            
            if '^VIX' in vix_pivot.columns and '^VIX3M' in vix_pivot.columns:
                vix_term_slope = vix_pivot['^VIX3M'] / vix_pivot['^VIX'] - 1
                results.append({
                    'name': 'vix_term_slope_3m',
                    'description': 'VIX 3M / VIX - 1 (term structure slope)',
                    'latest_value': float(vix_term_slope.dropna().iloc[-1]) if len(vix_term_slope.dropna()) > 0 else None,
                    'type': 'volatility_surface'
                })
        
        # 2. Enhanced Credit Spreads  
        rate_symbols = ["DGS30", "DGS10", "DGS5", "DGS2"]
        rate_query = f"""
            SELECT series_id, observation_date, value
            FROM observations
            WHERE series_id IN ({','.join(['?' for _ in rate_symbols])})
              AND observation_date >= ?
            ORDER BY observation_date
        """
        rate_df = pd.read_sql_query(rate_query, conn, params=rate_symbols + [start_date])
        if not rate_df.empty:
            rate_df['observation_date'] = pd.to_datetime(rate_df['observation_date'])
            rate_pivot = rate_df.pivot(index='observation_date', columns='series_id', values='value')
            
            if 'DGS30' in rate_pivot.columns and 'DGS10' in rate_pivot.columns:
                spread_30_10 = rate_pivot['DGS30'] - rate_pivot['DGS10']
                results.append({
                    'name': 'spread_30y_10y',
                    'description': '30Y-10Y Treasury spread',
                    'latest_value': float(spread_30_10.dropna().iloc[-1]) if len(spread_30_10.dropna()) > 0 else None,
                    'type': 'credit_spreads'
                })
        
        # 3. Real Economic Indicators
        macro_symbols = ["INDPRO", "PAYEMS", "HOUST"]
        for symbol in macro_symbols:
            macro_query = """
                SELECT observation_date, value
                FROM observations
                WHERE series_id = ? AND observation_date >= ?
                ORDER BY observation_date
            """
            macro_df = pd.read_sql_query(macro_query, conn, params=[symbol, start_date])
            if not macro_df.empty:
                macro_df['observation_date'] = pd.to_datetime(macro_df['observation_date'])
                macro_df = macro_df.set_index('observation_date').sort_index()
                
                # Calculate 3-month growth rate
                growth_3m = macro_df['value'].pct_change(3)
                results.append({
                    'name': f'{symbol.lower()}_growth_3m',
                    'description': f'{symbol} 3-month growth rate',
                    'latest_value': float(growth_3m.dropna().iloc[-1]) if len(growth_3m.dropna()) > 0 else None,
                    'type': 'real_economy'
                })
        
        # 4. Commodity Momentum
        commodity_symbols = ["GLD", "USO", "UNG"]
        for symbol in commodity_symbols:
            comm_query = """
                SELECT price_date, close
                FROM market_prices
                WHERE symbol = ? AND price_date >= ?
                ORDER BY price_date
            """
            comm_df = pd.read_sql_query(comm_query, conn, params=[symbol, start_date])
            if not comm_df.empty:
                comm_df['price_date'] = pd.to_datetime(comm_df['price_date'])
                comm_df = comm_df.set_index('price_date').sort_index()
                
                # 63-day momentum
                momentum = comm_df['close'].pct_change(63)
                results.append({
                    'name': f'{symbol.lower()}_mom_63d',
                    'description': f'{symbol} 63-day momentum', 
                    'latest_value': float(momentum.dropna().iloc[-1]) if len(momentum.dropna()) > 0 else None,
                    'type': 'cross_asset'
                })
        
        # 5. Currency Strength
        dxy_query = """
            SELECT price_date, close
            FROM market_prices
            WHERE symbol = 'DX-Y.NYB' AND price_date >= ?
            ORDER BY price_date
        """
        dxy_df = pd.read_sql_query(dxy_query, conn, params=[start_date])
        if not dxy_df.empty:
            dxy_df['price_date'] = pd.to_datetime(dxy_df['price_date'])
            dxy_df = dxy_df.set_index('price_date').sort_index()
            
            dxy_roc = dxy_df['close'].pct_change(21)
            results.append({
                'name': 'dxy_strength_21d',
                'description': 'Dollar index 21-day rate of change',
                'latest_value': float(dxy_roc.dropna().iloc[-1]) if len(dxy_roc.dropna()) > 0 else None,
                'type': 'currency'
            })
        
        # 6. MOVE Index (Bond Volatility)
        move_query = """
            SELECT price_date, close
            FROM market_prices
            WHERE symbol = '^MOVE' AND price_date >= ?
            ORDER BY price_date
        """
        move_df = pd.read_sql_query(move_query, conn, params=[start_date])
        if not move_df.empty:
            move_df['price_date'] = pd.to_datetime(move_df['price_date'])
            move_df = move_df.set_index('price_date').sort_index()
            
            move_zscore = (move_df['close'] - move_df['close'].rolling(252).mean()) / move_df['close'].rolling(252).std()
            results.append({
                'name': 'move_zscore',
                'description': 'MOVE index rolling z-score (bond volatility)',
                'latest_value': float(move_zscore.dropna().iloc[-1]) if len(move_zscore.dropna()) > 0 else None,
                'type': 'volatility'
            })
    
    except Exception as e:
        results.append({'error': str(e)})
    
    finally:
        conn.close()
    
    return {
        'enhanced_features': results,
        'feature_count': len([r for r in results if 'error' not in r]),
        'categories': list(set(r.get('type', 'unknown') for r in results if 'error' not in r))
    }


def get_all_database_features(start_date: str = "2016-01-01") -> Dict[str, Any]:
    """
    Get ALL quantitative time series from the entire database.
    Ensures every numeric feature that could be useful for factor timing is accessible.
    """
    conn = get_db_connection()
    additional_features = {}
    
    try:
        # Economic Calendar features (actual vs estimate surprises)
        econ_cal_query = """
            SELECT event_date, actual, estimate, previous,
                   (actual - estimate) as surprise,
                   (actual - previous) as change_from_prev
            FROM economic_calendar
            WHERE event_date >= ? AND actual IS NOT NULL AND estimate IS NOT NULL
            ORDER BY event_date
        """
        econ_cal = pd.read_sql_query(econ_cal_query, conn, params=[start_date])
        if not econ_cal.empty:
            econ_cal['event_date'] = pd.to_datetime(econ_cal['event_date'])
            econ_cal = econ_cal.set_index('event_date').resample('D').mean()
            for col in econ_cal.columns:
                additional_features[f"econ_{col}"] = econ_cal[col].dropna().iloc[-1] if len(econ_cal[col].dropna()) > 0 else None
        
        # Market microstructure features from market_prices
        micro_query = """
            SELECT price_date, symbol,
                   (high - low) / close as daily_range,
                   volume,
                   change_percent,
                   vwap / close - 1 as vwap_premium
            FROM market_prices
            WHERE price_date >= ? AND volume IS NOT NULL
            ORDER BY price_date
        """
        micro_data = pd.read_sql_query(micro_query, conn, params=[start_date])
        if not micro_data.empty:
            micro_data['price_date'] = pd.to_datetime(micro_data['price_date'])
            
            # Aggregate microstructure features across all symbols
            micro_agg = micro_data.groupby('price_date').agg({
                'daily_range': 'mean',
                'volume': 'mean', 
                'change_percent': ['mean', 'std'],
                'vwap_premium': 'mean'
            }).resample('D').ffill()
            
            # Flatten column names
            micro_agg.columns = ['_'.join(col).strip() for col in micro_agg.columns]
            for col in micro_agg.columns:
                additional_features[f"market_{col}"] = micro_agg[col].dropna().iloc[-1] if len(micro_agg[col].dropna()) > 0 else None
    
    except Exception as e:
        additional_features['error'] = str(e)
    
    finally:
        conn.close()
    
    return {
        'additional_features': additional_features,
        'feature_count': len([k for k in additional_features.keys() if k != 'error']),
        'description': 'Additional quantitative features from economic calendar and market microstructure'
    }


def list_available_base_features() -> Dict[str, Any]:
    """
    List all raw features available for creating derived features.
    """
    conn = get_db_connection()
    
    # Market symbols
    query = "SELECT DISTINCT symbol FROM market_prices ORDER BY symbol"
    market_df = pd.read_sql_query(query, conn)
    
    # Economic series - get distinct series from observations
    query = "SELECT DISTINCT series_id FROM observations ORDER BY series_id"
    econ_df = pd.read_sql_query(query, conn)
    
    conn.close()
    
    return {
        "market_symbols": market_df['symbol'].tolist(),
        "economic_series": econ_df['series_id'].tolist(),
        "available_transformations": [
            "ma_ratio - Price relative to moving average",
            "roc - Rate of change (% change)",
            "zscore - Rolling z-score",
            "volatility - Rolling standard deviation",
            "momentum - Returns over window",
            "diff - First difference",
            "log_diff - Log returns",
            "ema - Exponential moving average",
            "rsi - Relative strength index",
            "rank - Rolling percentile rank",
        ],
        "example_usage": "create_derived_feature('VIX', 'zscore', 21) or create_derived_feature('DGS10', 'roc', 63)",
    }

