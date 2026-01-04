"""
Analysis Tools
==============

Tools for analyzing current model performance and feature correlations.
"""

import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Tuple
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


def get_available_features(start_date: str = "2016-01-01") -> Dict[str, List[str]]:
    """
    Get all available features from the database.
    
    Returns dict with categories and their available features.
    """
    conn = get_db_connection()
    
    features = {}
    
    # Economic series
    query = """
        SELECT DISTINCT series_id FROM observations
        WHERE observation_date >= ?
    """
    df = pd.read_sql_query(query, conn, params=[start_date])
    features["economic_series"] = df["series_id"].tolist()
    
    # Market symbols
    query = """
        SELECT DISTINCT symbol FROM market_prices
        WHERE price_date >= ?
    """
    df = pd.read_sql_query(query, conn, params=[start_date])
    features["market_symbols"] = df["symbol"].tolist()
    
    # Fed document features
    query = """
        SELECT DISTINCT d.document_type
        FROM documents d
        JOIN document_features df ON d.id = df.document_id
        WHERE d.document_date >= ?
    """
    df = pd.read_sql_query(query, conn, params=[start_date])
    features["fed_documents"] = df["document_type"].tolist() if not df.empty else []
    
    conn.close()
    return features


def get_factor_data(
    factor: str,
    start_date: str = "2016-01-01",
    benchmark: str = "SPY",
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Load factor data and compute premium over benchmark.
    
    Returns: (features, factor_returns, premium_returns)
    """
    conn = get_db_connection()
    
    # Load prices
    symbols = [factor, benchmark]
    placeholders = ",".join("?" * len(symbols))
    query = f"""
        SELECT symbol, price_date, close
        FROM market_prices
        WHERE symbol IN ({placeholders})
          AND price_date >= ?
        ORDER BY price_date
    """
    df = pd.read_sql_query(query, conn, params=symbols + [start_date])
    df["price_date"] = pd.to_datetime(df["price_date"])
    prices = df.pivot(index="price_date", columns="symbol", values="close")
    
    # Calculate returns
    returns = prices.pct_change()
    factor_ret = returns[factor]
    benchmark_ret = returns[benchmark]
    premium = factor_ret - benchmark_ret
    
    # Build features
    features = _build_features(prices, start_date, conn)
    
    conn.close()
    
    # Align indices
    common_idx = features.index.intersection(premium.index)
    features = features.loc[common_idx]
    factor_ret = factor_ret.loc[common_idx]
    premium = premium.loc[common_idx]
    
    return features, factor_ret, premium


def _build_features(
    prices: pd.DataFrame,
    start_date: str,
    conn: sqlite3.Connection,
) -> pd.DataFrame:
    """Build comprehensive feature set."""
    features = pd.DataFrame(index=prices.index)
    
    # Market features
    if "SPY" in prices.columns:
        spy = prices["SPY"]
        spy_ret = spy.pct_change()
        features["spy_mom_21d"] = spy.pct_change(21)
        features["spy_mom_63d"] = spy.pct_change(63)
        features["spy_vol_21d"] = spy_ret.rolling(21).std()
        features["spy_vol_63d"] = spy_ret.rolling(63).std()
    
    # Load economic series
    rate_series = ["DGS10", "DGS2", "T10Y2Y", "MORTGAGE30US", "FEDFUNDS"]
    rates = _load_series(rate_series, start_date, conn)
    
    if "DGS10" in rates.columns:
        features["rate_10y"] = rates["DGS10"]
        features["rate_roc_21d"] = rates["DGS10"].pct_change(21)
        features["rate_roc_63d"] = rates["DGS10"].pct_change(63)
        features["rate_zscore"] = _zscore(rates["DGS10"], 252)
    
    if "T10Y2Y" in rates.columns:
        features["yield_curve"] = rates["T10Y2Y"]
        features["yield_curve_zscore"] = _zscore(rates["T10Y2Y"], 252)
    
    if "FEDFUNDS" in rates.columns:
        features["fed_funds"] = rates["FEDFUNDS"]
        features["fed_funds_zscore"] = _zscore(rates["FEDFUNDS"], 252)
    
    if "MORTGAGE30US" in rates.columns:
        features["mortgage_roc"] = rates["MORTGAGE30US"].pct_change(63)
        features["mortgage_zscore"] = _zscore(rates["MORTGAGE30US"], 252)
    
    # Volatility/Risk
    vol_series = ["VIXCLS", "BAMLH0A0HYM2", "NFCI"]
    vol_data = _load_series(vol_series, start_date, conn)
    
    if "VIXCLS" in vol_data.columns:
        features["vix"] = vol_data["VIXCLS"]
        features["vix_roc_21d"] = vol_data["VIXCLS"].pct_change(21)
        features["vix_roc_63d"] = vol_data["VIXCLS"].pct_change(63)
        features["vix_zscore"] = _zscore(vol_data["VIXCLS"], 252)
    
    if "BAMLH0A0HYM2" in vol_data.columns:
        features["hy_spread"] = vol_data["BAMLH0A0HYM2"]
        features["hy_spread_zscore"] = _zscore(vol_data["BAMLH0A0HYM2"], 252)
    
    if "NFCI" in vol_data.columns:
        features["nfci"] = vol_data["NFCI"]
        features["nfci_roc"] = vol_data["NFCI"].diff(21)
    
    # Sectors
    sector_symbols = ["XLB", "XLI", "XLF", "XLK", "XLP", "XLY", "XLE", "XLU", "SHY", "IEF", "TLT", "HYG", "IWM"]
    sectors = _load_market_data(sector_symbols, start_date, conn)
    
    for symbol in sectors.columns:
        price = sectors[symbol]
        ma_50 = price.rolling(50).mean()
        ma_200 = price.rolling(200).mean()
        features[f"{symbol}_ma_ratio"] = ma_50 / ma_200 - 1
        features[f"{symbol}_mom_63d"] = price.pct_change(63)
        features[f"{symbol}_vol_21d"] = price.pct_change().rolling(21).std()
    
    # Macro
    macro_series = ["UNRATE", "UMCSENT"]
    macro = _load_series(macro_series, start_date, conn)
    
    if "UNRATE" in macro.columns:
        features["unrate"] = macro["UNRATE"]
        features["unrate_delta"] = macro["UNRATE"].diff(3)  # 3-month change
    
    if "UMCSENT" in macro.columns:
        features["sentiment"] = macro["UMCSENT"]
        features["sentiment_zscore"] = _zscore(macro["UMCSENT"], 252)
    
    # Fed policy
    fed_features = _load_fed_features(start_date, conn)
    for col in fed_features.columns:
        features[col] = fed_features[col]
    
    return features.ffill()


def _load_series(series_ids: List[str], start_date: str, conn: sqlite3.Connection) -> pd.DataFrame:
    """Load economic series."""
    if not series_ids:
        return pd.DataFrame()
    placeholders = ",".join("?" * len(series_ids))
    query = f"""
        SELECT series_id, observation_date, value
        FROM observations
        WHERE series_id IN ({placeholders})
          AND observation_date >= ?
        ORDER BY observation_date
    """
    df = pd.read_sql_query(query, conn, params=series_ids + [start_date])
    if df.empty:
        return pd.DataFrame()
    df["observation_date"] = pd.to_datetime(df["observation_date"])
    df_wide = df.pivot(index="observation_date", columns="series_id", values="value")
    return df_wide.resample("D").ffill()


def _load_market_data(symbols: List[str], start_date: str, conn: sqlite3.Connection) -> pd.DataFrame:
    """Load market prices."""
    if not symbols:
        return pd.DataFrame()
    placeholders = ",".join("?" * len(symbols))
    query = f"""
        SELECT symbol, price_date, close
        FROM market_prices
        WHERE symbol IN ({placeholders})
          AND price_date >= ?
        ORDER BY price_date
    """
    df = pd.read_sql_query(query, conn, params=symbols + [start_date])
    if df.empty:
        return pd.DataFrame()
    df["price_date"] = pd.to_datetime(df["price_date"])
    return df.pivot(index="price_date", columns="symbol", values="close")


def _load_fed_features(start_date: str, conn: sqlite3.Connection) -> pd.DataFrame:
    """Load ALL quantitative Fed communication features from textual analysis."""
    query = """
        SELECT d.document_date, d.document_type,
               df.policy_stance, df.policy_delta, 
               df.growth_score, df.growth_delta,
               df.inflation_score, df.inflation_delta,
               df.labor_score, df.labor_delta,
               df.quadrant_confidence
        FROM document_features df
        JOIN documents d ON df.document_id = d.id
        WHERE d.document_date >= ?
        ORDER BY d.document_date
    """
    df = pd.read_sql_query(query, conn, params=[start_date])
    if df.empty:
        return pd.DataFrame()
    
    df["document_date"] = pd.to_datetime(df["document_date"])
    
    # Convert implied_quadrant to numeric encoding if present
    # Q1=1, Q2=2, Q3=3, Q4=4 for quantitative analysis
    
    # Separate by document type for richer features
    fomc_data = df[df["document_type"].str.contains("FOMC", na=False)]
    speech_data = df[df["document_type"] == "Fed Speech"]
    beige_data = df[df["document_type"] == "Beige Book"]
    
    features_list = []
    
    # FOMC features (most important for policy)
    if not fomc_data.empty:
        fomc_numeric = fomc_data.drop(columns=["document_type"], errors='ignore')
        fomc_grouped = fomc_numeric.groupby("document_date").mean(numeric_only=True)
        fomc_resampled = fomc_grouped.resample("D").ffill()
        fomc_resampled.columns = [f"fomc_{col}" for col in fomc_resampled.columns]
        features_list.append(fomc_resampled)
    
    # Fed Speech features (forward guidance)
    if not speech_data.empty:
        speech_numeric = speech_data.drop(columns=["document_type"], errors='ignore')
        speech_grouped = speech_numeric.groupby("document_date").mean(numeric_only=True)
        speech_resampled = speech_grouped.resample("D").ffill()
        speech_resampled.columns = [f"speech_{col}" for col in speech_resampled.columns]
        features_list.append(speech_resampled)
    
    # Beige Book features (economic conditions)
    if not beige_data.empty:
        beige_numeric = beige_data.drop(columns=["document_type"], errors='ignore')
        beige_grouped = beige_numeric.groupby("document_date").mean(numeric_only=True)
        beige_resampled = beige_grouped.resample("D").ffill()
        beige_resampled.columns = [f"beige_{col}" for col in beige_resampled.columns]
        features_list.append(beige_resampled)
    
    # Combine all textual features
    if features_list:
        all_features = pd.concat(features_list, axis=1)
        
        # Add legacy column names for backward compatibility
        if "fomc_policy_stance" in all_features.columns:
            all_features["fed_policy"] = all_features["fomc_policy_stance"]
        if "fomc_policy_delta" in all_features.columns:
            all_features["fed_policy_delta"] = all_features["fomc_policy_delta"]
        if "fomc_growth_score" in all_features.columns:
            all_features["fed_growth"] = all_features["fomc_growth_score"]
        if "fomc_labor_score" in all_features.columns:
            all_features["fed_labor"] = all_features["fomc_labor_score"]
        
        return all_features
    
    return pd.DataFrame()


def _zscore(series: pd.Series, window: int) -> pd.Series:
    """Compute rolling z-score."""
    return (series - series.rolling(window).mean()) / series.rolling(window).std()


def analyze_feature_correlations(
    factor: str,
    horizon: int = 21,
    start_date: str = "2016-01-01",
    top_n: int = 20,
    target_type: str = "premium",
    benchmark: str = "SPY",
) -> pd.DataFrame:
    """
    Analyze feature correlations (IC) with forward factor premium or ratio.

    Args:
        factor: Factor ETF symbol (VLUE, SIZE, etc.)
        horizon: Forward return horizon in days
        start_date: Start date for analysis
        top_n: Number of top features to return
        target_type: "premium" (factor - benchmark) or "ratio" (same, but framed as long/short)
        benchmark: Benchmark symbol

    Returns:
        DataFrame with IC, abs_ic, p-value for top features
    """
    features, _, premium = get_factor_data(factor, start_date, benchmark=benchmark)
    # For ratio mode, we use the same premium (factor - benchmark returns)
    
    # Compute forward returns
    forward_ret = premium.rolling(horizon).sum().shift(-horizon)
    
    results = []
    for col in features.columns:
        valid = features[col].notna() & forward_ret.notna()
        if valid.sum() < 100:
            continue
        
        ic, pval = stats.spearmanr(features.loc[valid, col], forward_ret[valid])
        results.append({
            "feature": col,
            "ic": ic,
            "abs_ic": abs(ic),
            "pval": pval,
            "significant": pval < 0.05,
            "sign": "+" if ic > 0 else "-",
        })
    
    df = pd.DataFrame(results).sort_values("abs_ic", ascending=False)
    return df.head(top_n)


def get_current_performance(
    factor: str = "VLUE",
    start_date: str = "2016-01-01",
    train_window: int = 504,
) -> Dict:
    """
    Get current heuristic model performance for a factor.
    
    Returns dict with:
    - strategy_sharpe
    - benchmark_sharpe
    - hit_rate
    - total_return
    - max_drawdown
    - signal_stats
    """
    features, factor_ret, premium = get_factor_data(factor, start_date)
    
    # Compute a simple heuristic signal based on top predictors
    signal = _compute_simple_signal(features, factor)
    
    # Lag signal by 1 day (no lookahead)
    positions = np.sign(signal.shift(1))
    
    # Only use OOS period
    oos_start = train_window
    positions_oos = positions.iloc[oos_start:]
    premium_oos = premium.iloc[oos_start:]
    
    # Strategy returns
    strategy_ret = positions_oos * premium_oos
    strategy_ret = strategy_ret.dropna()
    benchmark_ret = premium_oos.loc[strategy_ret.index]
    
    # Calculate metrics
    def calc_metrics(rets: pd.Series) -> Dict:
        ann_ret = rets.mean() * 252
        ann_vol = rets.std() * np.sqrt(252)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
        cum = (1 + rets).cumprod()
        peak = cum.expanding().max()
        max_dd = ((cum - peak) / peak).min()
        total_ret = (1 + rets).prod() - 1
        return {
            "annual_return": float(ann_ret),
            "annual_vol": float(ann_vol),
            "sharpe": float(sharpe),
            "max_drawdown": float(max_dd),
            "total_return": float(total_ret),
        }
    
    strat_m = calc_metrics(strategy_ret)
    bench_m = calc_metrics(benchmark_ret)
    
    # Hit rate
    signal_dir = np.sign(signal.shift(1).loc[premium_oos.index])
    actual_dir = np.sign(premium_oos)
    hit_rate = (signal_dir == actual_dir).mean()
    
    # Signal statistics
    signal_stats = {
        "mean": float(signal.mean()),
        "std": float(signal.std()),
        "pct_positive": float((signal > 0).mean()),
        "pct_negative": float((signal < 0).mean()),
    }
    
    return {
        "factor": factor,
        "period": f"{strategy_ret.index[0].date()} to {strategy_ret.index[-1].date()}",
        "days": len(strategy_ret),
        "strategy": strat_m,
        "benchmark": bench_m,
        "hit_rate": float(hit_rate),
        "signal_stats": signal_stats,
    }


def _compute_simple_signal(features: pd.DataFrame, factor: str) -> pd.Series:
    """Compute a simple heuristic signal for a factor."""
    signal = pd.Series(0.0, index=features.index)
    
    # Bond trend (empirically strongest for value)
    if "SHY_ma_ratio" in features.columns:
        shy_signal = features["SHY_ma_ratio"] / features["SHY_ma_ratio"].rolling(252).std()
        shy_signal = shy_signal.clip(-3, 3) / 3
        signal += 0.30 * shy_signal.fillna(0)
    
    # VIX change
    if "vix_roc_63d" in features.columns:
        vix_signal = features["vix_roc_63d"] / features["vix_roc_63d"].rolling(252).std()
        vix_signal = vix_signal.clip(-3, 3) / 3
        # Rising VIX = buying opportunity for value (counterintuitive)
        if factor == "VLUE":
            signal += 0.25 * vix_signal.fillna(0)
        else:
            signal -= 0.25 * vix_signal.fillna(0)
    
    # Fed policy
    if "fed_policy_delta" in features.columns:
        fed_signal = -features["fed_policy_delta"] / features["fed_policy_delta"].abs().rolling(50).mean()
        fed_signal = fed_signal.clip(-2, 2) / 2
        signal += 0.20 * fed_signal.fillna(0)
    
    # Rate regime
    if "rate_roc_63d" in features.columns:
        rate_signal = features["rate_roc_63d"] / features["rate_roc_63d"].rolling(252).std()
        rate_signal = rate_signal.clip(-2, 2) / 2
        # Factor-specific rate sensitivity
        if factor in ["VLUE", "SIZE"]:
            signal += 0.15 * rate_signal.fillna(0)  # Rising rates = bullish
        elif factor == "USMV":
            signal -= 0.15 * rate_signal.fillna(0)  # Rising rates = bearish for min vol
    
    # Financial conditions
    if "nfci" in features.columns:
        nfci_signal = features["nfci"] / features["nfci"].rolling(252).std()
        nfci_signal = nfci_signal.clip(-2, 2) / 2
        signal += 0.10 * nfci_signal.fillna(0)
    
    return signal

