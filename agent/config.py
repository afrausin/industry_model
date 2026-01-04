"""
Agent Configuration
"""

from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DB_PATH = PROJECT_ROOT / "storage" / "database" / "pit_macro.db"
HEURISTICS_ROOT = PROJECT_ROOT / "model_heuristics"
OUTPUT_DIR = HEURISTICS_ROOT / "output"
AGENT_OUTPUT_DIR = PROJECT_ROOT / "agent" / "output"

# Model configuration
DEFAULT_MODEL = "gemini-3-pro-preview"
MAX_TOKENS = 100_000
TEMPERATURE = 0.7

# Factors to analyze
FACTORS = ["VLUE", "SIZE", "QUAL", "USMV", "MTUM"]

# Feature categories for optimization
FEATURE_CATEGORIES = {
    "rates": ["DGS10", "DGS2", "DGS30", "T10Y2Y", "MORTGAGE30US", "FEDFUNDS"],
    "volatility": ["VIXCLS", "BAMLH0A0HYM2", "NFCI"],
    "sectors": ["XLB", "XLI", "XLF", "XLK", "XLP", "XLY", "XLE", "XLU"],
    "bonds": ["SHY", "IEF", "TLT", "HYG"],
    "market": ["SPY", "IWM"],
    "fed_policy": ["policy_stance", "policy_delta", "growth_score", "labor_score"],
    "macro": ["UNRATE", "UMCSENT", "PCE", "PAYEMS"],
}

# Backtest parameters
BACKTEST_DEFAULTS = {
    "start_date": "2016-01-01",
    "train_window": 504,  # 2 years
    "test_window": 63,    # 3 months
    "horizon": 21,        # 1 month forward returns
}

# Agent system prompts
SYSTEM_PROMPT = """You are an expert quantitative researcher specializing in factor investing and macro timing strategies.

Your task is to analyze and improve heuristic-based factor timing models that predict when to be long or short equity style factors (Value, Size, Quality, Min Vol, Momentum) relative to SPY.

You have access to:
1. A database with macro indicators (rates, Fed policy, sectors, volatility)
2. Current heuristic model implementations
3. Backtesting capabilities

Key principles:
- Focus on empirically-validated signals (check ICs and hit rates)
- Avoid overfitting - prefer simple, robust rules
- Walk-forward testing is essential for out-of-sample validation
- Factor premiums are often counter-cyclical (value works in distress, momentum in calm)
- Fed policy changes and rate regimes are key drivers

When suggesting improvements:
1. Start with the current model's performance metrics
2. Identify which signal components are working/not working
3. Propose specific, testable modifications
4. Prioritize changes that improve Sharpe ratio and reduce drawdowns
"""
