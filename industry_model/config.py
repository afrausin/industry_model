"""
Configuration for Industry ETF Relative Performance Model

Defines sector ETFs, signal parameters, quadrant preferences, and model settings.
"""

from dataclasses import dataclass, field
from typing import Dict, List
from pathlib import Path


# Sector ETF Universe (SPDR Select Sector)
SECTOR_ETFS = [
    "XLK",   # Technology
    "XLF",   # Financials
    "XLE",   # Energy
    "XLV",   # Healthcare
    "XLI",   # Industrials
    "XLP",   # Consumer Staples
    "XLY",   # Consumer Discretionary
    "XLU",   # Utilities
    "XLB",   # Materials
    "XLRE",  # Real Estate
    "XLC",   # Communication Services
]

SECTOR_NAMES = {
    "XLK": "Technology",
    "XLF": "Financials",
    "XLE": "Energy",
    "XLV": "Healthcare",
    "XLI": "Industrials",
    "XLP": "Consumer Staples",
    "XLY": "Consumer Discretionary",
    "XLU": "Utilities",
    "XLB": "Materials",
    "XLRE": "Real Estate",
    "XLC": "Communication Services",
}

# Sector classifications
CYCLICAL_SECTORS = ["XLK", "XLY", "XLI", "XLB", "XLE", "XLF", "XLC"]
DEFENSIVE_SECTORS = ["XLP", "XLU", "XLV", "XLRE"]
RATE_SENSITIVE_SECTORS = ["XLF", "XLRE", "XLU"]

# Quadrant sector preferences (scores from -1 to +1)
# Based on Hedgeye Quadrant Framework
QUADRANT_SECTOR_SCORES: Dict[int, Dict[str, float]] = {
    # Quad 1: Growth Rising, Inflation Falling (Goldilocks/Risk-On)
    1: {
        "XLK": 1.0,    # Tech thrives in low-rate, high-growth
        "XLY": 0.8,    # Consumer discretionary benefits
        "XLC": 0.7,    # Communication services
        "XLF": 0.5,    # Financials benefit from activity
        "XLI": 0.3,    # Industrials modestly positive
        "XLB": 0.2,    # Materials neutral-positive
        "XLE": 0.0,    # Energy neutral
        "XLV": -0.3,   # Healthcare underperforms in risk-on
        "XLP": -0.5,   # Staples underperform
        "XLRE": -0.3,  # REITs mixed (low rates good, risk-off bad)
        "XLU": -0.7,   # Utilities underperform in risk-on
    },
    # Quad 2: Growth Rising, Inflation Rising (Inflationary Boom)
    2: {
        "XLE": 1.0,    # Energy benefits from inflation
        "XLB": 0.8,    # Materials benefit from commodity prices
        "XLI": 0.6,    # Industrials benefit from activity
        "XLF": 0.4,    # Financials: mixed (activity good, curve flattening bad)
        "XLK": 0.0,    # Tech: growth helps, inflation hurts
        "XLY": 0.0,    # Discretionary: mixed
        "XLC": 0.0,    # Communication: mixed
        "XLP": -0.3,   # Staples underperform in risk-on
        "XLU": -0.5,   # Utilities hurt by rising rates
        "XLV": -0.3,   # Healthcare underperforms
        "XLRE": -0.7,  # REITs hurt by rising rates
    },
    # Quad 3: Growth Falling, Inflation Falling (Deflationary Slowdown)
    3: {
        "XLU": 1.0,    # Utilities: defensive + benefit from falling rates
        "XLP": 0.8,    # Staples: defensive
        "XLV": 0.7,    # Healthcare: defensive
        "XLRE": 0.6,   # REITs: benefit from falling rates
        "XLK": 0.0,    # Tech: low rates help, low growth hurts
        "XLF": -0.3,   # Financials: low rates hurt NIM
        "XLC": -0.3,   # Communication: slowing growth
        "XLY": -0.5,   # Discretionary: slowing consumer
        "XLI": -0.5,   # Industrials: slowing activity
        "XLB": -0.7,   # Materials: commodity demand falls
        "XLE": -0.8,   # Energy: demand destruction
    },
    # Quad 4: Growth Falling, Inflation Rising (Stagflation)
    4: {
        "XLE": 0.8,    # Energy: inflation hedge
        "XLU": 0.6,    # Utilities: defensive
        "XLP": 0.5,    # Staples: defensive
        "XLV": 0.3,    # Healthcare: defensive
        "XLB": 0.0,    # Materials: inflation helps, demand hurts
        "XLF": -0.3,   # Financials: slowing economy
        "XLRE": -0.3,  # REITs: rising rates hurt
        "XLC": -0.5,   # Communication: slowing
        "XLI": -0.5,   # Industrials: slowing
        "XLK": -0.5,   # Tech: multiple compression
        "XLY": -0.7,   # Discretionary: slowing consumer + inflation
    },
}

# Yield curve sector sensitivity
# Positive = benefits from steepening, Negative = benefits from flattening
YIELD_CURVE_SENSITIVITY: Dict[str, float] = {
    "XLF": 1.0,     # Financials benefit most from steeper curve (NIM)
    "XLI": 0.5,     # Industrials: steepening signals growth
    "XLK": 0.3,     # Tech: modest benefit from growth signal
    "XLY": 0.3,     # Discretionary: growth signal
    "XLC": 0.2,     # Communication: slight positive
    "XLB": 0.2,     # Materials: growth signal
    "XLE": 0.0,     # Energy: neutral
    "XLV": -0.3,    # Healthcare: prefers low rates
    "XLP": -0.5,    # Staples: defensive, prefers flat/inverted
    "XLRE": -0.7,   # REITs: hurt by rising rates
    "XLU": -0.8,    # Utilities: hurt by rising rates
}

# Credit spread sector sensitivity
# Positive = benefits from tightening spreads (risk-on)
CREDIT_SPREAD_SENSITIVITY: Dict[str, float] = {
    "XLY": 1.0,     # Consumer discretionary: max risk-on
    "XLK": 0.8,     # Tech: high beta
    "XLI": 0.6,     # Industrials: cyclical
    "XLF": 0.5,     # Financials: credit sensitive
    "XLB": 0.4,     # Materials: cyclical
    "XLC": 0.3,     # Communication: moderate beta
    "XLE": 0.2,     # Energy: commodity-driven
    "XLRE": 0.0,    # REITs: mixed
    "XLV": -0.5,    # Healthcare: defensive
    "XLP": -0.7,    # Staples: defensive
    "XLU": -0.8,    # Utilities: defensive
}

# VIX regime sector preferences
VIX_LOW_SECTORS = ["XLK", "XLY", "XLF"]      # Favor in VIX < 15
VIX_HIGH_SECTORS = ["XLU", "XLP", "XLV"]     # Favor in VIX > 25

# Dollar sensitivity (positive = benefits from weak dollar)
DOLLAR_SENSITIVITY: Dict[str, float] = {
    "XLI": 0.7,     # Industrials: export revenue
    "XLK": 0.6,     # Tech: global revenue
    "XLB": 0.5,     # Materials: commodity pricing
    "XLE": 0.3,     # Energy: dollar-denominated commodities
    "XLV": 0.2,     # Healthcare: some global exposure
    "XLY": 0.2,     # Discretionary: mixed
    "XLC": 0.1,     # Communication: some global
    "XLF": 0.0,     # Financials: neutral
    "XLRE": -0.2,   # REITs: domestic
    "XLP": -0.3,    # Staples: domestic defensive
    "XLU": -0.4,    # Utilities: domestic
}

# Housing sensitivity
HOUSING_SENSITIVITY: Dict[str, float] = {
    "XLB": 0.8,     # Materials: construction materials
    "XLI": 0.6,     # Industrials: construction equipment
    "XLRE": 0.5,    # REITs: housing market
    "XLF": 0.4,     # Financials: mortgage lending
    "XLY": 0.3,     # Discretionary: home improvement
    "XLU": 0.0,     # Utilities: neutral
    "XLP": 0.0,     # Staples: neutral
    "XLV": 0.0,     # Healthcare: neutral
    "XLK": -0.1,    # Tech: neutral-slight negative
    "XLC": -0.1,    # Communication: neutral
    "XLE": -0.2,    # Energy: less correlated
}


@dataclass
class ModelConfig:
    """Configuration for the industry rotation model."""

    # Paths
    project_root: Path = field(
        default_factory=lambda: Path(__file__).parent.parent
    )
    cache_dir: Path = field(
        default_factory=lambda: Path(__file__).parent / "cache"
    )

    # FMP API
    fmp_base_url: str = "https://financialmodelingprep.com/stable"

    # Backtest parameters
    rebalance_frequency: str = "W"  # Weekly
    transaction_cost_bps: float = 10.0  # 10 basis points
    slippage_bps: float = 5.0  # 5 basis points

    # Position constraints
    min_sector_weight: float = 0.05  # 5% minimum per sector
    max_sector_weight: float = 0.20  # 20% maximum per sector

    # Signal parameters
    growth_roc_periods: int = 12  # 12 weeks for growth rate-of-change
    inflation_roc_periods: int = 12  # 12 weeks for inflation rate-of-change
    yield_curve_ma_periods: int = 4  # 4-week MA for yield curve
    credit_spread_ma_periods: int = 4  # 4-week MA for credit spreads

    # Walk-forward parameters (dynamic based on data)
    min_window_weeks: int = 26  # Minimum 6 months
    max_is_weeks: int = 156  # Maximum 3 years in-sample
    is_oos_ratio: float = 0.8  # 80% in-sample, 20% out-of-sample

    # Risk limits (per Quant Constitution)
    max_drawdown_limit: float = 0.20  # 20% max drawdown
    min_wfe: float = 0.60  # 60% walk-forward efficiency
    min_sharpe: float = 1.0  # Minimum acceptable Sharpe

    # FRED series for signals
    growth_series: List[str] = field(default_factory=lambda: [
        "INDPRO", "GDPC1", "PAYEMS", "RSAFS", "DGORDER"
    ])
    inflation_series: List[str] = field(default_factory=lambda: [
        "CPIAUCSL", "CPILFESL", "PCEPI", "PCEPILFE", "T5YIFR", "MICH"
    ])
    rate_series: List[str] = field(default_factory=lambda: [
        "DGS2", "DGS10", "DGS30", "T10Y2Y", "T10Y3M", "FEDFUNDS", "DFII10"
    ])
    financial_series: List[str] = field(default_factory=lambda: [
        "VIXCLS", "BAMLH0A0HYM2", "NFCI", "SP500"
    ])
    housing_series: List[str] = field(default_factory=lambda: [
        "HOUST", "PERMIT", "MORTGAGE30US", "CSUSHPISA"
    ])
    money_series: List[str] = field(default_factory=lambda: [
        "M2SL", "WALCL", "TOTRESNS", "DPCREDIT"
    ])
    labor_series: List[str] = field(default_factory=lambda: [
        "PAYEMS", "UNRATE", "ICSA", "CCSA", "JTSJOL", "CIVPART"
    ])
    sentiment_series: List[str] = field(default_factory=lambda: [
        "UMCSENT", "MICH"
    ])
    trade_series: List[str] = field(default_factory=lambda: [
        "DTWEXBGS", "BOPGSTB"
    ])

    def __post_init__(self):
        """Ensure cache directory exists."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def all_fred_series(self) -> List[str]:
        """Return all FRED series used by the model."""
        all_series = set()
        for series_list in [
            self.growth_series,
            self.inflation_series,
            self.rate_series,
            self.financial_series,
            self.housing_series,
            self.money_series,
            self.labor_series,
            self.sentiment_series,
            self.trade_series,
        ]:
            all_series.update(series_list)
        return sorted(all_series)
