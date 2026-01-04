"""
Configuration for Macro Data Framework
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class MacroConfig:
    """Configuration for macro data analysis."""
    
    # Paths
    data_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "data")
    logs_dir: Path = field(default_factory=lambda: Path(__file__).parent / "logs")
    
    # Gemini API
    gemini_api_key: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    gemini_model: str = "gemini-2.5-flash"  # Latest Gemini 2.5 Flash
    gemini_temperature: float = 0.3
    
    # Rate of Change periods (in months for monthly data, quarters for quarterly)
    growth_roc_periods: List[int] = field(default_factory=lambda: [3, 6, 12])  # 3m, 6m, 12m
    inflation_roc_periods: List[int] = field(default_factory=lambda: [3, 6, 12])
    
    # Key series for growth tracking (core)
    growth_series: Dict[str, str] = field(default_factory=lambda: {
        "GDPC1": "Real GDP (Quarterly)",
        "INDPRO": "Industrial Production (Monthly)",
        "PAYEMS": "Nonfarm Payrolls (Monthly)",
        "RSAFS": "Retail Sales (Monthly)",
    })
    
    # Key series for inflation tracking (core)
    inflation_series: Dict[str, str] = field(default_factory=lambda: {
        "CPIAUCSL": "CPI All Items (Monthly)",
        "CPILFESL": "Core CPI (Monthly)",
        "PCEPI": "PCE Price Index (Monthly)",
        "PCEPILFE": "Core PCE (Monthly)",
    })
    
    # ALL available series organized by category (for comprehensive Gemini analysis)
    all_series: Dict[str, Dict[str, str]] = field(default_factory=lambda: {
        "growth": {
            "GDP": "Nominal GDP (Quarterly)",
            "GDPC1": "Real GDP (Quarterly)",
            "A191RL1Q225SBEA": "Real GDP Growth Rate (Quarterly)",
            "INDPRO": "Industrial Production Index (Monthly)",
            "DGORDER": "Durable Goods Orders (Monthly)",
            "RSAFS": "Retail Sales (Monthly)",
        },
        "inflation": {
            "CPIAUCSL": "CPI All Items (Monthly)",
            "CPILFESL": "Core CPI ex Food & Energy (Monthly)",
            "PCEPI": "PCE Price Index (Monthly)",
            "PCEPILFE": "Core PCE ex Food & Energy (Monthly)",
            "T5YIFR": "5-Year Forward Inflation Expectation (Daily)",
            "MICH": "U of Michigan Inflation Expectations (Monthly)",
        },
        "labor": {
            "PAYEMS": "Nonfarm Payrolls (Monthly)",
            "UNRATE": "Unemployment Rate (Monthly)",
            "ICSA": "Initial Jobless Claims (Weekly)",
            "CCSA": "Continued Claims (Weekly)",
            "JTSJOL": "JOLTS Job Openings (Monthly)",
            "CIVPART": "Labor Force Participation Rate (Monthly)",
        },
        "rates": {
            "DFF": "Federal Funds Effective Rate (Daily)",
            "FEDFUNDS": "Federal Funds Rate (Monthly)",
            "DGS2": "2-Year Treasury Yield (Daily)",
            "DGS10": "10-Year Treasury Yield (Daily)",
            "DGS30": "30-Year Treasury Yield (Daily)",
            "T10Y2Y": "10Y-2Y Treasury Spread (Daily)",
            "T10Y3M": "10Y-3M Treasury Spread (Daily)",
            "DFII10": "10-Year TIPS Yield (Daily)",
            "MORTGAGE30US": "30-Year Mortgage Rate (Weekly)",
        },
        "financial_conditions": {
            "NFCI": "Chicago Fed Financial Conditions Index (Weekly)",
            "VIXCLS": "VIX Volatility Index (Daily)",
            "SP500": "S&P 500 Index (Daily)",
            "BAMLH0A0HYM2": "High Yield Credit Spread (Daily)",
            "DTWEXBGS": "Trade Weighted Dollar Index (Daily)",
        },
        "housing": {
            "HOUST": "Housing Starts (Monthly)",
            "PERMIT": "Building Permits (Monthly)",
            "CSUSHPISA": "Case-Shiller Home Price Index (Monthly)",
        },
        "money_credit": {
            "M2SL": "M2 Money Supply (Monthly)",
            "WALCL": "Fed Balance Sheet Total Assets (Weekly)",
            "TOTRESNS": "Total Reserves (Monthly)",
            "DPCREDIT": "Consumer Credit (Monthly)",
        },
        "consumer_sentiment": {
            "UMCSENT": "U of Michigan Consumer Sentiment (Monthly)",
            "MICH": "U of Michigan Inflation Expectations (Monthly)",
        },
        "trade": {
            "BOPGSTB": "Trade Balance (Monthly)",
            "DTWEXBGS": "Trade Weighted Dollar Index (Daily)",
        },
    })
    
    # Qualitative data sources
    qualitative_sources: List[str] = field(default_factory=lambda: [
        "fomc_statements",
        "beige_book",
        "fomc_minutes",
        "fed_speeches",
        "gdpnow_current",
    ])
    
    # Quadrant definitions
    quadrant_names: Dict[int, str] = field(default_factory=lambda: {
        1: "Quad 1: Growth ↑, Inflation ↓",
        2: "Quad 2: Growth ↑, Inflation ↑",
        3: "Quad 3: Growth ↓, Inflation ↓",
        4: "Quad 4: Growth ↓, Inflation ↑",
    })
    
    quadrant_descriptions: Dict[int, str] = field(default_factory=lambda: {
        1: "Goldilocks / Risk-On: Best environment for equities and risk assets. Growth accelerating while inflation decelerates.",
        2: "Inflationary Boom: Good for commodities and cyclicals. Growth and inflation both accelerating.",
        3: "Deflationary Slowdown: Favor bonds and defensives. Growth and inflation both decelerating.",
        4: "Stagflation: Challenging environment. Growth decelerating while inflation accelerates. Favor cash and gold.",
    })
    
    # Asset class preferences by quadrant
    quadrant_assets: Dict[int, Dict[str, str]] = field(default_factory=lambda: {
        1: {"long": "Growth stocks, Tech, Small caps", "short": "Bonds, Utilities, Staples"},
        2: {"long": "Commodities, Energy, Materials, Inflation-protected", "short": "Growth stocks, Long-duration bonds"},
        3: {"long": "Bonds, Utilities, Staples, Quality", "short": "Commodities, Cyclicals"},
        4: {"long": "Cash, Gold, Short-term bonds", "short": "Equities, Corporate bonds"},
    })
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if not self.gemini_api_key:
            print("Warning: GEMINI_API_KEY not found in environment variables")
        
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

