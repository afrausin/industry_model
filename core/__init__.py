"""
Core - Macro Data Analysis Framework
=====================================

This module provides:
- Configuration for macro data analysis
- Data loading from FRED, Fed documents, and other sources
- Gemini AI analysis for qualitative document interpretation
"""

from .config import MacroConfig
from .data_loader import MacroDataLoader, QualitativeDocument, MacroDataPoint
from .gemini_analyzer import (
    GeminiMacroAnalyzer,
    QuadrantProbabilities,
    DocumentSummary,
    PeriodComparison,
    PortfolioRecommendations,
    PreprocessedDocument,
)
from .fmp_loader import FMPDataLoader, MarketQuote, EconomicEvent

__version__ = "0.2.0"
__all__ = [
    # Config
    "MacroConfig",
    # Data Loading
    "MacroDataLoader",
    "QualitativeDocument",
    "MacroDataPoint",
    # FMP Market Data
    "FMPDataLoader",
    "MarketQuote",
    "EconomicEvent",
    # Gemini Analysis
    "GeminiMacroAnalyzer",
    "QuadrantProbabilities",
    "DocumentSummary",
    "PeriodComparison",
    "PortfolioRecommendations",
    "PreprocessedDocument",
]
