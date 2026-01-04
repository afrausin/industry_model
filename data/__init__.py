"""
Macro Data Pipeline
===================

Data collection, storage, and analysis for US macroeconomic data.

Usage:
    # Run data collection
    python -m data.main --status

    # Update data pipeline
    python -m data.update_data --status

    # Import in code
    from data import MacroDataLoader, PITDatabase, MacroConfig
"""

from data.core import (
    MacroConfig,
    MacroDataLoader,
    FMPDataLoader,
    GeminiMacroAnalyzer,
    QualitativeDocument,
    MacroDataPoint,
)

from data.storage.db import (
    PITDatabase,
    PITDataLoader,
    PITObservation,
    SeriesMetadata,
)

__all__ = [
    # Core
    "MacroConfig",
    "MacroDataLoader",
    "FMPDataLoader",
    "GeminiMacroAnalyzer",
    "QualitativeDocument",
    "MacroDataPoint",
    # Storage
    "PITDatabase",
    "PITDataLoader",
    "PITObservation",
    "SeriesMetadata",
]
