"""
Data Loaders
============

Point-In-Time data access layer.

Usage:
    from storage import PITDatabase, PITDataLoader
    
    # Option 1: Direct database queries
    db = PITDatabase()
    df = db.query_pit(as_of_date="2024-06-15")
    
    # Option 2: DataFrame interface (compatible with MacroDataLoader)
    loader = PITDataLoader(as_of_date="2024-06-15")
    growth_data = loader.load_growth_data()
    inflation_data = loader.load_inflation_data()
    
    # Migrate existing JSON data (run once)
    db.migrate_from_json()
"""

from .pit_database import (
    PITDatabase,
    PITObservation,
    SeriesMetadata,
    create_database,
    migrate_all_data,
    PUBLICATION_LAGS,
)

from .pit_data_loader import (
    PITDataLoader,
    create_pit_loader,
)

__all__ = [
    # Database
    "PITDatabase",
    "PITObservation", 
    "SeriesMetadata",
    "create_database",
    "migrate_all_data",
    "PUBLICATION_LAGS",
    # Data Loader
    "PITDataLoader",
    "create_pit_loader",
]
