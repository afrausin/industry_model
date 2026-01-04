"""
PIT Data Loader - DataFrame interface for Point-In-Time queries
================================================================

A convenient wrapper around PITDatabase that provides a pandas DataFrame
interface compatible with the existing MacroDataLoader.

Usage:
    from database import PITDataLoader
    
    # Initialize with a point-in-time date
    loader = PITDataLoader(as_of_date="2024-06-15")
    
    # Load data just like MacroDataLoader
    growth_data = loader.load_growth_data()
    inflation_data = loader.load_inflation_data()
    
    # All returned data respects the PIT constraint
"""

from pathlib import Path
from datetime import datetime, date
from typing import Dict, List, Optional, Any, Union

import pandas as pd
import numpy as np

from .pit_database import PITDatabase


class PITDataLoader:
    """
    Point-In-Time Data Loader with pandas DataFrame interface.
    
    This class wraps PITDatabase to provide the same interface as 
    MacroDataLoader, but with PIT filtering to avoid look-ahead bias.
    """
    
    # Default series for each category
    DEFAULT_GROWTH_SERIES = ["GDPC1", "INDPRO", "PAYEMS", "RSAFS"]
    DEFAULT_INFLATION_SERIES = ["CPIAUCSL", "CPILFESL", "PCEPI", "PCEPILFE"]
    
    def __init__(
        self,
        as_of_date: Union[str, date, datetime],
        db_path: Optional[Path] = None,
    ):
        """
        Initialize PIT data loader.
        
        Args:
            as_of_date: The point-in-time date. All data returned will be
                       what was available on this date (no look-ahead bias).
            db_path: Optional path to SQLite database.
        """
        if isinstance(as_of_date, datetime):
            self.as_of_date = as_of_date.date()
        elif isinstance(as_of_date, str):
            self.as_of_date = datetime.strptime(as_of_date, "%Y-%m-%d").date()
        else:
            self.as_of_date = as_of_date
            
        self.db = PITDatabase(db_path=db_path)
        self._data_availability: Dict[str, str] = {}
    
    def load_fred_series(self, series_id: str) -> pd.DataFrame:
        """
        Load a single FRED series with PIT filtering.
        
        Args:
            series_id: The FRED series ID (e.g., "CPIAUCSL")
            
        Returns:
            DataFrame indexed by date with 'value' column
        """
        df = self.db.query_series(
            series_id=series_id.upper(),
            as_of_date=self.as_of_date,
        )
        
        # Record data availability
        if not df.empty:
            self._data_availability[series_id] = df.index[-1].strftime("%Y-%m-%d")
        else:
            self._data_availability[series_id] = "No data"
        
        # Rename to match MacroDataLoader format
        if 'value' in df.columns:
            df = df[['value']].copy()
            df['series_id'] = series_id.upper()
        
        return df
    
    def load_growth_data(
        self,
        series_ids: Optional[List[str]] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Load all growth-related series with PIT filtering.
        
        Args:
            series_ids: List of series to load. Defaults to standard growth series.
            
        Returns:
            Dict mapping series_id to DataFrame
        """
        if series_ids is None:
            series_ids = self.DEFAULT_GROWTH_SERIES
        
        result = {}
        for series_id in series_ids:
            df = self.load_fred_series(series_id)
            if not df.empty:
                result[series_id] = df
        
        return result
    
    def load_inflation_data(
        self,
        series_ids: Optional[List[str]] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Load all inflation-related series with PIT filtering.
        
        Args:
            series_ids: List of series to load. Defaults to standard inflation series.
            
        Returns:
            Dict mapping series_id to DataFrame
        """
        if series_ids is None:
            series_ids = self.DEFAULT_INFLATION_SERIES
        
        result = {}
        for series_id in series_ids:
            df = self.load_fred_series(series_id)
            if not df.empty:
                result[series_id] = df
        
        return result
    
    def load_all_available_series(self) -> Dict[str, pd.DataFrame]:
        """
        Load ALL available FRED series with PIT filtering.
        
        Returns:
            Dict mapping series_id to DataFrame
        """
        # Get all series in database
        series_list = self.db.get_series_list()
        
        result = {}
        for series_info in series_list:
            series_id = series_info['series_id']
            df = self.load_fred_series(series_id)
            if not df.empty:
                result[series_id] = df
        
        return result
    
    def load_documents(
        self,
        source: Optional[str] = None,
        document_type: Optional[str] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Load qualitative documents with PIT filtering.
        
        Args:
            source: Optional filter by source (e.g., "Federal Reserve")
            document_type: Optional filter by type (e.g., "FOMC Statement")
            limit: Maximum documents to return
            
        Returns:
            List of document dictionaries
        """
        return self.db.query_documents(
            as_of_date=self.as_of_date,
            source=source,
            document_type=document_type,
            limit=limit,
        )
    
    def load_fomc_statements(self, n_recent: int = 5) -> List[Dict[str, Any]]:
        """Load recent FOMC statements available as of the PIT date."""
        return self.load_documents(
            source="Federal Reserve",
            document_type="FOMC Statement",
            limit=n_recent,
        )
    
    def load_beige_book(self, n_recent: int = 3) -> List[Dict[str, Any]]:
        """Load recent Beige Book editions available as of the PIT date."""
        return self.load_documents(
            source="Federal Reserve",
            document_type="Beige Book",
            limit=n_recent,
        )
    
    def load_fomc_minutes(self, n_recent: int = 3) -> List[Dict[str, Any]]:
        """Load recent FOMC minutes available as of the PIT date."""
        return self.load_documents(
            source="Federal Reserve",
            document_type="FOMC Minutes",
            limit=n_recent,
        )
    
    def get_data_availability(self) -> Dict[str, str]:
        """
        Get summary of data availability at the PIT date.
        
        Returns:
            Dict mapping series_id to latest available observation date
        """
        return self._data_availability.copy()
    
    def get_latest_data_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the latest available data at the PIT date.
        
        Returns:
            Dict with summary information
        """
        summary = {
            "as_of_date": self.as_of_date.isoformat(),
            "growth_indicators": {},
            "inflation_indicators": {},
            "documents": [],
        }
        
        # Growth data
        growth_data = self.load_growth_data()
        for series_id, df in growth_data.items():
            if not df.empty:
                latest = df.iloc[-1]
                summary["growth_indicators"][series_id] = {
                    "date": latest.name.strftime("%Y-%m-%d"),
                    "value": float(latest["value"]),
                }
        
        # Inflation data
        inflation_data = self.load_inflation_data()
        for series_id, df in inflation_data.items():
            if not df.empty:
                latest = df.iloc[-1]
                summary["inflation_indicators"][series_id] = {
                    "date": latest.name.strftime("%Y-%m-%d"),
                    "value": float(latest["value"]),
                }
        
        # Documents
        docs = self.load_documents(limit=5)
        for doc in docs:
            summary["documents"].append({
                "type": doc["document_type"],
                "date": doc["document_date"],
                "title": doc.get("title", ""),
            })
        
        return summary


def create_pit_loader(as_of_date: Union[str, date, datetime]) -> PITDataLoader:
    """
    Create a PITDataLoader for a specific point in time.
    
    Args:
        as_of_date: The point-in-time date
        
    Returns:
        Configured PITDataLoader
    """
    return PITDataLoader(as_of_date=as_of_date)

