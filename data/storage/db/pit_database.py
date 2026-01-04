"""
Point-In-Time (PIT) SQLite Database for Macro Data
===================================================

A unified database that enables true point-in-time queries across all macro datasets.
This prevents look-ahead bias in backtesting by tracking when data was actually available.

Key Features:
- Tracks observation_date (what period the data refers to)
- Tracks release_date (when the data was actually published)
- Tracks revisions (preliminary, revised, final estimates)
- Enables PIT queries: "What data was available on date X?"

Usage:
    from database import PITDatabase
    
    db = PITDatabase()
    
    # Get all data as of a specific date
    df = db.query_pit(as_of_date="2024-06-15")
    
    # Get specific series
    df = db.query_series("CPIAUCSL", as_of_date="2024-06-15")
"""

import sqlite3
import json
from pathlib import Path
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from contextlib import contextmanager

import pandas as pd
import numpy as np


# Publication lag estimates for common series (days after observation period ends)
# These are approximate - actual release dates should be scraped when possible
PUBLICATION_LAGS = {
    # Monthly data (days after month end)
    "CPIAUCSL": 13,      # CPI released ~13th of following month
    "CPILFESL": 13,
    "PCEPI": 30,         # PCE released ~30 days after month end
    "PCEPILFE": 30,
    "PAYEMS": 5,         # Employment Situation first Friday
    "UNRATE": 5,
    "INDPRO": 15,        # Industrial Production mid-month
    "RSAFS": 16,         # Retail Sales mid-month
    "HOUST": 18,
    "PERMIT": 18,
    "ICSA": 5,           # Weekly claims, Thursday release
    "CCSA": 5,
    
    # Quarterly data (days after quarter end)
    "GDP": 30,           # Advance estimate ~30 days after quarter
    "GDPC1": 30,
    "A191RL1Q225SBEA": 30,
    
    # Daily data (available next business day)
    "DGS2": 1,
    "DGS10": 1,
    "DGS30": 1,
    "DFF": 1,
    "FEDFUNDS": 1,
    "VIXCLS": 1,
    "SP500": 1,
    
    # Weekly data
    "M2SL": 7,
    "WALCL": 7,
}


@dataclass
class SeriesMetadata:
    """Metadata for an economic time series."""
    series_id: str
    source: str
    description: str
    frequency: str
    units: str
    seasonal_adjustment: str
    last_updated: datetime
    
    
@dataclass 
class PITObservation:
    """A single point-in-time observation."""
    series_id: str
    observation_date: date
    release_date: date
    value: float
    revision_number: int = 0


class PITDatabase:
    """
    Point-In-Time database for macro economic data.
    
    Stores all data with release dates to enable true PIT queries
    that avoid look-ahead bias in backtesting.
    """
    
    SCHEMA_VERSION = 1
    
    def __init__(
        self,
        db_path: Optional[Path] = None,
        data_dir: Optional[Path] = None,
    ):
        """
        Initialize PIT database.
        
        Args:
            db_path: Path to SQLite database file. Defaults to data/pit_macro.db
            data_dir: Path to data directory with JSON files. Defaults to data/
        """
        if db_path is None:
            # Default location: storage/db/pit_macro.db
            self.db_path = Path(__file__).parent / "pit_macro.db"
        else:
            self.db_path = Path(db_path)

        if data_dir is None:
            # Default location: storage/raw/
            self.data_dir = Path(__file__).parent.parent / "raw"
        else:
            self.data_dir = Path(data_dir)
            
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_db()
    
    @contextmanager
    def _get_connection(self):
        """Get a database connection with context management."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
    
    def _init_db(self):
        """Initialize database schema."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Schema version tracking
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS schema_version (
                    version INTEGER PRIMARY KEY,
                    created_at TEXT NOT NULL
                )
            """)
            
            # Series metadata
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS series_metadata (
                    series_id TEXT PRIMARY KEY,
                    source TEXT NOT NULL,
                    description TEXT,
                    frequency TEXT,
                    units TEXT,
                    seasonal_adjustment TEXT,
                    first_observation TEXT,
                    last_observation TEXT,
                    last_updated TEXT,
                    notes TEXT
                )
            """)
            
            # Main observations table with PIT tracking
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS observations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    series_id TEXT NOT NULL,
                    observation_date TEXT NOT NULL,
                    release_date TEXT NOT NULL,
                    value REAL,
                    revision_number INTEGER DEFAULT 0,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(series_id, observation_date, release_date),
                    FOREIGN KEY (series_id) REFERENCES series_metadata(series_id)
                )
            """)
            
            # Indexes for fast PIT queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_obs_series_date 
                ON observations(series_id, observation_date)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_obs_release_date 
                ON observations(release_date)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_obs_pit 
                ON observations(series_id, release_date, observation_date)
            """)
            
            # Qualitative documents table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source TEXT NOT NULL,
                    document_type TEXT NOT NULL,
                    document_date TEXT NOT NULL,
                    release_date TEXT NOT NULL,
                    title TEXT,
                    content TEXT,
                    word_count INTEGER,
                    url TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(source, document_type, document_date)
                )
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_docs_release 
                ON documents(release_date)
            """)
            
            # Record schema version
            cursor.execute("SELECT MAX(version) FROM schema_version")
            result = cursor.fetchone()[0]
            if result is None:
                cursor.execute(
                    "INSERT INTO schema_version (version, created_at) VALUES (?, ?)",
                    (self.SCHEMA_VERSION, datetime.now().isoformat())
                )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("SELECT COUNT(DISTINCT series_id) FROM observations")
            n_series = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM observations")
            n_observations = cursor.fetchone()[0]
            
            cursor.execute("SELECT MIN(observation_date), MAX(observation_date) FROM observations")
            date_range = cursor.fetchone()
            
            cursor.execute("SELECT COUNT(*) FROM documents")
            n_documents = cursor.fetchone()[0]
            
            cursor.execute("""
                SELECT series_id, COUNT(*) as cnt 
                FROM observations 
                GROUP BY series_id 
                ORDER BY cnt DESC 
                LIMIT 10
            """)
            top_series = [dict(row) for row in cursor.fetchall()]
            
            return {
                "n_series": n_series,
                "n_observations": n_observations,
                "n_documents": n_documents,
                "date_range": {
                    "start": date_range[0],
                    "end": date_range[1],
                },
                "top_series": top_series,
                "db_size_mb": round(self.db_path.stat().st_size / 1024 / 1024, 2) if self.db_path.exists() else 0,
            }
    
    # =========================================================================
    # DATA INSERTION
    # =========================================================================
    
    def upsert_series_metadata(
        self,
        series_id: str,
        source: str,
        description: str = "",
        frequency: str = "",
        units: str = "",
        seasonal_adjustment: str = "",
        first_observation: str = "",
        last_observation: str = "",
        last_updated: str = "",
        notes: str = "",
    ):
        """Insert or update series metadata."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO series_metadata 
                (series_id, source, description, frequency, units, seasonal_adjustment,
                 first_observation, last_observation, last_updated, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(series_id) DO UPDATE SET
                    source = excluded.source,
                    description = excluded.description,
                    frequency = excluded.frequency,
                    units = excluded.units,
                    seasonal_adjustment = excluded.seasonal_adjustment,
                    first_observation = excluded.first_observation,
                    last_observation = excluded.last_observation,
                    last_updated = excluded.last_updated,
                    notes = excluded.notes
            """, (series_id, source, description, frequency, units, seasonal_adjustment,
                  first_observation, last_observation, last_updated, notes))
    
    def insert_observations(
        self,
        observations: List[PITObservation],
        batch_size: int = 1000,
    ) -> int:
        """
        Insert observations into the database.
        
        Args:
            observations: List of PITObservation objects
            batch_size: Number of observations to insert per batch
            
        Returns:
            Number of observations inserted
        """
        inserted = 0
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            for i in range(0, len(observations), batch_size):
                batch = observations[i:i + batch_size]
                cursor.executemany("""
                    INSERT OR REPLACE INTO observations 
                    (series_id, observation_date, release_date, value, revision_number)
                    VALUES (?, ?, ?, ?, ?)
                """, [
                    (o.series_id, 
                     o.observation_date.isoformat() if isinstance(o.observation_date, date) else o.observation_date,
                     o.release_date.isoformat() if isinstance(o.release_date, date) else o.release_date,
                     o.value, 
                     o.revision_number)
                    for o in batch
                ])
                inserted += len(batch)
        
        return inserted
    
    def insert_document(
        self,
        source: str,
        document_type: str,
        document_date: Union[str, date],
        release_date: Union[str, date],
        title: str = "",
        content: str = "",
        word_count: int = 0,
        url: str = "",
    ):
        """Insert a qualitative document."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            doc_date = document_date.isoformat() if isinstance(document_date, date) else document_date
            rel_date = release_date.isoformat() if isinstance(release_date, date) else release_date
            
            cursor.execute("""
                INSERT OR REPLACE INTO documents
                (source, document_type, document_date, release_date, title, content, word_count, url)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (source, document_type, doc_date, rel_date, title, content, word_count, url))
    
    # =========================================================================
    # POINT-IN-TIME QUERIES
    # =========================================================================
    
    def query_pit(
        self,
        as_of_date: Union[str, date],
        series_ids: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Query all data available as of a specific date.
        
        This returns the latest known value for each series/observation_date
        pair where the release_date <= as_of_date.
        
        Args:
            as_of_date: The point-in-time date
            series_ids: Optional list of series to filter
            
        Returns:
            DataFrame with columns: series_id, observation_date, value, release_date
        """
        if isinstance(as_of_date, date):
            as_of_str = as_of_date.isoformat()
        else:
            as_of_str = as_of_date
            
        with self._get_connection() as conn:
            # Query to get the latest revision for each observation available by as_of_date
            query = """
                WITH latest_revisions AS (
                    SELECT 
                        series_id,
                        observation_date,
                        MAX(release_date) as latest_release
                    FROM observations
                    WHERE release_date <= ?
                    GROUP BY series_id, observation_date
                )
                SELECT 
                    o.series_id,
                    o.observation_date,
                    o.value,
                    o.release_date,
                    o.revision_number
                FROM observations o
                INNER JOIN latest_revisions lr
                    ON o.series_id = lr.series_id
                    AND o.observation_date = lr.observation_date
                    AND o.release_date = lr.latest_release
            """
            
            params = [as_of_str]
            
            if series_ids:
                placeholders = ",".join("?" * len(series_ids))
                query += f" WHERE o.series_id IN ({placeholders})"
                params.extend(series_ids)
            
            query += " ORDER BY o.series_id, o.observation_date"
            
            df = pd.read_sql_query(query, conn, params=params)
            
            if not df.empty:
                df['observation_date'] = pd.to_datetime(df['observation_date'])
                df['release_date'] = pd.to_datetime(df['release_date'])
            
            return df
    
    def query_series(
        self,
        series_id: str,
        as_of_date: Optional[Union[str, date]] = None,
        start_date: Optional[Union[str, date]] = None,
        end_date: Optional[Union[str, date]] = None,
    ) -> pd.DataFrame:
        """
        Query a specific series with optional PIT filtering.
        
        Args:
            series_id: The series to query
            as_of_date: Optional PIT date (returns only data available by this date)
            start_date: Optional start date for observations
            end_date: Optional end date for observations
            
        Returns:
            DataFrame indexed by observation_date with value column
        """
        with self._get_connection() as conn:
            if as_of_date:
                as_of_str = as_of_date.isoformat() if isinstance(as_of_date, date) else as_of_date
                
                query = """
                    WITH latest_revisions AS (
                        SELECT 
                            observation_date,
                            MAX(release_date) as latest_release
                        FROM observations
                        WHERE series_id = ? AND release_date <= ?
                        GROUP BY observation_date
                    )
                    SELECT 
                        o.observation_date,
                        o.value,
                        o.release_date
                    FROM observations o
                    INNER JOIN latest_revisions lr
                        ON o.observation_date = lr.observation_date
                        AND o.release_date = lr.latest_release
                    WHERE o.series_id = ?
                """
                params = [series_id, as_of_str, series_id]
            else:
                # Get latest revision for each observation (current view)
                query = """
                    WITH latest_revisions AS (
                        SELECT 
                            observation_date,
                            MAX(release_date) as latest_release
                        FROM observations
                        WHERE series_id = ?
                        GROUP BY observation_date
                    )
                    SELECT 
                        o.observation_date,
                        o.value,
                        o.release_date
                    FROM observations o
                    INNER JOIN latest_revisions lr
                        ON o.observation_date = lr.observation_date
                        AND o.release_date = lr.latest_release
                    WHERE o.series_id = ?
                """
                params = [series_id, series_id]
            
            # Add date filters
            conditions = []
            if start_date:
                start_str = start_date.isoformat() if isinstance(start_date, date) else start_date
                conditions.append("o.observation_date >= ?")
                params.append(start_str)
            if end_date:
                end_str = end_date.isoformat() if isinstance(end_date, date) else end_date
                conditions.append("o.observation_date <= ?")
                params.append(end_str)
            
            if conditions:
                query += " AND " + " AND ".join(conditions)
            
            query += " ORDER BY o.observation_date"
            
            df = pd.read_sql_query(query, conn, params=params)
            
            if not df.empty:
                df['observation_date'] = pd.to_datetime(df['observation_date'])
                df = df.set_index('observation_date')
            
            return df
    
    def query_documents(
        self,
        as_of_date: Optional[Union[str, date]] = None,
        source: Optional[str] = None,
        document_type: Optional[str] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Query qualitative documents with optional PIT filtering.
        
        Args:
            as_of_date: Optional PIT date
            source: Optional source filter (e.g., "Federal Reserve")
            document_type: Optional type filter (e.g., "FOMC Statement")
            limit: Maximum number of documents to return
            
        Returns:
            List of document dictionaries
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            query = "SELECT * FROM documents WHERE 1=1"
            params = []
            
            if as_of_date:
                as_of_str = as_of_date.isoformat() if isinstance(as_of_date, date) else as_of_date
                query += " AND release_date <= ?"
                params.append(as_of_str)
            
            if source:
                query += " AND source = ?"
                params.append(source)
            
            if document_type:
                query += " AND document_type = ?"
                params.append(document_type)
            
            query += " ORDER BY document_date DESC LIMIT ?"
            params.append(limit)
            
            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]
    
    def get_series_list(self) -> List[Dict[str, Any]]:
        """Get list of all series with metadata."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT 
                    m.*,
                    COUNT(o.id) as observation_count,
                    MIN(o.observation_date) as actual_first_obs,
                    MAX(o.observation_date) as actual_last_obs
                FROM series_metadata m
                LEFT JOIN observations o ON m.series_id = o.series_id
                GROUP BY m.series_id
                ORDER BY m.series_id
            """)
            return [dict(row) for row in cursor.fetchall()]
    
    # =========================================================================
    # DATA MIGRATION FROM JSON
    # =========================================================================
    
    def migrate_from_json(
        self,
        verbose: bool = True,
    ) -> Dict[str, int]:
        """
        Migrate all existing JSON data to the SQLite database.
        
        This will:
        1. Load all FRED series from data/fred/
        2. Load qualitative documents (FOMC statements, Beige Book, etc.)
        3. Estimate release dates based on publication lags
        
        Args:
            verbose: Print progress messages
            
        Returns:
            Dict with migration statistics
        """
        stats = {
            "series_migrated": 0,
            "observations_migrated": 0,
            "documents_migrated": 0,
            "errors": [],
        }
        
        # Migrate FRED data
        fred_stats = self._migrate_fred_data(verbose)
        stats["series_migrated"] += fred_stats["series"]
        stats["observations_migrated"] += fred_stats["observations"]
        stats["errors"].extend(fred_stats.get("errors", []))
        
        # Migrate qualitative documents (FOMC, Beige Book, Minutes, Speeches)
        doc_stats = self._migrate_documents(verbose)
        stats["documents_migrated"] += doc_stats["documents"]
        stats["errors"].extend(doc_stats.get("errors", []))
        
        # Migrate Atlanta Fed GDPNow
        gdpnow_stats = self._migrate_gdpnow(verbose)
        stats["observations_migrated"] += gdpnow_stats["observations"]
        stats["errors"].extend(gdpnow_stats.get("errors", []))
        
        # Migrate NBER Business Cycles
        nber_stats = self._migrate_nber_cycles(verbose)
        stats["observations_migrated"] += nber_stats["observations"]
        stats["errors"].extend(nber_stats.get("errors", []))
        
        # Migrate FMP data (economic calendar, quotes)
        fmp_stats = self._migrate_fmp_data(verbose)
        stats["fmp_events"] = fmp_stats.get("events", 0)
        stats["fmp_quotes"] = fmp_stats.get("quotes", 0)
        stats["errors"].extend(fmp_stats.get("errors", []))
        
        if verbose:
            print(f"\n✅ Migration complete!")
            print(f"   Series: {stats['series_migrated']}")
            print(f"   Observations: {stats['observations_migrated']:,}")
            print(f"   Documents: {stats['documents_migrated']}")
            print(f"   FMP Events: {stats.get('fmp_events', 0)}")
            print(f"   FMP Quotes: {stats.get('fmp_quotes', 0)}")
            if stats["errors"]:
                print(f"   Errors: {len(stats['errors'])}")
        
        return stats
    
    def _migrate_fred_data(self, verbose: bool = True) -> Dict[str, int]:
        """Migrate FRED series from JSON files."""
        fred_dir = self.data_dir / "fred"
        stats = {"series": 0, "observations": 0, "errors": []}
        
        if not fred_dir.exists():
            if verbose:
                print(f"⚠️ FRED directory not found: {fred_dir}")
            return stats
        
        # Find all series files
        series_files = list(fred_dir.glob("series_*.json"))
        
        if verbose:
            print(f"📊 Migrating {len(series_files)} FRED series...")
        
        for file_path in series_files:
            series_id = file_path.stem.replace("series_", "").upper()
            
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                # Extract metadata
                info = data.get("info", {})
                series_info = info.get("seriess", [{}])[0] if info.get("seriess") else {}
                
                self.upsert_series_metadata(
                    series_id=series_id,
                    source="FRED",
                    description=data.get("description", series_info.get("title", "")),
                    frequency=series_info.get("frequency", ""),
                    units=series_info.get("units", ""),
                    seasonal_adjustment=series_info.get("seasonal_adjustment", ""),
                    first_observation=series_info.get("observation_start", ""),
                    last_observation=series_info.get("observation_end", ""),
                    last_updated=series_info.get("last_updated", ""),
                    notes=series_info.get("notes", "")[:1000] if series_info.get("notes") else "",
                )
                
                # Extract observations
                obs_data = data.get("observations", {})
                if isinstance(obs_data, dict):
                    observations_list = obs_data.get("observations", [])
                else:
                    observations_list = obs_data
                
                # Convert to PITObservation objects
                pit_observations = []
                download_date = data.get("downloaded_at", datetime.now().isoformat())[:10]
                frequency = series_info.get("frequency", "")
                
                for obs in observations_list:
                    try:
                        value_str = obs.get("value", "")
                        if value_str == "." or value_str == "":
                            continue
                        
                        value = float(value_str)
                        obs_date = obs.get("date")
                        
                        # Estimate release date based on publication lag
                        # Note: FRED's realtime_start in bulk queries is typically the query date,
                        # not the actual release date, so we estimate instead
                        release_date = self._estimate_release_date(
                            series_id, obs_date, download_date, frequency
                        )
                        
                        pit_observations.append(PITObservation(
                            series_id=series_id,
                            observation_date=obs_date,
                            release_date=release_date,
                            value=value,
                            revision_number=0,
                        ))
                    except (ValueError, TypeError) as e:
                        continue
                
                # Insert observations
                if pit_observations:
                    inserted = self.insert_observations(pit_observations)
                    stats["observations"] += inserted
                
                stats["series"] += 1
                
                if verbose and stats["series"] % 10 == 0:
                    print(f"   Processed {stats['series']} series...")
                    
            except Exception as e:
                stats["errors"].append(f"{series_id}: {str(e)}")
                if verbose:
                    print(f"   ⚠️ Error processing {series_id}: {e}")
        
        return stats
    
    def _estimate_release_date(
        self,
        series_id: str,
        observation_date: str,
        download_date: str,
        frequency: str = "",
    ) -> str:
        """
        Estimate the release date for an observation.
        
        Uses known publication lags for common series, otherwise
        uses a default estimate based on frequency.
        
        For historical data, we use the estimated lag without capping,
        since the data would have been available at the estimated time.
        We only cap for very recent data (within last 60 days).
        """
        try:
            obs_dt = datetime.strptime(observation_date, "%Y-%m-%d")
        except ValueError:
            return download_date
        
        # Get publication lag for this series
        lag_days = PUBLICATION_LAGS.get(series_id.upper())
        
        if lag_days is not None:
            # Use known lag
            release_dt = obs_dt + timedelta(days=lag_days)
        else:
            # Default estimates based on frequency
            freq_lower = frequency.lower() if frequency else ""
            
            if "daily" in freq_lower:
                lag_days = 1
            elif "weekly" in freq_lower:
                lag_days = 7
            elif "quarterly" in freq_lower:
                lag_days = 30  # ~1 month after quarter end
            else:
                # Default for monthly and other
                lag_days = 15
            
            release_dt = obs_dt + timedelta(days=lag_days)
        
        # Only cap at download date for recent observations (last 60 days)
        # For historical data, trust the estimated release date
        download_dt = datetime.strptime(download_date, "%Y-%m-%d")
        days_ago = (download_dt - obs_dt).days
        
        if days_ago < 60 and release_dt > download_dt:
            # For recent data, cap at download date
            release_dt = download_dt
        
        return release_dt.strftime("%Y-%m-%d")
    
    def _migrate_documents(self, verbose: bool = True) -> Dict[str, int]:
        """Migrate qualitative documents from JSON files."""
        stats = {"documents": 0, "errors": []}
        
        fed_dir = self.data_dir / "federal_reserve"
        
        if not fed_dir.exists():
            if verbose:
                print(f"⚠️ Federal Reserve directory not found: {fed_dir}")
            return stats
        
        if verbose:
            print(f"📄 Migrating qualitative documents...")
        
        # FOMC Statements
        statements_file = fed_dir / "fomc_statements.json"
        if statements_file.exists():
            try:
                with open(statements_file, 'r') as f:
                    data = json.load(f)
                
                for stmt in data.get("statements", []):
                    date_str = stmt.get("date", "")
                    if len(date_str) == 8:
                        doc_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
                        # FOMC statements are released same day
                        release_date = doc_date
                        
                        self.insert_document(
                            source="Federal Reserve",
                            document_type="FOMC Statement",
                            document_date=doc_date,
                            release_date=release_date,
                            title=f"FOMC Statement - {doc_date}",
                            content=stmt.get("content", ""),
                            word_count=len(stmt.get("content", "").split()),
                            url=stmt.get("url", ""),
                        )
                        stats["documents"] += 1
            except Exception as e:
                stats["errors"].append(f"FOMC Statements: {str(e)}")
        
        # Beige Book
        beige_file = fed_dir / "beige_book.json"
        if beige_file.exists():
            try:
                with open(beige_file, 'r') as f:
                    data = json.load(f)
                
                for edition in data.get("editions", []):
                    date_str = edition.get("date", "")
                    if len(date_str) == 6:
                        doc_date = f"{date_str[:4]}-{date_str[4:6]}-01"
                        # Beige Book released same day as publication date
                        release_date = doc_date
                        
                        self.insert_document(
                            source="Federal Reserve",
                            document_type="Beige Book",
                            document_date=doc_date,
                            release_date=release_date,
                            title=f"Beige Book - {doc_date}",
                            content=edition.get("content", ""),
                            word_count=edition.get("word_count", 0),
                            url=edition.get("url", ""),
                        )
                        stats["documents"] += 1
            except Exception as e:
                stats["errors"].append(f"Beige Book: {str(e)}")
        
        # FOMC Minutes
        minutes_file = fed_dir / "fomc_minutes.json"
        if minutes_file.exists():
            try:
                with open(minutes_file, 'r') as f:
                    data = json.load(f)
                
                for minutes in data.get("minutes", []):
                    date_str = minutes.get("date", "")
                    if len(date_str) == 8:
                        doc_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
                        # Minutes released ~3 weeks after meeting
                        doc_dt = datetime.strptime(doc_date, "%Y-%m-%d")
                        release_dt = doc_dt + timedelta(days=21)
                        release_date = release_dt.strftime("%Y-%m-%d")
                        
                        self.insert_document(
                            source="Federal Reserve",
                            document_type="FOMC Minutes",
                            document_date=doc_date,
                            release_date=release_date,
                            title=f"FOMC Minutes - {doc_date}",
                            content=minutes.get("content", ""),
                            word_count=minutes.get("word_count", 0),
                            url=minutes.get("url", ""),
                        )
                        stats["documents"] += 1
            except Exception as e:
                stats["errors"].append(f"FOMC Minutes: {str(e)}")
        
        if verbose:
            print(f"   Migrated {stats['documents']} documents")
        
        # Fed Speeches
        speeches_file = fed_dir / "fed_speeches.json"
        if speeches_file.exists():
            try:
                with open(speeches_file, 'r') as f:
                    data = json.load(f)
                
                speech_count = 0
                for speech in data.get("speeches", []):
                    date_str = speech.get("date", "")
                    if len(date_str) == 8:
                        doc_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
                        # Speeches are released same day
                        release_date = doc_date
                        
                        self.insert_document(
                            source="Federal Reserve",
                            document_type="Fed Speech",
                            document_date=doc_date,
                            release_date=release_date,
                            title=speech.get("title", ""),
                            content=speech.get("content", "")[:50000],  # Truncate long speeches
                            word_count=speech.get("word_count", 0),
                            url=speech.get("url", ""),
                        )
                        speech_count += 1
                        stats["documents"] += 1
                
                if verbose:
                    print(f"   Migrated {speech_count} Fed speeches")
            except Exception as e:
                stats["errors"].append(f"Fed Speeches: {str(e)}")
        
        return stats
    
    def _migrate_gdpnow(self, verbose: bool = True) -> Dict[str, int]:
        """Migrate Atlanta Fed GDPNow estimates."""
        stats = {"observations": 0, "errors": []}
        
        gdpnow_file = self.data_dir / "atlanta_fed" / "gdpnow_current.json"
        
        if not gdpnow_file.exists():
            return stats
        
        if verbose:
            print(f"📈 Migrating Atlanta Fed GDPNow...")
        
        try:
            with open(gdpnow_file, 'r') as f:
                data = json.load(f)
            
            estimate = data.get("estimate")
            last_updated = data.get("last_updated", "")
            quarter = data.get("quarter", "")
            download_date = data.get("downloaded_at", "")[:10]
            
            if estimate is not None and last_updated:
                # Parse the last_updated date (e.g., "April 29, 2025")
                try:
                    from dateutil import parser
                    update_dt = parser.parse(last_updated)
                    obs_date = update_dt.strftime("%Y-%m-%d")
                    
                    # Add series metadata
                    self.upsert_series_metadata(
                        series_id="GDPNOW",
                        source="Atlanta Fed",
                        description=f"GDPNow Real GDP Estimate ({quarter})",
                        frequency="Weekly",
                        units="Percent",
                    )
                    
                    # Insert observation
                    self.insert_observations([PITObservation(
                        series_id="GDPNOW",
                        observation_date=obs_date,
                        release_date=obs_date,  # Released same day
                        value=float(estimate),
                        revision_number=0,
                    )])
                    stats["observations"] += 1
                    
                    if verbose:
                        print(f"   GDPNow: {estimate}% ({last_updated})")
                except Exception as e:
                    stats["errors"].append(f"GDPNow parsing: {str(e)}")
        except Exception as e:
            stats["errors"].append(f"GDPNow: {str(e)}")
        
        return stats
    
    def _migrate_nber_cycles(self, verbose: bool = True) -> Dict[str, int]:
        """Migrate NBER business cycle dates."""
        stats = {"observations": 0, "errors": []}
        
        cycles_file = self.data_dir / "nber" / "business_cycles.json"
        
        if not cycles_file.exists():
            return stats
        
        if verbose:
            print(f"📉 Migrating NBER business cycles...")
        
        try:
            with open(cycles_file, 'r') as f:
                data = json.load(f)
            
            # Add series metadata
            self.upsert_series_metadata(
                series_id="NBER_RECESSION",
                source="NBER",
                description="NBER Business Cycle Peak/Trough Dates",
                frequency="Irregular",
                units="Date",
            )
            
            download_date = data.get("downloaded_at", "")[:10]
            
            for cycle in data.get("cycles", []):
                peak = cycle.get("peak", "")
                trough = cycle.get("trough", "")
                
                # Parse dates like "February 2020 (2019Q4)"
                import re
                
                # Extract peak date
                if peak and not peak.startswith("1854") and not peak.startswith("Red"):
                    match = re.search(r'(\w+)\s+(\d{4})', peak)
                    if match:
                        try:
                            from dateutil import parser
                            peak_dt = parser.parse(f"{match.group(1)} 1, {match.group(2)}")
                            
                            self.insert_observations([PITObservation(
                                series_id="NBER_PEAK",
                                observation_date=peak_dt.strftime("%Y-%m-%d"),
                                release_date=download_date,  # Use download date
                                value=1.0,  # 1 = peak
                                revision_number=0,
                            )])
                            stats["observations"] += 1
                        except:
                            pass
                
                # Extract trough date
                if trough and "Duration" not in trough:
                    match = re.search(r'(\w+)\s+(\d{4})', trough)
                    if match:
                        try:
                            from dateutil import parser
                            trough_dt = parser.parse(f"{match.group(1)} 1, {match.group(2)}")
                            
                            self.insert_observations([PITObservation(
                                series_id="NBER_TROUGH",
                                observation_date=trough_dt.strftime("%Y-%m-%d"),
                                release_date=download_date,
                                value=-1.0,  # -1 = trough
                                revision_number=0,
                            )])
                            stats["observations"] += 1
                        except:
                            pass
            
            if verbose:
                print(f"   Migrated {stats['observations']} cycle dates")
                
        except Exception as e:
            stats["errors"].append(f"NBER Cycles: {str(e)}")
        
        return stats
    
    def _migrate_fmp_data(self, verbose: bool = True) -> Dict[str, int]:
        """
        Migrate FMP (Financial Modeling Prep) data.
        
        Includes:
        - Economic Calendar (all countries, all impact levels)
        - Market Quotes (all indices with categorization)
        """
        stats = {"events": 0, "quotes": 0, "errors": []}
        
        # FMP data is in ingest/fmp/exploration_results/
        fmp_dir = self.data_dir.parent.parent / "ingest" / "fmp" / "exploration_results"
        
        if not fmp_dir.exists():
            if verbose:
                print(f"⚠️ FMP directory not found: {fmp_dir}")
            return stats
        
        if verbose:
            print(f"💹 Migrating FMP data...")
        
        # Economic Calendar - ALL events
        calendar_file = fmp_dir / "economic-calendar.json"
        if calendar_file.exists():
            try:
                with open(calendar_file, 'r') as f:
                    events = json.load(f)
                
                # Create table for economic events
                with self._get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        CREATE TABLE IF NOT EXISTS economic_calendar (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            event_date TEXT NOT NULL,
                            country TEXT,
                            event TEXT NOT NULL,
                            currency TEXT,
                            previous REAL,
                            estimate REAL,
                            actual REAL,
                            impact TEXT,
                            unit TEXT,
                            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                            UNIQUE(event_date, event, country)
                        )
                    """)
                    
                    cursor.execute("""
                        CREATE INDEX IF NOT EXISTS idx_econ_cal_date 
                        ON economic_calendar(event_date)
                    """)
                    
                    cursor.execute("""
                        CREATE INDEX IF NOT EXISTS idx_econ_cal_country 
                        ON economic_calendar(country)
                    """)
                    
                    cursor.execute("""
                        CREATE INDEX IF NOT EXISTS idx_econ_cal_impact 
                        ON economic_calendar(impact)
                    """)
                    
                    # Insert ALL events (not just US)
                    for event in events:
                        try:
                            cursor.execute("""
                                INSERT OR REPLACE INTO economic_calendar
                                (event_date, country, event, currency, previous, estimate, actual, impact, unit)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """, (
                                event.get("date", "")[:19],  # Include time if present
                                event.get("country", ""),
                                event.get("event", ""),
                                event.get("currency", ""),
                                event.get("previous"),
                                event.get("estimate"),
                                event.get("actual"),
                                event.get("impact", ""),
                                event.get("unit", ""),
                            ))
                            stats["events"] += 1
                        except:
                            pass
                
                if verbose:
                    # Count by country
                    us_count = len([e for e in events if e.get("country") == "US"])
                    print(f"   Migrated {stats['events']} economic calendar events ({us_count} US)")
                    
            except Exception as e:
                stats["errors"].append(f"Economic Calendar: {str(e)}")
        
        # Market Quotes - ALL quotes with categorization
        quotes_file = fmp_dir / "batch-index-quotes.json"
        if quotes_file.exists():
            try:
                with open(quotes_file, 'r') as f:
                    quotes = json.load(f)
                
                # Get download timestamp from file
                import os
                file_mtime = os.path.getmtime(quotes_file)
                quote_date = datetime.fromtimestamp(file_mtime).strftime("%Y-%m-%d")
                
                # Create table for market quotes with category
                with self._get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        CREATE TABLE IF NOT EXISTS market_quotes (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            symbol TEXT NOT NULL,
                            quote_date TEXT NOT NULL,
                            price REAL,
                            change REAL,
                            volume INTEGER,
                            category TEXT,
                            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                            UNIQUE(symbol, quote_date)
                        )
                    """)
                    
                    cursor.execute("""
                        CREATE INDEX IF NOT EXISTS idx_quotes_date 
                        ON market_quotes(quote_date)
                    """)
                    
                    cursor.execute("""
                        CREATE INDEX IF NOT EXISTS idx_quotes_symbol 
                        ON market_quotes(symbol)
                    """)
                    
                    cursor.execute("""
                        CREATE INDEX IF NOT EXISTS idx_quotes_category 
                        ON market_quotes(category)
                    """)
                    
                    # Categorize symbols
                    categories = {
                        # Volatility
                        "^VIX": "Volatility", "^VIX1D": "Volatility", "^VIX3M": "Volatility",
                        "^VIX6M": "Volatility", "^VVIX": "Volatility",
                        "^MOVE": "Bond Volatility", "^GVZ": "Gold Volatility", "^OVX": "Oil Volatility",
                        "^RVX": "Volatility",
                        # Treasury Yields
                        "^TNX": "Treasury Yields", "^TYX": "Treasury Yields",
                        "^IRX": "Treasury Yields", "^FVX": "Treasury Yields",
                        # US Equities
                        "^GSPC": "US Equities", "^DJI": "US Equities", "^IXIC": "US Equities",
                        "^RUT": "US Equities", "^NDX": "US Equities", "^NYA": "US Equities",
                        "^SP500TR": "US Equities",
                        # Global Markets
                        "^N225": "Global Markets", "^FTSE": "Global Markets",
                        "^GDAXI": "Global Markets", "^HSI": "Global Markets",
                        "^STOXX50E": "Global Markets", "^FCHI": "Global Markets",
                        "^AXJO": "Global Markets", "^KS11": "Global Markets",
                        # Dollar/FX
                        "DX-Y.NYB": "Dollar Index",
                        # Commodities
                        "^SPGSCI": "Commodities",
                    }
                    
                    # Insert ALL quotes
                    for quote in quotes:
                        symbol = quote.get("symbol", "")
                        category = categories.get(symbol, "Other")
                        
                        try:
                            cursor.execute("""
                                INSERT OR REPLACE INTO market_quotes
                                (symbol, quote_date, price, change, volume, category)
                                VALUES (?, ?, ?, ?, ?, ?)
                            """, (
                                symbol,
                                quote_date,
                                quote.get("price"),
                                quote.get("change"),
                                quote.get("volume", 0),
                                category,
                            ))
                            stats["quotes"] += 1
                        except:
                            pass
                
                if verbose:
                    print(f"   Migrated {stats['quotes']} market quotes")
                    
            except Exception as e:
                stats["errors"].append(f"Index Quotes: {str(e)}")
        
        return stats


# =============================================================================
# Convenience functions for common operations
# =============================================================================

def create_database(db_path: Optional[Path] = None) -> PITDatabase:
    """Create and initialize a new PIT database."""
    return PITDatabase(db_path=db_path)


def migrate_all_data(db_path: Optional[Path] = None, verbose: bool = True) -> Dict[str, int]:
    """Migrate all JSON data to SQLite database."""
    db = PITDatabase(db_path=db_path)
    return db.migrate_from_json(verbose=verbose)


# =============================================================================
# CLI
# =============================================================================

def main():
    """Command-line interface for PIT database operations."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Point-In-Time SQLite Database for Macro Data"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Migrate command
    migrate_parser = subparsers.add_parser("migrate", help="Migrate JSON data to SQLite")
    migrate_parser.add_argument("--db", type=str, help="Database path")
    
    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show database statistics")
    stats_parser.add_argument("--db", type=str, help="Database path")
    
    # Query command
    query_parser = subparsers.add_parser("query", help="Run a PIT query")
    query_parser.add_argument("--db", type=str, help="Database path")
    query_parser.add_argument("--as-of", type=str, required=True, help="Point-in-time date (YYYY-MM-DD)")
    query_parser.add_argument("--series", type=str, nargs="+", help="Series to query")
    
    args = parser.parse_args()
    
    if args.command == "migrate":
        db_path = Path(args.db) if args.db else None
        stats = migrate_all_data(db_path=db_path, verbose=True)
        print(f"\nMigration stats: {stats}")
        
    elif args.command == "stats":
        db_path = Path(args.db) if args.db else None
        db = PITDatabase(db_path=db_path)
        stats = db.get_stats()
        print("\n📊 Database Statistics:")
        print(f"   Series: {stats['n_series']}")
        print(f"   Observations: {stats['n_observations']:,}")
        print(f"   Documents: {stats['n_documents']}")
        print(f"   Date Range: {stats['date_range']['start']} to {stats['date_range']['end']}")
        print(f"   DB Size: {stats['db_size_mb']} MB")
        
    elif args.command == "query":
        db_path = Path(args.db) if args.db else None
        db = PITDatabase(db_path=db_path)
        df = db.query_pit(as_of_date=args.as_of, series_ids=args.series)
        print(f"\nData available as of {args.as_of}:")
        print(df.to_string())
        
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

