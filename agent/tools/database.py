"""
Database Access Tools
=====================

Direct database access for the agent to query any data.
"""

import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Any
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent
DB_PATH = PROJECT_ROOT / "storage" / "database" / "pit_macro.db"


def get_db_connection() -> sqlite3.Connection:
    """Get database connection."""
    return sqlite3.connect(DB_PATH)


def list_database_tables() -> Dict[str, Any]:
    """
    List all tables in the database with row counts.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = cursor.fetchall()
    
    result = {}
    for (table_name,) in tables:
        if table_name.startswith('sqlite'):
            continue
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        count = cursor.fetchone()[0]
        result[table_name] = count
    
    conn.close()
    return {"tables": result}


def list_economic_series() -> Dict[str, Any]:
    """
    List all available economic series with observation counts and date ranges.
    """
    conn = get_db_connection()
    
    query = """
        SELECT 
            o.series_id,
            COUNT(*) as observations,
            MIN(o.observation_date) as first_date,
            MAX(o.observation_date) as last_date,
            sm.title
        FROM observations o
        LEFT JOIN series_metadata sm ON o.series_id = sm.series_id
        GROUP BY o.series_id
        ORDER BY observations DESC
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    return {
        "total_series": len(df),
        "series": df.to_dict("records")
    }


def list_market_symbols() -> Dict[str, Any]:
    """
    List all available market symbols with data counts and date ranges.
    """
    conn = get_db_connection()
    
    query = """
        SELECT 
            symbol,
            COUNT(*) as trading_days,
            MIN(price_date) as first_date,
            MAX(price_date) as last_date
        FROM market_prices
        GROUP BY symbol
        ORDER BY trading_days DESC
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    return {
        "total_symbols": len(df),
        "symbols": df.to_dict("records")
    }


def get_economic_series(
    series_ids: List[str],
    start_date: str = "2010-01-01",
    end_date: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Get economic series data.
    
    Args:
        series_ids: List of FRED series IDs (e.g., ["DGS10", "UNRATE", "VIXCLS"])
        start_date: Start date
        end_date: End date (optional, defaults to latest)
        
    Returns:
        Dict with series data
    """
    conn = get_db_connection()
    
    placeholders = ",".join("?" * len(series_ids))
    query = f"""
        SELECT series_id, observation_date, value
        FROM observations
        WHERE series_id IN ({placeholders})
          AND observation_date >= ?
    """
    params = series_ids + [start_date]
    
    if end_date:
        query += " AND observation_date <= ?"
        params.append(end_date)
    
    query += " ORDER BY observation_date"
    
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    
    if df.empty:
        return {"error": f"No data found for series: {series_ids}"}
    
    # Pivot to wide format
    df_wide = df.pivot(index="observation_date", columns="series_id", values="value")
    
    return {
        "series_ids": series_ids,
        "date_range": f"{df['observation_date'].min()} to {df['observation_date'].max()}",
        "rows": len(df_wide),
        "data": df_wide.tail(20).to_dict()  # Last 20 rows for context
    }


def get_market_prices(
    symbols: List[str],
    start_date: str = "2010-01-01",
    end_date: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Get market price data.
    
    Args:
        symbols: List of symbols (e.g., ["SPY", "VLUE", "XLB"])
        start_date: Start date
        end_date: End date (optional)
        
    Returns:
        Dict with price data
    """
    conn = get_db_connection()
    
    placeholders = ",".join("?" * len(symbols))
    query = f"""
        SELECT symbol, price_date, open, high, low, close, volume
        FROM market_prices
        WHERE symbol IN ({placeholders})
          AND price_date >= ?
    """
    params = symbols + [start_date]
    
    if end_date:
        query += " AND price_date <= ?"
        params.append(end_date)
    
    query += " ORDER BY price_date"
    
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    
    if df.empty:
        return {"error": f"No data found for symbols: {symbols}"}
    
    # Pivot close prices
    close_prices = df.pivot(index="price_date", columns="symbol", values="close")
    
    return {
        "symbols": symbols,
        "date_range": f"{df['price_date'].min()} to {df['price_date'].max()}",
        "rows": len(close_prices),
        "data": close_prices.tail(20).to_dict()  # Last 20 rows
    }


def get_fed_documents(
    document_type: Optional[str] = None,
    start_date: str = "2010-01-01",
    with_features: bool = True,
) -> Dict[str, Any]:
    """
    Get Fed document data with extracted features.
    
    Args:
        document_type: Filter by type (e.g., "FOMC Statement", "FOMC Minutes", "Beige Book")
        start_date: Start date
        with_features: Include extracted features
        
    Returns:
        Dict with document data
    """
    conn = get_db_connection()
    
    if with_features:
        query = """
            SELECT 
                d.document_date,
                d.document_type,
                d.title,
                df.policy_stance,
                df.policy_delta,
                df.growth_score,
                df.inflation_score,
                df.labor_score,
                df.financial_conditions_score
            FROM documents d
            JOIN document_features df ON d.id = df.document_id
            WHERE d.document_date >= ?
        """
    else:
        query = """
            SELECT document_date, document_type, title
            FROM documents
            WHERE document_date >= ?
        """
    
    params = [start_date]
    
    if document_type:
        query += " AND d.document_type = ?"
        params.append(document_type)
    
    query += " ORDER BY d.document_date DESC"
    
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    
    return {
        "total_documents": len(df),
        "document_types": df["document_type"].value_counts().to_dict() if "document_type" in df.columns else {},
        "documents": df.head(30).to_dict("records")  # Most recent 30
    }


def get_economic_calendar(
    event_type: Optional[str] = None,
    start_date: str = "2020-01-01",
    limit: int = 50,
) -> Dict[str, Any]:
    """
    Get economic calendar events.
    
    Args:
        event_type: Filter by event type
        start_date: Start date
        limit: Maximum number of events to return
        
    Returns:
        Dict with calendar events
    """
    conn = get_db_connection()
    
    query = """
        SELECT event_date, event_type, actual, forecast, previous, surprise
        FROM economic_calendar
        WHERE event_date >= ?
    """
    params = [start_date]
    
    if event_type:
        query += " AND event_type LIKE ?"
        params.append(f"%{event_type}%")
    
    query += f" ORDER BY event_date DESC LIMIT {limit}"
    
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    
    # Get unique event types
    cursor = conn.cursor() if conn else get_db_connection().cursor()
    conn2 = get_db_connection()
    cursor2 = conn2.cursor()
    cursor2.execute("SELECT DISTINCT event_type FROM economic_calendar ORDER BY event_type")
    event_types = [row[0] for row in cursor2.fetchall()]
    conn2.close()
    
    return {
        "total_events": len(df),
        "available_event_types": event_types[:50],  # First 50 types
        "events": df.to_dict("records")
    }


def run_custom_query(
    query: str,
    limit: int = 100,
) -> Dict[str, Any]:
    """
    Run a custom SQL query on the database.
    
    Args:
        query: SQL SELECT query (read-only)
        limit: Maximum rows to return
        
    Returns:
        Query results
    """
    # Security: Only allow SELECT queries
    query_lower = query.lower().strip()
    if not query_lower.startswith("select"):
        return {"error": "Only SELECT queries are allowed"}
    
    # Prevent dangerous operations
    dangerous = ["drop", "delete", "update", "insert", "alter", "create", "truncate"]
    for word in dangerous:
        if word in query_lower:
            return {"error": f"Query contains forbidden keyword: {word}"}
    
    conn = get_db_connection()
    
    try:
        # Add LIMIT if not present
        if "limit" not in query_lower:
            query = f"{query} LIMIT {limit}"
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        return {
            "rows": len(df),
            "columns": list(df.columns),
            "data": df.to_dict("records")
        }
    except Exception as e:
        conn.close()
        return {"error": str(e)}

