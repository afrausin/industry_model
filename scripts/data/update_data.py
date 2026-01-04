#!/usr/bin/env python3
"""
Macro Data Update Pipeline
==========================

A unified script that:
1. Scrapes fresh data from all sources (FRED, Fed, Atlanta Fed, etc.)
2. Updates the SQLite PIT database with new data
3. Optionally updates FMP data (requires API key)

Usage:
    python scripts/update_data.py                  # Full update (scrape + migrate)
    python scripts/update_data.py --scrape-only    # Only scrape, don't update DB
    python scripts/update_data.py --db-only        # Only update DB from existing JSON
    python scripts/update_data.py --source fred    # Scrape specific source only
    python scripts/update_data.py --with-fmp       # Include FMP data refresh
    python scripts/update_data.py --with-fmp-historical  # Include FMP historical prices
    python scripts/update_data.py --with-fmp-historical --fmp-years 20  # 20 years of history
    python scripts/update_data.py --status         # Show current data status
    python scripts/update_data.py --schedule       # Show cron schedule suggestion

Can be scheduled via cron:
    # Daily at 6am (after most US data releases)
    0 6 * * * cd /path/to/macro && ./venv/bin/python scripts/update_data.py >> logs/update.log 2>&1
"""

import argparse
import logging
import os
import sys
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to path (scripts/ is one level down from project root)
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Configure logging
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / f"update_{datetime.now().strftime('%Y%m%d')}.log")
    ]
)
logger = logging.getLogger('UpdatePipeline')


def run_scrapers(sources: Optional[List[str]] = None, force: bool = False) -> Dict[str, bool]:
    """
    Run data scrapers for specified sources.
    
    Args:
        sources: List of sources to scrape. None = all sources.
        force: Force re-download even if unchanged.
        
    Returns:
        Dict mapping source name to success status.
    """
    from ingest.scrapers import DownloadTracker
    from ingest.scrapers.fed_scraper import FederalReserveScraper
    from ingest.scrapers.atlanta_fed_scraper import AtlantaFedScraper
    from ingest.scrapers.fred_scraper import FREDScraper
    from ingest.scrapers.ny_fed_scraper import NYFedScraper
    from ingest.scrapers.cbo_scraper import CBOScraper
    from ingest.scrapers.brookings_scraper import BrookingsScraper
    from ingest.scrapers.nber_scraper import NBERScraper
    from ingest.scrapers.piie_scraper import PIIEScraper
    from ingest.scrapers.imf_scraper import IMFScraper
    from ingest.scrapers.oecd_scraper import OECDScraper
    
    DATA_DIR = PROJECT_ROOT / 'storage' / 'raw'
    TRACKER_PATH = DATA_DIR / 'tracker.json'
    
    # Available scrapers
    scraper_classes = {
        'federal_reserve': FederalReserveScraper,
        'atlanta_fed': AtlantaFedScraper,
        'fred': FREDScraper,
        'ny_fed': NYFedScraper,
        'cbo': CBOScraper,
        'brookings': BrookingsScraper,
        'nber': NBERScraper,
        'piie': PIIEScraper,
        'imf': IMFScraper,
        'oecd': OECDScraper,
    }
    
    # Filter sources if specified
    if sources:
        scraper_classes = {k: v for k, v in scraper_classes.items() if k in sources}
    
    tracker = DownloadTracker(TRACKER_PATH)
    results = {}
    
    logger.info(f"Starting scraper run for {len(scraper_classes)} sources...")
    
    for source_name, scraper_class in scraper_classes.items():
        try:
            logger.info(f"  Scraping {source_name}...")
            
            # Create scraper with API key if needed
            if source_name == 'fred':
                scraper = scraper_class(DATA_DIR, tracker, api_key=os.environ.get('FRED_API_KEY'))
            else:
                scraper = scraper_class(DATA_DIR, tracker)
            
            # Download all datasets
            scraper.download_all(force=force)
            scraper.close()
            
            results[source_name] = True
            logger.info(f"  ✓ {source_name} complete")
            
        except Exception as e:
            results[source_name] = False
            logger.error(f"  ✗ {source_name} failed: {e}")
    
    return results


def run_fmp_update(include_historical: bool = False, historical_years: int = 10) -> Dict[str, int]:
    """
    Update FMP data (economic calendar, quotes, and optionally historical prices).
    
    Uses FMP's stable API (v3 legacy endpoints are deprecated).
    Requires FMP_API_KEY environment variable.
    
    Symbol list is defined in ingest/fmp/fmp_config.py.
    To add new symbols, update that config file.
    
    Args:
        include_historical: If True, also fetch historical price data for all symbols.
        historical_years: Number of years of historical data to fetch (default: 10).
    
    Returns:
        Dict with counts of updated items.
    """
    fmp_api_key = os.environ.get('FMP_API_KEY')
    
    if not fmp_api_key:
        logger.warning("FMP_API_KEY not set. Skipping FMP update.")
        return {"error": "No API key"}
    
    import requests
    import time
    
    FMP_DIR = PROJECT_ROOT / "storage" / "raw" / "fmp"
    FMP_DIR.mkdir(parents=True, exist_ok=True)
    
    HISTORICAL_DIR = FMP_DIR / "historical"
    HISTORICAL_DIR.mkdir(parents=True, exist_ok=True)
    
    stats = {"calendar_events": 0, "quotes": 0, "historical_symbols": 0, "historical_prices": 0}
    
    # Get symbols from centralized config
    try:
        from ingest.fmp import get_all_symbols
        MARKET_SYMBOLS = get_all_symbols()
        logger.info(f"  Loading {len(MARKET_SYMBOLS)} symbols from FMP config...")
    except ImportError as e:
        logger.warning(f"  Could not import FMP config: {e}. Using fallback list.")
        # Fallback to core symbols if config import fails
        MARKET_SYMBOLS = [
            "^VIX", "^VIX1D", "^VIX3M", "^VIX6M", "^VVIX",
            "^TNX", "^TYX", "^IRX", "^FVX",
            "^GSPC", "^DJI", "^IXIC", "^RUT", "^NDX",
            "SPY", "QQQ", "IWM", "TLT", "GLD",
            "VLUE", "QUAL", "MTUM", "SIZE", "USMV",
        ]
    
    try:
        # Update economic calendar (stable API)
        logger.info("  Fetching FMP economic calendar...")
        calendar_url = f"https://financialmodelingprep.com/stable/economic-calendar?apikey={fmp_api_key}"
        response = requests.get(calendar_url, timeout=60)
        response.raise_for_status()
        calendar_data = response.json()
        
        with open(FMP_DIR / "economic-calendar.json", 'w') as f:
            json.dump(calendar_data, f, indent=2)
        
        stats["calendar_events"] = len(calendar_data)
        logger.info(f"  ✓ Economic calendar: {len(calendar_data)} events")
        
    except Exception as e:
        logger.error(f"  Economic calendar failed: {e}")
        stats["calendar_error"] = str(e)
    
    try:
        # Fetch quotes for key market symbols (stable API)
        logger.info("  Fetching FMP market quotes...")
        all_quotes = []
        
        for symbol in MARKET_SYMBOLS:
            try:
                # URL encode the symbol (^ becomes %5E)
                encoded_symbol = symbol.replace("^", "%5E")
                quote_url = f"https://financialmodelingprep.com/stable/quote?symbol={encoded_symbol}&apikey={fmp_api_key}"
                response = requests.get(quote_url, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    if data:
                        all_quotes.extend(data)
            except Exception as e:
                logger.warning(f"    Failed to fetch {symbol}: {e}")
        
        with open(FMP_DIR / "batch-index-quotes.json", 'w') as f:
            json.dump(all_quotes, f, indent=2)
        
        stats["quotes"] = len(all_quotes)
        logger.info(f"  ✓ Market quotes: {len(all_quotes)} symbols")
        
    except Exception as e:
        logger.error(f"  Market quotes failed: {e}")
        stats["quotes_error"] = str(e)
    
    # Fetch historical price data if requested
    if include_historical:
        logger.info(f"  Fetching FMP historical prices ({historical_years} years)...")
        
        # Calculate date range
        from datetime import timedelta
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=historical_years * 365)).strftime("%Y-%m-%d")
        
        for i, symbol in enumerate(MARKET_SYMBOLS):
            try:
                # URL encode the symbol (^ becomes %5E)
                encoded_symbol = symbol.replace("^", "%5E")
                
                # Use the stable API historical-price-eod/full endpoint
                hist_url = (
                    f"https://financialmodelingprep.com/stable/historical-price-eod/full"
                    f"?symbol={encoded_symbol}&from={start_date}&to={end_date}&apikey={fmp_api_key}"
                )
                
                response = requests.get(hist_url, timeout=30)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # The stable API returns a list directly (not nested under "historical")
                    if data and isinstance(data, list) and len(data) > 0:
                        historical_data = data
                        
                        # Save to file (use safe filename for symbols with ^)
                        safe_symbol = symbol.replace("^", "_").replace(".", "_")
                        file_path = HISTORICAL_DIR / f"{safe_symbol}.json"
                        
                        with open(file_path, 'w') as f:
                            json.dump({
                                "symbol": symbol,
                                "fetched_at": datetime.now().isoformat(),
                                "start_date": start_date,
                                "end_date": end_date,
                                "count": len(historical_data),
                                "historical": historical_data
                            }, f, indent=2)
                        
                        stats["historical_symbols"] += 1
                        stats["historical_prices"] += len(historical_data)
                        
                        if (i + 1) % 10 == 0:
                            logger.info(f"    Progress: {i + 1}/{len(MARKET_SYMBOLS)} symbols")
                    else:
                        logger.warning(f"    No historical data for {symbol}")
                else:
                    logger.warning(f"    Failed to fetch {symbol}: HTTP {response.status_code}")
                    
                # Rate limiting - be nice to the API
                time.sleep(0.2)
                
            except Exception as e:
                logger.warning(f"    Failed to fetch historical for {symbol}: {e}")
        
        logger.info(f"  ✓ Historical prices: {stats['historical_symbols']} symbols, {stats['historical_prices']:,} data points")
    
    return stats


def update_database(verbose: bool = True) -> Dict[str, Any]:
    """
    Update the SQLite PIT database from JSON files.
    
    Returns:
        Migration statistics.
    """
    from storage import PITDatabase
    
    logger.info("Updating SQLite PIT database...")
    
    db = PITDatabase()
    
    # Get pre-migration stats
    pre_stats = db.get_stats()
    
    # Run migration (upserts new data)
    migration_stats = db.migrate_from_json(verbose=verbose)
    
    # Get post-migration stats
    post_stats = db.get_stats()
    
    # Calculate delta
    delta = {
        "new_observations": post_stats["n_observations"] - pre_stats["n_observations"],
        "new_documents": post_stats["n_documents"] - pre_stats["n_documents"],
    }
    
    migration_stats["delta"] = delta
    
    return migration_stats


def show_status():
    """Show current data status."""
    from storage import PITDatabase
    
    print("\n" + "=" * 70)
    print("MACRO DATA STATUS")
    print("=" * 70)
    
    # Check JSON data freshness
    data_dir = PROJECT_ROOT / "storage" / "raw"
    
    print("\n📂 DATA FILES (by last modified):")
    sources = [
        ("FRED", data_dir / "fred"),
        ("Federal Reserve", data_dir / "federal_reserve"),
        ("Atlanta Fed", data_dir / "atlanta_fed"),
        ("NY Fed", data_dir / "ny_fed"),
        ("NBER", data_dir / "nber"),
        ("Brookings", data_dir / "brookings"),
        ("FMP", data_dir / "fmp"),
    ]
    
    for name, path in sources:
        if path.exists():
            # Get most recent file
            files = list(path.glob("*.json"))
            if files:
                most_recent = max(files, key=lambda f: f.stat().st_mtime)
                mtime = datetime.fromtimestamp(most_recent.stat().st_mtime)
                age_days = (datetime.now() - mtime).days
                status = "✓" if age_days < 7 else "⚠️" if age_days < 30 else "❌"
                print(f"  {status} {name:20} Last update: {mtime.strftime('%Y-%m-%d %H:%M')} ({age_days}d ago)")
            else:
                print(f"  ❌ {name:20} No data files")
        else:
            print(f"  ❌ {name:20} Directory not found")
    
    # Database stats
    print("\n📊 SQLITE DATABASE:")
    try:
        db = PITDatabase()
        stats = db.get_stats()
        print(f"  Series:       {stats['n_series']}")
        print(f"  Observations: {stats['n_observations']:,}")
        print(f"  Documents:    {stats['n_documents']}")
        print(f"  Date Range:   {stats['date_range']['start']} to {stats['date_range']['end']}")
        print(f"  DB Size:      {stats['db_size_mb']} MB")
    except Exception as e:
        print(f"  ❌ Error reading database: {e}")
    
    # Environment variables
    print("\n🔑 API KEYS:")
    print(f"  FRED_API_KEY:   {'✓ Set' if os.environ.get('FRED_API_KEY') else '❌ Not set'}")
    print(f"  FMP_API_KEY:    {'✓ Set' if os.environ.get('FMP_API_KEY') else '❌ Not set'}")
    print(f"  GEMINI_API_KEY: {'✓ Set' if os.environ.get('GEMINI_API_KEY') else '❌ Not set'}")
    
    print("\n" + "=" * 70)


def show_schedule():
    """Show suggested cron schedule."""
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                    SUGGESTED UPDATE SCHEDULE                          ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                        ║
║  Add these lines to your crontab (crontab -e):                        ║
║                                                                        ║
║  # Daily macro data update at 6:00 AM (after most US releases)        ║
║  0 6 * * * cd /path/to/macro && ./venv/bin/python scripts/update_data.py ║
║                                                                        ║
║  # Weekly full refresh on Sunday at midnight                          ║
║  0 0 * * 0 cd /path/to/macro && ./venv/bin/python scripts/update_data.py --force ║
║                                                                        ║
║  # Hourly FMP update during market hours (9 AM - 5 PM ET, M-F)       ║
║  0 9-17 * * 1-5 cd /path/to/macro && ./venv/bin/python scripts/update_data.py --with-fmp --db-only ║
║                                                                        ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                        ║
║  Key release times (US Eastern):                                      ║
║  • CPI, PPI: 8:30 AM (mid-month)                                     ║
║  • Jobs Report: 8:30 AM (first Friday)                               ║
║  • FOMC: 2:00 PM (8 times/year)                                      ║
║  • GDP: 8:30 AM (quarterly)                                          ║
║                                                                        ║
╚══════════════════════════════════════════════════════════════════════╝
""")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Macro Data Update Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--source', '-s',
        type=str,
        nargs='+',
        help='Specific source(s) to update (e.g., fred federal_reserve)'
    )
    
    parser.add_argument(
        '--scrape-only',
        action='store_true',
        help='Only run scrapers, do not update database'
    )
    
    parser.add_argument(
        '--db-only',
        action='store_true',
        help='Only update database from existing JSON files'
    )
    
    parser.add_argument(
        '--with-fmp',
        action='store_true',
        help='Include FMP data refresh (requires FMP_API_KEY)'
    )
    
    parser.add_argument(
        '--with-fmp-historical',
        action='store_true',
        help='Include FMP historical price data (implies --with-fmp)'
    )
    
    parser.add_argument(
        '--fmp-years',
        type=int,
        default=10,
        help='Years of FMP historical data to fetch (default: 10)'
    )
    
    parser.add_argument(
        '--force', '-f',
        action='store_true',
        help='Force re-download even if data unchanged'
    )
    
    parser.add_argument(
        '--status',
        action='store_true',
        help='Show current data status'
    )
    
    parser.add_argument(
        '--schedule',
        action='store_true',
        help='Show suggested cron schedule'
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Reduce output verbosity'
    )
    
    parser.add_argument(
        '--with-features',
        action='store_true',
        help='Extract document features with Gemini (requires GEMINI_API_KEY)'
    )
    
    parser.add_argument(
        '--features-only',
        action='store_true',
        help='Only extract document features, skip scraping and DB migration'
    )
    
    parser.add_argument(
        '--feature-years',
        type=int,
        default=10,
        help='Years of history for feature extraction (default: 10)'
    )
    
    args = parser.parse_args()
    
    # Handle info commands
    if args.status:
        show_status()
        return
    
    if args.schedule:
        show_schedule()
        return
    
    # Run update pipeline
    start_time = datetime.now()
    
    print("\n" + "=" * 70)
    print(f"MACRO DATA UPDATE PIPELINE")
    print(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    results = {
        "started_at": start_time.isoformat(),
        "scraper_results": {},
        "fmp_results": {},
        "db_results": {},
    }
    
    # Step 1: Run scrapers (unless --db-only or --features-only)
    if not args.db_only and not args.features_only:
        print("\n📥 STEP 1: Scraping data sources...")
        scraper_results = run_scrapers(
            sources=args.source,
            force=args.force
        )
        results["scraper_results"] = scraper_results
        
        success = sum(1 for v in scraper_results.values() if v)
        total = len(scraper_results)
        print(f"\n   Scrapers: {success}/{total} successful")
    
    # Step 2: Update FMP (if requested and not --features-only)
    if (args.with_fmp or args.with_fmp_historical) and not args.features_only:
        include_historical = args.with_fmp_historical
        if include_historical:
            print(f"\n💹 STEP 2: Updating FMP data (including {args.fmp_years} years historical)...")
        else:
            print("\n💹 STEP 2: Updating FMP data...")
        fmp_results = run_fmp_update(
            include_historical=include_historical,
            historical_years=args.fmp_years
        )
        results["fmp_results"] = fmp_results
    
    # Step 3: Update database (unless --scrape-only or --features-only)
    if not args.scrape_only and not args.features_only:
        step_num = 3 if args.with_fmp else 2
        print(f"\n📊 STEP {step_num}: Updating SQLite database...")
        db_results = update_database(verbose=not args.quiet)
        results["db_results"] = db_results
        
        if db_results.get("delta"):
            delta = db_results["delta"]
            print(f"\n   New observations: +{delta.get('new_observations', 0):,}")
            print(f"   New documents:    +{delta.get('new_documents', 0)}")
    
    # Step 4: Extract document features (if requested)
    if args.with_features or args.features_only:
        step_num = 4 if (args.with_fmp and not args.features_only) else 3
        print(f"\n🧠 STEP {step_num}: Extracting document features (Gemini)...")
        
        if not os.environ.get('GEMINI_API_KEY'):
            logger.warning("  GEMINI_API_KEY not set. Skipping feature extraction.")
            results["feature_results"] = {"error": "No API key"}
        else:
            feature_results = update_document_features(
                years=args.feature_years,
                verbose=not args.quiet
            )
            results["feature_results"] = feature_results
            
            print(f"\n   Features extracted: {feature_results.get('extracted', 0)}")
            print(f"   Already up-to-date: {feature_results.get('skipped', 0)}")
    
    # Summary
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print("\n" + "=" * 70)
    print(f"UPDATE COMPLETE")
    print(f"Duration: {duration:.1f} seconds")
    print("=" * 70 + "\n")
    
    # Save results log
    results["ended_at"] = end_time.isoformat()
    results["duration_seconds"] = duration
    
    log_file = LOG_DIR / f"update_results_{start_time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(log_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"Results saved to: {log_file}")


if __name__ == "__main__":
    main()
