#!/usr/bin/env python3
"""
Macro Data Update Pipeline
==========================

A unified script that:
1. Scrapes fresh data from all sources (FRED, Fed, Atlanta Fed, etc.)
2. Updates the SQLite PIT database with new data
3. Optionally updates FMP data (requires API key)

Usage:
    python update_data.py                  # Full update (scrape + migrate)
    python update_data.py --scrape-only    # Only scrape, don't update DB
    python update_data.py --db-only        # Only update DB from existing JSON
    python update_data.py --source fred    # Scrape specific source only
    python update_data.py --with-fmp       # Include FMP data refresh
    python update_data.py --status         # Show current data status
    python update_data.py --schedule       # Show cron schedule suggestion

Can be scheduled via cron:
    # Daily at 6am (after most US data releases)
    0 6 * * * cd /path/to/macro && /path/to/venv/bin/python update_data.py >> logs/update.log 2>&1
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

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
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


def run_fmp_update() -> Dict[str, int]:
    """
    Update FMP data (economic calendar, quotes).
    
    Requires FMP_API_KEY environment variable.
    
    Returns:
        Dict with counts of updated items.
    """
    fmp_api_key = os.environ.get('FMP_API_KEY')
    
    if not fmp_api_key:
        logger.warning("FMP_API_KEY not set. Skipping FMP update.")
        return {"error": "No API key"}
    
    import requests
    
    FMP_DIR = PROJECT_ROOT / "ingest" / "fmp" / "exploration_results"
    FMP_DIR.mkdir(parents=True, exist_ok=True)
    
    stats = {"calendar_events": 0, "quotes": 0}
    
    try:
        # Update economic calendar
        logger.info("  Fetching FMP economic calendar...")
        calendar_url = f"https://financialmodelingprep.com/api/v3/economic_calendar?apikey={fmp_api_key}"
        response = requests.get(calendar_url, timeout=60)
        response.raise_for_status()
        calendar_data = response.json()
        
        with open(FMP_DIR / "economic-calendar.json", 'w') as f:
            json.dump(calendar_data, f, indent=2)
        
        stats["calendar_events"] = len(calendar_data)
        logger.info(f"  ✓ Economic calendar: {len(calendar_data)} events")
        
        # Update batch quotes
        logger.info("  Fetching FMP index quotes...")
        quotes_url = f"https://financialmodelingprep.com/api/v3/quotes/index?apikey={fmp_api_key}"
        response = requests.get(quotes_url, timeout=60)
        response.raise_for_status()
        quotes_data = response.json()
        
        with open(FMP_DIR / "batch-index-quotes.json", 'w') as f:
            json.dump(quotes_data, f, indent=2)
        
        stats["quotes"] = len(quotes_data)
        logger.info(f"  ✓ Index quotes: {len(quotes_data)} symbols")
        
    except Exception as e:
        logger.error(f"  FMP update failed: {e}")
        stats["error"] = str(e)
    
    return stats


def update_database(verbose: bool = True) -> Dict[str, Any]:
    """
    Update the SQLite PIT database from JSON files.
    
    Returns:
        Migration statistics.
    """
    from storage.db import PITDatabase
    
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
    from storage.db import PITDatabase
    
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
    
    # FMP data
    fmp_dir = PROJECT_ROOT / "ingest" / "fmp" / "exploration_results"
    if fmp_dir.exists():
        calendar_file = fmp_dir / "economic-calendar.json"
        if calendar_file.exists():
            mtime = datetime.fromtimestamp(calendar_file.stat().st_mtime)
            age_days = (datetime.now() - mtime).days
            status = "✓" if age_days < 7 else "⚠️"
            print(f"  {status} {'FMP Calendar':20} Last update: {mtime.strftime('%Y-%m-%d %H:%M')} ({age_days}d ago)")
    
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
║  0 6 * * * cd /path/to/macro && ./venv/bin/python update_data.py     ║
║                                                                        ║
║  # Weekly full refresh on Sunday at midnight                          ║
║  0 0 * * 0 cd /path/to/macro && ./venv/bin/python update_data.py --force ║
║                                                                        ║
║  # Hourly FMP update during market hours (9 AM - 5 PM ET, M-F)       ║
║  0 9-17 * * 1-5 cd /path/to/macro && ./venv/bin/python update_data.py --with-fmp --db-only ║
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
    
    # Step 1: Run scrapers (unless --db-only)
    if not args.db_only:
        print("\n📥 STEP 1: Scraping data sources...")
        scraper_results = run_scrapers(
            sources=args.source,
            force=args.force
        )
        results["scraper_results"] = scraper_results
        
        success = sum(1 for v in scraper_results.values() if v)
        total = len(scraper_results)
        print(f"\n   Scrapers: {success}/{total} successful")
    
    # Step 2: Update FMP (if requested)
    if args.with_fmp:
        print("\n💹 STEP 2: Updating FMP data...")
        fmp_results = run_fmp_update()
        results["fmp_results"] = fmp_results
    
    # Step 3: Update database (unless --scrape-only)
    if not args.scrape_only:
        step_num = 3 if args.with_fmp else 2
        print(f"\n📊 STEP {step_num}: Updating SQLite database...")
        db_results = update_database(verbose=not args.quiet)
        results["db_results"] = db_results
        
        if db_results.get("delta"):
            delta = db_results["delta"]
            print(f"\n   New observations: +{delta.get('new_observations', 0):,}")
            print(f"   New documents:    +{delta.get('new_documents', 0)}")
    
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

