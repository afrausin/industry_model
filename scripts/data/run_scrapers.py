#!/usr/bin/env python3
"""
Macro Data Scraper - Main Orchestrator
=======================================
Systematic download of US macroeconomic data from official sources.

Usage:
    python main.py                      # Download all data
    python main.py --source bea         # Download from specific source
    python main.py --list               # List available datasets
    python main.py --status             # Show download status
    python main.py --force              # Force re-download even if unchanged

Environment Variables:
    BEA_API_KEY     - Bureau of Economic Analysis API key
    BLS_API_KEY     - Bureau of Labor Statistics API key  
    FRED_API_KEY    - Federal Reserve Economic Data API key

Get free API keys:
    BEA:  https://apps.bea.gov/api/signup/
    BLS:  https://www.bls.gov/developers/api_signature_v2.htm
    FRED: https://fred.stlouisfed.org/docs/api/api_key.html
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent  # macro/
sys.path.insert(0, str(PROJECT_ROOT))

from data.ingest.scrapers import (
    DownloadTracker,
    FederalReserveScraper,
    AtlantaFedScraper,
    FREDScraper,
    NYFedScraper,
    CBOScraper,
)
from data.ingest.scrapers.brookings_scraper import BrookingsScraper
from data.ingest.scrapers.nber_scraper import NBERScraper
from data.ingest.scrapers.piie_scraper import PIIEScraper
from data.ingest.scrapers.imf_scraper import IMFScraper
from data.ingest.scrapers.oecd_scraper import OECDScraper
# from data.ingest.scrapers.spglobal_scraper import SPGlobalScraper  # Not available


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(PROJECT_ROOT / 'scraper.log')
    ]
)
logger = logging.getLogger('MacroScraper')


# Data directory
DATA_DIR = PROJECT_ROOT / 'data' / 'storage' / 'raw'
TRACKER_PATH = DATA_DIR / 'tracker.json'


def get_scraper_classes() -> dict:
    """Return mapping of source names to scraper classes."""
    return {
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
        'spglobal': SPGlobalScraper,
    }


def create_scraper(source: str, tracker: DownloadTracker):
    """Create a scraper instance for the given source."""
    scrapers = get_scraper_classes()
    
    if source not in scrapers:
        raise ValueError(f"Unknown source: {source}. Available: {list(scrapers.keys())}")
    
    scraper_class = scrapers[source]
    
    # Some scrapers need API keys
    if source == 'fred':
        return scraper_class(DATA_DIR, tracker, api_key=os.environ.get('FRED_API_KEY'))
    else:
        return scraper_class(DATA_DIR, tracker)


def list_datasets(source: Optional[str] = None):
    """List available datasets from all or specified sources."""
    tracker = DownloadTracker(TRACKER_PATH)
    scrapers = get_scraper_classes()
    
    sources_to_list = [source] if source else list(scrapers.keys())
    
    print("\n" + "=" * 70)
    print("AVAILABLE MACRO DATASETS")
    print("=" * 70)
    
    for src in sources_to_list:
        try:
            scraper = create_scraper(src, tracker)
            datasets = scraper.get_available_datasets()
            
            print(f"\n📊 {src.upper().replace('_', ' ')}")
            print("-" * 50)
            
            for ds in datasets:
                status = "✓" if tracker.is_downloaded(f"{src}/{ds['key']}") else "○"
                print(f"  {status} {ds['key']}: {ds['name']}")
                if ds.get('description'):
                    print(f"      {ds['description'][:60]}...")
            
            scraper.close()
            
        except Exception as e:
            print(f"\n❌ {src.upper()}: Error - {e}")
    
    print("\n" + "=" * 70)


def show_status():
    """Show current download status and statistics."""
    tracker = DownloadTracker(TRACKER_PATH)
    summary = tracker.get_summary()
    
    print("\n" + "=" * 70)
    print("DOWNLOAD STATUS")
    print("=" * 70)
    
    print(f"\n📈 Total Downloads: {summary['total_downloads']}")
    print(f"📅 Last Updated: {summary['last_updated']}")
    
    print("\n📊 Downloads by Source:")
    print("-" * 30)
    for source, count in sorted(summary['by_source'].items()):
        print(f"  {source}: {count} files")
    
    # Show recent downloads
    downloads = tracker.get_all_downloads()
    if downloads:
        print("\n📥 Recent Downloads:")
        print("-" * 50)
        
        # Sort by download time
        sorted_downloads = sorted(
            downloads.items(),
            key=lambda x: x[1].get('downloaded_at', ''),
            reverse=True
        )[:10]
        
        for key, info in sorted_downloads:
            status = "✓" if info.get('status') == 'success' else "✗"
            time_str = info.get('downloaded_at', 'Unknown')[:19]
            print(f"  {status} {key}")
            print(f"      Downloaded: {time_str}")
    
    print("\n" + "=" * 70)


def download_source(source: str, force: bool = False, dataset_key: Optional[str] = None):
    """Download data from a specific source."""
    tracker = DownloadTracker(TRACKER_PATH)
    
    try:
        scraper = create_scraper(source, tracker)
        
        print(f"\n🔄 Downloading from {source.upper().replace('_', ' ')}...")
        
        if dataset_key:
            # Download specific dataset
            result = scraper.download_dataset(dataset_key, force=force)
            if result:
                print(f"  ✓ Downloaded: {dataset_key}")
            else:
                print(f"  ○ Skipped (unchanged): {dataset_key}")
        else:
            # Download all datasets for this source
            results = scraper.download_all(force=force)
            
            for key, path in results.items():
                if path:
                    print(f"  ✓ Downloaded: {key}")
                else:
                    print(f"  ○ Skipped: {key}")
        
        scraper.close()
        return True
        
    except Exception as e:
        logger.error(f"Error downloading from {source}: {e}")
        print(f"  ❌ Error: {e}")
        return False


def download_all(force: bool = False):
    """Download from all sources."""
    scrapers = get_scraper_classes()
    
    print("\n" + "=" * 70)
    print("MACRO DATA DOWNLOAD")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    success_count = 0
    for source in scrapers.keys():
        if download_source(source, force=force):
            success_count += 1
    
    print("\n" + "=" * 70)
    print(f"COMPLETE: {success_count}/{len(scrapers)} sources processed")
    print("=" * 70 + "\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Macro Data Scraper - Download US macroeconomic data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--source', '-s',
        choices=list(get_scraper_classes().keys()),
        help='Download from specific source only'
    )
    
    parser.add_argument(
        '--dataset', '-d',
        help='Download specific dataset (use with --source)'
    )
    
    parser.add_argument(
        '--list', '-l',
        action='store_true',
        help='List available datasets'
    )
    
    parser.add_argument(
        '--status',
        action='store_true',
        help='Show download status'
    )
    
    parser.add_argument(
        '--force', '-f',
        action='store_true',
        help='Force re-download even if unchanged'
    )
    
    args = parser.parse_args()
    
    # Ensure data directory exists
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    if args.list:
        list_datasets(args.source)
    elif args.status:
        show_status()
    elif args.source:
        download_source(args.source, force=args.force, dataset_key=args.dataset)
    else:
        download_all(force=args.force)


if __name__ == '__main__':
    main()

