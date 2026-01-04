#!/usr/bin/env python3
"""
S&P Global PMI PDF Scraper Runner
==================================
Downloads PMI (Purchasing Managers' Index) reports from S&P Global.

Usage:
    python run_spglobal_pmi.py                 # Download all PMI PDFs
    python run_spglobal_pmi.py --check         # Check for new PDFs without downloading
    python run_spglobal_pmi.py --list          # List already downloaded PDFs
    python run_spglobal_pmi.py --us-only       # Download US PMI only
    python run_spglobal_pmi.py --global-only   # Download global/regional PMI only
    python run_spglobal_pmi.py --force         # Force re-download all PDFs
    python run_spglobal_pmi.py --watch         # Watch for new PDFs (runs periodically)
    python run_spglobal_pmi.py --add-url URL   # Add a PDF URL manually
    python run_spglobal_pmi.py --add-from-file FILE  # Add URLs from a file

Finding PDFs:
    Use Google search: filetype:pdf site:spglobal.com pmi
    Then add the PDF URLs using --add-url or edit storage/raw/spglobal/pmi_urls.txt
"""

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
SCRIPTS_DIR = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / 'ingest'))

from scrapers import DownloadTracker
from scrapers.spglobal_scraper import SPGlobalScraper


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(PROJECT_ROOT / 'logs' / 'spglobal_pmi.log')
    ]
)
logger = logging.getLogger('SPGlobalPMI')


# Data directory
DATA_DIR = PROJECT_ROOT / 'storage' / 'raw'
TRACKER_PATH = DATA_DIR / 'tracker.json'


def download_pdfs(dataset_key: str = 'all_pmi', force: bool = False) -> dict:
    """Download PMI PDFs."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    tracker = DownloadTracker(TRACKER_PATH)
    scraper = SPGlobalScraper(DATA_DIR, tracker)
    
    print("\n" + "=" * 60)
    print(f"S&P GLOBAL PMI PDF DOWNLOAD")
    print(f"Dataset: {dataset_key}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    try:
        result = scraper.download_dataset(dataset_key, force=force)
        
        if result:
            print(f"\n✓ Download complete!")
            print(f"  Manifest saved to: {result}")
            
            # Show downloaded PDFs
            pdfs = scraper.list_downloaded_pdfs()
            if pdfs:
                print(f"\n📄 Downloaded PDFs ({len(pdfs)} total):")
                for pdf in pdfs[:10]:  # Show first 10
                    size_kb = pdf['size_bytes'] / 1024
                    print(f"  - {pdf['filename']} ({size_kb:.1f} KB)")
                if len(pdfs) > 10:
                    print(f"  ... and {len(pdfs) - 10} more")
        else:
            print("\n○ No new PDFs to download")
        
        return {'success': True, 'result': result}
        
    except Exception as e:
        logger.error(f"Download failed: {e}")
        print(f"\n❌ Error: {e}")
        return {'success': False, 'error': str(e)}
    
    finally:
        scraper.close()
        print("\n" + "=" * 60)


def check_for_updates() -> dict:
    """Check for new PMI PDFs without downloading."""
    tracker = DownloadTracker(TRACKER_PATH)
    scraper = SPGlobalScraper(DATA_DIR, tracker)
    
    print("\n" + "=" * 60)
    print("CHECKING FOR NEW PMI PDFs")
    print("=" * 60)
    
    try:
        status = scraper.check_for_updates()
        
        print(f"\n📊 Status:")
        print(f"  Total discovered: {status['total_discovered']}")
        print(f"  Already downloaded: {status['already_downloaded']}")
        print(f"  New available: {status['new_available']}")
        
        if status['new_pdfs']:
            print(f"\n📄 New PDFs available:")
            for pdf in status['new_pdfs'][:10]:
                print(f"  - {pdf['title']}")
                print(f"    URL: {pdf['url']}")
            if len(status['new_pdfs']) > 10:
                print(f"  ... and {len(status['new_pdfs']) - 10} more")
        
        return status
        
    finally:
        scraper.close()
        print("\n" + "=" * 60)


def list_downloaded() -> list:
    """List all downloaded PDF files."""
    tracker = DownloadTracker(TRACKER_PATH)
    scraper = SPGlobalScraper(DATA_DIR, tracker)
    
    print("\n" + "=" * 60)
    print("DOWNLOADED PMI PDFs")
    print("=" * 60)
    
    try:
        pdfs = scraper.list_downloaded_pdfs()
        
        if not pdfs:
            print("\nNo PDFs downloaded yet.")
            return []
        
        print(f"\n📄 {len(pdfs)} PDFs downloaded:\n")
        
        for pdf in pdfs:
            size_kb = pdf['size_bytes'] / 1024
            print(f"  📁 {pdf['filename']}")
            print(f"     Size: {size_kb:.1f} KB | Modified: {pdf['modified'][:10]}")
        
        return pdfs
        
    finally:
        scraper.close()
        print("\n" + "=" * 60)


def watch_for_updates(interval_hours: float = 6.0):
    """
    Continuously watch for new PMI PDFs.
    
    Checks periodically and downloads any new PDFs found.
    Press Ctrl+C to stop.
    """
    print("\n" + "=" * 60)
    print("WATCHING FOR NEW PMI PDFs")
    print(f"Check interval: {interval_hours} hours")
    print("Press Ctrl+C to stop")
    print("=" * 60)
    
    interval_seconds = interval_hours * 3600
    
    while True:
        try:
            print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Checking for updates...")
            
            # Check for new PDFs
            tracker = DownloadTracker(TRACKER_PATH)
            scraper = SPGlobalScraper(DATA_DIR, tracker)
            
            try:
                status = scraper.check_for_updates()
                
                if status['new_available'] > 0:
                    print(f"  Found {status['new_available']} new PDFs, downloading...")
                    scraper.download_dataset('all_pmi', force=False)
                    print("  ✓ Download complete")
                else:
                    print("  No new PDFs found")
                    
            finally:
                scraper.close()
            
            # Wait for next check
            print(f"\n  Next check in {interval_hours} hours...")
            time.sleep(interval_seconds)
            
        except KeyboardInterrupt:
            print("\n\nStopping watch...")
            break


def add_url(url: str, title: str = None) -> bool:
    """Add a PDF URL manually."""
    tracker = DownloadTracker(TRACKER_PATH)
    scraper = SPGlobalScraper(DATA_DIR, tracker)
    
    try:
        result = scraper.add_pdf_url(url, title)
        if result:
            print(f"✓ Added URL: {url}")
        else:
            print(f"○ URL already exists: {url}")
        return result
    finally:
        scraper.close()


def add_urls_from_file(filepath: str) -> int:
    """Add URLs from a text file."""
    tracker = DownloadTracker(TRACKER_PATH)
    scraper = SPGlobalScraper(DATA_DIR, tracker)
    
    try:
        urls = []
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '.pdf' in line.lower():
                    urls.append(line)
        
        added = scraper.add_pdf_urls_batch(urls)
        print(f"✓ Added {added} new URLs from {filepath}")
        return added
    finally:
        scraper.close()


def show_urls_file_location():
    """Show the location of the manual URLs file."""
    urls_file = DATA_DIR / 'spglobal' / 'pmi_urls.txt'
    print(f"\n📁 Manual URLs file: {urls_file}")
    print("\nTo add PDF URLs:")
    print("  1. Search Google: filetype:pdf site:spglobal.com pmi")
    print("  2. Copy PDF URLs and add them to the file above")
    print("  3. Or use: python run_spglobal_pmi.py --add-url <URL>")
    print("\nThen run: python run_spglobal_pmi.py")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='S&P Global PMI PDF Scraper',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--check', '-c',
        action='store_true',
        help='Check for new PDFs without downloading'
    )
    
    parser.add_argument(
        '--list', '-l',
        action='store_true',
        help='List already downloaded PDFs'
    )
    
    parser.add_argument(
        '--us-only',
        action='store_true',
        help='Download US PMI PDFs only'
    )
    
    parser.add_argument(
        '--global-only',
        action='store_true',
        help='Download global/regional PMI PDFs only'
    )
    
    parser.add_argument(
        '--force', '-f',
        action='store_true',
        help='Force re-download even if already downloaded'
    )
    
    parser.add_argument(
        '--watch', '-w',
        action='store_true',
        help='Watch for new PDFs (runs periodically)'
    )
    
    parser.add_argument(
        '--interval',
        type=float,
        default=6.0,
        help='Watch interval in hours (default: 6)'
    )
    
    parser.add_argument(
        '--add-url',
        metavar='URL',
        help='Add a PDF URL manually'
    )
    
    parser.add_argument(
        '--add-from-file',
        metavar='FILE',
        help='Add URLs from a text file (one per line)'
    )
    
    parser.add_argument(
        '--show-config',
        action='store_true',
        help='Show configuration and URLs file location'
    )
    
    args = parser.parse_args()
    
    # Ensure directories exist
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / 'logs').mkdir(parents=True, exist_ok=True)
    
    if args.show_config:
        show_urls_file_location()
    elif args.add_url:
        add_url(args.add_url)
    elif args.add_from_file:
        add_urls_from_file(args.add_from_file)
    elif args.check:
        check_for_updates()
    elif args.list:
        list_downloaded()
    elif args.watch:
        watch_for_updates(interval_hours=args.interval)
    else:
        # Determine dataset key
        if args.us_only:
            dataset_key = 'us_pmi'
        elif args.global_only:
            dataset_key = 'global_pmi'
        else:
            dataset_key = 'all_pmi'
        
        download_pdfs(dataset_key=dataset_key, force=args.force)


if __name__ == '__main__':
    main()

