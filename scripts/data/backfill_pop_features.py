#!/usr/bin/env python3
"""
Backfill 20 Years of Period-over-Period (PoP) Features
=======================================================

This script backfills historical textual data and extracts PoP features
for the Hedgeye quadrant framework analysis.

Steps:
1. Scrape 20 years of Fed documents (FOMC statements, minutes, Beige Book)
2. Migrate all documents to the SQLite database
3. Extract PoP features using Gemini

Usage:
    # Full backfill (scrape + migrate + extract)
    python scripts/backfill_pop_features.py --full
    
    # Just scrape historical documents
    python scripts/backfill_pop_features.py --scrape
    
    # Just migrate to database
    python scripts/backfill_pop_features.py --migrate
    
    # Just extract features (for documents already in DB)
    python scripts/backfill_pop_features.py --extract
    
    # Extract features for specific document types
    python scripts/backfill_pop_features.py --extract --types "FOMC Statement,Beige Book"
"""

import sys
import re
import json
import argparse
import sqlite3
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
from bs4 import BeautifulSoup

import requests
import time

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from storage.loaders.pit_database import PITDatabase

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(PROJECT_ROOT / 'logs' / 'backfill_pop.log')
    ]
)
logger = logging.getLogger(__name__)


# =============================================================================
# HISTORICAL FED DOCUMENT SCRAPER
# =============================================================================

class HistoricalFedScraper:
    """
    Scrapes historical Fed documents going back 20 years.
    
    The Fed website has archives with different URL structures for older content.
    """
    
    BASE_URL = 'https://www.federalreserve.gov'
    RATE_LIMIT = 2.0  # seconds between requests
    
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir / 'federal_reserve'
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._last_request = 0
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        })
    
    def _rate_limit(self):
        """Enforce rate limiting."""
        elapsed = time.time() - self._last_request
        if elapsed < self.RATE_LIMIT:
            time.sleep(self.RATE_LIMIT - elapsed)
        self._last_request = time.time()
    
    def _fetch(self, url: str) -> Optional[requests.Response]:
        """Fetch a URL with rate limiting and error handling."""
        self._rate_limit()
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            return response
        except Exception as e:
            logger.warning(f"Failed to fetch {url}: {e}")
            return None
    
    def scrape_historical_fomc_statements(self, years_back: int = 20) -> List[Dict]:
        """
        Scrape FOMC statements going back specified years.
        
        The Fed has statements available from 1996 onwards.
        Historical URL patterns:
        - Recent (2017+): /newsevents/pressreleases/monetary{YYYYMMDD}a.htm
        - Older: /monetarypolicy/fomcYYYY.htm archives
        """
        logger.info(f"Scraping FOMC statements going back {years_back} years...")
        
        statements = []
        current_year = datetime.now().year
        start_year = max(1996, current_year - years_back)  # Fed has data from 1996
        
        # Collect all statement links from calendar/archive pages
        statement_links = []
        
        # Current calendar page
        calendar_url = f"{self.BASE_URL}/monetarypolicy/fomccalendars.htm"
        response = self._fetch(calendar_url)
        if response:
            soup = BeautifulSoup(response.text, 'html.parser')
            for link in soup.find_all('a', href=True):
                href = link['href']
                if '/newsevents/pressreleases/monetary' in href:
                    match = re.search(r'monetary(\d{8})a?\.htm$', href)
                    if match:
                        full_url = f"{self.BASE_URL}{href}" if href.startswith('/') else href
                        statement_links.append({
                            'date': match.group(1),
                            'url': full_url
                        })
        
        # Historical archive pages (one per year)
        for year in range(current_year, start_year - 1, -1):
            # Try different historical URL patterns
            archive_urls = [
                f"{self.BASE_URL}/monetarypolicy/fomchistorical{year}.htm",
                f"{self.BASE_URL}/monetarypolicy/fomccalendars.htm",  # Current
            ]
            
            # For older years, use the historical archive
            if year < 2017:
                archive_urls = [f"{self.BASE_URL}/monetarypolicy/fomchistorical{year}.htm"]
            
            for archive_url in archive_urls:
                response = self._fetch(archive_url)
                if response:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    for link in soup.find_all('a', href=True):
                        href = link['href']
                        text = link.get_text(strip=True).lower()
                        
                        # Look for statement links
                        if 'statement' in text or '/newsevents/pressreleases/monetary' in href:
                            # Extract date from various URL patterns
                            match = re.search(r'monetary(\d{8})a?\.htm$', href)
                            if match:
                                full_url = f"{self.BASE_URL}{href}" if href.startswith('/') else href
                                statement_links.append({
                                    'date': match.group(1),
                                    'url': full_url
                                })
                        
                        # Also check for fomc{date} patterns
                        match = re.search(r'fomc(\d{8})\.htm$', href)
                        if match:
                            full_url = f"{self.BASE_URL}{href}" if href.startswith('/') else href
                            statement_links.append({
                                'date': match.group(1),
                                'url': full_url
                            })
            
            logger.info(f"  Checked {year}: found {len(statement_links)} total links so far")
        
        # Deduplicate
        seen = set()
        unique_links = []
        for link in statement_links:
            if link['date'] not in seen:
                seen.add(link['date'])
                unique_links.append(link)
        
        unique_links.sort(key=lambda x: x['date'], reverse=True)
        logger.info(f"Found {len(unique_links)} unique FOMC statement links")
        
        # Download each statement
        for i, link in enumerate(unique_links):
            logger.info(f"  [{i+1}/{len(unique_links)}] Downloading statement from {link['date']}...")
            
            response = self._fetch(link['url'])
            if response:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Extract content
                content = None
                for selector in ['container__main', 'col-md-8', 'col-xs-12', 'content']:
                    content_div = soup.find('div', class_=selector)
                    if content_div:
                        content = content_div.get_text(separator='\n', strip=True)
                        if len(content) > 200:
                            break
                
                if not content or len(content) < 200:
                    content = soup.get_text(separator='\n', strip=True)
                
                statements.append({
                    'date': link['date'],
                    'url': link['url'],
                    'content': content,
                    'word_count': len(content.split()),
                })
        
        # Save to JSON
        self._save_statements(statements)
        return statements
    
    def scrape_historical_fomc_minutes(self, years_back: int = 20) -> List[Dict]:
        """
        Scrape FOMC minutes going back specified years.
        
        Minutes are released ~3 weeks after each FOMC meeting.
        """
        logger.info(f"Scraping FOMC minutes going back {years_back} years...")
        
        minutes = []
        current_year = datetime.now().year
        start_year = max(1996, current_year - years_back)
        
        minutes_links = []
        
        # Current calendar
        calendar_url = f"{self.BASE_URL}/monetarypolicy/fomccalendars.htm"
        response = self._fetch(calendar_url)
        if response:
            soup = BeautifulSoup(response.text, 'html.parser')
            for link in soup.find_all('a', href=True):
                href = link['href']
                text = link.get_text(strip=True).lower()
                if 'minutes' in text or 'fomcminutes' in href:
                    match = re.search(r'(\d{8})', href)
                    if match:
                        full_url = f"{self.BASE_URL}{href}" if href.startswith('/') else href
                        minutes_links.append({
                            'date': match.group(1),
                            'url': full_url
                        })
        
        # Historical archives
        for year in range(current_year, start_year - 1, -1):
            if year < 2017:
                archive_url = f"{self.BASE_URL}/monetarypolicy/fomchistorical{year}.htm"
            else:
                archive_url = f"{self.BASE_URL}/monetarypolicy/fomccalendars.htm"
            
            response = self._fetch(archive_url)
            if response:
                soup = BeautifulSoup(response.text, 'html.parser')
                for link in soup.find_all('a', href=True):
                    href = link['href']
                    text = link.get_text(strip=True).lower()
                    
                    if 'minutes' in text or 'fomcminutes' in href or 'minutes' in href:
                        match = re.search(r'(\d{8})', href)
                        if match:
                            full_url = f"{self.BASE_URL}{href}" if href.startswith('/') else href
                            minutes_links.append({
                                'date': match.group(1),
                                'url': full_url
                            })
            
            logger.info(f"  Checked {year}: found {len(minutes_links)} total links so far")
        
        # Deduplicate
        seen = set()
        unique_links = []
        for link in minutes_links:
            if link['date'] not in seen:
                seen.add(link['date'])
                unique_links.append(link)
        
        unique_links.sort(key=lambda x: x['date'], reverse=True)
        logger.info(f"Found {len(unique_links)} unique FOMC minutes links")
        
        # Download each minutes document
        for i, link in enumerate(unique_links):
            logger.info(f"  [{i+1}/{len(unique_links)}] Downloading minutes from {link['date']}...")
            
            response = self._fetch(link['url'])
            if response:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                content = None
                for selector in ['col-xs-12', 'container__main', 'content']:
                    content_div = soup.find('div', class_=selector)
                    if content_div:
                        content = content_div.get_text(separator='\n', strip=True)
                        if len(content) > 500:
                            break
                
                if not content or len(content) < 500:
                    content = soup.get_text(separator='\n', strip=True)
                
                minutes.append({
                    'date': link['date'],
                    'url': link['url'],
                    'content': content,
                    'word_count': len(content.split()),
                })
        
        self._save_minutes(minutes)
        return minutes
    
    def scrape_historical_beige_book(self, years_back: int = 20) -> List[Dict]:
        """
        Scrape Beige Book editions going back specified years.
        
        Published 8 times per year, available from 1996.
        """
        logger.info(f"Scraping Beige Book going back {years_back} years...")
        
        editions = []
        current_year = datetime.now().year
        start_year = max(1996, current_year - years_back)
        
        beige_links = []
        
        # Current page
        current_url = f"{self.BASE_URL}/monetarypolicy/beige-book-default.htm"
        response = self._fetch(current_url)
        if response:
            soup = BeautifulSoup(response.text, 'html.parser')
            for link in soup.find_all('a', href=True):
                href = link['href']
                if '/monetarypolicy/beigebook' in href and href.endswith('.htm'):
                    match = re.search(r'beigebook(\d{6})', href)
                    if match:
                        full_url = f"{self.BASE_URL}{href}" if href.startswith('/') else href
                        beige_links.append({
                            'date': match.group(1),
                            'url': full_url,
                            'title': link.get_text(strip=True)
                        })
        
        # Yearly archives
        for year in range(current_year, start_year - 1, -1):
            archive_url = f"{self.BASE_URL}/monetarypolicy/beigebook{year}.htm"
            
            response = self._fetch(archive_url)
            if response:
                soup = BeautifulSoup(response.text, 'html.parser')
                for link in soup.find_all('a', href=True):
                    href = link['href']
                    if '/monetarypolicy/beigebook' in href and href.endswith('.htm'):
                        match = re.search(r'beigebook(\d{6})', href)
                        if match:
                            full_url = f"{self.BASE_URL}{href}" if href.startswith('/') else href
                            beige_links.append({
                                'date': match.group(1),
                                'url': full_url,
                                'title': link.get_text(strip=True)
                            })
            
            logger.info(f"  Checked {year}: found {len(beige_links)} total links so far")
        
        # Deduplicate
        seen = set()
        unique_links = []
        for link in beige_links:
            if link['date'] not in seen:
                seen.add(link['date'])
                unique_links.append(link)
        
        unique_links.sort(key=lambda x: x['date'], reverse=True)
        logger.info(f"Found {len(unique_links)} unique Beige Book links")
        
        # Download each edition
        for i, link in enumerate(unique_links):
            logger.info(f"  [{i+1}/{len(unique_links)}] Downloading Beige Book {link['date']}...")
            
            response = self._fetch(link['url'])
            if response:
                soup = BeautifulSoup(response.text, 'html.parser')
                content = soup.get_text(separator='\n', strip=True)
                
                editions.append({
                    'date': link['date'],
                    'url': link['url'],
                    'title': link.get('title', f"Beige Book {link['date']}"),
                    'content': content,
                    'word_count': len(content.split()),
                })
        
        self._save_beige_book(editions)
        return editions
    
    def _save_statements(self, statements: List[Dict]):
        """Save FOMC statements to JSON, merging with existing."""
        filepath = self.data_dir / 'fomc_statements.json'
        existing = {}
        
        if filepath.exists():
            with open(filepath, 'r') as f:
                data = json.load(f)
                for stmt in data.get('statements', []):
                    existing[stmt['date']] = stmt
        
        # Merge new statements
        for stmt in statements:
            existing[stmt['date']] = stmt
        
        merged = sorted(existing.values(), key=lambda x: x['date'], reverse=True)
        
        data = {
            'downloaded_at': datetime.now().isoformat(),
            'source_url': f'{self.BASE_URL}/monetarypolicy/fomccalendars.htm',
            'statement_count': len(merged),
            'statements': merged,
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved {len(merged)} FOMC statements to {filepath}")
    
    def _save_minutes(self, minutes: List[Dict]):
        """Save FOMC minutes to JSON, merging with existing."""
        filepath = self.data_dir / 'fomc_minutes.json'
        existing = {}
        
        if filepath.exists():
            with open(filepath, 'r') as f:
                data = json.load(f)
                for m in data.get('minutes', []):
                    existing[m['date']] = m
        
        for m in minutes:
            existing[m['date']] = m
        
        merged = sorted(existing.values(), key=lambda x: x['date'], reverse=True)
        
        data = {
            'downloaded_at': datetime.now().isoformat(),
            'source_url': f'{self.BASE_URL}/monetarypolicy/fomccalendars.htm',
            'minutes_count': len(merged),
            'minutes': merged,
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved {len(merged)} FOMC minutes to {filepath}")
    
    def _save_beige_book(self, editions: List[Dict]):
        """Save Beige Book editions to JSON, merging with existing."""
        filepath = self.data_dir / 'beige_book.json'
        existing = {}
        
        if filepath.exists():
            with open(filepath, 'r') as f:
                data = json.load(f)
                for e in data.get('editions', []):
                    existing[e['date']] = e
        
        for e in editions:
            existing[e['date']] = e
        
        merged = sorted(existing.values(), key=lambda x: x['date'], reverse=True)
        
        data = {
            'downloaded_at': datetime.now().isoformat(),
            'source_url': f'{self.BASE_URL}/monetarypolicy/beige-book-default.htm',
            'edition_count': len(merged),
            'editions': merged,
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved {len(merged)} Beige Book editions to {filepath}")
    
    def scrape_all(self, years_back: int = 20):
        """Scrape all Fed document types."""
        logger.info(f"=" * 60)
        logger.info(f"Starting historical Fed document scrape ({years_back} years)")
        logger.info(f"=" * 60)
        
        # Scrape each document type
        self.scrape_historical_fomc_statements(years_back)
        self.scrape_historical_fomc_minutes(years_back)
        self.scrape_historical_beige_book(years_back)
        
        logger.info("Historical scrape complete!")


# =============================================================================
# DATABASE MIGRATION
# =============================================================================

def migrate_documents_to_db(db_path: Path = None):
    """
    Migrate all scraped documents to the SQLite database.
    
    This ensures all historical documents are in the documents table
    and ready for feature extraction.
    """
    logger.info("=" * 60)
    logger.info("Migrating documents to database")
    logger.info("=" * 60)
    
    db = PITDatabase(db_path=db_path)
    
    # Force re-migration of documents
    stats = db._migrate_documents(verbose=True)
    
    logger.info(f"Migrated {stats['documents']} documents")
    return stats


# =============================================================================
# FEATURE EXTRACTION
# =============================================================================

def extract_pop_features(
    doc_types: List[str] = None,
    limit_per_type: int = None,
    force: bool = False,
):
    """
    Extract period-over-period features from all documents.
    
    Uses Gemini to analyze each document and compare to its predecessor.
    """
    logger.info("=" * 60)
    logger.info("Extracting PoP features from documents")
    logger.info("=" * 60)
    
    from analysis.hedgeye.document_features import DocumentFeatureExtractor
    
    extractor = DocumentFeatureExtractor()
    
    # Default document types to process
    if doc_types is None:
        doc_types = [
            "FOMC Statement",
            "FOMC Minutes", 
            "Beige Book",
            "Fed Speech",
        ]
    
    results = {}
    for doc_type in doc_types:
        logger.info(f"\nProcessing {doc_type}...")
        result = extractor.extract_for_document_type(
            doc_type,
            limit=limit_per_type,
            skip_existing=not force,
        )
        results[doc_type] = len(result)
    
    logger.info("\n" + "=" * 60)
    logger.info("Feature extraction complete!")
    for doc_type, count in results.items():
        logger.info(f"  {doc_type}: {count} features extracted")
    logger.info("=" * 60)
    
    return results


# =============================================================================
# STATUS CHECK
# =============================================================================

def check_status(db_path: Path = None):
    """Check the current status of documents and features."""
    if db_path is None:
        db_path = PROJECT_ROOT / "storage" / "database" / "pit_macro.db"
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    print("\n" + "=" * 60)
    print("BACKFILL STATUS")
    print("=" * 60)
    
    # Documents by type
    print("\n📄 Documents in Database:")
    cursor.execute("""
        SELECT document_type, source, COUNT(*) as cnt,
               MIN(document_date) as earliest, MAX(document_date) as latest
        FROM documents
        GROUP BY document_type, source
        ORDER BY cnt DESC
    """)
    for row in cursor.fetchall():
        print(f"   {row[0]} ({row[1]}): {row[2]} docs, {row[3]} to {row[4]}")
    
    # Features extracted
    print("\n✨ Features Extracted:")
    cursor.execute("""
        SELECT document_type, COUNT(*) as cnt,
               MIN(document_date) as earliest, MAX(document_date) as latest
        FROM document_features
        GROUP BY document_type
    """)
    for row in cursor.fetchall():
        print(f"   {row[0]}: {row[1]} features, {row[2]} to {row[3]}")
    
    # Gap analysis
    print("\n📊 Gap Analysis:")
    cursor.execute("SELECT COUNT(*) FROM documents")
    total_docs = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM document_features")
    total_features = cursor.fetchone()[0]
    
    print(f"   Total documents: {total_docs}")
    print(f"   Total features: {total_features}")
    print(f"   Gap: {total_docs - total_features} documents need feature extraction")
    
    # By type gap
    print("\n   By Document Type:")
    cursor.execute("""
        SELECT 
            d.document_type,
            COUNT(DISTINCT d.id) as doc_count,
            COUNT(DISTINCT f.id) as feature_count
        FROM documents d
        LEFT JOIN document_features f ON d.id = f.document_id
        GROUP BY d.document_type
    """)
    for row in cursor.fetchall():
        gap = row[1] - row[2]
        status = "✅" if gap == 0 else f"⏳ {gap} pending"
        print(f"      {row[0]}: {row[1]} docs, {row[2]} features - {status}")
    
    conn.close()
    print("=" * 60 + "\n")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Backfill 20 years of PoP features from textual datasets"
    )
    
    parser.add_argument("--full", action="store_true",
                       help="Run full backfill (scrape + migrate + extract)")
    parser.add_argument("--scrape", action="store_true",
                       help="Scrape historical Fed documents")
    parser.add_argument("--migrate", action="store_true",
                       help="Migrate documents to database")
    parser.add_argument("--extract", action="store_true",
                       help="Extract PoP features from documents")
    parser.add_argument("--status", action="store_true",
                       help="Check current backfill status")
    
    parser.add_argument("--years", type=int, default=20,
                       help="Years of history to backfill (default: 20)")
    parser.add_argument("--types", type=str,
                       help="Comma-separated document types to process")
    parser.add_argument("--limit", type=int,
                       help="Limit documents per type (for testing)")
    parser.add_argument("--force", action="store_true",
                       help="Re-extract features for existing documents")
    
    args = parser.parse_args()
    
    # Create logs directory
    (PROJECT_ROOT / 'logs').mkdir(exist_ok=True)
    
    # Parse document types
    doc_types = None
    if args.types:
        doc_types = [t.strip() for t in args.types.split(',')]
    
    # Check status
    if args.status:
        check_status()
        return
    
    # Full backfill
    if args.full:
        # Step 1: Scrape
        data_dir = PROJECT_ROOT / 'storage' / 'raw'
        scraper = HistoricalFedScraper(data_dir)
        scraper.scrape_all(years_back=args.years)
        
        # Step 2: Migrate
        migrate_documents_to_db()
        
        # Step 3: Extract features
        extract_pop_features(
            doc_types=doc_types,
            limit_per_type=args.limit,
            force=args.force,
        )
        
        # Show final status
        check_status()
        return
    
    # Individual steps
    if args.scrape:
        data_dir = PROJECT_ROOT / 'storage' / 'raw'
        scraper = HistoricalFedScraper(data_dir)
        scraper.scrape_all(years_back=args.years)
    
    if args.migrate:
        migrate_documents_to_db()
    
    if args.extract:
        extract_pop_features(
            doc_types=doc_types,
            limit_per_type=args.limit,
            force=args.force,
        )
    
    # If no action specified, show help
    if not any([args.full, args.scrape, args.migrate, args.extract, args.status]):
        parser.print_help()
        print("\nCurrent status:")
        check_status()


if __name__ == "__main__":
    main()

