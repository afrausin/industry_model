"""
Federal Reserve Board of Governors Data Scraper
================================================
Downloads data from the Fed including:
- FOMC Statements and Minutes
- Summary of Economic Projections (SEP)
- Beige Book
- H.8 (Assets and Liabilities of Commercial Banks)
- H.15 (Selected Interest Rates)

Fed Data: https://www.federalreserve.gov/data.htm
"""

import re
from datetime import datetime
from pathlib import Path
from typing import Optional
from bs4 import BeautifulSoup

from .base_scraper import BaseScraper, DownloadTracker


class FederalReserveScraper(BaseScraper):
    """
    Scraper for Federal Reserve Board of Governors data.
    
    Sources:
    - FOMC meeting materials
    - Beige Book reports
    - Statistical releases (H.8, H.15, etc.)
    """
    
    SOURCE_NAME = 'federal_reserve'
    BASE_URL = 'https://www.federalreserve.gov'
    RATE_LIMIT_SECONDS = 2.0  # Be respectful to fed.gov
    
    # Key data endpoints
    ENDPOINTS = {
        'fomc_calendar': '/monetarypolicy/fomccalendars.htm',
        'fomc_statements': '/monetarypolicy/fomc.htm',
        'fomc_minutes': '/monetarypolicy/fomccalendars.htm',  # Minutes links from calendar
        'beige_book': '/monetarypolicy/beige-book-default.htm',
        'h8_release': '/releases/h8/current/default.htm',
        'h15_release': '/releases/h15/current/default.htm',
        'sep': '/monetarypolicy/fomcprojtabl.htm',
        'speeches': '/newsevents/speeches.htm',
    }
    
    def __init__(self, data_dir: Path, tracker: DownloadTracker):
        super().__init__(data_dir, tracker)
    
    def get_available_datasets(self) -> list[dict]:
        """Return list of available Fed datasets."""
        return [
            # Qualitative / Policy Documents
            {
                'name': 'FOMC Statements',
                'key': 'fomc_statements',
                'description': 'FOMC policy statements from recent meetings',
                'frequency': 'event-driven',
                'type': 'qualitative'
            },
            {
                'name': 'FOMC Minutes',
                'key': 'fomc_minutes',
                'description': 'Detailed FOMC meeting minutes with policy deliberations',
                'frequency': '8x per year (3 weeks after meeting)',
                'type': 'qualitative'
            },
            {
                'name': 'Fed Speeches',
                'key': 'fed_speeches',
                'description': 'Speeches by Fed officials (Powell, Governors, Presidents)',
                'frequency': 'ongoing',
                'type': 'qualitative'
            },
            {
                'name': 'Beige Book',
                'key': 'beige_book',
                'description': 'Summary of Commentary on Current Economic Conditions',
                'frequency': '8x per year',
                'type': 'qualitative'
            },
            # Economic Projections
            {
                'name': 'Summary of Economic Projections (SEP)',
                'key': 'sep',
                'description': 'FOMC participants economic projections (dot plot)',
                'frequency': '4x per year'
            },
            # Statistical Releases
            {
                'name': 'H.8 - Bank Assets and Liabilities',
                'key': 'h8_release',
                'description': 'Assets and Liabilities of Commercial Banks',
                'frequency': 'weekly'
            },
            {
                'name': 'H.15 - Selected Interest Rates',
                'key': 'h15_release',
                'description': 'Selected Interest Rates (Treasury yields, etc.)',
                'frequency': 'daily/weekly'
            },
            {
                'name': 'FOMC Calendar',
                'key': 'fomc_calendar',
                'description': 'Schedule of upcoming FOMC meetings',
                'frequency': 'annual'
            }
        ]
    
    def download_dataset(self, dataset_key: str, force: bool = False) -> Optional[Path]:
        """Download a specific Fed dataset."""
        
        if dataset_key == 'fomc_statements':
            return self._download_fomc_statements(force)
        elif dataset_key == 'fomc_minutes':
            return self._download_fomc_minutes(force)
        elif dataset_key == 'fed_speeches':
            return self._download_fed_speeches(force)
        elif dataset_key == 'beige_book':
            return self._download_beige_book(force)
        elif dataset_key == 'sep':
            return self._download_sep(force)
        elif dataset_key == 'h8_release':
            return self._download_statistical_release('h8', force)
        elif dataset_key == 'h15_release':
            return self._download_statistical_release('h15', force)
        elif dataset_key == 'fomc_calendar':
            return self._download_fomc_calendar(force)
        
        self.logger.warning(f"Unknown dataset key: {dataset_key}")
        return None
    
    def _download_fomc_statements(self, force: bool = False) -> Optional[Path]:
        """
        Download FOMC statements incrementally.
        
        Only downloads new statements not already in the local file.
        Uses the FOMC calendar page which contains links to all historical
        statements, including policy decisions and implementation notes.
        """
        try:
            # Get existing statement dates to avoid re-downloading
            existing_dates = self.get_existing_keys('fomc_statements.json', 'statements', 'date')
            if existing_dates and not force:
                self.logger.info(f"Have {len(existing_dates)} existing statements, checking for new...")
            
            # Use the calendar page which has all statement links
            url = f"{self.BASE_URL}{self.ENDPOINTS['fomc_calendar']}"
            response = self.fetch_url(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find links to statements
            statements = []
            for link in soup.find_all('a', href=True):
                href = link['href']
                text = link.get_text(strip=True)
                if '/newsevents/pressreleases/monetary' in href:
                    match = re.search(r'monetary(\d{8})a\.htm$', href)
                    if match:
                        date_str = match.group(1)
                        full_url = f"{self.BASE_URL}{href}" if href.startswith('/') else href
                        statements.append({
                            'date': date_str,
                            'url': full_url,
                            'link_text': text
                        })
            
            # Deduplicate and sort by date (newest first)
            seen = set()
            unique_statements = []
            for s in statements:
                if s['date'] not in seen:
                    seen.add(s['date'])
                    unique_statements.append(s)
            
            unique_statements.sort(key=lambda x: x['date'], reverse=True)
            
            # Filter to only new statements (unless force=True)
            if not force:
                to_download = [s for s in unique_statements[:80] if s['date'] not in existing_dates]
            else:
                to_download = unique_statements[:80]
            
            if not to_download:
                self.logger.info("No new FOMC statements to download")
                return self.data_dir / 'fomc_statements.json'
            
            self.logger.info(f"Downloading {len(to_download)} new statements...")
            
            # Download only new statements
            new_statements = []
            for stmt in to_download:
                try:
                    stmt_response = self.fetch_url(stmt['url'])
                    stmt_soup = BeautifulSoup(stmt_response.text, 'html.parser')
                    
                    content = None
                    for selector in ['container__main', 'col-md-8', 'col-xs-12']:
                        content_div = stmt_soup.find('div', class_=selector)
                        if content_div:
                            content = content_div.get_text(separator='\n', strip=True)
                            if len(content) > 200:
                                break
                    
                    if not content or len(content) < 200:
                        content = stmt_soup.get_text(separator='\n', strip=True)
                    
                    stmt['content'] = content
                    stmt['word_count'] = len(content.split())
                    new_statements.append(stmt)
                    self.logger.info(f"Downloaded statement from {stmt['date']}")
                    
                except Exception as e:
                    self.logger.warning(f"Failed to fetch statement {stmt['date']}: {e}")
            
            # Merge with existing data
            existing_data = self.load_existing_json('fomc_statements.json')
            merged_data = self.merge_items(
                existing_data, new_statements, 'statements', 'date', sort_field='date'
            )
            
            merged_data['downloaded_at'] = datetime.now().isoformat()
            merged_data['source_url'] = url
            merged_data['statement_count'] = len(merged_data['statements'])
            
            return self.save_json(
                data=merged_data,
                filename='fomc_statements.json',
                source_url=url,
                metadata={'statement_count': merged_data['statement_count'], 'type': 'qualitative'},
                force=True  # Always save since we've merged
            )
        except Exception as e:
            self.logger.error(f"Failed to download FOMC statements: {e}")
            return None
    
    def _download_beige_book(self, force: bool = False) -> Optional[Path]:
        """
        Download Beige Book editions incrementally.
        
        Only downloads new editions not already in the local file.
        The Beige Book is published 8 times per year.
        """
        try:
            # Get existing edition dates
            existing_dates = self.get_existing_keys('beige_book.json', 'editions', 'date')
            if existing_dates and not force:
                self.logger.info(f"Have {len(existing_dates)} existing editions, checking for new...")
            
            editions = []
            current_year = datetime.now().year
            
            # First get editions from the current default page
            try:
                url = f"{self.BASE_URL}{self.ENDPOINTS['beige_book']}"
                response = self.fetch_url(url)
                soup = BeautifulSoup(response.text, 'html.parser')
                
                for link in soup.find_all('a', href=True):
                    href = link['href']
                    if '/monetarypolicy/beigebook' in href and href.endswith('.htm'):
                        match = re.search(r'beigebook(\d{6})', href)
                        if match:
                            full_url = f"{self.BASE_URL}{href}" if href.startswith('/') else href
                            editions.append({
                                'date': match.group(1),
                                'url': full_url,
                                'title': link.get_text(strip=True)
                            })
            except Exception as e:
                self.logger.warning(f"Failed to fetch default page: {e}")
            
            # Then get editions from yearly archive pages (last 10 years)
            for year in range(current_year, current_year - 11, -1):
                year_url = f"{self.BASE_URL}/monetarypolicy/beigebook{year}.htm"
                try:
                    response = self.fetch_url(year_url)
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    for link in soup.find_all('a', href=True):
                        href = link['href']
                        if '/monetarypolicy/beigebook' in href and href.endswith('.htm'):
                            match = re.search(r'beigebook(\d{6})', href)
                            if match:
                                full_url = f"{self.BASE_URL}{href}" if href.startswith('/') else href
                                editions.append({
                                    'date': match.group(1),
                                    'url': full_url,
                                    'title': link.get_text(strip=True)
                                })
                    
                except Exception as e:
                    self.logger.warning(f"Failed to fetch {year} page: {e}")
            
            # Deduplicate and sort (newest first)
            seen = set()
            unique_editions = []
            for e in editions:
                if e['date'] not in seen:
                    seen.add(e['date'])
                    unique_editions.append(e)
            unique_editions.sort(key=lambda x: x['date'], reverse=True)
            
            # Filter to only new editions
            if not force:
                to_download = [e for e in unique_editions[:80] if e['date'] not in existing_dates]
            else:
                to_download = unique_editions[:80]
            
            if not to_download:
                self.logger.info("No new Beige Book editions to download")
                return self.data_dir / 'beige_book.json'
            
            self.logger.info(f"Downloading {len(to_download)} new editions...")
            
            # Download only new editions
            new_editions = []
            for edition in to_download:
                try:
                    bb_response = self.fetch_url(edition['url'])
                    bb_soup = BeautifulSoup(bb_response.text, 'html.parser')
                    
                    content = bb_soup.get_text(separator='\n', strip=True)
                    
                    edition['content'] = content
                    edition['word_count'] = len(content.split())
                    new_editions.append(edition)
                    
                    self.logger.info(f"Downloaded Beige Book {edition['date']}")
                    
                except Exception as e:
                    self.logger.warning(f"Failed to download Beige Book {edition['date']}: {e}")
            
            # Merge with existing data
            existing_data = self.load_existing_json('beige_book.json')
            merged_data = self.merge_items(
                existing_data, new_editions, 'editions', 'date', sort_field='date'
            )
            
            source_url = f"{self.BASE_URL}{self.ENDPOINTS['beige_book']}"
            merged_data['downloaded_at'] = datetime.now().isoformat()
            merged_data['source_url'] = source_url
            merged_data['edition_count'] = len(merged_data['editions'])
            
            return self.save_json(
                data=merged_data,
                filename='beige_book.json',
                source_url=source_url,
                metadata={'edition_count': merged_data['edition_count'], 'type': 'qualitative'},
                force=True
            )
            
        except Exception as e:
            self.logger.error(f"Failed to download Beige Book: {e}")
            return None
    
    def _download_sep(self, force: bool = False) -> Optional[Path]:
        """
        Download Summary of Economic Projections data.
        
        SEP is released quarterly at 4 of the 8 FOMC meetings (March, June, Sept, Dec).
        URLs have the format: fomcprojtabl20240918.htm
        """
        try:
            # Get SEP links from the FOMC calendar page
            url = f"{self.BASE_URL}{self.ENDPOINTS['fomc_calendar']}"
            response = self.fetch_url(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find links to SEP materials (fomcprojtabl*.htm)
            sep_links = []
            for link in soup.find_all('a', href=True):
                href = link['href']
                text = link.get_text(strip=True)
                if 'fomcprojtabl' in href and href.endswith('.htm'):
                    match = re.search(r'fomcprojtabl(\d{8})', href)
                    if match:
                        full_url = f"{self.BASE_URL}{href}" if href.startswith('/') else href
                        sep_links.append({
                            'date': match.group(1),
                            'url': full_url,
                            'text': text
                        })
            
            # Deduplicate and sort
            seen = set()
            unique_seps = []
            for s in sep_links:
                if s['date'] not in seen:
                    seen.add(s['date'])
                    unique_seps.append(s)
            unique_seps.sort(key=lambda x: x['date'], reverse=True)
            
            self.logger.info(f"Found {len(unique_seps)} SEP releases")
            
            # Download SEP content for each release
            downloaded_seps = []
            for sep in unique_seps[:20]:  # Last ~5 years (4 per year)
                try:
                    sep_response = self.fetch_url(sep['url'])
                    sep_soup = BeautifulSoup(sep_response.text, 'html.parser')
                    
                    content = sep_soup.get_text(separator='\n', strip=True)
                    
                    sep['content'] = content[:20000]
                    sep['word_count'] = len(content.split())
                    downloaded_seps.append(sep)
                    
                    self.logger.info(f"Downloaded SEP from {sep['date']}")
                    
                except Exception as e:
                    self.logger.warning(f"Failed to fetch SEP {sep['date']}: {e}")
            
            data = {
                'downloaded_at': datetime.now().isoformat(),
                'source_url': url,
                'sep_count': len(downloaded_seps),
                'projections': downloaded_seps
            }
            
            return self.save_json(
                data=data,
                filename='sep_projections.json',
                source_url=url,
                metadata={'sep_count': len(downloaded_seps), 'type': 'qualitative'},
                force=force
            )
            
        except Exception as e:
            self.logger.error(f"Failed to download SEP: {e}")
            return None
    
    def _download_statistical_release(self, release: str, force: bool = False) -> Optional[Path]:
        """Download a Fed statistical release (H.8, H.15, etc.)."""
        try:
            url = f"{self.BASE_URL}/releases/{release}/current/default.htm"
            response = self.fetch_url(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Look for data download links
            data_links = []
            for link in soup.find_all('a', href=True):
                href = link['href']
                if href.endswith(('.csv', '.xml', '.txt')):
                    full_url = f"{self.BASE_URL}{href}" if href.startswith('/') else href
                    data_links.append({
                        'url': full_url,
                        'text': link.get_text(strip=True)
                    })
            
            # Download CSV files if available
            downloaded_files = []
            for dl in data_links:
                if dl['url'].endswith('.csv'):
                    try:
                        file_response = self.fetch_url(dl['url'])
                        filename = f"{release}_{dl['url'].split('/')[-1]}"
                        file_path = self.data_dir / filename
                        with open(file_path, 'wb') as f:
                            f.write(file_response.content)
                        downloaded_files.append(filename)
                    except Exception as e:
                        self.logger.warning(f"Failed to download {dl['url']}: {e}")
            
            data = {
                'downloaded_at': datetime.now().isoformat(),
                'release': release.upper(),
                'source_url': url,
                'data_links': data_links,
                'downloaded_files': downloaded_files,
                'page_title': soup.title.string if soup.title else None
            }
            
            return self.save_json(
                data=data,
                filename=f"{release}_release.json",
                source_url=url,
                metadata={'release': release, 'files': downloaded_files},
                force=force
            )
            
        except Exception as e:
            self.logger.error(f"Failed to download {release} release: {e}")
            return None
    
    def _download_fomc_calendar(self, force: bool = False) -> Optional[Path]:
        """Download FOMC meeting calendar."""
        try:
            url = f"{self.BASE_URL}{self.ENDPOINTS['fomc_calendar']}"
            response = self.fetch_url(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Parse meeting dates from the calendar page
            meetings = []
            
            # Look for calendar tables
            for table in soup.find_all('table'):
                for row in table.find_all('tr'):
                    cells = row.find_all(['td', 'th'])
                    if len(cells) >= 2:
                        date_text = cells[0].get_text(strip=True)
                        if re.search(r'\d{4}', date_text) or re.search(r'(January|February|March|April|May|June|July|August|September|October|November|December)', date_text):
                            meetings.append({
                                'date': date_text,
                                'details': [c.get_text(strip=True) for c in cells[1:]]
                            })
            
            data = {
                'downloaded_at': datetime.now().isoformat(),
                'source_url': url,
                'meetings': meetings,
                'raw_content': soup.get_text(separator='\n', strip=True)[:5000]
            }
            
            return self.save_json(
                data=data,
                filename='fomc_calendar.json',
                source_url=url,
                metadata={'meeting_count': len(meetings)},
                force=force
            )
            
        except Exception as e:
            self.logger.error(f"Failed to download FOMC calendar: {e}")
            return None
    
    # =========================================================================
    # QUALITATIVE DATA: FOMC Minutes and Fed Speeches
    # =========================================================================
    
    def _download_fomc_minutes(self, force: bool = False) -> Optional[Path]:
        """
        Download FOMC meeting minutes incrementally.
        
        Only downloads new minutes not already in the local file.
        Minutes are released 3 weeks after each FOMC meeting.
        """
        try:
            # Get existing minutes dates
            existing_dates = self.get_existing_keys('fomc_minutes.json', 'minutes', 'date')
            if existing_dates and not force:
                self.logger.info(f"Have {len(existing_dates)} existing minutes, checking for new...")
            
            url = f"{self.BASE_URL}{self.ENDPOINTS['fomc_calendar']}"
            response = self.fetch_url(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find links to minutes
            minutes_links = []
            for link in soup.find_all('a', href=True):
                href = link['href']
                text = link.get_text(strip=True).lower()
                if 'minutes' in text or 'fomcminutes' in href:
                    match = re.search(r'(\d{8})', href)
                    if match:
                        full_url = f"{self.BASE_URL}{href}" if href.startswith('/') else href
                        minutes_links.append({
                            'date': match.group(1),
                            'url': full_url,
                            'text': link.get_text(strip=True)
                        })
            
            # Deduplicate and sort
            seen = set()
            unique_minutes = []
            for m in minutes_links:
                if m['date'] not in seen:
                    seen.add(m['date'])
                    unique_minutes.append(m)
            unique_minutes.sort(key=lambda x: x['date'], reverse=True)
            
            # Filter to only new minutes
            if not force:
                to_download = [m for m in unique_minutes[:80] if m['date'] not in existing_dates]
            else:
                to_download = unique_minutes[:80]
            
            if not to_download:
                self.logger.info("No new FOMC minutes to download")
                return self.data_dir / 'fomc_minutes.json'
            
            self.logger.info(f"Downloading {len(to_download)} new minutes...")
            
            # Download only new minutes
            new_minutes = []
            for minutes in to_download:
                try:
                    min_response = self.fetch_url(minutes['url'])
                    min_soup = BeautifulSoup(min_response.text, 'html.parser')
                    
                    content_div = min_soup.find('div', class_='col-xs-12')
                    if content_div:
                        content = content_div.get_text(separator='\n', strip=True)
                    else:
                        content = min_soup.get_text(separator='\n', strip=True)
                    
                    # Extract key sections
                    sections = {}
                    section_headers = [
                        'Staff Review of the Economic Situation',
                        'Staff Review of the Financial Situation',
                        'Staff Economic Outlook',
                        'Participants\' Views',
                        'Committee Policy Action'
                    ]
                    
                    for header in section_headers:
                        pattern = rf'{re.escape(header)}(.*?)(?={"|".join(re.escape(h) for h in section_headers)}|$)'
                        match = re.search(pattern, content, re.IGNORECASE | re.DOTALL)
                        if match:
                            sections[header] = match.group(1).strip()[:3000]
                    
                    minutes['content'] = content
                    minutes['sections'] = sections
                    minutes['word_count'] = len(content.split())
                    new_minutes.append(minutes)
                    
                    self.logger.info(f"Downloaded minutes from {minutes['date']}")
                    
                except Exception as e:
                    self.logger.warning(f"Failed to fetch minutes {minutes['date']}: {e}")
            
            # Merge with existing data
            existing_data = self.load_existing_json('fomc_minutes.json')
            merged_data = self.merge_items(
                existing_data, new_minutes, 'minutes', 'date', sort_field='date'
            )
            
            merged_data['downloaded_at'] = datetime.now().isoformat()
            merged_data['source_url'] = url
            merged_data['minutes_count'] = len(merged_data['minutes'])
            
            return self.save_json(
                data=merged_data,
                filename='fomc_minutes.json',
                source_url=url,
                metadata={'minutes_count': merged_data['minutes_count'], 'type': 'qualitative'},
                force=True
            )
            
        except Exception as e:
            self.logger.error(f"Failed to download FOMC minutes: {e}")
            return None
    
    def _download_fed_speeches(self, force: bool = False) -> Optional[Path]:
        """
        Download Fed speeches incrementally.
        
        Only downloads new speeches not already in the local file.
        Fed speeches are crucial for "Fedspeak" analysis.
        """
        try:
            # Get existing speech URLs to avoid re-downloading
            existing_urls = self.get_existing_keys('fed_speeches.json', 'speeches', 'url')
            if existing_urls and not force:
                self.logger.info(f"Have {len(existing_urls)} existing speeches, checking for new...")
            
            base_url = f"{self.BASE_URL}/newsevents/speech"
            current_year = datetime.now().year
            
            # Collect speeches from last 10 years
            all_speeches = []
            for year in range(current_year, current_year - 10, -1):
                year_url = f"{base_url}/{year}-speeches.htm"
                try:
                    response = self.fetch_url(year_url)
                    soup = BeautifulSoup(response.text, 'html.parser')
                    
                    for link in soup.find_all('a', href=True):
                        href = link['href']
                        title = link.get_text(strip=True)
                        
                        if '/newsevents/speech/' in href:
                            match = re.search(r'/newsevents/speech/([a-z]+)(\d{8})([a-z]?)\.htm$', href)
                            if match:
                                speaker_name = match.group(1)
                                date_str = match.group(2)
                                full_url = f"{self.BASE_URL}{href}" if href.startswith('/') else href
                                all_speeches.append({
                                    'date': date_str,
                                    'speaker_id': speaker_name,
                                    'url': full_url,
                                    'title': title if len(title) > 5 else f"Speech by {speaker_name}"
                                })
                    
                except Exception as e:
                    self.logger.warning(f"Failed to fetch {year} speeches: {e}")
            
            # Deduplicate by URL
            seen = set()
            unique_speeches = []
            for s in all_speeches:
                if s['url'] not in seen:
                    seen.add(s['url'])
                    unique_speeches.append(s)
            
            unique_speeches.sort(key=lambda x: x['date'], reverse=True)
            
            # Filter to only new speeches
            if not force:
                to_download = [s for s in unique_speeches[:200] if s['url'] not in existing_urls]
            else:
                to_download = unique_speeches[:200]
            
            if not to_download:
                self.logger.info("No new Fed speeches to download")
                return self.data_dir / 'fed_speeches.json'
            
            self.logger.info(f"Downloading {len(to_download)} new speeches...")
            
            # Download only new speeches
            new_speeches = []
            for speech in to_download:
                try:
                    speech_response = self.fetch_url(speech['url'])
                    speech_soup = BeautifulSoup(speech_response.text, 'html.parser')
                    
                    speaker = None
                    speaker_elem = speech_soup.find('p', class_='speaker')
                    if speaker_elem:
                        speaker = speaker_elem.get_text(strip=True)
                    
                    if not speaker:
                        byline = speech_soup.find('p', class_='article__byline')
                        if byline:
                            speaker = byline.get_text(strip=True)
                    
                    content = None
                    for selector in ['container__main', 'col-md-8', 'col-xs-12']:
                        content_div = speech_soup.find('div', class_=selector)
                        if content_div:
                            content = content_div.get_text(separator='\n', strip=True)
                            if len(content) > 500:
                                break
                    
                    if not content or len(content) < 500:
                        content = speech_soup.get_text(separator='\n', strip=True)
                    
                    speech['speaker'] = speaker or speech['speaker_id'].title()
                    speech['content'] = content[:15000]
                    speech['word_count'] = len(content.split())
                    new_speeches.append(speech)
                    
                    self.logger.info(f"Downloaded speech: {speech['title'][:50]}...")
                    
                except Exception as e:
                    self.logger.warning(f"Failed to fetch speech {speech['date']}: {e}")
            
            # Merge with existing data
            existing_data = self.load_existing_json('fed_speeches.json')
            merged_data = self.merge_items(
                existing_data, new_speeches, 'speeches', 'url', sort_field='date'
            )
            
            # Rebuild by_speaker index
            by_speaker = {}
            for s in merged_data['speeches']:
                speaker = s.get('speaker', s.get('speaker_id', 'Unknown'))
                if speaker not in by_speaker:
                    by_speaker[speaker] = []
                by_speaker[speaker].append({
                    'date': s['date'],
                    'title': s['title'],
                    'url': s['url']
                })
            
            merged_data['downloaded_at'] = datetime.now().isoformat()
            merged_data['source_url'] = f"{base_url}/{current_year}-speeches.htm"
            merged_data['speech_count'] = len(merged_data['speeches'])
            merged_data['by_speaker'] = by_speaker
            
            return self.save_json(
                data=merged_data,
                filename='fed_speeches.json',
                source_url=base_url,
                metadata={'speech_count': merged_data['speech_count'], 'type': 'qualitative'},
                force=True
            )
            
        except Exception as e:
            self.logger.error(f"Failed to download Fed speeches: {e}")
            return None

