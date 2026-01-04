"""
Organisation for Economic Co-operation and Development (OECD) Scraper
======================================================================
Downloads OECD economic data and research including:
- Economic Outlook
- Country surveys (US)
- Leading indicators

OECD: https://www.oecd.org/
"""

import re
from datetime import datetime
from pathlib import Path
from typing import Optional
from bs4 import BeautifulSoup

from .base_scraper import BaseScraper, DownloadTracker


class OECDScraper(BaseScraper):
    """
    Scraper for OECD economic data.
    
    OECD provides comparative economic analysis across
    developed economies, useful for benchmarking US performance.
    """
    
    SOURCE_NAME = 'oecd'
    BASE_URL = 'https://www.oecd.org'
    RATE_LIMIT_SECONDS = 2.0
    
    ENDPOINTS = {
        'economic_outlook': '/economic-outlook/',
        'us_survey': '/united-states/',
        'cli': '/leading-indicators/',
        'data': '/en/data/',
    }
    
    def __init__(self, data_dir: Path, tracker: DownloadTracker):
        super().__init__(data_dir, tracker)
    
    def get_available_datasets(self) -> list[dict]:
        """Return list of available OECD datasets."""
        return [
            {
                'name': 'Economic Outlook',
                'key': 'economic_outlook',
                'description': 'Semi-annual global economic projections',
                'frequency': 'semi-annual',
                'type': 'qualitative'
            },
            {
                'name': 'US Economic Survey',
                'key': 'us_survey',
                'description': 'OECD assessment of the US economy',
                'frequency': 'biennial',
                'type': 'qualitative'
            },
            {
                'name': 'Composite Leading Indicators',
                'key': 'cli',
                'description': 'Leading economic indicators for major economies',
                'frequency': 'monthly',
                'type': 'quantitative'
            }
        ]
    
    def download_dataset(self, dataset_key: str, force: bool = False) -> Optional[Path]:
        """Download a specific OECD dataset."""
        if dataset_key == 'economic_outlook':
            return self._download_economic_outlook(force)
        elif dataset_key == 'us_survey':
            return self._download_us_survey(force)
        elif dataset_key == 'cli':
            return self._download_cli(force)
        
        self.logger.warning(f"Unknown dataset key: {dataset_key}")
        return None
    
    def _download_economic_outlook(self, force: bool = False) -> Optional[Path]:
        """
        Download OECD Economic Outlook.
        
        The Economic Outlook provides semi-annual projections
        for GDP, inflation, and unemployment across OECD countries.
        """
        try:
            url = f"{self.BASE_URL}{self.ENDPOINTS['economic_outlook']}"
            response = self.fetch_url(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract content and key data
            content = soup.get_text(separator='\n', strip=True)
            
            # Find reports and publications
            reports = []
            for item in soup.find_all(['article', 'div', 'li'], class_=re.compile(r'(publication|report|item|card)')):
                title_elem = item.find(['h2', 'h3', 'h4', 'a'])
                if title_elem:
                    title = title_elem.get_text(strip=True)
                    
                    link = None
                    if title_elem.name == 'a':
                        link = title_elem.get('href')
                    else:
                        link_elem = item.find('a', href=True)
                        if link_elem:
                            link = link_elem['href']
                    
                    if link and not link.startswith('http'):
                        link = f"{self.BASE_URL}{link}"
                    
                    # Get date
                    date_elem = item.find(class_=re.compile(r'date'))
                    date = date_elem.get_text(strip=True) if date_elem else None
                    
                    if title and len(title) > 10:
                        reports.append({
                            'title': title,
                            'url': link,
                            'date': date
                        })
            
            # Look for key projections in the content
            projections = []
            # Try to find GDP growth projections
            gdp_patterns = [
                r'GDP.*?(\d+\.?\d*)%',
                r'growth.*?(\d+\.?\d*)%',
            ]
            for pattern in gdp_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                if matches:
                    projections.extend(matches[:5])
            
            # Deduplicate reports
            seen = set()
            unique_reports = []
            for r in reports:
                if r['title'] not in seen:
                    seen.add(r['title'])
                    unique_reports.append(r)
            
            data = {
                'downloaded_at': datetime.now().isoformat(),
                'source_url': url,
                'report_count': len(unique_reports),
                'reports': unique_reports[:20],
                'content_summary': content[:5000],
                'word_count': len(content.split())
            }
            
            return self.save_json(
                data=data,
                filename='economic_outlook.json',
                source_url=url,
                metadata={'report_count': len(unique_reports), 'type': 'qualitative'},
                force=force
            )
            
        except Exception as e:
            self.logger.error(f"Failed to download Economic Outlook: {e}")
            return None
    
    def _download_us_survey(self, force: bool = False) -> Optional[Path]:
        """Download OECD US Economic Survey."""
        try:
            url = f"{self.BASE_URL}{self.ENDPOINTS['us_survey']}"
            response = self.fetch_url(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            content = soup.get_text(separator='\n', strip=True)
            
            # Find reports about the US
            reports = []
            for item in soup.find_all(['article', 'div', 'li'], class_=re.compile(r'(publication|report|item|card)')):
                title_elem = item.find(['h2', 'h3', 'h4', 'a'])
                if title_elem:
                    title = title_elem.get_text(strip=True)
                    
                    link = None
                    if title_elem.name == 'a':
                        link = title_elem.get('href')
                    else:
                        link_elem = item.find('a', href=True)
                        if link_elem:
                            link = link_elem['href']
                    
                    if link and not link.startswith('http'):
                        link = f"{self.BASE_URL}{link}"
                    
                    # Get date
                    date_elem = item.find(class_=re.compile(r'date'))
                    date = date_elem.get_text(strip=True) if date_elem else None
                    
                    if title and len(title) > 10:
                        reports.append({
                            'title': title,
                            'url': link,
                            'date': date
                        })
            
            # Deduplicate
            seen = set()
            unique_reports = []
            for r in reports:
                if r['title'] not in seen:
                    seen.add(r['title'])
                    unique_reports.append(r)
            
            data = {
                'downloaded_at': datetime.now().isoformat(),
                'source_url': url,
                'country': 'United States',
                'report_count': len(unique_reports),
                'reports': unique_reports[:20],
                'content_summary': content[:5000],
                'word_count': len(content.split())
            }
            
            return self.save_json(
                data=data,
                filename='us_survey.json',
                source_url=url,
                metadata={'report_count': len(unique_reports), 'type': 'qualitative'},
                force=force
            )
            
        except Exception as e:
            self.logger.error(f"Failed to download US survey: {e}")
            return None
    
    def _download_cli(self, force: bool = False) -> Optional[Path]:
        """
        Download Composite Leading Indicators.
        
        The CLI is designed to provide early signals of 
        turning points in business cycles.
        """
        try:
            url = f"{self.BASE_URL}{self.ENDPOINTS['cli']}"
            response = self.fetch_url(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            content = soup.get_text(separator='\n', strip=True)
            
            # Find any data tables
            tables = []
            for table in soup.find_all('table'):
                rows = []
                for tr in table.find_all('tr'):
                    cells = [td.get_text(strip=True) for td in tr.find_all(['td', 'th'])]
                    if cells:
                        rows.append(cells)
                if rows:
                    tables.append(rows)
            
            # Find links to data/publications
            links = []
            for link in soup.find_all('a', href=True):
                text = link.get_text(strip=True)
                href = link['href']
                if ('indicator' in text.lower() or 'data' in text.lower() or 'cli' in text.lower()) and len(text) > 5:
                    if not href.startswith('http'):
                        href = f"{self.BASE_URL}{href}"
                    links.append({
                        'text': text,
                        'url': href
                    })
            
            # Deduplicate
            seen = set()
            unique_links = []
            for l in links:
                if l['text'] not in seen:
                    seen.add(l['text'])
                    unique_links.append(l)
            
            data = {
                'downloaded_at': datetime.now().isoformat(),
                'source_url': url,
                'title': 'Composite Leading Indicators',
                'description': 'Early signals of turning points in business cycles',
                'tables': tables,
                'related_links': unique_links[:20],
                'content_summary': content[:5000],
                'word_count': len(content.split())
            }
            
            return self.save_json(
                data=data,
                filename='cli.json',
                source_url=url,
                metadata={'type': 'quantitative'},
                force=force
            )
            
        except Exception as e:
            self.logger.error(f"Failed to download CLI: {e}")
            return None

