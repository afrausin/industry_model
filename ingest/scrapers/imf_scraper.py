"""
International Monetary Fund (IMF) Scraper
==========================================
Downloads IMF research and data including:
- Article IV Consultations (country reviews)
- World Economic Outlook
- Global Financial Stability Report

IMF: https://www.imf.org/
"""

import re
from datetime import datetime
from pathlib import Path
from typing import Optional
from bs4 import BeautifulSoup

from .base_scraper import BaseScraper, DownloadTracker


class IMFScraper(BaseScraper):
    """
    Scraper for International Monetary Fund data.
    
    The IMF provides independent assessment of the US economy
    through Article IV consultations and global outlooks.
    """
    
    SOURCE_NAME = 'imf'
    BASE_URL = 'https://www.imf.org'
    RATE_LIMIT_SECONDS = 2.0
    
    ENDPOINTS = {
        'us_article4': '/en/Countries/USA',
        'weo': '/en/Publications/WEO',
        'gfsr': '/en/Publications/GFSR',
        'data': '/en/Data',
    }
    
    def __init__(self, data_dir: Path, tracker: DownloadTracker):
        super().__init__(data_dir, tracker)
    
    def get_available_datasets(self) -> list[dict]:
        """Return list of available IMF datasets."""
        return [
            {
                'name': 'US Article IV Consultation',
                'key': 'us_article4',
                'description': 'IMF annual review of the US economy',
                'frequency': 'annual',
                'type': 'qualitative'
            },
            {
                'name': 'World Economic Outlook',
                'key': 'weo',
                'description': 'Global economic forecasts and analysis',
                'frequency': 'semi-annual',
                'type': 'qualitative'
            },
            {
                'name': 'Global Financial Stability Report',
                'key': 'gfsr',
                'description': 'Assessment of global financial system risks',
                'frequency': 'semi-annual',
                'type': 'qualitative'
            }
        ]
    
    def download_dataset(self, dataset_key: str, force: bool = False) -> Optional[Path]:
        """Download a specific IMF dataset."""
        if dataset_key == 'us_article4':
            return self._download_us_article4(force)
        elif dataset_key == 'weo':
            return self._download_weo(force)
        elif dataset_key == 'gfsr':
            return self._download_gfsr(force)
        
        self.logger.warning(f"Unknown dataset key: {dataset_key}")
        return None
    
    def _download_us_article4(self, force: bool = False) -> Optional[Path]:
        """
        Download US Article IV Consultation reports.
        
        Article IV consultations are the IMF's annual "health check"
        of member countries, providing independent fiscal assessment.
        """
        try:
            url = f"{self.BASE_URL}{self.ENDPOINTS['us_article4']}"
            response = self.fetch_url(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            reports = []
            
            # Find Article IV reports
            for item in soup.find_all(['article', 'div', 'li'], class_=re.compile(r'(publication|report|document|item)')):
                title_elem = item.find(['h2', 'h3', 'h4', 'a'])
                if title_elem:
                    title = title_elem.get_text(strip=True)
                    
                    # Check if it's an Article IV report
                    if 'article iv' in title.lower() or 'staff report' in title.lower():
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
                        
                        reports.append({
                            'title': title,
                            'url': link,
                            'date': date
                        })
            
            # Also search for any links containing "article iv" or "staff report"
            for link in soup.find_all('a', href=True):
                text = link.get_text(strip=True)
                if ('article iv' in text.lower() or 'staff report' in text.lower()) and len(text) > 10:
                    href = link['href']
                    if not href.startswith('http'):
                        href = f"{self.BASE_URL}{href}"
                    
                    if not any(r['title'] == text for r in reports):
                        reports.append({
                            'title': text,
                            'url': href,
                            'date': None
                        })
            
            self.logger.info(f"Found {len(reports)} US Article IV reports")
            
            data = {
                'downloaded_at': datetime.now().isoformat(),
                'source_url': url,
                'country': 'United States',
                'report_count': len(reports),
                'reports': reports[:20]
            }
            
            return self.save_json(
                data=data,
                filename='us_article4.json',
                source_url=url,
                metadata={'report_count': len(reports), 'type': 'qualitative'},
                force=force
            )
            
        except Exception as e:
            self.logger.error(f"Failed to download US Article IV: {e}")
            return None
    
    def _download_weo(self, force: bool = False) -> Optional[Path]:
        """Download World Economic Outlook reports."""
        try:
            url = f"{self.BASE_URL}{self.ENDPOINTS['weo']}"
            response = self.fetch_url(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            reports = []
            
            # Find WEO publications
            for item in soup.find_all(['article', 'div', 'li'], class_=re.compile(r'(publication|report|item)')):
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
                    
                    # Get summary
                    summary_elem = item.find(['p', 'div'], class_=re.compile(r'(summary|description|excerpt)'))
                    summary = summary_elem.get_text(strip=True) if summary_elem else None
                    
                    if title and len(title) > 10:
                        reports.append({
                            'title': title,
                            'url': link,
                            'date': date,
                            'summary': summary[:300] if summary else None
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
                'report_count': len(unique_reports),
                'reports': unique_reports[:20]
            }
            
            return self.save_json(
                data=data,
                filename='weo.json',
                source_url=url,
                metadata={'report_count': len(unique_reports), 'type': 'qualitative'},
                force=force
            )
            
        except Exception as e:
            self.logger.error(f"Failed to download WEO: {e}")
            return None
    
    def _download_gfsr(self, force: bool = False) -> Optional[Path]:
        """Download Global Financial Stability Report."""
        try:
            url = f"{self.BASE_URL}{self.ENDPOINTS['gfsr']}"
            response = self.fetch_url(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            reports = []
            
            # Find GFSR publications
            for item in soup.find_all(['article', 'div', 'li'], class_=re.compile(r'(publication|report|item)')):
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
                'report_count': len(unique_reports),
                'reports': unique_reports[:20]
            }
            
            return self.save_json(
                data=data,
                filename='gfsr.json',
                source_url=url,
                metadata={'report_count': len(unique_reports), 'type': 'qualitative'},
                force=force
            )
            
        except Exception as e:
            self.logger.error(f"Failed to download GFSR: {e}")
            return None

