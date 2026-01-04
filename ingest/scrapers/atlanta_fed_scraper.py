"""
Federal Reserve Bank of Atlanta GDPNow Scraper
==============================================
Downloads data from the Atlanta Fed including:
- GDPNow real-time GDP growth estimates
- GDPNow tracking data and history

GDPNow: https://www.atlantafed.org/cqer/research/gdpnow
"""

import re
from datetime import datetime
from pathlib import Path
from typing import Optional
from bs4 import BeautifulSoup

from .base_scraper import BaseScraper, DownloadTracker


class AtlantaFedScraper(BaseScraper):
    """
    Scraper for Federal Reserve Bank of Atlanta data.
    
    Primary focus: GDPNow nowcasting model
    - Real-time GDP growth estimates
    - Component-level tracking
    - Historical forecast evolution
    """
    
    SOURCE_NAME = 'atlanta_fed'
    BASE_URL = 'https://www.atlantafed.org'
    RATE_LIMIT_SECONDS = 2.0
    
    ENDPOINTS = {
        'gdpnow': '/cqer/research/gdpnow',
        'gdpnow_data': '/cqer/research/gdpnow/archives',
    }
    
    def __init__(self, data_dir: Path, tracker: DownloadTracker):
        super().__init__(data_dir, tracker)
    
    def get_available_datasets(self) -> list[dict]:
        """Return list of available Atlanta Fed datasets."""
        return [
            {
                'name': 'GDPNow Current Estimate',
                'key': 'gdpnow_current',
                'description': 'Latest real-time GDP growth estimate for current quarter',
                'frequency': 'multiple per month'
            },
            {
                'name': 'GDPNow Historical Forecasts',
                'key': 'gdpnow_history',
                'description': 'Historical evolution of GDPNow forecasts',
                'frequency': 'quarterly archives'
            },
            {
                'name': 'GDPNow Component Tracking',
                'key': 'gdpnow_components',
                'description': 'Detailed component contributions to GDP forecast',
                'frequency': 'multiple per month'
            }
        ]
    
    def download_dataset(self, dataset_key: str, force: bool = False) -> Optional[Path]:
        """Download a specific Atlanta Fed dataset."""
        
        if dataset_key == 'gdpnow_current':
            return self._download_gdpnow_current(force)
        elif dataset_key == 'gdpnow_history':
            return self._download_gdpnow_history(force)
        elif dataset_key == 'gdpnow_components':
            return self._download_gdpnow_components(force)
        
        self.logger.warning(f"Unknown dataset key: {dataset_key}")
        return None
    
    def _download_gdpnow_current(self, force: bool = False) -> Optional[Path]:
        """Download current GDPNow estimate."""
        try:
            url = f"{self.BASE_URL}{self.ENDPOINTS['gdpnow']}"
            response = self.fetch_url(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract the current estimate
            # The page typically has a prominent display of the current forecast
            estimate_data = {
                'downloaded_at': datetime.now().isoformat(),
                'source_url': url,
                'raw_text': '',
                'estimate': None,
                'quarter': None,
                'last_updated': None
            }
            
            # Look for the main estimate text
            page_text = soup.get_text(separator=' ', strip=True)
            
            # Try to extract GDP estimate (usually in format like "2.5 percent")
            gdp_match = re.search(
                r'(?:GDPNow|model|estimate|forecast)[^\d]*?(-?\d+\.?\d*)\s*percent',
                page_text,
                re.IGNORECASE
            )
            if gdp_match:
                estimate_data['estimate'] = float(gdp_match.group(1))
            
            # Try to extract quarter reference
            quarter_match = re.search(
                r'(Q[1-4]|first|second|third|fourth)\s*(quarter)?\s*(\d{4})?',
                page_text,
                re.IGNORECASE
            )
            if quarter_match:
                estimate_data['quarter'] = quarter_match.group(0)
            
            # Try to extract last update date
            date_match = re.search(
                r'(?:updated?|as of)[:\s]*(\w+\s+\d+,?\s*\d{4})',
                page_text,
                re.IGNORECASE
            )
            if date_match:
                estimate_data['last_updated'] = date_match.group(1)
            
            estimate_data['raw_text'] = page_text[:3000]
            
            # Look for downloadable data files
            data_links = []
            for link in soup.find_all('a', href=True):
                href = link['href']
                if any(ext in href.lower() for ext in ['.xlsx', '.xls', '.csv']):
                    full_url = f"{self.BASE_URL}{href}" if href.startswith('/') else href
                    data_links.append({
                        'url': full_url,
                        'text': link.get_text(strip=True)
                    })
            
            estimate_data['data_files'] = data_links
            
            # Download Excel data if available
            for dl in data_links:
                if 'gdpnow' in dl['url'].lower():
                    try:
                        file_response = self.fetch_url(dl['url'])
                        filename = dl['url'].split('/')[-1].split('?')[0]
                        file_path = self.data_dir / filename
                        with open(file_path, 'wb') as f:
                            f.write(file_response.content)
                        estimate_data['downloaded_data_file'] = filename
                        self.logger.info(f"Downloaded GDPNow data file: {filename}")
                    except Exception as e:
                        self.logger.warning(f"Failed to download data file: {e}")
            
            return self.save_json(
                data=estimate_data,
                filename='gdpnow_current.json',
                source_url=url,
                metadata={'estimate': estimate_data.get('estimate')},
                force=force
            )
            
        except Exception as e:
            self.logger.error(f"Failed to download GDPNow current: {e}")
            return None
    
    def _download_gdpnow_history(self, force: bool = False) -> Optional[Path]:
        """Download GDPNow historical forecasts."""
        try:
            url = f"{self.BASE_URL}{self.ENDPOINTS['gdpnow_data']}"
            response = self.fetch_url(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find archive links
            archives = []
            for link in soup.find_all('a', href=True):
                href = link['href']
                text = link.get_text(strip=True)
                if 'archive' in href.lower() or re.search(r'20\d{2}', text):
                    full_url = f"{self.BASE_URL}{href}" if href.startswith('/') else href
                    archives.append({
                        'url': full_url,
                        'text': text
                    })
            
            # Also look for downloadable data
            data_files = []
            for link in soup.find_all('a', href=True):
                href = link['href']
                if any(ext in href.lower() for ext in ['.xlsx', '.xls', '.csv']):
                    full_url = f"{self.BASE_URL}{href}" if href.startswith('/') else href
                    data_files.append({
                        'url': full_url,
                        'filename': href.split('/')[-1].split('?')[0]
                    })
            
            # Download available data files
            downloaded = []
            for df in data_files[:5]:  # Limit to prevent excessive downloads
                try:
                    file_response = self.fetch_url(df['url'])
                    file_path = self.data_dir / f"history_{df['filename']}"
                    with open(file_path, 'wb') as f:
                        f.write(file_response.content)
                    downloaded.append(df['filename'])
                except Exception as e:
                    self.logger.warning(f"Failed to download {df['filename']}: {e}")
            
            data = {
                'downloaded_at': datetime.now().isoformat(),
                'source_url': url,
                'archives': archives,
                'data_files': data_files,
                'downloaded_files': downloaded
            }
            
            return self.save_json(
                data=data,
                filename='gdpnow_history.json',
                source_url=url,
                metadata={'archive_count': len(archives)},
                force=force
            )
            
        except Exception as e:
            self.logger.error(f"Failed to download GDPNow history: {e}")
            return None
    
    def _download_gdpnow_components(self, force: bool = False) -> Optional[Path]:
        """Download GDPNow component breakdown."""
        try:
            # The components are usually on the main GDPNow page
            url = f"{self.BASE_URL}{self.ENDPOINTS['gdpnow']}"
            response = self.fetch_url(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Look for tables with component data
            tables_data = []
            for table in soup.find_all('table'):
                table_rows = []
                for row in table.find_all('tr'):
                    cells = [cell.get_text(strip=True) for cell in row.find_all(['td', 'th'])]
                    if cells:
                        table_rows.append(cells)
                if table_rows:
                    tables_data.append(table_rows)
            
            # Extract component information from text
            components = []
            page_text = soup.get_text(separator='\n', strip=True)
            
            # Common GDP components to look for
            component_names = [
                'Personal consumption',
                'Gross private domestic investment',
                'Residential investment',
                'Nonresidential investment',
                'Equipment',
                'Intellectual property',
                'Structures',
                'Change in private inventories',
                'Net exports',
                'Exports',
                'Imports',
                'Government consumption'
            ]
            
            for comp in component_names:
                # Try to find the component and its contribution
                pattern = rf'{re.escape(comp)}[^\d]*?(-?\d+\.?\d*)'
                match = re.search(pattern, page_text, re.IGNORECASE)
                if match:
                    components.append({
                        'component': comp,
                        'value': float(match.group(1))
                    })
            
            data = {
                'downloaded_at': datetime.now().isoformat(),
                'source_url': url,
                'tables': tables_data,
                'extracted_components': components,
                'raw_text': page_text[:5000]
            }
            
            return self.save_json(
                data=data,
                filename='gdpnow_components.json',
                source_url=url,
                metadata={'component_count': len(components)},
                force=force
            )
            
        except Exception as e:
            self.logger.error(f"Failed to download GDPNow components: {e}")
            return None

