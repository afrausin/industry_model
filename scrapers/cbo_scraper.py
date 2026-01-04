"""
Congressional Budget Office (CBO) Data Scraper
===============================================
Downloads data from the CBO including:
- Budget and Economic Outlook
- Long-Term Budget Projections
- Cost Estimates
- Economic data and forecasts

CBO Data: https://www.cbo.gov/data/budget-economic-data
"""

import re
from datetime import datetime
from pathlib import Path
from typing import Optional
from bs4 import BeautifulSoup

from .base_scraper import BaseScraper, DownloadTracker


class CBOScraper(BaseScraper):
    """
    Scraper for Congressional Budget Office data.
    
    Key datasets:
    - Budget and Economic Outlook (10-year projections)
    - Long-Term Budget Outlook
    - Historical Budget Data
    - Economic projections
    """
    
    SOURCE_NAME = 'cbo'
    BASE_URL = 'https://www.cbo.gov'
    RATE_LIMIT_SECONDS = 3.0  # Be extra respectful
    
    def __init__(self, data_dir: Path, tracker: DownloadTracker):
        super().__init__(data_dir, tracker)
        # CBO requires browser-like headers
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })
    
    ENDPOINTS = {
        'data_home': '/data/budget-economic-data',
        'budget_projections': '/data/budget-projections',
        'economic_projections': '/data/10-year-economic-projections',
        'historical_budget': '/data/historical-budget-data',
        'long_term': '/data/long-term-budget-projections',
    }
    
    def get_available_datasets(self) -> list[dict]:
        """Return list of available CBO datasets."""
        return [
            {
                'name': 'Budget Projections',
                'key': 'budget_projections',
                'description': '10-year federal budget projections',
                'frequency': 'twice yearly'
            },
            {
                'name': 'Economic Projections',
                'key': 'economic_projections',
                'description': '10-year economic forecast (GDP, inflation, rates)',
                'frequency': 'twice yearly'
            },
            {
                'name': 'Historical Budget Data',
                'key': 'historical_budget',
                'description': 'Historical federal revenues, outlays, deficits, debt',
                'frequency': 'annual'
            },
            {
                'name': 'Long-Term Budget Outlook',
                'key': 'long_term_outlook',
                'description': '30-year budget projections and fiscal sustainability',
                'frequency': 'annual'
            },
            {
                'name': 'All CBO Data Files',
                'key': 'all_data',
                'description': 'Comprehensive download of available CBO datasets',
                'frequency': 'varies'
            }
        ]
    
    def download_dataset(self, dataset_key: str, force: bool = False) -> Optional[Path]:
        """Download a specific CBO dataset."""
        
        if dataset_key == 'budget_projections':
            return self._download_budget_projections(force)
        elif dataset_key == 'economic_projections':
            return self._download_economic_projections(force)
        elif dataset_key == 'historical_budget':
            return self._download_historical_budget(force)
        elif dataset_key == 'long_term_outlook':
            return self._download_long_term_outlook(force)
        elif dataset_key == 'all_data':
            return self._download_all_data(force)
        
        self.logger.warning(f"Unknown dataset key: {dataset_key}")
        return None
    
    def _find_data_files(self, soup: BeautifulSoup) -> list[dict]:
        """Find downloadable data files on a CBO page."""
        data_files = []
        
        for link in soup.find_all('a', href=True):
            href = link['href']
            text = link.get_text(strip=True)
            
            # Look for Excel, CSV, or PDF files
            if any(ext in href.lower() for ext in ['.xlsx', '.xls', '.csv', '.zip']):
                full_url = f"{self.BASE_URL}{href}" if href.startswith('/') else href
                data_files.append({
                    'url': full_url,
                    'text': text,
                    'type': href.split('.')[-1].lower()
                })
        
        return data_files
    
    def _download_files(self, data_files: list[dict], prefix: str = '') -> list[str]:
        """Download a list of data files."""
        downloaded = []
        
        for df in data_files:
            try:
                response = self.fetch_url(df['url'])
                
                # Generate filename
                url_filename = df['url'].split('/')[-1].split('?')[0]
                filename = f"{prefix}_{url_filename}" if prefix else url_filename
                
                file_path = self.data_dir / filename
                with open(file_path, 'wb') as f:
                    f.write(response.content)
                
                downloaded.append(filename)
                self.logger.info(f"Downloaded: {filename}")
                
            except Exception as e:
                self.logger.warning(f"Failed to download {df['url']}: {e}")
        
        return downloaded
    
    def _download_budget_projections(self, force: bool = False) -> Optional[Path]:
        """Download CBO budget projections."""
        try:
            url = f"{self.BASE_URL}{self.ENDPOINTS['budget_projections']}"
            response = self.fetch_url(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find data files
            data_files = self._find_data_files(soup)
            
            # Filter for budget-related files
            budget_files = [
                df for df in data_files
                if any(kw in df['text'].lower() or kw in df['url'].lower()
                       for kw in ['budget', 'revenue', 'outlay', 'deficit', 'projection'])
            ]
            
            # Download files
            downloaded = self._download_files(budget_files[:10], 'budget')
            
            # Extract page summary
            page_text = soup.get_text(separator='\n', strip=True)
            
            data = {
                'downloaded_at': datetime.now().isoformat(),
                'source_url': url,
                'available_files': data_files,
                'downloaded_files': downloaded,
                'page_summary': page_text[:3000]
            }
            
            return self.save_json(
                data=data,
                filename='budget_projections.json',
                source_url=url,
                metadata={'file_count': len(downloaded)},
                force=force
            )
            
        except Exception as e:
            self.logger.error(f"Failed to download budget projections: {e}")
            return None
    
    def _download_economic_projections(self, force: bool = False) -> Optional[Path]:
        """Download CBO economic projections."""
        try:
            url = f"{self.BASE_URL}{self.ENDPOINTS['economic_projections']}"
            response = self.fetch_url(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find data files
            data_files = self._find_data_files(soup)
            
            # Filter for economic projection files
            econ_files = [
                df for df in data_files
                if any(kw in df['text'].lower() or kw in df['url'].lower()
                       for kw in ['economic', 'gdp', 'inflation', 'unemployment', 'interest'])
            ]
            
            # Download files
            downloaded = self._download_files(econ_files[:10], 'economic')
            
            # Try to extract key projections from page
            page_text = soup.get_text(separator='\n', strip=True)
            
            projections = {}
            
            # Look for GDP growth
            gdp_match = re.search(r'(?:real\s*)?GDP[^\d]*(\d+\.?\d*)\s*percent', page_text, re.IGNORECASE)
            if gdp_match:
                projections['gdp_growth'] = float(gdp_match.group(1))
            
            # Look for unemployment
            unemp_match = re.search(r'unemployment[^\d]*(\d+\.?\d*)\s*percent', page_text, re.IGNORECASE)
            if unemp_match:
                projections['unemployment'] = float(unemp_match.group(1))
            
            # Look for inflation
            infl_match = re.search(r'(?:CPI|inflation)[^\d]*(\d+\.?\d*)\s*percent', page_text, re.IGNORECASE)
            if infl_match:
                projections['inflation'] = float(infl_match.group(1))
            
            data = {
                'downloaded_at': datetime.now().isoformat(),
                'source_url': url,
                'extracted_projections': projections,
                'available_files': data_files,
                'downloaded_files': downloaded,
                'page_summary': page_text[:3000]
            }
            
            return self.save_json(
                data=data,
                filename='economic_projections.json',
                source_url=url,
                metadata={'projections': projections, 'file_count': len(downloaded)},
                force=force
            )
            
        except Exception as e:
            self.logger.error(f"Failed to download economic projections: {e}")
            return None
    
    def _download_historical_budget(self, force: bool = False) -> Optional[Path]:
        """Download CBO historical budget data."""
        try:
            url = f"{self.BASE_URL}{self.ENDPOINTS['historical_budget']}"
            response = self.fetch_url(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find data files
            data_files = self._find_data_files(soup)
            
            # Download Excel files (usually contain the comprehensive data)
            excel_files = [df for df in data_files if df['type'] in ['xlsx', 'xls']]
            downloaded = self._download_files(excel_files[:5], 'historical')
            
            data = {
                'downloaded_at': datetime.now().isoformat(),
                'source_url': url,
                'available_files': data_files,
                'downloaded_files': downloaded
            }
            
            return self.save_json(
                data=data,
                filename='historical_budget.json',
                source_url=url,
                metadata={'file_count': len(downloaded)},
                force=force
            )
            
        except Exception as e:
            self.logger.error(f"Failed to download historical budget: {e}")
            return None
    
    def _download_long_term_outlook(self, force: bool = False) -> Optional[Path]:
        """Download CBO long-term budget outlook."""
        try:
            url = f"{self.BASE_URL}{self.ENDPOINTS['long_term']}"
            response = self.fetch_url(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find data files
            data_files = self._find_data_files(soup)
            
            # Download files
            downloaded = self._download_files(data_files[:5], 'longterm')
            
            # Extract key long-term metrics
            page_text = soup.get_text(separator='\n', strip=True)
            
            metrics = {}
            
            # Look for debt-to-GDP projections
            debt_match = re.search(r'debt[^\d]*(\d+)\s*percent\s*(?:of\s*)?GDP', page_text, re.IGNORECASE)
            if debt_match:
                metrics['debt_to_gdp'] = int(debt_match.group(1))
            
            data = {
                'downloaded_at': datetime.now().isoformat(),
                'source_url': url,
                'key_metrics': metrics,
                'available_files': data_files,
                'downloaded_files': downloaded,
                'page_summary': page_text[:3000]
            }
            
            return self.save_json(
                data=data,
                filename='long_term_outlook.json',
                source_url=url,
                metadata={'metrics': metrics},
                force=force
            )
            
        except Exception as e:
            self.logger.error(f"Failed to download long-term outlook: {e}")
            return None
    
    def _download_all_data(self, force: bool = False) -> Optional[Path]:
        """Download all available CBO data from the data homepage."""
        try:
            url = f"{self.BASE_URL}{self.ENDPOINTS['data_home']}"
            response = self.fetch_url(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find all data files
            data_files = self._find_data_files(soup)
            
            # Download all (with reasonable limit)
            downloaded = self._download_files(data_files[:20], '')
            
            data = {
                'downloaded_at': datetime.now().isoformat(),
                'source_url': url,
                'total_files_found': len(data_files),
                'available_files': data_files,
                'downloaded_files': downloaded
            }
            
            return self.save_json(
                data=data,
                filename='cbo_all_data_index.json',
                source_url=url,
                metadata={'total_files': len(data_files), 'downloaded': len(downloaded)},
                force=force
            )
            
        except Exception as e:
            self.logger.error(f"Failed to download all CBO data: {e}")
            return None

