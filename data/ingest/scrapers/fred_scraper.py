"""
Federal Reserve Bank of St. Louis (FRED) Data Scraper
=====================================================
Downloads data from FRED (Federal Reserve Economic Data):
- Economic time series
- FRED-MD macroeconomic dataset
- Key indicators and vintage data

FRED API: https://fred.stlouisfed.org/docs/api/fred/
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from .base_scraper import BaseScraper, DownloadTracker


class FREDScraper(BaseScraper):
    """
    Scraper for Federal Reserve Bank of St. Louis FRED data.
    
    Uses the FRED API for structured data access.
    Requires a free API key from https://fred.stlouisfed.org/docs/api/api_key.html
    
    Key features:
    - Individual series downloads
    - FRED-MD macroeconomic panel
    - Vintage/real-time data (ALFRED)
    """
    
    SOURCE_NAME = 'fred'
    BASE_URL = 'https://api.stlouisfed.org/fred'
    RATE_LIMIT_SECONDS = 0.5
    
    # Key macro series for comprehensive analysis
    KEY_SERIES = {
        # Output and Growth
        'GDP': 'Gross Domestic Product',
        'GDPC1': 'Real Gross Domestic Product',
        'A191RL1Q225SBEA': 'Real GDP Growth Rate (quarterly)',
        'INDPRO': 'Industrial Production Index',
        
        # Labor Market
        'UNRATE': 'Unemployment Rate',
        'PAYEMS': 'Total Nonfarm Payrolls',
        'CIVPART': 'Labor Force Participation Rate',
        'ICSA': 'Initial Jobless Claims',
        'CCSA': 'Continued Jobless Claims',
        'JTSJOL': 'Job Openings (JOLTS)',
        
        # Inflation
        'CPIAUCSL': 'Consumer Price Index (All Urban)',
        'CPILFESL': 'Core CPI (Less Food and Energy)',
        'PCEPI': 'PCE Price Index',
        'PCEPILFE': 'Core PCE Price Index',
        'MICH': 'University of Michigan Inflation Expectations',
        'T5YIFR': '5-Year Breakeven Inflation Rate',
        
        # Interest Rates
        'FEDFUNDS': 'Federal Funds Effective Rate',
        'DFF': 'Federal Funds Rate (Daily)',
        'DGS2': '2-Year Treasury Yield',
        'DGS10': '10-Year Treasury Yield',
        'DGS30': '30-Year Treasury Yield',
        'T10Y2Y': '10-Year Minus 2-Year Treasury (Yield Curve)',
        'T10Y3M': '10-Year Minus 3-Month Treasury',
        'DFII10': '10-Year TIPS Yield (Real Rate)',
        
        # Money and Credit
        'M2SL': 'M2 Money Stock',
        'WALCL': 'Fed Balance Sheet (Total Assets)',
        'TOTRESNS': 'Total Reserves',
        'DPCREDIT': 'Consumer Credit',
        
        # Housing
        'HOUST': 'Housing Starts',
        'PERMIT': 'Building Permits',
        'CSUSHPISA': 'S&P Case-Shiller Home Price Index',
        'MORTGAGE30US': '30-Year Fixed Mortgage Rate',
        
        # Financial Conditions
        'BAMLH0A0HYM2': 'High Yield Corporate Bond Spread',
        'VIXCLS': 'VIX Volatility Index',
        'SP500': 'S&P 500 Index',
        'NFCI': 'Chicago Fed National Financial Conditions Index',
        
        # Trade
        'BOPGSTB': 'Trade Balance',
        'DTWEXBGS': 'Trade Weighted US Dollar Index',
        
        # Sentiment
        'UMCSENT': 'University of Michigan Consumer Sentiment',
        'RSAFS': 'Retail Sales',
        'DGORDER': 'Durable Goods Orders',
    }
    
    def __init__(self, data_dir: Path, tracker: DownloadTracker, api_key: Optional[str] = None):
        super().__init__(data_dir, tracker)
        self.api_key = api_key or os.environ.get('FRED_API_KEY')
        if not self.api_key:
            self.logger.warning(
                "No FRED API key provided. Set FRED_API_KEY environment variable. "
                "Get a free key at: https://fred.stlouisfed.org/docs/api/api_key.html"
            )
    
    def _make_api_request(self, endpoint: str, params: dict) -> dict:
        """Make a request to the FRED API."""
        if not self.api_key:
            raise ValueError("FRED API key required. Set FRED_API_KEY environment variable.")
        
        params['api_key'] = self.api_key
        params['file_type'] = 'json'
        
        url = f"{self.BASE_URL}/{endpoint}"
        response = self.fetch_url(url, params=params)
        return response.json()
    
    def get_series(
        self,
        series_id: str,
        observation_start: Optional[str] = None,
        observation_end: Optional[str] = None,
        frequency: Optional[str] = None
    ) -> dict:
        """
        Get observations for a FRED series.
        
        Args:
            series_id: FRED series ID (e.g., 'UNRATE')
            observation_start: Start date (YYYY-MM-DD)
            observation_end: End date (YYYY-MM-DD)
            frequency: Aggregation frequency (d, w, bw, m, q, sa, a)
        """
        params = {'series_id': series_id}
        
        if observation_start:
            params['observation_start'] = observation_start
        if observation_end:
            params['observation_end'] = observation_end
        if frequency:
            params['frequency'] = frequency
        
        return self._make_api_request('series/observations', params)
    
    def get_series_info(self, series_id: str) -> dict:
        """Get metadata for a FRED series."""
        params = {'series_id': series_id}
        return self._make_api_request('series', params)
    
    def search_series(self, search_text: str, limit: int = 100) -> dict:
        """Search for FRED series by text."""
        params = {
            'search_text': search_text,
            'limit': limit
        }
        return self._make_api_request('series/search', params)
    
    def get_available_datasets(self) -> list[dict]:
        """Return list of available FRED datasets."""
        datasets = []
        
        # Individual key series
        for series_id, description in self.KEY_SERIES.items():
            datasets.append({
                'name': f"{series_id}: {description}",
                'key': f"series_{series_id.lower()}",
                'description': description,
                'frequency': 'varies',
                'series_id': series_id
            })
        
        # Composite datasets
        datasets.extend([
            {
                'name': 'Macro Dashboard - Growth',
                'key': 'macro_growth',
                'description': 'Key growth and output indicators',
                'frequency': 'quarterly/monthly'
            },
            {
                'name': 'Macro Dashboard - Labor',
                'key': 'macro_labor',
                'description': 'Key labor market indicators',
                'frequency': 'monthly'
            },
            {
                'name': 'Macro Dashboard - Inflation',
                'key': 'macro_inflation',
                'description': 'Key inflation indicators',
                'frequency': 'monthly'
            },
            {
                'name': 'Macro Dashboard - Rates',
                'key': 'macro_rates',
                'description': 'Interest rates and yield curve',
                'frequency': 'daily/weekly'
            },
            {
                'name': 'Macro Dashboard - Financial',
                'key': 'macro_financial',
                'description': 'Financial conditions and credit',
                'frequency': 'daily/weekly'
            },
            {
                'name': 'Full Macro Dataset',
                'key': 'macro_full',
                'description': 'All key macroeconomic series',
                'frequency': 'varies'
            }
        ])
        
        return datasets
    
    def download_dataset(self, dataset_key: str, force: bool = False) -> Optional[Path]:
        """Download a specific FRED dataset."""
        
        if not self.api_key:
            self.logger.error("Cannot download: No FRED API key configured")
            return None
        
        # Handle composite datasets
        if dataset_key == 'macro_growth':
            series = ['GDP', 'GDPC1', 'A191RL1Q225SBEA', 'INDPRO']
            return self._download_series_bundle(series, 'macro_growth', force)
        
        elif dataset_key == 'macro_labor':
            series = ['UNRATE', 'PAYEMS', 'CIVPART', 'ICSA', 'CCSA', 'JTSJOL']
            return self._download_series_bundle(series, 'macro_labor', force)
        
        elif dataset_key == 'macro_inflation':
            series = ['CPIAUCSL', 'CPILFESL', 'PCEPI', 'PCEPILFE', 'MICH', 'T5YIFR']
            return self._download_series_bundle(series, 'macro_inflation', force)
        
        elif dataset_key == 'macro_rates':
            series = ['FEDFUNDS', 'DFF', 'DGS2', 'DGS10', 'DGS30', 'T10Y2Y', 'T10Y3M', 'DFII10']
            return self._download_series_bundle(series, 'macro_rates', force)
        
        elif dataset_key == 'macro_financial':
            series = ['M2SL', 'WALCL', 'BAMLH0A0HYM2', 'VIXCLS', 'NFCI']
            return self._download_series_bundle(series, 'macro_financial', force)
        
        elif dataset_key == 'macro_full':
            return self._download_series_bundle(list(self.KEY_SERIES.keys()), 'macro_full', force)
        
        # Handle individual series
        if dataset_key.startswith('series_'):
            series_id = dataset_key.replace('series_', '').upper()
            return self._download_single_series(series_id, force)
        
        self.logger.warning(f"Unknown dataset key: {dataset_key}")
        return None
    
    def _download_single_series(self, series_id: str, force: bool = False) -> Optional[Path]:
        """Download a single FRED series."""
        try:
            # Get series info
            info = self.get_series_info(series_id)
            
            # Get observations (last 30 years)
            observations = self.get_series(
                series_id,
                observation_start='1995-01-01'
            )
            
            data = {
                'downloaded_at': datetime.now().isoformat(),
                'series_id': series_id,
                'description': self.KEY_SERIES.get(series_id, ''),
                'info': info,
                'observations': observations
            }
            
            filename = f"series_{series_id.lower()}.json"
            return self.save_json(
                data=data,
                filename=filename,
                source_url=f"{self.BASE_URL}/series?series_id={series_id}",
                metadata={'series_id': series_id},
                force=force
            )
            
        except Exception as e:
            self.logger.error(f"Failed to download series {series_id}: {e}")
            return None
    
    def _download_series_bundle(
        self,
        series_ids: list[str],
        bundle_name: str,
        force: bool = False
    ) -> Optional[Path]:
        """Download a bundle of related FRED series."""
        bundle_data = {
            'downloaded_at': datetime.now().isoformat(),
            'bundle_name': bundle_name,
            'series_count': len(series_ids),
            'series': {}
        }
        
        success_count = 0
        for series_id in series_ids:
            try:
                observations = self.get_series(
                    series_id,
                    observation_start='1995-01-01'
                )
                
                bundle_data['series'][series_id] = {
                    'description': self.KEY_SERIES.get(series_id, ''),
                    'observations': observations
                }
                success_count += 1
                
            except Exception as e:
                self.logger.warning(f"Failed to fetch {series_id}: {e}")
                bundle_data['series'][series_id] = {'error': str(e)}
        
        if success_count == 0:
            self.logger.error(f"Failed to download any series for {bundle_name}")
            return None
        
        return self.save_json(
            data=bundle_data,
            filename=f"{bundle_name}.json",
            source_url=self.BASE_URL,
            metadata={
                'bundle_name': bundle_name,
                'success_count': success_count,
                'total_count': len(series_ids)
            },
            force=force
        )
    
    def download_fred_md(self, force: bool = False) -> Optional[Path]:
        """
        Download FRED-MD macroeconomic dataset.
        
        FRED-MD is a large macroeconomic database designed for
        empirical analysis in macroeconomics and finance.
        https://research.stlouisfed.org/econ/mccracken/fred-databases/
        """
        try:
            # FRED-MD is available as a direct CSV download
            url = 'https://files.stlouisfed.org/files/htdocs/fred-md/monthly/current.csv'
            response = self.fetch_url(url)
            
            filename = f"fred_md_{datetime.now().strftime('%Y%m')}.csv"
            return self.save_file(
                content=response.content,
                filename=filename,
                source_url=url,
                source_key=f"{self.SOURCE_NAME}/fred_md",
                metadata={'type': 'FRED-MD'},
                force=force
            )
            
        except Exception as e:
            self.logger.error(f"Failed to download FRED-MD: {e}")
            return None

