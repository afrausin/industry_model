"""
Macro Data Scrapers
====================
A collection of scrapers for US macroeconomic data sources.
"""

from .base_scraper import BaseScraper, DownloadTracker
from .fed_scraper import FederalReserveScraper
from .atlanta_fed_scraper import AtlantaFedScraper
from .fred_scraper import FREDScraper
from .ny_fed_scraper import NYFedScraper
from .cbo_scraper import CBOScraper

__all__ = [
    'BaseScraper',
    'DownloadTracker',
    'FederalReserveScraper',
    'AtlantaFedScraper',
    'FREDScraper',
    'NYFedScraper',
    'CBOScraper',
]
