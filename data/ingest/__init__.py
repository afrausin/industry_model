"""
Ingest - Data Collection Layer
==============================

This module handles all data ingestion from external sources.

Submodules:
    - scrapers: Web scrapers for FRED, Fed, Atlanta Fed, etc.
    - fmp: Financial Modeling Prep API integration
"""

from .scrapers import (
    DownloadTracker,
    BaseScraper,
    FREDScraper,
    FederalReserveScraper,
    AtlantaFedScraper,
    NYFedScraper,
    CBOScraper,
    BrookingsScraper,
    NBERScraper,
    PIIEScraper,
    IMFScraper,
    OECDScraper,
)

__all__ = [
    "DownloadTracker",
    "BaseScraper",
    "FREDScraper",
    "FederalReserveScraper",
    "AtlantaFedScraper",
    "NYFedScraper",
    "CBOScraper",
    "BrookingsScraper",
    "NBERScraper",
    "PIIEScraper",
    "IMFScraper",
    "OECDScraper",
]
