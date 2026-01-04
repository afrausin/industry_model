"""
Data Loader for Hedgeye Quadrant Framework

Consolidates macro data from various sources into a structured format
for analysis.
"""

import json
import re
import io
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

import pandas as pd
import numpy as np
import requests

try:
    import pdfplumber
    PDF_EXTRACTION_AVAILABLE = True
except ImportError:
    PDF_EXTRACTION_AVAILABLE = False
    print("Warning: pdfplumber not installed. PDF extraction will not be available.")


@dataclass
class MacroDataPoint:
    """Represents a single macro data observation."""
    date: datetime
    value: float
    series_id: str
    description: str


@dataclass
class QualitativeDocument:
    """Represents a qualitative document (Fed statement, Beige Book, etc.)."""
    date: datetime
    source: str
    title: str
    content: str
    word_count: int


def clean_text_content(text: str) -> str:
    """
    Clean text content by removing binary/non-printable characters.
    
    This is necessary because some scraped documents (especially FOMC minutes)
    contain embedded PDF/binary data.
    """
    if not text:
        return ""
    
    # Remove null bytes
    text = text.replace('\x00', '')
    
    # Remove other non-printable characters (except newlines, tabs)
    # Keep only printable ASCII + common whitespace + some extended chars
    cleaned = []
    for char in text:
        if char.isprintable() or char in '\n\r\t':
            cleaned.append(char)
        elif ord(char) > 127:
            # Replace non-ASCII with space (might be encoding issues)
            cleaned.append(' ')
    
    text = ''.join(cleaned)
    
    # Collapse multiple spaces
    text = re.sub(r' {3,}', '  ', text)
    
    # Collapse multiple newlines (more than 3)
    text = re.sub(r'\n{4,}', '\n\n\n', text)
    
    return text.strip()


def extract_text_from_pdf_url(url: str, cache_dir: Optional[Path] = None) -> Tuple[str, bytes]:
    """
    Download a PDF from URL and extract text using pdfplumber.
    
    Args:
        url: URL of the PDF to download
        cache_dir: Optional directory to cache downloaded PDFs
        
    Returns:
        Tuple of (extracted_text, pdf_bytes)
    """
    if not PDF_EXTRACTION_AVAILABLE:
        raise ImportError("pdfplumber is required for PDF extraction. Install with: pip install pdfplumber")
    
    # Create cache key from URL
    url_hash = hashlib.md5(url.encode()).hexdigest()[:12]
    
    pdf_bytes = None
    
    # Check cache first
    if cache_dir:
        cache_file = cache_dir / f"pdf_cache_{url_hash}.pdf"
        text_cache = cache_dir / f"pdf_text_{url_hash}.txt"
        
        # If we have cached text, return it
        if text_cache.exists():
            with open(text_cache, 'r', encoding='utf-8') as f:
                cached_text = f.read()
            if cache_file.exists():
                with open(cache_file, 'rb') as f:
                    pdf_bytes = f.read()
            return cached_text, pdf_bytes or b''
        
        # If we have cached PDF, use it
        if cache_file.exists():
            with open(cache_file, 'rb') as f:
                pdf_bytes = f.read()
    
    # Download PDF if not cached
    if pdf_bytes is None:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        pdf_bytes = response.content
        
        # Cache the PDF
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)
            with open(cache_file, 'wb') as f:
                f.write(pdf_bytes)
    
    # Extract text
    full_text = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                full_text.append(page_text)
    
    extracted_text = '\n\n'.join(full_text)
    
    # Cache the extracted text
    if cache_dir:
        with open(text_cache, 'w', encoding='utf-8') as f:
            f.write(extracted_text)
    
    return extracted_text, pdf_bytes


def get_pdf_bytes_from_url(url: str, cache_dir: Optional[Path] = None) -> bytes:
    """
    Download a PDF from URL and return raw bytes.
    Useful for sending PDFs directly to Gemini multimodal API.
    
    Args:
        url: URL of the PDF to download
        cache_dir: Optional directory to cache downloaded PDFs
        
    Returns:
        Raw PDF bytes
    """
    url_hash = hashlib.md5(url.encode()).hexdigest()[:12]
    
    # Check cache first
    if cache_dir:
        cache_file = cache_dir / f"pdf_cache_{url_hash}.pdf"
        if cache_file.exists():
            with open(cache_file, 'rb') as f:
                return f.read()
    
    # Download PDF
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    pdf_bytes = response.content
    
    # Cache the PDF
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
        with open(cache_file, 'wb') as f:
            f.write(pdf_bytes)
    
    return pdf_bytes


class MacroDataLoader:
    """Loads and consolidates macro data from various sources."""
    
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self._cache: Dict[str, Any] = {}
    
    def load_fred_series(self, series_id: str) -> pd.DataFrame:
        """Load a single FRED series from local data."""
        # Check individual series file first
        series_file = self.data_dir / "fred" / f"series_{series_id.lower()}.json"
        
        if series_file.exists():
            with open(series_file, 'r') as f:
                data = json.load(f)
            return self._parse_fred_observations(data, series_id)
        
        # Fall back to bundle files
        for bundle in ["macro_growth", "macro_inflation", "macro_labor", "macro_rates", "macro_full"]:
            bundle_file = self.data_dir / "fred" / f"{bundle}.json"
            if bundle_file.exists():
                with open(bundle_file, 'r') as f:
                    bundle_data = json.load(f)
                if "series" in bundle_data and series_id in bundle_data["series"]:
                    series_data = bundle_data["series"][series_id]
                    return self._parse_fred_observations(series_data, series_id)
        
        raise FileNotFoundError(f"Series {series_id} not found in local data")
    
    def _parse_fred_observations(self, data: Dict, series_id: str) -> pd.DataFrame:
        """Parse FRED observations into a DataFrame."""
        if "observations" in data:
            obs_data = data["observations"]
            if isinstance(obs_data, dict) and "observations" in obs_data:
                observations = obs_data["observations"]
            else:
                observations = obs_data
        else:
            observations = []
        
        records = []
        for obs in observations:
            try:
                value = float(obs["value"]) if obs["value"] != "." else np.nan
                records.append({
                    "date": pd.to_datetime(obs["date"]),
                    "value": value,
                    "series_id": series_id,
                })
            except (ValueError, KeyError):
                continue
        
        df = pd.DataFrame(records)
        if not df.empty:
            df = df.set_index("date").sort_index()
        return df
    
    def load_growth_data(self, series_ids: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        """Load all growth-related series."""
        if series_ids is None:
            series_ids = ["GDPC1", "INDPRO", "PAYEMS", "RSAFS"]
        
        result = {}
        for series_id in series_ids:
            try:
                result[series_id] = self.load_fred_series(series_id)
            except FileNotFoundError:
                print(f"Warning: Series {series_id} not found")
        return result
    
    def load_inflation_data(self, series_ids: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        """Load all inflation-related series."""
        if series_ids is None:
            series_ids = ["CPIAUCSL", "CPILFESL", "PCEPI", "PCEPILFE"]
        
        result = {}
        for series_id in series_ids:
            try:
                result[series_id] = self.load_fred_series(series_id)
            except FileNotFoundError:
                print(f"Warning: Series {series_id} not found")
        return result
    
    def load_all_series_by_category(
        self,
        all_series_config: Dict[str, Dict[str, str]],
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Load ALL available time series organized by category.
        
        Args:
            all_series_config: Dict from config.all_series mapping categories to series
            
        Returns:
            Dict[category, Dict[series_id, DataFrame]]
        """
        result = {}
        
        for category, series_dict in all_series_config.items():
            result[category] = {}
            for series_id, description in series_dict.items():
                try:
                    df = self.load_fred_series(series_id)
                    if not df.empty:
                        result[category][series_id] = df
                except FileNotFoundError:
                    pass  # Silently skip missing series
        
        return result
    
    def load_all_available_series(self) -> Dict[str, pd.DataFrame]:
        """
        Load ALL available FRED series from the data folder.
        
        Scans the fred directory for all series_*.json files and loads them.
        
        Returns:
            Dict[series_id, DataFrame]
        """
        result = {}
        fred_dir = self.data_dir / "fred"
        
        if not fred_dir.exists():
            return result
        
        # Find all series files
        for file_path in fred_dir.glob("series_*.json"):
            series_id = file_path.stem.replace("series_", "").upper()
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                df = self._parse_fred_observations(data, series_id)
                if not df.empty:
                    result[series_id] = df
            except Exception as e:
                print(f"Warning: Failed to load {series_id}: {e}")
        
        return result
    
    def load_gdpnow(self) -> Dict[str, Any]:
        """Load Atlanta Fed GDPNow estimate."""
        gdpnow_file = self.data_dir / "atlanta_fed" / "gdpnow_current.json"
        if gdpnow_file.exists():
            with open(gdpnow_file, 'r') as f:
                return json.load(f)
        raise FileNotFoundError("GDPNow data not found")
    
    def load_fomc_statements(self, n_recent: int = 5) -> List[QualitativeDocument]:
        """Load recent FOMC statements."""
        statements_file = self.data_dir / "federal_reserve" / "fomc_statements.json"
        if not statements_file.exists():
            raise FileNotFoundError("FOMC statements not found")
        
        with open(statements_file, 'r') as f:
            data = json.load(f)
        
        documents = []
        for stmt in data.get("statements", [])[:n_recent]:
            try:
                # Parse date from format YYYYMMDD
                date_str = stmt.get("date", "")
                if len(date_str) == 8:
                    date = datetime.strptime(date_str, "%Y%m%d")
                else:
                    continue
                
                # Clean content to remove any binary/non-printable chars
                raw_content = stmt.get("content", "")
                cleaned_content = clean_text_content(raw_content)
                
                documents.append(QualitativeDocument(
                    date=date,
                    source="FOMC Statement",
                    title=f"FOMC Statement - {date.strftime('%B %d, %Y')}",
                    content=cleaned_content,
                    word_count=len(cleaned_content.split()),
                ))
            except (ValueError, KeyError):
                continue
        
        return documents
    
    def load_beige_book(self, n_recent: int = 3) -> List[QualitativeDocument]:
        """Load recent Beige Book editions."""
        beige_file = self.data_dir / "federal_reserve" / "beige_book.json"
        if not beige_file.exists():
            raise FileNotFoundError("Beige Book not found")
        
        with open(beige_file, 'r') as f:
            data = json.load(f)
        
        documents = []
        for edition in data.get("editions", [])[:n_recent]:
            try:
                # Parse date from format YYYYMM
                date_str = edition.get("date", "")
                if len(date_str) == 6:
                    date = datetime.strptime(date_str + "01", "%Y%m%d")
                else:
                    continue
                
                documents.append(QualitativeDocument(
                    date=date,
                    source="Beige Book",
                    title=f"Beige Book - {date.strftime('%B %Y')}",
                    content=edition.get("content", ""),
                    word_count=edition.get("word_count", 0),
                ))
            except (ValueError, KeyError):
                continue
        
        return documents
    
    def load_fomc_minutes(
        self, 
        n_recent: int = 3,
        extract_from_pdf: bool = True,
    ) -> List[QualitativeDocument]:
        """
        Load recent FOMC minutes.
        
        Args:
            n_recent: Number of recent minutes to load
            extract_from_pdf: If True, download and extract text from PDF URLs.
                            If False, use stored content (may be corrupted binary).
        """
        minutes_file = self.data_dir / "federal_reserve" / "fomc_minutes.json"
        if not minutes_file.exists():
            raise FileNotFoundError("FOMC minutes not found")
        
        with open(minutes_file, 'r') as f:
            data = json.load(f)
        
        # Cache directory for PDFs
        cache_dir = self.data_dir / "federal_reserve" / "pdf_cache"
        
        documents = []
        for minutes in data.get("minutes", [])[:n_recent]:
            try:
                date_str = minutes.get("date", "")
                if len(date_str) == 8:
                    date = datetime.strptime(date_str, "%Y%m%d")
                else:
                    continue
                
                content = ""
                word_count = 0
                
                if extract_from_pdf and PDF_EXTRACTION_AVAILABLE:
                    pdf_url = minutes.get("url", "")
                    if pdf_url:
                        try:
                            print(f"Extracting text from FOMC Minutes PDF: {date_str}...")
                            content, _ = extract_text_from_pdf_url(pdf_url, cache_dir)
                            word_count = len(content.split())
                        except Exception as e:
                            print(f"Warning: Failed to extract PDF for {date_str}: {e}")
                            # Fall back to stored content
                            content = clean_text_content(minutes.get("content", ""))
                            word_count = len(content.split())
                else:
                    content = clean_text_content(minutes.get("content", ""))
                    word_count = minutes.get("word_count", len(content.split()))
                
                if content and word_count > 100:  # Skip if content is too short (likely corrupted)
                    documents.append(QualitativeDocument(
                        date=date,
                        source="FOMC Minutes",
                        title=f"FOMC Minutes - {date.strftime('%B %d, %Y')}",
                        content=content,
                        word_count=word_count,
                    ))
            except (ValueError, KeyError) as e:
                print(f"Warning: Failed to load FOMC minutes entry: {e}")
                continue
        
        return documents
    
    def get_fomc_minutes_pdf_bytes(
        self, 
        n_recent: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Get raw PDF bytes for FOMC minutes (for Gemini multimodal analysis).
        
        Returns:
            List of dicts with 'date', 'title', 'pdf_bytes', 'url'
        """
        minutes_file = self.data_dir / "federal_reserve" / "fomc_minutes.json"
        if not minutes_file.exists():
            raise FileNotFoundError("FOMC minutes not found")
        
        with open(minutes_file, 'r') as f:
            data = json.load(f)
        
        cache_dir = self.data_dir / "federal_reserve" / "pdf_cache"
        
        results = []
        for minutes in data.get("minutes", [])[:n_recent]:
            try:
                date_str = minutes.get("date", "")
                if len(date_str) == 8:
                    date = datetime.strptime(date_str, "%Y%m%d")
                else:
                    continue
                
                pdf_url = minutes.get("url", "")
                if pdf_url:
                    try:
                        print(f"Downloading FOMC Minutes PDF: {date_str}...")
                        pdf_bytes = get_pdf_bytes_from_url(pdf_url, cache_dir)
                        results.append({
                            "date": date,
                            "title": f"FOMC Minutes - {date.strftime('%B %d, %Y')}",
                            "pdf_bytes": pdf_bytes,
                            "url": pdf_url,
                        })
                    except Exception as e:
                        print(f"Warning: Failed to download PDF for {date_str}: {e}")
            except (ValueError, KeyError):
                continue
        
        return results
    
    def load_sep_projections(self) -> Dict[str, Any]:
        """Load Summary of Economic Projections."""
        sep_file = self.data_dir / "federal_reserve" / "sep_projections.json"
        if sep_file.exists():
            with open(sep_file, 'r') as f:
                return json.load(f)
        raise FileNotFoundError("SEP projections not found")
    
    def load_all_qualitative_docs(
        self,
        n_statements: int = 3,
        n_beige: int = 2,
        n_minutes: int = 2,
    ) -> List[QualitativeDocument]:
        """Load all qualitative documents for analysis."""
        documents = []
        
        try:
            documents.extend(self.load_fomc_statements(n_statements))
        except FileNotFoundError:
            print("Warning: FOMC statements not available")
        
        try:
            documents.extend(self.load_beige_book(n_beige))
        except FileNotFoundError:
            print("Warning: Beige Book not available")
        
        try:
            documents.extend(self.load_fomc_minutes(n_minutes))
        except FileNotFoundError:
            print("Warning: FOMC minutes not available")
        
        # Sort by date, most recent first
        documents.sort(key=lambda x: x.date, reverse=True)
        return documents
    
    def get_latest_data_summary(self) -> Dict[str, Any]:
        """Get a summary of the latest available data points."""
        summary = {
            "timestamp": datetime.now().isoformat(),
            "growth_indicators": {},
            "inflation_indicators": {},
            "gdpnow": None,
            "qualitative_docs": [],
        }
        
        # Growth data
        growth_data = self.load_growth_data()
        for series_id, df in growth_data.items():
            if not df.empty:
                latest = df.iloc[-1]
                summary["growth_indicators"][series_id] = {
                    "date": latest.name.isoformat(),
                    "value": float(latest["value"]),
                }
        
        # Inflation data
        inflation_data = self.load_inflation_data()
        for series_id, df in inflation_data.items():
            if not df.empty:
                latest = df.iloc[-1]
                summary["inflation_indicators"][series_id] = {
                    "date": latest.name.isoformat(),
                    "value": float(latest["value"]),
                }
        
        # GDPNow
        try:
            gdpnow = self.load_gdpnow()
            summary["gdpnow"] = {
                "estimate": gdpnow.get("estimate"),
                "quarter": gdpnow.get("quarter"),
                "last_updated": gdpnow.get("last_updated"),
            }
        except FileNotFoundError:
            pass
        
        # Qualitative docs summary
        docs = self.load_all_qualitative_docs()
        for doc in docs[:5]:
            summary["qualitative_docs"].append({
                "date": doc.date.isoformat(),
                "source": doc.source,
                "title": doc.title,
                "word_count": doc.word_count,
            })
        
        return summary

