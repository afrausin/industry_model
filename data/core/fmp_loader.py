"""
FMP Data Loader for Market Pricing Indicators

Loads real-time market data from Financial Modeling Prep API results
to understand what markets are currently pricing.
"""

import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass


@dataclass
class MarketQuote:
    """Single market quote."""
    symbol: str
    price: float
    change: float
    change_pct: Optional[float] = None
    volume: Optional[int] = None


@dataclass
class EconomicEvent:
    """Economic calendar event."""
    date: datetime
    country: str
    event: str
    previous: Optional[float]
    estimate: Optional[float]
    actual: Optional[float]
    impact: str
    unit: Optional[str]


class FMPDataLoader:
    """Loads market data from FMP exploration results."""
    
    def __init__(self, fmp_results_dir: Path):
        self.results_dir = fmp_results_dir
        self._cache: Dict[str, Any] = {}
    
    def load_index_quotes(self) -> Dict[str, MarketQuote]:
        """Load all index quotes from batch-index-quotes.json."""
        quotes_file = self.results_dir / "batch-index-quotes.json"
        if not quotes_file.exists():
            return {}
        
        with open(quotes_file, 'r') as f:
            data = json.load(f)
        
        result = {}
        for quote in data:
            symbol = quote.get("symbol", "")
            price = quote.get("price", 0)
            change = quote.get("change", 0)
            volume = quote.get("volume", 0)
            
            # Calculate percent change
            prev_price = price - change if price else 0
            change_pct = (change / prev_price * 100) if prev_price else 0
            
            result[symbol] = MarketQuote(
                symbol=symbol,
                price=price,
                change=change,
                change_pct=change_pct,
                volume=volume if volume else None,
            )
        
        return result
    
    def load_categorized_indices(self) -> Dict[str, List[MarketQuote]]:
        """Load indices organized by category."""
        cat_file = self.results_dir / "indices_categorized.json"
        if not cat_file.exists():
            return {}
        
        with open(cat_file, 'r') as f:
            data = json.load(f)
        
        result = {}
        for category, quotes in data.items():
            result[category] = []
            for q in quotes:
                price = q.get("price", 0)
                change = q.get("change", 0)
                prev = price - change if price else 0
                change_pct = (change / prev * 100) if prev else 0
                
                result[category].append(MarketQuote(
                    symbol=q.get("symbol", ""),
                    price=price,
                    change=change,
                    change_pct=change_pct,
                    volume=q.get("volume"),
                ))
        
        return result
    
    def load_economic_calendar(
        self,
        country: str = "US",
        days_ahead: int = 7,
        days_back: int = 7,
        impact_filter: Optional[List[str]] = None,
    ) -> List[EconomicEvent]:
        """
        Load economic calendar events.
        
        Args:
            country: Country code filter (e.g., "US")
            days_ahead: Number of days to look ahead
            days_back: Number of days to look back
            impact_filter: Filter by impact level (e.g., ["High", "Medium"])
            
        Returns:
            List of EconomicEvent objects
        """
        cal_file = self.results_dir / "economic-calendar.json"
        if not cal_file.exists():
            return []
        
        with open(cal_file, 'r') as f:
            data = json.load(f)
        
        now = datetime.now()
        start_date = now - timedelta(days=days_back)
        end_date = now + timedelta(days=days_ahead)
        
        if impact_filter is None:
            impact_filter = ["High", "Medium"]
        
        events = []
        for event in data:
            # Filter by country
            if event.get("country") != country:
                continue
            
            # Filter by impact
            impact = event.get("impact", "Low")
            if impact not in impact_filter:
                continue
            
            # Parse date
            try:
                date_str = event.get("date", "")
                event_date = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
            except (ValueError, TypeError):
                continue
            
            # Filter by date range
            if not (start_date <= event_date <= end_date):
                continue
            
            events.append(EconomicEvent(
                date=event_date,
                country=event.get("country", ""),
                event=event.get("event", ""),
                previous=event.get("previous"),
                estimate=event.get("estimate"),
                actual=event.get("actual"),
                impact=impact,
                unit=event.get("unit"),
            ))
        
        # Sort by date
        events.sort(key=lambda x: x.date)
        return events
    
    def get_market_snapshot(self) -> Dict[str, Any]:
        """
        Get a comprehensive market snapshot for LLM analysis.
        
        Returns structured data showing what markets are pricing.
        """
        indices = self.load_categorized_indices()
        
        snapshot = {
            "timestamp": datetime.now().isoformat(),
            "volatility": {},
            "us_equities": {},
            "treasury_yields": {},
            "global_equities": {},
            "risk_indicators": {},
        }
        
        # Volatility
        if "volatility" in indices:
            for q in indices["volatility"]:
                snapshot["volatility"][q.symbol] = {
                    "price": q.price,
                    "change": q.change,
                    "change_pct": q.change_pct,
                }
        
        # US Equities
        if "us_major" in indices:
            for q in indices["us_major"]:
                snapshot["us_equities"][q.symbol] = {
                    "price": q.price,
                    "change": q.change,
                    "change_pct": q.change_pct,
                }
        
        # Treasury Yields
        if "treasury" in indices:
            for q in indices["treasury"]:
                snapshot["treasury_yields"][q.symbol] = {
                    "yield": q.price,
                    "change_bp": q.change * 100,  # Convert to basis points
                }
        
        # Global Equities
        if "global" in indices:
            for q in indices["global"]:
                snapshot["global_equities"][q.symbol] = {
                    "price": q.price,
                    "change_pct": q.change_pct,
                }
        
        return snapshot
    
    def get_vix_term_structure(self) -> Dict[str, float]:
        """
        Get VIX term structure to understand volatility expectations.
        
        Returns:
            Dict with VIX at different tenors
        """
        indices = self.load_categorized_indices()
        
        vix_map = {
            "^VIX1D": "1D",
            "^VIX": "30D",
            "^VIX3M": "3M",
            "^VIX6M": "6M",
        }
        
        result = {}
        if "volatility" in indices:
            for q in indices["volatility"]:
                if q.symbol in vix_map:
                    result[vix_map[q.symbol]] = q.price
        
        return result
    
    def get_treasury_curve(self) -> Dict[str, float]:
        """
        Get Treasury yield curve.
        
        Returns:
            Dict mapping tenor to yield
        """
        indices = self.load_categorized_indices()
        
        tenor_map = {
            "^IRX": "3M",
            "^FVX": "5Y",
            "^TNX": "10Y",
            "^TYX": "30Y",
        }
        
        result = {}
        if "treasury" in indices:
            for q in indices["treasury"]:
                if q.symbol in tenor_map:
                    result[tenor_map[q.symbol]] = q.price
        
        return result

