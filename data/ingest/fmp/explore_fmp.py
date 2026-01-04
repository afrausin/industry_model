"""
Financial Modeling Prep API Explorer
====================================
Explores available endpoints and data from FMP's stable API.
"""

import os
import requests
import json
from pathlib import Path

# Load API key
API_KEY = os.environ.get('FMP_API_KEY')
if not API_KEY:
    try:
        from dotenv import load_dotenv
        load_dotenv()
        API_KEY = os.environ.get('FMP_API_KEY')
    except ImportError:
        pass

if not API_KEY:
    raise ValueError("FMP_API_KEY not found. Set it as environment variable.")

BASE_URL = "https://financialmodelingprep.com/stable"
OUTPUT_DIR = Path(__file__).parent / "exploration_results"
OUTPUT_DIR.mkdir(exist_ok=True)


def fmp_get(endpoint, params=None):
    """Make FMP API request."""
    url = f"{BASE_URL}/{endpoint}"
    params = params or {}
    params['apikey'] = API_KEY
    
    response = requests.get(url, params=params, timeout=30)
    
    if response.status_code == 200:
        data = response.json()
        count = len(data) if isinstance(data, list) else 'dict'
        print(f"  ✅ {endpoint}: {count} items")
        return data
    else:
        print(f"  ❌ {endpoint}: {response.status_code}")
        return None


def save_result(name, data):
    """Save result to JSON file."""
    path = OUTPUT_DIR / f"{name}.json"
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"     💾 Saved to {path.name}")


def explore_all_endpoints():
    """Try all known FMP stable endpoints."""
    
    endpoints = {
        # Index data
        "batch-index-quotes": "Real-time index quotes",
        "batch_index_quotes": "Real-time index quotes (underscore)",
        
        # Symbol lists
        "available-indexes": "Available indices list",
        "available_indexes": "Available indices (underscore)",
        "symbol/available-indexes": "Symbol/available indices",
        "available-commodities": "Commodities list",
        "available-forex-currency-pairs": "Forex pairs",
        "available-etfs": "ETF list",
        "available-cryptocurrencies": "Crypto list",
        "stock-list": "Stock list",
        "etf-list": "ETF list alt",
        
        # Economic data
        "economic-calendar": "Economic calendar",
        "economic_calendar": "Economic calendar (underscore)",
        "treasury": "Treasury rates",
        "treasury-rate": "Treasury rate",
        
        # Market data
        "sector-performance": "Sector performance",
        "sector_performance": "Sector performance (underscore)",
        "market-hours": "Market hours",
        "market-cap": "Market cap",
        "gainers": "Top gainers",
        "losers": "Top losers",
        "actives": "Most active",
        
        # Quotes
        "quote/%5EVIX": "VIX quote",
        "quote/%5EGSPC": "S&P 500 quote",
        "quotes/index": "All index quotes",
        "full-quote/%5EVIX": "VIX full quote",
        
        # Historical
        "historical-price-eod/%5EVIX": "VIX historical EOD",
        "historical-price-full/%5EVIX": "VIX historical full",
        "historical-chart/1day/%5EVIX": "VIX 1-day chart",
        
        # Fear & Greed
        "fear-greed-index": "Fear & Greed Index",
        
        # Commodities
        "commodities-quotes": "Commodity quotes",
        "commodity-quotes": "Commodity quotes alt",
        
        # Forex
        "forex-quotes": "Forex quotes",
        "fx-quotes": "FX quotes",
        
        # Crypto
        "crypto-quotes": "Crypto quotes",
        "cryptocurrency-quotes": "Cryptocurrency quotes",
    }
    
    results = {"working": [], "not_working": []}
    
    print("\n" + "="*60)
    print("🔍 EXPLORING FMP STABLE API ENDPOINTS")
    print("="*60 + "\n")
    
    for endpoint, description in endpoints.items():
        data = fmp_get(endpoint)
        if data and (isinstance(data, list) and len(data) > 0 or isinstance(data, dict) and data):
            results["working"].append({
                "endpoint": endpoint,
                "description": description,
                "sample_count": len(data) if isinstance(data, list) else 1
            })
            # Save successful results
            safe_name = endpoint.replace("/", "_").replace("%5E", "caret_")
            save_result(safe_name, data)
        else:
            results["not_working"].append(endpoint)
    
    return results


def explore_index_quotes_detail():
    """Deep dive into batch-index-quotes data."""
    print("\n" + "="*60)
    print("📊 DETAILED INDEX QUOTES ANALYSIS")
    print("="*60 + "\n")
    
    data = fmp_get("batch-index-quotes")
    if not data:
        return None
    
    # Analyze the data
    print(f"\nTotal indices: {len(data)}")
    print(f"Fields: {list(data[0].keys()) if data else 'N/A'}")
    
    # Find interesting indices
    keywords = {
        "volatility": ["vix", "vol", "fear"],
        "us_major": ["gspc", "dji", "ixic", "ndx", "rut"],
        "treasury": ["tnx", "tyx", "irx", "fvx"],
        "global": ["ftse", "dax", "nikkei", "hang", "stoxx"],
        "sector": ["xlf", "xle", "xlk", "xlv"],
    }
    
    categorized = {k: [] for k in keywords}
    categorized["other"] = []
    
    for item in data:
        symbol = item.get("symbol", "").lower()
        found = False
        for category, terms in keywords.items():
            if any(term in symbol for term in terms):
                categorized[category].append(item)
                found = True
                break
        if not found:
            categorized["other"].append(item)
    
    print("\n📌 CATEGORIZED INDICES:")
    for category, items in categorized.items():
        if items and category != "other":
            print(f"\n  {category.upper()} ({len(items)} indices):")
            for item in items[:10]:  # Show first 10
                print(f"    • {item.get('symbol'):12s} | Price: {item.get('price', 'N/A'):>12} | Change: {item.get('change', 'N/A'):>10}")
    
    print(f"\n  OTHER: {len(categorized['other'])} indices")
    
    # Save categorized results
    save_result("indices_categorized", categorized)
    
    return categorized


def explore_economic_calendar():
    """Explore economic calendar data."""
    print("\n" + "="*60)
    print("📅 ECONOMIC CALENDAR ANALYSIS")
    print("="*60 + "\n")
    
    data = fmp_get("economic-calendar")
    if not data:
        return None
    
    print(f"Total events: {len(data)}")
    print(f"Fields: {list(data[0].keys()) if data else 'N/A'}")
    
    # Sample events
    print("\n📌 SAMPLE EVENTS (first 10):")
    for event in data[:10]:
        print(f"  • {event.get('date', 'N/A')[:10]} | {event.get('country', 'N/A'):3s} | {event.get('event', 'N/A')[:50]}")
    
    # Unique countries
    countries = set(e.get('country') for e in data if e.get('country'))
    print(f"\n📌 Countries covered: {len(countries)}")
    print(f"   {sorted(countries)}")
    
    # Unique event types
    events = set(e.get('event') for e in data if e.get('event'))
    print(f"\n📌 Unique event types: {len(events)}")
    
    return data


def main():
    print(f"\n🔑 Using API key ending in ...{API_KEY[-4:]}")
    print(f"📁 Results will be saved to: {OUTPUT_DIR}")
    
    # Explore all endpoints
    results = explore_all_endpoints()
    
    # Summary
    print("\n" + "="*60)
    print("📋 ENDPOINT DISCOVERY SUMMARY")
    print("="*60)
    
    print(f"\n✅ WORKING ENDPOINTS ({len(results['working'])}):")
    for item in results['working']:
        print(f"   • {item['endpoint']}: {item['sample_count']} items - {item['description']}")
    
    print(f"\n❌ NOT AVAILABLE ({len(results['not_working'])}):")
    for ep in results['not_working'][:10]:
        print(f"   • {ep}")
    if len(results['not_working']) > 10:
        print(f"   ... and {len(results['not_working']) - 10} more")
    
    # Deep dive into working endpoints
    explore_index_quotes_detail()
    explore_economic_calendar()
    
    # Save summary
    save_result("endpoint_summary", results)
    
    print("\n" + "="*60)
    print("✅ EXPLORATION COMPLETE")
    print(f"📁 Results saved to: {OUTPUT_DIR}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()



