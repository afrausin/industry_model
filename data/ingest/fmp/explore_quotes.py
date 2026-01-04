"""
Explore FMP quote endpoints for stocks and ETFs
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


def fmp_get(endpoint, params=None):
    """Make FMP API request."""
    url = f"{BASE_URL}/{endpoint}"
    params = params or {}
    params['apikey'] = API_KEY
    
    response = requests.get(url, params=params, timeout=30)
    
    if response.status_code == 200:
        data = response.json()
        return data
    else:
        print(f"  ❌ {endpoint}: {response.status_code}")
        return None


def explore_quote_endpoints():
    """Try different quote endpoint patterns."""
    
    # Credit/bond ETFs we want
    test_symbols = ['HYG', 'LQD', 'JNK', 'TLT', 'IEF', 'SPY', 'AAPL']
    
    # Endpoint patterns to try
    patterns = [
        "quote/{symbol}",
        "quotes/{symbol}",
        "stock-quote/{symbol}",
        "stock/quote/{symbol}",
        "real-time-price/{symbol}",
        "batch-request-end-of-day-prices/{symbol}",
        "historical-price-eod/{symbol}",
        "eod/{symbol}",
        "full-quote/{symbol}",
        "simple-quote/{symbol}",
    ]
    
    print("\n" + "="*60)
    print("🔍 EXPLORING QUOTE ENDPOINTS")
    print("="*60)
    
    working = []
    
    for pattern in patterns:
        print(f"\n📡 Testing pattern: {pattern}")
        for symbol in test_symbols[:2]:  # Just test first 2
            endpoint = pattern.format(symbol=symbol)
            data = fmp_get(endpoint)
            if data and (isinstance(data, list) and len(data) > 0 or isinstance(data, dict) and data):
                print(f"  ✅ {symbol}: {data if isinstance(data, dict) else data[:1]}")
                working.append({"pattern": pattern, "symbol": symbol, "sample": data[:1] if isinstance(data, list) else data})
                break
    
    return working


def explore_batch_endpoints():
    """Try batch quote endpoints."""
    
    symbols = "HYG,LQD,JNK,TLT,IEF,SPY"
    
    endpoints = [
        f"batch-quote?symbols={symbols}",
        f"batch-request-end-of-day-prices?symbols={symbols}",
        f"quote?symbols={symbols}",
        f"quotes?symbols={symbols}",
        f"stock/quote?symbols={symbols}",
    ]
    
    print("\n" + "="*60)
    print("🔍 EXPLORING BATCH QUOTE ENDPOINTS")
    print("="*60)
    
    for endpoint in endpoints:
        print(f"\n📡 Testing: {endpoint}")
        data = fmp_get(endpoint)
        if data and (isinstance(data, list) and len(data) > 0 or isinstance(data, dict) and data):
            print(f"  ✅ SUCCESS: {len(data) if isinstance(data, list) else 'dict'} items")
            if isinstance(data, list):
                for item in data[:3]:
                    print(f"    • {item}")
            else:
                print(f"    • {data}")
            return {"endpoint": endpoint, "data": data}
    
    return None


def explore_historical():
    """Try historical price endpoints."""
    
    symbol = "HYG"
    
    endpoints = [
        f"historical-price-eod/{symbol}",
        f"historical-price-full/{symbol}",
        f"historical-chart/1day/{symbol}",
        f"historical/eod/{symbol}",
        f"eod/{symbol}",
    ]
    
    print("\n" + "="*60)
    print("🔍 EXPLORING HISTORICAL ENDPOINTS")
    print("="*60)
    
    for endpoint in endpoints:
        print(f"\n📡 Testing: {endpoint}")
        data = fmp_get(endpoint)
        if data:
            if isinstance(data, dict) and 'historical' in data:
                print(f"  ✅ SUCCESS: {len(data['historical'])} historical records")
                print(f"    Sample: {data['historical'][:2]}")
                return {"endpoint": endpoint, "data": data}
            elif isinstance(data, list) and len(data) > 0:
                print(f"  ✅ SUCCESS: {len(data)} records")
                print(f"    Sample: {data[:2]}")
                return {"endpoint": endpoint, "data": data}
    
    return None


def check_etf_list_for_credit():
    """Check if credit ETFs are in the etf-list."""
    
    print("\n" + "="*60)
    print("🔍 CHECKING ETF LIST FOR CREDIT ETFS")
    print("="*60)
    
    etf_list_path = OUTPUT_DIR / "etf-list.json"
    if etf_list_path.exists():
        with open(etf_list_path) as f:
            etf_list = json.load(f)
        
        credit_etfs = ['HYG', 'LQD', 'JNK', 'TLT', 'IEF', 'SHY', 'VCIT', 'VCSH']
        
        print(f"\nTotal ETFs in list: {len(etf_list)}")
        print("\nSearching for credit/bond ETFs...")
        
        found = []
        for etf in etf_list:
            symbol = etf.get('symbol', '')
            if symbol in credit_etfs:
                found.append(etf)
                print(f"  ✅ Found: {symbol} - {etf.get('name', 'N/A')}")
        
        if not found:
            # Search by name
            print("\n  Searching by name...")
            for etf in etf_list:
                name = etf.get('name', '').lower()
                if any(kw in name for kw in ['high yield', 'corporate bond', 'treasury', 'credit']):
                    print(f"  • {etf.get('symbol')}: {etf.get('name')}")
        
        return found
    else:
        print("ETF list not found. Run explore_fmp.py first.")
        return None


def main():
    print(f"\n🔑 Using API key ending in ...{API_KEY[-4:]}")
    
    # Check ETF list
    check_etf_list_for_credit()
    
    # Try quote endpoints
    quote_results = explore_quote_endpoints()
    
    # Try batch endpoints
    batch_results = explore_batch_endpoints()
    
    # Try historical endpoints
    hist_results = explore_historical()
    
    print("\n" + "="*60)
    print("📋 QUOTE ENDPOINT SUMMARY")
    print("="*60)
    
    if quote_results:
        print("\n✅ Working quote patterns:")
        for r in quote_results:
            print(f"  • {r['pattern']}")
    else:
        print("\n❌ No individual quote endpoints working")
    
    if batch_results:
        print(f"\n✅ Working batch endpoint: {batch_results['endpoint']}")
    else:
        print("\n❌ No batch quote endpoints working")
    
    if hist_results:
        print(f"\n✅ Working historical endpoint: {hist_results['endpoint']}")
    else:
        print("\n❌ No historical endpoints working")


if __name__ == "__main__":
    main()

