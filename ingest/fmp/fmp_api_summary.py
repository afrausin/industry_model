"""
FMP API Summary - Available Endpoints and Data
===============================================
Based on exploration with your API key.
"""

import os
import requests
import json
from pathlib import Path
from datetime import datetime

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
    raise ValueError("FMP_API_KEY not found")

BASE_URL = "https://financialmodelingprep.com/stable"
OUTPUT_DIR = Path(__file__).parent / "exploration_results"


def fmp_get(endpoint, params=None):
    """Make FMP API request."""
    url = f"{BASE_URL}/{endpoint}"
    params = params or {}
    params['apikey'] = API_KEY
    response = requests.get(url, params=params, timeout=30)
    if response.status_code == 200:
        return response.json()
    return None


def get_all_available_data():
    """Fetch all available data from working endpoints."""
    
    print("\n" + "="*70)
    print("📊 FMP API - COMPLETE DATA SUMMARY")
    print("="*70)
    print(f"\n⏰ Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = {}
    
    # 1. Index Quotes
    print("\n" + "-"*50)
    print("1️⃣  INDEX QUOTES (batch-index-quotes)")
    print("-"*50)
    
    index_data = fmp_get("batch-index-quotes")
    if index_data:
        results['indices'] = index_data
        print(f"   Total: {len(index_data)} indices")
        
        # Key indices
        key_symbols = ['^VIX', '^GSPC', '^DJI', '^IXIC', '^TNX', '^TYX', '^RUT']
        print("\n   📌 Key Indices:")
        for item in index_data:
            if item['symbol'] in key_symbols:
                print(f"      {item['symbol']:10s} = {item['price']:>12} (chg: {item['change']:+.2f})")
    
    # 2. Stock/ETF Quotes
    print("\n" + "-"*50)
    print("2️⃣  STOCK/ETF QUOTES (batch-quote)")
    print("-"*50)
    
    # Get credit/macro relevant ETFs
    macro_etfs = [
        # Credit/Bonds
        'HYG', 'LQD', 'JNK', 'TLT', 'IEF', 'SHY', 'VCIT', 'VCSH',
        # Equity
        'SPY', 'QQQ', 'IWM', 'DIA',
        # Sectors
        'XLF', 'XLE', 'XLK', 'XLV', 'XLI', 'XLP', 'XLU',
        # Commodities
        'GLD', 'SLV', 'USO', 'UNG',
        # International
        'EEM', 'EFA', 'VWO',
        # Volatility
        'VIXY', 'VXX', 'UVXY',
    ]
    
    etf_data = fmp_get(f"batch-quote?symbols={','.join(macro_etfs)}")
    if etf_data:
        results['etfs'] = etf_data
        print(f"   Total: {len(etf_data)} ETFs")
        
        print("\n   📌 Credit ETFs (for spread calculation):")
        for item in etf_data:
            if item['symbol'] in ['HYG', 'LQD', 'JNK', 'TLT', 'IEF']:
                print(f"      {item['symbol']:6s} = ${item['price']:>8.2f} (chg: {item['changePercentage']:+.2f}%)")
        
        print("\n   📌 Equity ETFs:")
        for item in etf_data:
            if item['symbol'] in ['SPY', 'QQQ', 'IWM', 'DIA']:
                print(f"      {item['symbol']:6s} = ${item['price']:>8.2f} (chg: {item['changePercentage']:+.2f}%)")
        
        print("\n   📌 Volatility ETFs:")
        for item in etf_data:
            if item['symbol'] in ['VIXY', 'VXX', 'UVXY']:
                print(f"      {item['symbol']:6s} = ${item['price']:>8.2f} (chg: {item['changePercentage']:+.2f}%)")
    
    # 3. Economic Calendar
    print("\n" + "-"*50)
    print("3️⃣  ECONOMIC CALENDAR (economic-calendar)")
    print("-"*50)
    
    econ_data = fmp_get("economic-calendar")
    if econ_data:
        results['economic_calendar'] = econ_data
        print(f"   Total: {len(econ_data)} events")
        
        # Filter to US events
        us_events = [e for e in econ_data if e.get('country') == 'US']
        print(f"   US Events: {len(us_events)}")
        
        print("\n   📌 Upcoming US Events:")
        for event in us_events[:10]:
            print(f"      {event['date'][:10]} | {event['event'][:50]}")
    
    # 4. Stock List
    print("\n" + "-"*50)
    print("4️⃣  STOCK LIST (stock-list)")
    print("-"*50)
    
    stock_data = fmp_get("stock-list")
    if stock_data:
        results['stocks'] = stock_data
        print(f"   Total: {len(stock_data)} stocks")
        
        # Count by exchange
        exchanges = {}
        for s in stock_data:
            ex = s.get('exchangeShortName', 'Unknown')
            exchanges[ex] = exchanges.get(ex, 0) + 1
        
        print("\n   📌 By Exchange:")
        for ex, count in sorted(exchanges.items(), key=lambda x: -x[1])[:10]:
            print(f"      {ex}: {count}")
    
    # 5. ETF List
    print("\n" + "-"*50)
    print("5️⃣  ETF LIST (etf-list)")
    print("-"*50)
    
    etf_list = fmp_get("etf-list")
    if etf_list:
        results['etf_list'] = etf_list
        print(f"   Total: {len(etf_list)} ETFs")
    
    # Summary
    print("\n" + "="*70)
    print("📋 SUMMARY - WORKING FMP ENDPOINTS")
    print("="*70)
    
    print("""
    ┌─────────────────────────────────────────────────────────────────┐
    │ ENDPOINT                        │ DATA                          │
    ├─────────────────────────────────┼───────────────────────────────┤
    │ batch-index-quotes              │ 196 global indices (VIX, etc) │
    │ batch-quote?symbols=X,Y,Z       │ Any stock/ETF real-time quote │
    │ economic-calendar               │ 8,844 economic events         │
    │ stock-list                      │ 87,638 stocks                 │
    │ etf-list                        │ 13,244 ETFs                   │
    └─────────────────────────────────┴───────────────────────────────┘
    """)
    
    print("""
    📌 KEY MACRO DATA AVAILABLE:
    
    VOLATILITY:
      • VIX Index (^VIX) - fear gauge
      • VIX Term Structure (^VIX1D, ^VIX3M, ^VIX6M)
      • VVIX - volatility of VIX
      • Volatility ETFs (VIXY, VXX, UVXY)
    
    TREASURY YIELDS:
      • 3-Month (^IRX)
      • 5-Year (^FVX)
      • 10-Year (^TNX)
      • 30-Year (^TYX)
    
    CREDIT SPREADS (via ETFs):
      • High Yield: HYG, JNK
      • Investment Grade: LQD, VCIT, VCSH
      • Treasury: TLT (20+Y), IEF (7-10Y), SHY (1-3Y)
      • Spread = HYG - TLT (approximate credit spread)
    
    ECONOMIC EVENTS:
      • 147 countries covered
      • Fed decisions, GDP, CPI, Employment, etc.
    
    ⚠️  NOT AVAILABLE (need higher tier):
      • Historical price data
      • Treasury rates endpoint
      • Sector performance
      • Fear & Greed Index
    """)
    
    return results


def calculate_credit_spreads(etf_data):
    """Calculate approximate credit spreads from ETF prices."""
    
    print("\n" + "-"*50)
    print("📊 CREDIT SPREAD PROXIES (ETF-based)")
    print("-"*50)
    
    # Get prices
    prices = {e['symbol']: e for e in etf_data}
    
    if 'HYG' in prices and 'TLT' in prices:
        # Use year-over-year returns as proxy for spread changes
        hyg = prices['HYG']
        tlt = prices['TLT']
        
        # Simple price ratio as spread indicator
        spread_ratio = hyg['price'] / tlt['price']
        
        print(f"\n   HYG (High Yield): ${hyg['price']:.2f} (YTD: {hyg['changePercentage']:+.2f}%)")
        print(f"   TLT (Treasury):   ${tlt['price']:.2f} (YTD: {tlt['changePercentage']:+.2f}%)")
        print(f"   HYG/TLT Ratio:    {spread_ratio:.4f}")
        print(f"   (Higher ratio = tighter spreads, lower credit risk)")
    
    if 'LQD' in prices and 'TLT' in prices:
        lqd = prices['LQD']
        tlt = prices['TLT']
        spread_ratio = lqd['price'] / tlt['price']
        
        print(f"\n   LQD (IG Corp):    ${lqd['price']:.2f} (YTD: {lqd['changePercentage']:+.2f}%)")
        print(f"   LQD/TLT Ratio:    {spread_ratio:.4f}")


if __name__ == "__main__":
    data = get_all_available_data()
    
    if 'etfs' in data:
        calculate_credit_spreads(data['etfs'])
    
    # Save summary
    summary_path = OUTPUT_DIR / "full_data_summary.json"
    with open(summary_path, 'w') as f:
        json.dump({k: v[:100] if isinstance(v, list) and len(v) > 100 else v 
                   for k, v in data.items()}, f, indent=2)
    print(f"\n💾 Summary saved to: {summary_path}")



