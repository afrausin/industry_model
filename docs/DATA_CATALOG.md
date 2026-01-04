# Data Catalog

Complete inventory of all data sources, datasets, and economic indicators available in the Macro Data Framework.

---

## Overview

| Category | Sources | Series/Datasets | Update Range |
|----------|---------|-----------------|--------------|
| Quantitative Time Series | FRED, FMP | 40+ economic indicators | Daily to Quarterly |
| Fed Policy Documents | Federal Reserve, Atlanta Fed, NY Fed | 10+ document types | Per event |
| Economic Projections | CBO, Atlanta Fed | Budget & GDP forecasts | Weekly to Annual |
| Research & Analysis | Brookings, NBER, PIIE, IMF, OECD | Papers, reports, blogs | Ongoing |

---

## 1. FRED (Federal Reserve Economic Data)

**Source:** Federal Reserve Bank of St. Louis
**API:** https://api.stlouisfed.org/fred
**Requires:** `FRED_API_KEY` environment variable
**Scraper:** `data/ingest/scrapers/fred_scraper.py`

### Growth & Output (7 series)

| Series ID | Description | Frequency |
|-----------|-------------|-----------|
| `GDP` | Gross Domestic Product (Nominal) | Quarterly |
| `GDPC1` | Real Gross Domestic Product | Quarterly |
| `A191RL1Q225SBEA` | Real GDP Growth Rate | Quarterly |
| `INDPRO` | Industrial Production Index | Monthly |
| `PAYEMS` | Total Nonfarm Payrolls | Monthly |
| `RSAFS` | Retail Sales | Monthly |
| `DGORDER` | Durable Goods Orders | Monthly |

### Labor Market (6 series)

| Series ID | Description | Frequency |
|-----------|-------------|-----------|
| `PAYEMS` | Total Nonfarm Payrolls | Monthly |
| `UNRATE` | Unemployment Rate | Monthly |
| `CIVPART` | Labor Force Participation Rate | Monthly |
| `ICSA` | Initial Jobless Claims | Weekly |
| `CCSA` | Continued Jobless Claims | Weekly |
| `JTSJOL` | Job Openings (JOLTS) | Monthly |

### Inflation (6 series)

| Series ID | Description | Frequency |
|-----------|-------------|-----------|
| `CPIAUCSL` | CPI All Urban Consumers | Monthly |
| `CPILFESL` | Core CPI (ex Food & Energy) | Monthly |
| `PCEPI` | PCE Price Index | Monthly |
| `PCEPILFE` | Core PCE Price Index | Monthly |
| `MICH` | U of Michigan Inflation Expectations | Monthly |
| `T5YIFR` | 5-Year Breakeven Inflation Rate | Daily |

### Interest Rates (9 series)

| Series ID | Description | Frequency |
|-----------|-------------|-----------|
| `FEDFUNDS` | Federal Funds Effective Rate | Monthly |
| `DFF` | Federal Funds Rate (Daily) | Daily |
| `DGS2` | 2-Year Treasury Yield | Daily |
| `DGS10` | 10-Year Treasury Yield | Daily |
| `DGS30` | 30-Year Treasury Yield | Daily |
| `T10Y2Y` | 10-Year minus 2-Year Spread | Daily |
| `T10Y3M` | 10-Year minus 3-Month Spread | Daily |
| `DFII10` | 10-Year TIPS Yield (Real Rate) | Daily |
| `MORTGAGE30US` | 30-Year Fixed Mortgage Rate | Weekly |

### Financial Conditions (5 series)

| Series ID | Description | Frequency |
|-----------|-------------|-----------|
| `VIXCLS` | VIX Volatility Index | Daily |
| `SP500` | S&P 500 Index | Daily |
| `BAMLH0A0HYM2` | High Yield Credit Spread | Daily |
| `NFCI` | Chicago Fed Financial Conditions Index | Weekly |
| `DTWEXBGS` | Trade Weighted Dollar Index | Daily |

### Housing (3 series)

| Series ID | Description | Frequency |
|-----------|-------------|-----------|
| `HOUST` | Housing Starts | Monthly |
| `PERMIT` | Building Permits | Monthly |
| `CSUSHPISA` | Case-Shiller Home Price Index | Monthly |

### Money & Credit (4 series)

| Series ID | Description | Frequency |
|-----------|-------------|-----------|
| `M2SL` | M2 Money Supply | Monthly |
| `WALCL` | Fed Balance Sheet Total Assets | Weekly |
| `TOTRESNS` | Total Reserves | Monthly |
| `DPCREDIT` | Consumer Credit | Monthly |

### Consumer Sentiment (2 series)

| Series ID | Description | Frequency |
|-----------|-------------|-----------|
| `UMCSENT` | U of Michigan Consumer Sentiment | Monthly |
| `MICH` | U of Michigan Inflation Expectations | Monthly |

### Trade (2 series)

| Series ID | Description | Frequency |
|-----------|-------------|-----------|
| `BOPGSTB` | Trade Balance | Monthly |
| `DTWEXBGS` | Trade Weighted Dollar Index | Daily |

### Composite Bundles

Pre-configured series groups for common analysis:

| Bundle | Series Count | Contents |
|--------|--------------|----------|
| `macro_growth` | 4 | GDP, Industrial Production, Payrolls, Retail Sales |
| `macro_labor` | 6 | Unemployment, payrolls, participation, claims, JOLTS |
| `macro_inflation` | 6 | CPI, Core CPI, PCE, Core PCE, inflation expectations |
| `macro_rates` | 8 | Fed funds, Treasury yields, spreads, TIPS |
| `macro_financial` | 5 | M2, Fed balance sheet, spreads, VIX, NFCI |
| `macro_full` | 40+ | All series combined |

---

## 2. Federal Reserve Board of Governors

**Source:** https://www.federalreserve.gov
**Scraper:** `data/ingest/scrapers/fed_scraper.py`
**Rate Limit:** 2.0 seconds per request

### FOMC Policy Documents

| Dataset | File | Description | Frequency |
|---------|------|-------------|-----------|
| FOMC Statements | `fomc_statements.json` | Monetary policy decisions and forward guidance | 8x/year |
| FOMC Minutes | `fomc_minutes.json` | Detailed meeting discussions with extracted sections | 8x/year (3 weeks post-meeting) |
| Beige Book | `beige_book.json` | Regional economic conditions summary | 8x/year |
| Fed Speeches | `fed_speeches.json` | Speeches by Powell, Governors, and Regional Presidents | Ongoing |

**FOMC Minutes Sections Extracted:**
- Staff Review of the Economic Situation
- Staff Review of the Financial Situation
- Participants' Views on Current Conditions
- Committee Policy Action

### Economic Projections

| Dataset | File | Description | Frequency |
|---------|------|-------------|-----------|
| SEP Projections | `sep_projections.json` | Summary of Economic Projections (dot plot, GDP, inflation, unemployment forecasts) | 4x/year (March, June, Sept, Dec) |

### Statistical Releases

| Dataset | File | Description | Frequency |
|---------|------|-------------|-----------|
| H.8 Release | `h8_release.json` | Assets & Liabilities of Commercial Banks | Weekly |
| H.15 Release | `h15_release.json` | Selected Interest Rates | Daily/Weekly |
| FOMC Calendar | `fomc_calendar.json` | FOMC meeting schedule | Annual |

---

## 3. Atlanta Fed

**Source:** https://www.atlantafed.org
**Scraper:** `data/ingest/scrapers/atlanta_fed_scraper.py`
**Rate Limit:** 2.0 seconds per request

### GDPNow Nowcasting Model

Real-time GDP growth estimates updated after each major economic data release.

| Dataset | File | Description | Frequency |
|---------|------|-------------|-----------|
| Current Estimate | `gdpnow_current.json` | Latest real-time GDP growth forecast | Multiple times/month |
| Forecast History | `gdpnow_history.json` | Evolution of forecasts through the quarter | Quarterly archives |
| Component Breakdown | `gdpnow_components.json` | Detailed GDP components (consumption, investment, trade, government) | Multiple times/month |

**Excel Data Files:**
- `GDPNowcastDataReleaseDates.xlsx` - Schedule of data releases affecting the forecast
- `GDPTrackingModelDataAndForecasts.xlsx` - Full tracking model data

---

## 4. New York Fed

**Source:** https://www.newyorkfed.org
**Scraper:** `data/ingest/scrapers/ny_fed_scraper.py`
**Rate Limit:** 2.0 seconds per request

### Survey of Consumer Expectations (SCE)

| Dataset | File | Description | Frequency |
|---------|------|-------------|-----------|
| SCE Overview | `sce_overview.json` | Consumer inflation expectations (1-year, 3-year ahead) | Monthly |
| SCE Inflation | `sce_inflation.json` | Detailed inflation expectations data | Monthly |

### Economic Indicators

| Dataset | File | Description | Frequency |
|---------|------|-------------|-----------|
| Weekly Economic Index | `weekly_economic_index.json` | High-frequency real economic activity measure (scaled to 4Q GDP growth) | Weekly |

### Research

| Dataset | File | Description | Frequency |
|---------|------|-------------|-----------|
| Liberty Street Economics | `liberty_street_posts.json` | NY Fed economist blog posts on labor, inflation, financial conditions | Ongoing |

---

## 5. Congressional Budget Office (CBO)

**Source:** https://www.cbo.gov
**Scraper:** `data/ingest/scrapers/cbo_scraper.py`
**Rate Limit:** 3.0 seconds per request

### Projections

| Dataset | File | Description | Frequency |
|---------|------|-------------|-----------|
| Budget Projections | `budget_projections.json` | 10-year federal revenue, outlay, and deficit forecasts | 2x/year |
| Economic Projections | `economic_projections.json` | 10-year GDP, inflation, unemployment forecasts | 2x/year |
| Long-Term Outlook | `long_term_outlook.json` | 30-year debt-to-GDP projections | Annual |
| Historical Budget | `historical_budget.json` | Historical federal revenues, outlays, deficits, debt | Annual |

---

## 6. Brookings Institution

**Source:** https://www.brookings.edu
**Scraper:** `data/ingest/scrapers/brookings_scraper.py`
**Rate Limit:** 2.0 seconds per request

### Hutchins Center on Fiscal & Monetary Policy

| Dataset | File | Type | Description | Frequency |
|---------|------|------|-------------|-----------|
| Fiscal Impact Measure | `fim.json` | Quantitative | Measures fiscal policy contribution to GDP growth | Quarterly |
| Hutchins Research | `hutchins_research.json` | Qualitative | Fiscal and monetary policy research papers | Ongoing |

**FIM Interpretation:**
- Positive values = Fiscal stimulus (government adding to growth)
- Negative values = Fiscal drag (government subtracting from growth)

---

## 7. NBER (National Bureau of Economic Research)

**Source:** https://www.nber.org
**Scraper:** `data/ingest/scrapers/nber_scraper.py`
**Rate Limit:** 2.0 seconds per request

### Reference Data

| Dataset | File | Type | Description | Frequency |
|---------|------|------|-------------|-----------|
| Business Cycles | `business_cycles.json` | Reference | Official US recession dates (peaks/troughs) | Event-driven |

### Research

| Dataset | File | Type | Description | Frequency |
|---------|------|------|-------------|-----------|
| Working Papers | `working_papers.json` | Qualitative | Frontier economic research | Weekly |
| NBER Digest | `digest.json` | Qualitative | Accessible research summaries | Monthly |

---

## 8. Peterson Institute (PIIE)

**Source:** https://www.piie.com
**Scraper:** `data/ingest/scrapers/piie_scraper.py`
**Rate Limit:** 2.0 seconds per request

### Research

| Dataset | File | Type | Description | Frequency |
|---------|------|------|-------------|-----------|
| Trade Research | `trade_research.json` | Qualitative | Tariffs, trade agreements, trade wars analysis | Ongoing |
| PIIE Charts | `charts.json` | Quantitative | Data visualizations on global economics | Ongoing |
| RealTime Economics | `blog.json` | Qualitative | Commentary on current economic events | Daily |

---

## 9. International Monetary Fund (IMF)

**Source:** https://www.imf.org
**Scraper:** `data/ingest/scrapers/imf_scraper.py`
**Rate Limit:** 2.0 seconds per request

### Reports

| Dataset | File | Type | Description | Frequency |
|---------|------|------|-------------|-----------|
| US Article IV | `us_article4.json` | Qualitative | IMF annual assessment of US economy | Annual |
| World Economic Outlook | `weo.json` | Qualitative | Global economic forecasts and analysis | Semi-annual |
| Global Financial Stability | `gfsr.json` | Qualitative | Global financial system risk assessment | Semi-annual |

---

## 10. OECD

**Source:** https://www.oecd.org
**Scraper:** `data/ingest/scrapers/oecd_scraper.py`
**Rate Limit:** 2.0 seconds per request

### Reports & Indicators

| Dataset | File | Type | Description | Frequency |
|---------|------|------|-------------|-----------|
| Economic Outlook | `economic_outlook.json` | Qualitative | Global economic projections | Semi-annual |
| US Economic Survey | `us_survey.json` | Qualitative | OECD assessment of US economy | Biennial |
| Composite Leading Indicators | `cli.json` | Quantitative | Leading indicators for business cycle turning points | Monthly |

---

## 11. Financial Modeling Prep (FMP)

**Source:** https://financialmodelingprep.com
**API:** `data/ingest/fmp/`
**Requires:** `FMP_API_KEY` environment variable

### Market Data

| Endpoint | Description | Coverage |
|----------|-------------|----------|
| Batch Index Quotes | Major global indices | 196 indices |
| Batch Stock/ETF Quotes | Real-time prices | 87,638 stocks, 13,244 ETFs |
| Economic Calendar | Scheduled economic events | 147 countries, 8,844 events |

### Key Indices Tracked

| Symbol | Description |
|--------|-------------|
| `^GSPC` | S&P 500 |
| `^DJI` | Dow Jones Industrial Average |
| `^IXIC` | Nasdaq Composite |
| `^VIX` | CBOE Volatility Index |
| `^TNX` | 10-Year Treasury Yield |
| `^TYX` | 30-Year Treasury Yield |
| `^RUT` | Russell 2000 |

### Macro-Relevant ETFs

**Fixed Income:**
| Symbol | Description |
|--------|-------------|
| `TLT` | 20+ Year Treasury |
| `IEF` | 7-10 Year Treasury |
| `SHY` | 1-3 Year Treasury |
| `HYG` | High Yield Corporate |
| `LQD` | Investment Grade Corporate |
| `JNK` | High Yield (Junk) Bonds |

**Equity:**
| Symbol | Description |
|--------|-------------|
| `SPY` | S&P 500 ETF |
| `QQQ` | Nasdaq 100 ETF |
| `IWM` | Russell 2000 ETF |
| `DIA` | Dow Jones ETF |

**Sectors:**
| Symbol | Description |
|--------|-------------|
| `XLF` | Financials |
| `XLE` | Energy |
| `XLK` | Technology |
| `XLV` | Healthcare |
| `XLI` | Industrials |
| `XLP` | Consumer Staples |
| `XLU` | Utilities |

**Commodities:**
| Symbol | Description |
|--------|-------------|
| `GLD` | Gold |
| `SLV` | Silver |
| `USO` | Oil |
| `UNG` | Natural Gas |

**International:**
| Symbol | Description |
|--------|-------------|
| `EEM` | Emerging Markets |
| `EFA` | Developed Markets ex-US |
| `VWO` | Emerging Markets (Vanguard) |

**Volatility:**
| Symbol | Description |
|--------|-------------|
| `VIXY` | VIX Short-Term Futures |
| `VXX` | VIX Short-Term |
| `UVXY` | Ultra VIX Short-Term |

### Calculated Metrics

| Metric | Calculation | Use Case |
|--------|-------------|----------|
| HYG/TLT Ratio | High Yield vs Treasury | Credit spread proxy |
| LQD/TLT Ratio | IG Corporate vs Treasury | Investment grade spread proxy |

---

## Data Storage Structure

```
data/storage/raw/
├── federal_reserve/          # Fed documents, FOMC, Beige Book, speeches
├── atlanta_fed/              # GDPNow estimates and components
├── ny_fed/                   # SCE surveys, WEI, Liberty Street
├── fred/                     # Individual FRED series and bundles
├── cbo/                      # Budget and economic projections
├── brookings/                # FIM, Hutchins research
├── nber/                     # Business cycles, working papers
├── piie/                     # Trade research, charts, blog
├── imf/                      # Article IV, WEO, GFSR
├── oecd/                     # Economic outlook, surveys, CLI
└── archive/                  # Timestamped versions (YYYYMMDD_HHMMSS)
```

---

## Data Characteristics Summary

| Aspect | Details |
|--------|---------|
| **Total Scrapers** | 10 official sources + FMP |
| **FRED Time Series** | 40+ economic indicators |
| **Qualitative Documents** | 15+ document types (statements, speeches, research) |
| **Update Frequency** | Daily (rates) → Weekly (claims) → Monthly (employment) → Quarterly (GDP) |
| **Historical Depth** | 30+ years (FRED), 10+ years (Fed docs), 5+ years (research) |
| **Archive Strategy** | Timestamped snapshots for all downloads |
| **Incremental Updates** | Most scrapers track existing content |

---

## API Keys Required

| Service | Environment Variable | Purpose | Required |
|---------|---------------------|---------|----------|
| FRED | `FRED_API_KEY` | Economic time series | Yes |
| FMP | `FMP_API_KEY` | Market data, economic calendar | Optional |
| Gemini | `GOOGLE_API_KEY` | AI document analysis | Optional |

**Get API Keys:**
- FRED: https://fred.stlouisfed.org/docs/api/api_key.html
- FMP: https://financialmodelingprep.com/
- Gemini: https://ai.google.dev/

---

## Usage Examples

### Load FRED Series
```python
from data import MacroDataLoader, MacroConfig

config = MacroConfig()
loader = MacroDataLoader(config.data_dir)

# Single series
gdp = loader.load_fred_series("GDPC1")

# Bundle
inflation_data = loader.load_fred_bundle("macro_inflation")
```

### Load Fed Documents
```python
from data import MacroDataLoader

loader = MacroDataLoader(data_dir)
docs = loader.load_all_qualitative_docs()

# Access specific document types
fomc_statements = docs.get("fomc_statements", [])
beige_book = docs.get("beige_book", [])
```

### Point-in-Time Query
```python
from data.storage.db import PITDataLoader

pit_loader = PITDataLoader("data/storage/db/pit_database.db")

# Get data as it was known on a specific date
historical_view = pit_loader.get_as_of("GDPC1", as_of_date="2023-06-30")
```
