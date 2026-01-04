# Macro Data Pipeline - Dataset Documentation

This document provides a comprehensive reference of all productionized datasets and time series available in this macroeconomic data collection system.

---

## Table of Contents

1. [Federal Reserve (FRED)](#federal-reserve-fred)
2. [Federal Reserve Board of Governors](#federal-reserve-board-of-governors)
3. [Bureau of Economic Analysis (BEA)](#bureau-of-economic-analysis-bea)
4. [Bureau of Labor Statistics (BLS)](#bureau-of-labor-statistics-bls)
5. [Atlanta Fed](#atlanta-fed)
6. [New York Fed](#new-york-fed)
7. [Brookings Institution](#brookings-institution)
8. [NBER](#national-bureau-of-economic-research-nber)
9. [PIIE](#peterson-institute-for-international-economics-piie)

---

## Federal Reserve (FRED)

**Source:** Federal Reserve Bank of St. Louis  
**API:** https://api.stlouisfed.org/fred  
**Data Directory:** `data/fred/`

### Individual Time Series

#### Output and Growth
| Series ID | Description | Frequency | File |
|-----------|-------------|-----------|------|
| `GDP` | Gross Domestic Product (Nominal) | Quarterly | `series_gdp.json` |
| `GDPC1` | Real Gross Domestic Product | Quarterly | `series_gdpc1.json` |
| `A191RL1Q225SBEA` | Real GDP Growth Rate | Quarterly | `series_a191rl1q225sbea.json` |
| `INDPRO` | Industrial Production Index | Monthly | `series_indpro.json` |

#### Labor Market
| Series ID | Description | Frequency | File |
|-----------|-------------|-----------|------|
| `UNRATE` | Unemployment Rate | Monthly | `series_unrate.json` |
| `PAYEMS` | Total Nonfarm Payrolls | Monthly | `series_payems.json` |
| `CIVPART` | Labor Force Participation Rate | Monthly | `series_civpart.json` |
| `ICSA` | Initial Jobless Claims | Weekly | `series_icsa.json` |
| `CCSA` | Continued Jobless Claims | Weekly | `series_ccsa.json` |
| `JTSJOL` | Job Openings (JOLTS) | Monthly | `series_jtsjol.json` |

#### Inflation
| Series ID | Description | Frequency | File |
|-----------|-------------|-----------|------|
| `CPIAUCSL` | Consumer Price Index (All Urban) | Monthly | `series_cpiaucsl.json` |
| `CPILFESL` | Core CPI (Less Food and Energy) | Monthly | `series_cpilfesl.json` |
| `PCEPI` | PCE Price Index | Monthly | `series_pcepi.json` |
| `PCEPILFE` | Core PCE Price Index | Monthly | `series_pcepilfe.json` |
| `MICH` | University of Michigan Inflation Expectations | Monthly | `series_mich.json` |
| `T5YIFR` | 5-Year Breakeven Inflation Rate | Daily | `series_t5yifr.json` |

#### Interest Rates
| Series ID | Description | Frequency | File |
|-----------|-------------|-----------|------|
| `FEDFUNDS` | Federal Funds Effective Rate | Monthly | `series_fedfunds.json` |
| `DFF` | Federal Funds Rate | Daily | `series_dff.json` |
| `DGS2` | 2-Year Treasury Yield | Daily | `series_dgs2.json` |
| `DGS10` | 10-Year Treasury Yield | Daily | `series_dgs10.json` |
| `DGS30` | 30-Year Treasury Yield | Daily | `series_dgs30.json` |
| `T10Y2Y` | 10-Year Minus 2-Year Treasury (Yield Curve) | Daily | `series_t10y2y.json` |
| `T10Y3M` | 10-Year Minus 3-Month Treasury | Daily | `series_t10y3m.json` |
| `DFII10` | 10-Year TIPS Yield (Real Rate) | Daily | `series_dfii10.json` |

#### Money and Credit
| Series ID | Description | Frequency | File |
|-----------|-------------|-----------|------|
| `M2SL` | M2 Money Stock | Monthly | `series_m2sl.json` |
| `WALCL` | Fed Balance Sheet (Total Assets) | Weekly | `series_walcl.json` |
| `TOTRESNS` | Total Reserves | Monthly | `series_totresns.json` |
| `DPCREDIT` | Consumer Credit | Monthly | `series_dpcredit.json` |

#### Housing
| Series ID | Description | Frequency | File |
|-----------|-------------|-----------|------|
| `HOUST` | Housing Starts | Monthly | `series_houst.json` |
| `PERMIT` | Building Permits | Monthly | `series_permit.json` |
| `CSUSHPISA` | S&P Case-Shiller Home Price Index | Monthly | `series_csushpisa.json` |
| `MORTGAGE30US` | 30-Year Fixed Mortgage Rate | Weekly | `series_mortgage30us.json` |

#### Financial Conditions
| Series ID | Description | Frequency | File |
|-----------|-------------|-----------|------|
| `BAMLH0A0HYM2` | High Yield Corporate Bond Spread | Daily | `series_bamlh0a0hym2.json` |
| `VIXCLS` | VIX Volatility Index | Daily | `series_vixcls.json` |
| `SP500` | S&P 500 Index | Daily | `series_sp500.json` |
| `NFCI` | Chicago Fed National Financial Conditions Index | Weekly | `series_nfci.json` |

#### Trade & Sentiment
| Series ID | Description | Frequency | File |
|-----------|-------------|-----------|------|
| `BOPGSTB` | Trade Balance | Monthly | `series_bopgstb.json` |
| `DTWEXBGS` | Trade Weighted US Dollar Index | Daily | `series_dtwexbgs.json` |
| `UMCSENT` | University of Michigan Consumer Sentiment | Monthly | `series_umcsent.json` |
| `RSAFS` | Retail Sales | Monthly | `series_rsafs.json` |
| `DGORDER` | Durable Goods Orders | Monthly | `series_dgorder.json` |

### Bundled Datasets

| Bundle Name | Description | Series Count | File |
|-------------|-------------|--------------|------|
| `macro_growth` | Key growth and output indicators | 4 | `macro_growth.json` |
| `macro_labor` | Labor market indicators | 6 | `macro_labor.json` |
| `macro_inflation` | Inflation indicators | 6 | `macro_inflation.json` |
| `macro_rates` | Interest rates and yield curve | 8 | `macro_rates.json` |
| `macro_financial` | Financial conditions and credit | 5 | `macro_financial.json` |
| `macro_full` | All key macroeconomic series | 41 | `macro_full.json` |

---

## Federal Reserve Board of Governors

**Source:** Federal Reserve Board  
**URL:** https://www.federalreserve.gov  
**Data Directory:** `data/federal_reserve/`

### Qualitative / Policy Documents

| Dataset | Description | Frequency | File |
|---------|-------------|-----------|------|
| **FOMC Statements** | Federal Open Market Committee policy statements with full text | 8x/year | `fomc_statements.json` |
| **FOMC Minutes** | Detailed meeting minutes with policy deliberations, includes parsed sections (Staff Review, Participants' Views, Committee Policy Action) | 8x/year (3 weeks post-meeting) | `fomc_minutes.json` |
| **Fed Speeches** | Speeches by Fed officials (Chair Powell, Governors, Regional Presidents) with full content | Ongoing | `fed_speeches.json` |
| **Beige Book** | Summary of Commentary on Current Economic Conditions by Federal Reserve District | 8x/year | `beige_book.json` |
| **Beige Book (Latest)** | Most recent Beige Book edition | Event-driven | `beige_book_YYYYMM.json` |

### Economic Projections

| Dataset | Description | Frequency | File |
|---------|-------------|-----------|------|
| **Summary of Economic Projections (SEP)** | FOMC participants' economic projections ("dot plot") including GDP, unemployment, inflation, and fed funds rate forecasts | 4x/year (March, June, Sept, Dec) | `sep_projections.json` |

### Statistical Releases

| Dataset | Description | Frequency | File |
|---------|-------------|-----------|------|
| **H.8 Release** | Assets and Liabilities of Commercial Banks in the U.S. | Weekly | `h8_release.json` |
| **H.15 Release** | Selected Interest Rates (Treasury yields, money market rates) | Daily/Weekly | `h15_release.json` |
| **FOMC Calendar** | Schedule of upcoming FOMC meetings | Annual | `fomc_calendar.json` |

---

## Bureau of Economic Analysis (BEA)

**Source:** U.S. Department of Commerce, BEA  
**API:** https://apps.bea.gov/api  
**Data Directory:** `data/bea/`

### NIPA Tables (National Income and Product Accounts)

#### GDP and Components
| Table ID | Description | Frequency | File |
|----------|-------------|-----------|------|
| `T10101` | Percent Change From Preceding Period in Real GDP | Quarterly | `nipa_t10101.json` |
| `T10102` | Contributions to Percent Change in Real GDP | Quarterly | `nipa_t10102.json` |
| `T10105` | GDP (Nominal) | Quarterly | `nipa_t10105.json` |
| `T10106` | Real GDP | Quarterly | `nipa_t10106.json` |
| `T10107` | GDP Chain-type Price Index | Quarterly | `nipa_t10107.json` |

#### Gross Domestic Income
| Table ID | Description | Frequency | File |
|----------|-------------|-----------|------|
| `T11000` | Gross Domestic Income by Type of Income | Quarterly | `nipa_t11000.json` |

#### Personal Income and Consumption
| Table ID | Description | Frequency | File |
|----------|-------------|-----------|------|
| `T20100` | Personal Income and Its Disposition | Monthly | `nipa_t20100.json` |
| `T20301` | Personal Consumption Expenditures by Major Type of Product | Monthly | `nipa_t20301.json` |
| `T20304` | Price Indexes for Personal Consumption Expenditures | Monthly | `nipa_t20304.json` |
| `T20305` | PCE Price Index by Major Type of Product | Monthly | `nipa_t20305.json` |
| `T20306` | Real Personal Consumption Expenditures by Major Type of Product | Monthly | `nipa_t20306.json` |

#### Government
| Table ID | Description | Frequency | File |
|----------|-------------|-----------|------|
| `T30100` | Government Current Receipts and Expenditures | Quarterly | `nipa_t30100.json` |
| `T30200` | Federal Government Current Receipts and Expenditures | Quarterly | `nipa_t30200.json` |
| `T30300` | State and Local Government Current Receipts and Expenditures | Quarterly | `nipa_t30300.json` |

### Underlying Detail Tables
| Table ID | Description | Frequency | File |
|----------|-------------|-----------|------|
| `U20305` | PCE by Type of Product (detailed) | Quarterly | `underlying_u20305.json` |

### Composite Datasets
| Dataset | Description | File |
|---------|-------------|------|
| **GDP Summary** | GDP headline numbers and major components (T10101, T10105, T10106, T10107) | `gdp_summary.json` |

---

## Bureau of Labor Statistics (BLS)

**Source:** U.S. Department of Labor, BLS  
**API:** https://api.bls.gov/publicAPI/v2  
**Data Directory:** `data/bls/`

### Press Releases (Qualitative)

| Dataset | Description | Frequency | File |
|---------|-------------|-----------|------|
| **Employment Situation** | Full narrative text of the Jobs Report with key highlights and tables | Monthly (First Friday) | `release_employment_situation.json` |
| **CPI Release** | Consumer Price Index news release with detailed breakdowns | Monthly | `release_cpi.json` |
| **PPI Release** | Producer Price Index news release | Monthly | `release_ppi.json` |
| **JOLTS Release** | Job Openings and Labor Turnover Survey release | Monthly | `release_jolts.json` |
| **All Releases** | Combined bundle of all major BLS press releases | Monthly | `all_releases.json` |

### Time Series Data

#### Employment
| Series ID | Description | File |
|-----------|-------------|------|
| `CES0000000001` | Total Nonfarm Payrolls (Thousands) | `nonfarm_payrolls.json` |
| `CES0500000001` | Total Private Payrolls (Thousands) | `private_payrolls.json` |
| `LNS14000000` | Unemployment Rate (Seasonally Adjusted) | `unemployment_rate.json` |
| `LNS11300000` | Labor Force Participation Rate | `labor_force_participation.json` |
| `CES0500000003` | Average Hourly Earnings, Private | `average_hourly_earnings.json` |
| `CES0500000002` | Average Weekly Hours, Private | `average_weekly_hours.json` |

#### CPI Components
| Series ID | Description | File |
|-----------|-------------|------|
| `CUUR0000SA0` | CPI-U, All Items | `cpi_all_urban.json` |
| `CUUR0000SA0L1E` | CPI-U, All Items Less Food and Energy (Core) | `cpi_core.json` |
| `CUUR0000SAF1` | CPI-U, Food | `cpi_food.json` |
| `CUUR0000SA0E` | CPI-U, Energy | `cpi_energy.json` |
| `CUUR0000SAH1` | CPI-U, Shelter | `cpi_shelter.json` |
| `CUUR0000SAM` | CPI-U, Medical Care | `cpi_medical.json` |

#### PPI
| Series ID | Description | File |
|-----------|-------------|------|
| `WPSFD4` | PPI Final Demand | `ppi_final_demand.json` |
| `WPSFD49104` | PPI Final Demand Less Foods and Energy (Core) | `ppi_core.json` |

#### JOLTS
| Series ID | Description | File |
|-----------|-------------|------|
| `JTS000000000000000JOL` | Job Openings: Total Nonfarm | `job_openings.json` |
| `JTS000000000000000QUR` | Quits Rate: Total Nonfarm | `quits_rate.json` |
| `JTS000000000000000HIR` | Hires Rate: Total Nonfarm | `hires_rate.json` |

---

## Atlanta Fed

**Source:** Federal Reserve Bank of Atlanta  
**URL:** https://www.atlantafed.org  
**Data Directory:** `data/atlanta_fed/`

### GDPNow Nowcasting Model

| Dataset | Description | Frequency | File |
|---------|-------------|-----------|------|
| **GDPNow Current** | Latest real-time GDP growth estimate for current quarter | Multiple per month (after key data releases) | `gdpnow_current.json` |
| **GDPNow History** | Historical evolution of GDPNow forecasts across quarters | Quarterly archives | `gdpnow_history.json` |
| **GDPNow Components** | Detailed component contributions to GDP forecast (Personal consumption, Investment, Net exports, Government) | Multiple per month | `gdpnow_components.json` |

### Excel Data Files
- `GDPNowcastDataReleaseDates.xlsx` - Schedule of data releases affecting GDPNow
- `GDPTrackingModelDataAndForecasts.xlsx` - Full tracking data and forecasts

---

## New York Fed

**Source:** Federal Reserve Bank of New York  
**URL:** https://www.newyorkfed.org  
**Data Directory:** `data/ny_fed/`

### Survey Data

| Dataset | Description | Frequency | File |
|---------|-------------|-----------|------|
| **Survey of Consumer Expectations (SCE)** | Consumer inflation and economic expectations overview | Monthly | `sce_overview.json` |
| **SCE Inflation Expectations** | 1-year and 3-year ahead inflation expectations | Monthly | `sce_inflation.json` |

### Economic Indicators

| Dataset | Description | Frequency | File |
|---------|-------------|-----------|------|
| **Weekly Economic Index (WEI)** | High-frequency measure of real economic activity (scaled to 4-quarter GDP growth) | Weekly | `weekly_economic_index.json` |

### Research (Qualitative)

| Dataset | Description | Frequency | File |
|---------|-------------|-----------|------|
| **Liberty Street Economics** | Research blog posts from NY Fed economists on labor markets, inflation, financial conditions, consumer behavior | Ongoing | `liberty_street_posts.json` |

### Excel Data Files
- `frbny-sce-data.xlsx` - Full SCE survey data
- `frbny-sce-public-microdata-complete-*.xlsx` - Microdata files

---

## Brookings Institution

**Source:** Hutchins Center on Fiscal and Monetary Policy  
**URL:** https://www.brookings.edu/centers/the-hutchins-center-on-fiscal-and-monetary-policy/  
**Data Directory:** `data/brookings/`

| Dataset | Description | Type | Frequency | File |
|---------|-------------|------|-----------|------|
| **Fiscal Impact Measure (FIM)** | Measures how much fiscal policy adds to or subtracts from GDP growth. Positive = fiscal stimulus, Negative = fiscal drag | Quantitative | Quarterly | `fim.json` |
| **Hutchins Center Research** | Recent fiscal and monetary policy research papers and analysis | Qualitative | Ongoing | `hutchins_research.json` |

---

## National Bureau of Economic Research (NBER)

**Source:** NBER  
**URL:** https://www.nber.org  
**Data Directory:** `data/nber/`

| Dataset | Description | Type | Frequency | File |
|---------|-------------|------|-----------|------|
| **Business Cycle Dates** | Official US recession and expansion dates determined by the NBER Business Cycle Dating Committee. Includes peak/trough dates, contraction/expansion durations | Reference | Event-driven | `business_cycles.json` |
| **Working Papers** | Recent NBER working papers - frontier economic research with titles, authors, and abstracts | Qualitative | Weekly | `working_papers.json` |
| **NBER Digest** | Accessible summaries of recent NBER research | Qualitative | Monthly | `digest.json` |

---

## Peterson Institute for International Economics (PIIE)

**Source:** PIIE  
**URL:** https://www.piie.com  
**Data Directory:** `data/piie/`

| Dataset | Description | Type | Frequency | File |
|---------|-------------|------|-----------|------|
| **Trade Policy Research** | Analysis of tariffs, trade agreements, and trade wars | Qualitative | Ongoing | `trade_research.json` |
| **PIIE Charts** | Data visualizations on global economics and trade | Quantitative | Ongoing | `charts.json` |
| **RealTime Economics Blog** | Commentary on current economic events and policy | Qualitative | Daily | `blog.json` |

---

## Data Organization

### Directory Structure

```
data/
├── atlanta_fed/          # GDPNow nowcasting data
├── bea/                  # National accounts (NIPA tables)
├── bls/                  # Employment, inflation, labor data
├── brookings/            # Fiscal policy research
├── cbo/                  # Congressional Budget Office (placeholder)
├── federal_reserve/      # FOMC materials, Beige Book, speeches
├── fred/                 # Economic time series from FRED
├── imf/                  # International Monetary Fund (placeholder)
├── nber/                 # Business cycles, working papers
├── ny_fed/               # Consumer surveys, WEI
├── oecd/                 # OECD data (placeholder)
├── piie/                 # Trade policy research
└── tracker.json          # Download metadata and status
```

### Archive System

Each data source directory contains an `archive/` subdirectory with timestamped versions of downloaded files, enabling:
- Data versioning and historical comparison
- Rollback capabilities
- Audit trail of data changes

### Tracker File

`tracker.json` contains metadata for all downloads:
- Source URL
- Local file path
- Content hash (for change detection)
- Download timestamp
- Status (success/failure)
- Custom metadata per dataset

---

## API Keys Required

| Source | Environment Variable | Registration URL |
|--------|---------------------|------------------|
| FRED | `FRED_API_KEY` | https://fred.stlouisfed.org/docs/api/api_key.html |
| BEA | `BEA_API_KEY` | https://apps.bea.gov/api/signup/ |
| BLS | `BLS_API_KEY` | https://www.bls.gov/developers/api_signature_v2.htm |

---

## Update Frequency Summary

| Frequency | Datasets |
|-----------|----------|
| **Daily** | Treasury yields, Fed funds rate, VIX, Dollar index, Corporate spreads |
| **Weekly** | Jobless claims, WEI, Fed balance sheet, Mortgage rates, NFCI |
| **Monthly** | Employment situation, CPI, PPI, PCE, Industrial production, Housing, Retail sales, JOLTS, Consumer sentiment |
| **Quarterly** | GDP, NIPA tables, GDPNow, SEP, FIM, Business cycles |
| **8x/Year** | FOMC statements/minutes, Beige Book |
| **Ongoing** | Fed speeches, Research papers, Blog posts |

---

## Data Types

### Quantitative
Time series data with numerical observations:
- FRED series (GDP, unemployment, inflation, rates)
- BEA NIPA tables
- GDPNow estimates
- WEI

### Qualitative
Text-based content for NLP/sentiment analysis:
- FOMC statements and minutes
- Fed speeches
- Beige Book
- BLS press releases
- Research papers and blog posts

### Reference
Static or rarely-changing reference data:
- NBER business cycle dates
- FOMC calendar

---

## Usage Notes

1. **Incremental Downloads**: Most scrapers support incremental updates, only downloading new content since the last fetch.

2. **Rate Limiting**: All scrapers implement rate limiting (0.5-2.0 seconds between requests) to respect source servers.

3. **Error Handling**: Failed downloads are logged and tracked; partial successes are saved.

4. **Data Freshness**: Check `tracker.json` for the `downloaded_at` timestamp of each dataset.

5. **Full Content**: Qualitative datasets include full text content (truncated for very long documents) suitable for text analysis.

