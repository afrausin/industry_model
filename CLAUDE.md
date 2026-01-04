# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Macro Data Framework** - A system for collecting, storing, and analyzing US macroeconomic data with a point-in-time database for backtesting.

---

## Governance Documentation

> **CRITICAL**: Before developing any quantitative signals, backtesting strategies, or data pipelines, read and follow these governance frameworks.

### 📖 Quantitative Research Constitution

**[docs/QUANT_RESEARCHER_CONSTITUTION.md](docs/QUANT_RESEARCHER_CONSTITUTION.md)**

The definitive rule book for strategy development. Codifies scientific inquiry, data integrity, risk management, and regulatory compliance.

| Principle | Requirement |
|-----------|-------------|
| Hypothesis-Driven | Every strategy needs an a priori economic rationale before backtesting |
| Point-in-Time Data | Strict PIT compliance to prevent look-ahead bias |
| Seven Sins Prevention | Protocols against survivorship bias, overfitting, storytelling bias, outlier dependence, transaction cost neglect, shorting cost asymmetry |
| Risk Management | Hard limits: Max Drawdown <20%, Gross Exposure <200%, Single Position <5% |
| Walk-Forward Validation | Rolling in-sample/out-of-sample testing with WFE >60% |
| Transaction Cost Modeling | Square root law market impact, realistic slippage |

### 📖 AI Data Pipeline Framework

**[docs/AI_DATA_PIPELINE_FRAMEWORK.md](docs/AI_DATA_PIPELINE_FRAMEWORK.md)**

Technical standards for "Agent-Ready" financial data architecture.

| Pattern | Implementation |
|---------|----------------|
| Context Engineering | llms.txt standards, MCP protocol, self-describing databases |
| API Usage | Refinitiv (`get_history` for timeseries), S&P (`paginate=True`), FMP (standardized vs as-reported) |
| Symbology | Time-aware mapping using FIGI/CIK/PermID, never rely on tickers for historical analysis |
| PIT Alignment | `pd.merge_asof` with `direction='backward'` |
| Data Validation | Pandera schemas for statistical typing, Great Expectations for pipeline contracts |
| Vectorization | Prefer `pandas`/`numpy` operations over loops |

### 📖 Data Catalog

**[docs/DATA_CATALOG.md](docs/DATA_CATALOG.md)**

Complete inventory of all available data sources and datasets.

| Category | Sources | Series/Datasets |
|----------|---------|-----------------|
| Quantitative Time Series | FRED (40+ series), FMP | GDP, CPI, employment, rates, financial conditions |
| Fed Policy Documents | Federal Reserve, Atlanta Fed, NY Fed | FOMC statements, minutes, Beige Book, speeches |
| Economic Projections | CBO, Atlanta Fed GDPNow | Budget forecasts, real-time GDP estimates |
| Research & Analysis | Brookings, NBER, PIIE, IMF, OECD | Working papers, policy analysis, reports |

---

## Quick Start

### Setup
```bash
pip install -r requirements.txt
```

**Required environment variables** (in `.env`):
```
FRED_API_KEY=      # Required for FRED data
FMP_API_KEY=       # Optional for Financial Modeling Prep
GOOGLE_API_KEY=    # Optional for Gemini analyzer
```

### Data Collection & Updates
```bash
python -m data.main                      # Download all data from scrapers
python -m data.main --source fred        # Download specific source
python -m data.main --list               # List available datasets
python -m data.update_data               # Full update (scrape + migrate to DB)
python -m data.update_data --status      # Show data freshness status
python -m data.update_data --db-only     # Only update SQLite from existing JSON
```

---

## Architecture

### Data Flow Pipeline
```
data/ingest/scrapers → data/storage/raw (JSON) → data/storage/db (SQLite PIT) → data/core/data_loader
```

### Module Structure

| Module | Purpose | Key Files |
|--------|---------|-----------|
| **data/** | Data pipeline package | `main.py`, `update_data.py`, `__init__.py` |
| **data/ingest/** | Data collection from official sources | `scrapers/base_scraper.py`, 10 scrapers (FRED, Fed, Atlanta Fed, NY Fed, CBO, Brookings, NBER, PIIE, IMF, OECD), `fmp/` |
| **data/storage/** | Data persistence layer | `raw/` (JSON by source), `db/pit_database.py` (PIT SQLite), `db/pit_data_loader.py` |
| **data/core/** | Data loading and analysis utilities | `config.py` (40+ FRED series), `data_loader.py`, `fmp_loader.py`, `gemini_analyzer.py` |
| **docs/** | Governance & reference documentation | `QUANT_RESEARCHER_CONSTITUTION.md`, `AI_DATA_PIPELINE_FRAMEWORK.md`, `DATA_CATALOG.md` |

### Key Concepts

**Point-In-Time (PIT) Database**
- Tracks both observation dates AND release dates
- Critical for backtesting - prevents look-ahead bias
- Only uses data that was available as of a given historical date

---

## Data Sources

> **See [docs/DATA_CATALOG.md](docs/DATA_CATALOG.md) for complete data inventory.**

| Source | Key Data | Update Frequency |
|--------|----------|------------------|
| **FRED** | 40+ series: GDP, CPI, employment, Treasury yields, financial conditions | Daily to Quarterly |
| **Federal Reserve** | FOMC statements, minutes, Beige Book, speeches, SEP projections | Per event |
| **Atlanta Fed** | GDPNow real-time GDP nowcast with component breakdown | Weekly |
| **NY Fed** | Consumer expectations survey, Weekly Economic Index | Weekly/Monthly |
| **CBO** | 10-year budget projections, economic forecasts, long-term outlook | Semi-annual |
| **Brookings** | Fiscal Impact Measure (FIM), Hutchins Center research | Quarterly |
| **NBER** | Official business cycle dates, working papers, digest | Event-driven |
| **PIIE** | Trade policy research, charts, RealTime Economics blog | Ongoing |
| **IMF** | US Article IV consultation, World Economic Outlook, GFSR | Semi-annual |
| **OECD** | Economic outlook, US survey, Composite Leading Indicators | Monthly to Biennial |
| **FMP** | Economic calendar, index/ETF quotes, market data | Real-time |

---

## Code Standards

Per the governance documentation, all code must:

1. **Data Integrity**
   - Never drop rows silently - log all dropped data
   - Validate schemas using Pandera at ingestion and egress
   - Check for NaN, Inf, and negative values in Price/Volume fields

2. **Time Series Handling**
   - Use `pd.merge_asof` with `direction='backward'` for PIT merges
   - Ensure DatetimeIndex is timezone-aware (prefer UTC)
   - Explicitly handle look-ahead bias by verifying release dates vs observation dates

3. **Performance**
   - Prefer vectorized operations (Pandas/NumPy) over loops
   - Use functional patterns for data transformations
   - Structure code into modular ETL pipelines

4. **Strategy Development**
   - Document economic rationale BEFORE writing code
   - Limit optimized parameters to 2-3 (avoid overfitting)
   - Use walk-forward analysis, not simple train/test splits
