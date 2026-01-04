# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Macro Data Framework** - A system for collecting, storing, and analyzing US macroeconomic data with AI-powered factor timing optimization for ETF forecasting.

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
ANTHROPIC_API_KEY= # Required for Claude agent
GOOGLE_API_KEY=    # Required for Gemini agent/analyzer
```

### Data Collection & Updates
```bash
python main.py                      # Download all data from scrapers
python main.py --source fred        # Download specific source
python main.py --list               # List available datasets
python update_data.py               # Full update (scrape + migrate to DB)
python update_data.py --status      # Show data freshness status
python update_data.py --db-only     # Only update SQLite from existing JSON
```

### AI Optimization Agents
```bash
# Claude agent (recommended)
python -m agent --claude --factor VLUE --max-iterations 30
python -m agent --claude --interactive

# Gemini agent (legacy)
python -m agent --gemini --smart VLUE --max-iterations 50
```

---

## Architecture

### Data Flow Pipeline
```
ingest/scrapers → storage/raw (JSON) → storage/db (SQLite PIT) → core/data_loader → agent/
```

### Module Structure

| Module | Purpose | Key Files |
|--------|---------|-----------|
| **ingest/** | Data collection from official sources | `scrapers/base_scraper.py`, 10 scrapers (FRED, Fed, Atlanta Fed, NY Fed, CBO, Brookings, NBER, PIIE, IMF, OECD), `fmp/` |
| **storage/** | Data persistence layer | `raw/` (JSON by source), `db/pit_database.py` (PIT SQLite), `db/pit_data_loader.py` |
| **core/** | Data loading and analysis utilities | `config.py` (40+ FRED series), `data_loader.py`, `fmp_loader.py`, `gemini_analyzer.py` |
| **agent/** | AI optimization agents | `claude_agent.py`, `heuristic_agent.py`, `tools/` (10 modules), Bayesian/genetic optimization |
| **docs/** | Governance documentation | `QUANT_RESEARCHER_CONSTITUTION.md`, `AI_DATA_PIPELINE_FRAMEWORK.md` |

### Key Concepts

**Point-In-Time (PIT) Database**
- Tracks both observation dates AND release dates
- Critical for backtesting - prevents look-ahead bias
- Only uses data that was available as of a given historical date

**Factor Timing**
- Optimizes timing signals for factor ETFs: VLUE, SIZE, QUAL, USMV, MTUM
- Uses macroeconomic features as predictors
- Follows hypothesis-driven development per the Constitution

**Dual LLM Support**
- Claude (Anthropic) - recommended for new work
- Gemini (Google) - legacy support

---

## Data Sources

| Source | Key Data | Update Frequency |
|--------|----------|------------------|
| FRED | GDP, CPI, unemployment, Treasury yields, 40+ series | Daily |
| Federal Reserve | FOMC statements, minutes, Beige Book, speeches | Per event |
| Atlanta Fed | GDPNow real-time GDP forecasts | Weekly |
| NY Fed | Consumer expectations, Weekly Economic Index | Weekly |
| FMP | Economic calendar, market quotes | Real-time |

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
