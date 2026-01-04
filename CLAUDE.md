# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Macro Data Framework - A system for collecting, storing, and analyzing US macroeconomic data with AI-powered factor timing optimization for ETF forecasting.

## Common Commands

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

### Setup
```bash
pip install -r requirements.txt
# Required env vars in .env: FRED_API_KEY, FMP_API_KEY, ANTHROPIC_API_KEY (or GOOGLE_API_KEY)
```

## Architecture

### Data Flow Pipeline
```
ingest/scrapers → storage/raw (JSON) → storage/db (SQLite PIT) → core/data_loader → agent/
```

### Module Structure

**ingest/** - Data collection from official sources
- `scrapers/base_scraper.py` - Abstract base class for all scrapers
- 10 scrapers: FRED, Fed, Atlanta Fed, NY Fed, CBO, Brookings, NBER, PIIE, IMF, OECD
- `fmp/` - Financial Modeling Prep API integration

**storage/** - Data persistence layer
- `raw/` - JSON files from scrapers, organized by source
- `db/pit_database.py` - Point-In-Time SQLite database (prevents look-ahead bias in backtesting)
- `db/pit_data_loader.py` - Data access layer

**core/** - Data loading and analysis utilities
- `config.py` - MacroConfig dataclass with 40+ FRED series definitions
- `data_loader.py` - MacroDataLoader for FRED series and Fed documents
- `fmp_loader.py` - FMPDataLoader for market data
- `gemini_analyzer.py` - Gemini-based document analysis (2.5K lines)

**agent/** - AI optimization agents for factor timing
- `claude_agent.py` - Claude/Anthropic agent with structured tool use
- `heuristic_agent.py` - Gemini agent (legacy)
- `tools/` - 10 tool modules: analysis, backtest, optimization, database, code, features, strategy_tools
- Smart optimization: Bayesian optimization, genetic algorithms, 4-phase exploration, self-reflection

### Key Concepts

**Point-In-Time (PIT) Database**: Tracks both observation dates AND release dates. Critical for backtesting - prevents look-ahead bias by only using data available as of a given date.

**Factor Timing**: Optimizes timing signals for factor ETFs (VLUE, SIZE, QUAL, USMV, MTUM) using macroeconomic features.

**Dual LLM Support**: Both Claude (Anthropic) and Gemini (Google) agents available. Claude is recommended for new work.

## Environment Variables

```
FRED_API_KEY=      # Required for FRED data
FMP_API_KEY=       # Optional for Financial Modeling Prep
ANTHROPIC_API_KEY= # Required for Claude agent
GOOGLE_API_KEY=    # Required for Gemini agent/analyzer
```

## Data Sources

| Source | Key Data |
|--------|----------|
| FRED | GDP, CPI, unemployment, Treasury yields, 40+ series |
| Federal Reserve | FOMC statements, minutes, Beige Book, speeches |
| Atlanta Fed | GDPNow real-time forecasts |
| NY Fed | Consumer expectations, Weekly Economic Index |
| FMP | Economic calendar, market quotes |
