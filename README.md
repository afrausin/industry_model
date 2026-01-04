# Macro Data Framework

A framework for collecting, storing, and analyzing US macroeconomic data with a point-in-time database for backtesting.

## Core Components

### Data Pipeline (`data/`)

All data-related code is organized under the `data/` package:

#### 1. Data Ingestion (`data/ingest/`)
Collects data from official sources:
- **FRED** - Federal Reserve Economic Data (GDP, CPI, employment, rates)
- **Federal Reserve** - FOMC statements, Beige Book, minutes
- **Atlanta Fed** - GDPNow real-time forecasts
- **NY Fed** - Economic surveys and nowcasts
- **CBO** - Congressional Budget Office projections
- **Brookings, NBER, PIIE** - Economic research
- **IMF, OECD** - International economic data
- **FMP** - Financial Modeling Prep market data

#### 2. Data Storage (`data/storage/`)
Point-In-Time (PIT) database system:
- `raw/` - JSON files from scrapers
- `db/` - SQLite PIT database
- Tracks observation dates and release dates
- Prevents look-ahead bias in backtesting

#### 3. Core Analysis (`data/core/`)
- **MacroDataLoader** - Load FRED series and Fed documents
- **FMPDataLoader** - Market data from Financial Modeling Prep
- **GeminiMacroAnalyzer** - AI analysis of Fed documents

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set API keys in .env
FRED_API_KEY=your_key
FMP_API_KEY=your_key       # Optional
GOOGLE_API_KEY=your_key    # Optional for Gemini analyzer

# Scrape all data
python -m data.main

# Update data and database
python -m data.update_data

# Check status
python -m data.update_data --status
```

## Project Structure

```
macro/
├── data/                   # Data pipeline package
│   ├── __init__.py         # Package exports
│   ├── main.py             # Scraper CLI
│   ├── update_data.py      # Update pipeline
│   ├── ingest/             # Data collection
│   │   ├── scrapers/       # Web scrapers
│   │   └── fmp/            # FMP API integration
│   ├── storage/            # Data storage
│   │   ├── raw/            # JSON data files
│   │   └── db/             # SQLite PIT database
│   └── core/               # Data loading & Gemini analysis
├── docs/                   # Governance documentation
├── scripts/                # Utility scripts
└── requirements.txt        # Dependencies
```

## Usage Examples

### Load FRED Data
```python
from data import MacroDataLoader, MacroConfig

config = MacroConfig()
loader = MacroDataLoader(config.data_dir)

# Load growth series
gdp = loader.load_fred_series("GDPC1")
print(gdp.tail())
```

### Analyze Fed Documents with Gemini
```python
from data import GeminiMacroAnalyzer, MacroDataLoader

loader = MacroDataLoader(data_dir)
docs = loader.load_all_qualitative_docs()

analyzer = GeminiMacroAnalyzer(api_key=os.getenv("GOOGLE_API_KEY"))
analysis = analyzer.analyze_documents(docs)
```

### Load Market Data
```python
from data import FMPDataLoader
from pathlib import Path

fmp = FMPDataLoader(Path("data/ingest/fmp/exploration_results"))
snapshot = fmp.get_market_snapshot()
calendar = fmp.load_economic_calendar(country="US", days_ahead=7)
```

## Data Sources

> **See [docs/DATA_CATALOG.md](docs/DATA_CATALOG.md) for complete data inventory with all series and datasets.**

### Quantitative Time Series

| Source | Key Indicators | Frequency |
|--------|----------------|-----------|
| **FRED** | 40+ series: GDP, CPI, employment, Treasury yields, financial conditions, housing | Daily to Quarterly |
| **FMP** | Index quotes, ETF prices, economic calendar | Real-time |
| **Atlanta Fed** | GDPNow real-time GDP estimate with components | Weekly |
| **NY Fed** | Weekly Economic Index, consumer expectations | Weekly/Monthly |

### Policy Documents & Research

| Source | Content | Frequency |
|--------|---------|-----------|
| **Federal Reserve** | FOMC statements, minutes, Beige Book, Fed speeches, SEP projections | Per event |
| **CBO** | 10-year budget projections, economic forecasts, long-term outlook | Semi-annual |
| **Brookings** | Fiscal Impact Measure, Hutchins Center research | Quarterly |
| **NBER** | Official business cycle dates, working papers | Event-driven |
| **PIIE** | Trade policy research, charts, economics blog | Ongoing |
| **IMF** | US Article IV, World Economic Outlook, Global Financial Stability | Semi-annual |
| **OECD** | Economic outlook, US survey, Composite Leading Indicators | Monthly to Biennial |

## Documentation

| Document | Description |
|----------|-------------|
| [docs/DATA_CATALOG.md](docs/DATA_CATALOG.md) | Complete inventory of all data sources, series, and datasets |
| [docs/QUANT_RESEARCHER_CONSTITUTION.md](docs/QUANT_RESEARCHER_CONSTITUTION.md) | Rules for quantitative research and strategy development |
| [docs/AI_DATA_PIPELINE_FRAMEWORK.md](docs/AI_DATA_PIPELINE_FRAMEWORK.md) | Technical standards for agent-ready data architecture |

## License

Private use only.
