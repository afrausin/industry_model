# Macro Data Framework

A framework for collecting, storing, and analyzing US macroeconomic data with a point-in-time database for backtesting.

## Core Components

### 1. Data Ingestion (`ingest/`)
Collects data from official sources:
- **FRED** - Federal Reserve Economic Data (GDP, CPI, employment, rates)
- **Federal Reserve** - FOMC statements, Beige Book, minutes
- **Atlanta Fed** - GDPNow real-time forecasts
- **NY Fed** - Economic surveys and nowcasts
- **CBO** - Congressional Budget Office projections
- **Brookings, NBER, PIIE** - Economic research
- **IMF, OECD** - International economic data
- **FMP** - Financial Modeling Prep market data

### 2. Data Storage (`storage/`)
Point-In-Time (PIT) database system:
- `raw/` - JSON files from scrapers
- `db/` - SQLite PIT database
- Tracks observation dates and release dates
- Prevents look-ahead bias in backtesting

### 3. Core Analysis (`core/`)
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
python main.py

# Update data and database
python update_data.py

# Check status
python update_data.py --status
```

## Project Structure

```
macro/
├── ingest/             # Data collection
│   ├── scrapers/       # Web scrapers
│   └── fmp/            # FMP API integration
├── storage/            # Data storage
│   ├── raw/            # JSON data files
│   └── db/             # SQLite PIT database
├── core/               # Data loading & Gemini analysis
├── docs/               # Governance documentation
├── scripts/            # Utility scripts
├── main.py             # Scraper CLI
├── update_data.py      # Update pipeline
└── requirements.txt    # Dependencies
```

## Usage Examples

### Load FRED Data
```python
from core import MacroDataLoader, MacroConfig

config = MacroConfig()
loader = MacroDataLoader(config.data_dir)

# Load growth series
gdp = loader.load_fred_series("GDPC1")
print(gdp.tail())
```

### Analyze Fed Documents with Gemini
```python
from core import GeminiMacroAnalyzer, MacroDataLoader

loader = MacroDataLoader(data_dir)
docs = loader.load_all_qualitative_docs()

analyzer = GeminiMacroAnalyzer(api_key=os.getenv("GOOGLE_API_KEY"))
analysis = analyzer.analyze_documents(docs)
```

### Load Market Data
```python
from core import FMPDataLoader
from pathlib import Path

fmp = FMPDataLoader(Path("ingest/fmp/exploration_results"))
snapshot = fmp.get_market_snapshot()
calendar = fmp.load_economic_calendar(country="US", days_ahead=7)
```

## Data Sources

| Source | Data Type | Update Frequency |
|--------|-----------|------------------|
| FRED | Economic series (GDP, CPI, employment) | Daily |
| Federal Reserve | FOMC statements, Beige Book, speeches | Per release |
| Atlanta Fed | GDPNow forecasts | Weekly |
| NY Fed | Various surveys | Monthly |
| FMP | Economic calendar, market quotes | Real-time |

## License

Private use only.
