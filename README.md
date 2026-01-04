# Macro Data Framework

A framework for collecting, storing, and analyzing US macroeconomic data with AI-powered analysis.

## Core Components

### 1. Data Scraping (`scrapers/`)
Collects data from official sources:
- **FRED** - Federal Reserve Economic Data (GDP, CPI, employment, rates)
- **Federal Reserve** - FOMC statements, Beige Book, minutes
- **Atlanta Fed** - GDPNow real-time forecasts
- **NY Fed** - Economic surveys and nowcasts
- **CBO** - Congressional Budget Office projections
- **Brookings, NBER, PIIE** - Economic research
- **IMF, OECD** - International economic data

### 2. Data Storage (`database/`, `storage/`)
Point-In-Time (PIT) database system:
- Tracks observation dates and release dates
- Prevents look-ahead bias in backtesting
- SQLite-based for simplicity

### 3. Core Analysis (`core/`)
- **MacroDataLoader** - Load FRED series and Fed documents
- **FMPDataLoader** - Market data from Financial Modeling Prep
- **GeminiMacroAnalyzer** - AI analysis of Fed documents

### 4. AI Agent (`agent/`)
Gemini-powered optimization agent with 80+ tool functions.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set API keys in .env
FRED_API_KEY=your_key
FMP_API_KEY=your_key
GEMINI_API_KEY=your_key

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
├── scrapers/           # Data collection layer
├── database/           # PIT database implementation
├── storage/            # Storage interface & loaders
├── core/               # Data loading & Gemini analysis
├── src_fmp/            # FMP API exploration
├── agent/              # Gemini AI agent
├── scripts/data/       # Data update scripts
├── data/               # Scraped data (JSON)
├── logs/               # Application logs
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

analyzer = GeminiMacroAnalyzer(api_key=os.getenv("GEMINI_API_KEY"))
analysis = analyzer.analyze_documents(docs)
```

### Load Market Data
```python
from core import FMPDataLoader
from pathlib import Path

fmp = FMPDataLoader(Path("src_fmp/exploration_results"))
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

## Backup Location

Previous forecasting/quadrant code backed up to: `_backup_before_etf_refactor_*/`

To restore: copy files back from the backup folder.

## License

Private use only.
