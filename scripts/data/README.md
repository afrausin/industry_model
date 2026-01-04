# Data Scripts

Scripts for data collection, processing, and management.

## Available Scripts

### `update_data.py`
Main data update pipeline that orchestrates all data collection.

```bash
# Full update (all sources + database)
python update_data.py

# Check current status
python update_data.py --status

# Update specific source
python update_data.py --source fred

# Include FMP data refresh  
python update_data.py --with-fmp
```

### `run_scrapers.py`
Execute all configured web scrapers for economic data.

```bash
python run_scrapers.py
```

### `run_spglobal_pmi.py`
Specialized scraper for S&P Global PMI data and reports.

```bash
python run_spglobal_pmi.py
```

### `backfill_pop_features.py`
Backfill population-based features for existing data.

```bash
python backfill_pop_features.py
```