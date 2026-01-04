# Scripts Directory

Organized scripts by domain for better maintainability:

## Structure

```
scripts/
├── data/           # Data collection and management
├── analysis/       # Analysis and backtesting  
├── dashboards/     # User interface applications
└── utilities/      # Development and maintenance tools
```

## Quick Access

### Data Management
```bash
# Update all data sources
python scripts/data/update_data.py

# Run all scrapers
python scripts/data/run_scrapers.py

# Backfill population features
python scripts/data/backfill_pop_features.py
```

### Analysis & Backtesting
```bash
# Barra factor analysis
python scripts/analysis/run_barra_backtest.py

# Industry rotation backtest
python scripts/analysis/run_industry_backtest.py

# Forward-looking analysis
python scripts/analysis/run_barra_forward_analysis.py
```

### Dashboards
```bash
# Main data dashboard
python scripts/dashboards/run_dashboard.py

# Hedgeye quadrant dashboard
python scripts/dashboards/run_hedgeye_dashboard.py

# Web dashboard
python scripts/dashboards/run_web_dashboard.py
```

### Utilities
```bash
# Database viewer/explorer
python scripts/utilities/db_viewer.py
```