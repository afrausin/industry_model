# Hedgeye Quadrant Framework

A Python implementation of the Hedgeye Risk Range quadrant framework for analyzing US macroeconomic conditions.

## Quadrant Framework

The Hedgeye framework divides the macro environment into four quadrants based on the **rate of change** (acceleration/deceleration) of growth and inflation:

| Quadrant | Growth | Inflation | Description | Best Assets |
|----------|--------|-----------|-------------|-------------|
| **Quad 1** | ↑ Accelerating | ↓ Decelerating | Goldilocks | Growth stocks, Tech, Small caps |
| **Quad 2** | ↑ Accelerating | ↑ Accelerating | Inflationary Boom | Commodities, Energy, Materials |
| **Quad 3** | ↓ Decelerating | ↓ Decelerating | Deflation | Bonds, Utilities, Quality |
| **Quad 4** | ↓ Decelerating | ↑ Accelerating | Stagflation | Cash, Gold |

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Gemini API (for qualitative analysis):**
   Add your Gemini API key to `.env`:
   ```
   GEMINI_API_KEY=your_api_key_here
   ```

## Usage

### Command Line Analysis

Run a full analysis from the command line:

```bash
# Full analysis with Gemini
python -m src_hedgeye.run_analysis

# Quantitative only (no Gemini API required)
python -m src_hedgeye.run_analysis --no-gemini

# Quiet mode (JSON output only)
python -m src_hedgeye.run_analysis --quiet

# Save to specific file
python -m src_hedgeye.run_analysis --output my_analysis.json
```

### Streamlit Dashboard

Launch the interactive dashboard:

```bash
streamlit run src_hedgeye/dashboard.py
```

Or use the runner script:
```bash
python -m src_hedgeye.run_dashboard
```

### Python API

```python
from src_hedgeye import (
    HedgeyeConfig,
    MacroDataLoader,
    RateOfChangeCalculator,
    QuadrantProbabilityEngine,
)

# Initialize with default config
config = HedgeyeConfig()
engine = QuadrantProbabilityEngine(config)

# Run full analysis
analysis = engine.run_full_analysis()

# Access results
print(f"Most likely quadrant: Quad {analysis.most_likely_quadrant}")
print(f"Probability: {analysis.confidence:.1%}")
print(f"GDP Now: {analysis.gdpnow_estimate}%")

# View probability distribution
for quad, prob in analysis.final_probabilities.items():
    print(f"Quad {quad}: {prob:.1%}")

# Save results
output_path = engine.save_analysis(analysis)
```

## Data Sources

The framework uses data from your existing scrapers:

- **Growth indicators**: GDP, Industrial Production, Payrolls, Retail Sales (FRED)
- **Inflation indicators**: CPI, Core CPI, PCE, Core PCE (FRED)
- **GDPNow**: Atlanta Fed real-time GDP tracking
- **Qualitative**: FOMC statements, Beige Book, FOMC minutes

## How It Works

### 1. Quantitative Analysis (Rate of Change)

Calculates the 3-month annualized rate of change for key macro series:
- If current RoC > prior RoC → **Accelerating**
- If current RoC < prior RoC → **Decelerating**

### 2. Qualitative Analysis (Gemini AI)

Uses Gemini to analyze Fed documents:
- Summarizes FOMC statements and Beige Book
- Compares current vs. prior period conditions
- Extracts forward-looking language
- Identifies risk factors for quadrant transitions

### 3. Probability Estimation

Combines quantitative and qualitative signals:
- Default weights: 40% quantitative, 60% qualitative
- Outputs probability distribution across all 4 quadrants
- Includes reasoning and market implications

## File Structure

```
src_hedgeye/
├── __init__.py          # Package initialization
├── config.py            # Configuration and settings
├── data_loader.py       # Data loading from JSON files
├── roc_calculator.py    # Rate of Change calculations
├── gemini_analyzer.py   # Gemini AI integration
├── quadrant_engine.py   # Main probability engine
├── dashboard.py         # Streamlit dashboard
├── run_analysis.py      # CLI runner
├── run_dashboard.py     # Dashboard launcher
└── results/             # Saved analysis outputs
```

## Configuration

Key settings in `HedgeyeConfig`:

```python
# Rate of Change lookback periods
growth_roc_periods = [3, 6, 12]  # months
inflation_roc_periods = [3, 6, 12]

# Analysis weights
quantitative_weight = 0.4
qualitative_weight = 0.6

# Gemini model
gemini_model = "gemini-2.0-flash"
```

## Example Output

```
Most Likely Quadrant: Quad 4
Probability: 45.0%
GDP Now: 3.9%

Probability Distribution:
  Quad 1: 20.0% ████████
  Quad 2: 20.0% ████████
  Quad 3: 15.0% ██████
  Quad 4: 45.0% ██████████████████

Reasoning: Based on accelerating inflation signals from CPI/PCE 
combined with mixed growth signals, the economy appears to be 
transitioning toward stagflationary conditions...
```

