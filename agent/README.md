# AI Optimization Agents

AI agents that use **advanced optimization strategies** to improve factor timing models. Supports both Claude (Anthropic) and Gemini (Google) for flexible deployment.

## 🧠 Smart Optimization Features

### 1. Bayesian Optimization
Models the Sharpe ratio surface and suggests experiments that balance exploration vs exploitation.

### 2. Genetic Algorithms
Treats configurations as "genes" - evolves through selection, crossover, and mutation.

### 3. Structured Exploration Phases
```
Phase 1 (0-25%):  Feature Discovery    - Find best predictive features
Phase 2 (25-50%): Weight Optimization  - Optimize feature weights  
Phase 3 (50-75%): Parameter Tuning     - Find optimal threshold/hold period
Phase 4 (75-100%): Refinement          - Fine-tune and validate
```

### 4. Self-Reflection
Every 5 iterations, the agent analyzes:
- Which features work best
- Why certain configs succeed/fail
- What new approaches to try

### 5. Ensemble Building
Combines top 5 configurations for more robust predictions.

### 6. Out-of-Sample Validation
Splits data into train/validation/test to detect overfitting.

## Usage

### Claude Optimization (Recommended)

```bash
# Smart optimization with Claude
python -m agent --claude --factor VLUE --max-iterations 30

# Interactive mode with Claude
python -m agent --claude --interactive

# Target specific Sharpe ratio
python -m agent --claude --factor VLUE --target-sharpe 0.8
```

### Gemini Optimization (Legacy)

```bash
# Smart optimization with Gemini
python -m agent --gemini --smart VLUE --max-iterations 50

# Standard iterative optimization  
python -m agent --gemini --iterate --factor VLUE --min-iterations 30

# Interactive mode
python -m agent --gemini --interactive
```

### Setup

```bash
# For Claude (recommended)
export ANTHROPIC_API_KEY=your_anthropic_key

# For Gemini (legacy)
export GOOGLE_API_KEY=your_google_key

# Install dependencies
pip install anthropic  # for Claude
pip install google-generativeai  # for Gemini
```

## New Smart Tools

| Tool | Description |
|------|-------------|
| `initialize_smart_optimization` | Start Bayesian/Genetic optimizers |
| `suggest_next_experiment` | Get AI-guided suggestions |
| `record_experiment_result` | Record results for learning |
| `get_genetic_population` | Get genetic algorithm population |
| `evolve_genetic_population` | Evolve to next generation |
| `build_ensemble` | Combine top configs |
| `test_ensemble` | Backtest ensemble |
| `get_reflection_prompt` | Analyze what's working |
| `validate_out_of_sample` | Check for overfitting |
| `get_optimization_summary` | Full optimization stats |

## All Available Tools

### Analysis & Backtesting
- `get_current_performance` - Sharpe, hit rate, drawdowns
- `analyze_feature_correlations` - Find top features by IC
- `run_backtest` - Test custom strategies
- `run_parameter_sweep` - Grid search threshold/hold
- `compare_strategies` - Compare multiple configs
- `generate_new_signal` - Create IC-weighted signal
- `suggest_improvements` - Get actionable suggestions

### Database Access
- `list_database_tables` - Show all tables
- `list_economic_series` - List FRED series
- `list_market_symbols` - List ETFs, indices
- `get_economic_series` - Get data values
- `get_market_prices` - Get price data
- `get_fed_documents` - Get Fed communication features
- `get_economic_calendar` - Get economic events
- `run_custom_query` - Execute SQL

### Feature Engineering
- `create_derived_feature` - Create zscore, momentum, etc.
- `combine_features` - Combine multiple features
- `test_feature_predictive_power` - Test IC at multiple horizons
- `list_available_base_features` - List raw features

### Code Modification
- `list_model_scripts` - List editable scripts
- `read_file` - Read Python files
- `write_file` - Write files (with backup)
- `search_replace_in_file` - Find and replace
- `insert_code_at_line` - Insert at line
- `run_script` - Execute scripts
- `lint_file` - Check syntax

## Example Smart Optimization

```
🧠 SMART OPTIMIZATION: VLUE
Strategies: Bayesian=True, Genetic=True, Reflection=True
Max iterations: 50

📊 Getting baseline performance...
   Baseline Sharpe: -0.013

📈 ITERATION 1/50 | Phase: FEATURE_DISCOVERY
🔧 Tool: initialize_smart_optimization
🔧 Tool: analyze_feature_correlations
   Found: XLY_ma_ratio (IC=0.18), IEF_vol_21d (IC=-0.16)...

📈 ITERATION 10/50 | Phase: FEATURE_DISCOVERY
   Sharpe: 0.357 (Δ=+0.370)
   🏆 New best!

📈 ITERATION 15/50 | Phase: WEIGHT_OPTIMIZATION
🔧 Tool: suggest_next_experiment
🔧 Tool: run_backtest
   Sharpe: 0.521 (Δ=+0.164)
   🏆 New best!

📈 ITERATION 25/50 | Phase: PARAMETER_TUNING
🔧 Tool: run_parameter_sweep
   Sharpe: 0.779 (Δ=+0.258)
   🏆 New best!

📈 ITERATION 35/50 | Phase: REFINEMENT
🔧 Tool: get_reflection_prompt
🔧 Tool: test_ensemble
🔧 Tool: validate_out_of_sample
   Train: 0.82, Val: 0.71, Test: 0.68
   Status: ROBUST ✅

🎯 SMART OPTIMIZATION COMPLETE
Baseline: -0.013 → Best: 0.779
Improvement: +0.792
✅ Ensemble tested
✅ Out-of-sample validated
```

## Configuration

Edit `agent/config.py`:

```python
DEFAULT_MODEL = "gemini-3-pro-preview"
MAX_TOKENS = 100_000
TEMPERATURE = 0.7
```

## Programmatic Use

```python
from agent.heuristic_agent import HeuristicOptimizationAgent

agent = HeuristicOptimizationAgent()

# Smart optimization with all strategies
result = agent.smart_optimize(
    factor="VLUE",
    max_iterations=50,
    use_bayesian=True,
    use_genetic=True,
    use_reflection=True,
    validate_oos=True,
    build_ensemble=True,
)

print(f"Best Sharpe: {result['best_sharpe']}")
print(f"Best Config: {result['best_config']}")
```

## Data Available

- **43+ Economic Series**: Interest rates, unemployment, inflation
- **30+ Market Symbols**: Factors, sectors, indices, bonds
- **700+ Fed Documents**: FOMC statements, minutes, Beige Book
- **11,000+ Calendar Events**: CPI, NFP, GDP releases
