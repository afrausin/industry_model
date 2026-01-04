"""
Claude Optimization Agent
=========================

An AI agent powered by Claude (Anthropic) that analyzes and improves 
the performance of factor timing strategies using Claude's reasoning capabilities.

Uses the Anthropic SDK with tool use for systematic optimization.
"""

import json
import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime

# Set up file logging for all LLM interactions
LOG_DIR = Path(__file__).parent / "output" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / f"llm_interactions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

def log_interaction(direction: str, content: str):
    """Log an interaction to the file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a") as f:
        f.write(f"\n{'='*80}\n")
        f.write(f"[{timestamp}] {direction}\n")
        f.write(f"{'='*80}\n")
        f.write(content[:10000] if len(content) > 10000 else content)  # Limit size
        f.write("\n")

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv is optional

try:
    import anthropic
except ImportError:
    raise ImportError("Please install anthropic: pip install anthropic")

from .config import (
    FACTORS,
    AGENT_OUTPUT_DIR,
    BACKTEST_DEFAULTS,
    FEATURE_CATEGORIES,
)

# Claude-compatible tool definitions
CLAUDE_TOOLS = [
    {
        "name": "get_current_performance",
        "description": "Get baseline performance metrics for a factor ETF vs SPY (Sharpe ratio, hit rate, max drawdown).",
        "input_schema": {
            "type": "object",
            "properties": {
                "factor": {
                    "type": "string", 
                    "description": "Factor ETF symbol",
                    "enum": ["VLUE", "SIZE", "QUAL", "USMV", "MTUM"]
                },
                "start_date": {
                    "type": "string",
                    "description": "Start date in YYYY-MM-DD format. Default: 2016-01-01"
                }
            },
            "required": ["factor"]
        }
    },
    {
        "name": "analyze_feature_correlations", 
        "description": "Calculate Information Coefficients between features and factor forward returns. Returns top predictive features.",
        "input_schema": {
            "type": "object",
            "properties": {
                "factor": {
                    "type": "string",
                    "description": "Factor ETF symbol", 
                    "enum": ["VLUE", "SIZE", "QUAL", "USMV", "MTUM"]
                },
                "horizon": {
                    "type": "integer",
                    "description": "Forward return horizon in days. Default: 21"
                },
                "top_n": {
                    "type": "integer", 
                    "description": "Number of top features to return. Default: 20"
                }
            },
            "required": ["factor"]
        }
    },
    {
        "name": "run_backtest",
        "description": "Run a backtest for a factor timing strategy with specified signal weights and parameters. Supports both premium mode (factor vs benchmark) and ratio mode (long factor/short benchmark pair trade).",
        "input_schema": {
            "type": "object",
            "properties": {
                "factor": {
                    "type": "string",
                    "description": "Factor ETF symbol",
                    "enum": ["VLUE", "SIZE", "QUAL", "USMV", "MTUM"]
                },
                "signal_weights": {
                    "type": "object",
                    "description": "Dict mapping feature names to weights. Positive weight = bullish when feature increases."
                },
                "threshold": {
                    "type": "number",
                    "description": "Signal threshold for taking positions. 0 = always in market. Default: 0"
                },
                "hold_period": {
                    "type": "integer",
                    "description": "Minimum holding period in days. Default: 0"
                },
                "start_date": {
                    "type": "string",
                    "description": "Start date in YYYY-MM-DD format. Default: 2016-01-01"
                },
                "target_type": {
                    "type": "string",
                    "description": "Target type: 'premium' (factor - benchmark) or 'ratio' (long factor/short benchmark pair trade)",
                    "enum": ["premium", "ratio"],
                    "default": "premium"
                },
                "benchmark": {
                    "type": "string",
                    "description": "Benchmark symbol for ratio calculation. Default: SPY"
                }
            },
            "required": ["factor", "signal_weights"]
        }
    },
    {
        "name": "initialize_smart_optimization",
        "description": "Initialize smart optimization with Bayesian and Genetic algorithms for systematic strategy improvement.",
        "input_schema": {
            "type": "object",
            "properties": {
                "factor": {
                    "type": "string",
                    "description": "Factor ETF symbol",
                    "enum": ["VLUE", "SIZE", "QUAL", "USMV", "MTUM"]
                },
                "available_features": {
                    "type": "array",
                    "description": "List of available features (uses default if not provided)",
                    "items": {"type": "string"}
                }
            },
            "required": ["factor"]
        }
    },
    {
        "name": "suggest_next_experiment",
        "description": "Get AI-guided suggestions for next experiment based on optimization history.",
        "input_schema": {
            "type": "object",
            "properties": {
                "factor": {
                    "type": "string",
                    "description": "Factor ETF symbol",
                    "enum": ["VLUE", "SIZE", "QUAL", "USMV", "MTUM"]
                },
                "phase": {
                    "type": "string",
                    "description": "Current optimization phase",
                    "enum": ["FEATURE_DISCOVERY", "WEIGHT_OPTIMIZATION", "PARAMETER_TUNING", "REFINEMENT"]
                }
            },
            "required": ["factor"]
        }
    },
    {
        "name": "validate_out_of_sample",
        "description": "Validate strategy performance on out-of-sample data to detect overfitting.",
        "input_schema": {
            "type": "object",
            "properties": {
                "factor": {
                    "type": "string",
                    "description": "Factor ETF symbol",
                    "enum": ["VLUE", "SIZE", "QUAL", "USMV", "MTUM"]
                },
                "signal_weights": {
                    "type": "object",
                    "description": "Signal weights to validate"
                },
                "threshold": {
                    "type": "number",
                    "description": "Signal threshold. Default: 0"
                }
            },
            "required": ["factor", "signal_weights"]
        }
    },
    {
        "name": "get_optimization_summary",
        "description": "Get comprehensive summary of optimization results including best configurations and performance metrics.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "create_derived_feature",
        "description": "Create a new derived feature from existing data (zscore, momentum, volatility, etc.).",
        "input_schema": {
            "type": "object",
            "properties": {
                "base_feature": {
                    "type": "string",
                    "description": "Base feature name to derive from (e.g., 'VIXCLS', 'SPY', 'DGS10')"
                },
                "transformation": {
                    "type": "string",
                    "description": "Transformation type",
                    "enum": ["ma_ratio", "roc", "zscore", "volatility", "momentum", "diff", "log_diff", "ema", "rsi", "rank"]
                },
                "window": {
                    "type": "integer",
                    "description": "Window size in days. Default: 21"
                },
                "feature_name": {
                    "type": "string",
                    "description": "Optional custom name for the new feature"
                }
            },
            "required": ["base_feature", "transformation"]
        }
    },
    {
        "name": "test_feature_predictive_power",
        "description": "Test the predictive power of a feature for factor prediction.",
        "input_schema": {
            "type": "object",
            "properties": {
                "feature_spec": {
                    "type": "object",
                    "description": "Feature specification: {\"base\": \"VIX\", \"transform\": \"zscore\", \"window\": 21} for derived feature or {\"features\": {\"VIX\": 0.5, \"DGS10\": -0.3}} for combined"
                },
                "target_factor": {
                    "type": "string", 
                    "description": "Factor to predict",
                    "enum": ["VLUE", "SIZE", "QUAL", "USMV", "MTUM"]
                },
                "forward_days": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Forward return periods. Default: [5, 10, 21, 63]"
                }
            },
            "required": ["feature_spec", "target_factor"]
        }
    },
    {
        "name": "get_existing_derived_features",
        "description": "Get all 69 existing derived features from analysis.py that Gemini uses. This provides access to the same feature set.",
        "input_schema": {
            "type": "object",
            "properties": {
                "start_date": {
                    "type": "string",
                    "description": "Start date in YYYY-MM-DD format. Default: 2016-01-01"
                }
            },
            "required": []
        }
    },
    {
        "name": "create_enhanced_features",
        "description": "Create advanced features missing from current set: VIX term structure, enhanced credit spreads, real economy, commodities, currency, bond volatility.",
        "input_schema": {
            "type": "object",
            "properties": {
                "start_date": {
                    "type": "string",
                    "description": "Start date in YYYY-MM-DD format. Default: 2016-01-01"
                }
            },
            "required": []
        }
    },
    {
        "name": "get_fed_documents",
        "description": "Get Federal Reserve documents with sentiment features.",
        "input_schema": {
            "type": "object",
            "properties": {
                "document_type": {
                    "type": "string",
                    "description": "Type of Fed document",
                    "enum": ["FOMC_statement", "FOMC_minutes", "Beige_Book", "speeches"]
                },
                "start_date": {
                    "type": "string",
                    "description": "Start date in YYYY-MM-DD format"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of documents. Default: 50"
                }
            },
            "required": ["document_type"]
        }
    },
    {
        "name": "record_experiment_result",
        "description": "Record the result of an experiment for enhanced learning and pattern recognition.",
        "input_schema": {
            "type": "object",
            "properties": {
                "weights": {
                    "type": "object",
                    "description": "Feature weights used"
                },
                "threshold": {
                    "type": "number",
                    "description": "Threshold used"
                },
                "hold_period": {
                    "type": "integer",
                    "description": "Hold period used"
                },
                "sharpe": {
                    "type": "number",
                    "description": "Resulting Sharpe ratio"
                },
                "max_drawdown": {
                    "type": "number",
                    "description": "Maximum drawdown"
                },
                "annual_return": {
                    "type": "number",
                    "description": "Annual return"
                },
                "volatility": {
                    "type": "number",
                    "description": "Strategy volatility"
                },
                "win_rate": {
                    "type": "number",
                    "description": "Win rate of trades"
                },
                "source_method": {
                    "type": "string",
                    "description": "Method that generated this config (bayesian, genetic, manual, etc.)"
                }
            },
            "required": ["weights", "threshold", "hold_period", "sharpe"]
        }
    },
    {
        "name": "enable_continuous_refinement",
        "description": "Enable autonomous continuous refinement mode for hands-free optimization.",
        "input_schema": {
            "type": "object",
            "properties": {
                "factor": {
                    "type": "string",
                    "description": "Factor to optimize (VLUE, SIZE, etc.)",
                    "enum": ["VLUE", "SIZE", "QUAL", "USMV", "MTUM"]
                },
                "target_sharpe": {
                    "type": "number",
                    "description": "Target Sharpe ratio to achieve. Default: 1.5"
                },
                "max_iterations": {
                    "type": "integer",
                    "description": "Maximum iterations before stopping. Default: 50"
                }
            },
            "required": ["factor"]
        }
    },
    {
        "name": "execute_autonomous_cycle",
        "description": "Execute one autonomous optimization cycle with AI decision making.",
        "input_schema": {
            "type": "object",
            "properties": {
                "factor": {
                    "type": "string",
                    "description": "Factor to optimize",
                    "enum": ["VLUE", "SIZE", "QUAL", "USMV", "MTUM"]
                }
            },
            "required": ["factor"]
        }
    },
    {
        "name": "deploy_best_strategy",
        "description": "Deploy the best optimized strategy when criteria are met (minimum Sharpe, confidence, robustness).",
        "input_schema": {
            "type": "object",
            "properties": {
                "factor": {
                    "type": "string",
                    "description": "Factor to deploy strategy for",
                    "enum": ["VLUE", "SIZE", "QUAL", "USMV", "MTUM"]
                },
                "minimum_sharpe": {
                    "type": "number",
                    "description": "Minimum Sharpe ratio required for deployment. Default: 0.5"
                },
                "confidence_threshold": {
                    "type": "number",
                    "description": "Minimum confidence score for deployment. Default: 0.8"
                }
            },
            "required": ["factor"]
        }
    },
    {
        "name": "auto_deploy_on_improvement",
        "description": "Automatically deploy strategy when significant improvement is detected.",
        "input_schema": {
            "type": "object",
            "properties": {
                "factor": {
                    "type": "string",
                    "description": "Factor to monitor for auto-deployment",
                    "enum": ["VLUE", "SIZE", "QUAL", "USMV", "MTUM"]
                },
                "auto_deploy": {
                    "type": "boolean",
                    "description": "Whether to actually deploy or just recommend. Default: true"
                },
                "minimum_improvement": {
                    "type": "number",
                    "description": "Minimum Sharpe improvement to trigger deployment. Default: 0.05"
                }
            },
            "required": ["factor"]
        }
    },
    {
        "name": "get_deployed_strategy",
        "description": "Get the currently deployed strategy configuration for a factor.",
        "input_schema": {
            "type": "object",
            "properties": {
                "factor": {
                    "type": "string",
                    "description": "Factor to get deployed strategy for",
                    "enum": ["VLUE", "SIZE", "QUAL", "USMV", "MTUM"]
                }
            },
            "required": ["factor"]
        }
    },
    {
        "name": "detect_overfitting_signals",
        "description": "Detect overfitting signals in a strategy configuration or backtest results.",
        "input_schema": {
            "type": "object",
            "properties": {
                "config": {
                    "type": "object",
                    "description": "Strategy configuration to check for overfitting"
                },
                "backtest_result": {
                    "type": "object",
                    "description": "Optional backtest results to validate"
                }
            },
            "required": ["config"]
        }
    },
    {
        "name": "apply_overfitting_filters",
        "description": "Apply overfitting filters to a configuration (feature count limits, weight caps).",
        "input_schema": {
            "type": "object",
            "properties": {
                "config": {
                    "type": "object",
                    "description": "Original configuration to filter"
                }
            },
            "required": ["config"]
        }
    },
    {
        "name": "validate_robustness",
        "description": "Validate strategy robustness across different time periods to detect overfitting.",
        "input_schema": {
            "type": "object",
            "properties": {
                "factor": {
                    "type": "string",
                    "description": "Factor to validate",
                    "enum": ["VLUE", "SIZE", "QUAL", "USMV", "MTUM"]
                },
                "config": {
                    "type": "object",
                    "description": "Configuration to validate"
                },
                "min_train_val_correlation": {
                    "type": "number",
                    "description": "Minimum correlation between train/val performance. Default: 0.5"
                }
            },
            "required": ["factor", "config"]
        }
    },
    {
        "name": "get_overfitting_summary",
        "description": "Get summary of overfitting patterns detected in current optimization session.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "diagnose_underperformance",
        "description": "Analyze periods where the strategy underperformed and diagnose WHY. Returns drawdown periods with feature values and position analysis to help improve the strategy.",
        "input_schema": {
            "type": "object",
            "properties": {
                "factor": {
                    "type": "string",
                    "description": "Factor ETF symbol",
                    "enum": ["VLUE", "SIZE", "QUAL", "USMV", "MTUM"]
                },
                "signal_weights": {
                    "type": "object",
                    "description": "Current signal weights to diagnose"
                },
                "threshold": {
                    "type": "number",
                    "description": "Signal threshold. Default: 0.03"
                },
                "target_type": {
                    "type": "string",
                    "description": "Target type: 'premium' or 'ratio'",
                    "enum": ["premium", "ratio"]
                }
            },
            "required": ["factor", "signal_weights"]
        }
    }
]

# System prompt for Claude - will be formatted with target_type
CLAUDE_SYSTEM_PROMPT_TEMPLATE = """You are an expert quantitative researcher specializing in factor investing and macro timing strategies.

Your task is to systematically optimize heuristic-based factor timing models that predict when to be long or short equity style factors (Value, Size, Quality, Min Vol, Momentum) relative to SPY.

TARGET MODE: {target_mode_description}

POSITION INTERPRETATION:
- **LONG position (+1)**: {long_description}
- **SHORT position (-1)**: {short_description}
- **FLAT position (0)**: No exposure

You have access to:
1. A comprehensive database with macro indicators (rates, Fed policy, sectors, volatility)
2. Current heuristic model implementations
3. Advanced backtesting and optimization tools
4. Smart optimization algorithms (Bayesian, Genetic)

KEY PRINCIPLES:
- Focus on empirically-validated signals (check ICs and statistical significance)
- Avoid overfitting - prefer simple, robust rules over complex models
- Walk-forward testing is essential for out-of-sample validation
- Factor premiums are often counter-cyclical (value works in distress, momentum in calm periods)
- Fed policy changes and interest rate regimes are primary drivers
- Use ensemble methods to improve robustness

OPTIMIZATION PROCESS:
1. **Feature Discovery**: Identify best predictive features using IC analysis
2. **Weight Optimization**: Find optimal feature weightings using smart algorithms
3. **Parameter Tuning**: Optimize thresholds and holding periods
4. **Validation**: Ensure robustness through out-of-sample testing

PERFORMANCE METRICS TO OPTIMIZE:
- Primary: Sharpe ratio (risk-adjusted returns)
- Secondary: Maximum drawdown, hit rate, total return
- Validation: Out-of-sample performance consistency

When suggesting improvements:
1. Start with current baseline performance
2. Identify which signal components work/don't work
3. Propose specific, testable modifications with clear rationale
4. Prioritize changes that improve risk-adjusted returns
5. Always validate with walk-forward testing"""

# Default system prompt (premium mode)
CLAUDE_SYSTEM_PROMPT = CLAUDE_SYSTEM_PROMPT_TEMPLATE.format(
    target_mode_description="PREMIUM MODE - Optimizing factor premium (VLUE - SPY) timing",
    long_description="Long the factor ETF (bullish on factor outperformance vs SPY)",
    short_description="Avoid the factor ETF (bearish on factor vs SPY)"
)


class ClaudeOptimizationAgent:
    """
    AI agent for optimizing factor timing heuristics using Claude.
    
    Uses Claude's advanced reasoning with tool use to:
    1. Systematically analyze model performance
    2. Discover predictive features through IC analysis
    3. Optimize signal weights using smart algorithms
    4. Validate strategies with out-of-sample testing
    """
    
    def __init__(
        self,
        model: str = "claude-sonnet-4-5",
        api_key: Optional[str] = None,
        verbose: bool = True,
        target_type: str = "premium",
        benchmark: str = "SPY",
    ):
        """
        Initialize the Claude optimization agent.

        Args:
            model: Claude model to use
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
            verbose: Print detailed progress
            target_type: "premium" (factor - benchmark) or "ratio" (long factor/short benchmark)
            benchmark: Benchmark symbol (default: SPY)
        """
        self.model_name = model
        self.verbose = verbose
        self.target_type = target_type
        self.benchmark = benchmark

        # Configure system prompt based on target type
        if target_type == "ratio":
            self.system_prompt = CLAUDE_SYSTEM_PROMPT_TEMPLATE.format(
                target_mode_description=f"RATIO MODE - Optimizing FACTOR/{benchmark} pair trade (long factor, short {benchmark})",
                long_description=f"LONG FACTOR / SHORT {benchmark} - profit when factor outperforms {benchmark}",
                short_description=f"SHORT FACTOR / LONG {benchmark} - profit when {benchmark} outperforms factor"
            )
        else:
            self.system_prompt = CLAUDE_SYSTEM_PROMPT_TEMPLATE.format(
                target_mode_description=f"PREMIUM MODE - Optimizing factor premium (factor - {benchmark}) timing",
                long_description=f"Long the factor ETF (bullish on factor outperformance vs {benchmark})",
                short_description=f"Avoid the factor ETF (bearish on factor vs {benchmark})"
            )

        # Configure Claude
        api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")

        self.client = anthropic.Anthropic(api_key=api_key)

        # Optimization state
        self.conversation_history = []
        self.optimization_results = {}

        AGENT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    def _call_tool(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """Execute a tool function and return results."""
        try:
            # Import tools dynamically to avoid dependency issues
            from .tools import (
                get_current_performance,
                analyze_feature_correlations, 
                run_backtest,
                create_derived_feature,
                test_feature_predictive_power,
                get_fed_documents,
            )
            from .tools.features import get_existing_derived_features, create_enhanced_features
            from .tools.strategy_tools import (
                initialize_smart_optimization,
                suggest_next_experiment,
                record_experiment_result,
                validate_out_of_sample,
                get_optimization_summary,
                enable_continuous_refinement,
                execute_autonomous_cycle,
                deploy_best_strategy,
                auto_deploy_on_improvement,
                get_deployed_strategy,
                detect_overfitting_signals,
                apply_overfitting_filters,
                validate_robustness,
                get_overfitting_summary,
                diagnose_underperformance,
            )
            
            tool_functions = {
                "get_current_performance": get_current_performance,
                "analyze_feature_correlations": analyze_feature_correlations, 
                "run_backtest": run_backtest,
                "initialize_smart_optimization": initialize_smart_optimization,
                "suggest_next_experiment": suggest_next_experiment,
                "validate_out_of_sample": validate_out_of_sample,
                "get_optimization_summary": get_optimization_summary,
                "record_experiment_result": record_experiment_result,
                "enable_continuous_refinement": enable_continuous_refinement,
                "execute_autonomous_cycle": execute_autonomous_cycle,
                "deploy_best_strategy": deploy_best_strategy,
                "auto_deploy_on_improvement": auto_deploy_on_improvement,
                "get_deployed_strategy": get_deployed_strategy,
                "detect_overfitting_signals": detect_overfitting_signals,
                "apply_overfitting_filters": apply_overfitting_filters,
                "validate_robustness": validate_robustness,
                "get_overfitting_summary": get_overfitting_summary,
                "diagnose_underperformance": diagnose_underperformance,
                "create_derived_feature": create_derived_feature,
                "test_feature_predictive_power": test_feature_predictive_power,
                "get_fed_documents": get_fed_documents,
                "get_existing_derived_features": get_existing_derived_features,
                "create_enhanced_features": create_enhanced_features,
            }
            
            if tool_name not in tool_functions:
                return {"error": f"Unknown tool: {tool_name}"}
            
            result = tool_functions[tool_name](**kwargs)
            return result if isinstance(result, dict) else {"result": result}
            
        except ImportError as e:
            return {"error": f"Tool import failed: {e}"}
        except Exception as e:
            return {"error": str(e)}
    
    def _send_message(self, message: str, max_tokens: int = 8192) -> Dict[str, Any]:
        """Send message to Claude and handle tool use."""
        try:
            if self.verbose:
                print(f"🔄 Sending message (max_tokens: {max_tokens})", flush=True)

            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=max_tokens,
                system=self.system_prompt,
                messages=[{"role": "user", "content": message}],
                tools=CLAUDE_TOOLS
            )
            
            if self.verbose:
                print(f"✅ Response received (stop_reason: {response.stop_reason})", flush=True)
                # Debug: show content types
                for block in response.content:
                    if block.type == "text":
                        text_preview = block.text[:200] if len(block.text) > 200 else block.text
                        print(f"   💬 Text: {text_preview}...", flush=True)
                    elif block.type == "tool_use":
                        print(f"   🔧 Tool call: {block.name}", flush=True)
            
            # Handle tool use
            if response.stop_reason == "tool_use":
                tool_results = []
                
                for block in response.content:
                    if block.type == "tool_use":
                        tool_name = block.name
                        tool_input = block.input
                        
                        if self.verbose:
                            print(f"🔧 Tool: {tool_name}")
                            if tool_input:
                                print(f"   Input: {json.dumps(tool_input, indent=2)}")
                        
                        # Execute tool
                        result = self._call_tool(tool_name, **tool_input)

                        # Print key results
                        if self.verbose and result:
                            if 'sharpe' in str(result).lower():
                                if isinstance(result, dict):
                                    if 'strategy' in result and 'sharpe' in result['strategy']:
                                        print(f"   📊 Sharpe: {result['strategy']['sharpe']:.3f}")
                                    elif 'sharpe' in result:
                                        print(f"   📊 Sharpe: {result['sharpe']:.3f}")
                                    elif 'best_sharpe' in result:
                                        print(f"   📊 Best Sharpe: {result['best_sharpe']:.3f}")

                        tool_results.append({
                            "tool_use_id": block.id,
                            "content": json.dumps(result, indent=2, default=str)
                        })
                
                # Send tool results back to Claude
                if tool_results:
                    follow_up = self.client.messages.create(
                        model=self.model_name,
                        max_tokens=max_tokens,
                        system=self.system_prompt,
                        messages=[
                            {"role": "user", "content": message},
                            {"role": "assistant", "content": response.content},
                            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": tr["tool_use_id"], "content": tr["content"]} for tr in tool_results]}
                        ]
                    )
                    return {"response": follow_up.content[0].text if follow_up.content else ""}
            
            # Regular text response
            return {"response": response.content[0].text if response.content else ""}

        except Exception as e:
            if self.verbose:
                print(f"❌ Error in _send_message: {e}")
                import traceback
                traceback.print_exc()
            return {"error": str(e)}

    def visualize_best_runs(self, factor: str, top_n: int = 5):
        """Visualize the best backtest runs with their parameters."""
        import matplotlib.pyplot as plt
        from agent.tools.backtest import run_backtest

        # Load history
        history = self._load_optimization_history(factor)
        all_runs = history.get('all_runs', [])

        if not all_runs:
            print("No runs to visualize. Run optimization first.")
            return

        # Sort by Sharpe and get top N
        sorted_runs = sorted(all_runs, key=lambda x: x.get('sharpe', -999), reverse=True)[:top_n]

        print(f"\n{'='*80}")
        print(f"📊 TOP {top_n} BACKTEST RESULTS FOR {factor}")
        print(f"{'='*80}\n")

        # Create figure with subplots
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))

        # Colors for different strategies
        colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#f39c12']

        equity_curves = []
        labels = []

        for i, run in enumerate(sorted_runs):
            sharpe = run.get('sharpe', 0)
            weights = run.get('weights', {})
            threshold = run.get('threshold', 0.05)
            hold_period = run.get('hold_period', 5)

            print(f"{'─'*80}")
            print(f"🏆 RANK #{i+1} | Sharpe: {sharpe:.3f}")
            print(f"{'─'*80}")
            print(f"   threshold = {threshold}")
            print(f"   hold_period = {hold_period}")
            print(f"   signal_weights = {{")
            for feat, w in weights.items():
                print(f'       "{feat}": {w},')
            print(f"   }}")

            # Run backtest to get equity curve
            try:
                result = run_backtest(
                    factor=factor,
                    signal_weights=weights,
                    threshold=threshold,
                    hold_period=hold_period
                )

                if 'equity_curve' in result:
                    equity_curves.append(result['equity_curve'])
                    labels.append(f"#{i+1} Sharpe={sharpe:.2f}")

                # Print performance metrics
                strat = result.get('strategy', {})
                bench = result.get('benchmark', {})
                print(f"\n   Performance:")
                print(f"   ├── Annual Return: {strat.get('annual_return', 0):.2f}%")
                print(f"   ├── Annual Vol: {strat.get('annual_vol', 0):.2f}%")
                print(f"   ├── Max Drawdown: {strat.get('max_drawdown', 0):.2f}%")
                print(f"   ├── Calmar Ratio: {strat.get('annual_return', 0) / abs(strat.get('max_drawdown', 1)):.2f}")
                print(f"   └── vs Benchmark: {strat.get('annual_return', 0) - bench.get('annual_return', 0):+.2f}%")

            except Exception as e:
                print(f"   Error running backtest: {e}")

        print(f"\n{'='*80}")

        # Plot equity curves
        if equity_curves:
            ax1 = axes[0]
            for i, (curve, label) in enumerate(zip(equity_curves, labels)):
                ax1.plot(curve.index, curve.values, label=label, color=colors[i % len(colors)], linewidth=2)

            ax1.set_title(f'{factor} - Top {len(equity_curves)} Strategies Equity Curves', fontsize=14, fontweight='bold')
            ax1.set_ylabel('Cumulative Return')
            ax1.legend(loc='upper left')
            ax1.grid(True, alpha=0.3)
            ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

            # Plot drawdowns
            ax2 = axes[1]
            for i, (curve, label) in enumerate(zip(equity_curves, labels)):
                rolling_max = curve.cummax()
                drawdown = (curve - rolling_max) / (1 + rolling_max) * 100
                ax2.fill_between(drawdown.index, drawdown.values, 0, alpha=0.3, color=colors[i % len(colors)], label=label)

            ax2.set_title('Drawdowns', fontsize=14, fontweight='bold')
            ax2.set_ylabel('Drawdown %')
            ax2.set_xlabel('Date')
            ax2.legend(loc='lower left')
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()

            # Save figure
            output_dir = Path(__file__).parent / "output"
            output_dir.mkdir(exist_ok=True)
            fig_path = output_dir / f"best_backtests_{factor}.png"
            plt.savefig(fig_path, dpi=150, bbox_inches='tight')
            print(f"\n📈 Chart saved to: {fig_path}")
            plt.show()

        # Print best config for copy-paste
        if sorted_runs:
            best = sorted_runs[0]
            print(f"\n{'='*80}")
            print("📋 BEST CONFIGURATION (copy-paste ready):")
            print(f"{'='*80}")
            print(f"""
# Best {factor} Strategy - Sharpe {best.get('sharpe', 0):.3f}
factor = "{factor}"
threshold = {best.get('threshold', 0.05)}
hold_period = {best.get('hold_period', 5)}

signal_weights = {{""")
            for feat, w in best.get('weights', {}).items():
                print(f'    "{feat}": {w},')
            print("}")

    def _get_history_file(self, factor: str) -> Path:
        """Get path to optimization history file for a factor."""
        output_dir = Path(__file__).parent / "output"
        output_dir.mkdir(exist_ok=True)
        # Include target_type in filename to separate ratio vs premium histories
        suffix = f"_{self.target_type}" if self.target_type != "premium" else ""
        return output_dir / f"optimization_history_{factor}{suffix}.json"

    def _load_optimization_history(self, factor: str) -> Dict[str, Any]:
        """Load previous optimization runs from disk."""
        history_file = self._get_history_file(factor)
        if history_file.exists():
            try:
                with open(history_file, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        return {
            "factor": factor,
            "best_sharpe": -999.0,
            "best_config": None,
            "all_runs": [],
            "last_updated": None
        }

    def _save_optimization_history(self, factor: str, history: Dict[str, Any]):
        """Save optimization history to disk."""
        history_file = self._get_history_file(factor)
        history["last_updated"] = datetime.now().isoformat()
        with open(history_file, 'w') as f:
            json.dump(history, f, indent=2, default=str)

    def _extract_config_from_response(self, response_text: str) -> Dict[str, Any]:
        """Extract a single configuration from model response text (legacy method)."""
        configs = self._extract_all_configs_from_response(response_text)
        return configs[0] if configs else None

    def _extract_all_configs_from_response(self, response_text: str) -> List[Dict[str, Any]]:
        """Extract ALL configurations from model response text (supports multiple configs per response)."""
        import re
        configs = []

        # Find all signal_weights blocks in the response
        # Pattern: signal_weights = { ... } or weights = { ... } or just { "feature": weight, ... }
        weights_blocks = re.findall(
            r'(?:signal_)?weights\s*[=:]\s*\{([^}]+)\}',
            response_text,
            re.DOTALL | re.IGNORECASE
        )

        # Also look for standalone dict patterns like {"feature": 0.5, ...}
        standalone_dicts = re.findall(
            r'\{(["\'][a-zA-Z_]+["\']:\s*-?[\d.]+(?:,\s*["\'][a-zA-Z_]+["\']:\s*-?[\d.]+)*)\}',
            response_text
        )
        weights_blocks.extend(standalone_dicts)

        for weights_str in weights_blocks:
            weights = {}
            # Parse key-value pairs with quotes
            for kv in re.findall(r'["\']([^"\']+)["\']:\s*(-?[\d.]+)', weights_str):
                feat_name = kv[0]
                # Filter out non-feature keys
                if feat_name.lower() not in ['threshold', 'hold_period', 'hold', 'sharpe', 'time_in_market']:
                    weights[feat_name] = float(kv[1])

            # Also try without quotes
            if not weights:
                for kv in re.findall(r'([a-zA-Z_][a-zA-Z0-9_]*):\s*(-?[\d.]+)', weights_str):
                    if kv[0].lower() not in ['threshold', 'hold_period', 'hold', 'sharpe', 'time_in_market']:
                        weights[kv[0]] = float(kv[1])

            if weights and len(weights) >= 2:  # Require at least 2 features
                config = {'weights': weights}

                # Look for associated threshold nearby in the text
                # Find position of this weights block
                pos = response_text.find(weights_str)
                context = response_text[max(0, pos-200):min(len(response_text), pos+len(weights_str)+200)]

                threshold_match = re.search(r'threshold[:\s=]+(-?[\d.]+)', context, re.IGNORECASE)
                if threshold_match:
                    try:
                        config['threshold'] = float(threshold_match.group(1))
                    except:
                        config['threshold'] = 0.03  # Default

                hold_match = re.search(r'hold[_\s]?period[:\s=]+(\d+)', context, re.IGNORECASE)
                if hold_match:
                    try:
                        config['hold_period'] = int(hold_match.group(1))
                    except:
                        config['hold_period'] = 5  # Default

                tim_match = re.search(r'[Tt]ime[_\s]?[Ii]n[_\s]?[Mm]arket[:\s=]+(\d+\.?\d*)%?', context)
                if tim_match:
                    try:
                        tim_val = float(tim_match.group(1))
                        config['time_in_market'] = tim_val / 100 if tim_val > 1 else tim_val
                    except:
                        pass

                # Avoid duplicates
                if not any(c.get('weights') == weights for c in configs):
                    configs.append(config)

        return configs

    def _get_feature_signature(self, weights: Dict[str, float]) -> frozenset:
        """Get a signature of features used (ignoring weights) for diversity calculation."""
        return frozenset(weights.keys())

    def _select_diverse_top_runs(self, runs: list, top_n: int = 10, diversity_penalty: float = 0.1) -> list:
        """
        Select top runs with diversity - not just highest Sharpe, but diverse feature combinations.

        Uses a greedy selection that penalizes runs with similar feature sets to already selected runs.
        """
        if not runs:
            return []

        # Filter out invalid runs (time_in_market < 50%)
        valid_runs = [r for r in runs if r.get('time_in_market', 1.0) >= 0.5]
        if not valid_runs:
            valid_runs = runs  # Fall back to all runs if none are valid

        # Sort by Sharpe initially
        sorted_runs = sorted(valid_runs, key=lambda x: x.get('sharpe', -999), reverse=True)

        selected = []
        selected_signatures = []

        for run in sorted_runs:
            if len(selected) >= top_n:
                break

            weights = run.get('weights', {})
            if not weights:
                continue

            signature = self._get_feature_signature(weights)

            # Calculate diversity score (how different from already selected)
            if selected_signatures:
                # Jaccard similarity with each selected run
                max_similarity = max(
                    len(signature & sig) / len(signature | sig) if signature | sig else 0
                    for sig in selected_signatures
                )
                # Penalize score for similar runs
                adjusted_sharpe = run.get('sharpe', 0) - (diversity_penalty * max_similarity)
            else:
                adjusted_sharpe = run.get('sharpe', 0)

            # Always include top 3 by raw Sharpe, then consider diversity
            if len(selected) < 3 or adjusted_sharpe > 0:
                selected.append(run)
                selected_signatures.append(signature)

        return selected

    def _format_previous_runs(self, runs: list, top_n: int = 10) -> str:
        """Format previous runs for the prompt - now with diversity selection."""
        if not runs:
            return "No previous runs recorded."

        # Select diverse top runs instead of just top by Sharpe
        diverse_runs = self._select_diverse_top_runs(runs, top_n=top_n)

        lines = ["## TOP DIVERSE RUNS (best Sharpe + feature diversity):"]
        lines.append("(⚠️ = time_in_market < 50% - INVALID)")
        for i, run in enumerate(diverse_runs, 1):
            sharpe = run.get('sharpe', 0)
            weights = run.get('weights', {})
            threshold = run.get('threshold', 0)
            time_in_market = run.get('time_in_market', 1.0)
            valid = "✓" if time_in_market >= 0.5 else "⚠️"
            weights_str = ", ".join([f"{k}: {v}" for k, v in list(weights.items())[:3]])
            lines.append(f"{i}. {valid} Sharpe={sharpe:.3f} | th={threshold} | TIM={time_in_market:.0%} | {weights_str}...")

        return "\n".join(lines)

    def optimize_factor(
        self,
        factor: str,
        max_iterations: int = 30,
        use_smart_optimization: bool = True,
        oos_start_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Optimize a factor timing strategy using Claude with conversation history.
        Persists results across runs. OOS evaluation happens only at the end.

        Args:
            factor: Factor ETF symbol
            max_iterations: Maximum optimization iterations
            use_smart_optimization: Use smart optimization techniques
            oos_start_date: OOS period start date (e.g., "2024-01-01"). Final evaluation only.
        """
        # Load previous optimization history
        opt_history = self._load_optimization_history(factor)

        # Determine target mode description
        if self.target_type == "ratio":
            target_mode = f"RATIO MODE: Long {factor} / Short {self.benchmark}"
            position_desc = f"Long={factor}/Short{self.benchmark}, Short=Short{factor}/Long{self.benchmark}"
        else:
            target_mode = f"PREMIUM MODE: {factor} - {self.benchmark}"
            position_desc = f"Long={factor}, Short=avoid {factor}"

        print(f"🧠 CLAUDE OPTIMIZATION: {factor}", flush=True)
        print(f"📊 Target Mode: {target_mode}", flush=True)
        print(f"📍 Positions: {position_desc}", flush=True)
        print(f"Max iterations: {max_iterations}", flush=True)
        if oos_start_date:
            print(f"📅 OOS Evaluation: {oos_start_date}+ (evaluated at end only)", flush=True)
        print(f"Previous runs loaded: {len(opt_history.get('all_runs', []))}", flush=True)
        if opt_history.get('best_sharpe', -999) > -999:
            print(f"Previous best Sharpe: {opt_history['best_sharpe']:.3f}", flush=True)
        print("=" * 60, flush=True)
        print("\n📋 COMPLEX RULES ENABLED:", flush=True)
        print("   - Interactions: feat1_X_feat2", flush=True)
        print("   - AND/OR logic: feat1_AND_feat2, feat1_OR_feat2", flush=True)
        print("   - Conditionals: IF_feat1_THEN_feat2", flush=True)
        print("   - Thresholds: feat1_GT_0.5, feat1_LT_0", flush=True)
        print("   - Transforms: feat1_SQ, feat1_INV, feat1_ABS, feat1_SIGN", flush=True)
        print("=" * 60, flush=True)

        # Track best results - start from previous best
        best_sharpe = opt_history.get('best_sharpe', -999.0)
        best_config = opt_history.get('best_config', None)
        all_runs = opt_history.get('all_runs', [])
        conversation_history = []

        # Format previous runs for context
        previous_runs_context = self._format_previous_runs(all_runs)

        # Determine ratio mode context for prompts
        if self.target_type == "ratio":
            ratio_context = f"""
TARGET: Optimize {factor}/{self.benchmark} PAIR TRADE (long {factor}, short {self.benchmark})
- LONG position (+1): Long {factor}, Short {self.benchmark} - profit when {factor} outperforms
- SHORT position (-1): Short {factor}, Long {self.benchmark} - profit when {self.benchmark} outperforms

Use target_type="ratio" in run_backtest calls!
"""
        else:
            ratio_context = f"""
TARGET: Optimize {factor} factor premium ({factor} - {self.benchmark})
- LONG position (+1): Long {factor} (bullish on factor)
- SHORT position (-1): Avoid {factor} (bearish on factor)
"""

        # Note: OOS evaluation happens only at the end, not during optimization

        # Initial analysis - customize based on whether we have previous runs
        if best_config and best_sharpe > 0:
            # Continue from previous best
            best_weights_str = json.dumps(best_config.get('weights', {}), indent=2)
            initial_prompt = f"""Continue optimizing {factor} factor timing. Maximize Sharpe ratio.
{ratio_context}

## CURRENT BEST CONFIG (Sharpe {best_sharpe:.3f}) - START FROM HERE:
```python
signal_weights = {best_weights_str}
threshold = {best_config.get('threshold', 0.03)}
hold_period = {best_config.get('hold_period', 5)}
```

{previous_runs_context}

YOUR TASK: Improve on the best config above. Try:
1. Small weight adjustments (+/- 0.1 to 0.3)
2. Adding one new feature or complex rule
3. Removing weak features
4. Adjusting threshold slightly

Run 3 variations of the best config and report results.
DO NOT start from scratch - build on what works!"""
        else:
            # Fresh start
            initial_prompt = f"""Optimize {factor} factor timing to maximize Sharpe ratio.
{ratio_context}
{previous_runs_context}

STEP 1: Call these tools NOW:
1. get_current_performance(factor="{factor}") - get baseline
2. analyze_feature_correlations(factor="{factor}", horizon=21, top_n=15) - find predictive features

After getting results, tell me the baseline Sharpe and top 5 features by IC.
Learn from previous runs above - avoid configurations that performed poorly!"""

        print("📨 Sending initial analysis request...", flush=True)
        response = self._send_message_with_history(initial_prompt, conversation_history)
        print("📊 Initial Analysis:", flush=True)
        print(response.get("response", "")[:500], flush=True)

        # Iterative optimization with conversation history
        for iteration in range(max_iterations):
            print(f"\n📈 ITERATION {iteration + 1}/{max_iterations} | Best Sharpe: {best_sharpe:.3f}", flush=True)

            # Simple, direct prompt for each iteration
            # Add target_type to backtest calls
            target_type_param = f', target_type="{self.target_type}"' if self.target_type == "ratio" else ""

            # Complex rules syntax block - used in all prompts after iteration 0
            complex_rules_syntax = """
## COMPLEX RULE SYNTAX (use these for non-linear strategies!):

1. **INTERACTIONS** (multiply features): "feat1_X_feat2"
   Example: "yield_curve_X_XLY_ma_ratio": 0.5

2. **AND LOGIC** (both must be positive): "feat1_AND_feat2"
   Example: "yield_curve_AND_rate_roc_63d": 0.4

3. **OR LOGIC** (either positive): "feat1_OR_feat2"
   Example: "XLY_ma_ratio_OR_IWM_ma_ratio": 0.3

4. **CONDITIONAL IF/THEN**: "IF_condition_THEN_value"
   Example: "IF_yield_curve_THEN_XLY_ma_ratio": 0.5

5. **THRESHOLDS**: "feat_GT_0.5" or "feat_LT_0"
   Example: "VIX_GT_0.3": 0.4

6. **TRANSFORMS**: "_SQ" (squared), "_INV" (inverse), "_ABS", "_SIGN"
   Example: "yield_curve_SQ": 0.3
"""

            if iteration == 0:
                optimization_prompt = f"""Now run 3 DIFFERENT backtests with the top features you found.
{ratio_context}

Run 3 backtests with DIFFERENT configurations:
1. **Conservative**: Top 3 features with moderate weights
2. **Aggressive**: Top 5 features with stronger weights
3. **Complex**: Use at least one complex rule (interaction, AND/OR, IF/THEN)
{complex_rules_syntax}

For EACH backtest, use:
- factor: "{factor}"
- threshold: LOW (0.02-0.05) to ensure >50% time in market
- hold_period: 5 (WEEKLY REBALANCING)
- target_type: "{self.target_type}"

⚠️ CONSTRAINT: ALL strategies MUST have TIME IN MARKET > 50%!

EXECUTE ALL 3 BACKTESTS NOW and report each Sharpe ratio AND time_in_market."""
            elif iteration < 5:
                optimization_prompt = f"""Current best Sharpe: {best_sharpe:.3f}. Maximize it!
{ratio_context}
{self._format_previous_runs(all_runs, top_n=10)}

Run 3 DIFFERENT configurations to beat the current best:
1. **Variation**: Modify weights of best-performing features
2. **New combo**: Try different feature combination not tested yet
3. **Complex**: Use interaction or conditional rules
{complex_rules_syntax}

⚠️ CONSTRAINTS:
- TIME IN MARKET MUST BE > 50%
- Use threshold <= 0.05
- ALWAYS use target_type="{self.target_type}"

EXECUTE ALL 3 BACKTESTS NOW. Report each config's Sharpe and time_in_market."""
            else:
                # After iteration 5, focus on complex strategies with 3 configs per cycle
                optimization_prompt = f"""Current best Sharpe: {best_sharpe:.3f}. Maximize it!
{ratio_context}
{self._format_previous_runs(all_runs, top_n=10)}

Simple linear combinations may have plateaued. Run 3 COMPLEX strategies:

1. **Interaction-based**: Use at least 2 interaction terms (feat1_X_feat2)
2. **Conditional**: Use IF/THEN or AND/OR logic
3. **Hybrid**: Mix simple features with one complex transformation
{complex_rules_syntax}

## EXAMPLE COMPLEX STRATEGIES:
```python
# Strategy 1: Interaction-focused
{{"yield_curve_X_XLY_ma_ratio": 0.5, "rate_roc_63d_X_IWM_ma_ratio": 0.4, "TLT_mom_63d": -0.3}}

# Strategy 2: Conditional logic
{{"IF_yield_curve_THEN_XLY_ma_ratio": 0.5, "rate_roc_63d_AND_IWM_ma_ratio": 0.4}}

# Strategy 3: Hybrid
{{"yield_curve": 0.3, "XLY_ma_ratio_SQ": 0.4, "rate_roc_63d": 0.3}}
```

⚠️ CONSTRAINTS:
- TIME IN MARKET MUST BE > 50%
- Use threshold <= 0.05
- ALWAYS use target_type="{self.target_type}"

EXECUTE ALL 3 BACKTESTS NOW. Report each config's Sharpe and time_in_market."""

            print("   📨 Sending optimization request...", flush=True)
            response = self._send_message_with_history(optimization_prompt, conversation_history)
            response_text = response.get("response", "")
            backtest_results = response.get("backtest_results", [])
            print(response_text[:400] if len(response_text) > 400 else response_text, flush=True)

            # Process backtest results directly from tool calls (much more reliable than text extraction)
            configs_added = 0
            for bt_result in backtest_results:
                sharpe = bt_result.get('sharpe', 0)
                time_in_market = bt_result.get('time_in_market', 1.0)
                weights = bt_result.get('signal_weights', {})

                if not weights or len(weights) < 2:
                    continue

                # Skip invalid configs (time in market < 50%)
                if time_in_market < 0.5:
                    continue

                config = {
                    'weights': weights,
                    'threshold': bt_result.get('threshold', 0.03),
                    'hold_period': bt_result.get('hold_period', 5),
                    'sharpe': sharpe,
                    'time_in_market': time_in_market,
                    'iteration': iteration + 1,
                    'timestamp': datetime.now().isoformat(),
                }

                # Check if this is a new best
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_config = config.copy()
                    print(f"   🏆 NEW BEST: {best_sharpe:.3f}", flush=True)

                all_runs.append(config)
                configs_added += 1

            if configs_added > 0:
                print(f"   📝 Added {configs_added} config(s) to history", flush=True)

            # Save history periodically (every 3 iterations now since we have more configs)
            if iteration % 3 == 0:
                print(f"   💾 Saving history ({len(all_runs)} runs)...", flush=True)
                self._save_optimization_history(factor, {
                    "factor": factor,
                    "best_sharpe": best_sharpe,
                    "best_config": best_config,
                    "all_runs": all_runs
                })

        # Save final history
        self._save_optimization_history(factor, {
            "factor": factor,
            "best_sharpe": best_sharpe,
            "best_config": best_config,
            "all_runs": all_runs
        })

        # Final summary
        final_prompt = f"""Optimization complete. Best in-sample Sharpe achieved: {best_sharpe:.3f}

Call get_optimization_summary() and provide:
1. The best configuration found (weights, threshold, hold_period)
2. Key insights about what works for {factor}
3. Whether this strategy is robust or potentially overfit"""

        final_response = self._send_message_with_history(final_prompt, conversation_history)

        # OOS Evaluation - only at the end if oos_start_date is specified
        oos_sharpe = None
        oos_result = None
        if oos_start_date and best_config:
            print("\n" + "=" * 60)
            print(f"📊 OUT-OF-SAMPLE EVALUATION ({oos_start_date}+)")
            print("=" * 60)
            try:
                from agent.tools.backtest import run_backtest
                oos_result = run_backtest(
                    factor=factor,
                    signal_weights=best_config.get('weights', {}),
                    threshold=best_config.get('threshold', 0.03),
                    hold_period=best_config.get('hold_period', 5),
                    target_type=self.target_type,
                    benchmark=self.benchmark,
                    oos_start_date=oos_start_date,
                )
                oos_sharpe = oos_result.get('strategy', {}).get('sharpe', None)
                oos_return = oos_result.get('strategy', {}).get('annual_return', 0)
                oos_drawdown = oos_result.get('strategy', {}).get('max_drawdown', 0)
                oos_time_in_market = oos_result.get('time_in_market', 0)

                print(f"   In-Sample Sharpe:  {best_sharpe:.3f}")
                print(f"   OOS Sharpe:        {oos_sharpe:.3f}")
                print(f"   OOS Annual Return: {oos_return:.2f}%")
                print(f"   OOS Max Drawdown:  {oos_drawdown:.2f}%")
                print(f"   OOS Time in Mkt:   {oos_time_in_market:.1%}")

                # Warn if there's a big gap (potential overfitting)
                if best_sharpe > 0 and oos_sharpe is not None:
                    degradation = (best_sharpe - oos_sharpe) / best_sharpe * 100
                    if degradation > 50:
                        print(f"   ⚠️  WARNING: {degradation:.0f}% Sharpe degradation - potential overfitting!")
                    elif degradation > 25:
                        print(f"   ⚠️  CAUTION: {degradation:.0f}% Sharpe degradation")
                    else:
                        print(f"   ✅ Good generalization ({degradation:.0f}% degradation)")
            except Exception as e:
                print(f"   ❌ OOS evaluation failed: {e}")

        if self.verbose:
            print("\n" + "=" * 60)
            print("🎯 OPTIMIZATION COMPLETE")
            print("=" * 60)
            print(f"\n📊 BEST IN-SAMPLE SHARPE: {best_sharpe:.3f}")
            if oos_sharpe is not None:
                print(f"📊 OOS SHARPE ({oos_start_date}+): {oos_sharpe:.3f}")
            print(f"📁 History saved to: {self._get_history_file(factor)}")
            print(f"📈 Total runs recorded: {len(all_runs)}")

            if best_config:
                print("\n" + "=" * 60)
                print("🏆 BEST CONFIGURATION:")
                print("=" * 60)
                print(f"\nfactor = \"{factor}\"")
                print(f"threshold = {best_config.get('threshold', 0.05)}")
                print(f"hold_period = {best_config.get('hold_period', 5)}")
                print("\nsignal_weights = {")
                for feat, weight in best_config.get('weights', {}).items():
                    print(f'    "{feat}": {weight},')
                print("}")
                print("\n# To run this config:")
                print(f"# python -m agent.tools.backtest run_backtest --factor {factor} \\")
                print(f"#   --threshold {best_config.get('threshold', 0.05)} \\")
                print(f"#   --hold_period {best_config.get('hold_period', 5)}")

            print("\n" + "=" * 60)
            print(final_response.get("response", "")[:1000])

        result = {
            "factor": factor,
            "iterations": max_iterations,
            "best_sharpe": best_sharpe,
            "best_config": best_config,
            "all_runs_count": len(all_runs),
            "history_file": str(self._get_history_file(factor)),
            "final_response": final_response.get("response", ""),
            "timestamp": datetime.now().isoformat()
        }

        # Add OOS results if available
        if oos_sharpe is not None:
            result["oos_start_date"] = oos_start_date
            result["oos_sharpe"] = oos_sharpe
            result["oos_result"] = oos_result

        return result

    def _send_message_with_history(self, message: str, history: list, max_tokens: int = 16384) -> Dict[str, Any]:
        """Send message with conversation history for continuity."""
        try:
            # Log input
            log_interaction("INPUT", message)

            # Collect backtest results during this conversation turn
            backtest_results = []

            # Add user message to history
            history.append({"role": "user", "content": message})

            if self.verbose:
                print(f"🔄 Sending message...", flush=True)

            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=max_tokens,
                system=self.system_prompt,
                messages=history,
                tools=CLAUDE_TOOLS
            )

            if self.verbose:
                print(f"✅ Response (stop_reason: {response.stop_reason})", flush=True)

            # Handle tool use with loop for multi-turn
            while response.stop_reason == "tool_use":
                tool_results = []
                assistant_content = response.content

                for block in response.content:
                    if block.type == "tool_use":
                        tool_name = block.name
                        tool_input = block.input

                        if self.verbose:
                            print(f"   🔧 {tool_name}", flush=True)

                        # Log tool call
                        log_interaction("TOOL_CALL", f"{tool_name}\nInput: {json.dumps(tool_input, indent=2, default=str)}")

                        result = self._call_tool(tool_name, **tool_input)

                        # Log tool result
                        log_interaction("TOOL_RESULT", f"{tool_name}\nOutput: {json.dumps(result, indent=2, default=str)[:5000]}")

                        # Print Sharpe results and collect backtest results
                        if result and isinstance(result, dict):
                            if 'strategy' in result and 'sharpe' in result.get('strategy', {}):
                                sharpe = result['strategy']['sharpe']
                                if self.verbose:
                                    print(f"      📊 Sharpe: {sharpe:.3f}", flush=True)
                                # Collect backtest result with config
                                backtest_results.append({
                                    'sharpe': sharpe,
                                    'time_in_market': result.get('time_in_market', 1.0),
                                    'signal_weights': tool_input.get('signal_weights', {}),
                                    'threshold': tool_input.get('threshold', 0),
                                    'hold_period': tool_input.get('hold_period', 5),
                                    'annual_return': result['strategy'].get('annual_return', 0),
                                    'max_drawdown': result['strategy'].get('max_drawdown', 0),
                                })
                            elif 'sharpe' in result:
                                if self.verbose:
                                    print(f"      📊 Sharpe: {result['sharpe']:.3f}", flush=True)

                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": json.dumps(result, indent=2, default=str)[:10000]  # Limit size
                        })

                # Add assistant response and tool results to history
                history.append({"role": "assistant", "content": assistant_content})
                history.append({"role": "user", "content": tool_results})

                # Continue conversation
                response = self.client.messages.create(
                    model=self.model_name,
                    max_tokens=max_tokens,
                    system=self.system_prompt,
                    messages=history,
                    tools=CLAUDE_TOOLS
                )

                if self.verbose:
                    print(f"   ↪️ Continue (stop_reason: {response.stop_reason})", flush=True)

            # Extract final text response
            final_text = ""
            for block in response.content:
                if hasattr(block, 'text'):
                    final_text += block.text

            # Log output
            log_interaction("OUTPUT", final_text)

            # Add final assistant response to history
            history.append({"role": "assistant", "content": response.content})

            return {"response": final_text, "backtest_results": backtest_results}

        except Exception as e:
            if self.verbose:
                print(f"❌ Error: {e}", flush=True)
            return {"error": str(e)}

    def _optimize_factor_old(
        self,
        factor: str,
        max_iterations: int = 30,
        target_sharpe: float = 0.5,
        use_smart_optimization: bool = True,
    ) -> Dict[str, Any]:
        """OLD VERSION - kept for reference."""
        pass

        # Final validation
        final_prompt = f"""
Complete the optimization for {factor}:

1. Get the final optimization summary
2. Validate the best strategy with validate_out_of_sample  
3. Provide a comprehensive summary of:
   - Best configuration found
   - Performance metrics (Sharpe, drawdown, hit rate)
   - Key insights about what works for {factor}
   - Recommendations for implementation
        """
        
        final_response = self._send_message(final_prompt)
        
        if self.verbose:
            print("\n🎯 OPTIMIZATION COMPLETE")
            print("=" * 60)
            print(final_response.get("response", ""))
        
        return {
            "factor": factor,
            "iterations": max_iterations,
            "final_response": final_response.get("response", ""),
            "timestamp": datetime.now().isoformat()
        }
    
    def interactive_mode(self):
        """Start interactive mode for manual optimization."""
        print("🤖 Claude Factor Optimization Agent - Interactive Mode")
        print("Available factors:", ", ".join(FACTORS))
        print("Type 'quit' to exit, 'help' for commands")
        print("-" * 50)
        
        while True:
            try:
                user_input = input("\n> ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                
                if user_input.lower() in ['help', 'h']:
                    print("\nCommands:")
                    print("  optimize <factor>  - Start optimization for factor")
                    print("  analyze <factor>   - Analyze factor performance") 
                    print("  features <factor>  - Show top predictive features")
                    print("  quit              - Exit")
                    continue
                
                if user_input.startswith('optimize '):
                    factor = user_input.split(' ')[1].upper()
                    if factor in FACTORS:
                        self.optimize_factor(factor)
                    else:
                        print(f"Invalid factor. Choose from: {FACTORS}")
                    continue
                
                # Send any other input to Claude
                response = self._send_message(user_input)
                print("\n🤖 Claude:", response.get("response", response.get("error", "No response")))
                
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")


def main():
    """CLI entry point for Claude optimization."""
    import argparse

    parser = argparse.ArgumentParser(description="Claude Factor Optimization Agent")
    parser.add_argument("--factor", choices=FACTORS, help="Factor to optimize")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    parser.add_argument("--max-iterations", type=int, default=30, help="Max iterations")
    parser.add_argument("--model", default="claude-sonnet-4-5", help="Claude model")
    parser.add_argument("--target-type", choices=["premium", "ratio"], default="premium",
                       help="Target type: 'premium' (factor - benchmark) or 'ratio' (long factor/short benchmark)")
    parser.add_argument("--benchmark", default="SPY", help="Benchmark symbol for ratio mode")
    parser.add_argument("--oos-start-date", type=str, default=None,
                       help="OOS period start date in YYYY-MM-DD format (e.g., '2024-01-01'). Evaluated at end only.")

    args = parser.parse_args()

    # Initialize agent with target type
    agent = ClaudeOptimizationAgent(
        model=args.model,
        target_type=args.target_type,
        benchmark=args.benchmark
    )

    if args.interactive:
        agent.interactive_mode()
    elif args.factor:
        agent.optimize_factor(
            factor=args.factor,
            max_iterations=args.max_iterations,
            oos_start_date=args.oos_start_date
        )
    else:
        print("Specify --factor or --interactive")


if __name__ == "__main__":
    main()