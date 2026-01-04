"""
Heuristic Optimization Agent
============================

An AI agent powered by Google Gemini that analyzes and improves 
the performance of model_heuristics factor timing strategies.

Uses the Google Generative AI SDK with function calling.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

import google.generativeai as genai

from .config import (
    DEFAULT_MODEL,
    MAX_TOKENS,
    TEMPERATURE,
    SYSTEM_PROMPT,
    FACTORS,
    AGENT_OUTPUT_DIR,
)
from .tools import (
    get_current_performance,
    analyze_feature_correlations,
    get_available_features,
    run_backtest,
    run_parameter_sweep,
    compare_strategies,
    generate_new_signal,
    test_signal_combination,
    optimize_weights,
    # Database tools
    list_database_tables,
    list_economic_series,
    list_market_symbols,
    get_economic_series,
    get_market_prices,
    get_fed_documents,
    get_economic_calendar,
    run_custom_query,
    # Code modification tools
    list_model_scripts,
    read_file,
    write_file,
    search_replace_in_file,
    insert_code_at_line,
    run_script,
    update_signal_weights,
    lint_file,
    # Feature engineering tools
    create_derived_feature,
    combine_features,
    test_feature_predictive_power,
    list_available_base_features,
    # Smart optimization strategy tools
    initialize_smart_optimization,
    suggest_next_experiment,
    record_experiment_result,
    get_genetic_population,
    evolve_genetic_population,
    build_ensemble,
    test_ensemble,
    get_reflection_prompt,
    validate_out_of_sample,
    get_optimization_summary,
)
from .tools.optimization import suggest_improvements


# Tool definitions for Gemini (using function declarations)
TOOL_FUNCTIONS = [
    {
        "name": "get_current_performance",
        "description": "Get the current heuristic model performance for a factor. Returns strategy Sharpe, benchmark Sharpe, hit rate, drawdowns, and signal statistics.",
        "parameters": {
            "type": "object",
            "properties": {
                "factor": {
                    "type": "string",
                    "description": "Factor ETF symbol (VLUE, SIZE, QUAL, USMV, MTUM)",
                    "enum": ["VLUE", "SIZE", "QUAL", "USMV", "MTUM"],
                },
                "start_date": {
                    "type": "string",
                    "description": "Start date in YYYY-MM-DD format. Default: 2016-01-01",
                },
            },
            "required": ["factor"],
        },
    },
    {
        "name": "analyze_feature_correlations",
        "description": "Analyze feature correlations (Information Coefficient) with forward factor premium returns. Returns top features ranked by predictive power.",
        "parameters": {
            "type": "object",
            "properties": {
                "factor": {
                    "type": "string",
                    "description": "Factor ETF symbol",
                    "enum": ["VLUE", "SIZE", "QUAL", "USMV", "MTUM"],
                },
                "horizon": {
                    "type": "integer",
                    "description": "Forward return horizon in days. Default: 21 (1 month)",
                },
                "top_n": {
                    "type": "integer",
                    "description": "Number of top features to return. Default: 20",
                },
            },
            "required": ["factor"],
        },
    },
    {
        "name": "get_available_features",
        "description": "Get all available features from the database, categorized by type (economic series, market symbols, fed documents).",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "run_backtest",
        "description": "Run a backtest for a factor timing strategy with specified signal weights and parameters.",
        "parameters": {
            "type": "object",
            "properties": {
                "factor": {
                    "type": "string",
                    "description": "Factor ETF symbol",
                    "enum": ["VLUE", "SIZE", "QUAL", "USMV", "MTUM"],
                },
                "signal_weights": {
                    "type": "object",
                    "description": "Dict mapping feature names to weights. Positive weight = bullish when feature increases.",
                },
                "threshold": {
                    "type": "number",
                    "description": "Signal threshold for taking positions. 0 = always in market. Default: 0",
                },
                "hold_period": {
                    "type": "integer",
                    "description": "Minimum holding period in days. Default: 0",
                },
            },
            "required": ["factor"],
        },
    },
    {
        "name": "run_parameter_sweep",
        "description": "Run a parameter sweep over threshold and holding period values to find optimal parameters.",
        "parameters": {
            "type": "object",
            "properties": {
                "factor": {
                    "type": "string",
                    "description": "Factor ETF symbol",
                    "enum": ["VLUE", "SIZE", "QUAL", "USMV", "MTUM"],
                },
                "thresholds": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "List of threshold values to test. Default: [0.0, 0.1, 0.2, 0.3, 0.5]",
                },
                "hold_periods": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "List of holding periods to test. Default: [0, 5, 10, 21]",
                },
            },
            "required": ["factor"],
        },
    },
    {
        "name": "compare_strategies",
        "description": "Compare multiple signal weight configurations and generate a comparison plot.",
        "parameters": {
            "type": "object",
            "properties": {
                "factor": {
                    "type": "string",
                    "description": "Factor ETF symbol",
                    "enum": ["VLUE", "SIZE", "QUAL", "USMV", "MTUM"],
                },
                "strategies": {
                    "type": "object",
                    "description": "Dict mapping strategy names to weight dicts",
                },
            },
            "required": ["factor", "strategies"],
        },
    },
    {
        "name": "generate_new_signal",
        "description": "Generate a new signal based on top correlated features using IC-weighted approach.",
        "parameters": {
            "type": "object",
            "properties": {
                "factor": {
                    "type": "string",
                    "description": "Factor ETF symbol",
                    "enum": ["VLUE", "SIZE", "QUAL", "USMV", "MTUM"],
                },
                "top_n_features": {
                    "type": "integer",
                    "description": "Number of top features to include. Default: 5",
                },
                "use_ic_weights": {
                    "type": "boolean",
                    "description": "Weight features by their IC. Default: true",
                },
            },
            "required": ["factor"],
        },
    },
    {
        "name": "test_signal_combination",
        "description": "Test a specific feature weight combination with given parameters.",
        "parameters": {
            "type": "object",
            "properties": {
                "factor": {
                    "type": "string",
                    "description": "Factor ETF symbol",
                    "enum": ["VLUE", "SIZE", "QUAL", "USMV", "MTUM"],
                },
                "feature_weights": {
                    "type": "object",
                    "description": "Dict mapping feature names to weights",
                },
                "threshold": {
                    "type": "number",
                    "description": "Signal threshold. Default: 0",
                },
                "hold_period": {
                    "type": "integer",
                    "description": "Holding period in days. Default: 0",
                },
            },
            "required": ["factor", "feature_weights"],
        },
    },
    {
        "name": "suggest_improvements",
        "description": "Analyze current strategy and suggest specific improvements based on feature analysis and parameter tuning.",
        "parameters": {
            "type": "object",
            "properties": {
                "factor": {
                    "type": "string",
                    "description": "Factor ETF symbol",
                    "enum": ["VLUE", "SIZE", "QUAL", "USMV", "MTUM"],
                },
                "current_sharpe": {
                    "type": "number",
                    "description": "Current strategy Sharpe ratio for comparison",
                },
            },
            "required": ["factor", "current_sharpe"],
        },
    },
    # === DATABASE ACCESS TOOLS ===
    {
        "name": "list_database_tables",
        "description": "List all tables in the database with row counts. Use this to understand what data is available.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "list_economic_series",
        "description": "List all available economic series (FRED data) with observation counts and date ranges. Includes series like interest rates, unemployment, inflation, etc.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "list_market_symbols",
        "description": "List all available market symbols (ETFs, indices) with trading day counts. Includes factors (VLUE, SIZE, etc.), sectors (XLB, XLK, etc.), and indices.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "get_economic_series",
        "description": "Get actual data for specific economic series. Returns recent values.",
        "parameters": {
            "type": "object",
            "properties": {
                "series_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of FRED series IDs (e.g., ['DGS10', 'UNRATE', 'VIXCLS'])",
                },
                "start_date": {
                    "type": "string",
                    "description": "Start date in YYYY-MM-DD format. Default: 2010-01-01",
                },
            },
            "required": ["series_ids"],
        },
    },
    {
        "name": "get_market_prices",
        "description": "Get price data for specific market symbols.",
        "parameters": {
            "type": "object",
            "properties": {
                "symbols": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of symbols (e.g., ['SPY', 'VLUE', 'XLB'])",
                },
                "start_date": {
                    "type": "string",
                    "description": "Start date in YYYY-MM-DD format. Default: 2010-01-01",
                },
            },
            "required": ["symbols"],
        },
    },
    {
        "name": "get_fed_documents",
        "description": "Get Fed communication data with extracted sentiment features (policy stance, growth score, etc.). Includes FOMC Statements, FOMC Minutes, Beige Book.",
        "parameters": {
            "type": "object",
            "properties": {
                "document_type": {
                    "type": "string",
                    "description": "Filter by type: 'FOMC Statement', 'FOMC Minutes', 'Beige Book', or None for all",
                },
                "start_date": {
                    "type": "string",
                    "description": "Start date. Default: 2010-01-01",
                },
            },
            "required": [],
        },
    },
    {
        "name": "get_economic_calendar",
        "description": "Get economic calendar events with actual vs forecast data and surprises.",
        "parameters": {
            "type": "object",
            "properties": {
                "event_type": {
                    "type": "string",
                    "description": "Filter by event type (e.g., 'CPI', 'NFP', 'GDP')",
                },
                "start_date": {
                    "type": "string",
                    "description": "Start date. Default: 2020-01-01",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum events to return. Default: 50",
                },
            },
            "required": [],
        },
    },
    {
        "name": "run_custom_query",
        "description": "Run a custom SQL SELECT query on the database. Tables: observations, market_prices, documents, document_features, economic_calendar, series_metadata.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "SQL SELECT query. Example: SELECT series_id, AVG(value) FROM observations GROUP BY series_id",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum rows. Default: 100",
                },
            },
            "required": ["query"],
        },
    },
    # === CODE MODIFICATION TOOLS ===
    {
        "name": "list_model_scripts",
        "description": "List all model scripts in model_heuristics/scripts/ that can be read and modified.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "read_file",
        "description": "Read a Python file's contents. Returns numbered lines.",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path relative to project root (e.g., 'model_heuristics/scripts/value_heuristic_model.py')",
                },
                "start_line": {
                    "type": "integer",
                    "description": "Starting line number (1-indexed). Default: 1",
                },
                "end_line": {
                    "type": "integer",
                    "description": "Ending line number. Default: end of file",
                },
            },
            "required": ["file_path"],
        },
    },
    {
        "name": "write_file",
        "description": "Overwrite a file with new content. Creates backup automatically. Only works on allowed model scripts.",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path relative to project root",
                },
                "content": {
                    "type": "string",
                    "description": "New file content",
                },
            },
            "required": ["file_path", "content"],
        },
    },
    {
        "name": "search_replace_in_file",
        "description": "Search and replace text in a file. Creates backup. Only works on allowed model scripts.",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path relative to project root",
                },
                "old_text": {
                    "type": "string",
                    "description": "Text to find and replace",
                },
                "new_text": {
                    "type": "string",
                    "description": "Replacement text",
                },
            },
            "required": ["file_path", "old_text", "new_text"],
        },
    },
    {
        "name": "insert_code_at_line",
        "description": "Insert code at a specific line number. Creates backup.",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path relative to project root",
                },
                "line_number": {
                    "type": "integer",
                    "description": "Line number to insert at (1-indexed)",
                },
                "code": {
                    "type": "string",
                    "description": "Code to insert",
                },
            },
            "required": ["file_path", "line_number", "code"],
        },
    },
    {
        "name": "run_script",
        "description": "Run a Python script and capture output. Use to test changes.",
        "parameters": {
            "type": "object",
            "properties": {
                "script_path": {
                    "type": "string",
                    "description": "Path relative to project root",
                },
                "args": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Command line arguments",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in seconds. Default: 120",
                },
            },
            "required": ["script_path"],
        },
    },
    {
        "name": "lint_file",
        "description": "Check a Python file for syntax errors.",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path relative to project root",
                },
            },
            "required": ["file_path"],
        },
    },
    # === FEATURE ENGINEERING TOOLS ===
    {
        "name": "create_derived_feature",
        "description": "Create a new derived feature from an existing one. Transformations: ma_ratio, roc, zscore, volatility, momentum, diff, log_diff, ema, rsi, rank.",
        "parameters": {
            "type": "object",
            "properties": {
                "base_feature": {
                    "type": "string",
                    "description": "Name of existing feature (e.g., 'VIX', 'DGS10', 'SPY')",
                },
                "transformation": {
                    "type": "string",
                    "description": "Type: ma_ratio, roc, zscore, volatility, momentum, diff, log_diff, ema, rsi, rank",
                },
                "window": {
                    "type": "integer",
                    "description": "Window size for rolling calcs. Default: 21",
                },
                "feature_name": {
                    "type": "string",
                    "description": "Custom name for the feature (optional)",
                },
            },
            "required": ["base_feature", "transformation"],
        },
    },
    {
        "name": "combine_features",
        "description": "Combine multiple features into one composite feature using weighted_sum, product, mean, or zscore_sum.",
        "parameters": {
            "type": "object",
            "properties": {
                "features": {
                    "type": "object",
                    "description": "Dict of feature_name -> weight, e.g., {'VIX': 0.5, 'DGS10': -0.3}",
                },
                "operation": {
                    "type": "string",
                    "description": "How to combine: weighted_sum, product, mean, zscore_sum. Default: weighted_sum",
                },
            },
            "required": ["features"],
        },
    },
    {
        "name": "test_feature_predictive_power",
        "description": "Test a new feature's predictive power. Returns IC correlations at multiple horizons.",
        "parameters": {
            "type": "object",
            "properties": {
                "feature_spec": {
                    "type": "object",
                    "description": "Either {'base': 'VIX', 'transform': 'zscore', 'window': 21} OR {'features': {'VIX': 0.5, 'DGS10': -0.3}}",
                },
                "target_factor": {
                    "type": "string",
                    "description": "Factor to predict (VLUE, SIZE, QUAL, etc.). Default: VLUE",
                },
                "forward_days": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Forward return periods to test. Default: [5, 10, 21, 63]",
                },
            },
            "required": ["feature_spec"],
        },
    },
    {
        "name": "list_available_base_features",
        "description": "List all raw features (market symbols, economic series) available for creating derived features.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    # === SMART OPTIMIZATION STRATEGY TOOLS ===
    {
        "name": "initialize_smart_optimization",
        "description": "Initialize smart optimization with Bayesian and Genetic strategies. Call at start of optimization.",
        "parameters": {
            "type": "object",
            "properties": {
                "factor": {
                    "type": "string",
                    "description": "Factor to optimize (VLUE, SIZE, etc.). Default: VLUE",
                },
            },
            "required": [],
        },
    },
    {
        "name": "suggest_next_experiment",
        "description": "Get AI-guided suggestions for next experiment using Bayesian optimization. Balances exploration/exploitation.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "record_experiment_result",
        "description": "Record experiment result for learning. Call after each backtest to update the optimizer.",
        "parameters": {
            "type": "object",
            "properties": {
                "weights": {
                    "type": "object",
                    "description": "Feature weights dict",
                },
                "threshold": {
                    "type": "number",
                    "description": "Threshold used",
                },
                "hold_period": {
                    "type": "integer",
                    "description": "Hold period used",
                },
                "sharpe": {
                    "type": "number",
                    "description": "Resulting Sharpe ratio",
                },
            },
            "required": ["weights", "sharpe"],
        },
    },
    {
        "name": "get_genetic_population",
        "description": "Get current genetic algorithm population configs for testing.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "evolve_genetic_population",
        "description": "Evolve genetic population after testing. Pass Sharpe scores to breed next generation.",
        "parameters": {
            "type": "object",
            "properties": {
                "fitness_scores": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Sharpe ratios for each config in population order",
                },
            },
            "required": ["fitness_scores"],
        },
    },
    {
        "name": "build_ensemble",
        "description": "Build ensemble from top configs. Methods: weighted_average, simple_average, best_features.",
        "parameters": {
            "type": "object",
            "properties": {
                "top_n": {
                    "type": "integer",
                    "description": "Number of top configs to include. Default: 5",
                },
                "method": {
                    "type": "string",
                    "description": "Combination method. Default: weighted_average",
                },
            },
            "required": [],
        },
    },
    {
        "name": "test_ensemble",
        "description": "Build and backtest an ensemble of top configs.",
        "parameters": {
            "type": "object",
            "properties": {
                "factor": {
                    "type": "string",
                    "description": "Factor to test. Default: VLUE",
                },
                "top_n": {
                    "type": "integer",
                    "description": "Number of configs. Default: 5",
                },
                "method": {
                    "type": "string",
                    "description": "Ensemble method. Default: weighted_average",
                },
            },
            "required": [],
        },
    },
    {
        "name": "get_reflection_prompt",
        "description": "Get self-reflection prompt to analyze what's working. Includes feature importance analysis.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "validate_out_of_sample",
        "description": "Validate config on out-of-sample data (train/val/test split). Detects overfitting.",
        "parameters": {
            "type": "object",
            "properties": {
                "factor": {
                    "type": "string",
                    "description": "Factor to test. Default: VLUE",
                },
            },
            "required": [],
        },
    },
    {
        "name": "get_optimization_summary",
        "description": "Get full summary: n experiments, best config, top 5 configs, Sharpe progression.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
]


def _convert_gemini_types(obj):
    """Convert Gemini protobuf types to Python native types."""
    if obj is None:
        return None
    
    # Handle MapComposite and other protobuf-like objects
    if hasattr(obj, 'items'):
        try:
            return {str(k): _convert_gemini_types(v) for k, v in obj.items()}
        except Exception:
            pass
    
    # Handle RepeatedComposite (list-like)
    if hasattr(obj, '__iter__') and not isinstance(obj, (str, dict)):
        try:
            return [_convert_gemini_types(item) for item in obj]
        except Exception:
            pass
    
    # Convert floats that should be ints
    if isinstance(obj, float) and obj.is_integer():
        return int(obj)
    
    return obj


def execute_tool(tool_name: str, tool_input: Dict) -> str:
    """Execute a tool and return the result as a string."""
    # Convert Gemini types to native Python types
    tool_input = _convert_gemini_types(tool_input)
    
    try:
        if tool_name == "get_current_performance":
            result = get_current_performance(
                factor=tool_input.get("factor", "VLUE"),
                start_date=tool_input.get("start_date", "2016-01-01"),
            )
        
        elif tool_name == "analyze_feature_correlations":
            result = analyze_feature_correlations(
                factor=tool_input.get("factor", "VLUE"),
                horizon=tool_input.get("horizon", 21),
                top_n=tool_input.get("top_n", 20),
            )
            # Convert DataFrame to dict for JSON serialization
            if hasattr(result, "to_dict"):
                result = result.to_dict("records")
        
        elif tool_name == "get_available_features":
            result = get_available_features()
        
        elif tool_name == "run_backtest":
            result = run_backtest(
                factor=tool_input.get("factor", "VLUE"),
                signal_weights=tool_input.get("signal_weights"),
                threshold=tool_input.get("threshold", 0.0),
                hold_period=tool_input.get("hold_period", 0),
            )
        
        elif tool_name == "run_parameter_sweep":
            result = run_parameter_sweep(
                factor=tool_input.get("factor", "VLUE"),
                thresholds=tool_input.get("thresholds", [0.0, 0.1, 0.2, 0.3, 0.5]),
                hold_periods=tool_input.get("hold_periods", [0, 5, 10, 21]),
            )
            if hasattr(result, "to_dict"):
                result = result.to_dict("records")
        
        elif tool_name == "compare_strategies":
            result = compare_strategies(
                factor=tool_input.get("factor", "VLUE"),
                strategies=tool_input.get("strategies", {}),
            )
        
        elif tool_name == "generate_new_signal":
            result = generate_new_signal(
                factor=tool_input.get("factor", "VLUE"),
                top_n_features=tool_input.get("top_n_features", 5),
                use_ic_weights=tool_input.get("use_ic_weights", True),
            )
        
        elif tool_name == "test_signal_combination":
            result = test_signal_combination(
                factor=tool_input.get("factor", "VLUE"),
                feature_weights=tool_input.get("feature_weights", {}),
                threshold=tool_input.get("threshold", 0.0),
                hold_period=tool_input.get("hold_period", 0),
            )
        
        elif tool_name == "suggest_improvements":
            result = suggest_improvements(
                factor=tool_input.get("factor", "VLUE"),
                current_sharpe=tool_input.get("current_sharpe", 0.0),
            )
        
        # === DATABASE ACCESS TOOLS ===
        elif tool_name == "list_database_tables":
            result = list_database_tables()
        
        elif tool_name == "list_economic_series":
            result = list_economic_series()
        
        elif tool_name == "list_market_symbols":
            result = list_market_symbols()
        
        elif tool_name == "get_economic_series":
            result = get_economic_series(
                series_ids=tool_input.get("series_ids", []),
                start_date=tool_input.get("start_date", "2010-01-01"),
            )
        
        elif tool_name == "get_market_prices":
            result = get_market_prices(
                symbols=tool_input.get("symbols", []),
                start_date=tool_input.get("start_date", "2010-01-01"),
            )
        
        elif tool_name == "get_fed_documents":
            result = get_fed_documents(
                document_type=tool_input.get("document_type"),
                start_date=tool_input.get("start_date", "2010-01-01"),
            )
        
        elif tool_name == "get_economic_calendar":
            result = get_economic_calendar(
                event_type=tool_input.get("event_type"),
                start_date=tool_input.get("start_date", "2020-01-01"),
                limit=tool_input.get("limit", 50),
            )
        
        elif tool_name == "run_custom_query":
            result = run_custom_query(
                query=tool_input.get("query", ""),
                limit=tool_input.get("limit", 100),
            )
        
        # === CODE MODIFICATION TOOLS ===
        elif tool_name == "list_model_scripts":
            result = list_model_scripts()
        
        elif tool_name == "read_file":
            result = read_file(
                file_path=tool_input.get("file_path", ""),
                start_line=tool_input.get("start_line", 1),
                end_line=tool_input.get("end_line"),
            )
        
        elif tool_name == "write_file":
            result = write_file(
                file_path=tool_input.get("file_path", ""),
                content=tool_input.get("content", ""),
            )
        
        elif tool_name == "search_replace_in_file":
            result = search_replace_in_file(
                file_path=tool_input.get("file_path", ""),
                old_text=tool_input.get("old_text", ""),
                new_text=tool_input.get("new_text", ""),
            )
        
        elif tool_name == "insert_code_at_line":
            result = insert_code_at_line(
                file_path=tool_input.get("file_path", ""),
                line_number=tool_input.get("line_number", 1),
                code=tool_input.get("code", ""),
            )
        
        elif tool_name == "run_script":
            result = run_script(
                script_path=tool_input.get("script_path", ""),
                args=tool_input.get("args"),
                timeout=tool_input.get("timeout", 120),
            )
        
        elif tool_name == "lint_file":
            result = lint_file(
                file_path=tool_input.get("file_path", ""),
            )
        
        # === FEATURE ENGINEERING TOOLS ===
        elif tool_name == "create_derived_feature":
            result = create_derived_feature(
                base_feature=tool_input.get("base_feature", ""),
                transformation=tool_input.get("transformation", "zscore"),
                window=tool_input.get("window", 21),
                feature_name=tool_input.get("feature_name"),
            )
        
        elif tool_name == "combine_features":
            result = combine_features(
                features=tool_input.get("features", {}),
                operation=tool_input.get("operation", "weighted_sum"),
            )
        
        elif tool_name == "test_feature_predictive_power":
            result = test_feature_predictive_power(
                feature_spec=tool_input.get("feature_spec", {}),
                target_factor=tool_input.get("target_factor", "VLUE"),
                forward_days=tool_input.get("forward_days", [5, 10, 21, 63]),
            )
        
        elif tool_name == "list_available_base_features":
            result = list_available_base_features()
        
        # === SMART OPTIMIZATION STRATEGY TOOLS ===
        elif tool_name == "initialize_smart_optimization":
            result = initialize_smart_optimization(
                factor=tool_input.get("factor", "VLUE"),
            )
        
        elif tool_name == "suggest_next_experiment":
            result = suggest_next_experiment()
        
        elif tool_name == "record_experiment_result":
            result = record_experiment_result(
                weights=tool_input.get("weights", {}),
                threshold=tool_input.get("threshold", 0.0),
                hold_period=tool_input.get("hold_period", 0),
                sharpe=tool_input.get("sharpe", 0.0),
            )
        
        elif tool_name == "get_genetic_population":
            result = get_genetic_population()
        
        elif tool_name == "evolve_genetic_population":
            result = evolve_genetic_population(
                fitness_scores=tool_input.get("fitness_scores", []),
            )
        
        elif tool_name == "build_ensemble":
            result = build_ensemble(
                top_n=tool_input.get("top_n", 5),
                method=tool_input.get("method", "weighted_average"),
            )
        
        elif tool_name == "test_ensemble":
            result = test_ensemble(
                factor=tool_input.get("factor", "VLUE"),
                top_n=tool_input.get("top_n", 5),
                method=tool_input.get("method", "weighted_average"),
            )
        
        elif tool_name == "get_reflection_prompt":
            result = get_reflection_prompt()
        
        elif tool_name == "validate_out_of_sample":
            result = validate_out_of_sample(
                factor=tool_input.get("factor", "VLUE"),
            )
        
        elif tool_name == "get_optimization_summary":
            result = get_optimization_summary()
        
        else:
            result = {"error": f"Unknown tool: {tool_name}"}
        
        return json.dumps(result, indent=2, default=str)
    
    except Exception as e:
        return json.dumps({"error": str(e)})


class HeuristicOptimizationAgent:
    """
    AI agent for optimizing factor timing heuristics.
    
    Uses Gemini with function calling to:
    1. Analyze current model performance
    2. Identify top predictive features
    3. Suggest and test improvements
    4. Generate optimized signal weights
    """
    
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
        verbose: bool = True,
    ):
        """
        Initialize the agent.
        
        Args:
            model: Gemini model to use
            api_key: Google API key (defaults to GOOGLE_API_KEY env var)
            verbose: Print detailed progress
        """
        self.model_name = model
        self.verbose = verbose
        
        # Configure Gemini
        api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")
        
        genai.configure(api_key=api_key)
        
        # Create the model with tools
        self.model = genai.GenerativeModel(
            model_name=model,
            system_instruction=SYSTEM_PROMPT,
            tools=[{"function_declarations": TOOL_FUNCTIONS}],
        )
        
        # Start a chat session
        self.chat_session = self.model.start_chat(history=[])
        
        AGENT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    def _log(self, message: str):
        """Log message if verbose."""
        if self.verbose:
            print(message)
    
    def chat(self, user_message: str, max_tool_iterations: int = 20) -> str:
        """
        Send a message to the agent and get a response.
        
        Handles function calling automatically.
        """
        self._log(f"\n{'='*60}")
        self._log(f"User: {user_message}")
        self._log(f"{'='*60}")
        
        try:
            # Send message
            response = self.chat_session.send_message(user_message)
        except Exception as e:
            self._log(f"Error sending message: {e}")
            return f"Error: {e}"
        
        tool_iteration = 0
        
        # Loop for function calls
        while response.candidates[0].content.parts and tool_iteration < max_tool_iterations:
            # Check if there are function calls
            function_calls = [
                part.function_call 
                for part in response.candidates[0].content.parts 
                if hasattr(part, 'function_call') and part.function_call.name
            ]
            
            if not function_calls:
                # No function calls, extract text response
                break
            
            # Process function calls
            function_responses = []
            for fc in function_calls:
                tool_name = fc.name
                tool_input = _convert_gemini_types(dict(fc.args))
                
                self._log(f"\n🔧 Tool: {tool_name}")
                self._log(f"   Input: {json.dumps(tool_input, indent=2, default=str)}")
                
                # Execute tool
                result = execute_tool(tool_name, tool_input)
                
                self._log(f"   Result: {result[:200]}..." if len(result) > 200 else f"   Result: {result}")
                
                function_responses.append(
                    genai.protos.Part(
                        function_response=genai.protos.FunctionResponse(
                            name=tool_name,
                            response={"result": result}
                        )
                    )
                )
            
            # Send function responses back
            try:
                response = self.chat_session.send_message(function_responses)
                tool_iteration += 1
            except Exception as e:
                self._log(f"\n⚠️ API error: {e}")
                # Try to continue without the failed response
                break
        
        # Extract final text response
        text_response = ""
        for part in response.candidates[0].content.parts:
            if hasattr(part, 'text'):
                text_response += part.text
        
        self._log(f"\n{'='*60}")
        self._log(f"Assistant: {text_response}")
        self._log(f"{'='*60}")
        
        return text_response
    
    def analyze_factor(self, factor: str) -> str:
        """
        Run a comprehensive analysis of a factor's heuristic model.
        """
        prompt = f"""Please analyze the {factor} factor timing strategy:

1. First, get the current performance of the {factor} heuristic model
2. Analyze which features have the highest predictive power (IC) for {factor}
3. Compare the current signal weights with the empirically optimal weights
4. Suggest specific improvements to increase Sharpe ratio

Focus on actionable, specific recommendations.
"""
        return self.chat(prompt)
    
    def optimize_factor(self, factor: str) -> str:
        """
        Optimize a factor's signal weights.
        """
        prompt = f"""I want to optimize the {factor} factor timing strategy.

Please:
1. Get current performance as a baseline
2. Generate a new signal based on top IC features
3. Test different threshold and holding period combinations
4. Compare the new strategy against the current one
5. Provide the optimal weights I should use

Be specific about which features to include and their weights.
"""
        return self.chat(prompt)
    
    def compare_all_factors(self) -> str:
        """
        Compare performance across all factors.
        """
        prompt = """Please analyze all factor timing strategies (VLUE, SIZE, QUAL, USMV).

For each factor:
1. Get current performance
2. Identify which is performing best/worst

Then provide:
- A ranking of factors by strategy Sharpe ratio
- Which factor has the most room for improvement
- Key differences in what drives each factor
"""
        return self.chat(prompt)
    
    def save_session(self, filename: Optional[str] = None) -> Path:
        """Save the conversation history to a file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"agent_session_{timestamp}.json"
        
        filepath = AGENT_OUTPUT_DIR / filename
        
        # Serialize conversation history
        history = []
        for message in self.chat_session.history:
            parts_data = []
            for part in message.parts:
                if hasattr(part, 'text'):
                    parts_data.append({"type": "text", "content": part.text})
                elif hasattr(part, 'function_call') and part.function_call.name:
                    parts_data.append({
                        "type": "function_call",
                        "name": part.function_call.name,
                        "args": dict(part.function_call.args)
                    })
                elif hasattr(part, 'function_response'):
                    parts_data.append({
                        "type": "function_response",
                        "name": part.function_response.name,
                        "response": str(part.function_response.response)
                    })
            history.append({
                "role": message.role,
                "parts": parts_data
            })
        
        with open(filepath, "w") as f:
            json.dump({
                "model": self.model_name,
                "timestamp": datetime.now().isoformat(),
                "history": history,
            }, f, indent=2, default=str)
        
        self._log(f"\nSession saved to: {filepath}")
        return filepath
    
    def reset(self):
        """Reset conversation history."""
        self.chat_session = self.model.start_chat(history=[])
        self._log("Conversation history reset.")
    
    def iterative_optimize(
        self,
        factor: str,
        min_iterations: int = 3,
        max_iterations: int = 10,
        min_improvement: float = 0.05,
        save_best: bool = True,
        apply_to_code: bool = True,
    ) -> Dict[str, Any]:
        """
        Iteratively optimize a factor until improvement plateaus.
        
        The agent will keep trying to improve the Sharpe ratio until:
        1. At least min_iterations have been completed
        2. The improvement in the last iteration is below min_improvement
        3. Or max_iterations is reached
        
        Args:
            factor: Factor ETF symbol (VLUE, SIZE, QUAL, USMV, MTUM)
            min_iterations: Minimum number of optimization iterations
            max_iterations: Maximum iterations (safety limit)
            min_improvement: Stop if Sharpe improvement < this after min_iterations
            save_best: Save the best configuration found
            apply_to_code: Update the actual model scripts with best config
            
        Returns:
            Dict with optimization history and best configuration
        """
        self._log(f"\n{'='*70}")
        self._log(f"🚀 ITERATIVE OPTIMIZATION: {factor}")
        self._log(f"{'='*70}")
        self._log(f"Settings: min_iter={min_iterations}, max_iter={max_iterations}, min_improvement={min_improvement}")
        self._log(f"Will modify code: {apply_to_code}")
        
        # Track optimization history with FULL configs
        history = []
        tested_configs = []  # Keep all tested configurations
        best_sharpe = float('-inf')
        best_config = None
        best_iteration = 0
        
        # Reset for fresh start but DON'T reset between iterations
        self.reset()
        
        # Get baseline performance
        self._log(f"\n📊 Getting baseline performance...")
        baseline = get_current_performance(factor)
        baseline_sharpe = baseline["strategy"]["sharpe"]
        
        history.append({
            "iteration": 0,
            "type": "baseline",
            "sharpe": baseline_sharpe,
            "improvement": 0.0,
            "config": None,
        })
        
        self._log(f"   Baseline Sharpe: {baseline_sharpe:.3f}")
        
        current_sharpe = baseline_sharpe
        
        for iteration in range(1, max_iterations + 1):
            self._log(f"\n{'='*70}")
            self._log(f"📈 ITERATION {iteration}/{max_iterations}")
            self._log(f"{'='*70}")
            self._log(f"Current best Sharpe: {max(current_sharpe, best_sharpe):.3f}")
            
            # Build prompt with FULL history of tested configs
            tested_configs_summary = ""
            if tested_configs:
                tested_configs_summary = "\n\nALREADY TESTED CONFIGURATIONS (do not repeat these):\n"
                for i, cfg in enumerate(tested_configs[-10:], 1):  # Last 10 configs
                    weights_str = ", ".join([f"{k}:{v:.2f}" for k, v in (cfg.get("weights") or {}).items()])
                    tested_configs_summary += f"  {i}. [{weights_str}] th={cfg.get('threshold', 0)}, hold={cfg.get('hold_period', 0)} → Sharpe={cfg.get('sharpe', 0):.3f}\n"
            
            if iteration == 1:
                prompt = f"""You are optimizing the {factor} factor timing strategy.

Current baseline Sharpe: {baseline_sharpe:.3f}

You have FULL CONTROL over the codebase. You can:
- Read and analyze the model scripts with read_file
- Modify the signal weights in the code with search_replace_in_file
- Run scripts to test changes with run_script
- Query any data with run_custom_query

This is iteration 1. Please:
1. First, read the current model: read_file("model_heuristics/scripts/value_heuristic_model.py", 200, 320)
2. Analyze feature correlations to find the best predictors
3. Generate a new signal using the top features weighted by IC
4. Test the new signal with run_backtest
5. Report the new Sharpe ratio achieved

Focus on finding features with strong, statistically significant ICs.
At the end, provide the exact weights you used in JSON format like:
{{"weights": {{"feature1": 0.3, "feature2": -0.2, ...}}, "sharpe": X.XX}}
"""
            else:
                # Full history of what was tried
                history_summary = "\n".join([
                    f"  Iter {h['iteration']}: Sharpe={h['sharpe']:.3f} (Δ={h['improvement']:+.3f})"
                    for h in history
                ])
                
                prompt = f"""You are optimizing the {factor} factor timing strategy.

FULL OPTIMIZATION HISTORY:
{history_summary}
{tested_configs_summary}

Current best Sharpe: {best_sharpe:.3f} (from iteration {best_iteration})
Target: Beat {best_sharpe:.3f}

This is iteration {iteration}. You have full control:
- Read model code: read_file("model_heuristics/scripts/value_heuristic_model.py")
- Modify code: search_replace_in_file or write_file  
- Run tests: run_script or run_backtest
- Query data: run_custom_query

DO NOT repeat configurations you've already tested. Try something NEW:
- Different feature combinations
- Different thresholds (0, 0.1, 0.2, 0.3) or holding periods (0, 5, 10, 21)
- Add/remove features
- Try inverse signals
- Explore the database for new features

Test your new configuration and report:
{{"weights": {{"feature1": 0.3, ...}}, "threshold": X, "hold_period": X, "sharpe": X.XX}}
"""
            
            # Run the optimization iteration
            response = self.chat(prompt)
            
            # Try to extract Sharpe from the response
            new_sharpe = self._extract_sharpe_from_response(response)
            new_config = self._extract_config_from_response(response)
            
            if new_sharpe is None:
                # Fallback: run a backtest to get actual Sharpe
                self._log("   Could not extract Sharpe, running verification backtest...")
                if new_config and "weights" in new_config:
                    result = run_backtest(
                        factor=factor,
                        signal_weights=new_config.get("weights"),
                        threshold=new_config.get("threshold", 0.0),
                        hold_period=new_config.get("hold_period", 0),
                    )
                    new_sharpe = result["strategy"]["sharpe"]
                else:
                    new_sharpe = current_sharpe  # No improvement
            
            # Store config with its sharpe for future reference
            if new_config:
                new_config["sharpe"] = new_sharpe
                tested_configs.append(new_config)
            
            improvement = new_sharpe - current_sharpe
            
            history.append({
                "iteration": iteration,
                "type": "optimization",
                "sharpe": new_sharpe,
                "improvement": improvement,
                "config": new_config,
            })
            
            self._log(f"\n   Iteration {iteration} Sharpe: {new_sharpe:.3f} (Δ={improvement:+.3f})")
            
            # Track best
            if new_sharpe > best_sharpe:
                best_sharpe = new_sharpe
                best_config = new_config
                best_iteration = iteration
                self._log(f"   🏆 New best! Sharpe: {best_sharpe:.3f}")
            
            current_sharpe = new_sharpe
            
            # Check stopping conditions
            if iteration >= min_iterations:
                # Calculate recent improvement (average of last 2 iterations)
                recent_improvements = [h["improvement"] for h in history[-2:]]
                avg_recent_improvement = sum(recent_improvements) / len(recent_improvements)
                
                if abs(avg_recent_improvement) < min_improvement:
                    self._log(f"\n✅ Stopping: Average recent improvement ({avg_recent_improvement:.3f}) < threshold ({min_improvement})")
                    break
        
        # Final summary
        self._log(f"\n{'='*70}")
        self._log(f"🎯 OPTIMIZATION COMPLETE")
        self._log(f"{'='*70}")
        self._log(f"Total iterations: {len(history) - 1}")
        self._log(f"Baseline Sharpe: {baseline_sharpe:.3f}")
        self._log(f"Best Sharpe: {best_sharpe:.3f} (iteration {best_iteration})")
        self._log(f"Total improvement: {best_sharpe - baseline_sharpe:+.3f}")
        self._log(f"Configurations tested: {len(tested_configs)}")
        
        if best_config:
            self._log(f"\nBest configuration:")
            self._log(json.dumps(best_config, indent=2))
        
        # Apply best config to code if requested
        if apply_to_code and best_config and "weights" in best_config:
            self._log(f"\n📝 Applying best configuration to code...")
            apply_prompt = f"""The optimization is complete. The best configuration found was:

{json.dumps(best_config, indent=2)}

Please apply this to the actual model script:
1. Read the current model: read_file("model_heuristics/scripts/value_heuristic_model.py")
2. Find the compute_heuristic_signal method (around line 216-310)
3. Update the signal weights to match the best configuration
4. Use search_replace_in_file to make the changes
5. Lint the file to check for errors
6. Run the script with --backtest to verify it works

Make sure to preserve the code structure and just update the weights.
"""
            self.chat(apply_prompt)
        
        # Save results
        if save_best:
            result_path = AGENT_OUTPUT_DIR / f"optimization_{factor}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(result_path, "w") as f:
                json.dump({
                    "factor": factor,
                    "baseline_sharpe": baseline_sharpe,
                    "best_sharpe": best_sharpe,
                    "best_iteration": best_iteration,
                    "best_config": best_config,
                    "all_tested_configs": tested_configs,
                    "history": history,
                    "timestamp": datetime.now().isoformat(),
                }, f, indent=2, default=str)
            self._log(f"\nResults saved to: {result_path}")
        
        return {
            "factor": factor,
            "baseline_sharpe": baseline_sharpe,
            "best_sharpe": best_sharpe,
            "improvement": best_sharpe - baseline_sharpe,
            "best_config": best_config,
            "best_iteration": best_iteration,
            "total_iterations": len(history) - 1,
            "history": history,
        }
    
    def _extract_sharpe_from_response(self, response: str) -> Optional[float]:
        """Try to extract Sharpe ratio from agent response."""
        import re
        
        # Look for JSON with sharpe
        json_match = re.search(r'\{[^{}]*"sharpe":\s*([-\d.]+)[^{}]*\}', response)
        if json_match:
            try:
                return float(json_match.group(1))
            except ValueError:
                pass
        
        # Look for "Sharpe: X.XX" or "Sharpe ratio: X.XX"
        sharpe_match = re.search(r'[Ss]harpe(?:\s+[Rr]atio)?[:\s]+([+-]?\d+\.?\d*)', response)
        if sharpe_match:
            try:
                return float(sharpe_match.group(1))
            except ValueError:
                pass
        
        return None
    
    def _extract_config_from_response(self, response: str) -> Optional[Dict]:
        """Try to extract configuration from agent response."""
        import re
        
        # Look for JSON block with weights
        json_patterns = [
            r'\{[^{}]*"weights":\s*\{[^{}]+\}[^{}]*\}',
            r'```json\s*(\{[^`]+\})\s*```',
            r'```\s*(\{[^`]+\})\s*```',
        ]
        
        for pattern in json_patterns:
            match = re.search(pattern, response, re.DOTALL)
            if match:
                try:
                    json_str = match.group(1) if match.lastindex else match.group(0)
                    return json.loads(json_str)
                except (json.JSONDecodeError, IndexError):
                    continue
        
        return None
    
    def smart_optimize(
        self,
        factor: str,
        max_iterations: int = 50,
        use_bayesian: bool = True,
        use_genetic: bool = True,
        use_reflection: bool = True,
        validate_oos: bool = True,
        build_ensemble: bool = True,
        apply_to_code: bool = True,
    ) -> Dict[str, Any]:
        """
        Smart optimization using advanced strategies:
        
        1. Bayesian Optimization - Model Sharpe surface, pick optimal experiments
        2. Genetic Algorithms - Evolve configs through mutation/crossover
        3. Structured Phases - Feature discovery → Weight optimization → Parameter tuning
        4. Self-Reflection - Analyze why configs work/fail
        5. Ensemble Building - Combine top configs for robustness
        6. Out-of-Sample Validation - Detect overfitting
        
        Args:
            factor: Factor to optimize (VLUE, SIZE, QUAL, etc.)
            max_iterations: Maximum optimization iterations
            use_bayesian: Use Bayesian optimization for suggestions
            use_genetic: Use genetic algorithm for evolution
            use_reflection: Include self-reflection prompts
            validate_oos: Validate on out-of-sample data
            build_ensemble: Build ensemble of top configs at end
            apply_to_code: Update model scripts with best config
            
        Returns:
            Dict with optimization results, best config, ensemble, OOS validation
        """
        self._log(f"\n{'='*70}")
        self._log(f"🧠 SMART OPTIMIZATION: {factor}")
        self._log(f"{'='*70}")
        self._log(f"Strategies: Bayesian={use_bayesian}, Genetic={use_genetic}, Reflection={use_reflection}")
        self._log(f"Max iterations: {max_iterations}")
        
        # Reset for fresh start
        self.reset()
        
        # Phase definitions
        phases = [
            ("feature_discovery", 0.25),     # First 25%
            ("weight_optimization", 0.50),   # Next 25%
            ("parameter_tuning", 0.75),      # Next 25%
            ("refinement", 1.0),             # Final 25%
        ]
        
        def get_current_phase(iteration: int) -> str:
            progress = iteration / max_iterations
            for phase_name, threshold in phases:
                if progress <= threshold:
                    return phase_name
            return "refinement"
        
        # Get baseline
        self._log(f"\n📊 Getting baseline performance...")
        baseline = get_current_performance(factor)
        baseline_sharpe = baseline["strategy"]["sharpe"]
        self._log(f"   Baseline Sharpe: {baseline_sharpe:.3f}")
        
        # Initialize smart optimization
        init_prompt = f"""You are starting a SMART optimization of the {factor} factor timing strategy.

Baseline Sharpe: {baseline_sharpe:.3f}

You have access to advanced optimization tools:
1. initialize_smart_optimization - Start Bayesian/Genetic optimization
2. suggest_next_experiment - Get AI-guided experiment suggestions
3. record_experiment_result - Record results for learning
4. get_reflection_prompt - Analyze what's working and why
5. test_ensemble - Build and test ensemble of top configs
6. validate_out_of_sample - Check for overfitting

The optimization has 4 phases:
- Phase 1 (Feature Discovery): Find best predictive features using analyze_feature_correlations
- Phase 2 (Weight Optimization): Optimize weights for top features
- Phase 3 (Parameter Tuning): Find optimal threshold and holding period
- Phase 4 (Refinement): Fine-tune and validate

START by calling:
1. initialize_smart_optimization(factor="{factor}")
2. analyze_feature_correlations(factor="{factor}", top_n=10)
3. Then test the top features with run_backtest
4. Record results with record_experiment_result

Report each result as: {{"weights": {{...}}, "threshold": X, "hold_period": X, "sharpe": X.XX}}
"""
        
        # Run initial setup
        self.chat(init_prompt)
        
        # Track results
        history = []
        best_sharpe = baseline_sharpe
        best_config = None
        best_iteration = 0
        
        history.append({
            "iteration": 0,
            "phase": "baseline",
            "sharpe": baseline_sharpe,
            "config": None,
        })
        
        current_sharpe = baseline_sharpe
        
        for iteration in range(1, max_iterations + 1):
            current_phase = get_current_phase(iteration)
            
            self._log(f"\n{'='*70}")
            self._log(f"📈 ITERATION {iteration}/{max_iterations} | Phase: {current_phase.upper()}")
            self._log(f"{'='*70}")
            self._log(f"Best Sharpe: {best_sharpe:.3f}")
            
            # Build phase-specific prompt
            if current_phase == "feature_discovery":
                phase_prompt = f"""
## PHASE: FEATURE DISCOVERY (Iteration {iteration})

Best Sharpe so far: {best_sharpe:.3f}

GOAL: Find the best predictive features for {factor}.
- Use analyze_feature_correlations to identify high-IC features
- Focus on features with |IC| > 0.05 and p-value < 0.1
- Test different feature combinations
- Use suggest_next_experiment for AI-guided suggestions

After testing, record results with record_experiment_result.
Report: {{"weights": {{...}}, "threshold": 0, "hold_period": 0, "sharpe": X.XX}}
"""
            elif current_phase == "weight_optimization":
                phase_prompt = f"""
## PHASE: WEIGHT OPTIMIZATION (Iteration {iteration})

Best Sharpe: {best_sharpe:.3f}

GOAL: Optimize weights for the top features.
- Use suggest_next_experiment for suggestions based on history
- Try different weight values (-0.3 to +0.3)
- Test flipping signs on features
- Focus on the features that worked in Phase 1

Record results with record_experiment_result.
Report: {{"weights": {{...}}, "threshold": 0, "hold_period": 0, "sharpe": X.XX}}
"""
            elif current_phase == "parameter_tuning":
                phase_prompt = f"""
## PHASE: PARAMETER TUNING (Iteration {iteration})

Best Sharpe: {best_sharpe:.3f}
Best config: {json.dumps(best_config, indent=2) if best_config else 'None'}

GOAL: Find optimal threshold and holding period.
- Use run_parameter_sweep with thresholds [0, 0.02, 0.04, 0.05, 0.1]
- Test holding periods [0, 5, 10, 21, 42, 63]
- Keep the best weights from Phase 2

Report: {{"weights": {{...}}, "threshold": X, "hold_period": X, "sharpe": X.XX}}
"""
            else:  # refinement
                phase_prompt = f"""
## PHASE: REFINEMENT (Iteration {iteration})

Best Sharpe: {best_sharpe:.3f}

GOAL: Final refinement and validation.
- Make small adjustments to best config
- Use get_reflection_prompt to analyze what's working
- Test ensemble: test_ensemble(factor="{factor}", top_n=5)
- Validate: validate_out_of_sample(factor="{factor}")

Report final best config.
"""
            
            # Add reflection if enabled
            if use_reflection and iteration % 5 == 0 and iteration > 1:
                phase_prompt += """

## REFLECTION TIME
Call get_reflection_prompt to analyze:
- Which features consistently appear in top configs?
- What patterns do you see in successful vs unsuccessful configs?
- What NEW approach should we try?
"""
            
            # Run iteration
            response = self.chat(phase_prompt)
            
            # Extract results
            new_sharpe = self._extract_sharpe_from_response(response)
            new_config = self._extract_config_from_response(response)
            
            if new_sharpe is None and new_config and "weights" in new_config:
                # Verify with backtest
                result = run_backtest(
                    factor=factor,
                    signal_weights=new_config.get("weights"),
                    threshold=new_config.get("threshold", 0.0),
                    hold_period=new_config.get("hold_period", 0),
                )
                new_sharpe = result["strategy"]["sharpe"]
            elif new_sharpe is None:
                new_sharpe = current_sharpe
            
            improvement = new_sharpe - current_sharpe
            
            history.append({
                "iteration": iteration,
                "phase": current_phase,
                "sharpe": new_sharpe,
                "improvement": improvement,
                "config": new_config,
            })
            
            self._log(f"\n   Sharpe: {new_sharpe:.3f} (Δ={improvement:+.3f})")
            
            if new_sharpe > best_sharpe:
                best_sharpe = new_sharpe
                best_config = new_config
                best_iteration = iteration
                self._log(f"   🏆 New best! Sharpe: {best_sharpe:.3f}")
            
            current_sharpe = new_sharpe
            
            # Early stopping if no improvement for 10 iterations
            if iteration > 20:
                recent_best = max([h["sharpe"] for h in history[-10:]])
                if recent_best <= best_sharpe - 0.01:
                    self._log(f"\n⚠️ No improvement in 10 iterations, continuing to next phase...")
        
        # Final steps
        results = {
            "factor": factor,
            "baseline_sharpe": baseline_sharpe,
            "best_sharpe": best_sharpe,
            "best_iteration": best_iteration,
            "best_config": best_config,
            "improvement": best_sharpe - baseline_sharpe,
            "history": history,
        }
        
        # Build ensemble if requested
        if build_ensemble:
            self._log(f"\n📦 Building ensemble of top configs...")
            ensemble_prompt = f"""
Build and test an ensemble of the top 5 configurations:
1. Call test_ensemble(factor="{factor}", top_n=5, method="weighted_average")
2. Report the ensemble's Sharpe ratio
"""
            ensemble_response = self.chat(ensemble_prompt)
            results["ensemble_tested"] = True
        
        # Validate out-of-sample if requested
        if validate_oos:
            self._log(f"\n🔍 Validating on out-of-sample data...")
            oos_prompt = f"""
Validate the best configuration on out-of-sample data:
1. Call validate_out_of_sample(factor="{factor}")
2. Report train/validation/test Sharpes
3. Is the config robust or overfit?
"""
            oos_response = self.chat(oos_prompt)
            results["oos_validated"] = True
        
        # Apply to code if requested
        if apply_to_code and best_config:
            self._log(f"\n📝 Applying best config to code...")
            apply_prompt = f"""
Apply the best configuration to the model code:

Best config:
{json.dumps(best_config, indent=2)}

Steps:
1. Read the model: read_file("model_heuristics/scripts/value_heuristic_model.py", 230, 270)
2. Update the signal weights using search_replace_in_file
3. Lint the file to check for errors
"""
            self.chat(apply_prompt)
        
        # Save results
        result_path = AGENT_OUTPUT_DIR / f"smart_optimization_{factor}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(result_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        self._log(f"\nResults saved to: {result_path}")
        
        # Final summary
        self._log(f"\n{'='*70}")
        self._log(f"🎯 SMART OPTIMIZATION COMPLETE")
        self._log(f"{'='*70}")
        self._log(f"Baseline Sharpe: {baseline_sharpe:.3f}")
        self._log(f"Best Sharpe: {best_sharpe:.3f}")
        self._log(f"Improvement: {best_sharpe - baseline_sharpe:+.3f}")
        self._log(f"Best iteration: {best_iteration}")
        if best_config:
            self._log(f"\nBest config:")
            self._log(json.dumps(best_config, indent=2))
        
        return results
