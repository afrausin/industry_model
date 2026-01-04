"""
Agent Tools
===========

Tools for the heuristic optimization agent to interact with the codebase.
"""

from .analysis import (
    get_current_performance,
    analyze_feature_correlations,
    get_available_features,
    get_factor_data,
)

from .backtest import (
    run_backtest,
    run_parameter_sweep,
    compare_strategies,
)

from .optimization import (
    generate_new_signal,
    test_signal_combination,
    optimize_weights,
)

from .database import (
    list_database_tables,
    list_economic_series,
    list_market_symbols,
    get_economic_series,
    get_market_prices,
    get_fed_documents,
    get_economic_calendar,
    run_custom_query,
)

from .code import (
    list_model_scripts,
    read_file,
    write_file,
    search_replace_in_file,
    insert_code_at_line,
    run_script,
    update_signal_weights,
    lint_file,
)

from .features import (
    create_derived_feature,
    combine_features,
    test_feature_predictive_power,
    list_available_base_features,
)

from .strategy_tools import (
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
    diagnose_underperformance,
)

__all__ = [
    # Analysis
    "get_current_performance",
    "analyze_feature_correlations",
    "get_available_features",
    "get_factor_data",
    # Backtest
    "run_backtest",
    "run_parameter_sweep",
    "compare_strategies",
    # Optimization
    "generate_new_signal",
    "test_signal_combination",
    "optimize_weights",
    # Database
    "list_database_tables",
    "list_economic_series",
    "list_market_symbols",
    "get_economic_series",
    "get_market_prices",
    "get_fed_documents",
    "get_economic_calendar",
    "run_custom_query",
    # Code modification
    "list_model_scripts",
    "read_file",
    "write_file",
    "search_replace_in_file",
    "insert_code_at_line",
    "run_script",
    "update_signal_weights",
    "lint_file",
    # Feature engineering
    "create_derived_feature",
    "combine_features",
    "test_feature_predictive_power",
    "list_available_base_features",
    # Smart optimization strategies
    "initialize_smart_optimization",
    "suggest_next_experiment",
    "record_experiment_result",
    "get_genetic_population",
    "evolve_genetic_population",
    "build_ensemble",
    "test_ensemble",
    "get_reflection_prompt",
    "validate_out_of_sample",
    "get_optimization_summary",
    "diagnose_underperformance",
]

