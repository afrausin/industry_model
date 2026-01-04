"""
Visualization Module for Industry Model

Generate charts and reports for backtest analysis.
"""

from .charts import (
    plot_cumulative_returns,
    plot_drawdowns,
    plot_sector_weights,
    plot_signal_heatmap,
    plot_rolling_metrics,
    plot_sector_performance,
    create_rules_summary_chart,
)
from .report import generate_html_report, generate_dashboard, save_charts_to_files

__all__ = [
    # Charts
    "plot_cumulative_returns",
    "plot_drawdowns",
    "plot_sector_weights",
    "plot_signal_heatmap",
    "plot_rolling_metrics",
    "plot_sector_performance",
    "create_rules_summary_chart",
    # Reports
    "generate_html_report",
    "generate_dashboard",
    "save_charts_to_files",
]
