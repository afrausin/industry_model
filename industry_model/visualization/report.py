"""
Report Generation for Industry Model

Generate HTML reports and dashboards for backtest analysis.
"""

import base64
import io
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List

import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

from .charts import (
    plot_cumulative_returns,
    plot_drawdowns,
    plot_sector_weights,
    plot_signal_heatmap,
    plot_rolling_metrics,
    plot_sector_performance,
    create_rules_summary_chart,
)


def fig_to_base64(fig: plt.Figure) -> str:
    """Convert matplotlib figure to base64 encoded PNG."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_base64


def format_pct(value: float, decimals: int = 2) -> str:
    """Format value as percentage string."""
    return f"{value * 100:.{decimals}f}%"


def format_number(value: float, decimals: int = 2) -> str:
    """Format number with specified decimals."""
    return f"{value:.{decimals}f}"


def generate_metrics_table(metrics: Dict[str, float], title: str) -> str:
    """Generate HTML table for metrics."""
    rows = []
    for key, value in metrics.items():
        # Clean up key name
        display_key = key.replace('_', ' ').title()

        # Format value based on type
        if 'return' in key.lower() or 'vol' in key.lower() or 'drawdown' in key.lower():
            display_value = format_pct(value)
        elif 'ratio' in key.lower() or 'sharpe' in key.lower():
            display_value = format_number(value)
        elif 'rate' in key.lower():
            display_value = format_pct(value)
        elif 'periods' in key.lower() or 'n_' in key.lower():
            display_value = str(int(value))
        else:
            display_value = format_number(value, 4)

        rows.append(f"<tr><td>{display_key}</td><td>{display_value}</td></tr>")

    return f"""
    <div class="metrics-card">
        <h3>{title}</h3>
        <table class="metrics-table">
            <tbody>
                {''.join(rows)}
            </tbody>
        </table>
    </div>
    """


def generate_html_report(
    results: Dict[str, Any],
    signal_scores: Optional[pd.DataFrame] = None,
    output_path: Optional[str] = None,
    title: str = "Industry ETF Model Backtest Report",
) -> str:
    """
    Generate comprehensive HTML report for backtest results.

    Args:
        results: Backtest results dictionary from BacktestEngine.run_backtest()
        signal_scores: Optional DataFrame of signal scores by sector
        output_path: Path to save HTML file. If None, returns HTML string.
        title: Report title

    Returns:
        HTML string if output_path is None, otherwise path to saved file.
    """
    # Extract data from results
    strategy_returns = results['strategy_returns']
    benchmark_returns = results['benchmark_returns']
    weights_df = results['weights']
    strategy_metrics = results['strategy_metrics']
    benchmark_metrics = results['benchmark_metrics']
    relative_metrics = results.get('relative_metrics', {})
    turnover = results.get('turnover', pd.Series())

    # Generate charts
    charts = {}

    # 1. Cumulative returns
    fig = plot_cumulative_returns(strategy_returns, benchmark_returns)
    charts['cumulative'] = fig_to_base64(fig)

    # 2. Drawdowns
    fig = plot_drawdowns(strategy_returns, benchmark_returns)
    charts['drawdowns'] = fig_to_base64(fig)

    # 3. Sector weights
    fig = plot_sector_weights(weights_df)
    charts['weights'] = fig_to_base64(fig)

    # 4. Rolling metrics
    fig = plot_rolling_metrics(strategy_returns, benchmark_returns)
    charts['rolling'] = fig_to_base64(fig)

    # 5. Signal heatmap (if provided)
    if signal_scores is not None and not signal_scores.empty:
        fig = plot_signal_heatmap(signal_scores)
        charts['signals'] = fig_to_base64(fig)

    # 6. Sector performance (need to reconstruct returns from weights)
    # Skip if we don't have proper data

    # 7. Rules summary
    fig = create_rules_summary_chart()
    charts['rules'] = fig_to_base64(fig)

    # Calculate additional metrics
    avg_turnover = turnover.mean() if not turnover.empty else 0

    # Build HTML
    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        :root {{
            --primary: #1a365d;
            --secondary: #2c5282;
            --success: #276749;
            --danger: #c53030;
            --bg: #f7fafc;
            --card-bg: #ffffff;
            --text: #2d3748;
            --text-light: #718096;
            --border: #e2e8f0;
        }}

        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: var(--bg);
            color: var(--text);
            line-height: 1.6;
        }}

        .container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }}

        header {{
            background: var(--primary);
            color: white;
            padding: 30px;
            margin-bottom: 30px;
            border-radius: 8px;
        }}

        header h1 {{
            font-size: 2rem;
            margin-bottom: 10px;
        }}

        header .subtitle {{
            color: #a0aec0;
            font-size: 1rem;
        }}

        .section {{
            margin-bottom: 40px;
        }}

        .section-title {{
            font-size: 1.5rem;
            color: var(--primary);
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid var(--border);
        }}

        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}

        .metrics-card {{
            background: var(--card-bg);
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}

        .metrics-card h3 {{
            color: var(--secondary);
            margin-bottom: 15px;
            font-size: 1.1rem;
        }}

        .metrics-table {{
            width: 100%;
            border-collapse: collapse;
        }}

        .metrics-table td {{
            padding: 8px 0;
            border-bottom: 1px solid var(--border);
        }}

        .metrics-table td:last-child {{
            text-align: right;
            font-weight: 600;
            font-family: 'SF Mono', Monaco, monospace;
        }}

        .chart-container {{
            background: var(--card-bg);
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}

        .chart-container img {{
            width: 100%;
            height: auto;
            display: block;
        }}

        .chart-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(600px, 1fr));
            gap: 20px;
        }}

        .highlight-box {{
            background: linear-gradient(135deg, var(--secondary), var(--primary));
            color: white;
            padding: 30px;
            border-radius: 8px;
            margin-bottom: 30px;
        }}

        .highlight-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 20px;
            text-align: center;
        }}

        .highlight-item {{
            padding: 15px;
        }}

        .highlight-value {{
            font-size: 2rem;
            font-weight: 700;
            margin-bottom: 5px;
        }}

        .highlight-label {{
            font-size: 0.9rem;
            opacity: 0.9;
        }}

        footer {{
            text-align: center;
            padding: 30px;
            color: var(--text-light);
            border-top: 1px solid var(--border);
            margin-top: 40px;
        }}

        @media (max-width: 768px) {{
            .chart-grid {{
                grid-template-columns: 1fr;
            }}

            .metrics-grid {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>{title}</h1>
            <div class="subtitle">
                Period: {results['start_date']} to {results['end_date']} |
                Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
            </div>
        </header>

        <!-- Key Highlights -->
        <div class="highlight-box">
            <div class="highlight-grid">
                <div class="highlight-item">
                    <div class="highlight-value">{format_pct(strategy_metrics.get('strategy_ann_return', 0))}</div>
                    <div class="highlight-label">Strategy Annual Return</div>
                </div>
                <div class="highlight-item">
                    <div class="highlight-value">{format_number(strategy_metrics.get('strategy_sharpe', 0))}</div>
                    <div class="highlight-label">Sharpe Ratio</div>
                </div>
                <div class="highlight-item">
                    <div class="highlight-value">{format_pct(strategy_metrics.get('strategy_max_drawdown', 0))}</div>
                    <div class="highlight-label">Max Drawdown</div>
                </div>
                <div class="highlight-item">
                    <div class="highlight-value">{format_pct(avg_turnover)}</div>
                    <div class="highlight-label">Avg Weekly Turnover</div>
                </div>
            </div>
        </div>

        <!-- Performance Metrics -->
        <div class="section">
            <h2 class="section-title">Performance Metrics</h2>
            <div class="metrics-grid">
                {generate_metrics_table(strategy_metrics, "Strategy")}
                {generate_metrics_table(benchmark_metrics, "Benchmark (Equal-Weight)")}
                {generate_metrics_table(relative_metrics, "Relative Performance") if relative_metrics else ""}
            </div>
        </div>

        <!-- Cumulative Returns -->
        <div class="section">
            <h2 class="section-title">Cumulative Returns</h2>
            <div class="chart-container">
                <img src="data:image/png;base64,{charts['cumulative']}" alt="Cumulative Returns">
            </div>
        </div>

        <!-- Drawdowns -->
        <div class="section">
            <h2 class="section-title">Drawdown Analysis</h2>
            <div class="chart-container">
                <img src="data:image/png;base64,{charts['drawdowns']}" alt="Drawdowns">
            </div>
        </div>

        <!-- Rolling Metrics -->
        <div class="section">
            <h2 class="section-title">Rolling Performance Metrics</h2>
            <div class="chart-container">
                <img src="data:image/png;base64,{charts['rolling']}" alt="Rolling Metrics">
            </div>
        </div>

        <!-- Portfolio Weights -->
        <div class="section">
            <h2 class="section-title">Portfolio Weights Over Time</h2>
            <div class="chart-container">
                <img src="data:image/png;base64,{charts['weights']}" alt="Portfolio Weights">
            </div>
        </div>

        {"" if 'signals' not in charts else f'''
        <!-- Signal Heatmap -->
        <div class="section">
            <h2 class="section-title">Signal Scores by Sector</h2>
            <div class="chart-container">
                <img src="data:image/png;base64,{charts['signals']}" alt="Signal Heatmap">
            </div>
        </div>
        '''}

        <!-- Signal Rules -->
        <div class="section">
            <h2 class="section-title">Model Signal Rules</h2>
            <div class="chart-container">
                <img src="data:image/png;base64,{charts['rules']}" alt="Signal Rules">
            </div>
        </div>

        <footer>
            <p>Industry ETF Relative Performance Model | Heuristic-Based Signals</p>
            <p>Generated by industry_model visualization module</p>
        </footer>
    </div>
</body>
</html>
    """

    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(html)
        return str(path)

    return html


def generate_dashboard(
    results: Dict[str, Any],
    signal_history: Optional[pd.DataFrame] = None,
    output_dir: str = "dashboard",
) -> str:
    """
    Generate a multi-page dashboard with interactive elements.

    Args:
        results: Backtest results dictionary
        signal_history: Historical signal scores DataFrame
        output_dir: Directory to save dashboard files

    Returns:
        Path to main dashboard HTML file.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate main report
    report_path = generate_html_report(
        results=results,
        signal_scores=signal_history,
        output_path=str(output_path / "index.html"),
        title="Industry ETF Model Dashboard",
    )

    # Generate individual chart pages
    strategy_returns = results['strategy_returns']
    benchmark_returns = results['benchmark_returns']
    weights_df = results['weights']

    # Save individual charts as PNGs
    charts_dir = output_path / "charts"
    charts_dir.mkdir(exist_ok=True)

    chart_files = []

    # 1. Cumulative returns
    fig = plot_cumulative_returns(strategy_returns, benchmark_returns)
    fig.savefig(charts_dir / "cumulative_returns.png", dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    chart_files.append("cumulative_returns.png")

    # 2. Drawdowns
    fig = plot_drawdowns(strategy_returns, benchmark_returns)
    fig.savefig(charts_dir / "drawdowns.png", dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    chart_files.append("drawdowns.png")

    # 3. Weights
    fig = plot_sector_weights(weights_df)
    fig.savefig(charts_dir / "weights.png", dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    chart_files.append("weights.png")

    # 4. Rolling metrics
    fig = plot_rolling_metrics(strategy_returns, benchmark_returns)
    fig.savefig(charts_dir / "rolling_metrics.png", dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    chart_files.append("rolling_metrics.png")

    # 5. Signal heatmap
    if signal_history is not None and not signal_history.empty:
        fig = plot_signal_heatmap(signal_history)
        fig.savefig(charts_dir / "signal_heatmap.png", dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        chart_files.append("signal_heatmap.png")

    # 6. Rules summary
    fig = create_rules_summary_chart()
    fig.savefig(charts_dir / "rules_summary.png", dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    chart_files.append("rules_summary.png")

    # Export metrics to CSV
    metrics_df = pd.DataFrame({
        'Strategy': results['strategy_metrics'],
        'Benchmark': results['benchmark_metrics'],
    })
    metrics_df.to_csv(output_path / "metrics.csv")

    # Export weights to CSV
    weights_df.to_csv(output_path / "weights.csv")

    # Export returns to CSV
    returns_df = pd.DataFrame({
        'strategy': strategy_returns,
        'benchmark': benchmark_returns,
        'excess': results.get('excess_returns', strategy_returns - benchmark_returns),
    })
    returns_df.to_csv(output_path / "returns.csv")

    print(f"Dashboard generated at: {output_path}")
    print(f"  - Main report: {output_path / 'index.html'}")
    print(f"  - Charts: {charts_dir}")
    print(f"  - Metrics: {output_path / 'metrics.csv'}")
    print(f"  - Weights: {output_path / 'weights.csv'}")
    print(f"  - Returns: {output_path / 'returns.csv'}")

    return str(output_path / "index.html")


def save_charts_to_files(
    results: Dict[str, Any],
    signal_scores: Optional[pd.DataFrame] = None,
    output_dir: str = "charts",
    format: str = "png",
    dpi: int = 150,
) -> List[str]:
    """
    Save all charts as individual image files.

    Args:
        results: Backtest results dictionary
        signal_scores: Optional signal scores DataFrame
        output_dir: Directory to save charts
        format: Image format ('png', 'pdf', 'svg')
        dpi: Resolution for raster formats

    Returns:
        List of saved file paths.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    saved_files = []

    strategy_returns = results['strategy_returns']
    benchmark_returns = results['benchmark_returns']
    weights_df = results['weights']

    charts_to_save = [
        ('cumulative_returns', lambda: plot_cumulative_returns(strategy_returns, benchmark_returns)),
        ('drawdowns', lambda: plot_drawdowns(strategy_returns, benchmark_returns)),
        ('weights', lambda: plot_sector_weights(weights_df)),
        ('rolling_metrics', lambda: plot_rolling_metrics(strategy_returns, benchmark_returns)),
        ('rules_summary', lambda: create_rules_summary_chart()),
    ]

    if signal_scores is not None and not signal_scores.empty:
        charts_to_save.append(
            ('signal_heatmap', lambda: plot_signal_heatmap(signal_scores))
        )

    for name, chart_func in charts_to_save:
        fig = chart_func()
        filepath = output_path / f"{name}.{format}"
        fig.savefig(filepath, dpi=dpi, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        saved_files.append(str(filepath))
        print(f"Saved: {filepath}")

    return saved_files
