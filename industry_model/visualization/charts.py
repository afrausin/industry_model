"""
Chart Generation for Industry Model

Creates matplotlib visualizations for backtest analysis.
"""

from typing import Dict, Optional, List, Any
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec

from ..config import SECTOR_ETFS, SECTOR_NAMES


# Style configuration
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = {
    'strategy': '#2E86AB',
    'benchmark': '#A23B72',
    'positive': '#28A745',
    'negative': '#DC3545',
    'neutral': '#6C757D',
}

SECTOR_COLORS = {
    'XLK': '#0066CC',   # Tech - Blue
    'XLF': '#006600',   # Financials - Green
    'XLE': '#CC6600',   # Energy - Orange
    'XLV': '#CC0066',   # Healthcare - Pink
    'XLI': '#666666',   # Industrials - Gray
    'XLP': '#009999',   # Staples - Teal
    'XLY': '#9900CC',   # Discretionary - Purple
    'XLU': '#CCCC00',   # Utilities - Yellow
    'XLB': '#996633',   # Materials - Brown
    'XLRE': '#FF6666',  # Real Estate - Light Red
    'XLC': '#6699FF',   # Communication - Light Blue
}


def plot_cumulative_returns(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    title: str = "Strategy vs Benchmark Performance",
    save_path: Optional[Path] = None,
    figsize: tuple = (12, 6),
) -> plt.Figure:
    """
    Plot cumulative returns comparison.

    Args:
        strategy_returns: Strategy return series.
        benchmark_returns: Benchmark return series.
        title: Chart title.
        save_path: Path to save the figure.
        figsize: Figure size.

    Returns:
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Calculate cumulative returns
    strategy_cum = (1 + strategy_returns).cumprod()
    benchmark_cum = (1 + benchmark_returns).cumprod()

    # Plot
    ax.plot(strategy_cum.index, strategy_cum.values,
            label='Strategy', color=COLORS['strategy'], linewidth=2)
    ax.plot(benchmark_cum.index, benchmark_cum.values,
            label='Benchmark (Equal-Weight)', color=COLORS['benchmark'],
            linewidth=2, linestyle='--')

    # Fill between
    ax.fill_between(strategy_cum.index, strategy_cum.values, benchmark_cum.values,
                    where=strategy_cum.values >= benchmark_cum.values,
                    alpha=0.3, color=COLORS['positive'], label='Outperformance')
    ax.fill_between(strategy_cum.index, strategy_cum.values, benchmark_cum.values,
                    where=strategy_cum.values < benchmark_cum.values,
                    alpha=0.3, color=COLORS['negative'], label='Underperformance')

    # Formatting
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=11)
    ax.set_ylabel('Cumulative Return', fontsize=11)
    ax.legend(loc='upper left', fontsize=10)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=45)

    # Add final values annotation
    final_strat = strategy_cum.iloc[-1]
    final_bench = benchmark_cum.iloc[-1]
    ax.annotate(f'Strategy: {(final_strat-1)*100:.1f}%',
                xy=(strategy_cum.index[-1], final_strat),
                xytext=(10, 10), textcoords='offset points',
                fontsize=10, color=COLORS['strategy'])
    ax.annotate(f'Benchmark: {(final_bench-1)*100:.1f}%',
                xy=(benchmark_cum.index[-1], final_bench),
                xytext=(10, -15), textcoords='offset points',
                fontsize=10, color=COLORS['benchmark'])

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_drawdowns(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    title: str = "Drawdown Analysis",
    save_path: Optional[Path] = None,
    figsize: tuple = (12, 5),
) -> plt.Figure:
    """
    Plot drawdown comparison.

    Args:
        strategy_returns: Strategy return series.
        benchmark_returns: Benchmark return series.
        title: Chart title.
        save_path: Path to save the figure.
        figsize: Figure size.

    Returns:
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=figsize)

    def calculate_drawdown(returns):
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        return drawdown

    strategy_dd = calculate_drawdown(strategy_returns)
    benchmark_dd = calculate_drawdown(benchmark_returns)

    ax.fill_between(strategy_dd.index, strategy_dd.values * 100, 0,
                    alpha=0.5, color=COLORS['strategy'], label='Strategy')
    ax.fill_between(benchmark_dd.index, benchmark_dd.values * 100, 0,
                    alpha=0.3, color=COLORS['benchmark'], label='Benchmark')

    ax.plot(strategy_dd.index, strategy_dd.values * 100,
            color=COLORS['strategy'], linewidth=1.5)
    ax.plot(benchmark_dd.index, benchmark_dd.values * 100,
            color=COLORS['benchmark'], linewidth=1.5, linestyle='--')

    # Formatting
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=11)
    ax.set_ylabel('Drawdown (%)', fontsize=11)
    ax.legend(loc='lower left', fontsize=10)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=45)

    # Add max drawdown annotations
    max_strat_dd = strategy_dd.min()
    max_bench_dd = benchmark_dd.min()
    ax.axhline(y=max_strat_dd * 100, color=COLORS['strategy'],
               linestyle=':', alpha=0.7)
    ax.axhline(y=max_bench_dd * 100, color=COLORS['benchmark'],
               linestyle=':', alpha=0.7)

    ax.text(strategy_dd.index[0], max_strat_dd * 100 - 1,
            f'Max DD: {max_strat_dd*100:.1f}%', fontsize=9,
            color=COLORS['strategy'])

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_sector_weights(
    weights_df: pd.DataFrame,
    title: str = "Sector Weights Over Time",
    save_path: Optional[Path] = None,
    figsize: tuple = (14, 7),
) -> plt.Figure:
    """
    Plot sector weight evolution as stacked area chart.

    Args:
        weights_df: DataFrame with dates as index and sectors as columns.
        title: Chart title.
        save_path: Path to save the figure.
        figsize: Figure size.

    Returns:
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Ensure columns are in consistent order
    sectors = [s for s in SECTOR_ETFS if s in weights_df.columns]
    weights_ordered = weights_df[sectors]

    # Stacked area chart
    ax.stackplot(weights_ordered.index, weights_ordered.T.values,
                 labels=[f"{s} ({SECTOR_NAMES.get(s, s)})" for s in sectors],
                 colors=[SECTOR_COLORS.get(s, '#999999') for s in sectors],
                 alpha=0.8)

    # Add equal-weight line
    equal_weight = 1.0 / len(sectors)
    ax.axhline(y=equal_weight * len(sectors) / 2, color='black',
               linestyle='--', linewidth=1, alpha=0.5, label='Equal Weight')

    # Formatting
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=11)
    ax.set_ylabel('Portfolio Weight', fontsize=11)
    ax.set_ylim(0, 1)
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=45)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_signal_heatmap(
    signal_scores: pd.DataFrame,
    title: str = "Signal Scores by Sector",
    save_path: Optional[Path] = None,
    figsize: tuple = (12, 8),
) -> plt.Figure:
    """
    Plot signal scores as heatmap.

    Args:
        signal_scores: DataFrame with sectors as index and signals as columns.
        title: Chart title.
        save_path: Path to save the figure.
        figsize: Figure size.

    Returns:
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Ensure consistent ordering
    sectors = [s for s in SECTOR_ETFS if s in signal_scores.index]
    scores = signal_scores.loc[sectors]

    # Create heatmap
    im = ax.imshow(scores.values, cmap='RdYlGn', aspect='auto',
                   vmin=-1, vmax=1)

    # Set ticks
    ax.set_xticks(np.arange(len(scores.columns)))
    ax.set_yticks(np.arange(len(sectors)))
    ax.set_xticklabels(scores.columns, rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels([f"{s} ({SECTOR_NAMES.get(s, '')})" for s in sectors],
                       fontsize=10)

    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.set_ylabel('Score', rotation=-90, va="bottom", fontsize=11)

    # Add value annotations
    for i in range(len(sectors)):
        for j in range(len(scores.columns)):
            value = scores.iloc[i, j]
            color = 'white' if abs(value) > 0.5 else 'black'
            ax.text(j, i, f'{value:.2f}', ha='center', va='center',
                    color=color, fontsize=9)

    ax.set_title(title, fontsize=14, fontweight='bold')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_rolling_metrics(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    window: int = 52,
    title: str = "Rolling Performance Metrics",
    save_path: Optional[Path] = None,
    figsize: tuple = (12, 10),
) -> plt.Figure:
    """
    Plot rolling Sharpe ratio and returns.

    Args:
        strategy_returns: Strategy return series.
        benchmark_returns: Benchmark return series.
        window: Rolling window size.
        title: Chart title.
        save_path: Path to save the figure.
        figsize: Figure size.

    Returns:
        Matplotlib figure.
    """
    fig, axes = plt.subplots(3, 1, figsize=figsize)

    # Rolling returns (annualized)
    ax1 = axes[0]
    rolling_strat_ret = strategy_returns.rolling(window).mean() * 52
    rolling_bench_ret = benchmark_returns.rolling(window).mean() * 52

    ax1.plot(rolling_strat_ret.index, rolling_strat_ret.values * 100,
             label='Strategy', color=COLORS['strategy'], linewidth=2)
    ax1.plot(rolling_bench_ret.index, rolling_bench_ret.values * 100,
             label='Benchmark', color=COLORS['benchmark'], linewidth=2, linestyle='--')
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax1.set_ylabel('Rolling Annual Return (%)', fontsize=11)
    ax1.set_title(f'{title} ({window}-week window)', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=10)

    # Rolling Sharpe
    ax2 = axes[1]
    rolling_strat_sharpe = (strategy_returns.rolling(window).mean() /
                            strategy_returns.rolling(window).std()) * np.sqrt(52)
    rolling_bench_sharpe = (benchmark_returns.rolling(window).mean() /
                            benchmark_returns.rolling(window).std()) * np.sqrt(52)

    ax2.plot(rolling_strat_sharpe.index, rolling_strat_sharpe.values,
             label='Strategy', color=COLORS['strategy'], linewidth=2)
    ax2.plot(rolling_bench_sharpe.index, rolling_bench_sharpe.values,
             label='Benchmark', color=COLORS['benchmark'], linewidth=2, linestyle='--')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.axhline(y=1, color='green', linestyle=':', linewidth=1, alpha=0.7,
                label='Sharpe = 1')
    ax2.set_ylabel('Rolling Sharpe Ratio', fontsize=11)
    ax2.legend(loc='upper left', fontsize=10)

    # Rolling excess return
    ax3 = axes[2]
    excess_returns = strategy_returns - benchmark_returns
    rolling_excess = excess_returns.rolling(window).mean() * 52

    ax3.fill_between(rolling_excess.index, rolling_excess.values * 100, 0,
                     where=rolling_excess.values >= 0,
                     alpha=0.5, color=COLORS['positive'])
    ax3.fill_between(rolling_excess.index, rolling_excess.values * 100, 0,
                     where=rolling_excess.values < 0,
                     alpha=0.5, color=COLORS['negative'])
    ax3.plot(rolling_excess.index, rolling_excess.values * 100,
             color='black', linewidth=1)
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax3.set_ylabel('Rolling Excess Return (%)', fontsize=11)
    ax3.set_xlabel('Date', fontsize=11)

    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))

    plt.xticks(rotation=45)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_sector_performance(
    returns_df: pd.DataFrame,
    weights_df: pd.DataFrame,
    title: str = "Sector Performance Contribution",
    save_path: Optional[Path] = None,
    figsize: tuple = (14, 6),
) -> plt.Figure:
    """
    Plot sector performance contribution.

    Args:
        returns_df: DataFrame with sector returns.
        weights_df: DataFrame with sector weights.
        title: Chart title.
        save_path: Path to save the figure.
        figsize: Figure size.

    Returns:
        Matplotlib figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Calculate total returns per sector
    sectors = [s for s in SECTOR_ETFS if s in returns_df.columns]
    total_returns = ((1 + returns_df[sectors]).prod() - 1) * 100

    # Left: Sector returns bar chart
    ax1 = axes[0]
    colors = [COLORS['positive'] if r > 0 else COLORS['negative']
              for r in total_returns.values]
    bars = ax1.bar(range(len(sectors)), total_returns.values, color=colors, alpha=0.8)
    ax1.set_xticks(range(len(sectors)))
    ax1.set_xticklabels(sectors, rotation=45, ha='right')
    ax1.set_ylabel('Total Return (%)', fontsize=11)
    ax1.set_title('Sector Total Returns', fontsize=12, fontweight='bold')
    ax1.axhline(y=0, color='black', linewidth=0.5)

    # Add value labels
    for bar, val in zip(bars, total_returns.values):
        ypos = val + 1 if val > 0 else val - 3
        ax1.text(bar.get_x() + bar.get_width()/2, ypos, f'{val:.1f}%',
                 ha='center', va='bottom' if val > 0 else 'top', fontsize=9)

    # Right: Average weights
    ax2 = axes[1]
    avg_weights = weights_df[sectors].mean() * 100
    equal_weight = 100 / len(sectors)

    ax2.bar(range(len(sectors)), avg_weights.values,
            color=[SECTOR_COLORS.get(s, '#999999') for s in sectors], alpha=0.8)
    ax2.axhline(y=equal_weight, color='red', linestyle='--', linewidth=2,
                label=f'Equal Weight ({equal_weight:.1f}%)')
    ax2.set_xticks(range(len(sectors)))
    ax2.set_xticklabels(sectors, rotation=45, ha='right')
    ax2.set_ylabel('Average Weight (%)', fontsize=11)
    ax2.set_title('Average Portfolio Weights', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=10)

    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def create_rules_summary_chart(
    save_path: Optional[Path] = None,
    figsize: tuple = (14, 10),
) -> plt.Figure:
    """
    Create a visual summary of the model rules/signals.

    Args:
        save_path: Path to save the figure.
        figsize: Figure size.

    Returns:
        Matplotlib figure.
    """
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(4, 3, figure=fig, hspace=0.4, wspace=0.3)

    # Title
    fig.suptitle('Industry Model Signal Rules', fontsize=16, fontweight='bold', y=0.98)

    # Signal descriptions
    signals = [
        ("Quadrant\n(Hedgeye)", "Growth + Inflation\nmomentum determines\nmacro regime",
         "Q1: Tech, Discretionary\nQ2: Energy, Materials\nQ3: Utilities, Staples\nQ4: Energy, Utilities"),
        ("Yield Curve", "10Y-2Y spread\nsteepening/flattening",
         "Steep: Financials +\nFlat: Utilities, REITs +"),
        ("Credit", "HY Spreads + NFCI\nconditions",
         "Tight: Risk-on sectors\nWide: Defensives"),
        ("High-Freq", "WEI + Claims\n+ GDPNow",
         "Rising: Cyclicals\nFalling: Defensives"),
        ("Housing", "Starts, Permits\nMortgage rates",
         "Strong: Materials, Ind.\nWeak: Less correlated"),
        ("Consumer", "Sentiment\nRetail Sales",
         "High: Discretionary\nLow: Staples"),
        ("Dollar", "Trade-weighted\nDollar Index",
         "Weak: Exporters\nStrong: Domestic"),
        ("Liquidity", "Fed Balance Sheet\nM2 Growth",
         "QE: Risk-on\nQT: Defensives"),
        ("Volatility", "VIX regime\nlevels",
         "<15: High beta\n>25: Low beta"),
        ("Gemini AI", "AI quadrant probs\n+ recommendations",
         "Based on Fed docs\nand macro analysis"),
        ("Labor", "Payrolls, Claims\nJob openings",
         "Strong: Cyclicals\nWeak: Defensives"),
        ("Energy", "Inflation + Dollar\ncorrelation",
         "Rising inflation:\nEnergy, Materials"),
    ]

    for i, (name, description, logic) in enumerate(signals):
        row = i // 3
        col = i % 3
        ax = fig.add_subplot(gs[row, col])

        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')

        # Box
        rect = plt.Rectangle((0.5, 0.5), 9, 9, fill=True,
                              facecolor='#f8f9fa', edgecolor='#dee2e6',
                              linewidth=2, alpha=0.8)
        ax.add_patch(rect)

        # Signal name
        ax.text(5, 8.5, name, ha='center', va='top', fontsize=11,
                fontweight='bold', color=COLORS['strategy'])

        # Description
        ax.text(5, 6.5, description, ha='center', va='top', fontsize=9,
                color='#495057', style='italic')

        # Logic
        ax.text(5, 3.5, logic, ha='center', va='top', fontsize=9,
                color='#212529', family='monospace')

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig
