#!/usr/bin/env python3
"""
Industry ETF Relative Performance Model - CLI Entry Point

Usage:
    python -m industry_model.run --backtest
    python -m industry_model.run --walk-forward
    python -m industry_model.run --signals
"""

import argparse
import os
import sys
from datetime import date, timedelta
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from industry_model.config import ModelConfig, SECTOR_ETFS
from industry_model.portfolio.combiner import SignalCombiner
from industry_model.portfolio.weights import PortfolioWeights


def get_api_keys():
    """Get API keys from environment."""
    fmp_key = os.environ.get("FMP_API_KEY")
    gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

    if not fmp_key:
        print("Warning: FMP_API_KEY not set. ETF data will not be available.")

    return fmp_key, gemini_key


def run_signals(as_of_date: date = None, gemini_api_key: str = None):
    """Generate and display current signals."""
    as_of_date = as_of_date or date.today()

    print(f"\n{'=' * 60}")
    print(f"SIGNAL GENERATION - As of {as_of_date}")
    print("=" * 60)

    config = ModelConfig()
    combiner = SignalCombiner(config=config, gemini_api_key=gemini_api_key)

    # Get signal breakdown
    try:
        scores_df = combiner.generate_combined_scores(as_of_date, return_individual=True)

        print(f"\nActive Signals: {', '.join(combiner.get_active_signals())}")
        print(f"\n--- Individual Signal Scores ---")
        print(scores_df.to_string())

        print(f"\n--- Sector Rankings ---")
        rankings = combiner.generate_rankings(as_of_date)
        rankings_sorted = rankings.sort_values()
        for i, (sector, rank) in enumerate(rankings_sorted.items(), 1):
            score = scores_df.loc[sector, "combined_score"]
            print(f"{i:2}. {sector}: Rank {int(rank)}, Score {score:+.3f}")

    except Exception as e:
        print(f"Error generating signals: {e}")
        import traceback
        traceback.print_exc()


def run_backtest(
    start_date: str = None,
    end_date: str = None,
    fmp_api_key: str = None,
    gemini_api_key: str = None,
):
    """Run full backtest."""
    from industry_model.backtest.engine import BacktestEngine

    # Default dates
    if not end_date:
        end_date = date.today().strftime("%Y-%m-%d")
    if not start_date:
        start_date = (date.today() - timedelta(days=365 * 5)).strftime("%Y-%m-%d")

    print(f"\n{'=' * 60}")
    print(f"BACKTEST - {start_date} to {end_date}")
    print("=" * 60)

    try:
        engine = BacktestEngine(
            fmp_api_key=fmp_api_key,
            gemini_api_key=gemini_api_key,
        )

        results = engine.run_backtest(
            start_date=start_date,
            end_date=end_date,
        )

        engine.print_results(results)

        return results

    except Exception as e:
        print(f"Error running backtest: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_walk_forward(
    start_date: str = None,
    end_date: str = None,
    n_windows: int = 5,
    fmp_api_key: str = None,
    gemini_api_key: str = None,
):
    """Run walk-forward analysis."""
    from industry_model.backtest.walk_forward import WalkForwardAnalyzer

    # Default dates
    if not end_date:
        end_date = date.today().strftime("%Y-%m-%d")
    if not start_date:
        start_date = (date.today() - timedelta(days=365 * 5)).strftime("%Y-%m-%d")

    print(f"\n{'=' * 60}")
    print(f"WALK-FORWARD ANALYSIS - {start_date} to {end_date}")
    print(f"Windows: {n_windows}")
    print("=" * 60)

    try:
        analyzer = WalkForwardAnalyzer(
            fmp_api_key=fmp_api_key,
            gemini_api_key=gemini_api_key,
        )

        results = analyzer.run_walk_forward(
            start_date=start_date,
            end_date=end_date,
            n_windows=n_windows,
        )

        analyzer.print_walk_forward_report(results)

        return results

    except Exception as e:
        print(f"Error running walk-forward analysis: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_portfolio_weights(as_of_date: date = None, gemini_api_key: str = None):
    """Generate and display current portfolio weights."""
    as_of_date = as_of_date or date.today()

    print(f"\n{'=' * 60}")
    print(f"PORTFOLIO WEIGHTS - As of {as_of_date}")
    print("=" * 60)

    config = ModelConfig()
    weights_gen = PortfolioWeights(config=config, gemini_api_key=gemini_api_key)

    try:
        weights = weights_gen.generate_weights(as_of_date)
        active = weights_gen.get_active_weights(weights)

        print(f"\n--- Target Weights ---")
        weights_sorted = weights.sort_values(ascending=False)
        for sector in weights_sorted.index:
            w = weights[sector]
            a = active[sector]
            sign = "+" if a > 0 else ""
            print(f"{sector}: {w:.1%} ({sign}{a:.1%} active)")

        print(f"\nTotal: {weights.sum():.1%}")
        print(f"Benchmark: {1/len(SECTOR_ETFS):.1%} each sector")

    except Exception as e:
        print(f"Error generating weights: {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(
        description="Industry ETF Relative Performance Model"
    )

    parser.add_argument(
        "--signals",
        action="store_true",
        help="Generate current signals",
    )
    parser.add_argument(
        "--weights",
        action="store_true",
        help="Generate current portfolio weights",
    )
    parser.add_argument(
        "--backtest",
        action="store_true",
        help="Run full backtest",
    )
    parser.add_argument(
        "--walk-forward",
        action="store_true",
        help="Run walk-forward analysis",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="End date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--as-of",
        type=str,
        default=None,
        help="As-of date for signals/weights (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--windows",
        type=int,
        default=5,
        help="Number of walk-forward windows",
    )

    args = parser.parse_args()

    # Get API keys
    fmp_key, gemini_key = get_api_keys()

    # Parse as-of date
    as_of_date = None
    if args.as_of:
        from datetime import datetime
        as_of_date = datetime.strptime(args.as_of, "%Y-%m-%d").date()

    # Run requested action
    if args.signals:
        run_signals(as_of_date=as_of_date, gemini_api_key=gemini_key)
    elif args.weights:
        run_portfolio_weights(as_of_date=as_of_date, gemini_api_key=gemini_key)
    elif args.backtest:
        run_backtest(
            start_date=args.start_date,
            end_date=args.end_date,
            fmp_api_key=fmp_key,
            gemini_api_key=gemini_key,
        )
    elif args.walk_forward:
        run_walk_forward(
            start_date=args.start_date,
            end_date=args.end_date,
            n_windows=args.windows,
            fmp_api_key=fmp_key,
            gemini_api_key=gemini_key,
        )
    else:
        # Default: show signals
        print("Industry ETF Relative Performance Model")
        print("-" * 40)
        print("Usage:")
        print("  python -m industry_model.run --signals     # Current signals")
        print("  python -m industry_model.run --weights     # Target weights")
        print("  python -m industry_model.run --backtest    # Full backtest")
        print("  python -m industry_model.run --walk-forward # Walk-forward analysis")
        print("\nRun with --help for all options")


if __name__ == "__main__":
    main()
