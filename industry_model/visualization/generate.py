#!/usr/bin/env python3
"""
Generate Visualizations for Industry Model

Run backtest and generate all charts/reports.

Usage:
    python -m industry_model.visualization.generate
    python -m industry_model.visualization.generate --output-dir reports
    python -m industry_model.visualization.generate --charts-only
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def main():
    parser = argparse.ArgumentParser(
        description="Generate Industry Model backtest visualizations"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="industry_model/output",
        help="Output directory for reports and charts",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default="2020-01-01",
        help="Backtest start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="Backtest end date (YYYY-MM-DD), defaults to today",
    )
    parser.add_argument(
        "--charts-only",
        action="store_true",
        help="Only save individual chart files, not HTML report",
    )
    parser.add_argument(
        "--html-only",
        action="store_true",
        help="Only generate HTML report, not individual charts",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="rank_tilt",
        choices=["rank_tilt", "equal_weight", "signal_weighted"],
        help="Portfolio weighting method",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="png",
        choices=["png", "pdf", "svg"],
        help="Chart output format",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Chart resolution (DPI)",
    )

    args = parser.parse_args()

    # Set default end date
    if args.end_date is None:
        args.end_date = datetime.now().strftime("%Y-%m-%d")

    print("=" * 60)
    print("INDUSTRY MODEL VISUALIZATION GENERATOR")
    print("=" * 60)
    print(f"Period: {args.start_date} to {args.end_date}")
    print(f"Output: {args.output_dir}")
    print(f"Method: {args.method}")
    print()

    # Check for API keys
    fmp_api_key = os.environ.get("FMP_API_KEY")
    gemini_api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")

    if not fmp_api_key:
        print("ERROR: FMP_API_KEY environment variable not set")
        print("Set it with: export FMP_API_KEY=your_key_here")
        sys.exit(1)

    # Import after path setup
    try:
        from industry_model.backtest.engine import BacktestEngine
        from industry_model.visualization.report import (
            generate_html_report,
            generate_dashboard,
            save_charts_to_files,
        )
    except ImportError as e:
        print(f"ERROR: Failed to import modules: {e}")
        print("Make sure you're running from the project root directory")
        sys.exit(1)

    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Run backtest
    print("Running backtest...")
    try:
        engine = BacktestEngine(
            fmp_api_key=fmp_api_key,
            gemini_api_key=gemini_api_key,
        )

        results = engine.run_backtest(
            start_date=args.start_date,
            end_date=args.end_date,
            method=args.method,
        )

        # Print summary
        engine.print_results(results)

    except Exception as e:
        print(f"ERROR: Backtest failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Generate visualizations
    print()
    print("Generating visualizations...")

    try:
        if args.charts_only:
            # Save individual charts
            charts_dir = output_path / "charts"
            saved = save_charts_to_files(
                results=results,
                output_dir=str(charts_dir),
                format=args.format,
                dpi=args.dpi,
            )
            print(f"\nSaved {len(saved)} charts to {charts_dir}")

        elif args.html_only:
            # Generate HTML report only
            report_path = generate_html_report(
                results=results,
                output_path=str(output_path / "report.html"),
            )
            print(f"\nHTML report saved to: {report_path}")

        else:
            # Generate full dashboard (HTML + charts + CSVs)
            dashboard_path = generate_dashboard(
                results=results,
                output_dir=str(output_path),
            )
            print(f"\nDashboard generated at: {dashboard_path}")

    except Exception as e:
        print(f"ERROR: Visualization generation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print()
    print("=" * 60)
    print("COMPLETE")
    print("=" * 60)
    print(f"Open {output_path / 'index.html'} in your browser to view the report")


if __name__ == "__main__":
    main()
