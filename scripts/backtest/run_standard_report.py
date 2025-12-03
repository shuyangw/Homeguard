"""
Standalone Standard Backtest Report Script.

Runs any registered strategy and generates monthly/overall performance report.

Usage:
    python scripts/backtest/run_standard_report.py --strategy MovingAverageCrossover --symbols iex_etfs
    python scripts/backtest/run_standard_report.py --strategy RSIMeanReversion --symbols leveraged_etfs --start 2017-01-01
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import pandas as pd
from datetime import datetime

from src.utils.logger import logger
from src.strategies.registry import get_strategy_class
from src.backtesting.engine.backtest_engine import BacktestEngine
from src.backtesting.reporting.standard_report import StandardReportGenerator
from src.settings import PROJECT_ROOT as PROJ_ROOT


def load_symbol_list(list_name: str) -> list:
    """Load symbols from backtest_lists/ directory."""
    backtest_lists_dir = PROJ_ROOT / "backtest_lists"

    # Try exact match first
    if Path(list_name).exists():
        csv_path = Path(list_name)
    else:
        matches = list(backtest_lists_dir.glob(f"{list_name}*.csv"))
        if not matches:
            matches = list(backtest_lists_dir.glob(f"*{list_name}*.csv"))

        if not matches:
            raise FileNotFoundError(f"No backtest list found matching: {list_name}")

        csv_path = matches[0]

    df = pd.read_csv(csv_path)
    logger.info(f"Loading symbols from: {csv_path.name}")

    # Find symbol column
    for col in ['Symbol', 'symbol', 'SYMBOL', 'ticker', 'Ticker']:
        if col in df.columns:
            symbols = df[col].dropna().astype(str).tolist()
            logger.info(f"Loaded {len(symbols)} symbols")
            return symbols

    # Fallback to first column
    symbols = df.iloc[:, 0].dropna().astype(str).tolist()
    logger.info(f"Loaded {len(symbols)} symbols")
    return symbols


def main():
    parser = argparse.ArgumentParser(description='Run standard backtest report')
    parser.add_argument('--strategy', required=True, help='Strategy name from registry')
    parser.add_argument('--symbols', required=True, help='Symbol list name or comma-separated symbols')
    parser.add_argument('--start', default='2017-01-01', help='Start date (default: 2017-01-01)')
    parser.add_argument('--end', default=None, help='End date (default: today)')
    parser.add_argument('--capital', type=float, default=100000, help='Initial capital')
    parser.add_argument('--fees', type=float, default=0.001, help='Transaction fees')
    parser.add_argument('--output-dir', default=None, help='Output directory for reports (default: settings.ini output_dir)')
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("STANDARD BACKTEST REPORT GENERATOR")
    logger.info("=" * 70)

    # Load symbols
    if ',' in args.symbols:
        symbols = [s.strip() for s in args.symbols.split(',')]
        logger.info(f"Using {len(symbols)} symbols from command line")
    else:
        symbols = load_symbol_list(args.symbols)

    # Get strategy
    logger.info(f"Loading strategy: {args.strategy}")
    try:
        strategy_cls = get_strategy_class(args.strategy)
        strategy = strategy_cls()
        logger.info(f"Strategy loaded: {strategy_cls.__name__}")
    except Exception as e:
        logger.error(f"Failed to load strategy '{args.strategy}': {e}")
        return 1

    # Run backtest
    end_date = args.end or datetime.now().strftime('%Y-%m-%d')

    engine = BacktestEngine(
        initial_capital=args.capital,
        fees=args.fees,
        slippage=0.0005
    )

    logger.info(f"\nRunning backtest...")
    logger.info(f"  Strategy: {args.strategy}")
    logger.info(f"  Symbols: {len(symbols)}")
    logger.info(f"  Period: {args.start} to {end_date}")
    logger.info(f"  Capital: ${args.capital:,.0f}")

    try:
        portfolio = engine.run(
            strategy=strategy,
            symbols=symbols,
            start_date=args.start,
            end_date=end_date
        )
    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Check if equity curve exists
    if portfolio.equity_curve is None or len(portfolio.equity_curve) == 0:
        logger.error("No equity curve generated - backtest may have failed")
        return 1

    # Generate report
    logger.info("\nGenerating standard report...")
    output_dir = Path(args.output_dir) if args.output_dir else None
    reporter = StandardReportGenerator(output_dir=output_dir)

    try:
        results = reporter.generate_report(
            equity_curve=portfolio.equity_curve,
            strategy_name=args.strategy,
            symbols=symbols,
            start_date=args.start,
            end_date=end_date,
            initial_capital=args.capital
        )
    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    logger.success("\nReport generation complete!")
    logger.info(f"  Markdown: {results['markdown_path']}")
    logger.info(f"  CSV: {results['csv_path']}")

    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
