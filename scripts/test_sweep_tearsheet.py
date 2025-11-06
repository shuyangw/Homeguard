"""
Quick test script to verify sweep backtest tearsheet generation.

This script runs a minimal sweep backtest and checks if tearsheets are generated.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from backtesting.engine.backtest_engine import BacktestEngine
from backtesting.optimization.sweep_runner import SweepRunner
from backtesting.base.strategy import LongOnlyStrategy
from backtesting.utils.indicators import Indicators
from utils import logger


# Simple test strategy
class SimpleBreakout(LongOnlyStrategy):
    """Simple breakout strategy for testing."""

    def __init__(self, sma_period: int = 20):
        super().__init__(sma_period=sma_period)

    def validate_parameters(self) -> None:
        pass

    def calculate_indicators(self, data: "pd.DataFrame") -> "pd.DataFrame":
        # Handle both lowercase and uppercase column names
        close_col = 'close' if 'close' in data.columns else 'Close'
        data['sma'] = Indicators.sma(data[close_col], self.params['sma_period'])
        return data

    def generate_long_signals(self, data: "pd.DataFrame") -> "pd.DataFrame":
        # Handle both lowercase and uppercase column names
        close_col = 'close' if 'close' in data.columns else 'Close'

        # Entry: Price crosses above SMA
        entry = ((data[close_col] > data['sma']) &
                (data[close_col].shift(1) <= data['sma'].shift(1))).astype(int)

        # Exit: Price crosses below SMA
        exit_signal = ((data[close_col] < data['sma']) &
                      (data[close_col].shift(1) >= data['sma'].shift(1))).astype(int)

        data['entries'] = entry
        data['exits'] = exit_signal
        return data


def main():
    """Run a small sweep backtest to test tearsheet generation."""

    logger.header("SWEEP BACKTEST TEARSHEET TEST")
    logger.blank()

    # Setup
    engine = BacktestEngine(
        initial_capital=100000,
        fees=0.001
    )

    # Create sweep runner
    runner = SweepRunner(
        engine=engine,
        max_workers=2,
        show_progress=True
    )

    # Create strategy
    strategy = SimpleBreakout(sma_period=20)

    # Define test parameters
    symbols = ["AAPL", "MSFT"]  # Small test with 2 symbols
    start_date = "2023-01-01"
    end_date = "2023-03-31"

    logger.info(f"Running sweep for: {symbols}")
    logger.info(f"Period: {start_date} to {end_date}")
    logger.blank()

    # Run sweep with full reports
    output_dir = Path("c:/Users/qwqw1/Dropbox/cs/stonk/logs") / "test_sweep_tearsheet"

    df = runner.run_and_report(
        strategy=strategy,
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        output_dir=output_dir,
        export_csv=True,
        export_html=True,
        parallel=False  # Sequential for clearer logging
    )

    logger.blank()
    logger.separator()
    logger.header("VERIFICATION")
    logger.separator()

    # Check what was generated
    if output_dir.exists():
        logger.success(f"Output directory created: {output_dir}")

        # List all files
        all_files = list(output_dir.rglob("*"))
        logger.info(f"Total files generated: {len(all_files)}")

        # Check for tearsheets directory
        tearsheets_dir = output_dir / "tearsheets"
        if tearsheets_dir.exists():
            tearsheet_files = list(tearsheets_dir.glob("*.html"))
            logger.success(f"Tearsheets directory exists with {len(tearsheet_files)} files")
            for tf in tearsheet_files:
                logger.info(f"  - {tf.name}")
        else:
            logger.error("TEARSHEETS DIRECTORY NOT FOUND!")
            logger.error("This means portfolio objects were not available for tearsheet generation")

        # Check for trades directory
        trades_dir = output_dir / "trades"
        if trades_dir.exists():
            trade_files = list(trades_dir.glob("*.csv"))
            logger.success(f"Trades directory exists with {len(trade_files)} files")
        else:
            logger.error("TRADES DIRECTORY NOT FOUND!")

        logger.blank()
        logger.info("Directory structure:")
        for item in sorted(output_dir.rglob("*")):
            if item.is_file():
                rel_path = item.relative_to(output_dir)
                logger.info(f"  {rel_path}")
    else:
        logger.error(f"Output directory not created: {output_dir}")

    logger.blank()
    logger.separator()
    logger.header("TEST COMPLETE")
    logger.separator()
    logger.blank()


if __name__ == "__main__":
    main()
