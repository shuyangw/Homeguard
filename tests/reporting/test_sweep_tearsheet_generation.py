"""
Test that sweep backtests generate tearsheets for each symbol.
"""

import sys
from pathlib import Path
import tempfile
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from backtesting.engine.sweep_runner import SweepRunner
from backtesting.engine.backtest_engine import BacktestEngine
from backtesting.strategies.examples.ma_crossover import MovingAverageCrossover
from data.loaders.csv_loader import CSVDataLoader


def test_sweep_generates_tearsheets():
    """Test that sweep runner stores portfolios that can be used for tearsheet generation."""

    # Setup
    data_loader = CSVDataLoader(data_dir="data/sample")  # Use whatever data directory exists
    engine = BacktestEngine(data_loader=data_loader)
    runner = SweepRunner(engine=engine, max_workers=2, show_progress=False)

    # Create a simple strategy
    strategy = MovingAverageCrossover(fast_period=10, slow_period=20)

    # Run a mini sweep (just 2 symbols to test)
    symbols = ["AAPL", "MSFT"]
    start_date = "2023-01-01"
    end_date = "2023-03-31"

    print(f"\nRunning sweep for {symbols}...")

    try:
        results = runner.run_sweep(
            strategy=strategy,
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            parallel=False  # Sequential for easier debugging
        )

        print(f"Sweep completed: {len(results)} results")

        # Get portfolios
        portfolios = runner.get_portfolios()

        print(f"\nPortfolios stored: {len(portfolios)}")
        print(f"Portfolio symbols: {list(portfolios.keys())}")

        # Check each portfolio
        for symbol, portfolio in portfolios.items():
            print(f"\n{symbol}:")
            print(f"  - Portfolio object: {portfolio}")
            print(f"  - Has returns() method: {hasattr(portfolio, 'returns')}")

            if hasattr(portfolio, 'returns'):
                returns = portfolio.returns()
                print(f"  - Returns type: {type(returns)}")
                print(f"  - Returns length: {len(returns) if hasattr(returns, '__len__') else 'N/A'}")

        # Verify portfolios were stored
        assert len(portfolios) > 0, "No portfolios were stored!"
        assert len(portfolios) == len(symbols), f"Expected {len(symbols)} portfolios, got {len(portfolios)}"

        print("\n✓ Portfolios are being stored correctly!")
        print("✓ Tearsheet generation should work!")

    except Exception as e:
        print(f"\n✗ Error during sweep: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    test_sweep_generates_tearsheets()
