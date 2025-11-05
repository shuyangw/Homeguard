"""
Basic backtest example using the backtesting framework.

This example demonstrates:
1. Loading data
2. Creating a strategy
3. Running a backtest
4. Viewing results
"""

from backtesting.engine.backtest_engine import BacktestEngine
from backtesting.engine.data_loader import DataLoader
from strategies.base_strategies.moving_average import MovingAverageCrossover


def main():
    print("=" * 80)
    print("Basic Backtest Example")
    print("=" * 80)

    # Step 1: Create a strategy
    print("\n1. Creating Moving Average Crossover strategy...")
    strategy = MovingAverageCrossover(
        fast_window=20,
        slow_window=50,
        ma_type='sma'
    )
    print(f"   Strategy: {strategy}")

    # Step 2: Initialize backtesting engine
    print("\n2. Initializing backtest engine...")
    engine = BacktestEngine(
        initial_capital=100000,  # $100k starting capital
        fees=0.001,              # 0.1% trading fees
        slippage=0.0005          # 0.05% slippage
    )
    print("   Engine initialized")

    # Step 3: Run backtest
    print("\n3. Running backtest...")
    portfolio = engine.run(
        strategy=strategy,
        symbols=['AAPL'],
        start_date='2023-01-01',
        end_date='2024-01-01'
    )

    # Step 4: View detailed statistics
    print("\n4. Detailed Statistics:")
    print("-" * 80)
    stats = portfolio.stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")

    # Step 5: Optional - Plot results
    print("\n5. Plotting results...")
    print("   (Close the plot window to continue)")
    try:
        portfolio.plot().show()
    except Exception as e:
        print(f"   Could not display plot: {e}")

    print("\n" + "=" * 80)
    print("Backtest complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
