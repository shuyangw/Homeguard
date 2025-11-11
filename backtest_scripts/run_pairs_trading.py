"""
Example script demonstrating pairs trading with the Homeguard framework.

This script shows how to:
1. Run a simple pairs trading backtest
2. Optimize pairs trading parameters
3. Analyze results

Usage:
    conda activate fintech
    python backtest_scripts/run_pairs_trading.py
"""

from src.backtesting.engine.backtest_engine import BacktestEngine
from src.strategies.advanced.pairs_trading import PairsTrading
from src.backtesting.optimization.grid_search import GridSearchOptimizer


def simple_pairs_backtest():
    """Run a simple pairs trading backtest."""
    print("\n" + "="*80)
    print("SIMPLE PAIRS TRADING BACKTEST")
    print("="*80 + "\n")

    # Initialize engine
    engine = BacktestEngine(
        initial_capital=100000,
        fees=0.001,  # 0.1% per trade
        slippage=0.001
    )

    # Create pairs trading strategy
    strategy = PairsTrading(
        pair_selection_window=252,
        cointegration_pvalue=0.05,
        entry_zscore=2.0,
        exit_zscore=0.5,
        stop_loss_zscore=3.5,
        zscore_window=20
    )

    # Run backtest on a correlated pair (example: tech stocks)
    # NOTE: Replace with actual symbols you want to test
    symbols = ['AAPL', 'MSFT']  # Example pair
    start_date = '2020-01-01'
    end_date = '2022-12-31'

    print(f"Testing pair: {symbols[0]} / {symbols[1]}")
    print(f"Period: {start_date} to {end_date}")
    print(f"Initial Capital: ${engine.initial_capital:,.0f}\n")

    try:
        # Run the backtest
        portfolio = engine.run(
            strategy=strategy,
            symbols=symbols,
            start_date=start_date,
            end_date=end_date
        )

        # Display results
        stats = portfolio.stats()

        print("\nBACKTEST RESULTS:")
        print("-" * 80)
        print(f"Total Return: {stats.get('Total Return [%]', 0):.2f}%")
        print(f"Sharpe Ratio: {stats.get('Sharpe Ratio', 0):.2f}")
        print(f"Max Drawdown: {stats.get('Max Drawdown [%]', 0):.2f}%")
        print(f"Win Rate: {stats.get('Win Rate [%]', 0):.2f}%")
        print(f"Total Trades: {stats.get('# Trades', 0)}")
        print(f"Final Equity: ${stats.get('Equity Final [$]', 0):,.2f}")
        print("-" * 80)

        # Show some trades if any
        if len(portfolio.trades) > 0:
            print(f"\nFirst 3 Trades:")
            for i, trade in enumerate(portfolio.trades[:3], 1):
                print(f"  {i}. {trade}")

    except Exception as e:
        print(f"Error during backtest: {e}")
        print("This might be due to:")
        print("  - Data not available for these symbols")
        print("  - Symbols not cointegrated")
        print("  - Insufficient data for the period")


def optimize_pairs_parameters():
    """Optimize pairs trading parameters."""
    print("\n" + "="*80)
    print("PAIRS TRADING PARAMETER OPTIMIZATION")
    print("="*80 + "\n")

    # Initialize engine and optimizer
    engine = BacktestEngine(
        initial_capital=100000,
        fees=0.001,
        slippage=0.001
    )

    optimizer = GridSearchOptimizer(engine)

    # Define parameter grid to search
    param_grid = {
        'entry_zscore': [1.5, 2.0, 2.5],
        'exit_zscore': [0.25, 0.5, 0.75],
        'zscore_window': [15, 20, 30]
    }

    symbols = ['AAPL', 'MSFT']  # Example pair
    start_date = '2020-01-01'
    end_date = '2022-12-31'

    print(f"Optimizing pair: {symbols[0]} / {symbols[1]}")
    print(f"Parameter grid: {param_grid}")
    print(f"Total combinations: {3 * 3 * 3} = 27\n")

    try:
        # Run optimization
        result = optimizer.optimize_parallel(
            strategy_class=PairsTrading,
            param_grid=param_grid,
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            metric='sharpe_ratio',
            max_workers=4,
            export_results=True,
            use_cache=True
        )

        # Display best results
        print("\nOPTIMIZATION RESULTS:")
        print("-" * 80)
        print(f"Best Parameters: {result['best_params']}")
        print(f"Best Sharpe Ratio: {result['best_value']:.3f}")
        print(f"Total Time: {result['total_time']:.1f}s")
        print(f"Cache Hits: {result['cache_hits']}")
        print("-" * 80)

        # Show top 5 parameter combinations
        if result.get('all_results'):
            sorted_results = sorted(
                result['all_results'],
                key=lambda x: x.get('value', float('-inf')),
                reverse=True
            )

            print("\nTop 5 Parameter Combinations:")
            for i, res in enumerate(sorted_results[:5], 1):
                print(f"  {i}. Params: {res['params']}")
                print(f"     Sharpe: {res['value']:.3f}")

    except Exception as e:
        print(f"Error during optimization: {e}")


def main():
    """Main execution function."""
    print("\n" + "="*80)
    print("PAIRS TRADING FRAMEWORK DEMONSTRATION")
    print("Homeguard Backtesting System")
    print("="*80)

    # Run simple backtest
    simple_pairs_backtest()

    # Ask user if they want to run optimization
    print("\n" + "="*80)
    response = input("\nRun parameter optimization? (y/n): ").strip().lower()

    if response == 'y':
        optimize_pairs_parameters()
    else:
        print("Skipping optimization.")

    print("\n" + "="*80)
    print("DEMONSTRATION COMPLETE")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
