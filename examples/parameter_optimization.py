"""
Parameter optimization example.

This example demonstrates:
1. Defining a parameter grid
2. Running optimization
3. Comparing optimal vs default parameters
"""

from backtesting.engine.backtest_engine import BacktestEngine
from strategies.base_strategies.moving_average import MovingAverageCrossover


def main():
    print("=" * 80)
    print("Parameter Optimization Example")
    print("=" * 80)

    # Step 1: Define parameter grid
    print("\n1. Defining parameter grid...")
    param_grid = {
        'fast_window': [10, 15, 20, 25, 30],
        'slow_window': [40, 50, 60, 70]
    }
    print(f"   Grid: {param_grid}")
    total_combinations = len(param_grid['fast_window']) * len(param_grid['slow_window'])
    print(f"   Total combinations to test: {total_combinations}")

    # Step 2: Initialize engine
    print("\n2. Initializing backtest engine...")
    engine = BacktestEngine(
        initial_capital=100000,
        fees=0.001
    )

    # Step 3: Run optimization
    print("\n3. Running optimization (this may take a while)...")
    results = engine.optimize(
        strategy_class=MovingAverageCrossover,
        param_grid=param_grid,
        symbols=['AAPL'],
        start_date='2022-01-01',
        end_date='2023-01-01',
        metric='sharpe_ratio'
    )

    # Step 4: Display results
    print("\n4. Optimization Results:")
    print("-" * 80)
    print(f"   Best parameters: {results['best_params']}")
    print(f"   Best Sharpe Ratio: {results['best_value']:.4f}")

    # Step 5: Test optimal parameters on validation period
    print("\n5. Testing optimal parameters on validation period (2023)...")
    optimal_strategy = MovingAverageCrossover(**results['best_params'])

    validation_portfolio = engine.run(
        strategy=optimal_strategy,
        symbols=['AAPL'],
        start_date='2023-01-01',
        end_date='2024-01-01'
    )

    # Step 6: Compare with default parameters
    print("\n6. Comparing with default parameters...")
    default_strategy = MovingAverageCrossover(fast_window=20, slow_window=50)

    default_portfolio = engine.run(
        strategy=default_strategy,
        symbols=['AAPL'],
        start_date='2023-01-01',
        end_date='2024-01-01'
    )

    # Step 7: Comparison
    print("\n7. Performance Comparison (Validation Period):")
    print("-" * 80)
    print(f"{'Metric':<25} {'Optimal':<15} {'Default':<15}")
    print("-" * 80)

    optimal_stats = validation_portfolio.stats()
    default_stats = default_portfolio.stats()

    metrics = ['Total Return [%]', 'Sharpe Ratio', 'Max Drawdown [%]', 'Win Rate [%]']
    for metric in metrics:
        print(f"{metric:<25} {optimal_stats[metric]:<15.2f} {default_stats[metric]:<15.2f}")

    print("\n" + "=" * 80)
    print("Optimization complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
