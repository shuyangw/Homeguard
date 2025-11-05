"""
Multi-strategy comparison example.

This example demonstrates:
1. Testing multiple strategies on the same data
2. Comparing performance metrics
3. Visualizing results
"""

from backtesting.engine.backtest_engine import BacktestEngine
from backtesting.engine.metrics import PerformanceMetrics
from strategies.base_strategies.moving_average import MovingAverageCrossover
from strategies.base_strategies.mean_reversion import MeanReversion, RSIMeanReversion
from strategies.base_strategies.momentum import MomentumStrategy


def main():
    print("=" * 80)
    print("Multi-Strategy Comparison Example")
    print("=" * 80)

    # Step 1: Define strategies to compare
    print("\n1. Defining strategies...")
    strategies = {
        'MA Crossover': MovingAverageCrossover(fast_window=20, slow_window=50),
        'Bollinger Bands': MeanReversion(window=20, num_std=2.0),
        'RSI Mean Reversion': RSIMeanReversion(rsi_window=14, oversold=30, overbought=70),
        'MACD Momentum': MomentumStrategy(fast=12, slow=26, signal=9)
    }

    for name, strategy in strategies.items():
        print(f"   - {name}: {strategy}")

    # Step 2: Initialize engine
    print("\n2. Initializing backtest engine...")
    engine = BacktestEngine(
        initial_capital=100000,
        fees=0.001
    )

    # Step 3: Run backtests
    print("\n3. Running backtests...")
    portfolios = {}

    for name, strategy in strategies.items():
        print(f"\n   Testing {name}...")
        portfolio = engine.run(
            strategy=strategy,
            symbols=['AAPL'],
            start_date='2023-01-01',
            end_date='2024-01-01'
        )
        portfolios[name] = portfolio

    # Step 4: Compare strategies
    print("\n4. Comparing strategies...")
    comparison = PerformanceMetrics.compare_strategies(portfolios)

    print("\n" + "=" * 80)
    print("STRATEGY COMPARISON")
    print("=" * 80)
    print(comparison.to_string())

    # Step 5: Identify best strategy
    print("\n5. Best Performing Strategies:")
    print("-" * 80)

    best_sharpe = comparison['sharpe_ratio'].idxmax()
    best_return = comparison['total_return_pct'].idxmax()
    best_drawdown = comparison['max_drawdown_pct'].idxmin()

    print(f"   Best Sharpe Ratio:    {best_sharpe} ({comparison.loc[best_sharpe, 'sharpe_ratio']:.2f})")
    print(f"   Best Total Return:    {best_return} ({comparison.loc[best_return, 'total_return_pct']:.2f}%)")
    print(f"   Best Max Drawdown:    {best_drawdown} ({comparison.loc[best_drawdown, 'max_drawdown_pct']:.2f}%)")

    # Step 6: Save detailed report
    print("\n6. Saving detailed report...")
    output_path = 'strategy_comparison_report.csv'
    comparison.to_csv(output_path)
    print(f"   Report saved to: {output_path}")

    # Step 7: Optional - Plot equity curves
    print("\n7. Plotting equity curves...")
    print("   (Close plot windows to continue)")

    for name, portfolio in portfolios.items():
        try:
            PerformanceMetrics.plot_equity_curve(portfolio, title=f"{name} - Equity Curve")
        except Exception as e:
            print(f"   Could not plot {name}: {e}")

    print("\n" + "=" * 80)
    print("Strategy comparison complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
