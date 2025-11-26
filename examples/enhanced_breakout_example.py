"""
Example: Enhanced Breakout Strategy with Volatility Filter

This demonstrates the enhanced BreakoutStrategy with:
- Volatility filter to avoid choppy markets
- Volume confirmation for stronger signals
- ATR-based stop loss

Strategy: Donchian Channel breakout with filters
Symbol: AMZN
Period: 2023-2024
"""

from backtesting.engine.backtest_engine import BacktestEngine
from strategies import BreakoutStrategy
from src.settings import get_output_dir


def main():
    # Initialize backtest engine
    engine = BacktestEngine(
        initial_capital=100000,
        fees=0.001
    )

    # Create enhanced breakout strategy with filters
    strategy = BreakoutStrategy(
        breakout_window=20,
        exit_window=10,
        volatility_filter=True,          # Enable volatility filter
        volatility_window=20,
        min_volatility=0.01,              # Min 1% annualized volatility
        max_volatility=0.10,              # Max 10% annualized volatility
        volume_confirmation=True,         # Require volume spike
        volume_threshold=1.5,             # 1.5x average volume
        use_atr_stop=True,                # Use ATR trailing stop
        atr_multiplier=2.0                # 2x ATR for stop
    )

    # Run backtest
    output_dir = str(get_output_dir() / 'examples' / 'enhanced_breakout_example')
    portfolio = engine.run_and_report(
        strategy=strategy,
        symbols=['AMZN'],
        start_date='2023-01-01',
        end_date='2024-01-01',
        quantstats=True,
        output_dir=output_dir
    )

    print("\nBacktest complete!")
    print(f"See QuantStats report in: {output_dir}")


if __name__ == '__main__':
    main()
