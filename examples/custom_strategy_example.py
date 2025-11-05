"""
Custom strategy development example.

This example demonstrates:
1. Creating a custom strategy
2. Implementing custom indicators
3. Testing the custom strategy
"""

import pandas as pd
import numpy as np
from typing import Tuple

from backtesting.base.strategy import LongOnlyStrategy
from backtesting.utils.indicators import Indicators
from backtesting.utils.validation import validate_positive_int, validate_range
from backtesting.engine.backtest_engine import BacktestEngine


class VolatilityBreakout(LongOnlyStrategy):
    """
    Custom volatility breakout strategy.

    Entry: Price breaks above upper band (SMA + ATR)
    Exit: Price falls below lower band (SMA - ATR)
    """

    def __init__(self, sma_window: int = 20, atr_window: int = 14, atr_multiplier: float = 2.0):
        """
        Initialize strategy.

        Args:
            sma_window: SMA period for center line
            atr_window: ATR period for volatility
            atr_multiplier: ATR multiplier for bands
        """
        super().__init__(
            sma_window=sma_window,
            atr_window=atr_window,
            atr_multiplier=atr_multiplier
        )

    def validate_parameters(self) -> None:
        """
        Validate strategy parameters.
        """
        validate_positive_int(self.params['sma_window'], 'sma_window')
        validate_positive_int(self.params['atr_window'], 'atr_window')
        validate_range(self.params['atr_multiplier'], 'atr_multiplier', 0.1, 10.0)

    def generate_long_signals(self, data: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """
        Generate long entry and exit signals.
        """
        close = data['close']
        high = data['high']
        low = data['low']

        sma = Indicators.sma(close, self.params['sma_window'])
        atr = Indicators.atr(high, low, close, self.params['atr_window'])

        upper_band = sma + (atr * self.params['atr_multiplier'])
        lower_band = sma - (atr * self.params['atr_multiplier'])

        entries = (close > upper_band) & (close.shift(1) <= upper_band.shift(1))
        exits = (close < lower_band) & (close.shift(1) >= lower_band.shift(1))

        return entries, exits


def main():
    print("=" * 80)
    print("Custom Strategy Development Example")
    print("=" * 80)

    # Step 1: Create custom strategy
    print("\n1. Creating custom Volatility Breakout strategy...")
    strategy = VolatilityBreakout(
        sma_window=20,
        atr_window=14,
        atr_multiplier=2.0
    )
    print(f"   Strategy: {strategy}")

    # Step 2: Initialize engine
    print("\n2. Initializing backtest engine...")
    engine = BacktestEngine(
        initial_capital=100000,
        fees=0.001
    )

    # Step 3: Run backtest
    print("\n3. Running backtest...")
    portfolio = engine.run(
        strategy=strategy,
        symbols=['AAPL'],
        start_date='2023-01-01',
        end_date='2024-01-01'
    )

    # Step 4: Analyze results
    print("\n4. Analyzing results...")
    stats = portfolio.stats()

    print("\n" + "=" * 80)
    print("CUSTOM STRATEGY RESULTS")
    print("=" * 80)
    print(f"Total Return:       {stats['Total Return [%]']:.2f}%")
    print(f"Sharpe Ratio:       {stats['Sharpe Ratio']:.2f}")
    print(f"Max Drawdown:       {stats['Max Drawdown [%]']:.2f}%")
    print(f"Win Rate:           {stats['Win Rate [%]']:.2f}%")
    print(f"Total Trades:       {stats['Total Trades']}")

    # Step 5: Optimize parameters
    print("\n5. Optimizing parameters...")
    param_grid = {
        'sma_window': [15, 20, 25],
        'atr_multiplier': [1.5, 2.0, 2.5]
    }

    results = engine.optimize(
        strategy_class=VolatilityBreakout,
        param_grid=param_grid,
        symbols=['AAPL'],
        start_date='2022-01-01',
        end_date='2023-01-01',
        metric='sharpe_ratio'
    )

    print(f"\n   Best parameters: {results['best_params']}")
    print(f"   Best Sharpe Ratio: {results['best_value']:.4f}")

    # Step 6: Test optimized strategy
    print("\n6. Testing optimized strategy on validation period...")
    optimized_strategy = VolatilityBreakout(**results['best_params'])

    validation_portfolio = engine.run(
        strategy=optimized_strategy,
        symbols=['AAPL'],
        start_date='2023-01-01',
        end_date='2024-01-01'
    )

    validation_stats = validation_portfolio.stats()

    print("\n" + "=" * 80)
    print("OPTIMIZED STRATEGY RESULTS (Validation Period)")
    print("=" * 80)
    print(f"Total Return:       {validation_stats['Total Return [%]']:.2f}%")
    print(f"Sharpe Ratio:       {validation_stats['Sharpe Ratio']:.2f}")
    print(f"Max Drawdown:       {validation_stats['Max Drawdown [%]']:.2f}%")
    print(f"Win Rate:           {validation_stats['Win Rate [%]']:.2f}%")

    print("\n" + "=" * 80)
    print("Custom strategy development complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
