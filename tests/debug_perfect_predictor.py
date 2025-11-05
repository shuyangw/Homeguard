"""
Debug script to understand Perfect Predictor strategy behavior.
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from backtesting.base.strategy import LongOnlyStrategy
from typing import Tuple


def create_sine_wave_prices(
    periods: int = 10,
    amplitude: float = 0.10,
    base_price: float = 100.0,
    bars_per_period: int = 20
) -> pd.DataFrame:
    """Create synthetic sine wave price data."""
    n_bars = periods * bars_per_period
    dates = pd.date_range(start='2023-01-01', periods=n_bars, freq='D')

    # Create sine wave
    x = np.linspace(0, periods * 2 * np.pi, n_bars)
    sine_wave = np.sin(x)

    # Scale to price amplitude
    prices = base_price + (sine_wave * base_price * amplitude)

    # Create OHLCV
    df = pd.DataFrame({
        'open': prices,
        'high': prices + 0.5,
        'low': prices - 0.5,
        'close': prices,
        'volume': np.ones(n_bars) * 1000000
    }, index=dates)

    return df


def generate_perfect_sine_signals(
    data: pd.DataFrame,
    bars_per_period: int = 20
) -> Tuple[pd.Series, pd.Series]:
    """Generate perfect entry/exit signals for sine wave data."""
    close = data['close']

    # Find local minima and maxima using centered rolling window
    # Note: Uses center=True (lookahead) but acceptable for test validation
    window = 5
    rolling_min = close.rolling(window=window, center=True).min()
    rolling_max = close.rolling(window=window, center=True).max()

    entries = (close == rolling_min).fillna(False)
    exits = (close == rolling_max).fillna(False)

    return entries, exits


class PerfectPredictorStrategy(LongOnlyStrategy):
    """Perfect predictor strategy for sine wave data."""

    def __init__(self, bars_per_period: int = 20):
        super().__init__(bars_per_period=bars_per_period)
        self.bars_per_period = bars_per_period

    def generate_long_signals(self, data: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        return generate_perfect_sine_signals(data, self.bars_per_period)


# Create test data
print("Creating sine wave data...")
data = create_sine_wave_prices(periods=5, amplitude=0.10, base_price=100.0, bars_per_period=20)

print(f"Data shape: {data.shape}")
print(f"Price range: ${data['close'].min():.2f} - ${data['close'].max():.2f}")
print(f"Expected valleys (at $90): ~5")
print(f"Expected peaks (at $110): ~5")
print()

# Generate signals
strategy = PerfectPredictorStrategy(bars_per_period=20)
entries, exits = strategy.generate_signals(data)

# Count signals
entry_count = entries.sum()
exit_count = exits.sum()

print(f"Entry signals: {entry_count}")
print(f"Exit signals: {exit_count}")
print(f"Expected: 5 entries, 5 exits (10 total trades)")
print()

# Show where entries happen
print("ENTRY SIGNALS (Buy at valleys):")
print("-" * 70)
entry_dates = data.index[entries]
for date in entry_dates:
    idx = data.index.get_loc(date)
    price = data.loc[date, 'close']
    prev_price = data.iloc[idx-1]['close'] if idx > 0 else price
    next_price = data.iloc[idx+1]['close'] if idx < len(data)-1 else price
    print(f"  {date.date()} | Bar {idx:3d} | Price: ${price:6.2f} | Prev: ${prev_price:6.2f} | Next: ${next_price:6.2f}")

print()
print("EXIT SIGNALS (Sell at peaks):")
print("-" * 70)
exit_dates = data.index[exits]
for date in exit_dates:
    idx = data.index.get_loc(date)
    price = data.loc[date, 'close']
    prev_price = data.iloc[idx-1]['close'] if idx > 0 else price
    next_price = data.iloc[idx+1]['close'] if idx < len(data)-1 else price
    print(f"  {date.date()} | Bar {idx:3d} | Price: ${price:6.2f} | Prev: ${prev_price:6.2f} | Next: ${next_price:6.2f}")

print()
print("DERIVATIVE ANALYSIS (first 30 bars):")
print("-" * 70)
derivative = data['close'].diff()
for i in range(min(30, len(data))):
    price = data.iloc[i]['close']
    deriv = derivative.iloc[i]
    prev_deriv = derivative.iloc[i-1] if i > 0 else np.nan
    entry = entries.iloc[i]
    exit_sig = exits.iloc[i]

    marker = ""
    if entry:
        marker = " <-- ENTRY"
    if exit_sig:
        marker = " <-- EXIT"

    print(f"  Bar {i:3d} | Price: ${price:6.2f} | Deriv: {deriv:6.2f} | Prev Deriv: {prev_deriv:6.2f}{marker}")

# Calculate actual local minima and maxima using a simple rolling window
print("\n" + "="*70)
print("ACTUAL LOCAL EXTREMA (5-bar rolling window):")
print("="*70)

window = 5
rolling_min_idx = data['close'].rolling(window=window, center=True).apply(
    lambda x: 1 if len(x) == window and x.iloc[window//2] == x.min() else 0
)
rolling_max_idx = data['close'].rolling(window=window, center=True).apply(
    lambda x: 1 if len(x) == window and x.iloc[window//2] == x.max() else 0
)

local_minima = data.index[rolling_min_idx == 1]
local_maxima = data.index[rolling_max_idx == 1]

print(f"\nLocal minima (valleys): {len(local_minima)}")
for date in local_minima:
    idx = data.index.get_loc(date)
    price = data.loc[date, 'close']
    print(f"  {date.date()} | Bar {idx:3d} | Price: ${price:6.2f}")

print(f"\nLocal maxima (peaks): {len(local_maxima)}")
for date in local_maxima:
    idx = data.index.get_loc(date)
    price = data.loc[date, 'close']
    print(f"  {date.date()} | Bar {idx:3d} | Price: ${price:6.2f}")
