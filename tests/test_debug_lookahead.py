"""
Debug script to understand why some lookahead bugs aren't detected.
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from backtesting.base.strategy import LongOnlyStrategy
from typing import Tuple


class LookaheadStrategy(LongOnlyStrategy):
    """Uses .shift(-1) to peek at next bar."""

    def generate_long_signals(self, data: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        close = data['close']
        future_close = close.shift(-1)
        entries = (future_close > close).fillna(False)
        exits = (future_close < close).fillna(False)
        return entries, exits


# Create test data
dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
np.random.seed(42)
trend = np.linspace(100, 150, 100)
noise = np.random.randn(100) * 2
close_prices = trend + noise

test_data = pd.DataFrame({
    'open': close_prices - 0.2,
    'high': close_prices + np.abs(np.random.randn(100) * 0.5),
    'low': close_prices - np.abs(np.random.randn(100) * 0.5),
    'close': close_prices,
    'volume': np.random.randint(1000000, 5000000, 100)
}, index=dates)

strategy = LookaheadStrategy()

# Full data
entries_full, exits_full = strategy.generate_signals(test_data)

# Partial data
entries_partial, exits_partial = strategy.generate_signals(test_data.iloc[:51])

print("Full data entries around bar 50:")
print(entries_full.iloc[48:53])
print()

print("Partial data entries around bar 50:")
print(entries_partial.iloc[48:51])
print()

print(f"Entry at bar 50 - Full: {entries_full.iloc[50]}, Partial: {entries_partial.iloc[50]}")
print(f"Do they match? {entries_full.iloc[50] == entries_partial.iloc[50]}")
print()

# Check close values
print("Close values around bar 50:")
print(test_data['close'].iloc[48:53])
print()

# The issue: With .shift(-1), at bar 50:
# Full data: future_close[50] = close[51] → exists
# Partial data: future_close[50] = close[51] → NaN (doesn't exist) → fillna(False)

# So if close[51] > close[50], entry would be True in full, False in partial
# But if fillna catches it, both become False

print("Future close (full data):")
future_full = test_data['close'].shift(-1)
print(future_full.iloc[48:53])
print()

print("Future close (partial data):")
future_partial = test_data.iloc[:51]['close'].shift(-1)
print(future_partial.iloc[48:51])
print()

# Check the comparison
print("Comparison (future > current) full:")
print((future_full > test_data['close']).iloc[48:53])
print()

print("Comparison (future > current) partial:")
print((future_partial > test_data.iloc[:51]['close']).iloc[48:51])
