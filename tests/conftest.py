"""
Pytest fixtures for backtesting unit tests.
"""

import sys
from pathlib import Path
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


@pytest.fixture
def simple_price_data():
    """
    Generate simple trending price data for testing.

    Returns 100 days of data with a clear uptrend.
    """
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    np.random.seed(42)

    base_price = 100.0
    trend = np.linspace(0, 20, 100)
    noise = np.random.randn(100) * 2
    close_prices = base_price + trend + noise

    high_prices = close_prices + np.abs(np.random.randn(100) * 0.5)
    low_prices = close_prices - np.abs(np.random.randn(100) * 0.5)
    open_prices = close_prices + np.random.randn(100) * 0.3
    volume = np.random.randint(1000000, 5000000, 100)

    df = pd.DataFrame({
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volume
    }, index=dates)

    return df


@pytest.fixture
def oscillating_price_data():
    """
    Generate oscillating price data (good for mean reversion testing).

    Returns 100 days of data oscillating around 100.
    """
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    np.random.seed(42)

    base_price = 100.0
    oscillation = 10 * np.sin(np.linspace(0, 4 * np.pi, 100))
    noise = np.random.randn(100) * 1
    close_prices = base_price + oscillation + noise

    high_prices = close_prices + np.abs(np.random.randn(100) * 0.5)
    low_prices = close_prices - np.abs(np.random.randn(100) * 0.5)
    open_prices = close_prices + np.random.randn(100) * 0.3
    volume = np.random.randint(1000000, 5000000, 100)

    df = pd.DataFrame({
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volume
    }, index=dates)

    return df


@pytest.fixture
def multi_symbol_data():
    """
    Generate multi-symbol price data for portfolio testing.

    Returns MultiIndex DataFrame with 3 symbols.
    """
    dates = pd.date_range(start='2023-01-01', periods=50, freq='D')
    np.random.seed(42)

    symbols = ['AAPL', 'MSFT', 'GOOGL']
    dfs = []

    for i, symbol in enumerate(symbols):
        base_price = 100.0 + (i * 20)
        trend = np.linspace(0, 10, 50)
        noise = np.random.randn(50) * 2
        close_prices = base_price + trend + noise

        high_prices = close_prices + np.abs(np.random.randn(50) * 0.5)
        low_prices = close_prices - np.abs(np.random.randn(50) * 0.5)
        open_prices = close_prices + np.random.randn(50) * 0.3
        volume = np.random.randint(1000000, 5000000, 50)

        df = pd.DataFrame({
            'open': open_prices,
            'high': high_prices,
            'low': low_prices,
            'close': close_prices,
            'volume': volume,
            'symbol': symbol
        }, index=dates)

        dfs.append(df)

    combined = pd.concat(dfs)
    combined = combined.set_index('symbol', append=True)
    combined = combined.swaplevel()
    combined = combined.sort_index()

    return combined


@pytest.fixture
def flat_price_data():
    """
    Generate flat price data (no trend, for testing no-signal strategies).

    Returns 50 days of data with minimal movement.
    """
    dates = pd.date_range(start='2023-01-01', periods=50, freq='D')
    np.random.seed(42)

    base_price = 100.0
    noise = np.random.randn(50) * 0.1
    close_prices = base_price + noise

    high_prices = close_prices + np.abs(np.random.randn(50) * 0.05)
    low_prices = close_prices - np.abs(np.random.randn(50) * 0.05)
    open_prices = close_prices + np.random.randn(50) * 0.05
    volume = np.random.randint(1000000, 5000000, 50)

    df = pd.DataFrame({
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volume
    }, index=dates)

    return df
