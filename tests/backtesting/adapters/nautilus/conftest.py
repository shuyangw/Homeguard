"""
Test fixtures for NautilusTrader adapter tests.

Provides common fixtures used across all adapter tests.
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List

import pandas as pd
import pytest

from tests.backtesting.adapters.nautilus.mocks import (
    MockBar,
    MockCache,
    MockPortfolio,
    MockStrategy,
    MockInstrument,
)


@pytest.fixture
def sample_symbols() -> List[str]:
    """Sample stock symbols for testing."""
    return ["AAPL", "MSFT", "GOOGL"]


@pytest.fixture
def sample_bar_data() -> Dict[str, List[Dict]]:
    """Sample bar data for multiple symbols."""
    base_date = datetime(2024, 1, 2, 9, 30)  # Market open
    data = {}

    for symbol in ["AAPL", "MSFT", "GOOGL"]:
        bars = []
        price = {"AAPL": 185.0, "MSFT": 375.0, "GOOGL": 140.0}[symbol]

        for i in range(100):
            timestamp = base_date + timedelta(minutes=i)
            bars.append({
                "symbol": symbol,
                "open": price + i * 0.1,
                "high": price + i * 0.1 + 0.5,
                "low": price + i * 0.1 - 0.3,
                "close": price + i * 0.1 + 0.2,
                "volume": 10000.0 + i * 100,
                "timestamp": timestamp,
            })
        data[symbol] = bars

    return data


@pytest.fixture
def sample_bars(sample_bar_data) -> List[MockBar]:
    """Create MockBar objects from sample data."""
    bars = []
    for symbol, bar_list in sample_bar_data.items():
        for bar_dict in bar_list:
            bars.append(MockBar.from_ohlcv(
                symbol=bar_dict["symbol"],
                open_price=bar_dict["open"],
                high=bar_dict["high"],
                low=bar_dict["low"],
                close=bar_dict["close"],
                volume=bar_dict["volume"],
                timestamp=bar_dict["timestamp"],
            ))
    return bars


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """Sample OHLCV DataFrame for testing."""
    dates = pd.date_range("2024-01-02", periods=100, freq="1min", tz="UTC")
    return pd.DataFrame({
        "timestamp": dates,
        "open": [100 + i * 0.1 for i in range(100)],
        "high": [100 + i * 0.1 + 0.5 for i in range(100)],
        "low": [100 + i * 0.1 - 0.3 for i in range(100)],
        "close": [100 + i * 0.1 + 0.2 for i in range(100)],
        "volume": [10000 + i * 100 for i in range(100)],
    })


@pytest.fixture
def mock_cache(sample_symbols) -> MockCache:
    """Create MockCache with sample instruments."""
    cache = MockCache()
    for symbol in sample_symbols:
        instrument = MockInstrument.from_symbol(symbol)
        cache.add_instrument(instrument)
    return cache


@pytest.fixture
def mock_portfolio() -> MockPortfolio:
    """Create MockPortfolio with default balance."""
    return MockPortfolio(initial_balance=100000.0)


@pytest.fixture
def mock_strategy(mock_cache, mock_portfolio) -> MockStrategy:
    """Create MockStrategy with cache and portfolio."""
    strategy = MockStrategy(strategy_id="TEST")
    strategy.cache = mock_cache
    strategy.portfolio = mock_portfolio
    return strategy


@pytest.fixture
def sample_homeguard_parquet(tmp_path, sample_dataframe) -> Path:
    """
    Create temporary Hive-partitioned parquet structure.

    Creates: tmp_path/equities_1min/symbol=TEST/year=2024/month=1/data.parquet
    """
    # Create directory structure
    symbol_dir = tmp_path / "equities_1min" / "symbol=TEST" / "year=2024" / "month=1"
    symbol_dir.mkdir(parents=True, exist_ok=True)

    # Write parquet file
    parquet_path = symbol_dir / "data.parquet"
    sample_dataframe.to_parquet(parquet_path, index=False)

    return tmp_path


@pytest.fixture
def sample_signals():
    """Sample trading signals for testing."""
    from src.strategies.core.signal import Signal

    return [
        Signal(
            timestamp=datetime(2024, 1, 2, 15, 50),
            symbol="AAPL",
            direction="BUY",
            confidence=0.85,
            price=185.50,
        ),
        Signal(
            timestamp=datetime(2024, 1, 2, 15, 50),
            symbol="MSFT",
            direction="SELL",
            confidence=0.75,
            price=376.20,
        ),
        Signal(
            timestamp=datetime(2024, 1, 2, 15, 50),
            symbol="GOOGL",
            direction="HOLD",
            confidence=0.50,
            price=140.80,
        ),
    ]
