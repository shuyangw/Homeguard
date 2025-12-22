"""
Pytest fixtures for screening module tests.
"""

from datetime import datetime, timedelta
from typing import Dict, List
from unittest.mock import MagicMock, Mock

import numpy as np
import pandas as pd
import pytest

from src.screening.alpaca_client import SnapshotData
from src.screening.filters import (
    FundamentalFilter,
    PriceFilter,
    ScreenerConfig,
    TechnicalFilter,
    VolumeFilter,
)


@pytest.fixture
def sample_symbols() -> List[str]:
    """Sample list of stock symbols."""
    return ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "JPM", "V", "WMT"]


@pytest.fixture
def sample_snapshot_data() -> Dict[str, SnapshotData]:
    """Sample snapshot data for multiple symbols."""
    now = datetime.now()
    return {
        "AAPL": SnapshotData(
            symbol="AAPL",
            latest_trade_price=175.50,
            latest_trade_size=100,
            latest_trade_time=now,
            bid=175.48,
            ask=175.52,
            daily_bar_open=174.00,
            daily_bar_high=177.00,
            daily_bar_low=173.50,
            daily_bar_close=175.50,
            daily_bar_volume=50_000_000,
            prev_daily_close=172.00,
            minute_bar_close=175.45,
            minute_bar_volume=100_000,
        ),
        "MSFT": SnapshotData(
            symbol="MSFT",
            latest_trade_price=380.25,
            latest_trade_size=50,
            latest_trade_time=now,
            bid=380.20,
            ask=380.30,
            daily_bar_open=378.00,
            daily_bar_high=382.00,
            daily_bar_low=377.00,
            daily_bar_close=380.25,
            daily_bar_volume=25_000_000,
            prev_daily_close=375.00,
            minute_bar_close=380.20,
            minute_bar_volume=50_000,
        ),
        "PENNY": SnapshotData(
            symbol="PENNY",
            latest_trade_price=2.50,
            latest_trade_size=1000,
            latest_trade_time=now,
            bid=2.48,
            ask=2.52,
            daily_bar_open=2.40,
            daily_bar_high=2.60,
            daily_bar_low=2.35,
            daily_bar_close=2.50,
            daily_bar_volume=100_000,
            prev_daily_close=2.45,
            minute_bar_close=2.50,
            minute_bar_volume=10_000,
        ),
        "LOWVOL": SnapshotData(
            symbol="LOWVOL",
            latest_trade_price=50.00,
            latest_trade_size=10,
            latest_trade_time=now,
            bid=49.95,
            ask=50.05,
            daily_bar_open=49.50,
            daily_bar_high=50.50,
            daily_bar_low=49.00,
            daily_bar_close=50.00,
            daily_bar_volume=10_000,
            prev_daily_close=49.00,
            minute_bar_close=50.00,
            minute_bar_volume=100,
        ),
    }


@pytest.fixture
def sample_historical_bars() -> pd.DataFrame:
    """Sample historical OHLCV DataFrame for testing indicators."""
    np.random.seed(42)
    n_days = 60
    dates = pd.date_range(end=datetime.now(), periods=n_days, freq="D")

    # Generate trending price data
    base_price = 100.0
    returns = np.random.normal(0.001, 0.02, n_days)
    prices = base_price * np.cumprod(1 + returns)

    # Add some noise for OHLC
    opens = prices * (1 + np.random.uniform(-0.01, 0.01, n_days))
    highs = np.maximum(prices, opens) * (1 + np.random.uniform(0, 0.02, n_days))
    lows = np.minimum(prices, opens) * (1 - np.random.uniform(0, 0.02, n_days))
    closes = prices

    return pd.DataFrame(
        {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": np.random.randint(1_000_000, 10_000_000, n_days),
        },
        index=dates,
    )


@pytest.fixture
def oversold_historical_bars() -> pd.DataFrame:
    """Historical data with RSI in oversold territory."""
    np.random.seed(123)
    n_days = 60
    dates = pd.date_range(end=datetime.now(), periods=n_days, freq="D")

    # Generate declining price data for oversold RSI
    base_price = 150.0
    prices = base_price * np.cumprod(1 + np.random.normal(-0.015, 0.01, n_days))

    return pd.DataFrame(
        {
            "open": prices * 1.005,
            "high": prices * 1.01,
            "low": prices * 0.99,
            "close": prices,
            "volume": np.random.randint(1_000_000, 5_000_000, n_days),
        },
        index=dates,
    )


@pytest.fixture
def mock_alpaca_client(sample_snapshot_data, sample_historical_bars):
    """Mocked Alpaca screener client."""
    client = MagicMock()
    client.get_all_tradable_assets.return_value = list(sample_snapshot_data.keys())
    client.get_snapshots.return_value = sample_snapshot_data
    client.get_historical_bars.return_value = {
        symbol: sample_historical_bars.copy() for symbol in sample_snapshot_data.keys()
    }
    return client


@pytest.fixture
def basic_price_filter() -> PriceFilter:
    """Basic price filter: $10-500."""
    return PriceFilter(min_price=10, max_price=500)


@pytest.fixture
def volume_filter() -> VolumeFilter:
    """Volume filter: min 1M shares."""
    return VolumeFilter(min_volume=1_000_000)


@pytest.fixture
def technical_filter_oversold() -> TechnicalFilter:
    """Technical filter for oversold stocks."""
    return TechnicalFilter(rsi_14=("lt", 30))


@pytest.fixture
def basic_config(basic_price_filter) -> ScreenerConfig:
    """Basic screener configuration."""
    return ScreenerConfig(
        universe=["AAPL", "MSFT", "PENNY", "LOWVOL"],
        price=basic_price_filter,
        max_results=10,
    )
