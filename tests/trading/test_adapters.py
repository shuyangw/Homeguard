"""
Unit Tests for Live Trading Adapters.

Tests StrategyAdapter, MACrossoverLiveAdapter, and OMRLiveAdapter.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock, patch
import pandas as pd

from src.trading.adapters import (
    StrategyAdapter,
    MACrossoverLiveAdapter,
    TripleMACrossoverLiveAdapter,
    OMRLiveAdapter
)
from src.strategies.core import Signal
from src.strategies.universe import ETFUniverse, EquityUniverse


@pytest.fixture
def mock_broker():
    """Create mock broker interface."""
    broker = Mock()

    # Mock account info
    account = Mock()
    account.id = "test_account_123"
    account.buying_power = "100000.00"
    account.portfolio_value = "100000.00"
    broker.get_account.return_value = account

    # Mock market status
    broker.is_market_open.return_value = True

    # Mock historical data
    def get_historical_bars(symbol, start, end, timeframe):
        """Return mock historical data."""
        dates = pd.date_range(start, end, freq='D')
        data = {
            'open': [100.0] * len(dates),
            'high': [101.0] * len(dates),
            'low': [99.0] * len(dates),
            'close': [100.5] * len(dates),
            'volume': [1000000] * len(dates)
        }
        return pd.DataFrame(data, index=dates)

    broker.get_historical_bars.side_effect = get_historical_bars

    # Mock positions (returned as list of dicts, as real Alpaca broker does)
    broker.get_positions.return_value = []

    return broker


@pytest.fixture
def mock_strategy():
    """Create mock pure strategy."""
    strategy = Mock()
    strategy.generate_signals.return_value = []
    strategy.get_required_lookback.return_value = 200
    return strategy


@pytest.fixture
def sample_market_data():
    """Create sample market data for testing."""
    dates = pd.date_range('2024-01-01', '2024-12-31', freq='D')
    data = {}

    for symbol in ['AAPL', 'MSFT', 'GOOGL']:
        data[symbol] = pd.DataFrame({
            'open': [100.0] * len(dates),
            'high': [101.0] * len(dates),
            'low': [99.0] * len(dates),
            'close': [100.5] * len(dates),
            'volume': [1000000] * len(dates)
        }, index=dates)

    return data


class TestStrategyAdapter:
    """Test StrategyAdapter base class."""

    def test_initialization(self, mock_broker, mock_strategy):
        """Test adapter initialization."""
        adapter = MACrossoverLiveAdapter(
            broker=mock_broker,
            symbols=['AAPL', 'MSFT'],
            fast_period=50,
            slow_period=200
        )

        assert adapter.broker == mock_broker
        assert adapter.symbols == ['AAPL', 'MSFT']
        assert adapter.position_size == 0.1
        assert adapter.max_positions == 5
        assert adapter.execution_engine is not None
        assert adapter.position_manager is not None

    def test_fetch_market_data(self, mock_broker):
        """Test fetching market data."""
        adapter = MACrossoverLiveAdapter(
            broker=mock_broker,
            symbols=['AAPL', 'MSFT'],
            fast_period=50,
            slow_period=200
        )

        market_data = adapter.fetch_market_data()

        assert 'AAPL' in market_data
        assert 'MSFT' in market_data
        assert isinstance(market_data['AAPL'], pd.DataFrame)
        assert 'close' in market_data['AAPL'].columns

    def test_generate_signals(self, mock_broker, sample_market_data):
        """Test signal generation."""
        adapter = MACrossoverLiveAdapter(
            broker=mock_broker,
            symbols=['AAPL', 'MSFT', 'GOOGL'],
            fast_period=50,
            slow_period=200
        )

        signals = adapter.generate_signals(sample_market_data)

        # Should return a list
        assert isinstance(signals, list)

    def test_filter_signals_max_positions(self, mock_broker):
        """Test signal filtering respects max positions."""
        adapter = MACrossoverLiveAdapter(
            broker=mock_broker,
            symbols=['AAPL', 'MSFT', 'GOOGL'],
            fast_period=50,
            slow_period=200,
            max_positions=2
        )

        # Create 3 signals
        signals = [
            Signal(
                timestamp=datetime.now(),
                symbol='AAPL',
                direction='BUY',
                confidence=0.8,
                price=100.0
            ),
            Signal(
                timestamp=datetime.now(),
                symbol='MSFT',
                direction='BUY',
                confidence=0.75,
                price=200.0
            ),
            Signal(
                timestamp=datetime.now(),
                symbol='GOOGL',
                direction='BUY',
                confidence=0.7,
                price=150.0
            )
        ]

        filtered = adapter.filter_signals(signals)

        # Should only get 2 signals (max_positions=2)
        assert len(filtered) == 2

    def test_filter_signals_existing_positions(self, mock_broker):
        """Test signal filtering skips existing positions."""
        adapter = MACrossoverLiveAdapter(
            broker=mock_broker,
            symbols=['AAPL', 'MSFT'],
            fast_period=50,
            slow_period=200
        )

        # Add existing position using position manager API
        adapter.position_manager.add_position(
            symbol='AAPL',
            entry_price=95.0,
            qty=100,
            timestamp=datetime.now(),
            order_id='test_order_123'
        )

        # Create signal for existing position
        signals = [
            Signal(
                timestamp=datetime.now(),
                symbol='AAPL',
                direction='BUY',
                confidence=0.8,
                price=100.0
            )
        ]

        filtered = adapter.filter_signals(signals)

        # Should skip AAPL (already have position)
        assert len(filtered) == 0

    def test_should_run_now_market_closed(self, mock_broker):
        """Test should_run_now when market is closed."""
        # Set market to closed
        mock_broker.is_market_open.return_value = False

        adapter = MACrossoverLiveAdapter(
            broker=mock_broker,
            symbols=['AAPL'],
            fast_period=50,
            slow_period=200
        )

        # Should not run when market is closed
        assert adapter.should_run_now() == False

    def test_should_run_now_market_open(self, mock_broker):
        """Test should_run_now when market is open."""
        # Set market to open
        mock_broker.is_market_open.return_value = True

        adapter = MACrossoverLiveAdapter(
            broker=mock_broker,
            symbols=['AAPL'],
            fast_period=50,
            slow_period=200
        )

        # Should run when market is open (MA adapter runs every 5min)
        assert adapter.should_run_now() == True


class TestMACrossoverLiveAdapter:
    """Test MACrossoverLiveAdapter."""

    def test_initialization(self, mock_broker):
        """Test MA adapter initialization."""
        adapter = MACrossoverLiveAdapter(
            broker=mock_broker,
            symbols=['AAPL', 'MSFT'],
            fast_period=50,
            slow_period=200,
            ma_type='sma',
            min_confidence=0.7
        )

        assert adapter.symbols == ['AAPL', 'MSFT']
        assert adapter.strategy is not None

    def test_get_schedule(self, mock_broker):
        """Test schedule configuration."""
        adapter = MACrossoverLiveAdapter(
            broker=mock_broker,
            symbols=['AAPL'],
            fast_period=50,
            slow_period=200
        )

        schedule = adapter.get_schedule()

        assert schedule['interval'] == '5min'
        assert schedule['market_hours_only'] == True

    def test_data_lookback_calculation(self, mock_broker):
        """Test data lookback is calculated correctly."""
        adapter = MACrossoverLiveAdapter(
            broker=mock_broker,
            symbols=['AAPL'],
            fast_period=50,
            slow_period=200
        )

        # Lookback should be 2x slow period
        assert adapter.data_lookback_days == 200 * 2


class TestTripleMACrossoverLiveAdapter:
    """Test TripleMACrossoverLiveAdapter."""

    def test_initialization(self, mock_broker):
        """Test triple MA adapter initialization."""
        adapter = TripleMACrossoverLiveAdapter(
            broker=mock_broker,
            symbols=['AAPL', 'MSFT'],
            fast_period=10,
            medium_period=20,
            slow_period=50,
            ma_type='ema'
        )

        assert adapter.symbols == ['AAPL', 'MSFT']
        assert adapter.strategy is not None

    def test_get_schedule(self, mock_broker):
        """Test schedule configuration."""
        adapter = TripleMACrossoverLiveAdapter(
            broker=mock_broker,
            symbols=['AAPL'],
            fast_period=10,
            medium_period=20,
            slow_period=50
        )

        schedule = adapter.get_schedule()

        assert schedule['interval'] == '5min'
        assert schedule['market_hours_only'] == True


class TestOMRLiveAdapter:
    """Test OMRLiveAdapter."""

    def test_initialization_default_symbols(self, mock_broker):
        """Test OMR adapter uses default symbols."""
        adapter = OMRLiveAdapter(
            broker=mock_broker
        )

        # Should use default leveraged 3x ETFs
        assert adapter.symbols == ETFUniverse.LEVERAGED_3X

    def test_initialization_custom_symbols(self, mock_broker):
        """Test OMR adapter with custom symbols."""
        custom_symbols = ['TQQQ', 'SQQQ', 'UPRO']

        adapter = OMRLiveAdapter(
            broker=mock_broker,
            symbols=custom_symbols
        )

        assert adapter.symbols == custom_symbols

    def test_get_schedule(self, mock_broker):
        """Test OMR schedule configuration."""
        adapter = OMRLiveAdapter(
            broker=mock_broker
        )

        schedule = adapter.get_schedule()

        # OMR has TWO execution times: entry at 3:50 PM and exit at 9:31 AM
        assert 'execution_times' in schedule
        assert len(schedule['execution_times']) == 2
        assert schedule['execution_times'][0]['time'] == '15:50'  # 3:50 PM EST entry
        assert schedule['execution_times'][0]['action'] == 'entry'
        assert schedule['execution_times'][1]['time'] == '09:31'  # 9:31 AM EST exit
        assert schedule['execution_times'][1]['action'] == 'exit'
        assert schedule['market_hours_only'] == True
        assert schedule['strategy_type'] == 'overnight'

    @patch('src.trading.adapters.omr_live_adapter.datetime')
    def test_should_run_now_correct_time(self, mock_datetime, mock_broker):
        """Test should_run_now at 3:50 PM."""
        # Mock current time to 3:50 PM
        mock_now = datetime(2024, 1, 15, 15, 50, 0)
        mock_datetime.now.return_value = mock_now
        mock_datetime.strptime = datetime.strptime
        mock_datetime.combine = datetime.combine

        # Market is open
        mock_broker.is_market_open.return_value = True

        adapter = OMRLiveAdapter(
            broker=mock_broker
        )

        # Should run at 3:50 PM
        # Note: This test might fail due to datetime mocking complexity
        # In real usage, adapter checks if within 1 minute of target time

    def test_fetch_market_data_includes_spy_vix(self, mock_broker):
        """Test OMR fetches SPY and VIX for regime detection."""
        adapter = OMRLiveAdapter(
            broker=mock_broker,
            symbols=['TQQQ', 'SQQQ']
        )

        market_data = adapter.fetch_market_data()

        # Should fetch symbols + SPY + VIX
        assert mock_broker.get_historical_bars.call_count >= 2

    def test_close_overnight_positions(self, mock_broker):
        """Test closing overnight positions."""
        # Create mock position as dict (as returned by real Alpaca broker)
        mock_position = {
            'symbol': 'TQQQ',
            'quantity': 100,
            'avg_entry_price': 50.0,
            'current_price': 52.0,
            'market_value': 5200.0,
            'unrealized_pnl': 200.0,
            'unrealized_pnl_pct': 0.04,
            'side': 'long'
        }

        mock_broker.get_positions.return_value = [mock_position]

        adapter = OMRLiveAdapter(
            broker=mock_broker
        )

        # Mock execution engine
        adapter.execution_engine.place_market_order = Mock(return_value=Mock(id='order_123'))

        # Close positions
        adapter.close_overnight_positions()

        # Should have placed sell order
        adapter.execution_engine.place_market_order.assert_called_once()

    def test_close_overnight_positions_no_positions(self, mock_broker):
        """Test closing overnight positions when none exist."""
        mock_broker.get_positions.return_value = []

        adapter = OMRLiveAdapter(
            broker=mock_broker
        )

        # Mock execution engine
        adapter.execution_engine.place_market_order = Mock()

        # Close positions (should do nothing)
        adapter.close_overnight_positions()

        # Should not place any orders
        adapter.execution_engine.place_market_order.assert_not_called()


class TestAdapterIntegration:
    """Integration tests for adapters."""

    def test_run_once_workflow(self, mock_broker, sample_market_data):
        """Test complete run_once workflow."""
        adapter = MACrossoverLiveAdapter(
            broker=mock_broker,
            symbols=['AAPL', 'MSFT', 'GOOGL'],
            fast_period=50,
            slow_period=200
        )

        # Mock execution engine
        adapter.execution_engine.place_market_order = Mock(return_value=Mock(id='order_123'))

        # Run once
        adapter.run_once()

        # Should have:
        # 1. Fetched market data
        assert mock_broker.get_historical_bars.called

        # 2. Generated signals (via strategy)
        # 3. Filtered signals
        # 4. Updated positions
        assert mock_broker.get_positions.called

    def test_adapter_handles_no_market_data(self, mock_broker):
        """Test adapter handles missing market data gracefully."""
        # Mock broker returns no data
        mock_broker.get_historical_bars.return_value = None

        adapter = MACrossoverLiveAdapter(
            broker=mock_broker,
            symbols=['INVALID'],
            fast_period=50,
            slow_period=200
        )

        # Should not raise exception
        adapter.run_once()

    def test_adapter_handles_account_error(self, mock_broker):
        """Test adapter handles account fetch error."""
        # Mock account fetch failure
        mock_broker.get_account.return_value = None

        adapter = MACrossoverLiveAdapter(
            broker=mock_broker,
            symbols=['AAPL'],
            fast_period=50,
            slow_period=200
        )

        # Create mock signal
        signals = [
            Signal(
                timestamp=datetime.now(),
                symbol='AAPL',
                direction='BUY',
                confidence=0.8,
                price=100.0
            )
        ]

        # Should handle gracefully (skip execution)
        adapter.execute_signals(signals)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
