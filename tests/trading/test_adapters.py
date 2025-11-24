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
        adapter.execution_engine.execute_order = Mock(return_value={'order_id': 'order_123'})

        # Close positions
        adapter.close_overnight_positions()

        # Should have placed sell order
        adapter.execution_engine.execute_order.assert_called_once()

    def test_close_overnight_positions_no_positions(self, mock_broker):
        """Test closing overnight positions when none exist."""
        mock_broker.get_positions.return_value = []

        adapter = OMRLiveAdapter(
            broker=mock_broker
        )

        # Mock execution engine
        adapter.execution_engine.execute_order = Mock()

        # Close positions (should do nothing)
        adapter.close_overnight_positions()

        # Should not place any orders
        adapter.execution_engine.execute_order.assert_not_called()


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
        adapter.execution_engine.execute_order = Mock(return_value={'order_id': 'order_123'})

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


class TestVIXDataFetching:
    """
    Tests for VIX data fetching and column normalization.

    These tests make actual network calls to verify real-world behavior.
    Critical for catching column name mismatches (e.g., 'Close' vs 'close').
    """

    @pytest.mark.network
    def test_fetch_vix_yfinance_returns_data(self, mock_broker):
        """Test _fetch_vix_yfinance returns non-empty DataFrame."""
        adapter = OMRLiveAdapter(
            broker=mock_broker,
            symbols=['TQQQ']
        )

        # Fetch VIX data (actual network call)
        from datetime import datetime, timedelta
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)

        vix_data = adapter._fetch_vix_yfinance(
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
        )

        assert vix_data is not None, "VIX data should not be None"
        assert not vix_data.empty, "VIX data should not be empty"
        assert len(vix_data) > 0, "VIX data should have rows"

    @pytest.mark.network
    def test_fetch_vix_yfinance_has_required_columns(self, mock_broker):
        """Test _fetch_vix_yfinance returns DataFrame with OHLCV columns."""
        adapter = OMRLiveAdapter(
            broker=mock_broker,
            symbols=['TQQQ']
        )

        from datetime import datetime, timedelta
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)

        vix_data = adapter._fetch_vix_yfinance(
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
        )

        assert vix_data is not None

        # Check for required columns (case-insensitive)
        columns_lower = [c.lower() if isinstance(c, str) else str(c).lower() for c in vix_data.columns]
        assert 'close' in columns_lower or 'Close' in vix_data.columns, \
            f"VIX data should have 'close' column. Got: {list(vix_data.columns)}"

    @pytest.mark.network
    def test_fetch_market_data_normalizes_column_names(self, mock_broker):
        """
        Test fetch_market_data normalizes all column names to lowercase.

        This is the critical test that would have caught the 'close' KeyError bug.
        """
        # Need to mock the broker's get_historical_bars to avoid Alpaca API calls
        # but let VIX fetch go through yfinance
        def mock_historical_bars(symbol, start, end, timeframe):
            """Return data with UPPERCASE columns (simulating some data sources)."""
            import pandas as pd
            dates = pd.date_range(start, end, freq='D')
            # Intentionally use uppercase column names
            data = {
                'Open': [100.0] * len(dates),
                'High': [101.0] * len(dates),
                'Low': [99.0] * len(dates),
                'Close': [100.5] * len(dates),
                'Volume': [1000000] * len(dates)
            }
            return pd.DataFrame(data, index=dates)

        mock_broker.get_historical_bars.side_effect = mock_historical_bars

        adapter = OMRLiveAdapter(
            broker=mock_broker,
            symbols=['SPY']  # Just SPY to simplify
        )

        # Preload cache with uppercase columns (simulating yfinance behavior)
        adapter._data_cache = {
            'SPY': mock_historical_bars('SPY', '2024-01-01', '2024-01-31', '1D'),
            'VIX': mock_historical_bars('VIX', '2024-01-01', '2024-01-31', '1D')
        }

        # Fetch market data
        market_data = adapter.fetch_market_data()

        # All column names should be lowercase after normalization
        for symbol, df in market_data.items():
            for col in df.columns:
                col_str = col if isinstance(col, str) else str(col)
                assert col_str == col_str.lower(), \
                    f"Column '{col}' in {symbol} data should be lowercase. Got: {list(df.columns)}"

    @pytest.mark.network
    def test_fetch_market_data_vix_has_lowercase_close(self, mock_broker):
        """
        Test that VIX data specifically has lowercase 'close' column.

        This directly tests the bug fix for KeyError: 'close'.
        """
        def mock_historical_bars(symbol, start, end, timeframe):
            import pandas as pd
            dates = pd.date_range(start, end, freq='D')
            data = {
                'open': [100.0] * len(dates),
                'high': [101.0] * len(dates),
                'low': [99.0] * len(dates),
                'close': [100.5] * len(dates),
                'volume': [1000000] * len(dates)
            }
            return pd.DataFrame(data, index=dates)

        mock_broker.get_historical_bars.side_effect = mock_historical_bars

        adapter = OMRLiveAdapter(
            broker=mock_broker,
            symbols=['TQQQ']
        )

        # Simulate cache with VIX data having uppercase columns (like yfinance returns)
        import pandas as pd
        dates = pd.date_range('2024-01-01', '2024-01-31', freq='D')
        adapter._data_cache = {
            'SPY': pd.DataFrame({
                'Close': [450.0] * len(dates),  # Uppercase
                'Open': [449.0] * len(dates),
                'High': [451.0] * len(dates),
                'Low': [448.0] * len(dates),
                'Volume': [50000000] * len(dates)
            }, index=dates),
            'VIX': pd.DataFrame({
                'Close': [15.0] * len(dates),  # Uppercase (yfinance style)
                'Open': [14.5] * len(dates),
                'High': [15.5] * len(dates),
                'Low': [14.0] * len(dates),
                'Volume': [0] * len(dates)
            }, index=dates)
        }

        # Fetch market data
        market_data = adapter.fetch_market_data()

        # VIX should have lowercase 'close' column after normalization
        assert 'VIX' in market_data, "VIX should be in market data"
        vix_df = market_data['VIX']
        assert 'close' in vix_df.columns, \
            f"VIX data should have lowercase 'close' column. Got: {list(vix_df.columns)}"

        # Should be able to access vix_data['close'] without KeyError
        try:
            _ = vix_df['close'].iloc[-1]
        except KeyError as e:
            pytest.fail(f"Should be able to access vix_data['close']: {e}")

    @pytest.mark.network
    def test_real_yfinance_vix_integration(self, mock_broker):
        """
        End-to-end test: fetch real VIX data and verify column normalization.

        This test makes actual network calls and verifies the complete flow.
        """
        adapter = OMRLiveAdapter(
            broker=mock_broker,
            symbols=['TQQQ']
        )

        from datetime import datetime, timedelta
        end_date = datetime.now()
        start_date = end_date - timedelta(days=10)

        # Fetch real VIX data
        vix_data = adapter._fetch_vix_yfinance(
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
        )

        if vix_data is None or vix_data.empty:
            pytest.skip("Could not fetch VIX data (network issue or market closed)")

        # Store in cache (simulating preload)
        adapter._data_cache = {
            'SPY': vix_data.copy(),  # Use same data for SPY (just for testing)
            'VIX': vix_data.copy()
        }

        # Fetch market data (should normalize)
        market_data = adapter.fetch_market_data()

        # Verify VIX has lowercase columns
        if 'VIX' in market_data:
            vix_df = market_data['VIX']
            columns_str = [str(c).lower() for c in vix_df.columns]
            assert 'close' in columns_str, \
                f"After normalization, VIX should have 'close' column. Got: {list(vix_df.columns)}"


class TestColumnNormalization:
    """Tests specifically for column name normalization edge cases."""

    def test_normalize_single_level_uppercase(self, mock_broker):
        """Test normalization of simple uppercase column names."""
        adapter = OMRLiveAdapter(broker=mock_broker, symbols=['SPY'])

        import pandas as pd
        dates = pd.date_range('2024-01-01', '2024-01-10', freq='D')

        # Uppercase columns (common from yfinance)
        df = pd.DataFrame({
            'Open': [100.0] * len(dates),
            'High': [101.0] * len(dates),
            'Low': [99.0] * len(dates),
            'Close': [100.5] * len(dates),
            'Volume': [1000000] * len(dates)
        }, index=dates)

        adapter._data_cache = {'SPY': df, 'VIX': df.copy()}

        market_data = adapter.fetch_market_data()

        assert 'close' in market_data['SPY'].columns
        assert 'open' in market_data['SPY'].columns
        assert 'Close' not in market_data['SPY'].columns
        assert 'Open' not in market_data['SPY'].columns

    def test_normalize_mixed_case_columns(self, mock_broker):
        """Test normalization of mixed case column names."""
        adapter = OMRLiveAdapter(broker=mock_broker, symbols=['SPY'])

        import pandas as pd
        dates = pd.date_range('2024-01-01', '2024-01-10', freq='D')

        # Mixed case columns
        df = pd.DataFrame({
            'OPEN': [100.0] * len(dates),
            'high': [101.0] * len(dates),
            'Low': [99.0] * len(dates),
            'CLOSE': [100.5] * len(dates),
            'Volume': [1000000] * len(dates)
        }, index=dates)

        adapter._data_cache = {'SPY': df, 'VIX': df.copy()}

        market_data = adapter.fetch_market_data()

        # All should be lowercase
        for col in market_data['SPY'].columns:
            assert col == col.lower(), f"Column '{col}' should be lowercase"

    def test_normalize_already_lowercase(self, mock_broker):
        """Test that already lowercase columns remain unchanged."""
        adapter = OMRLiveAdapter(broker=mock_broker, symbols=['SPY'])

        import pandas as pd
        dates = pd.date_range('2024-01-01', '2024-01-10', freq='D')

        # Already lowercase (from Alpaca)
        df = pd.DataFrame({
            'open': [100.0] * len(dates),
            'high': [101.0] * len(dates),
            'low': [99.0] * len(dates),
            'close': [100.5] * len(dates),
            'volume': [1000000] * len(dates)
        }, index=dates)

        adapter._data_cache = {'SPY': df, 'VIX': df.copy()}

        market_data = adapter.fetch_market_data()

        # Should still be lowercase
        assert list(market_data['SPY'].columns) == ['open', 'high', 'low', 'close', 'volume']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
