"""
Tests for StrategyAdapter base class.

Tests the core adapter functionality that connects pure strategies
to live trading infrastructure.
"""

import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from datetime import datetime, timedelta
import pandas as pd
import pytz

from src.trading.adapters.strategy_adapter import StrategyAdapter
from src.strategies.core import Signal


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def mock_broker():
    """Create a mock broker with controllable behavior."""
    broker = MagicMock()

    # Default account info
    broker.get_account.return_value = {
        'buying_power': 100000.0,
        'portfolio_value': 100000.0,
        'cash': 100000.0
    }

    # Default positions (empty)
    broker.get_positions.return_value = []

    # Market is open by default
    broker.is_market_open.return_value = True

    # Default historical data
    def get_bars(symbol, start, end, timeframe):
        dates = pd.date_range(start=start, end=end, freq='D')
        return pd.DataFrame({
            'open': [100.0] * len(dates),
            'high': [105.0] * len(dates),
            'low': [95.0] * len(dates),
            'close': [102.0] * len(dates),
            'volume': [1000000] * len(dates)
        }, index=dates)

    broker.get_historical_bars.side_effect = get_bars

    return broker


@pytest.fixture
def mock_strategy():
    """Create a mock pure strategy that generates controllable signals."""
    strategy = MagicMock()
    strategy.__class__.__name__ = 'MockStrategy'

    # Default: return empty signals
    strategy.generate_signals.return_value = []

    return strategy


@pytest.fixture
def sample_signals():
    """Create sample trading signals for testing."""
    now = datetime.now()
    return [
        Signal(timestamp=now, symbol='AAPL', direction='BUY', price=150.0, confidence=0.8),
        Signal(timestamp=now, symbol='MSFT', direction='BUY', price=350.0, confidence=0.75),
        Signal(timestamp=now, symbol='GOOGL', direction='BUY', price=140.0, confidence=0.7),
    ]


class ConcreteStrategyAdapter(StrategyAdapter):
    """Concrete implementation for testing the abstract base class."""

    def __init__(self, *args, schedule_config=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._schedule_config = schedule_config or {
            'market_hours_only': True,
            'specific_time': None,
            'interval': '1d'
        }

    def get_schedule(self):
        return self._schedule_config


@pytest.fixture
def adapter(mock_broker, mock_strategy):
    """Create a concrete adapter instance for testing."""
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']

    with patch('src.trading.adapters.strategy_adapter.ExecutionEngine'):
        with patch('src.trading.adapters.strategy_adapter.PositionManager') as mock_pm:
            # Configure position manager mock
            mock_pm_instance = MagicMock()
            mock_pm_instance.get_open_positions.return_value = []
            mock_pm.return_value = mock_pm_instance

            adapter = ConcreteStrategyAdapter(
                strategy=mock_strategy,
                broker=mock_broker,
                symbols=symbols,
                position_size=0.1,
                max_positions=5,
                data_lookback_days=365
            )
            adapter.position_manager = mock_pm_instance

            return adapter


# =============================================================================
# INITIALIZATION TESTS
# =============================================================================

class TestAdapterInitialization:
    """Tests for StrategyAdapter initialization."""

    def test_init_with_required_params(self, mock_broker, mock_strategy):
        """Test adapter initializes with required parameters."""
        symbols = ['AAPL', 'MSFT']

        with patch('src.trading.adapters.strategy_adapter.ExecutionEngine'):
            with patch('src.trading.adapters.strategy_adapter.PositionManager'):
                adapter = ConcreteStrategyAdapter(
                    strategy=mock_strategy,
                    broker=mock_broker,
                    symbols=symbols
                )

        assert adapter.strategy == mock_strategy
        assert adapter.broker == mock_broker
        assert adapter.symbols == symbols
        assert adapter.position_size == 0.1  # default
        assert adapter.max_positions == 5   # default
        assert adapter.data_lookback_days == 365  # default

    def test_init_with_custom_params(self, mock_broker, mock_strategy):
        """Test adapter initializes with custom parameters."""
        symbols = ['SPY', 'QQQ']

        with patch('src.trading.adapters.strategy_adapter.ExecutionEngine'):
            with patch('src.trading.adapters.strategy_adapter.PositionManager'):
                adapter = ConcreteStrategyAdapter(
                    strategy=mock_strategy,
                    broker=mock_broker,
                    symbols=symbols,
                    position_size=0.2,
                    max_positions=10,
                    data_lookback_days=180
                )

        assert adapter.position_size == 0.2
        assert adapter.max_positions == 10
        assert adapter.data_lookback_days == 180

    def test_init_with_data_provider(self, mock_broker, mock_strategy):
        """Test adapter initializes with optional data provider."""
        mock_provider = MagicMock()
        mock_provider.name = 'TestProvider'

        with patch('src.trading.adapters.strategy_adapter.ExecutionEngine'):
            with patch('src.trading.adapters.strategy_adapter.PositionManager'):
                adapter = ConcreteStrategyAdapter(
                    strategy=mock_strategy,
                    broker=mock_broker,
                    symbols=['AAPL'],
                    data_provider=mock_provider
                )

        assert adapter._data_provider == mock_provider

    def test_init_creates_execution_engine(self, mock_broker, mock_strategy):
        """Test adapter creates ExecutionEngine with broker."""
        with patch('src.trading.adapters.strategy_adapter.ExecutionEngine') as mock_ee:
            with patch('src.trading.adapters.strategy_adapter.PositionManager'):
                ConcreteStrategyAdapter(
                    strategy=mock_strategy,
                    broker=mock_broker,
                    symbols=['AAPL']
                )

        # metrics_registry is forwarded so ExecutionEngine can record
        # hg_orders_* metrics. It defaults to None, which is behaviourally
        # identical to the previous ExecutionEngine(broker) call.
        mock_ee.assert_called_once_with(mock_broker, metrics_registry=None)

    def test_init_creates_position_manager(self, mock_broker, mock_strategy):
        """Test adapter creates PositionManager with correct config."""
        with patch('src.trading.adapters.strategy_adapter.ExecutionEngine'):
            with patch('src.trading.adapters.strategy_adapter.PositionManager') as mock_pm:
                ConcreteStrategyAdapter(
                    strategy=mock_strategy,
                    broker=mock_broker,
                    symbols=['AAPL'],
                    position_size=0.15,
                    max_positions=8
                )

        # Check position manager was created with correct config
        call_args = mock_pm.call_args[0][0]
        assert call_args['max_position_size_pct'] == 0.15
        assert call_args['max_concurrent_positions'] == 8
        assert call_args['max_total_exposure_pct'] == 0.15 * 8


# =============================================================================
# DATA PRELOADING TESTS
# =============================================================================

class TestPreloadHistoricalData:
    """Tests for preload_historical_data method."""

    def test_preload_fetches_data_for_all_symbols(self, adapter):
        """Test that preload fetches data for each symbol."""
        adapter.preload_historical_data()

        # Should have called get_historical_bars for each symbol
        assert adapter.broker.get_historical_bars.call_count == len(adapter.symbols)

    def test_preload_stores_in_cache(self, adapter):
        """Test that preloaded data is stored in cache."""
        adapter.preload_historical_data()

        assert adapter._data_cache is not None
        assert len(adapter._data_cache) == len(adapter.symbols)
        assert adapter._cache_date is not None

    def test_preload_handles_empty_data(self, adapter):
        """Test preload handles symbols with no data."""
        # Make one symbol return empty data
        def selective_bars(symbol, start, end, timeframe):
            if symbol == 'META':
                return pd.DataFrame()  # Empty
            dates = pd.date_range(start=start, end=end, freq='D')
            return pd.DataFrame({
                'open': [100.0] * len(dates),
                'high': [105.0] * len(dates),
                'low': [95.0] * len(dates),
                'close': [102.0] * len(dates),
                'volume': [1000000] * len(dates)
            }, index=dates)

        adapter.broker.get_historical_bars.side_effect = selective_bars
        adapter.preload_historical_data()

        # Should have 4 symbols (not META)
        assert len(adapter._data_cache) == 4
        assert 'META' not in adapter._data_cache

    def test_preload_handles_exceptions(self, adapter):
        """Test preload continues after exception for one symbol."""
        def failing_bars(symbol, start, end, timeframe):
            if symbol == 'GOOGL':
                raise Exception("API Error")
            dates = pd.date_range(start=start, end=end, freq='D')
            return pd.DataFrame({
                'open': [100.0] * len(dates),
                'high': [105.0] * len(dates),
                'low': [95.0] * len(dates),
                'close': [102.0] * len(dates),
                'volume': [1000000] * len(dates)
            }, index=dates)

        adapter.broker.get_historical_bars.side_effect = failing_bars
        adapter.preload_historical_data()

        # Should have 4 symbols (not GOOGL)
        assert len(adapter._data_cache) == 4
        assert 'GOOGL' not in adapter._data_cache


# =============================================================================
# DATA FETCHING TESTS
# =============================================================================

class TestFetchMarketData:
    """Tests for fetch_market_data method."""

    def test_fetch_uses_cache_when_valid(self, adapter):
        """Test fetch uses cache when cache date matches today."""
        # Pre-populate cache
        adapter._data_cache = {
            'AAPL': pd.DataFrame({'close': [150.0]}, index=[datetime.now()])
        }
        adapter._cache_date = datetime.now().date()

        result = adapter.fetch_market_data()

        # Should have data for AAPL
        assert 'AAPL' in result

    def test_fetch_performs_full_fetch_without_cache(self, adapter):
        """Test fetch performs full fetch when no cache."""
        adapter._data_cache = None
        adapter._cache_date = None

        result = adapter.fetch_market_data()

        # Should have fetched all symbols
        assert len(result) == len(adapter.symbols)

    def test_fetch_invalidates_stale_cache(self, adapter):
        """Test fetch invalidates cache from previous day."""
        # Set cache from yesterday
        adapter._data_cache = {'AAPL': pd.DataFrame({'close': [150.0]})}
        adapter._cache_date = (datetime.now() - timedelta(days=1)).date()

        result = adapter.fetch_market_data()

        # Should have performed full fetch
        assert adapter.broker.get_historical_bars.call_count == len(adapter.symbols)

    def test_fetch_returns_empty_on_error(self, adapter):
        """Test fetch returns empty dict on error."""
        adapter._data_cache = None
        adapter.broker.get_historical_bars.side_effect = Exception("API Error")

        result = adapter.fetch_market_data()

        assert result == {}


# =============================================================================
# SIGNAL GENERATION TESTS
# =============================================================================

class TestGenerateSignals:
    """Tests for generate_signals method."""

    def test_generate_delegates_to_strategy(self, adapter, sample_signals):
        """Test signal generation delegates to pure strategy."""
        market_data = {'AAPL': pd.DataFrame({'close': [150.0]})}
        adapter.strategy.generate_signals.return_value = sample_signals

        result = adapter.generate_signals(market_data)

        # Strategy should be called with market data
        adapter.strategy.generate_signals.assert_called_once()
        call_args = adapter.strategy.generate_signals.call_args
        assert call_args[0][0] == market_data

        # Should return strategy's signals
        assert result == sample_signals

    def test_generate_returns_empty_on_error(self, adapter):
        """Test generate_signals returns empty list on error."""
        adapter.strategy.generate_signals.side_effect = Exception("Strategy Error")

        result = adapter.generate_signals({'AAPL': pd.DataFrame()})

        assert result == []

    def test_generate_passes_timestamp(self, adapter):
        """Test generate_signals passes timestamp to strategy."""
        market_data = {'AAPL': pd.DataFrame({'close': [150.0]})}

        with patch('src.trading.adapters.strategy_adapter.datetime') as mock_dt:
            mock_now = datetime(2024, 1, 15, 15, 50)
            mock_dt.now.return_value = mock_now

            adapter.generate_signals(market_data)

        call_args = adapter.strategy.generate_signals.call_args
        assert call_args[0][1] == mock_now


# =============================================================================
# SIGNAL FILTERING TESTS
# =============================================================================

class TestFilterSignals:
    """Tests for filter_signals method."""

    def test_filter_respects_max_positions(self, adapter, sample_signals):
        """Test filter respects max_positions limit."""
        # Set max_positions to 2
        adapter.max_positions = 2

        # No existing positions
        adapter.position_manager.get_open_positions.return_value = []

        result = adapter.filter_signals(sample_signals)

        # Should only keep 2 signals
        assert len(result) == 2

    def test_filter_skips_existing_positions(self, adapter, sample_signals):
        """Test filter skips symbols with existing positions."""
        # Already have position in AAPL
        adapter.position_manager.get_open_positions.return_value = [
            {'symbol': 'AAPL', 'quantity': 100, 'avg_entry_price': 145.0}
        ]

        result = adapter.filter_signals(sample_signals)

        # AAPL should be filtered out
        result_symbols = [s.symbol for s in result]
        assert 'AAPL' not in result_symbols
        assert 'MSFT' in result_symbols

    def test_filter_counts_existing_positions(self, adapter, sample_signals):
        """Test filter counts existing positions toward max."""
        adapter.max_positions = 3

        # Already have 2 positions
        adapter.position_manager.get_open_positions.return_value = [
            {'symbol': 'TSLA', 'quantity': 50, 'avg_entry_price': 200.0},
            {'symbol': 'NVDA', 'quantity': 30, 'avg_entry_price': 500.0}
        ]

        result = adapter.filter_signals(sample_signals)

        # Can only add 1 more position
        assert len(result) == 1

    def test_filter_returns_empty_when_maxed(self, adapter, sample_signals):
        """Test filter returns empty when at max positions."""
        adapter.max_positions = 2

        # Already at max positions
        adapter.position_manager.get_open_positions.return_value = [
            {'symbol': 'TSLA', 'quantity': 50, 'avg_entry_price': 200.0},
            {'symbol': 'NVDA', 'quantity': 30, 'avg_entry_price': 500.0}
        ]

        result = adapter.filter_signals(sample_signals)

        assert len(result) == 0


# =============================================================================
# SIGNAL EXECUTION TESTS
# =============================================================================

class TestExecuteSignals:
    """Tests for execute_signals method."""

    def test_execute_calculates_position_size(self, adapter, sample_signals):
        """Test execute calculates correct position size."""
        # 100k buying power, 10% position size, $150 stock = 66 shares
        adapter.broker.get_account.return_value = {'buying_power': 100000.0}
        adapter.position_size = 0.1

        # Take just the AAPL signal
        signals = [sample_signals[0]]  # AAPL @ $150

        adapter.execute_signals(signals)

        # Should have called execute_order with correct qty
        adapter.execution_engine.execute_order.assert_called_once()
        call_kwargs = adapter.execution_engine.execute_order.call_args[1]
        assert call_kwargs['quantity'] == 66  # 10000 / 150 = 66

    def test_execute_skips_zero_quantity(self, adapter, sample_signals):
        """Test execute skips orders with zero quantity."""
        # Very low buying power results in 0 shares
        adapter.broker.get_account.return_value = {'buying_power': 100.0}

        adapter.execute_signals(sample_signals)

        # Should not have executed any orders
        adapter.execution_engine.execute_order.assert_not_called()

    def test_execute_handles_buy_direction(self, adapter, sample_signals):
        """Test execute creates BUY orders correctly."""
        from src.trading.brokers.broker_interface import OrderSide

        adapter.broker.get_account.return_value = {'buying_power': 100000.0}
        signals = [Signal(timestamp=datetime.now(), symbol='AAPL', direction='BUY', price=150.0, confidence=0.8)]

        adapter.execute_signals(signals)

        call_kwargs = adapter.execution_engine.execute_order.call_args[1]
        assert call_kwargs['side'] == OrderSide.BUY

    def test_execute_handles_sell_direction(self, adapter, sample_signals):
        """Test execute creates SELL orders correctly."""
        from src.trading.brokers.broker_interface import OrderSide

        adapter.broker.get_account.return_value = {'buying_power': 100000.0}
        signals = [Signal(timestamp=datetime.now(), symbol='AAPL', direction='SELL', price=150.0, confidence=0.8)]

        adapter.execute_signals(signals)

        call_kwargs = adapter.execution_engine.execute_order.call_args[1]
        assert call_kwargs['side'] == OrderSide.SELL

    def test_execute_skips_unknown_direction(self, adapter):
        """Test execute skips signals with unknown direction."""
        adapter.broker.get_account.return_value = {'buying_power': 100000.0}
        signals = [Signal(timestamp=datetime.now(), symbol='AAPL', direction='HOLD', price=150.0, confidence=0.8)]

        adapter.execute_signals(signals)

        adapter.execution_engine.execute_order.assert_not_called()

    def test_execute_handles_no_account(self, adapter, sample_signals):
        """Test execute handles missing account info."""
        adapter.broker.get_account.return_value = None

        adapter.execute_signals(sample_signals)

        # Should not have executed any orders
        adapter.execution_engine.execute_order.assert_not_called()

    def test_execute_empty_signals(self, adapter):
        """Test execute handles empty signals list."""
        adapter.execute_signals([])

        adapter.execution_engine.execute_order.assert_not_called()


# =============================================================================
# SCHEDULING TESTS
# =============================================================================

class TestShouldRunNow:
    """Tests for should_run_now method."""

    def test_should_run_respects_market_hours(self, mock_broker, mock_strategy):
        """Test should_run_now checks market hours when required."""
        with patch('src.trading.adapters.strategy_adapter.ExecutionEngine'):
            with patch('src.trading.adapters.strategy_adapter.PositionManager'):
                adapter = ConcreteStrategyAdapter(
                    strategy=mock_strategy,
                    broker=mock_broker,
                    symbols=['AAPL'],
                    schedule_config={'market_hours_only': True}
                )

        # Market closed
        mock_broker.is_market_open.return_value = False
        assert adapter.should_run_now() == False

        # Market open
        mock_broker.is_market_open.return_value = True
        assert adapter.should_run_now() == True

    def test_should_run_ignores_market_hours_when_disabled(self, mock_broker, mock_strategy):
        """Test should_run_now ignores market hours when not required."""
        with patch('src.trading.adapters.strategy_adapter.ExecutionEngine'):
            with patch('src.trading.adapters.strategy_adapter.PositionManager'):
                adapter = ConcreteStrategyAdapter(
                    strategy=mock_strategy,
                    broker=mock_broker,
                    symbols=['AAPL'],
                    schedule_config={'market_hours_only': False}
                )

        # Market closed but should still run
        mock_broker.is_market_open.return_value = False
        assert adapter.should_run_now() == True

    def test_should_run_at_specific_time(self, mock_broker, mock_strategy):
        """Test should_run_now checks specific time correctly."""
        with patch('src.trading.adapters.strategy_adapter.ExecutionEngine'):
            with patch('src.trading.adapters.strategy_adapter.PositionManager'):
                adapter = ConcreteStrategyAdapter(
                    strategy=mock_strategy,
                    broker=mock_broker,
                    symbols=['AAPL'],
                    schedule_config={
                        'market_hours_only': False,
                        'specific_time': '15:50'
                    }
                )

        # At target time
        with patch('src.trading.adapters.strategy_adapter.datetime') as mock_dt:
            mock_now = datetime(2024, 1, 15, 15, 50, 30)
            mock_dt.now.return_value = mock_now
            mock_dt.strptime = datetime.strptime
            mock_dt.combine = datetime.combine

            assert adapter.should_run_now() == True

    def test_should_not_run_outside_time_window(self, mock_broker, mock_strategy):
        """Test should_run_now returns False outside time window."""
        with patch('src.trading.adapters.strategy_adapter.ExecutionEngine'):
            with patch('src.trading.adapters.strategy_adapter.PositionManager'):
                adapter = ConcreteStrategyAdapter(
                    strategy=mock_strategy,
                    broker=mock_broker,
                    symbols=['AAPL'],
                    schedule_config={
                        'market_hours_only': False,
                        'specific_time': '15:50'
                    }
                )

        # 5 minutes away from target
        with patch('src.trading.adapters.strategy_adapter.datetime') as mock_dt:
            mock_now = datetime(2024, 1, 15, 15, 45, 0)  # 5 min early
            mock_dt.now.return_value = mock_now
            mock_dt.strptime = datetime.strptime
            mock_dt.combine = datetime.combine

            assert adapter.should_run_now() == False


# =============================================================================
# RUN ONCE INTEGRATION TESTS
# =============================================================================

class TestRunOnce:
    """Integration tests for run_once method."""

    def test_run_once_full_workflow(self, adapter, sample_signals):
        """Test run_once executes full workflow."""
        # Setup
        adapter.strategy.generate_signals.return_value = sample_signals[:1]
        adapter.broker.get_account.return_value = {'buying_power': 100000.0}
        adapter.execution_engine.execute_order.return_value = {'order_id': 'TEST123'}

        # Preload data first
        adapter.preload_historical_data()

        # Run
        adapter.run_once()

        # Verify workflow
        assert adapter.strategy.generate_signals.called
        assert adapter.execution_engine.execute_order.called

    def test_run_once_handles_no_data(self, adapter):
        """Test run_once handles missing market data gracefully."""
        adapter._data_cache = None
        # Clear side_effect first, then set return_value
        adapter.broker.get_historical_bars.side_effect = None
        adapter.broker.get_historical_bars.return_value = None

        # Should not raise
        adapter.run_once()

        # Strategy should not be called with no data
        adapter.strategy.generate_signals.assert_not_called()

    def test_run_once_handles_no_signals(self, adapter):
        """Test run_once handles no signals gracefully."""
        adapter.strategy.generate_signals.return_value = []
        adapter.preload_historical_data()

        adapter.run_once()

        # Execution should not be called
        adapter.execution_engine.execute_order.assert_not_called()


# =============================================================================
# INTRADAY DATA TESTS
# =============================================================================

class TestPrefetchIntradayData:
    """Tests for prefetch_intraday_data method."""

    def test_prefetch_uses_data_provider_when_available(self, adapter):
        """Test prefetch uses data provider with fallback chain."""
        mock_provider = MagicMock()
        mock_provider.name = 'TestProvider'
        mock_provider.get_historical_bars_batch.return_value = {
            'AAPL': pd.DataFrame({'close': [150.0]})
        }
        adapter._data_provider = mock_provider

        adapter.prefetch_intraday_data()

        mock_provider.get_historical_bars_batch.assert_called_once()
        assert adapter._intraday_cache is not None

    def test_prefetch_falls_back_to_broker(self, adapter):
        """Test prefetch falls back to broker when no data provider."""
        adapter._data_provider = None

        adapter.prefetch_intraday_data()

        # Should have called broker for each symbol
        assert adapter.broker.get_historical_bars.call_count == len(adapter.symbols)

    def test_prefetch_respects_force_refresh(self, adapter):
        """Test prefetch passes force_refresh to data provider."""
        mock_provider = MagicMock()
        mock_provider.name = 'TestProvider'
        mock_provider.get_historical_bars_batch.return_value = {}
        adapter._data_provider = mock_provider

        adapter.prefetch_intraday_data(force_refresh=True)

        call_kwargs = mock_provider.get_historical_bars_batch.call_args[1]
        assert call_kwargs.get('force_refresh') == True


# =============================================================================
# UPDATE POSITIONS TESTS
# =============================================================================

class TestUpdatePositions:
    """Tests for update_positions method."""

    def test_update_logs_broker_positions(self, adapter):
        """Test update_positions logs positions from broker."""
        adapter.broker.get_positions.return_value = [
            {
                'symbol': 'AAPL',
                'quantity': 100,
                'avg_entry_price': 145.0,
                'current_price': 150.0
            }
        ]

        # Should not raise
        adapter.update_positions()

        adapter.broker.get_positions.assert_called_once()

    def test_update_handles_empty_positions(self, adapter):
        """Test update_positions handles no positions."""
        adapter.broker.get_positions.return_value = []

        # Should not raise
        adapter.update_positions()

    def test_update_handles_error(self, adapter):
        """Test update_positions handles exceptions."""
        adapter.broker.get_positions.side_effect = Exception("API Error")

        # Should not raise
        adapter.update_positions()
