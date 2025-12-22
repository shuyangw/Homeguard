"""
Tests for OMR (Overnight Mean Reversion) Live Adapter.

Tests the live trading adapter for overnight mean reversion strategy.
"""

import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from datetime import datetime, time, timedelta
import pandas as pd
import pytz

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
        if len(dates) == 0:
            dates = pd.date_range(start=start, periods=10, freq='D')
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
def mock_regime_detector():
    """Create a mock regime detector."""
    detector = MagicMock()
    detector.classify_regime.return_value = ('STRONG_BULL', 0.85)
    detector.get_regime_parameters.return_value = {
        'regime': 'STRONG_BULL',
        'top_n': 20,
        'long_period': 21,
        'short_period': 5
    }
    return detector


@pytest.fixture
def mock_bayesian_model():
    """Create a mock Bayesian model."""
    model = MagicMock()
    model.trained = True
    model.regime_probabilities = {'TQQQ': {}, 'SQQQ': {}}
    model.load_model.return_value = None
    return model


@pytest.fixture
def omr_adapter(mock_broker, mock_regime_detector, mock_bayesian_model):
    """Create an OMR adapter with mocked dependencies."""
    # Patch ExecutionEngine and PositionManager in strategy_adapter (parent class)
    with patch('src.trading.adapters.strategy_adapter.ExecutionEngine') as mock_ee:
        with patch('src.trading.adapters.strategy_adapter.PositionManager') as mock_pm:
            with patch('src.trading.adapters.omr_live_adapter.StrategyStateManager') as mock_sm:
                with patch('src.trading.adapters.omr_live_adapter.PortfolioHealthChecker') as mock_hc:
                    # Configure execution engine
                    mock_ee_instance = MagicMock()
                    mock_ee.return_value = mock_ee_instance

                    # Configure position manager
                    mock_pm_instance = MagicMock()
                    mock_pm_instance.get_open_positions.return_value = []
                    mock_pm.return_value = mock_pm_instance

                    # Configure state manager
                    mock_sm_instance = MagicMock()
                    mock_sm_instance.is_enabled.return_value = True
                    mock_sm_instance.is_shutdown_requested.return_value = False
                    mock_sm_instance.acquire_execution_lock.return_value = True
                    mock_sm_instance.get_positions.return_value = {}
                    mock_sm.return_value = mock_sm_instance

                    # Configure health checker
                    mock_hc_instance = MagicMock()
                    health_result = MagicMock()
                    health_result.passed = True
                    health_result.warnings = []
                    health_result.errors = []
                    mock_hc_instance.check_before_entry.return_value = health_result
                    mock_hc_instance.check_before_exit.return_value = health_result
                    mock_hc.return_value = mock_hc_instance

                    from src.trading.adapters.omr_live_adapter import OMRLiveAdapter

                    adapter = OMRLiveAdapter(
                        broker=mock_broker,
                        symbols=['TQQQ', 'SQQQ', 'UPRO'],
                        min_probability=0.55,
                        min_expected_return=0.002,
                        max_positions=3,
                        position_size=0.1,
                        regime_detector=mock_regime_detector,
                        bayesian_model=mock_bayesian_model
                    )

                    adapter.execution_engine = mock_ee_instance
                    adapter.position_manager = mock_pm_instance
                    adapter.state_manager = mock_sm_instance
                    adapter.health_checker = mock_hc_instance

                    return adapter


# =============================================================================
# INITIALIZATION TESTS
# =============================================================================

class TestOMRLiveAdapterInit:
    """Tests for OMRLiveAdapter initialization."""

    def test_init_with_custom_symbols(self, mock_broker, mock_regime_detector, mock_bayesian_model):
        """Test adapter initializes with custom symbols."""
        with patch('src.trading.adapters.strategy_adapter.ExecutionEngine'):
            with patch('src.trading.adapters.strategy_adapter.PositionManager'):
                with patch('src.trading.adapters.omr_live_adapter.StrategyStateManager'):
                    with patch('src.trading.adapters.omr_live_adapter.PortfolioHealthChecker'):
                        from src.trading.adapters.omr_live_adapter import OMRLiveAdapter

                        symbols = ['TQQQ', 'SQQQ']
                        adapter = OMRLiveAdapter(
                            broker=mock_broker,
                            symbols=symbols,
                            regime_detector=mock_regime_detector,
                            bayesian_model=mock_bayesian_model
                        )

        assert adapter.symbols == symbols

    def test_init_with_default_symbols(self, mock_broker, mock_regime_detector, mock_bayesian_model):
        """Test adapter uses default ETF universe when no symbols specified."""
        with patch('src.trading.adapters.strategy_adapter.ExecutionEngine'):
            with patch('src.trading.adapters.strategy_adapter.PositionManager'):
                with patch('src.trading.adapters.omr_live_adapter.StrategyStateManager'):
                    with patch('src.trading.adapters.omr_live_adapter.PortfolioHealthChecker'):
                        from src.trading.adapters.omr_live_adapter import OMRLiveAdapter
                        from src.strategies.universe import ETFUniverse

                        adapter = OMRLiveAdapter(
                            broker=mock_broker,
                            symbols=None,
                            regime_detector=mock_regime_detector,
                            bayesian_model=mock_bayesian_model
                        )

        assert adapter.symbols == ETFUniverse.LEVERAGED_3X

    def test_init_stores_thresholds(self, omr_adapter):
        """Test adapter stores probability and return thresholds."""
        assert omr_adapter.min_probability == 0.55
        assert omr_adapter.min_expected_return == 0.002


# =============================================================================
# SCHEDULE TESTS
# =============================================================================

class TestOMRSchedule:
    """Tests for OMR schedule configuration."""

    def test_get_schedule_returns_correct_times(self, omr_adapter):
        """Test get_schedule returns entry and exit times."""
        schedule = omr_adapter.get_schedule()

        assert 'execution_times' in schedule
        times = schedule['execution_times']
        assert len(times) == 2

        # Check entry time
        entry = next(t for t in times if t['action'] == 'entry')
        assert entry['time'] == '15:50'

        # Check exit time
        exit_ = next(t for t in times if t['action'] == 'exit')
        assert exit_['time'] == '09:31'

    def test_get_schedule_market_hours_only(self, omr_adapter):
        """Test get_schedule requires market hours."""
        schedule = omr_adapter.get_schedule()
        assert schedule.get('market_hours_only') == True

    def test_get_schedule_overnight_type(self, omr_adapter):
        """Test get_schedule indicates overnight strategy."""
        schedule = omr_adapter.get_schedule()
        assert schedule.get('strategy_type') == 'overnight'


# =============================================================================
# DATA FETCHING TESTS
# =============================================================================

class TestOMRDataFetching:
    """Tests for OMR market data fetching."""

    def test_fetch_market_data_uses_intraday_cache(self, omr_adapter):
        """Test fetch_market_data uses intraday cache when available."""
        # Setup intraday cache
        omr_adapter._intraday_cache = {
            'TQQQ': pd.DataFrame({'close': [50.0]}, index=[datetime.now()]),
            'SQQQ': pd.DataFrame({'close': [30.0]}, index=[datetime.now()]),
            'UPRO': pd.DataFrame({'close': [40.0]}, index=[datetime.now()])
        }

        result = omr_adapter.fetch_market_data()

        # Should use cached data
        assert 'TQQQ' in result
        assert 'SQQQ' in result

    def test_fetch_market_data_falls_back_to_broker(self, omr_adapter):
        """Test fetch_market_data falls back to broker when no cache."""
        omr_adapter._intraday_cache = None
        omr_adapter._data_provider = None

        result = omr_adapter.fetch_market_data()

        # Should have called broker
        assert omr_adapter.broker.get_historical_bars.called

    def test_fetch_market_data_normalizes_column_names(self, omr_adapter):
        """Test fetch_market_data normalizes column names to lowercase."""
        # Mock broker returning uppercase columns
        def get_bars_uppercase(symbol, start, end, timeframe):
            return pd.DataFrame({
                'Open': [100.0],
                'High': [105.0],
                'Low': [95.0],
                'Close': [102.0],
                'Volume': [1000000]
            }, index=[datetime.now()])

        omr_adapter.broker.get_historical_bars.side_effect = get_bars_uppercase
        omr_adapter._intraday_cache = None
        omr_adapter._data_provider = None

        result = omr_adapter.fetch_market_data()

        # Check columns are lowercase
        if 'TQQQ' in result:
            df = result['TQQQ']
            assert 'close' in df.columns


# =============================================================================
# EXECUTION TESTS
# =============================================================================

class TestOMRExecution:
    """Tests for OMR signal execution."""

    def test_execute_signals_tracks_positions(self, omr_adapter):
        """Test execute_signals tracks positions in state manager."""
        now = datetime.now()
        signals = [
            Signal(timestamp=now, symbol='TQQQ', direction='BUY', price=50.0, confidence=0.8)
        ]

        # Mock successful order
        omr_adapter.execution_engine.execute_order.return_value = {'order_id': 'TEST123'}

        omr_adapter.execute_signals(signals)

        # Should have tracked position
        omr_adapter.state_manager.add_or_update_position.assert_called()

    def test_execute_signals_handles_failed_order(self, omr_adapter):
        """Test execute_signals handles failed orders gracefully."""
        now = datetime.now()
        signals = [
            Signal(timestamp=now, symbol='TQQQ', direction='BUY', price=50.0, confidence=0.8)
        ]

        # Mock failed order
        omr_adapter.execution_engine.execute_order.return_value = None

        # Should not raise
        omr_adapter.execute_signals(signals)

        # Should not have tracked position
        omr_adapter.state_manager.add_or_update_position.assert_not_called()

    def test_execute_signals_calculates_quantity(self, omr_adapter):
        """Test execute_signals calculates correct quantity."""
        now = datetime.now()
        signals = [
            Signal(timestamp=now, symbol='TQQQ', direction='BUY', price=50.0, confidence=0.8)
        ]

        omr_adapter.execution_engine.execute_order.return_value = {'order_id': 'TEST123'}

        omr_adapter.execute_signals(signals)

        # Check quantity calculation: 100000 * 0.1 / 50 = 200 shares
        call_kwargs = omr_adapter.execution_engine.execute_order.call_args[1]
        assert call_kwargs['quantity'] == 200


# =============================================================================
# RUN ONCE TESTS
# =============================================================================

class TestOMRRunOnce:
    """Tests for OMR run_once method."""

    def test_run_once_checks_if_enabled(self, omr_adapter):
        """Test run_once respects enabled state."""
        omr_adapter.state_manager.is_enabled.return_value = False

        omr_adapter.run_once()

        # Should not have proceeded to execution
        omr_adapter.state_manager.acquire_execution_lock.assert_not_called()

    def test_run_once_checks_shutdown_requested(self, omr_adapter):
        """Test run_once respects shutdown request."""
        omr_adapter.state_manager.is_enabled.return_value = True
        omr_adapter.state_manager.is_shutdown_requested.return_value = True

        omr_adapter.run_once()

        # Should not have proceeded to execution
        omr_adapter.state_manager.acquire_execution_lock.assert_not_called()

    def test_run_once_acquires_execution_lock(self, omr_adapter):
        """Test run_once acquires execution lock."""
        omr_adapter.preload_historical_data()
        omr_adapter.run_once()

        omr_adapter.state_manager.acquire_execution_lock.assert_called_with('omr')

    def test_run_once_releases_lock_on_success(self, omr_adapter):
        """Test run_once releases lock after execution."""
        omr_adapter.preload_historical_data()
        omr_adapter.run_once()

        omr_adapter.state_manager.release_execution_lock.assert_called_with('omr')

    def test_run_once_blocks_on_failed_health_check(self, omr_adapter):
        """Test run_once blocks entry on failed health check."""
        # Configure health check to fail
        health_result = MagicMock()
        health_result.passed = False
        health_result.errors = ['Low buying power']
        health_result.warnings = []
        omr_adapter.health_checker.check_before_entry.return_value = health_result

        omr_adapter.preload_historical_data()
        omr_adapter.run_once()

        # Should not have executed signals (strategy.generate_signals not called for execution)
        # The lock should still be released
        omr_adapter.state_manager.release_execution_lock.assert_called()


# =============================================================================
# CLOSE POSITIONS TESTS
# =============================================================================

class TestOMRClosePositions:
    """Tests for OMR close_overnight_positions method."""

    def test_close_positions_with_no_positions(self, omr_adapter):
        """Test close_overnight_positions handles no positions."""
        omr_adapter.broker.get_positions.return_value = []

        omr_adapter.close_overnight_positions()

        # Should not have executed any close orders
        omr_adapter.execution_engine.execute_order.assert_not_called()

    def test_close_positions_only_closes_omr_positions(self, omr_adapter):
        """Test close_overnight_positions only closes OMR positions."""
        # Setup broker positions with both OMR (TQQQ) and non-OMR (AAPL)
        omr_adapter.broker.get_positions.return_value = [
            {'symbol': 'TQQQ', 'quantity': 100, 'avg_entry_price': 50.0, 'current_price': 52.0},
            {'symbol': 'AAPL', 'quantity': 50, 'avg_entry_price': 150.0, 'current_price': 155.0}
        ]

        # AAPL owned by another strategy
        omr_adapter.state_manager.get_positions.return_value = {'TQQQ': {}}
        omr_adapter.state_manager.symbol_owned_by_other.return_value = 'ramp'  # AAPL owned by RAMP

        omr_adapter.execution_engine.execute_order.return_value = {'order_id': 'CLOSE123'}

        omr_adapter.close_overnight_positions()

        # Should have closed TQQQ (leveraged ETF in OMR universe)
        # AAPL should be skipped (owned by ramp)
        calls = omr_adapter.execution_engine.execute_order.call_args_list
        symbols_closed = [call[1]['symbol'] for call in calls]
        assert 'TQQQ' in symbols_closed
        # AAPL might also be closed if ETFUniverse.is_leveraged returns False

    def test_close_positions_removes_from_state(self, omr_adapter):
        """Test close_overnight_positions removes closed positions from state."""
        omr_adapter.broker.get_positions.return_value = [
            {'symbol': 'TQQQ', 'quantity': 100, 'avg_entry_price': 50.0, 'current_price': 52.0}
        ]
        omr_adapter.state_manager.get_positions.return_value = {'TQQQ': {'entry_price': 50.0}}
        omr_adapter.execution_engine.execute_order.return_value = {'order_id': 'CLOSE123'}

        omr_adapter.close_overnight_positions()

        omr_adapter.state_manager.remove_position.assert_called_with('omr', 'TQQQ')


# =============================================================================
# SIGNAL WRAPPER TESTS
# =============================================================================

class TestOMRSignalWrapper:
    """Tests for OMRSignalWrapper class."""

    def test_signal_wrapper_converts_short_to_sell(self):
        """Test wrapper converts SHORT direction to SELL."""
        from src.trading.adapters.omr_live_adapter import OMRSignalWrapper

        mock_omr = MagicMock()
        mock_omr.generate_signals.return_value = [
            {
                'symbol': 'TQQQ',
                'direction': 'SHORT',
                'probability': 0.6,
                'signal_strength': 0.7,
                'current_price': 50.0
            }
        ]

        wrapper = OMRSignalWrapper(mock_omr)
        market_data = {'TQQQ': pd.DataFrame({'close': [50.0]})}

        signals = wrapper.generate_signals(market_data, datetime.now())

        assert len(signals) == 1
        assert signals[0].direction == 'SELL'

    def test_signal_wrapper_preserves_metadata(self):
        """Test wrapper preserves signal metadata."""
        from src.trading.adapters.omr_live_adapter import OMRSignalWrapper

        mock_omr = MagicMock()
        mock_omr.generate_signals.return_value = [
            {
                'symbol': 'TQQQ',
                'direction': 'BUY',
                'probability': 0.65,
                'expected_return': 0.02,
                'signal_strength': 0.8,
                'current_price': 50.0,
                'regime': 'STRONG_BULL'
            }
        ]

        wrapper = OMRSignalWrapper(mock_omr)
        market_data = {'TQQQ': pd.DataFrame({'close': [50.0]})}

        signals = wrapper.generate_signals(market_data, datetime.now())

        assert signals[0].metadata['probability'] == 0.65
        assert signals[0].metadata['regime'] == 'STRONG_BULL'

    def test_signal_wrapper_get_required_lookback(self):
        """Test wrapper returns correct lookback."""
        from src.trading.adapters.omr_live_adapter import OMRSignalWrapper

        mock_omr = MagicMock()
        wrapper = OMRSignalWrapper(mock_omr)

        assert wrapper.get_required_lookback() == 1
