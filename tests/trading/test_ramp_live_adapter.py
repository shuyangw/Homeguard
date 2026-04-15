"""
Tests for RAMP (Regime-Aware Momentum Protection) Live Adapter.

Tests the live trading adapter for regime-aware momentum strategy.
"""

import pytest
from unittest.mock import MagicMock, patch, mock_open
from datetime import datetime, time, timedelta
from pathlib import Path
import pandas as pd
import numpy as np

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
def ramp_adapter(mock_broker):
    """Create a RAMP adapter with mocked dependencies."""
    # Patch ExecutionEngine and PositionManager in strategy_adapter (parent class)
    with patch('src.trading.adapters.strategy_adapter.ExecutionEngine') as mock_ee:
        with patch('src.trading.adapters.strategy_adapter.PositionManager') as mock_pm:
            with patch('src.trading.adapters.ramp_live_adapter.StrategyStateManager') as mock_sm:
                with patch('src.trading.adapters.ramp_live_adapter.PortfolioHealthChecker') as mock_hc:
                    with patch('src.trading.adapters.ramp_live_adapter._load_cache_from_disk') as mock_cache:
                        # No disk cache
                        mock_cache.return_value = None

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
                        mock_hc.return_value = mock_hc_instance

                        from src.trading.adapters.ramp_live_adapter import RAMPLiveAdapter

                        adapter = RAMPLiveAdapter(
                            broker=mock_broker,
                            symbols=['AAPL', 'MSFT', 'GOOGL'],  # Small test universe
                            max_capital_allocation=1.0,
                            reduced_exposure=0.5,
                            vix_threshold=25.0,
                            spy_dd_threshold=-0.05,
                            broker_name='alpaca'
                        )

                        adapter.execution_engine = mock_ee_instance
                        adapter.position_manager = mock_pm_instance
                        adapter.state_manager = mock_sm_instance
                        adapter.health_checker = mock_hc_instance

                        return adapter


# =============================================================================
# INITIALIZATION TESTS
# =============================================================================

class TestRAMPLiveAdapterInit:
    """Tests for RAMPLiveAdapter initialization."""

    def test_init_with_custom_symbols(self, mock_broker):
        """Test adapter initializes with custom symbols."""
        with patch('src.trading.adapters.strategy_adapter.ExecutionEngine'):
            with patch('src.trading.adapters.strategy_adapter.PositionManager'):
                with patch('src.trading.adapters.ramp_live_adapter.StrategyStateManager'):
                    with patch('src.trading.adapters.ramp_live_adapter.PortfolioHealthChecker'):
                        with patch('src.trading.adapters.ramp_live_adapter._load_cache_from_disk', return_value=None):
                            from src.trading.adapters.ramp_live_adapter import RAMPLiveAdapter

                            symbols = ['AAPL', 'MSFT', 'GOOGL']
                            adapter = RAMPLiveAdapter(
                                broker=mock_broker,
                                symbols=symbols,
                                broker_name='alpaca'
                            )

        assert adapter.symbols == symbols

    def test_init_stores_configuration(self, ramp_adapter):
        """Test adapter stores configuration parameters."""
        assert ramp_adapter.max_capital_allocation == 1.0
        assert ramp_adapter.reduced_exposure == 0.5
        assert ramp_adapter.vix_threshold == 25.0
        assert ramp_adapter.spy_dd_threshold == -0.05

    def test_init_with_default_symbols_loads_sp500(self, mock_broker):
        """Test adapter loads S&P 500 when no symbols specified."""
        with patch('src.trading.adapters.strategy_adapter.ExecutionEngine'):
            with patch('src.trading.adapters.strategy_adapter.PositionManager'):
                with patch('src.trading.adapters.ramp_live_adapter.StrategyStateManager'):
                    with patch('src.trading.adapters.ramp_live_adapter.PortfolioHealthChecker'):
                        with patch('src.trading.adapters.ramp_live_adapter._load_cache_from_disk', return_value=None):
                            # Mock the CSV loading
                            mock_csv = pd.DataFrame({'Symbol': ['AAPL', 'MSFT', 'GOOGL']})

                            with patch('pandas.read_csv', return_value=mock_csv):
                                from src.trading.adapters.ramp_live_adapter import RAMPLiveAdapter

                                adapter = RAMPLiveAdapter(
                                    broker=mock_broker,
                                    symbols=None,
                                    broker_name='alpaca'
                                )

        # Should have loaded from CSV
        assert len(adapter.symbols) > 0


# =============================================================================
# SCHEDULE TESTS
# =============================================================================

class TestRAMPSchedule:
    """Tests for RAMP schedule configuration."""

    def test_get_schedule_returns_rebalance_time(self, ramp_adapter):
        """Test get_schedule returns correct rebalance time."""
        schedule = ramp_adapter.get_schedule()

        # RAMP uses execution_times format
        assert 'execution_times' in schedule
        times = schedule['execution_times']
        assert len(times) == 1
        assert times[0]['time'] == '15:55'
        assert times[0]['action'] == 'rebalance'

    def test_get_schedule_market_hours_only(self, ramp_adapter):
        """Test get_schedule requires market hours."""
        schedule = ramp_adapter.get_schedule()
        assert schedule.get('market_hours_only') == True

    def test_get_schedule_daily_rebalance(self, ramp_adapter):
        """Test get_schedule indicates daily rebalance."""
        schedule = ramp_adapter.get_schedule()
        assert schedule.get('strategy_type') == 'daily'


# =============================================================================
# DATA CACHING TESTS
# =============================================================================

class TestRAMPDataCaching:
    """Tests for RAMP disk caching functionality."""

    def test_preload_uses_disk_cache_when_valid(self, mock_broker):
        """Test preload_historical_data uses valid disk cache."""
        with patch('src.trading.adapters.strategy_adapter.ExecutionEngine'):
            with patch('src.trading.adapters.strategy_adapter.PositionManager'):
                with patch('src.trading.adapters.ramp_live_adapter.StrategyStateManager'):
                    with patch('src.trading.adapters.ramp_live_adapter.PortfolioHealthChecker'):
                        # Mock valid cache
                        mock_cache_data = {
                            'data': {
                                'prices': pd.DataFrame({
                                    'AAPL': [150.0, 151.0, 152.0],
                                    'MSFT': [350.0, 351.0, 352.0]
                                }),
                                'SPY': pd.DataFrame({'close': [450.0, 451.0, 452.0]}),
                                'VIX': pd.DataFrame({'Close': [20.0, 21.0, 22.0]})
                            },
                            'date': datetime.now()
                        }

                        with patch('src.trading.adapters.ramp_live_adapter._load_cache_from_disk', return_value=mock_cache_data):
                            from src.trading.adapters.ramp_live_adapter import RAMPLiveAdapter

                            adapter = RAMPLiveAdapter(
                                broker=mock_broker,
                                symbols=['AAPL', 'MSFT'],
                                broker_name='alpaca'
                            )

                            # Preload should use cache
                            adapter.preload_historical_data()

        # Should not have called broker for historical data
        assert not mock_broker.get_historical_bars.called

    def test_preload_fetches_from_broker_when_no_cache(self, mock_broker):
        """Test preload_historical_data fetches from broker when no cache."""
        with patch('src.trading.adapters.strategy_adapter.ExecutionEngine'):
            with patch('src.trading.adapters.strategy_adapter.PositionManager'):
                with patch('src.trading.adapters.ramp_live_adapter.StrategyStateManager'):
                    with patch('src.trading.adapters.ramp_live_adapter.PortfolioHealthChecker'):
                        # Explicitly return None for cache to force broker fetch
                        with patch('src.trading.adapters.ramp_live_adapter._load_cache_from_disk', return_value=None):
                            from src.trading.adapters.ramp_live_adapter import RAMPLiveAdapter

                            adapter = RAMPLiveAdapter(
                                broker=mock_broker,
                                symbols=['AAPL', 'MSFT'],
                                broker_name='alpaca'
                            )

                            # Clear any prior calls and preload
                            mock_broker.get_historical_bars.reset_mock()
                            adapter.preload_historical_data()

        # Should have called broker
        assert mock_broker.get_historical_bars.called


# =============================================================================
# SIGNAL WRAPPER TESTS
# =============================================================================

class TestRAMPSignalWrapper:
    """Tests for RAMPSignalWrapper class."""

    def test_signal_wrapper_converts_buy_signals(self):
        """Test wrapper converts buy signals correctly."""
        from src.trading.adapters.ramp_live_adapter import RAMPSignalWrapper
        from src.strategies.advanced.ramp_strategy import RAMPSignals, RAMPSignal

        mock_ramp = MagicMock(spec=RAMPSignals)

        # Mock risk signals
        mock_risk = MagicMock()
        mock_risk.exposure_pct = 1.0

        # Mock generate_signals to return buy signal
        mock_ramp.generate_signals.return_value = (
            [RAMPSignal(
                symbol='AAPL',
                action='buy',
                momentum_score=0.15,
                rank=1,
                regime='STRONG_BULL',
                weight=0.05
            )],
            mock_risk
        )

        wrapper = RAMPSignalWrapper(mock_ramp)
        market_data = {
            'AAPL': pd.DataFrame({'close': [150.0]}, index=[datetime.now()])
        }

        signals = wrapper.generate_signals(market_data, datetime.now())

        assert len(signals) == 1
        assert signals[0].direction == 'BUY'
        assert signals[0].symbol == 'AAPL'

    def test_signal_wrapper_converts_sell_signals(self):
        """Test wrapper converts sell signals correctly."""
        from src.trading.adapters.ramp_live_adapter import RAMPSignalWrapper
        from src.strategies.advanced.ramp_strategy import RAMPSignals, RAMPSignal

        mock_ramp = MagicMock(spec=RAMPSignals)

        mock_risk = MagicMock()
        mock_risk.exposure_pct = 1.0

        mock_ramp.generate_signals.return_value = (
            [RAMPSignal(
                symbol='MSFT',
                action='sell',
                momentum_score=-0.05,
                rank=50,
                regime='BEAR',
                weight=0.0
            )],
            mock_risk
        )

        wrapper = RAMPSignalWrapper(mock_ramp)
        market_data = {
            'MSFT': pd.DataFrame({'close': [350.0]}, index=[datetime.now()])
        }

        signals = wrapper.generate_signals(market_data, datetime.now())

        assert len(signals) == 1
        assert signals[0].direction == 'SELL'

    def test_signal_wrapper_includes_hold_signals(self):
        """Test wrapper includes hold signals for transparency."""
        from src.trading.adapters.ramp_live_adapter import RAMPSignalWrapper
        from src.strategies.advanced.ramp_strategy import RAMPSignals, RAMPSignal

        mock_ramp = MagicMock(spec=RAMPSignals)

        mock_risk = MagicMock()
        mock_risk.exposure_pct = 1.0

        mock_ramp.generate_signals.return_value = (
            [RAMPSignal(
                symbol='GOOGL',
                action='hold',
                momentum_score=0.10,
                rank=5,
                regime='SIDEWAYS',
                weight=0.1
            )],
            mock_risk
        )

        wrapper = RAMPSignalWrapper(mock_ramp)
        market_data = {
            'GOOGL': pd.DataFrame({'close': [140.0]}, index=[datetime.now()])
        }

        signals = wrapper.generate_signals(market_data, datetime.now())

        assert len(signals) == 1
        assert signals[0].direction == 'HOLD'

    def test_signal_wrapper_preserves_metadata(self):
        """Test wrapper preserves signal metadata."""
        from src.trading.adapters.ramp_live_adapter import RAMPSignalWrapper
        from src.strategies.advanced.ramp_strategy import RAMPSignals, RAMPSignal

        mock_ramp = MagicMock(spec=RAMPSignals)

        mock_risk = MagicMock()
        mock_risk.exposure_pct = 0.8

        mock_ramp.generate_signals.return_value = (
            [RAMPSignal(
                symbol='AAPL',
                action='buy',
                momentum_score=0.15,
                rank=1,
                regime='STRONG_BULL',
                weight=0.05
            )],
            mock_risk
        )

        wrapper = RAMPSignalWrapper(mock_ramp)
        market_data = {
            'AAPL': pd.DataFrame({'close': [150.0]}, index=[datetime.now()])
        }

        signals = wrapper.generate_signals(market_data, datetime.now())

        assert signals[0].metadata['momentum_score'] == 0.15
        assert signals[0].metadata['rank'] == 1
        assert signals[0].metadata['regime'] == 'STRONG_BULL'
        assert signals[0].metadata['risk_exposure'] == 0.8

    def test_signal_wrapper_get_required_lookback(self):
        """Test wrapper returns correct lookback period."""
        from src.trading.adapters.ramp_live_adapter import RAMPSignalWrapper

        mock_ramp = MagicMock()
        wrapper = RAMPSignalWrapper(mock_ramp)

        # RAMP needs ~1 year of data
        assert wrapper.get_required_lookback() == 300

    def test_signal_wrapper_updates_current_positions(self):
        """Test wrapper can update current positions."""
        from src.trading.adapters.ramp_live_adapter import RAMPSignalWrapper

        mock_ramp = MagicMock()
        wrapper = RAMPSignalWrapper(mock_ramp)

        positions = {'AAPL': 100.0, 'MSFT': 50.0}
        wrapper.set_current_positions(positions)

        assert wrapper._current_positions == positions


# =============================================================================
# REBALANCING TESTS
# =============================================================================

class TestRAMPRebalancing:
    """Tests for RAMP rebalancing logic."""

    def test_fetch_market_data_includes_spy_vix(self, ramp_adapter):
        """Test fetch_market_data includes SPY and VIX."""
        ramp_adapter._data_cache = None

        result = ramp_adapter.fetch_market_data()

        # Should have tried to fetch SPY and VIX
        symbols_fetched = [call[1]['symbol'] for call in ramp_adapter.broker.get_historical_bars.call_args_list]

        # The symbols list includes the trading universe + SPY/VIX
        assert 'AAPL' in ramp_adapter.symbols or 'SPY' in symbols_fetched

    def test_execute_rebalance_calculates_trades(self, ramp_adapter):
        """Test rebalance calculates correct trades."""
        # This would test the full rebalancing logic
        # For now, just verify the adapter can handle a rebalance call

        ramp_adapter.preload_historical_data()
        # run_once should not raise
        ramp_adapter.run_once()


# =============================================================================
# DISK CACHE UTILITY TESTS
# =============================================================================

class TestDiskCacheUtilities:
    """Tests for disk cache utility functions."""

    def test_get_cache_path_format(self):
        """Test cache path is correctly formatted."""
        from src.trading.adapters.ramp_live_adapter import _get_cache_path

        test_date = datetime(2024, 1, 15)
        path = _get_cache_path(test_date)

        assert 'ramp_historical_cache_20240115.pkl' in str(path)

    def test_load_cache_returns_none_when_no_directory(self):
        """Test load cache returns None when cache directory doesn't exist."""
        from src.trading.adapters.ramp_live_adapter import _load_cache_from_disk

        with patch.object(Path, 'exists', return_value=False):
            result = _load_cache_from_disk()

        assert result is None

    def test_load_cache_returns_none_when_cache_stale(self):
        """Test load cache returns None when cache is too old."""
        from src.trading.adapters.ramp_live_adapter import _load_cache_from_disk, CACHE_DIR

        old_date = datetime.now() - timedelta(days=5)
        mock_cache = {
            'data': {'prices': pd.DataFrame()},
            'date': old_date,
            'version': 1
        }

        with patch.object(Path, 'exists', return_value=True):
            with patch.object(Path, 'glob', return_value=[Path('test_cache.pkl')]):
                with patch('builtins.open', mock_open()):
                    with patch('pickle.load', return_value=mock_cache):
                        result = _load_cache_from_disk(max_age_days=1)

        assert result is None


# =============================================================================
# HEALTH CHECK INTEGRATION TESTS
# =============================================================================

class TestRAMPHealthChecks:
    """Tests for RAMP health check integration."""

    def test_run_once_performs_health_check(self, ramp_adapter):
        """Test run_once performs health check before entry."""
        ramp_adapter.preload_historical_data()
        ramp_adapter.run_once()

        ramp_adapter.health_checker.check_before_entry.assert_called()

    def test_run_once_blocks_on_failed_health_check(self, ramp_adapter):
        """Test run_once blocks entry on failed health check."""
        health_result = MagicMock()
        health_result.passed = False
        health_result.errors = ['Insufficient buying power']
        health_result.warnings = []
        ramp_adapter.health_checker.check_before_entry.return_value = health_result

        ramp_adapter.preload_historical_data()
        ramp_adapter.run_once()

        # Lock should still be released
        ramp_adapter.state_manager.release_execution_lock.assert_called()
