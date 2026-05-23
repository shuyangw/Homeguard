"""
TDD tests for RAMPLiveAdapter._position_open_dates bookkeeping.

Mirrors the reference logic in src/research/ramp_phase4/engine.py::apply_trades:
- Set on 0 -> n>0 transition (new open).
- Preserved on top-ups (existing position grows; original open date stays).
- Cleared on n -> 0 transition (full exit).
"""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from src.trading.adapters.ramp_live_adapter import (
    RAMPLiveAdapter,
    update_position_open_dates,
)


@pytest.fixture
def mock_broker():
    broker = MagicMock()
    broker.get_account.return_value = {
        'buying_power': 100000.0,
        'portfolio_value': 100000.0,
        'cash': 100000.0,
    }
    broker.get_positions.return_value = []
    broker.is_market_open.return_value = True
    return broker


@pytest.fixture
def ramp_adapter(mock_broker):
    with patch('src.trading.adapters.strategy_adapter.ExecutionEngine'):
        with patch('src.trading.adapters.strategy_adapter.PositionManager'):
            with patch('src.trading.adapters.ramp_live_adapter.StrategyStateManager') as mock_sm:
                with patch('src.trading.adapters.ramp_live_adapter.PortfolioHealthChecker'):
                    with patch('src.trading.adapters.ramp_live_adapter._load_cache_from_disk') as mock_cache:
                        mock_cache.return_value = None
                        mock_sm_instance = MagicMock()
                        mock_sm_instance.is_enabled.return_value = True
                        mock_sm_instance.is_shutdown_requested.return_value = False
                        mock_sm_instance.acquire_execution_lock.return_value = True
                        mock_sm_instance.get_positions.return_value = {}
                        mock_sm_instance.get_runner_session_state.return_value = {
                            'peak_equity_usd': None,
                            'session_open_equity_usd': None,
                            'session_open_date': None,
                            'position_open_dates': None,
                        }
                        mock_sm.return_value = mock_sm_instance

                        adapter = RAMPLiveAdapter(
                            broker=mock_broker,
                            symbols=['AAPL', 'MSFT', 'GOOGL'],
                            max_capital_allocation=1.0,
                            broker_name='alpaca',
                        )
                        adapter.state_manager = mock_sm_instance
                        return adapter


class TestUpdatePositionOpenDatesHelper:
    """Pure helper used by the live adapter to maintain _position_open_dates."""

    def test_new_position_sets_open_date(self):
        previous_positions = {}
        new_positions = {'AAPL': 100}
        previous_open_dates = {}
        current_date = datetime(2026, 1, 15)

        result = update_position_open_dates(
            previous_positions=previous_positions,
            new_positions=new_positions,
            previous_open_dates=previous_open_dates,
            current_date=current_date,
        )

        assert result == {'AAPL': datetime(2026, 1, 15)}

    def test_topup_preserves_original_open_date(self):
        previous_positions = {'AAPL': 100}
        new_positions = {'AAPL': 150}
        previous_open_dates = {'AAPL': datetime(2026, 1, 10)}
        current_date = datetime(2026, 1, 15)

        result = update_position_open_dates(
            previous_positions=previous_positions,
            new_positions=new_positions,
            previous_open_dates=previous_open_dates,
            current_date=current_date,
        )

        assert result == {'AAPL': datetime(2026, 1, 10)}

    def test_full_exit_clears_open_date(self):
        previous_positions = {'AAPL': 100}
        new_positions = {}
        previous_open_dates = {'AAPL': datetime(2026, 1, 10)}
        current_date = datetime(2026, 1, 15)

        result = update_position_open_dates(
            previous_positions=previous_positions,
            new_positions=new_positions,
            previous_open_dates=previous_open_dates,
            current_date=current_date,
        )

        assert result == {}


class TestRAMPLiveAdapterPositionOpenDates:
    """Adapter-level tests: field initialization, transitions, persistence."""

    def test_position_open_dates_initialized_empty(self, ramp_adapter):
        assert hasattr(ramp_adapter, '_position_open_dates')
        assert ramp_adapter._position_open_dates == {}

    def test_position_open_dates_set_on_new_position(self, ramp_adapter):
        ramp_adapter._position_open_dates = {}
        ramp_adapter._refresh_position_open_dates(
            previous_positions={},
            new_positions={'AAPL': 100},
            current_date=datetime(2026, 1, 15),
        )
        assert ramp_adapter._position_open_dates == {'AAPL': datetime(2026, 1, 15)}

    def test_position_open_dates_preserved_on_topup(self, ramp_adapter):
        ramp_adapter._position_open_dates = {'AAPL': datetime(2026, 1, 10)}
        ramp_adapter._refresh_position_open_dates(
            previous_positions={'AAPL': 100},
            new_positions={'AAPL': 150},
            current_date=datetime(2026, 1, 15),
        )
        assert ramp_adapter._position_open_dates == {'AAPL': datetime(2026, 1, 10)}

    def test_position_open_dates_cleared_on_full_exit(self, ramp_adapter):
        ramp_adapter._position_open_dates = {'AAPL': datetime(2026, 1, 10)}
        ramp_adapter._refresh_position_open_dates(
            previous_positions={'AAPL': 100},
            new_positions={},
            current_date=datetime(2026, 1, 15),
        )
        assert ramp_adapter._position_open_dates == {}

    def test_position_open_dates_persisted_to_state_manager(self, ramp_adapter):
        ramp_adapter._position_open_dates = {'AAPL': datetime(2026, 1, 10)}
        ramp_adapter._refresh_position_open_dates(
            previous_positions={'AAPL': 100},
            new_positions={'AAPL': 100, 'MSFT': 50},
            current_date=datetime(2026, 1, 15),
        )

        assert ramp_adapter.state_manager.update_runner_session_state.called
        kwargs = ramp_adapter.state_manager.update_runner_session_state.call_args.kwargs
        assert kwargs.get('position_open_dates') == {
            'AAPL': '2026-01-10T00:00:00',
            'MSFT': '2026-01-15T00:00:00',
        }

    def test_position_open_dates_loaded_from_state_manager(self, mock_broker):
        with patch('src.trading.adapters.strategy_adapter.ExecutionEngine'):
            with patch('src.trading.adapters.strategy_adapter.PositionManager'):
                with patch('src.trading.adapters.ramp_live_adapter.StrategyStateManager') as mock_sm:
                    with patch('src.trading.adapters.ramp_live_adapter.PortfolioHealthChecker'):
                        with patch('src.trading.adapters.ramp_live_adapter._load_cache_from_disk') as mock_cache:
                            mock_cache.return_value = None
                            mock_sm_instance = MagicMock()
                            mock_sm_instance.is_enabled.return_value = True
                            mock_sm_instance.is_shutdown_requested.return_value = False
                            mock_sm_instance.acquire_execution_lock.return_value = True
                            mock_sm_instance.get_positions.return_value = {}
                            mock_sm_instance.get_runner_session_state.return_value = {
                                'peak_equity_usd': None,
                                'session_open_equity_usd': None,
                                'session_open_date': None,
                                'position_open_dates': {
                                    'AAPL': '2026-01-10T00:00:00',
                                    'MSFT': '2026-01-12T00:00:00',
                                },
                            }
                            mock_sm.return_value = mock_sm_instance

                            adapter = RAMPLiveAdapter(
                                broker=mock_broker,
                                symbols=['AAPL', 'MSFT', 'GOOGL'],
                                max_capital_allocation=1.0,
                                broker_name='alpaca',
                            )

                            assert adapter._position_open_dates == {
                                'AAPL': datetime(2026, 1, 10),
                                'MSFT': datetime(2026, 1, 12),
                            }
