"""Integration test: MomentumLiveAdapter emits a DecisionRecord."""
from unittest.mock import MagicMock
import pandas as pd

import pytest

from src.trading.adapters.momentum_live_adapter import MomentumLiveAdapter
from src.trading.decision_log.reader import latest


@pytest.fixture
def mp_adapter(tmp_decisions_dir):
    broker = MagicMock()
    broker.get_account.return_value = {"portfolio_value": 100000.0, "buying_power": 100000.0, "cash": 100000.0}
    broker.get_positions.return_value = []

    sm = MagicMock()
    sm.is_enabled.return_value = True
    sm.is_shutdown_requested.return_value = False
    sm.acquire_execution_lock.return_value = True
    sm.get_positions.return_value = {}
    sm.symbol_owned_by_other.return_value = None
    sm.sync_with_broker.return_value = {"removed": [], "added": []}

    adapter = MomentumLiveAdapter.__new__(MomentumLiveAdapter)
    adapter.broker = broker
    adapter.state_manager = sm
    adapter._broker_name = "ibkr"
    adapter.initial_capital = 100000.0
    adapter.symbols = ["AAPL", "MSFT"]
    adapter._data_cache = {
        "prices": pd.DataFrame({"AAPL": [100.0, 101.0], "MSFT": [200.0, 202.0]},
                               index=pd.to_datetime(["2026-04-23", "2026-04-24"])),
        "SPY": pd.DataFrame({"close": [430.0, 431.0]}, index=pd.to_datetime(["2026-04-23", "2026-04-24"])),
    }
    adapter._data_provider = None
    adapter.health_checker = MagicMock()
    adapter.health_checker.check_before_entry.return_value = MagicMock(passed=True, errors=[], warnings=[])
    adapter.strategy = MagicMock()
    adapter.strategy.generate_signals.return_value = []
    adapter.execution_engine = MagicMock()
    adapter._momentum_signals = MagicMock()
    adapter._momentum_signals.calculate_momentum_scores.return_value = pd.Series({"AAPL": 0.05, "MSFT": 0.03})

    # Stub fetch_todays_closes so test does not call broker network methods
    adapter.fetch_todays_closes = MagicMock(return_value=True)
    # Stub fetch_market_data to return minimal dict
    adapter.fetch_market_data = MagicMock(return_value={
        "AAPL": pd.DataFrame({"close": [100.0, 101.0]}, index=pd.to_datetime(["2026-04-23", "2026-04-24"])),
        "MSFT": pd.DataFrame({"close": [200.0, 202.0]}, index=pd.to_datetime(["2026-04-23", "2026-04-24"])),
    })
    return adapter


def test_run_once_emits_record(mp_adapter):
    mp_adapter.run_once()
    rec = latest("mp")
    assert rec is not None
    assert rec.strategy == "mp"
