"""Integration test: OMRLiveAdapter emits records for both entry and exit.

The exit record's parent_decision_id should match the prior entry's decision_id.

OMR shape note: run_once() accepts action="entry" (default, 15:50 ET) or
action="exit" (09:31 ET).  Before this task the entry path lived in run_once()
and the exit path lived in close_overnight_positions().  Task 10 unified them
under run_once(action=...) and added decision-log emission to both paths.
"""
from unittest.mock import MagicMock
import pytest

from src.trading.adapters.omr_live_adapter import OMRLiveAdapter
from src.trading.decision_log.reader import latest


@pytest.fixture
def omr_adapter(tmp_decisions_dir):
    broker = MagicMock()
    broker.get_account.return_value = {
        "portfolio_value": 100000.0,
        "buying_power": 100000.0,
        "cash": 100000.0,
    }
    broker.get_positions.return_value = []
    broker.get_latest_quote.return_value = {"bid": 100.0, "ask": 100.05}

    sm = MagicMock()
    sm.is_enabled.return_value = True
    sm.is_shutdown_requested.return_value = False
    sm.acquire_execution_lock.return_value = True
    sm.get_positions.return_value = {}
    sm.symbol_owned_by_other.return_value = None
    sm.sync_with_broker.return_value = {"removed": [], "added": []}

    adapter = OMRLiveAdapter.__new__(OMRLiveAdapter)
    adapter.broker = broker
    adapter.state_manager = sm
    adapter._broker_name = "ibkr"
    adapter.initial_capital = 100000.0
    adapter.symbols = ["TQQQ", "SPXL"]
    adapter._omr_signals = MagicMock()
    adapter._omr_signals.generate_signals.return_value = []
    adapter._data_provider = None
    adapter._data_cache = {}
    adapter._intraday_cache = None
    adapter._last_entry_decision_id = None

    adapter.health_checker = MagicMock()
    adapter.health_checker.check_before_entry.return_value = MagicMock(
        passed=True, errors=[], warnings=[]
    )
    adapter.health_checker.check_before_exit.return_value = MagicMock(
        passed=True, errors=[], warnings=[]
    )

    adapter.strategy = MagicMock()
    adapter.strategy.generate_signals.return_value = []

    adapter.execution_engine = MagicMock()
    adapter.execution_engine.execute_order.return_value = None

    return adapter


def test_entry_run_emits_record_with_kind_entry(omr_adapter):
    omr_adapter.run_once(action="entry")
    rec = latest("omr")
    assert rec is not None
    assert rec.trigger.kind == "scheduled_entry"


def test_exit_run_emits_record_with_kind_exit(omr_adapter):
    omr_adapter.run_once(action="exit")
    rec = latest("omr")
    assert rec is not None
    assert rec.trigger.kind == "scheduled_exit"


def test_exit_record_links_to_prior_entry(omr_adapter):
    omr_adapter.run_once(action="entry")
    entry_rec = latest("omr")
    entry_id = entry_rec.decision_id

    omr_adapter.run_once(action="exit")
    exit_rec = latest("omr")
    assert exit_rec.parent_decision_id == entry_id
