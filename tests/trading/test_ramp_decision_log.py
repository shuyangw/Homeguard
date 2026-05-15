"""Integration test: RAMPLiveAdapter.run_once() emits a complete DecisionRecord."""
from unittest.mock import MagicMock, patch
import pandas as pd

import pytest

from src.trading.adapters.ramp_live_adapter import RAMPLiveAdapter
from src.trading.decision_log.record import DecisionRecord
from src.trading.decision_log.reader import latest


@pytest.fixture
def ramp_adapter(tmp_decisions_dir, monkeypatch):
    """RAMPLiveAdapter with mock broker, state_manager, and pre-loaded cache."""
    broker = MagicMock()
    broker.get_account.return_value = {
        "portfolio_value": 1014176.25,
        "buying_power": 4049365.68,
        "cash": 1012341.42,
    }
    broker.get_positions.return_value = []
    broker.get_latest_quote.return_value = {"bid": 100.0, "ask": 100.05}

    sm = MagicMock()
    sm.is_enabled.return_value = True
    sm.is_shutdown_requested.return_value = False
    sm.acquire_execution_lock.return_value = True
    sm.get_positions.return_value = {}
    sm.symbol_owned_by_other.return_value = None

    adapter = RAMPLiveAdapter.__new__(RAMPLiveAdapter)
    adapter.broker = broker
    adapter.state_manager = sm
    adapter._broker_name = "ibkr"
    adapter.initial_capital = 100000.0
    adapter.max_capital_allocation = 1.0
    adapter.symbols = ["AAPL", "MSFT"]
    adapter._data_provider = None
    adapter._data_cache = {
        "prices": pd.DataFrame(
            {"AAPL": [100.0, 101.0], "MSFT": [200.0, 202.0]},
            index=pd.to_datetime(["2026-04-23", "2026-04-24"]),
        ),
        "SPY": pd.DataFrame({"close": [430.0, 431.0]}, index=pd.to_datetime(["2026-04-23", "2026-04-24"])),
        "VIX": pd.DataFrame({"Close": [14.0, 15.0]}, index=pd.to_datetime(["2026-04-23", "2026-04-24"])),
    }
    adapter._cache_date = None

    # Mock signal generator
    adapter._ramp_signals = MagicMock()
    adapter._ramp_signals.generate_signals.return_value = ([], MagicMock(
        regime="WEAK_BULL", regime_confidence=0.75, exposure_pct=1.0,
        params={"top_n": 10},
    ))
    adapter._ramp_signals.detect_regime.return_value = ("WEAK_BULL", 0.75)
    adapter._ramp_signals.current_top_n = 10

    # Mock health checker
    adapter.health_checker = MagicMock()
    adapter.health_checker.check_before_entry.return_value = MagicMock(
        passed=True, errors=[], warnings=[],
    )

    adapter.strategy = MagicMock()
    adapter.strategy.set_current_positions = MagicMock()
    adapter.strategy.generate_signals.return_value = []

    adapter.execution_engine = MagicMock()
    adapter.execution_engine.execute_order.return_value = None

    return adapter


def test_run_once_emits_decision_record(ramp_adapter):
    ramp_adapter.run_once()
    rec = latest("ramp")
    assert rec is not None
    assert rec.strategy == "ramp"
    assert rec.trigger.kind == "scheduled_rebalance"


def test_run_once_records_preconditions_pass(ramp_adapter):
    ramp_adapter.run_once()
    rec = latest("ramp")
    assert rec.preconditions.all_passed is True


def test_run_once_records_inputs(ramp_adapter):
    ramp_adapter.run_once()
    rec = latest("ramp")
    assert rec.inputs.regime == "WEAK_BULL"
    assert rec.inputs.regime_confidence == 0.75


def test_run_once_writes_record_on_exception(ramp_adapter, monkeypatch):
    """Even if an exception fires mid-rebalance, the record must be written."""
    # Inject a failure during inputs stage
    ramp_adapter.fetch_todays_closes = MagicMock(side_effect=ValueError("test boom"))

    with pytest.raises(Exception):
        ramp_adapter.run_once()

    rec = latest("ramp")
    assert rec is not None, "record must be written even when run_once raises"
    assert rec.error is not None
    assert "boom" in rec.error.message


def test_run_once_records_blocked_when_health_check_fails(ramp_adapter):
    ramp_adapter.health_checker.check_before_entry.return_value = MagicMock(
        passed=False, errors=["missing get_open_orders"], warnings=[],
    )
    ramp_adapter.run_once()
    rec = latest("ramp")
    assert rec.preconditions.health_check.passed is False
    assert rec.executions == []


# ---------------------------------------------------------------------------
# F5 schema enrichment tests (SCHEMA_VERSION 2)
# ---------------------------------------------------------------------------

from src.trading.decision_log.record import SCHEMA_VERSION, StrategyInputs, LogicDecisions  # noqa: E402


class TestF5SchemaEnrichment:
    def test_schema_version_is_2(self):
        assert SCHEMA_VERSION == 2

    def test_strategy_inputs_has_new_f5_fields(self):
        si = StrategyInputs()
        assert si.regime_scores == {}
        assert si.vix_percentile is None
        assert si.breadth_pct_above_50d is None
        assert si.exposure_multiplier is None
        assert si.fallback_mode_used is None

    def test_strategy_inputs_accepts_new_field_values(self):
        si = StrategyInputs(
            regime="STRONG_BULL",
            regime_scores={
                "STRONG_BULL": 0.9,
                "WEAK_BULL": 0.05,
                "SIDEWAYS": 0.02,
                "UNPREDICTABLE": 0.02,
                "BEAR": 0.01,
            },
            vix_percentile=0.45,
            breadth_pct_above_50d=0.62,
            exposure_multiplier=1.0,
            fallback_mode_used=None,
        )
        assert si.regime_scores["STRONG_BULL"] == 0.9
        assert si.vix_percentile == 0.45
        assert si.breadth_pct_above_50d == 0.62

    def test_logic_decisions_has_new_f5_fields(self):
        ld = LogicDecisions(
            top_n=10,
            target_symbols=["AAPL", "MSFT"],
            target_weights={"AAPL": 0.1, "MSFT": 0.1},
            target_value_usd={"AAPL": 10000.0, "MSFT": 10000.0},
            reduce_exposure=False,
            exposure_pct=1.0,
            exit_signals=[],
            hold_signals=[],
            skip_reasons={},
        )
        assert ld.realized_weights == {}
        assert ld.realized_turnover_usd is None
        assert ld.expected_cost_bps is None
        assert ld.actual_cost_usd is None
        assert ld.cash_after_rebalance_usd is None
