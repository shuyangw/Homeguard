"""Tests for the decision-log helpers added to StrategyAdapter base.

These cover _begin_decision, _stage, _check_common_preconditions, and
_write_decision -- helpers used by all stock-strategy adapters.
"""
from unittest.mock import MagicMock, patch

import pytest

from src.trading.adapters.strategy_adapter import StrategyAdapter
from src.trading.decision_log.record import DecisionRecord, GateResult


class _MinimalAdapter(StrategyAdapter):
    STRATEGY_NAME = "test_strat"

    def __init__(self, broker, state_manager):
        self.broker = broker
        self.state_manager = state_manager
        self._broker_name = "ibkr"
        self.initial_capital = 100000.0
        self.STRATEGY_VERSION = 1

    def get_schedule(self):
        return {"interval": "1d", "market_hours_only": True}


@pytest.fixture
def adapter():
    broker = MagicMock()
    sm = MagicMock()
    sm.is_enabled.return_value = True
    sm.is_shutdown_requested.return_value = False
    sm.acquire_execution_lock.return_value = True
    return _MinimalAdapter(broker, sm)


class TestBeginDecision:
    def test_returns_record_with_correct_strategy(self, adapter):
        rec = adapter._begin_decision("scheduled_rebalance", schedule_time="15:55")
        assert rec.strategy == "test_strat"
        assert rec.trigger.kind == "scheduled_rebalance"
        assert rec.trigger.schedule_time == "15:55"

    def test_record_carries_initial_capital(self, adapter):
        rec = adapter._begin_decision("scheduled_rebalance")
        assert rec.metadata.initial_capital_usd == 100000.0

    def test_record_carries_broker_name(self, adapter):
        rec = adapter._begin_decision("scheduled_rebalance")
        assert rec.metadata.broker_name == "ibkr"


class TestCommonPreconditions:
    def test_all_pass_when_state_clean(self, adapter):
        rec = adapter._begin_decision("scheduled_rebalance")
        ok = adapter._check_common_preconditions(rec)
        assert ok is True
        assert rec.preconditions.strategy_enabled.passed is True
        assert rec.preconditions.execution_lock_acquired.passed is True

    def test_strategy_disabled_blocks(self, adapter):
        adapter.state_manager.is_enabled.return_value = False
        rec = adapter._begin_decision("scheduled_rebalance")
        ok = adapter._check_common_preconditions(rec)
        assert ok is False
        assert rec.preconditions.strategy_enabled.passed is False

    def test_lock_unavailable_blocks(self, adapter):
        adapter.state_manager.acquire_execution_lock.return_value = False
        rec = adapter._begin_decision("scheduled_rebalance")
        ok = adapter._check_common_preconditions(rec)
        assert ok is False


class TestWriteDecision:
    def test_writes_record_to_disk(self, adapter, tmp_path, monkeypatch):
        from src.trading.decision_log import paths as paths_mod
        monkeypatch.setattr(paths_mod, "decisions_dir", lambda: tmp_path)
        latest = tmp_path / "_latest"
        latest.mkdir()
        monkeypatch.setattr(paths_mod, "latest_dir", lambda: latest)

        rec = adapter._begin_decision("scheduled_rebalance", schedule_time="15:55")
        adapter._write_decision(rec)

        # File should exist for today
        files = list(tmp_path.glob("test_strat_*.jsonl"))
        assert len(files) == 1


class TestStageContextManager:
    def test_stage_sets_current_stage_attr(self, adapter):
        rec = adapter._begin_decision("scheduled_rebalance")
        with adapter._stage(rec, "inputs"):
            assert getattr(rec, "_current_stage", None) == "inputs"
        # After exit, stage stays set so a later exception capture sees it
        assert getattr(rec, "_current_stage", None) == "inputs"
