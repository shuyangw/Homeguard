"""Tests for the reader API: latest, by_id, for_date, iter_records, filter, summary."""
from datetime import date
import json
from pathlib import Path

import pytest

from src.trading.decision_log import paths
from src.trading.decision_log.record import (
    DecisionRecord, TriggerInfo, PreconditionResults, GateResult,
    StrategyInputs, LogicDecisions, RunMetadata,
)
from src.trading.decision_log.writer import append
from src.trading.decision_log.reader import (
    latest, by_id, for_date, iter_records, filter_records, summary,
)


def _make(strategy="ramp", decision_id="abc", timestamp="2026-04-24T15:55:56-04:00",
          regime=None, executions_count=0, all_passed=True, error_stage=None):
    from src.trading.decision_log.record import (
        Execution, ErrorInfo,
    )
    rec = DecisionRecord(
        schema_version=1,
        decision_id=decision_id,
        strategy=strategy,
        timestamp=timestamp,
        trigger=TriggerInfo(
            kind="scheduled_rebalance",
            schedule_time="15:55",
            actual_fire_time=timestamp,
            delay_seconds=0.0,
            consecutive_fire_count=1,
        ),
        preconditions=PreconditionResults(
            all_passed=all_passed,
            strategy_enabled=GateResult(passed=True, details={}),
            shutdown_requested=GateResult(passed=True, details={}),
            execution_lock_acquired=GateResult(passed=True, details={}),
            health_check=GateResult(passed=all_passed, details={}),
            data_freshness=GateResult(passed=True, details={}),
            extra={},
        ),
        inputs=StrategyInputs.empty(),
        logic_decisions=None,
        executions=[
            Execution(
                symbol=f"S{i}", action="buy", target_qty=1, target_value_usd=10.0,
                status="filled", fill_price=10.0,
            )
            for i in range(executions_count)
        ],
        post_state=None,
        error=ErrorInfo("ValueError", "x", "...", error_stage) if error_stage else None,
        metadata=RunMetadata(
            broker_name="ibkr", data_provider="alpaca-market-iex", git_sha="abc",
            initial_capital_usd=100000.0, strategy_version=1, process_pid=1, hostname="h",
        ),
    )
    if regime is not None:
        rec.inputs.regime = regime
    return rec


class TestLatest:
    def test_latest_reads_from_latest_dir(self, tmp_decisions_dir):
        # regime set -> substantive record (a blocked/empty record no longer
        # updates _latest; see writer._is_substantive).
        append(_make(decision_id="newest", regime="STRONG_BULL"))
        rec = latest("ramp")
        assert rec is not None
        assert rec.decision_id == "newest"

    def test_latest_returns_none_when_no_records(self, tmp_decisions_dir):
        assert latest("ramp") is None


class TestByID:
    def test_by_id_finds_record_in_correct_day_file(self, tmp_decisions_dir):
        append(_make(decision_id="d1", timestamp="2026-04-24T15:55:56-04:00"))
        append(_make(decision_id="d2", timestamp="2026-04-25T15:55:56-04:00"))
        found = by_id("d2")
        assert found is not None
        assert found.decision_id == "d2"

    def test_by_id_returns_none_for_missing(self, tmp_decisions_dir):
        append(_make(decision_id="real"))
        assert by_id("nonexistent") is None


class TestForDate:
    def test_for_date_returns_records_for_that_day(self, tmp_decisions_dir):
        append(_make(decision_id="a", timestamp="2026-04-24T15:55:56-04:00"))
        append(_make(decision_id="b", timestamp="2026-04-24T15:56:00-04:00"))
        append(_make(decision_id="c", timestamp="2026-04-25T15:55:56-04:00"))
        records = for_date("ramp", date(2026, 4, 24))
        ids = [r.decision_id for r in records]
        assert ids == ["a", "b"]

    def test_for_date_empty_for_unknown_day(self, tmp_decisions_dir):
        assert for_date("ramp", date(2024, 1, 1)) == []


class TestIterRecords:
    def test_iter_records_yields_lazily_across_days(self, tmp_decisions_dir):
        for i, ts in enumerate([
            "2026-04-22T15:55:56-04:00",
            "2026-04-23T15:55:56-04:00",
            "2026-04-24T15:55:56-04:00",
        ]):
            append(_make(decision_id=f"d{i}", timestamp=ts))
        records = list(iter_records("ramp", since=date(2026, 4, 22), until=date(2026, 4, 24)))
        assert [r.decision_id for r in records] == ["d0", "d1", "d2"]

    def test_iter_records_skips_corrupt_lines(self, tmp_decisions_dir):
        target = tmp_decisions_dir / "ramp_20260424.jsonl"
        target.write_text(
            '{"not": "a valid record"}\n'  # invalid: missing required keys
            + _make(decision_id="good").to_jsonl_line()
        )
        records = list(iter_records("ramp", since=date(2026, 4, 24), until=date(2026, 4, 24)))
        assert len(records) == 1
        assert records[0].decision_id == "good"


class TestFilterDSL:
    def test_filter_by_regime(self, tmp_decisions_dir):
        append(_make(decision_id="bull", regime="WEAK_BULL"))
        append(_make(decision_id="bear", regime="BEAR"))
        records = list(iter_records("ramp", since=date(2026, 4, 24), until=date(2026, 4, 24)))
        bears = list(filter_records(records, "regime=BEAR"))
        assert [r.decision_id for r in bears] == ["bear"]

    def test_filter_by_executions_length_zero(self, tmp_decisions_dir):
        append(_make(decision_id="empty", executions_count=0))
        append(_make(decision_id="full", executions_count=3))
        records = list(iter_records("ramp", since=date(2026, 4, 24), until=date(2026, 4, 24)))
        empty = list(filter_records(records, "executions:length=0"))
        assert [r.decision_id for r in empty] == ["empty"]

    def test_filter_by_error_stage(self, tmp_decisions_dir):
        append(_make(decision_id="ok"))
        append(_make(decision_id="errored", error_stage="logic"))
        records = list(iter_records("ramp", since=date(2026, 4, 24), until=date(2026, 4, 24)))
        with_error = list(filter_records(records, "error.stage=logic"))
        assert [r.decision_id for r in with_error] == ["errored"]

    def test_filter_by_preconditions_all_passed(self, tmp_decisions_dir):
        append(_make(decision_id="pass", all_passed=True))
        append(_make(decision_id="fail", all_passed=False))
        records = list(iter_records("ramp", since=date(2026, 4, 24), until=date(2026, 4, 24)))
        blocked = list(filter_records(records, "preconditions.all_passed=false"))
        assert [r.decision_id for r in blocked] == ["fail"]


class TestSummary:
    def test_summary_aggregates_per_strategy(self, tmp_decisions_dir):
        append(_make(strategy="ramp", decision_id="r1", executions_count=5))
        append(_make(strategy="ramp", decision_id="r2", error_stage="logic"))
        append(_make(strategy="omr", decision_id="o1", executions_count=2))
        result = summary(start=date(2026, 4, 24), end=date(2026, 4, 24))
        assert result["ramp"].triggers == 2
        assert result["ramp"].errored == 1
        assert result["omr"].triggers == 1
        assert result["omr"].errored == 0


class TestLegacyCSCM:
    def test_legacy_cscm_signal_jsonl_is_translated(self, tmp_decisions_dir, tmp_path, monkeypatch):
        """Old cscm_signals_YYYYMMDD.jsonl files load as DecisionRecord via legacy path."""
        from src.trading.decision_log.reader import load_legacy_cscm

        # Old format -- one signal entry per line
        legacy_dir = tmp_path / "legacy_cscm"
        legacy_dir.mkdir()
        legacy_file = legacy_dir / "cscm_signals_20260420.jsonl"
        legacy_file.write_text(
            '{"timestamp": "2026-04-20T10:00:56-04:00", "type": "signal", '
            '"regime": "bullish", "btc_price": 75230.89, "btc_sma": 70932.83, '
            '"is_rebalance_day": true, "reduce_exposure": false, '
            '"exposure_pct": 1.0, "top_symbols": ["SUSHI/USD", "ETH/USD", "BTC/USD"], '
            '"momentum_scores": {"BTC/USD": 0.06, "ETH/USD": 0.07}}\n'
        )

        records = list(load_legacy_cscm(legacy_file))
        assert len(records) == 1
        rec = records[0]
        assert rec.strategy == "cscm"
        assert rec.inputs.regime == "bullish"
        assert rec.inputs.extra["btc_price"] == 75230.89
        assert rec.inputs.extra["is_rebalance_day"] is True
        assert rec.inputs.momentum_scores["BTC/USD"] == 0.06

    def test_legacy_cscm_skips_non_signal_entries(self, tmp_path):
        from src.trading.decision_log.reader import load_legacy_cscm
        legacy_file = tmp_path / "cscm_signals_20260420.jsonl"
        legacy_file.write_text(
            '{"timestamp": "2026-04-20T10:00:00-04:00", "type": "heartbeat"}\n'
            '{"timestamp": "2026-04-20T10:01:00-04:00", "type": "signal", '
            '"regime": "bullish", "btc_price": 75000.0, "btc_sma": 70000.0, '
            '"is_rebalance_day": false, "reduce_exposure": false, '
            '"exposure_pct": 1.0, "top_symbols": [], "momentum_scores": {}}\n'
        )
        records = list(load_legacy_cscm(legacy_file))
        assert len(records) == 1
        assert records[0].inputs.regime == "bullish"
