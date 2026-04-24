"""Tests for the writer.

Verifies atomicity guarantees, _latest/ snapshot updates, and
retention cleanup. Uses tmp_decisions_dir fixture from conftest.
"""
import json
from datetime import date, timedelta
from pathlib import Path

import pytest

from src.trading.decision_log import paths
from src.trading.decision_log.record import (
    DecisionRecord, TriggerInfo, PreconditionResults, GateResult,
    StrategyInputs, RunMetadata,
)
from src.trading.decision_log.writer import append, _cleanup_old_files


def _record(strategy="ramp", timestamp="2026-04-24T15:55:56-04:00", decision_id="abc-1"):
    return DecisionRecord(
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
        preconditions=PreconditionResults.empty(),
        inputs=StrategyInputs.empty(),
        logic_decisions=None,
        executions=[],
        post_state=None,
        error=None,
        metadata=RunMetadata(
            broker_name="ibkr", data_provider="alpaca-market-iex", git_sha="9533ec0",
            initial_capital_usd=100000.0, strategy_version=1, process_pid=1, hostname="h",
        ),
    )


class TestAppend:
    def test_append_creates_file_for_strategy_and_date(self, tmp_decisions_dir):
        rec = _record(strategy="ramp", timestamp="2026-04-24T15:55:56-04:00")
        append(rec)

        # Day extracted from timestamp -> ramp_20260424.jsonl
        target = tmp_decisions_dir / "ramp_20260424.jsonl"
        assert target.exists()
        line = target.read_text()
        assert line.count("\n") == 1
        assert json.loads(line.strip())["decision_id"] == "abc-1"

    def test_append_appends_to_existing_file(self, tmp_decisions_dir):
        rec1 = _record(decision_id="a")
        rec2 = _record(decision_id="b")
        append(rec1)
        append(rec2)
        target = tmp_decisions_dir / "ramp_20260424.jsonl"
        lines = target.read_text().strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0])["decision_id"] == "a"
        assert json.loads(lines[1])["decision_id"] == "b"

    def test_append_separate_strategies_separate_files(self, tmp_decisions_dir):
        append(_record(strategy="ramp", decision_id="r1"))
        append(_record(strategy="omr", decision_id="o1"))
        assert (tmp_decisions_dir / "ramp_20260424.jsonl").exists()
        assert (tmp_decisions_dir / "omr_20260424.jsonl").exists()

    def test_append_separate_dates_separate_files(self, tmp_decisions_dir):
        append(_record(timestamp="2026-04-24T15:55:56-04:00", decision_id="d1"))
        append(_record(timestamp="2026-04-25T15:55:56-04:00", decision_id="d2"))
        assert (tmp_decisions_dir / "ramp_20260424.jsonl").exists()
        assert (tmp_decisions_dir / "ramp_20260425.jsonl").exists()


class TestLatestSnapshot:
    def test_append_writes_latest_snapshot(self, tmp_decisions_dir):
        rec = _record(strategy="ramp", decision_id="latest-1")
        append(rec)
        latest = tmp_decisions_dir / "_latest" / "ramp.json"
        assert latest.exists()
        loaded = json.loads(latest.read_text())
        assert loaded["decision_id"] == "latest-1"

    def test_append_overwrites_latest_snapshot(self, tmp_decisions_dir):
        append(_record(decision_id="first"))
        append(_record(decision_id="second"))
        latest = tmp_decisions_dir / "_latest" / "ramp.json"
        assert json.loads(latest.read_text())["decision_id"] == "second"


class TestRetention:
    def test_cleanup_removes_files_older_than_retention(self, tmp_decisions_dir):
        # Create old + new files
        old = tmp_decisions_dir / "ramp_20240101.jsonl"
        new = tmp_decisions_dir / "ramp_20260424.jsonl"
        old.write_text('{"x":1}\n')
        new.write_text('{"y":2}\n')

        _cleanup_old_files(retention_days=30, today=date(2026, 4, 24))

        assert not old.exists(), "old file beyond retention should be deleted"
        assert new.exists(), "recent file must remain"

    def test_cleanup_keeps_files_within_retention(self, tmp_decisions_dir):
        recent = tmp_decisions_dir / "ramp_20260420.jsonl"
        recent.write_text('{"x":1}\n')
        _cleanup_old_files(retention_days=30, today=date(2026, 4, 24))
        assert recent.exists()

    def test_cleanup_ignores_latest_dir(self, tmp_decisions_dir):
        # _latest/ subdir must not be cleaned
        latest_file = tmp_decisions_dir / "_latest" / "ramp.json"
        latest_file.write_text('{"x":1}\n')
        _cleanup_old_files(retention_days=1, today=date(2030, 1, 1))
        assert latest_file.exists()


class TestErrorRecord:
    def test_append_writes_record_with_error_populated(self, tmp_decisions_dir):
        from src.trading.decision_log.record import ErrorInfo
        rec = _record(decision_id="errored")
        rec.error = ErrorInfo(
            type="ValueError", message="bad", traceback="...", stage="logic",
        )
        append(rec)
        target = tmp_decisions_dir / "ramp_20260424.jsonl"
        loaded = json.loads(target.read_text().strip())
        assert loaded["error"]["stage"] == "logic"
