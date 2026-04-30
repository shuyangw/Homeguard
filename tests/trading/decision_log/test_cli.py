"""Tests for the CLI subcommands. Uses fixture jsonl + golden text."""
import io
import json
import shutil
import sys
from datetime import date
from pathlib import Path

import pytest

from src.trading.decision_log import cli
from src.trading.decision_log.writer import append
from src.trading.decision_log.record import DecisionRecord


FIXTURES = Path(__file__).parent / "fixtures"


def _seed_from_fixture(tmp_decisions_dir, fixture_name):
    """Copy fixture jsonl into the temp decisions dir as today's record."""
    src = FIXTURES / fixture_name
    text = src.read_text()
    rec = DecisionRecord.from_jsonl_line(text)
    append(rec)


class TestShow:
    def test_show_latest_ramp_matches_golden(self, tmp_decisions_dir, capsys):
        _seed_from_fixture(tmp_decisions_dir, "sample_ramp_clean.jsonl")
        rc = cli.main(["show", "ramp"])
        assert rc == 0
        out = capsys.readouterr().out
        expected = (FIXTURES / "expected_show_ramp_clean.txt").read_text()
        # Compare line-by-line for clearer diffs
        assert out.strip().split("\n") == expected.strip().split("\n")

    def test_show_with_json_flag_emits_raw_json(self, tmp_decisions_dir, capsys):
        _seed_from_fixture(tmp_decisions_dir, "sample_ramp_clean.jsonl")
        rc = cli.main(["show", "ramp", "--json"])
        assert rc == 0
        out = capsys.readouterr().out
        loaded = json.loads(out)
        assert loaded["decision_id"] == "7c2f9d"

    def test_show_no_records_returns_nonzero(self, tmp_decisions_dir, capsys):
        rc = cli.main(["show", "ramp"])
        assert rc != 0


class TestList:
    def test_list_summary_table(self, tmp_decisions_dir, capsys):
        _seed_from_fixture(tmp_decisions_dir, "sample_ramp_clean.jsonl")
        rc = cli.main(["list", "ramp", "--days", "7"])
        assert rc == 0
        out = capsys.readouterr().out
        # Table header
        assert "DATE" in out and "TRIGGER" in out and "REGIME" in out
        # The seed record's regime should appear
        assert "WEAK_BULL" in out


class TestStatus:
    def test_status_no_records_returns_nonzero(self, tmp_decisions_dir, capsys):
        rc = cli.main(["status"])
        assert rc != 0
        err = capsys.readouterr().err
        assert "No decision records" in err

    def test_status_single_strategy_emits_one_row(self, tmp_decisions_dir, capsys):
        _seed_from_fixture(tmp_decisions_dir, "sample_ramp_clean.jsonl")
        rc = cli.main(["status"])
        assert rc == 0
        out = capsys.readouterr().out
        # Header columns
        assert "STRAT" in out and "LAST FIRE" in out and "REGIME" in out
        assert "EQUITY" in out and "STATUS" in out
        # The seeded record's strategy and regime should appear
        assert "ramp" in out
        assert "WEAK_BULL" in out
        # Clean record -> status "clean"
        assert "clean" in out
        # Equity formatting (initial_capital_usd in fixture is $99,970)
        assert "$99,970" in out

    def test_status_multi_strategy_lists_each(self, tmp_decisions_dir, capsys):
        # Seed one record for ramp; cscm/omr/mp absent -> they should be silently
        # skipped (rather than printed with "-").
        _seed_from_fixture(tmp_decisions_dir, "sample_ramp_clean.jsonl")
        rc = cli.main(["status"])
        assert rc == 0
        out = capsys.readouterr().out
        # Only one data row (header + 1 row -> 2 non-empty lines)
        data_lines = [ln for ln in out.strip().split("\n") if ln.strip()]
        assert len(data_lines) == 2

    def test_status_blocked_record_shows_gates(self, tmp_decisions_dir, capsys):
        # Build a record with a failing precondition and persist via append.
        from src.trading.decision_log.writer import append
        from src.trading.decision_log.record import (
            DecisionRecord, TriggerInfo, PreconditionResults, GateResult,
            StrategyInputs, RunMetadata,
        )
        rec = DecisionRecord(
            schema_version=1,
            decision_id="blocked-1",
            strategy="ramp",
            timestamp="2026-04-29T15:55:00-04:00",
            trigger=TriggerInfo(
                kind="scheduled_rebalance",
                schedule_time="15:55",
                actual_fire_time="2026-04-29T15:55:00-04:00",
                delay_seconds=0.0,
            ),
            preconditions=PreconditionResults(
                all_passed=False,
                strategy_enabled=GateResult(passed=True),
                shutdown_requested=GateResult(passed=True),
                execution_lock_acquired=GateResult(passed=True),
                health_check=GateResult(passed=False, details={}, error="no buying power"),
                data_freshness=GateResult(passed=True),
            ),
            inputs=StrategyInputs(),
            logic_decisions=None,
            executions=[],
            post_state=None,
            error=None,
            metadata=RunMetadata(
                broker_name="ibkr", data_provider="alpaca", git_sha="abc",
                initial_capital_usd=100000.0, strategy_version=1,
                process_pid=1, hostname="test",
            ),
        )
        append(rec)
        rc = cli.main(["status"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "blocked: health_check" in out

    def test_status_filters_by_toggle_when_present(
        self, tmp_decisions_dir, tmp_path, monkeypatch, capsys,
    ):
        # Seed records for ramp AND cscm (and one for "omr" via direct write).
        from src.trading.decision_log.writer import append
        from src.trading.decision_log.record import (
            DecisionRecord, TriggerInfo, PreconditionResults, GateResult,
            StrategyInputs, RunMetadata,
        )

        def _make(strategy: str) -> DecisionRecord:
            return DecisionRecord(
                schema_version=1,
                decision_id=f"{strategy}-1",
                strategy=strategy,
                timestamp="2026-04-29T15:55:00-04:00",
                trigger=TriggerInfo(
                    kind="scheduled_rebalance",
                    schedule_time="15:55",
                    actual_fire_time="2026-04-29T15:55:00-04:00",
                    delay_seconds=0.0,
                ),
                preconditions=PreconditionResults(
                    all_passed=True,
                    strategy_enabled=GateResult(passed=True),
                    shutdown_requested=GateResult(passed=True),
                    execution_lock_acquired=GateResult(passed=True),
                    health_check=GateResult(passed=True),
                    data_freshness=GateResult(passed=True),
                ),
                inputs=StrategyInputs(),
                logic_decisions=None,
                executions=[],
                post_state=None,
                error=None,
                metadata=RunMetadata(
                    broker_name="ibkr", data_provider="alpaca", git_sha="abc",
                    initial_capital_usd=100000.0, strategy_version=1,
                    process_pid=1, hostname="test",
                ),
            )

        for s in ("ramp", "cscm", "omr", "mp"):
            append(_make(s))

        # Stub a toggle file with only ramp + cscm enabled.
        toggle_yaml = tmp_path / "strategy_toggle.yaml"
        toggle_yaml.write_text(
            "strategies:\n"
            "  ramp:\n    enabled: true\n    shutdown_requested: false\n"
            "  cscm:\n    enabled: true\n    shutdown_requested: false\n"
            "  omr:\n    enabled: false\n    shutdown_requested: false\n"
            "  mp:\n    enabled: false\n    shutdown_requested: false\n"
        )
        # Patch the toggle resolver: omr & mp are disabled in the toggle
        # (and governed by it). CSCM is also "false" in the toggle but
        # NOT governed by it (separate service), so it must still show.
        monkeypatch.setattr(cli, "_disabled_strategies", lambda: {"omr", "mp"})

        rc = cli.main(["status"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "ramp" in out and "cscm" in out
        # Disabled toggle-governed strategies should NOT appear by default.
        assert "omr" not in out and " mp " not in out

    def test_status_all_flag_includes_disabled(
        self, tmp_decisions_dir, monkeypatch, capsys,
    ):
        from src.trading.decision_log.writer import append
        from src.trading.decision_log.record import (
            DecisionRecord, TriggerInfo, PreconditionResults, GateResult,
            StrategyInputs, RunMetadata,
        )

        def _make(strategy: str) -> DecisionRecord:
            return DecisionRecord(
                schema_version=1, decision_id=f"{strategy}-1", strategy=strategy,
                timestamp="2026-04-29T15:55:00-04:00",
                trigger=TriggerInfo(
                    kind="scheduled_rebalance", schedule_time="15:55",
                    actual_fire_time="2026-04-29T15:55:00-04:00", delay_seconds=0.0,
                ),
                preconditions=PreconditionResults(
                    all_passed=True,
                    strategy_enabled=GateResult(passed=True),
                    shutdown_requested=GateResult(passed=True),
                    execution_lock_acquired=GateResult(passed=True),
                    health_check=GateResult(passed=True),
                    data_freshness=GateResult(passed=True),
                ),
                inputs=StrategyInputs(),
                logic_decisions=None, executions=[], post_state=None, error=None,
                metadata=RunMetadata(
                    broker_name="ibkr", data_provider="alpaca", git_sha="abc",
                    initial_capital_usd=100000.0, strategy_version=1,
                    process_pid=1, hostname="test",
                ),
            )

        for s in ("ramp", "omr"):
            append(_make(s))

        # Toggle says omr is disabled; without --all, omr would be hidden.
        monkeypatch.setattr(cli, "_disabled_strategies", lambda: {"omr"})

        rc = cli.main(["status", "--all"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "ramp" in out and "omr" in out  # --all overrides the toggle

    def test_status_age_humanized(self, tmp_decisions_dir, capsys):
        _seed_from_fixture(tmp_decisions_dir, "sample_ramp_clean.jsonl")
        rc = cli.main(["status"])
        assert rc == 0
        out = capsys.readouterr().out
        # The fixture timestamp is 2026-04-24 which is days/months in the past.
        # Verify some humanized age token is present (one of d/h/m/s suffixes).
        # The status row has "ramp ..." so check the row directly.
        ramp_row = [ln for ln in out.split("\n") if ln.startswith("ramp")][0]
        # Row format: "<strat> <YYYY-MM-DD> <HH:MM> <age> <trigger> ..."
        # The age token is field index 3 after splitting on whitespace.
        # _humanize_age emits suffix s/m/h/d -- one of them must be present.
        age_token = ramp_row.split()[3]
        assert any(age_token.endswith(suf) for suf in ("s", "m", "h", "d"))
