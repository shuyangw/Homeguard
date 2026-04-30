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


class TestLogicRendering:
    """Renderer must compute per-buy from len(target_value_usd), NOT top_n.

    Regression test for the case that surfaced today: RAMP rebalanced from
    SIDEWAYS (top_n=5) to STRONG_BULL (top_n=20) with 3 held positions
    retained from the prior session. target_value_usd had 17 entries
    (only the new buys) summing to $85,000. The pre-fix renderer divided
    by top_n=20 and reported "$4,250 each" -- but each new buy was
    actually $5,000 = $85k / 17.
    """

    def _make_record_with_holds(self):
        from src.trading.decision_log.record import (
            DecisionRecord, TriggerInfo, PreconditionResults, GateResult,
            StrategyInputs, LogicDecisions, RunMetadata,
        )
        new_buys = [f"BUY{i}" for i in range(17)]
        holds = ["HOLD1", "HOLD2", "HOLD3"]
        return DecisionRecord(
            schema_version=1,
            decision_id="logic-test-1",
            strategy="ramp",
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
            logic_decisions=LogicDecisions(
                top_n=20,
                target_symbols=new_buys,
                target_weights={s: 0.05 for s in new_buys},
                target_value_usd={s: 5000.0 for s in new_buys},
                reduce_exposure=False,
                exposure_pct=1.0,
                exit_signals=[],
                hold_signals=holds,
                skip_reasons={},
            ),
            executions=[],
            post_state=None,
            error=None,
            metadata=RunMetadata(
                broker_name="ibkr", data_provider="alpaca", git_sha="abc",
                initial_capital_usd=100000.0, strategy_version=1,
                process_pid=1, hostname="test",
            ),
        )

    def test_per_buy_uses_buy_count_not_top_n(self):
        rec = self._make_record_with_holds()
        out = cli._render_record(rec)
        # The fixed renderer must use 17 (len of target_value_usd) as the
        # divisor, producing $5,000 per new buy.
        assert "17 new @ $5,000 each" in out
        assert "($85,000) + 3 held" in out
        # The broken pre-fix value MUST NOT appear ($85k / 20 = $4,250).
        assert "$4,250" not in out
        # And exposure_pct lives on its own line, not concatenated with sum.
        assert "Target exposure: 100% of initial_capital" in out

    def test_hold_line_renders_when_holds_present(self):
        rec = self._make_record_with_holds()
        out = cli._render_record(rec)
        assert "Hold: HOLD1 HOLD2 HOLD3" in out

    def test_no_held_branch_when_holds_empty(self):
        """Pure rebalance with no holds uses the simpler one-line summary."""
        rec = self._make_record_with_holds()
        rec.logic_decisions.hold_signals = []
        out = cli._render_record(rec)
        assert "+ 0 held" not in out
        assert "Target 20 new buys @ $5,000 each = $85,000" in out
        # No Hold: line emitted.
        assert "Hold:" not in out
