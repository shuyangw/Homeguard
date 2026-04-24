"""Tests for DecisionRecord schema and serialization.

Schema is the contract every consumer (writer, reader, CLI, strategies)
relies on. These tests pin down JSON shape and Optional/None semantics.
"""
import json
from datetime import datetime, timezone

import pytest

from src.trading.decision_log.record import (
    DecisionRecord,
    TriggerInfo,
    GateResult,
    PreconditionResults,
    StrategyInputs,
    LogicDecisions,
    Execution,
    PostState,
    PositionSnapshot,
    ErrorInfo,
    RunMetadata,
)


def _minimal_record(strategy="ramp") -> DecisionRecord:
    """Build the smallest valid record (preconditions blocked, no logic/exec/post)."""
    return DecisionRecord(
        schema_version=1,
        decision_id="00000000-0000-0000-0000-000000000001",
        strategy=strategy,
        timestamp="2026-04-24T15:55:56-04:00",
        trigger=TriggerInfo(
            kind="scheduled_rebalance",
            schedule_time="15:55",
            actual_fire_time="2026-04-24T15:55:56-04:00",
            delay_seconds=0.84,
            consecutive_fire_count=1,
        ),
        preconditions=PreconditionResults(
            all_passed=False,
            strategy_enabled=GateResult(passed=True, details={}),
            shutdown_requested=GateResult(passed=True, details={}),
            execution_lock_acquired=GateResult(passed=True, details={}),
            health_check=GateResult(passed=False, details={}, error="missing get_open_orders"),
            data_freshness=GateResult(passed=True, details={"valid_pct": 96.2}),
            extra={},
        ),
        inputs=StrategyInputs.empty(),
        logic_decisions=None,
        executions=[],
        post_state=None,
        error=None,
        metadata=RunMetadata(
            broker_name="ibkr",
            data_provider="alpaca-market-iex",
            git_sha="9533ec0",
            initial_capital_usd=100000.0,
            strategy_version=1,
            process_pid=12345,
            hostname="ip-172-31-21-22",
        ),
    )


class TestSerialization:
    def test_minimal_record_round_trip(self):
        """A blocked-preconditions record serializes and parses back identically."""
        rec = _minimal_record()
        line = rec.to_jsonl_line()

        # Must be a single line of valid JSON ending in newline
        assert line.endswith("\n")
        assert line.count("\n") == 1
        json.loads(line)  # raises if invalid

        rec2 = DecisionRecord.from_jsonl_line(line)
        assert rec2 == rec

    def test_full_record_round_trip(self):
        """A clean run with executions + post_state round-trips."""
        rec = _minimal_record()
        rec.preconditions.all_passed = True
        rec.preconditions.health_check = GateResult(passed=True, details={"buying_power": 1000000})
        rec.inputs = StrategyInputs(
            regime="WEAK_BULL",
            regime_confidence=0.75,
            regime_params={"long_p": 21, "short_p": 5},
            vix=19.40,
            spy_drawdown_pct=-2.3,
            universe_size=503,
            data_completeness_pct=96.2,
            cache_source="rest_alpaca",
            momentum_scores={"AAPL": 0.014, "MSFT": 0.023},
            extra={},
        )
        rec.logic_decisions = LogicDecisions(
            top_n=10,
            target_symbols=["AAPL", "MSFT"],
            target_weights={"AAPL": 0.10, "MSFT": 0.10},
            target_value_usd={"AAPL": 10000.0, "MSFT": 10000.0},
            reduce_exposure=False,
            exposure_pct=1.0,
            exit_signals=[],
            hold_signals=[],
            skip_reasons={},
        )
        rec.executions = [
            Execution(
                symbol="AAPL", action="buy", target_qty=37, target_value_usd=10000.0,
                status="filled", fill_price=272.50, filled_qty=37,
                order_id="ibkr-12345", reason=None, latency_ms=124.0,
            ),
        ]
        rec.post_state = PostState(
            positions_after={
                "AAPL": PositionSnapshot(qty=37, avg_price=272.50, unrealized_pnl=0.0)
            },
            cash_after=89925.0,
            strategy_equity_after=99925.0,
            state_writes=["add_or_update_position(ramp, AAPL, 37, 272.50)"],
        )

        line = rec.to_jsonl_line()
        rec2 = DecisionRecord.from_jsonl_line(line)
        assert rec2 == rec
        assert rec2.executions[0].fill_price == 272.50

    def test_error_record_captures_stage(self):
        rec = _minimal_record()
        rec.error = ErrorInfo(
            type="ValueError",
            message="bad data",
            traceback="Traceback...\n",
            stage="logic",
        )
        rec2 = DecisionRecord.from_jsonl_line(rec.to_jsonl_line())
        assert rec2.error.stage == "logic"

    def test_unknown_strategy_extra_fields_round_trip(self):
        """The extra: Dict escape hatch carries strategy-specific fields."""
        rec = _minimal_record(strategy="cscm")
        rec.inputs = StrategyInputs.empty()
        rec.inputs.extra = {
            "btc_price": 77234.40,
            "btc_sma": 70932.83,
            "is_rebalance_day": True,
        }
        rec2 = DecisionRecord.from_jsonl_line(rec.to_jsonl_line())
        assert rec2.inputs.extra["btc_price"] == 77234.40

    def test_schema_version_field_present(self):
        rec = _minimal_record()
        line = rec.to_jsonl_line()
        loaded = json.loads(line)
        assert loaded["schema_version"] == 1


class TestEquality:
    def test_equal_records_compare_equal(self):
        a = _minimal_record()
        b = _minimal_record()
        assert a == b

    def test_different_strategy_compares_unequal(self):
        a = _minimal_record(strategy="ramp")
        b = _minimal_record(strategy="omr")
        assert a != b


class TestEmptyHelpers:
    def test_strategy_inputs_empty_returns_no_data(self):
        empty = StrategyInputs.empty()
        assert empty.universe_size == 0
        assert empty.momentum_scores == {}

    def test_precondition_results_empty_returns_failed(self):
        empty = PreconditionResults.empty()
        assert empty.all_passed is False
