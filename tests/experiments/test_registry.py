"""Tests for the experiment registry.

Each test points the registry at a fresh per-test DB path so we don't
touch the real output/experiments.duckdb.
"""
from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from src.experiments import (
    TRIAL_RULE_OPTIMIZER_ONLY,
    TrialCountUnavailableError,
    append_run,
    incumbent_return_streams,
    init_db,
    n_trials_project_wide,
)


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "experiments.duckdb"


def test_init_db_creates_file_and_tables(db_path: Path) -> None:
    assert not db_path.exists()
    init_db(db_path)
    assert db_path.exists()

    import duckdb
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        tables = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
    finally:
        con.close()
    assert "runs" == tables.intersection({"runs"}).pop()
    assert "return_streams" in tables


def test_append_run_round_trip(db_path: Path) -> None:
    rs = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-02", periods=3, freq="B"),
            "return_pct": [0.001, -0.002, 0.0005],
            "position_count": [5, 6, 4],
        }
    )

    run_id = append_run(
        db_path=db_path,
        strategy_name="ramp",
        agent_name="backtest-driver",
        phase="final",
        params={"momentum_window": 60, "top_n": 10},
        universe_name="sp500-2025",
        asset_class="equities",
        data_frequency="1day",
        window_start="2018-01-01",
        window_end="2024-12-31",
        oos_start="2023-01-01",
        oos_end="2024-12-31",
        n_folds=8,
        metrics={"sharpe": 0.85, "psr": 0.98, "dsr": 0.91, "trade_count": 248},
        cost_tier_used="large_cap",
        cost_bps=10.0,
        combinations_in_run=1,
        config_payload={"strategy": "ramp", "v": 1},
        data_snapshot_date="2024-12-31",
        random_seeds={"numpy": 42},
        wall_clock_start=datetime(2024, 12, 31, 12, 0, tzinfo=timezone.utc),
        wall_clock_end=datetime(2024, 12, 31, 12, 5, tzinfo=timezone.utc),
        verdict="PASS",
        return_stream=rs,
    )

    assert run_id and len(run_id) == 36

    import duckdb
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        row = con.execute(
            "SELECT strategy_name, agent_name, phase, params, metrics, "
            "n_folds, cost_tier_used, verdict, config_sha "
            "FROM runs WHERE run_id = ?",
            [run_id],
        ).fetchone()
        stream = con.execute(
            "SELECT date, return_pct, position_count FROM return_streams "
            "WHERE run_id = ? ORDER BY date",
            [run_id],
        ).fetch_df()
    finally:
        con.close()

    import json as _json
    strat, agent, phase, params_json, metrics_json, n_folds, tier, verdict, config_sha = row
    assert (strat, agent, phase, n_folds, tier, verdict) == ("ramp", "backtest-driver", "final", 8, "large_cap", "PASS")
    assert _json.loads(params_json)["momentum_window"] == 60
    assert _json.loads(metrics_json)["sharpe"] == 0.85
    assert config_sha != "unknown" and len(config_sha) == 16

    assert len(stream) == 3
    assert list(stream["position_count"]) == [5, 6, 4]


def test_n_trials_project_wide_accumulates(db_path: Path) -> None:
    """NEGATIVE CONTROL for the 2026-07-30 fix.

    The superseded implementation filtered on `agent_name =
    'backtest-optimizer'`, an agent that has never written a single registry
    row, so it returned 0 forever. Every assertion below returns 0 under that
    implementation, so this test has demonstrable power against it.
    """
    with pytest.raises(TrialCountUnavailableError):
        n_trials_project_wide(db_path)

    # Every writer counts. The superseded version returned 0 here.
    append_run(
        db_path=db_path,
        strategy_name="ramp",
        agent_name="backtest-driver",
        metrics={"sharpe": 1.0},
        combinations_in_run=5,
    )
    assert n_trials_project_wide(db_path) == 5

    append_run(
        db_path=db_path,
        strategy_name="ramp",
        agent_name="backtest-optimizer",
        metrics={"best_sharpe": 1.2},
        combinations_in_run=120,
    )
    append_run(
        db_path=db_path,
        strategy_name="omr",
        agent_name="futures-harness",
        metrics={"best_sharpe": 0.9},
        combinations_in_run=80,
    )
    assert n_trials_project_wide(db_path) == 205


def test_n_trials_counts_an_agent_name_nobody_has_invented_yet(db_path: Path) -> None:
    """No allowlist. A brand-new writer must be counted the day it appears.

    The original defect was an agent-name filter that silently stopped
    matching. Re-introducing any allowlist breaks this test.
    """
    append_run(
        db_path=db_path,
        strategy_name="whatever",
        agent_name="some-harness-invented-in-2027",
        metrics={"sharpe": 0.4},
        combinations_in_run=7,
    )
    assert n_trials_project_wide(db_path) == 7


def test_n_trials_treats_a_missing_combination_count_as_one_trial(db_path: Path) -> None:
    """A writer that forgets `combinations_in_run` still ran a trial.

    Defaulting the NULL to 0 would silently under-deflate -- the exact
    direction of the original bug.
    """
    append_run(
        db_path=db_path,
        strategy_name="ramp",
        agent_name="fx-harness",
        metrics={"sharpe": 0.4},
    )
    assert n_trials_project_wide(db_path) == 1


def test_n_trials_honors_an_explicit_zero_for_a_rerun(db_path: Path) -> None:
    """`combinations_in_run = 0` marks an exact rerun of an already-counted
    spec (the backfill's convention) and must not add a trial."""
    append_run(
        db_path=db_path,
        strategy_name="ramp",
        agent_name="fx-harness",
        metrics={"sharpe": 0.4},
        combinations_in_run=3,
    )
    append_run(
        db_path=db_path,
        strategy_name="ramp",
        agent_name="fx-harness",
        metrics={"sharpe": 0.4},
        combinations_in_run=0,
        notes="apparatus-correction rerun",
    )
    assert n_trials_project_wide(db_path) == 3


def test_n_trials_never_shrinks_below_a_backfilled_cumulative_count(db_path: Path) -> None:
    """`combinations_project` is a floor: N never shrinks.

    The 2026-07-28 backfill wrote a reconstructed cumulative counter that is
    larger than anything derivable from `combinations_in_run` alone on rows it
    did not touch. Reading only the summed column would discard it.
    """
    append_run(
        db_path=db_path,
        strategy_name="ramp",
        agent_name="trial-count-reconstruction",
        metrics={"sharpe": 0.0},
        combinations_in_run=1,
        combinations_project=349,
    )
    assert n_trials_project_wide(db_path) == 349


def test_n_trials_empty_registry_is_loud(db_path: Path) -> None:
    """An absent count must never masquerade as 'zero trials, no deflation'."""
    with pytest.raises(TrialCountUnavailableError):
        n_trials_project_wide(db_path)


def test_n_trials_empty_registry_non_strict_returns_zero(db_path: Path) -> None:
    """Bookkeeping call sites (recording provenance, not grading) opt out."""
    assert n_trials_project_wide(db_path, strict=False) == 0


def test_incumbent_return_streams_returns_latest_run(db_path: Path) -> None:
    rs_old = pd.DataFrame({"date": pd.date_range("2023-01-02", periods=2, freq="B"), "return_pct": [0.01, -0.01]})
    rs_new = pd.DataFrame({"date": pd.date_range("2024-01-02", periods=2, freq="B"), "return_pct": [0.02, -0.005]})

    append_run(
        db_path=db_path,
        strategy_name="ramp",
        agent_name="backtest-driver",
        metrics={"sharpe": 0.5},
        return_stream=rs_old,
    )
    append_run(
        db_path=db_path,
        strategy_name="ramp",
        agent_name="backtest-driver",
        metrics={"sharpe": 0.85},
        return_stream=rs_new,
    )
    append_run(
        db_path=db_path,
        strategy_name="omr",
        agent_name="backtest-driver",
        metrics={"sharpe": 0.4},
        return_stream=rs_old,
    )

    got = incumbent_return_streams("ramp", db_path=db_path)
    assert len(got) == 2
    assert got.iloc[0]["return_pct"] == pytest.approx(0.02)
    assert got.iloc[1]["return_pct"] == pytest.approx(-0.005)


def test_incumbent_return_streams_empty_when_no_run(db_path: Path) -> None:
    got = incumbent_return_streams("does_not_exist", db_path=db_path)
    assert got.empty
    assert list(got.columns) == ["date", "return_pct", "position_count"]


def test_append_run_rejects_malformed_return_stream(db_path: Path) -> None:
    bad = pd.DataFrame({"date": [date(2024, 1, 2)], "returns": [0.01]})  # wrong column name
    with pytest.raises(ValueError, match="return_pct"):
        append_run(
            db_path=db_path,
            strategy_name="ramp",
            agent_name="backtest-driver",
            metrics={"sharpe": 1.0},
            return_stream=bad,
        )
    # And the run row must not be visible after the failure -- the transaction rolled back.
    import duckdb
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        n = con.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
    finally:
        con.close()
    assert n == 0
