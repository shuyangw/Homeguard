"""Experiment registry: append-only log of every backtest and optimization run.

See docs/methodology/backtesting.md Section 9 for schema and append protocol.
Section 9.3: "If the append fails, the run fails. No silent success." -- this
module raises on every failure; callers must not catch and continue.

Concurrent writers: DuckDB does not support concurrent writes from multiple
processes on the same file. Callers must serialize their backtests (the
backtest_runner is single-process today).
"""
from __future__ import annotations

import hashlib
import json
import socket
import subprocess
import sys
import uuid
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import duckdb
import pandas as pd

DEFAULT_DB_PATH = Path("output/experiments.duckdb")
_SCHEMA_PATH = Path(__file__).parent / "schema.sql"


def init_db(db_path: Path = DEFAULT_DB_PATH) -> None:
    """Create the registry file and schema if missing. Idempotent."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    schema_sql = _SCHEMA_PATH.read_text(encoding="utf-8")
    con = duckdb.connect(str(db_path))
    try:
        con.execute(schema_sql)
    finally:
        con.close()


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return out.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        return "unknown"


def _python_env_hash() -> str:
    try:
        out = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        return hashlib.sha256(out.stdout.encode("utf-8")).hexdigest()[:16]
    except (subprocess.SubprocessError, FileNotFoundError):
        return "unknown"


def _config_sha(config_payload: Any) -> str:
    if config_payload is None:
        return "unknown"
    if isinstance(config_payload, (str, bytes)):
        data = config_payload if isinstance(config_payload, bytes) else config_payload.encode("utf-8")
    else:
        data = json.dumps(config_payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(data).hexdigest()[:16]


def _json_or_none(value: Any) -> Optional[str]:
    if value is None:
        return None
    return json.dumps(value, default=str)


def _to_date(value: Any) -> Optional[date]:
    if value is None:
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    return pd.to_datetime(value).date()


def append_run(
    *,
    strategy_name: str,
    agent_name: str,
    metrics: Mapping[str, Any],
    db_path: Path = DEFAULT_DB_PATH,
    run_id: Optional[str] = None,
    phase: Optional[str] = None,
    parent_run_id: Optional[str] = None,
    params: Optional[Mapping[str, Any]] = None,
    universe_name: Optional[str] = None,
    asset_class: Optional[str] = None,
    data_frequency: Optional[str] = None,
    window_start: Any = None,
    window_end: Any = None,
    is_start: Any = None,
    is_end: Any = None,
    oos_start: Any = None,
    oos_end: Any = None,
    n_folds: Optional[int] = None,
    regime_breakdown: Optional[Mapping[str, Any]] = None,
    fold_metrics: Optional[Mapping[str, Any]] = None,
    cost_tier_used: Optional[str] = None,
    cost_bps: Optional[float] = None,
    cost_sensitivity: Optional[Mapping[str, Any]] = None,
    combinations_in_run: Optional[int] = None,
    combinations_project: Optional[int] = None,
    config_payload: Any = None,
    data_snapshot_date: Any = None,
    random_seeds: Optional[Mapping[str, Any]] = None,
    wall_clock_start: Optional[datetime] = None,
    wall_clock_end: Optional[datetime] = None,
    verdict: Optional[str] = None,
    verdict_reasons: Optional[Mapping[str, Any]] = None,
    notes: Optional[str] = None,
    return_stream: Optional[pd.DataFrame] = None,
) -> str:
    """Append one run to the registry. Return the run_id.

    Raises on any failure -- no silent success per methodology Section 9.3.

    `return_stream`, when supplied, must have columns ``date`` and
    ``return_pct`` (with optional ``position_count``). It's written to
    ``return_streams`` keyed by the new run_id.
    """
    init_db(db_path)
    run_id = run_id or str(uuid.uuid4())

    row = {
        "run_id": run_id,
        "timestamp_utc": datetime.now(tz=timezone.utc),
        "strategy_name": strategy_name,
        "agent_name": agent_name,
        "phase": phase,
        "parent_run_id": parent_run_id,
        "params": _json_or_none(dict(params) if params else None),
        "universe_name": universe_name,
        "asset_class": asset_class,
        "data_frequency": data_frequency,
        "window_start": _to_date(window_start),
        "window_end": _to_date(window_end),
        "is_start": _to_date(is_start),
        "is_end": _to_date(is_end),
        "oos_start": _to_date(oos_start),
        "oos_end": _to_date(oos_end),
        "n_folds": n_folds,
        "metrics": _json_or_none(dict(metrics)),
        "regime_breakdown": _json_or_none(dict(regime_breakdown) if regime_breakdown else None),
        "fold_metrics": _json_or_none(dict(fold_metrics) if fold_metrics else None),
        "cost_tier_used": cost_tier_used,
        "cost_bps": cost_bps,
        "cost_sensitivity": _json_or_none(dict(cost_sensitivity) if cost_sensitivity else None),
        "combinations_in_run": combinations_in_run,
        "combinations_project": combinations_project,
        "git_sha": _git_sha(),
        "config_sha": _config_sha(config_payload),
        "data_snapshot_date": _to_date(data_snapshot_date),
        "python_env_hash": _python_env_hash(),
        "random_seeds": _json_or_none(dict(random_seeds) if random_seeds else None),
        "wall_clock_start": wall_clock_start,
        "wall_clock_end": wall_clock_end,
        "host": socket.gethostname(),
        "verdict": verdict,
        "verdict_reasons": _json_or_none(dict(verdict_reasons) if verdict_reasons else None),
        "notes": notes,
    }

    columns = list(row.keys())
    placeholders = ", ".join("?" for _ in columns)
    insert_sql = f"INSERT INTO runs ({', '.join(columns)}) VALUES ({placeholders})"

    con = duckdb.connect(str(db_path))
    try:
        con.begin()
        con.execute(insert_sql, [row[c] for c in columns])
        if return_stream is not None and len(return_stream):
            _insert_return_stream(con, run_id, return_stream)
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()

    return run_id


def _insert_return_stream(con: duckdb.DuckDBPyConnection, run_id: str, stream: pd.DataFrame) -> None:
    if "date" not in stream.columns or "return_pct" not in stream.columns:
        raise ValueError("return_stream must have columns 'date' and 'return_pct'")
    pos_col = "position_count" if "position_count" in stream.columns else None
    rows = [
        (
            run_id,
            _to_date(r["date"]),
            float(r["return_pct"]),
            int(r[pos_col]) if pos_col and pd.notna(r[pos_col]) else None,
        )
        for _, r in stream.iterrows()
    ]
    con.executemany(
        "INSERT INTO return_streams (run_id, date, return_pct, position_count) VALUES (?, ?, ?, ?)",
        rows,
    )


def n_trials_project_wide(db_path: Path = DEFAULT_DB_PATH) -> int:
    """Cumulative optimizer trial count across the whole project.

    Use as the N argument to expected_max_sharpe() per methodology Section 9.4.
    Returns 0 if the registry is empty.
    """
    init_db(db_path)
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        row = con.execute(
            "SELECT COALESCE(SUM(combinations_in_run), 0) FROM runs "
            "WHERE agent_name = 'backtest-optimizer'"
        ).fetchone()
        return int(row[0] or 0)
    finally:
        con.close()


def incumbent_return_streams(
    strategy_name: str,
    since_date: Optional[date] = None,
    db_path: Path = DEFAULT_DB_PATH,
) -> pd.DataFrame:
    """Daily OOS returns for an incumbent strategy, used by portfolio-integrator.

    Pulls return_streams for the most recent run of `strategy_name` whose
    timestamp_utc >= since_date. Returns columns (date, return_pct, position_count).
    Empty DataFrame if no qualifying run exists.
    """
    init_db(db_path)
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        run_id_row = con.execute(
            """
            SELECT run_id FROM runs
            WHERE strategy_name = ?
              AND (? IS NULL OR timestamp_utc >= ?)
            ORDER BY timestamp_utc DESC
            LIMIT 1
            """,
            [strategy_name, since_date, since_date],
        ).fetchone()
        if not run_id_row:
            return pd.DataFrame(columns=["date", "return_pct", "position_count"])
        run_id = run_id_row[0]
        return con.execute(
            "SELECT date, return_pct, position_count FROM return_streams "
            "WHERE run_id = ? ORDER BY date",
            [run_id],
        ).fetch_df()
    finally:
        con.close()
