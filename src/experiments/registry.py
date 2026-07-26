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
from typing import Any, Callable, Dict, Mapping, Optional

import duckdb
import pandas as pd

from src.utils import logger

DEFAULT_DB_PATH = Path("output/experiments.duckdb")
_SCHEMA_PATH = Path(__file__).parent / "schema.sql"

# Retry policy for transient Windows file-lock contention. The registry
# can live inside a Dropbox-synced folder; Dropbox's indexer periodically
# opens the file for read, which collides with DuckDB's exclusive write.
# Catch the OSError / IOError / duckdb.IOException, backoff briefly, retry.
_RETRY_ATTEMPTS = 5
_RETRY_BASE_DELAY = 0.2  # seconds; doubled each attempt


def _is_lock_error(exc: BaseException) -> bool:
    """True if `exc` is a transient file-lock error we should retry on."""
    msg = str(exc).lower()
    return (
        "being used by another process" in msg
        or "io error" in msg
        or "cannot open file" in msg
        or "permission denied" in msg
    )


def _connect_with_retry(db_path: Path, *, read_only: bool = False):
    """Open a DuckDB connection, retrying briefly on transient lock errors."""
    import time as _time
    last_exc: Optional[BaseException] = None
    for attempt in range(_RETRY_ATTEMPTS):
        try:
            return duckdb.connect(str(db_path), read_only=read_only)
        except (OSError, IOError, duckdb.IOException) as exc:
            if not _is_lock_error(exc) or attempt == _RETRY_ATTEMPTS - 1:
                raise
            last_exc = exc
            _time.sleep(_RETRY_BASE_DELAY * (2 ** attempt))
    # Unreachable; the loop either returns or raises.
    raise RuntimeError(f"unreachable: lock retries exhausted (last={last_exc})")


def init_db(db_path: Path = DEFAULT_DB_PATH) -> None:
    """Create the registry file and schema if missing. Idempotent."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    schema_sql = _SCHEMA_PATH.read_text(encoding="utf-8")
    con = _connect_with_retry(db_path)
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


_SPEC_IDENTITY = ("strategy_name", "params", "window_start", "window_end",
                  "asset_class", "data_frequency")


def _spec_identity_sql(alias: str = "") -> str:
    p = f"{alias}." if alias else ""
    return ", ".join(f"{p}{c}" for c in _SPEC_IDENTITY)


def duplicate_spec_run_ids(db_path: Path = DEFAULT_DB_PATH) -> list[dict]:
    """Groups of runs sharing an identical spec identity, most recent first.

    Audit helper. Duplicates are NOT removed: they inflate the trial count N,
    which is the safe direction, and N never shrinks. Surfacing them lets a
    reviewer decide whether a re-run was intentional.

    Rows with NULL params are EXCLUDED. Without params there is no recoverable
    spec identity, and grouping on the null would merge genuinely different runs:
    RAMP-V31 has 47 such rows carrying 45 distinct metric sets, so treating them
    as one repeated spec would be plainly wrong.
    """
    con = _connect_with_retry(db_path, read_only=True)
    try:
        rows = con.execute(
            f"SELECT {_spec_identity_sql()}, list(run_id) AS run_ids, count(*) AS n "
            f"FROM runs WHERE params IS NOT NULL GROUP BY {_spec_identity_sql()} "
            "HAVING count(*) > 1 ORDER BY n DESC").fetchall()
        cols = list(_SPEC_IDENTITY) + ["run_ids", "n"]
        return [dict(zip(cols, r)) for r in rows]
    finally:
        con.close()


def _warn_if_duplicate_spec(con, row: Mapping[str, Any]) -> None:
    if row.get("params") is None:
        # No params, no recoverable identity. Claiming a duplicate here would be
        # a guess, and a wrong one 45 times out of 47 on the existing RAMP rows.
        return
    where = " AND ".join(
        f"{c} IS NOT DISTINCT FROM ?" for c in _SPEC_IDENTITY)
    prior = con.execute(f"SELECT run_id FROM runs WHERE {where}",
                        [row[c] for c in _SPEC_IDENTITY]).fetchall()
    if prior:
        logger.error(
            f"[registry] DUPLICATE spec: {row['strategy_name']!r} over "
            f"{row['window_start']}..{row['window_end']} with identical params "
            f"already exists as {[p[0] for p in prior]}. The row is still being "
            "inserted (N never shrinks); confirm the re-run was intentional "
            "before counting it as a new trial.")


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

    con = _connect_with_retry(db_path)
    try:
        con.begin()
        _warn_if_duplicate_spec(con, row)
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
    con = _connect_with_retry(db_path, read_only=True)
    try:
        row = con.execute(
            "SELECT COALESCE(SUM(combinations_in_run), 0) FROM runs "
            "WHERE agent_name = 'backtest-optimizer'"
        ).fetchone()
        return int(row[0] or 0)
    finally:
        con.close()


def make_trial_callback(
    *,
    parent_run_id: str,
    strategy_name: str,
    combinations_in_run: int,
    start_date: Any = None,
    end_date: Any = None,
    wall_clock_start: Optional[datetime] = None,
    cost_tier_used: Optional[str] = None,
    cost_bps: Optional[float] = None,
    phase: str = "optimization",
    db_path: Path = DEFAULT_DB_PATH,
    on_failure: str = "warn",
) -> Callable[[Mapping[str, Any], Mapping[str, Any]], None]:
    """Build a per-trial registry-append callback for optimizers.

    Returns a `(params, stats) -> None` closure that appends one child
    `runs` row per optimizer trial, sharing the supplied `parent_run_id`.
    Optimizers stay registry-agnostic -- they accept this callback as
    an opt-in hook. Callers that don't want registry writes (research
    sandboxes, tests) pass nothing and the optimizer skips the call.

    Section 9.3 says "no silent success" but the runner-level parent-row
    append (in `src/backtest_runner.py`) is where that gate fires. A
    single failed per-trial append should not abort a 5000-config sweep,
    so the default behavior here is `on_failure='warn'`: log and continue.
    Pass `on_failure='raise'` if you want the optimizer to crash on any
    registry write failure.

    Args:
        parent_run_id: UUID shared by every trial in this sweep.
        strategy_name: name of the strategy being optimized.
        combinations_in_run: total trial count for the sweep (used by DSR).
        start_date / end_date: window for the rows.
        wall_clock_start: start of the sweep; defaults to now() if None.
        cost_tier_used / cost_bps: provenance; read from engine by caller.
        phase: free-form label (e.g. 'optimization_grid_search').
        db_path: registry file path.
        on_failure: 'warn' (default) -- log and continue -- or 'raise'.
    """
    import logging

    log = logging.getLogger(__name__)
    sweep_start = wall_clock_start or datetime.now(tz=timezone.utc)

    def _callback(params: Mapping[str, Any], stats: Mapping[str, Any]) -> None:
        if stats is None:
            return
        try:
            metrics = {
                "sharpe": float(stats.get("Sharpe Ratio", 0.0) or 0.0),
                "total_return_pct": float(stats.get("Total Return [%]", 0.0) or 0.0),
                "annual_return_pct": float(stats.get("Annual Return [%]", 0.0) or 0.0),
                "max_drawdown_pct": float(stats.get("Max Drawdown [%]", 0.0) or 0.0),
                "win_rate_pct": float(stats.get("Win Rate [%]", 0.0) or 0.0),
                "trade_count": int(stats.get("Total Trades", 0) or 0),
            }
            append_run(
                db_path=db_path,
                strategy_name=strategy_name,
                agent_name="backtest-optimizer",
                phase=phase,
                parent_run_id=parent_run_id,
                params=dict(params),
                window_start=start_date,
                window_end=end_date,
                metrics=metrics,
                combinations_in_run=combinations_in_run,
                cost_tier_used=cost_tier_used,
                cost_bps=cost_bps,
                wall_clock_start=sweep_start,
                wall_clock_end=datetime.now(tz=timezone.utc),
                verdict="PENDING",
            )
        except Exception as e:
            if on_failure == "raise":
                raise
            log.warning(f"[registry] trial-callback append failed: {e}")

    return _callback


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
    con = _connect_with_retry(db_path, read_only=True)
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
