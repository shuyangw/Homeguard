"""Writer for decision log JSONL files.

Append-only with O_APPEND atomic writes for records < 4KB. Larger
records use tmp-file + atomic rename. Retention cleanup runs lazily
on each append.
"""
from __future__ import annotations

import json
import os
import random
import tempfile
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, Optional

from src.trading.decision_log import paths
from src.trading.decision_log.record import DecisionRecord
from src.utils.timezone import tz


# Posix PIPE_BUF guarantees atomic appends below this size.
ATOMIC_APPEND_THRESHOLD_BYTES = 4096

# Default 365 days; overridable via DECISION_LOG_RETENTION_DAYS env var.
DEFAULT_RETENTION_DAYS = 365


def append(rec: DecisionRecord, *, retention_days: Optional[int] = None) -> None:
    """Append a DecisionRecord to its strategy/date JSONL file.

    Also updates _latest/<strategy>.json with a one-line snapshot.
    Triggers lazy retention cleanup on every Nth call (cheap probability test).
    """
    line = rec.to_jsonl_line().encode("utf-8")
    day = _date_from_iso(rec.timestamp)

    target = paths.jsonl_path(rec.strategy, day)
    target.parent.mkdir(parents=True, exist_ok=True)

    if len(line) <= ATOMIC_APPEND_THRESHOLD_BYTES:
        # Single small write: O_APPEND guarantees atomicity below PIPE_BUF
        with open(target, "ab") as f:
            f.write(line)
    else:
        # Larger record: serialize append via tmp-file + atomic concat
        _atomic_append_large(target, line)

    _update_latest(rec)

    # Lazy retention -- run on ~1 in 50 appends
    if random.random() < 0.02:
        try:
            _cleanup_old_files(retention_days=retention_days)
        except Exception:
            # Retention cleanup must never fail an append
            pass


def _atomic_append_large(target: Path, line_bytes: bytes) -> None:
    """For records exceeding PIPE_BUF, write via tmp + rename."""
    target.parent.mkdir(parents=True, exist_ok=True)
    # Read existing content (may be empty)
    existing = target.read_bytes() if target.exists() else b""
    tmp = tempfile.NamedTemporaryFile(
        mode="wb", dir=target.parent, prefix=".tmp_", suffix=".jsonl",
        delete=False,
    )
    try:
        tmp.write(existing)
        tmp.write(line_bytes)
        tmp.flush()
        os.fsync(tmp.fileno())
    finally:
        tmp.close()
    os.replace(tmp.name, target)


def _is_substantive(rec: DecisionRecord) -> bool:
    """True if the record carries a real decision worth snapshotting to _latest.

    A blocked or errored run (failed preconditions -- strategy disabled, lock not
    acquired, health check failed) early-returns before reaching the inputs or
    logic stages, so its `inputs` are empty (StrategyInputs.empty()) and
    `logic_decisions` is None. Such a record must NOT overwrite _latest: the
    runner polls every 15s and `_should_run_now` matches the whole rebalance
    minute, so a second poll firing while the first rebalance still holds the
    execution lock emits exactly such an empty record -- and unconditionally
    overwriting _latest with it clobbers the day's real rebalance decision that
    the A7 comparator (scripts/trading/compare_paper_vs_plan.py) reads, producing
    a spurious VACUOUS verdict. The full record is still appended to the day's
    JSONL for diagnostics regardless; only the _latest snapshot is protected.
    """
    ld = rec.logic_decisions
    if ld is not None and ld.target_weights:
        return True
    inp = rec.inputs
    if inp is not None and (inp.momentum_scores or inp.regime is not None):
        return True
    return False


def _update_latest(rec: DecisionRecord) -> None:
    """Atomically rewrite _latest/<strategy>.json with this record.

    Skips non-substantive (blocked/errored) records so they cannot clobber the
    day's real decision -- see _is_substantive. The record is still in the day's
    JSONL either way.
    """
    if not _is_substantive(rec):
        return
    target = paths.latest_path(rec.strategy)
    target.parent.mkdir(parents=True, exist_ok=True)
    line = rec.to_jsonl_line()
    # Write to tmp then rename (POSIX atomic same-fs rename)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(line, encoding="utf-8")
    os.replace(tmp, target)


def write_position_state(
    strategy: str,
    positions: Dict[str, float],
    position_open_dates: Dict[str, datetime],
) -> None:
    """Atomically rewrite _latest/<strategy>_position_state.json.

    Companion snapshot to _update_latest so a separate process (e.g. the
    V11 comparator) can read the live adapter's current positions and
    per-symbol open dates. Same tmp + os.replace semantics as
    _update_latest. Empty dicts are valid (Day-1 state).
    """
    target = paths.position_state_path(strategy)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "strategy": strategy,
        "timestamp": tz.now().isoformat(),
        "positions": {sym: float(qty) for sym, qty in positions.items()},
        "position_open_dates": {
            sym: dt.isoformat() for sym, dt in position_open_dates.items()
        },
    }
    line = json.dumps(payload)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(line, encoding="utf-8")
    os.replace(tmp, target)


def _cleanup_old_files(
    retention_days: Optional[int] = None,
    today: Optional[date] = None,
) -> int:
    """Remove decision JSONL files older than retention_days.

    Returns count of files deleted. _latest/ subdirectory is preserved.
    """
    if retention_days is None:
        retention_days = int(os.environ.get(
            "DECISION_LOG_RETENTION_DAYS", DEFAULT_RETENTION_DAYS,
        ))
    if today is None:
        today = date.today()

    cutoff = today - timedelta(days=retention_days)
    decisions = paths.decisions_dir()
    if not decisions.exists():
        return 0

    deleted = 0
    for entry in decisions.iterdir():
        if entry.is_dir():
            continue  # skip _latest/
        # Filename format: <strategy>_<YYYYMMDD>.jsonl
        try:
            stem = entry.stem  # strategy_20260424
            date_part = stem.rsplit("_", 1)[1]
            file_date = datetime.strptime(date_part, "%Y%m%d").date()
        except (IndexError, ValueError):
            continue
        if file_date < cutoff:
            entry.unlink()
            deleted += 1
    return deleted


def _date_from_iso(timestamp_iso: str) -> date:
    """Extract local date from ISO8601 timestamp.

    Records are bucketed by the date in the timestamp (which is in
    market-local time for stock strategies). Don't convert to UTC.
    """
    dt = datetime.fromisoformat(timestamp_iso)
    return dt.date()
