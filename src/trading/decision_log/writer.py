"""Writer for decision log JSONL files.

Append-only with O_APPEND atomic writes for records < 4KB. Larger
records use tmp-file + atomic rename. Retention cleanup runs lazily
on each append.
"""
from __future__ import annotations

import os
import random
import tempfile
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

from src.trading.decision_log import paths
from src.trading.decision_log.record import DecisionRecord


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


def _update_latest(rec: DecisionRecord) -> None:
    """Atomically rewrite _latest/<strategy>.json with this record."""
    target = paths.latest_path(rec.strategy)
    target.parent.mkdir(parents=True, exist_ok=True)
    line = rec.to_jsonl_line()
    # Write to tmp then rename (POSIX atomic same-fs rename)
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
