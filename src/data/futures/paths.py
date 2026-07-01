"""Consolidated futures data paths (post-2026 consolidation).

Single source of truth for the futures/databento/* and futures/definitions
layout. All futures readers resolve paths through here so a future
reorganization is a one-file change instead of a repo-wide grep.
"""
from __future__ import annotations

from pathlib import Path

from src.settings import get_local_storage_dir


def _futures_root() -> Path:
    return get_local_storage_dir() / "futures"


def continuous_1min_dir() -> Path:
    """.v.0 volume-roll continuous minute bars."""
    return _futures_root() / "databento" / "1min"


def per_contract_1min_dir() -> Path:
    """Per-contract (raw CME symbol) minute bars."""
    return _futures_root() / "databento" / "per_contract_1min"


def statistics_dir() -> Path:
    """Databento statistics (settle / OI / volume events)."""
    return _futures_root() / "databento" / "statistics"


def definitions_dir() -> Path:
    """Contract definition events (expiration, tick size, etc.)."""
    return _futures_root() / "definitions"


def roll_calendar_dir() -> Path:
    """Cached per-root roll calendar artifacts (built in Phase 1)."""
    return _futures_root() / "roll_calendar"
