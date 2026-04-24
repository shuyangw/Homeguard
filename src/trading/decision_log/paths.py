"""Path constants and resolvers for the decision log package.

Decision records live under data/trading/decisions/ as one JSONL file
per strategy per UTC date. The _latest/ subdirectory holds one-line
copies of the most recent record per strategy for quick status checks.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path


def decisions_dir() -> Path:
    """Root directory for decision JSONL files."""
    project_root = Path(__file__).resolve().parents[3]
    return project_root / "data" / "trading" / "decisions"


def latest_dir() -> Path:
    """Subdirectory holding latest-record snapshots per strategy."""
    return decisions_dir() / "_latest"


def jsonl_path(strategy: str, day: date) -> Path:
    """Path to the per-day JSONL file for a strategy."""
    return decisions_dir() / f"{strategy}_{day.strftime('%Y%m%d')}.jsonl"


def latest_path(strategy: str) -> Path:
    """Path to the latest-record snapshot for a strategy."""
    return latest_dir() / f"{strategy}.json"
