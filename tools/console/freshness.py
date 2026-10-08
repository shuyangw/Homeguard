"""Freshness classes from the parent spec: what a value means once its reading is old."""
from __future__ import annotations

from datetime import datetime

INSTANCE_OWNED = "instance"  # process state, decisions, switches: still true as of the reading
MARKET_OWNED = "market"      # equity, P&L, drawdown: drift once prices move
MEASURED = "measured"        # gateway, heartbeat, stream, memory: only valid while measured

_STALE_CLASS = {INSTANCE_OWNED: "frozen", MARKET_OWNED: "drifting", MEASURED: "unknown"}


def classify(kind: str, live: bool) -> str:
    return "live" if live else _STALE_CLASS[kind]


def age_text(as_of: datetime, now: datetime) -> str:
    seconds = max(int((now - as_of).total_seconds()), 0)
    if seconds < 60:
        return f"{seconds} s ago"
    if seconds < 3600:
        return f"{seconds // 60} min ago"
    return f"{seconds // 3600} h {seconds % 3600 // 60} min ago"
