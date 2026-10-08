"""The eight exception checks. Pure: a status document, liveness and the time in, levels out."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone

from tools.console import freshness
from tools.console.decision_times import DECISION_TIMES
from tools.console.schedule import EASTERN, nyse_session

HEARTBEAT_MAX_AGE = timedelta(seconds=120)
DECISION_GRACE = timedelta(minutes=5)
DRAWDOWN_CAUTION_PCT = -10.0
DRAWDOWN_WARNING_PCT = -20.0
TRADING_UNIT_MEMORY_CAP_BYTES = 1 << 30  # homeguard-multi MemoryMax=1G
MEMORY_CAUTION = 0.80
MEMORY_WARNING = 0.95
GATEWAY_UNIT = "homeguard-gateway.service"
TRADING_UNIT = "homeguard-multi.service"
MEASURED_CHECKS = {"IB Gateway", "Broker heartbeat", "Market data stream", "Host memory"}


@dataclass(frozen=True)
class Check:
    name: str
    level: str
    detail: str


def run_checks(document: dict, live: bool, as_of: datetime, now: datetime) -> list[Check]:
    units = document.get("units") or []
    strategies = document.get("strategies") or {}
    checks = [
        gateway_check(units),
        heartbeat_check(strategies, now),
        stream_check(strategies, now),
        decisions_check(strategies, now),
        rejects_check(strategies),
        drawdown_check(strategies),
        memory_check(units),
        Check("Metrics scrape", "unknown", "Measured from Phase 2b"),
    ]
    if freshness.classify(freshness.MEASURED, live) != "unknown":
        return checks
    return [_unknown_since(check, as_of) if check.name in MEASURED_CHECKS else check for check in checks]


def _unknown_since(check: Check, as_of: datetime) -> Check:
    return Check(check.name, "unknown", f"Unknown since {as_of.astimezone(EASTERN):%H:%M}; last reading: {check.detail}")


def _metric_values(strategies: dict, kind: str, name: str) -> list[float]:
    values: list[float] = []
    for entry in strategies.values():
        snapshot = entry.get("snapshot") or {}
        values.extend((snapshot.get(kind) or {}).get(name, {}).values())
    return values


def _unit(units: list[dict], name: str) -> dict | None:
    return next((unit for unit in units if unit.get("unit") == name), None)


def gateway_check(units: list[dict]) -> Check:
    unit = _unit(units, GATEWAY_UNIT)
    if unit is None:
        return Check("IB Gateway", "warning", f"{GATEWAY_UNIT} not reported")
    if unit.get("active_state") != "active":
        return Check("IB Gateway", "warning", f"{unit.get('active_state')} ({unit.get('sub_state')})")
    return Check("IB Gateway", "normal", "Running")


def heartbeat_check(strategies: dict, now: datetime) -> Check:
    beats = _metric_values(strategies, "gauges", "hg_broker_last_heartbeat_timestamp")
    if not beats:
        return Check("Broker heartbeat", "normal", "Not reported")
    age = now - datetime.fromtimestamp(max(beats), tz=timezone.utc)
    seconds = int(age.total_seconds())
    if age > HEARTBEAT_MAX_AGE:
        return Check("Broker heartbeat", "warning", f"Last heartbeat {seconds} s ago")
    return Check("Broker heartbeat", "normal", f"{seconds} s ago")


def stream_check(strategies: dict, now: datetime) -> Check:
    connected = _metric_values(strategies, "gauges", "hg_websocket_connected")
    if not connected:
        return Check("Market data stream", "normal", "Not reported")
    if min(connected) > 0:
        return Check("Market data stream", "normal", "Connected")
    session = nyse_session(now.astimezone(EASTERN).date())
    if session is not None and session[0] <= now < session[1]:
        return Check("Market data stream", "caution", "Disconnected during the session")
    return Check("Market data stream", "normal", "Disconnected (market closed)")


def _decided_at(entry: dict) -> datetime | None:
    decision = entry.get("last_decision") or {}
    try:
        decided = datetime.fromisoformat(decision["timestamp"])
    except (KeyError, TypeError, ValueError):
        return None
    return decided if decided.tzinfo is not None else decided.replace(tzinfo=EASTERN)


def decisions_check(strategies: dict, now: datetime) -> Check:
    local_now = now.astimezone(EASTERN)
    session = nyse_session(local_now.date())
    if session is None:
        return Check("Decisions on schedule", "normal", "No session today")
    missed = []
    for name, entry in sorted(strategies.items()):
        if not entry.get("units"):
            continue
        for slot in DECISION_TIMES.get(name, ()):
            due = datetime.combine(local_now.date(), time.fromisoformat(slot), EASTERN)
            if not session[0] <= due < session[1] or local_now < due + DECISION_GRACE:
                continue
            decided = _decided_at(entry)
            if decided is None or decided < due:
                missed.append(f"{name} {slot}")
    if missed:
        return Check("Decisions on schedule", "warning", f"Missed: {', '.join(missed)}")
    return Check("Decisions on schedule", "normal", "On schedule")


def rejects_check(strategies: dict) -> Check:
    rejected = sum(_metric_values(strategies, "counters", "hg_orders_rejected_total"))
    if rejected > 0:
        return Check("Order rejects", "caution", f"{int(rejected)} rejected since the process started")
    return Check("Order rejects", "normal", "None")


def drawdown_check(strategies: dict) -> Check:
    drawdowns = _metric_values(strategies, "gauges", "hg_portfolio_drawdown_pct")
    if not drawdowns:
        return Check("Drawdown", "normal", "Not reported")
    worst = min(drawdowns)
    if worst <= DRAWDOWN_WARNING_PCT:
        return Check("Drawdown", "warning", f"{worst:.1f}%")
    if worst <= DRAWDOWN_CAUTION_PCT:
        return Check("Drawdown", "caution", f"{worst:.1f}%")
    return Check("Drawdown", "normal", f"{worst:.1f}%")


def memory_check(units: list[dict]) -> Check:
    unit = _unit(units, TRADING_UNIT)
    if unit is None or unit.get("memory_bytes") is None:
        return Check("Host memory", "normal", "Not reported")
    fraction = unit["memory_bytes"] / TRADING_UNIT_MEMORY_CAP_BYTES
    detail = f"{TRADING_UNIT} at {fraction:.0%} of 1G"
    if fraction > MEMORY_WARNING:
        return Check("Host memory", "warning", detail)
    if fraction > MEMORY_CAUTION:
        return Check("Host memory", "caution", detail)
    return Check("Host memory", "normal", detail)
