"""Template contexts from the ConsoleState. Pure: state and time in, plain dicts out."""
from __future__ import annotations

from datetime import date, datetime, time

from src.console_agent.status import READ_ERRORS, summarize_decision
from tools.console import freshness
from tools.console.checks import run_checks
from tools.console.decision_times import DECISION_TIMES
from tools.console.poller import ConsoleState
from tools.console.schedule import EASTERN, expected_state, instance_window, next_event, nyse_session, power_lock, power_status

PANELS = ("header", "schedule", "exceptions", "strategies", "account", "host")
RAIL_START_HOUR = 6
RAIL_END_HOUR = 22
RAIL_WIDTH = 1000
START_LAMBDA_LOG_GROUP = "/aws/lambda/homeguard-start-instance"
ACCOUNT_GAUGES = (
    ("Equity", "hg_portfolio_equity_usd", "${:,.0f}", freshness.MARKET_OWNED),
    ("Day P&L", "hg_portfolio_day_pnl_usd", "${:+,.0f}", freshness.MARKET_OWNED),
    ("Drawdown", "hg_portfolio_drawdown_pct", "{:.1f}%", freshness.MARKET_OWNED),
    ("Positions", "hg_strategy_positions_count", "{:.0f}", freshness.INSTANCE_OWNED),
)


def page_context(state: ConsoleState, now: datetime, region: str) -> dict:
    document = state.document or {}
    live = state.is_live(now)
    return {
        "header": header_context(state, now, live),
        "rail": rail_context(state, now),
        "power": power_context(state, now, region),
        "checks": run_checks(document, live, state.as_of, now) if state.document else None,
        "strategies": strategy_rows(state),
        "account": account_rows(state, now, live),
        "units": document.get("units") or [],
        "live": live,
        "stamp": stamp_text(state, live),
        "has_document": state.document is not None,
    }


def stamp_text(state: ConsoleState, live: bool) -> str | None:
    if live or state.document is None:
        return None
    taken = state.as_of.astimezone(EASTERN).strftime("%H:%M:%S ET")
    if state.source != "s3":
        return f"as of {taken}, last agent reading"
    reason = f" {state.reason}" if state.reason else ""
    return f"as of {taken}, S3{reason} snapshot"


def header_context(state: ConsoleState, now: datetime, live: bool) -> dict:
    errors = [f"{source}: {error}" for source, error in sorted(state.errors.items())]
    errors += [f"{e.get('source')}: {e.get('error')}" for e in (state.document or {}).get("errors") or []]
    if state.document is None:
        return {"badge": "No status yet", "level": "unknown", "errors": errors}
    age = freshness.age_text(state.as_of, now)
    if live:
        return {"badge": f"Live from agent, {age}", "level": "normal", "errors": errors}
    if state.source == "s3":
        taken = state.as_of.astimezone(EASTERN).strftime("%Y-%m-%d %H:%M:%S ET")
        reason = f" ({state.reason})" if state.reason else ""
        return {"badge": f"Snapshot from S3, taken {taken}{reason}, {age}", "level": "caution", "errors": errors}
    if state.agent_down_since is None:
        return {"badge": f"Agent reading {age.removesuffix(' ago')} old", "level": "caution", "errors": errors}
    return {"badge": f"Agent unreachable; last agent reading {age}", "level": "caution", "errors": errors}


def _rail_x(moment: datetime, day: date) -> float:
    start = datetime.combine(day, time(RAIL_START_HOUR), EASTERN)
    end = datetime.combine(day, time(RAIL_END_HOUR), EASTERN)
    fraction = (moment - start) / (end - start)
    return round(RAIL_WIDTH * min(max(fraction, 0.0), 1.0), 1)


def _band(cls: str, label: str, span: tuple[datetime, datetime], y: int, day: date) -> dict:
    x = _rail_x(span[0], day)
    return {"cls": cls, "label": f"{label} {span[0]:%H:%M}-{span[1]:%H:%M}", "x": x,
            "w": round(_rail_x(span[1], day) - x, 1), "y": y, "h": 10}


def rail_context(state: ConsoleState, now: datetime) -> dict:
    local_now = now.astimezone(EASTERN)
    day = local_now.date()
    bands = []
    for cls, label, span, y in (
        ("instance", "Instance window", instance_window(state.schedules, day), 12),
        ("lock", "Power lock", power_lock(day), 26),
        ("session", "NYSE session", nyse_session(day), 40),
    ):
        if span is not None:
            bands.append(_band(cls, label, span, y, day))
    ticks = []
    strategies = (state.document or {}).get("strategies") or {}
    if nyse_session(day) is not None:
        for name, entry in sorted(strategies.items()):
            if entry.get("units"):
                for slot in DECISION_TIMES.get(name, ()):
                    moment = datetime.combine(day, time.fromisoformat(slot), EASTERN)
                    ticks.append({"x": _rail_x(moment, day), "label": f"{name} {slot}"})
    hours = [{"x": _rail_x(datetime.combine(day, time(hour), EASTERN), day), "label": f"{hour:02d}"}
             for hour in range(RAIL_START_HOUR, RAIL_END_HOUR + 1, 2)]
    rail_start = datetime.combine(day, time(RAIL_START_HOUR), EASTERN)
    rail_end = datetime.combine(day, time(RAIL_END_HOUR), EASTERN)
    now_x = _rail_x(local_now, day) if rail_start <= local_now <= rail_end else None
    return {"bands": bands, "ticks": ticks, "hours": hours, "now_x": now_x}


def power_context(state: ConsoleState, now: datetime, region: str) -> dict:
    down_for = now - state.agent_down_since if state.agent_down_since else None
    status = power_status(state.instance_state, expected_state(state.schedules, now), down_for)
    upcoming = next_event(state.schedules, now)
    log_url = None
    if status.code == "missed_start":
        group = START_LAMBDA_LOG_GROUP.replace("/", "$252F")
        log_url = f"https://console.aws.amazon.com/cloudwatch/home?region={region}#logsV2:log-groups/log-group/{group}"
    return {
        "level": status.level,
        "text": status.text,
        "next": f"Next scheduled {upcoming[0]}: {upcoming[1]:%a %H:%M} ET" if upcoming else None,
        "log_url": log_url,
    }


def _short_time(timestamp: str | None) -> str:
    try:
        moment = datetime.fromisoformat(timestamp)
    except (TypeError, ValueError):
        return "-"
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=EASTERN)
    return moment.astimezone(EASTERN).strftime("%m-%d %H:%M")


def _unit_text(name: str, unit: dict | None) -> str:
    if unit is None:
        return f"{name}: not reported"
    return f"{name}: {unit.get('active_state')} ({unit.get('sub_state')})"


def strategy_rows(state: ConsoleState) -> list[dict]:
    reported = {unit.get("unit"): unit for unit in (state.document or {}).get("units") or []}
    rows = []
    for name, entry in sorted(((state.document or {}).get("strategies") or {}).items()):
        decision = entry.get("last_decision") or {}
        listed = entry.get("units") or []
        running = any((reported.get(unit) or {}).get("active_state") == "active" for unit in listed)
        rows.append({
            "name": name,
            "enabled": entry.get("enabled"),
            "process": ", ".join(_unit_text(unit, reported.get(unit)) for unit in listed) or "no unit",
            "variant": entry.get("variant"),
            "decided_at": _short_time(decision.get("timestamp")),
            "passed": decision.get("all_passed"),
            "caution": bool(entry.get("enabled")) and not running,
            "has_gates": state.decisions.get(name) is not None,
        })
    return rows


def _first_value(values: dict | None, template: str) -> str:
    if not values:
        return "-"
    return template.format(next(iter(values.values())))


def _reading_time(entry: dict, as_of: datetime) -> datetime:
    taken = freshness.snapshot_time(entry)
    return min(taken, as_of) if taken else as_of


def account_rows(state: ConsoleState, now: datetime, live: bool) -> list[dict]:
    rows = []
    for name, entry in sorted(((state.document or {}).get("strategies") or {}).items()):
        gauges = (entry.get("snapshot") or {}).get("gauges") or {}
        if not entry.get("units") or not gauges:
            continue
        row_live = live and freshness.snapshot_is_live(entry, now)
        cells = [{"label": label, "value": _first_value(gauges.get(metric), template),
                  "cls": freshness.classify(kind, row_live)}
                 for label, metric, template, kind in ACCOUNT_GAUGES]
        age = "" if row_live else freshness.age_text(_reading_time(entry, state.as_of), now)
        rows.append({"name": name, "cells": cells, "age": age})
    return rows


def gates_context(state: ConsoleState, strategy: str) -> dict | None:
    record = state.decisions.get(strategy)
    if record is None:
        return None
    try:
        summary = summarize_decision(record)
    except READ_ERRORS as e:
        return {"strategy": strategy, "summary": None, "gates": {}, "error": repr(e)}
    return {"strategy": strategy, "summary": summary, "gates": summary["gates"], "error": None}
