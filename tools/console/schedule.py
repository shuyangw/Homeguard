"""Expected instance state from the EventBridge schedules, and the NYSE session.

Only the cron forms the Homeguard schedules use are supported (fixed minute and
hour, '?' day of month, '*' month and year, named days of week); anything else
raises instead of guessing.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from functools import lru_cache
from zoneinfo import ZoneInfo

import pandas_market_calendars as mcal

EASTERN = ZoneInfo("America/New_York")
POWER_LOCK_BUFFER = timedelta(minutes=15)
AGENT_UNREACHABLE_AFTER = timedelta(minutes=2)
START_GRACE = timedelta(minutes=5)
SCHEDULE_ACTIONS = {
    "homeguard-start-instance": "start",
    "homeguard-stop-instance": "stop",
    "homeguard-start-instance-sunday": "start",
    "homeguard-stop-instance-sunday": "stop",
}
_DAYS = {"MON": 0, "TUE": 1, "WED": 2, "THU": 3, "FRI": 4, "SAT": 5, "SUN": 6}
_CRON = re.compile(r"^cron\((\d{1,2}) (\d{1,2}) \? \* ([A-Z,\-]+) \*\)$")


@dataclass(frozen=True)
class CronSchedule:
    name: str
    action: str
    minute: int
    hour: int
    weekdays: frozenset[int]
    zone: ZoneInfo


@dataclass(frozen=True)
class PowerStatus:
    level: str
    text: str
    code: str


def parse_cron(name: str, action: str, expression: str, timezone: str) -> CronSchedule:
    match = _CRON.match(expression)
    if match is None:
        raise ValueError(f"{name}: unsupported schedule expression {expression!r}")
    minute, hour, days = match.groups()
    return CronSchedule(name, action, int(minute), int(hour), _parse_weekdays(name, days), ZoneInfo(timezone))


def _parse_weekdays(name: str, field: str) -> frozenset[int]:
    weekdays: set[int] = set()
    for part in field.split(","):
        first, _, last = part.partition("-")
        last = last or first
        if first not in _DAYS or last not in _DAYS or _DAYS[last] < _DAYS[first]:
            raise ValueError(f"{name}: unsupported day-of-week field {field!r}")
        weekdays.update(range(_DAYS[first], _DAYS[last] + 1))
    return frozenset(weekdays)


def _fire_on_local_day(schedule: CronSchedule, local_day: date) -> datetime | None:
    if local_day.weekday() not in schedule.weekdays:
        return None
    return datetime.combine(local_day, time(schedule.hour, schedule.minute), schedule.zone)


def last_fire(schedule: CronSchedule, now: datetime) -> datetime:
    local_now = now.astimezone(schedule.zone)
    for days_back in range(8):
        fire = _fire_on_local_day(schedule, local_now.date() - timedelta(days=days_back))
        if fire is not None and fire <= local_now:
            return fire
    raise ValueError(f"{schedule.name} has no fire time in the last week")


def next_fire(schedule: CronSchedule, now: datetime) -> datetime:
    local_now = now.astimezone(schedule.zone)
    for days_ahead in range(8):
        fire = _fire_on_local_day(schedule, local_now.date() + timedelta(days=days_ahead))
        if fire is not None and fire > local_now:
            return fire
    raise ValueError(f"{schedule.name} has no fire time in the next week")


def expected_state(schedules: list[CronSchedule], now: datetime) -> str | None:
    if not schedules:
        return None
    latest = max(schedules, key=lambda schedule: last_fire(schedule, now))
    return "running" if latest.action == "start" else "stopped"


def last_start(schedules: list[CronSchedule], now: datetime) -> datetime | None:
    return max((last_fire(schedule, now) for schedule in schedules if schedule.action == "start"), default=None)


def next_event(schedules: list[CronSchedule], now: datetime) -> tuple[str, datetime] | None:
    if not schedules:
        return None
    upcoming = min(schedules, key=lambda schedule: next_fire(schedule, now))
    return upcoming.action, next_fire(upcoming, now).astimezone(EASTERN)


def fires_on(schedule: CronSchedule, day: date) -> list[datetime]:
    """Fire times of a schedule that fall on the given Eastern calendar day."""
    fires = []
    for offset in (-1, 0, 1):
        fire = _fire_on_local_day(schedule, day + timedelta(days=offset))
        if fire is not None and fire.astimezone(EASTERN).date() == day:
            fires.append(fire.astimezone(EASTERN))
    return fires


def instance_window(schedules: list[CronSchedule], day: date) -> tuple[datetime, datetime] | None:
    starts = [fire for s in schedules if s.action == "start" for fire in fires_on(s, day)]
    stops = [fire for s in schedules if s.action == "stop" for fire in fires_on(s, day)]
    if not starts or not stops:
        return None
    return min(starts), max(stops)


@lru_cache(maxsize=16)
def nyse_session(day: date) -> tuple[datetime, datetime] | None:
    schedule = mcal.get_calendar("NYSE").schedule(start_date=day, end_date=day)
    if schedule.empty:
        return None
    row = schedule.iloc[0]
    return row["market_open"].to_pydatetime().astimezone(EASTERN), row["market_close"].to_pydatetime().astimezone(EASTERN)


def power_lock(day: date) -> tuple[datetime, datetime] | None:
    session = nyse_session(day)
    if session is None:
        return None
    return session[0] - POWER_LOCK_BUFFER, session[1] + POWER_LOCK_BUFFER


def power_status(actual: str | None, expected: str | None, agent_down_for: timedelta | None,
                 since_start: timedelta | None = None) -> PowerStatus:
    if actual is None or expected is None:
        return PowerStatus("unknown", "Instance state unknown", "unknown")
    if actual == "running" and agent_down_for is not None and agent_down_for > AGENT_UNREACHABLE_AFTER:
        return PowerStatus("warning", "Instance is up but the console cannot reach the agent", "agent_unreachable")
    start_overdue = since_start is None or since_start > START_GRACE
    if actual == "stopped" and expected == "running" and start_overdue:
        return PowerStatus("warning", "Instance should be running", "missed_start")
    if actual == "running" and expected == "stopped":
        return PowerStatus("caution", "Instance is running outside its schedule", "off_schedule")
    if actual == expected:
        return PowerStatus("normal", f"Instance {actual}, as scheduled", "as_scheduled")
    return PowerStatus("caution", f"Instance {actual} (expected {expected})", "transition")
