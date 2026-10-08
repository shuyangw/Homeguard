from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from tools.console.schedule import (
    EASTERN, SCHEDULE_ACTIONS, expected_state, instance_window, next_event, nyse_session,
    parse_cron, power_lock, power_status,
)

UTC = ZoneInfo("UTC")
REAL = [
    ("homeguard-start-instance", "cron(0 8 ? * MON-FRI *)", "America/New_York"),
    ("homeguard-stop-instance", "cron(0 20 ? * MON-FRI *)", "America/New_York"),
    ("homeguard-start-instance-sunday", "cron(0 23 ? * SAT *)", "UTC"),
    ("homeguard-stop-instance-sunday", "cron(10 0 ? * SUN *)", "UTC"),
]
SCHEDULES = [parse_cron(name, SCHEDULE_ACTIONS[name], expr, zone) for name, expr, zone in REAL]


def et(*args):
    return datetime(*args, tzinfo=EASTERN)


def utc(*args):
    return datetime(*args, tzinfo=UTC)


def test_the_four_real_expressions_parse():
    assert [s.weekdays for s in SCHEDULES] == [frozenset(range(5)), frozenset(range(5)), frozenset({5}), frozenset({6})]
    assert (SCHEDULES[3].hour, SCHEDULES[3].minute) == (0, 10)


@pytest.mark.parametrize("expression", [
    "cron(0 8 * * MON-FRI *)", "cron(0/5 8 ? * MON-FRI *)", "cron(0 8 ? * 2-6 *)",
    "cron(0 8 ? * FRI-MON *)", "rate(5 minutes)", "cron(0 8 ? JAN MON *)",
])
def test_unsupported_expressions_raise(expression):
    with pytest.raises(ValueError):
        parse_cron("x", "start", expression, "UTC")


@pytest.mark.parametrize("now,expected", [
    (et(2026, 10, 8, 7, 59), "stopped"),
    (et(2026, 10, 8, 8, 0), "running"),
    (et(2026, 10, 8, 19, 59), "running"),
    (et(2026, 10, 8, 20, 0), "stopped"),
    (utc(2026, 10, 10, 23, 30), "running"),
    (utc(2026, 10, 11, 0, 10), "stopped"),
    (et(2026, 10, 11, 12, 0), "stopped"),
    (et(2026, 11, 26, 12, 0), "running"),
], ids=["weekday-0759", "weekday-0800", "weekday-1959", "weekday-2000", "sat-2330utc", "sun-0010utc",
        "sunday-noon", "thanksgiving"])
def test_expected_state(now, expected):
    assert expected_state(SCHEDULES, now) == expected


def test_expected_state_across_the_november_dst_change():
    assert expected_state(SCHEDULES, et(2026, 11, 2, 7, 59)) == "stopped"
    assert expected_state(SCHEDULES, et(2026, 11, 2, 8, 0)) == "running"
    assert instance_window(SCHEDULES, date(2026, 10, 31)) == (et(2026, 10, 31, 19, 0), et(2026, 10, 31, 20, 10))
    assert instance_window(SCHEDULES, date(2026, 11, 7)) == (et(2026, 11, 7, 18, 0), et(2026, 11, 7, 19, 10))


def test_no_schedules_means_unknown_expected_state():
    assert expected_state([], et(2026, 10, 8, 12, 0)) is None
    assert next_event([], et(2026, 10, 8, 12, 0)) is None


def test_next_event_after_the_evening_stop_is_the_next_morning_start():
    action, when = next_event(SCHEDULES, et(2026, 10, 8, 20, 30))

    assert action == "start"
    assert when == et(2026, 10, 9, 8, 0)


def test_weekday_instance_window():
    assert instance_window(SCHEDULES, date(2026, 10, 8)) == (et(2026, 10, 8, 8, 0), et(2026, 10, 8, 20, 0))
    assert instance_window(SCHEDULES, date(2026, 10, 11)) is None


def test_session_and_power_lock_on_regular_early_close_and_holiday_days():
    assert nyse_session(date(2026, 10, 8)) == (et(2026, 10, 8, 9, 30), et(2026, 10, 8, 16, 0))
    assert power_lock(date(2026, 10, 8)) == (et(2026, 10, 8, 9, 15), et(2026, 10, 8, 16, 15))
    assert power_lock(date(2026, 11, 27))[1] == et(2026, 11, 27, 13, 15)
    assert nyse_session(date(2026, 11, 26)) is None
    assert power_lock(date(2026, 11, 26)) is None


@pytest.mark.parametrize("actual,expected,down_for,code,level", [
    ("stopped", "stopped", None, "as_scheduled", "normal"),
    ("running", "running", None, "as_scheduled", "normal"),
    ("stopped", "running", None, "missed_start", "warning"),
    ("running", "stopped", None, "off_schedule", "caution"),
    ("running", "running", timedelta(minutes=3), "agent_unreachable", "warning"),
    ("running", "stopped", timedelta(minutes=3), "agent_unreachable", "warning"),
    ("running", "running", timedelta(minutes=1), "as_scheduled", "normal"),
    ("pending", "running", None, "transition", "caution"),
    (None, "running", None, "unknown", "unknown"),
    ("running", None, None, "unknown", "unknown"),
])
def test_power_status(actual, expected, down_for, code, level):
    status = power_status(actual, expected, down_for)

    assert (status.code, status.level) == (code, level)
    assert status.text
