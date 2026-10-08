from datetime import datetime, timedelta, timezone

import pytest

from tools.console.checks import run_checks
from tools.console.schedule import EASTERN

GIB = 1 << 30


def et(*args):
    return datetime(*args, tzinfo=EASTERN)


def document(*, gateway="active", heartbeat_age=10.0, websocket=None, rejected=0, drawdown=None,
             memory=GIB // 2, decided_at="2026-10-08T15:55:04-04:00", now=et(2026, 10, 8, 12, 0), ramp_units=True):
    gauges = {"hg_broker_last_heartbeat_timestamp": {"{}": now.timestamp() - heartbeat_age}}
    if websocket is not None:
        gauges["hg_websocket_connected"] = {"{}": websocket}
    if drawdown is not None:
        gauges["hg_portfolio_drawdown_pct"] = {"{}": drawdown}
    return {
        "units": [
            {"unit": "homeguard-gateway.service", "active_state": gateway, "sub_state": "running", "memory_bytes": None},
            {"unit": "homeguard-multi.service", "active_state": "active", "sub_state": "running", "memory_bytes": memory},
        ],
        "strategies": {
            "ramp": {
                "units": ["homeguard-multi.service"] if ramp_units else [],
                "snapshot": {"gauges": gauges, "counters": {"hg_orders_rejected_total": {'{"reason": "x"}': rejected}}},
                "last_decision": {"timestamp": decided_at, "all_passed": True} if decided_at else None,
            },
            "mp": {"units": [], "snapshot": None, "last_decision": None},
        },
    }


def levels(doc, now, live=True):
    return {check.name: check.level for check in run_checks(doc, live, now, now)}


def check(doc, now, name, live=True):
    return next(c for c in run_checks(doc, live, now, now) if c.name == name)


NOON = et(2026, 10, 8, 12, 0)


def test_healthy_document_is_all_normal_except_scrape():
    result = levels(document(), NOON)

    assert len(result) == 8
    assert result.pop("Metrics scrape") == "unknown"
    assert set(result.values()) == {"normal"}


def test_gateway_not_active_is_a_warning():
    assert levels(document(gateway="failed"), NOON)["IB Gateway"] == "warning"


def test_gateway_missing_from_units_is_a_warning():
    doc = document()
    doc["units"] = doc["units"][1:]
    assert levels(doc, NOON)["IB Gateway"] == "warning"


@pytest.mark.parametrize("age,level", [(119, "normal"), (121, "warning")])
def test_heartbeat_age_boundary(age, level):
    assert levels(document(heartbeat_age=age), NOON)["Broker heartbeat"] == level


def test_absent_heartbeat_reads_as_not_reported():
    doc = document()
    doc["strategies"]["ramp"]["snapshot"]["gauges"].pop("hg_broker_last_heartbeat_timestamp")
    result = check(doc, NOON, "Broker heartbeat")
    assert (result.level, result.detail) == ("normal", "Not reported")


def test_stream_disconnected_in_session_is_a_caution_and_after_close_is_normal():
    assert levels(document(websocket=0), NOON)["Market data stream"] == "caution"
    assert levels(document(websocket=0), et(2026, 10, 8, 17, 0))["Market data stream"] == "normal"
    assert check(document(), NOON, "Market data stream").detail == "Not reported"


@pytest.mark.parametrize("value,level", [(-9.9, "normal"), (-10.0, "caution"), (-19.9, "caution"), (-20.0, "warning")])
def test_drawdown_uses_negative_bounds(value, level):
    assert levels(document(drawdown=value), NOON)["Drawdown"] == level


def test_positive_drawdown_value_never_alerts():
    assert levels(document(drawdown=25.0), NOON)["Drawdown"] == "normal"


@pytest.mark.parametrize("fraction,level", [(0.80, "normal"), (0.81, "caution"), (0.95, "caution"), (0.96, "warning")])
def test_memory_boundaries(fraction, level):
    assert levels(document(memory=int(GIB * fraction)), NOON)["Host memory"] == level


def test_rejects_since_process_start_are_a_caution_with_the_count():
    result = check(document(rejected=3), NOON, "Order rejects")
    assert result.level == "caution"
    assert "3" in result.detail


def test_missed_rebalance_after_grace_is_a_warning():
    doc = document(decided_at="2026-10-07T15:55:04-04:00")
    assert levels(doc, et(2026, 10, 8, 15, 59))["Decisions on schedule"] == "normal"
    assert levels(doc, et(2026, 10, 8, 16, 0, 1))["Decisions on schedule"] == "warning"


def test_strategy_without_a_unit_is_not_expected_to_decide():
    doc = document(decided_at=None, ramp_units=False)
    assert levels(doc, et(2026, 10, 8, 17, 0))["Decisions on schedule"] == "normal"


def test_no_decision_expectation_on_a_holiday_or_after_an_early_close():
    doc = document(decided_at="2026-10-01T15:55:04-04:00")
    assert levels(doc, et(2026, 11, 26, 17, 0))["Decisions on schedule"] == "normal"
    assert levels(doc, et(2026, 11, 27, 17, 0))["Decisions on schedule"] == "normal"


def test_naive_decision_timestamp_is_read_as_eastern():
    doc = document(decided_at="2026-10-08T15:55:04")
    assert levels(doc, et(2026, 10, 8, 17, 0))["Decisions on schedule"] == "normal"


def test_unparseable_decision_timestamp_reads_as_missed():
    doc = document(decided_at="not a time")
    assert levels(doc, et(2026, 10, 8, 17, 0))["Decisions on schedule"] == "warning"


def test_measured_checks_become_unknown_on_a_snapshot_and_keep_the_last_reading():
    as_of = et(2026, 10, 8, 20, 0)
    results = {c.name: c for c in run_checks(document(gateway="failed"), False, as_of, as_of + timedelta(hours=3))}

    for name in ("IB Gateway", "Broker heartbeat", "Market data stream", "Host memory"):
        assert results[name].level == "unknown"
        assert results[name].detail.startswith("Unknown since 20:00")
    assert "failed" in results["IB Gateway"].detail
    assert results["Order rejects"].level == "normal"
    assert results["Decisions on schedule"].level == "normal"


def test_document_without_strategies_or_units_still_returns_eight_checks():
    assert len(run_checks({}, True, NOON, NOON)) == 8


def two_strategy_document(ramp_age, cscm_age, now=NOON):
    doc = document(heartbeat_age=ramp_age, now=now)
    doc["strategies"]["cscm"] = {
        "units": ["homeguard-cscm.service"],
        "snapshot": {"gauges": {"hg_broker_last_heartbeat_timestamp": {"{}": now.timestamp() - cscm_age}}},
        "last_decision": None,
    }
    return doc


def test_a_stale_strategy_heartbeat_is_not_masked_by_a_fresh_one():
    result = check(two_strategy_document(900, 10), NOON, "Broker heartbeat")

    assert result.level == "warning"
    assert result.detail == "ramp: last heartbeat 900 s ago"


def test_fresh_heartbeats_list_each_strategy():
    result = check(two_strategy_document(12, 8), NOON, "Broker heartbeat")

    assert (result.level, result.detail) == ("normal", "cscm 8 s, ramp 12 s")


def test_a_heartbeat_ahead_of_the_local_clock_reads_zero_seconds():
    assert check(document(heartbeat_age=-3), NOON, "Broker heartbeat").detail == "ramp 0 s"
