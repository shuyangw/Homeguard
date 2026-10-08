import copy
from datetime import datetime, timedelta, timezone

from tools.console.poller import ConsoleState
from tools.console.schedule import EASTERN, SCHEDULE_ACTIONS, parse_cron
from tools.console.views import gates_context, page_context

NOW = datetime(2026, 10, 8, 16, 30, tzinfo=timezone.utc)  # 12:30 ET
SCHEDULES = [
    parse_cron("homeguard-start-instance", "start", "cron(0 8 ? * MON-FRI *)", "America/New_York"),
    parse_cron("homeguard-stop-instance", "stop", "cron(0 20 ? * MON-FRI *)", "America/New_York"),
]
DOCUMENT = {
    "units": [{"unit": "homeguard-multi.service", "active_state": "active", "sub_state": "running",
               "restarts": 0, "memory_bytes": 1 << 29, "active_since": "Thu 2026-10-08 08:01:12 EDT"}],
    "strategies": {
        "ramp": {"enabled": False, "variant": "v11", "units": ["homeguard-multi.service"],
                 "snapshot": {"timestamp": NOW.timestamp() - 5,
                              "gauges": {"hg_portfolio_equity_usd": {"{}": 101234.0},
                                         "hg_strategy_positions_count": {'{"strategy": "ramp"}': 26.0}}},
                 "last_decision": {"timestamp": "2026-10-07T15:55:04-04:00", "all_passed": False}},
        "mp": {"enabled": True, "variant": "v01", "units": [], "snapshot": None, "last_decision": None},
    },
    "errors": [{"source": "snapshot:cscm", "error": "FileNotFoundError()"}],
}
RECORD = {"decision_id": "ramp-1", "timestamp": "2026-10-07T15:55:04-04:00",
          "trigger": {"kind": "scheduled_rebalance", "schedule_time": "15:55"},
          "preconditions": {"all_passed": False, "strategy_enabled": {"passed": False, "error": "disabled"}, "extra": {}}}


def live_state(**changes):
    state = ConsoleState(document=DOCUMENT, source="agent", as_of=NOW - timedelta(seconds=4),
                         instance_state="running", schedules=SCHEDULES, decisions={"ramp": RECORD})
    for name, value in changes.items():
        setattr(state, name, value)
    return state


def test_live_header_badge_and_error_strip():
    header = page_context(live_state(), NOW, "us-east-1")["header"]

    assert header["badge"] == "Live from agent, 4 s ago"
    assert header["level"] == "normal"
    assert "snapshot:cscm: FileNotFoundError()" in header["errors"]


def test_s3_header_badge_names_the_upload_and_reason():
    uploaded = datetime(2026, 10, 8, 0, 0, 12, tzinfo=timezone.utc)
    state = live_state(source="s3", as_of=uploaded, reason="shutdown", agent_down_since=NOW)

    header = page_context(state, NOW, "us-east-1")["header"]

    assert header["badge"].startswith("Snapshot from S3, taken 2026-10-07 20:00:12 ET (shutdown)")
    assert header["level"] == "caution"


def test_no_document_shows_no_status_and_no_checks():
    context = page_context(ConsoleState(), NOW, "us-east-1")

    assert context["header"]["badge"] == "No status yet"
    assert context["checks"] is None


def test_rail_has_instance_lock_and_session_bands_and_a_ramp_tick():
    rail = page_context(live_state(), NOW, "us-east-1")["rail"]

    assert [band["cls"] for band in rail["bands"]] == ["instance", "lock", "session"]
    assert [tick["label"] for tick in rail["ticks"]] == ["ramp 15:55"]
    assert rail["now_x"] is not None


def test_missed_start_power_tile_links_the_start_lambda_log():
    state = live_state(instance_state="stopped", source="s3", agent_down_since=NOW)

    power = page_context(state, NOW, "us-east-1")["power"]

    assert power["level"] == "warning"
    assert "homeguard-start-instance" in power["log_url"]


def test_strategy_rows_flag_a_switch_on_with_no_unit():
    rows = {row["name"]: row for row in page_context(live_state(), NOW, "us-east-1")["strategies"]}

    assert rows["mp"]["caution"] is True
    assert rows["mp"]["process"] == "no unit"
    assert rows["ramp"]["caution"] is False
    assert rows["ramp"]["decided_at"] == "10-07 15:55"


def test_account_values_drift_on_the_snapshot_and_positions_freeze():
    state = live_state(source="s3", agent_down_since=NOW, as_of=NOW - timedelta(hours=1))

    rows = page_context(state, NOW, "us-east-1")["account"]

    cells = {cell["label"]: cell for cell in rows[0]["cells"]}
    assert cells["Equity"]["value"] == "$101,234"
    assert cells["Equity"]["cls"] == "drifting"
    assert cells["Positions"]["cls"] == "frozen"
    assert rows[0]["age"] == "1 h 0 min ago"


def test_gates_context_summarizes_the_full_record():
    context = gates_context(live_state(), "ramp")

    assert context["gates"]["strategy_enabled"] == {"passed": False, "error": "disabled"}
    assert gates_context(live_state(), "nope") is None


def test_an_old_agent_reading_is_not_live():
    state = live_state(as_of=NOW - timedelta(hours=9))

    context = page_context(state, NOW, "us-east-1")

    assert context["header"]["level"] == "caution"
    assert context["header"]["badge"] == "Agent reading 9 h 0 min old"
    assert context["live"] is False
    measured = {c.name: c.level for c in context["checks"] if c.name in ("Broker heartbeat", "Host memory")}
    assert measured == {"Broker heartbeat": "unknown", "Host memory": "unknown"}


def stale_snapshot_state():
    document = copy.deepcopy(DOCUMENT)
    snapshot = document["strategies"]["ramp"]["snapshot"]
    snapshot["timestamp"] = NOW.timestamp() - 300
    snapshot["gauges"]["hg_broker_last_heartbeat_timestamp"] = {"{}": NOW.timestamp() - 300}
    snapshot["gauges"]["hg_websocket_connected"] = {"{}": 1.0}
    return live_state(document=document)


def test_a_stale_snapshot_from_a_live_agent_is_not_live():
    context = page_context(stale_snapshot_state(), NOW, "us-east-1")

    checks = {c.name: c for c in context["checks"]}
    for name in ("Broker heartbeat", "Market data stream"):
        assert checks[name].level == "unknown"
        assert checks[name].detail.startswith("Unknown since 12:25; last reading:")
    cells = {cell["label"]: cell for cell in context["account"][0]["cells"]}
    assert cells["Equity"]["cls"] == "drifting"
    assert context["account"][0]["age"] == "5 min ago"


def test_a_strategy_without_a_unit_has_no_account_row():
    document = copy.deepcopy(DOCUMENT)
    document["strategies"]["mp"]["snapshot"] = {"timestamp": NOW.timestamp(),
                                                "gauges": {"hg_portfolio_equity_usd": {"{}": 5.0}}}

    rows = page_context(live_state(document=document), NOW, "us-east-1")["account"]

    assert [row["name"] for row in rows] == ["ramp"]


def strategy_row(unit_state):
    document = copy.deepcopy(DOCUMENT)
    document["strategies"]["ramp"]["enabled"] = True
    document["units"][0]["active_state"], document["units"][0]["sub_state"] = unit_state
    rows = page_context(live_state(document=document), NOW, "us-east-1")["strategies"]
    return next(row for row in rows if row["name"] == "ramp")


def test_a_failed_unit_shows_its_state_and_a_caution():
    row = strategy_row(("failed", "failed"))

    assert row["process"] == "homeguard-multi.service: failed (failed)"
    assert row["caution"] is True


def test_an_active_unit_shows_running_and_no_caution():
    row = strategy_row(("active", "running"))

    assert row["process"] == "homeguard-multi.service: active (running)"
    assert row["caution"] is False


def test_stamp_names_the_source_and_time_when_not_live():
    uploaded = datetime(2026, 10, 8, 0, 0, 12, tzinfo=timezone.utc)
    s3 = live_state(source="s3", as_of=uploaded, reason="shutdown", agent_down_since=NOW)
    agent = live_state(as_of=uploaded, agent_down_since=NOW)

    assert page_context(live_state(), NOW, "us-east-1")["stamp"] is None
    assert page_context(s3, NOW, "us-east-1")["stamp"] == "as of 20:00:12 ET, S3 shutdown snapshot"
    assert page_context(agent, NOW, "us-east-1")["stamp"] == "as of 20:00:12 ET, last agent reading"


def morning_power(hour, minute, instance_state, launched_at=None):
    state = live_state(instance_state=instance_state, source="s3", launched_at=launched_at,
                       agent_down_since=datetime(2026, 10, 7, 20, 0, tzinfo=EASTERN))
    return page_context(state, datetime(2026, 10, 8, hour, minute, tzinfo=EASTERN), "us-east-1")["power"]


def test_the_agent_down_clock_starts_at_launch_not_at_last_nights_stop():
    launched = datetime(2026, 10, 8, 8, 0, 30, tzinfo=EASTERN)

    assert morning_power(8, 1, "running", launched)["text"] != "Instance is up but the console cannot reach the agent"
    assert morning_power(8, 4, "running", launched)["text"] == "Instance is up but the console cannot reach the agent"


def test_a_scheduled_start_has_five_minutes_before_it_is_missed():
    assert morning_power(8, 2, "stopped")["log_url"] is None
    assert morning_power(8, 6, "stopped")["log_url"] is not None
