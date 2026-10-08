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
                 "snapshot": {"gauges": {"hg_portfolio_equity_usd": {"{}": 101234.0},
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
