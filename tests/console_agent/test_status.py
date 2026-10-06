"""Unit tests for the on-disk readers and the /status document."""

import json
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from src.console_agent import status, units
from tests.console_agent.conftest import make_decision_record

NOW = datetime(2026, 10, 5, 14, 12, tzinfo=timezone(timedelta(hours=-4)))


def write_lock(state_path, lock):
    state_path.write_text(json.dumps({"version": 1, "strategies": {}, "execution_lock": lock}))


def test_config_from_env_reads_and_strips_the_login():
    config = status.config_from_env({"CONSOLE_OPERATOR_LOGIN": " op@example.com \n"}, Path("/r"), Path("/s"))
    assert config.operator_login == "op@example.com"
    assert config.toggle_path == Path("/r/config/trading/strategy_toggle.yaml")
    assert config.state_path == Path("/r/data/trading/strategy_positions.json")
    assert config.latest_dir == Path("/r/data/trading/decisions/_latest")


@pytest.mark.parametrize("environ", [{}, {"CONSOLE_OPERATOR_LOGIN": ""}, {"CONSOLE_OPERATOR_LOGIN": "   "}])
def test_config_from_env_refuses_a_missing_login(environ):
    with pytest.raises(ValueError, match="CONSOLE_OPERATOR_LOGIN"):
        status.config_from_env(environ, Path("/r"), Path("/s"))


def test_null_lock_reads_as_free(tmp_path):
    state_path = tmp_path / "state.json"
    write_lock(state_path, None)
    assert status.read_execution_lock(state_path, NOW) == {"state": "free"}


def test_expired_lock_reads_as_free(tmp_path):
    state_path = tmp_path / "state.json"
    write_lock(state_path, {"holder": "ramp", "acquired": "2026-10-05T13:55:00-04:00", "expires": "2026-10-05T13:59:00-04:00"})
    assert status.read_execution_lock(state_path, NOW)["state"] == "free"


def test_unexpired_lock_reads_as_held_with_holder(tmp_path):
    state_path = tmp_path / "state.json"
    write_lock(state_path, {"holder": "ramp", "acquired": "2026-10-05T14:11:00-04:00", "expires": "2026-10-05T14:15:00-04:00"})
    lock = status.read_execution_lock(state_path, NOW)
    assert lock == {
        "state": "held",
        "holder": "ramp",
        "acquired": "2026-10-05T14:11:00-04:00",
        "expires": "2026-10-05T14:15:00-04:00",
    }


@pytest.mark.parametrize(
    "contents",
    [
        "{not json",
        json.dumps({"version": 1, "strategies": {}}),
        json.dumps([1, 2]),
        json.dumps({"execution_lock": {"holder": "ramp"}}),
        json.dumps({"execution_lock": {"holder": "ramp", "expires": "2026-10-05T14:15:00"}}),
        json.dumps({"execution_lock": {"holder": "ramp", "expires": 12345}}),
        json.dumps({"execution_lock": "ramp"}),
    ],
    ids=["bad-json", "missing-field", "not-a-mapping", "no-expires", "naive-expires", "numeric-expires", "string-lock"],
)
def test_malformed_lock_reads_as_error_never_free(tmp_path, contents):
    state_path = tmp_path / "state.json"
    state_path.write_text(contents)
    lock = status.read_execution_lock(state_path, NOW)
    assert lock["state"] == "error"
    assert lock["error"]


def test_missing_state_file_reads_as_error(tmp_path):
    assert status.read_execution_lock(tmp_path / "absent.json", NOW)["state"] == "error"


def test_read_toggle_rejects_a_file_without_strategies(tmp_path):
    toggle_path = tmp_path / "t.yaml"
    toggle_path.write_text("modified_by: auto\n")
    with pytest.raises(ValueError, match="strategies"):
        status.read_toggle(toggle_path)


def test_status_lists_every_toggle_strategy_including_mp_with_no_unit(agent_config, fake_systemd):
    doc = status.build_status(agent_config, NOW)

    assert set(doc["strategies"]) == {"cscm", "mp", "omr", "ramp"}
    assert doc["strategies"]["mp"]["enabled"] is True
    assert doc["strategies"]["mp"]["units"] == []
    assert doc["strategies"]["ramp"]["units"] == ["homeguard-multi.service"]
    assert doc["strategies"]["ramp"]["variant"] == "v11"
    assert doc["strategies"]["cscm"]["units"] == ["homeguard-cscm.service"]
    assert doc["generated_at"] == NOW.isoformat()
    assert doc["execution_lock"] == {"state": "free"}


def test_status_reports_units_once_each(agent_config, fake_systemd):
    doc = status.build_status(agent_config, NOW)

    show_call = next(args for args in fake_systemd if args[1] == "show")
    requested = show_call[2:-2]
    assert len(requested) == len(set(requested))
    assert requested[:3] == ["homeguard-multi.service", "homeguard-cscm.service", "homeguard-gateway.service"]
    assert set(units.INFRA_UNITS) <= set(requested)
    assert [unit["unit"] for unit in doc["units"]] == [
        "homeguard-multi.service", "homeguard-cscm.service", "homeguard-gateway.service"
    ]


def test_status_carries_latest_decision_gates(agent_config, fake_systemd):
    decision = status.build_status(agent_config, NOW)["strategies"]["ramp"]["last_decision"]

    assert decision["all_passed"] is False
    assert decision["trigger_kind"] == "scheduled_rebalance"
    assert decision["schedule_time"] == "15:55"
    assert decision["gates"]["health_check"] == {"passed": False, "error": "Insufficient buying power"}
    assert decision["gates"]["strategy_enabled"] == {"passed": True, "error": None}
    assert set(decision["gates"]) == {
        "strategy_enabled", "shutdown_requested", "execution_lock_acquired", "health_check", "data_freshness"
    }


def test_extra_gates_are_included(agent_config, fake_systemd):
    record = make_decision_record("cscm", extra={"btc_regime": {"passed": False, "details": {}, "error": "bear"}})
    (agent_config.latest_dir / "cscm.json").write_text(json.dumps(record))

    gates = status.build_status(agent_config, NOW)["strategies"]["cscm"]["last_decision"]["gates"]

    assert gates["btc_regime"] == {"passed": False, "error": "bear"}


def test_snapshot_is_included_without_histograms(agent_config, fake_systemd):
    snapshot = status.build_status(agent_config, NOW)["strategies"]["ramp"]["snapshot"]

    assert snapshot["gauges"]["hg_broker_last_heartbeat_timestamp"] == {"{}": 1791230098.0}
    assert "histograms" not in snapshot


def test_decoy_and_tmp_files_in_latest_are_not_strategies(agent_config, fake_systemd):
    doc = status.build_status(agent_config, NOW)

    assert "ramp_position_state" not in doc["strategies"]
    assert doc["strategies"]["omr"]["last_decision"] is None
    assert not [e for e in doc["errors"] if e["source"] == "decision:omr"]


def test_missing_snapshots_are_reported_in_errors(agent_config, fake_systemd):
    doc = status.build_status(agent_config, NOW)

    sources = {error["source"] for error in doc["errors"]}
    assert {"snapshot:mp", "snapshot:omr"} <= sources
    assert doc["strategies"]["mp"]["snapshot"] is None


def test_corrupt_latest_decision_is_an_error_not_a_failure(agent_config, fake_systemd):
    (agent_config.latest_dir / "ramp.json").write_text('{"schema_version": 2, "decision_id": "ramp-tru')

    doc = status.build_status(agent_config, NOW)

    assert doc["strategies"]["ramp"]["last_decision"] is None
    assert any(error["source"] == "decision:ramp" for error in doc["errors"])
    assert doc["strategies"]["cscm"]["last_decision"] is not None


def test_unparseable_toggle_still_returns_units_and_lock(agent_config, fake_systemd):
    agent_config.toggle_path.write_text("strategies: [unclosed\n")

    doc = status.build_status(agent_config, NOW)

    assert doc["strategies"] == {}
    assert any(error["source"] == "toggle" for error in doc["errors"])
    assert len(doc["units"]) == 3
    assert doc["execution_lock"] == {"state": "free"}


def test_toggle_entry_that_is_not_a_mapping_is_an_error(agent_config, fake_systemd):
    agent_config.toggle_path.write_text("strategies:\n  mp: true\n  ramp:\n    enabled: true\n")

    doc = status.build_status(agent_config, NOW)

    assert doc["strategies"]["mp"]["enabled"] is None
    assert any(error["source"] == "toggle:mp" for error in doc["errors"])
    assert doc["strategies"]["ramp"]["enabled"] is True


def test_systemctl_timeout_is_an_error_and_strategies_still_return(agent_config, monkeypatch):
    def timeout_run(args):
        raise subprocess.TimeoutExpired(args, 5)

    monkeypatch.setattr(units, "run_command", timeout_run)

    doc = status.build_status(agent_config, NOW)

    assert doc["units"] == []
    assert any(error["source"] == "systemd" for error in doc["errors"])
    assert doc["strategies"]["ramp"]["units"] == []
    assert doc["strategies"]["ramp"]["last_decision"] is not None


def test_missing_systemctl_binary_is_an_error(agent_config, monkeypatch):
    def missing_run(args):
        raise FileNotFoundError("systemctl")

    monkeypatch.setattr(units, "run_command", missing_run)

    assert any(error["source"] == "systemd" for error in status.build_status(agent_config, NOW)["errors"])
