"""Shared fixtures: a temporary repo layout and a faked systemd."""

import json

import pytest

from src.console_agent import units
from src.console_agent.status import AgentConfig

OPERATOR_LOGIN = "operator@example.com"

TOGGLE_YAML = """last_modified: '2026-05-23T00:00:00-04:00'
modified_by: claude-code
strategies:
  cscm:
    enabled: false
    shutdown_requested: false
    variant: v01
  mp:
    enabled: true
    shutdown_requested: false
    variant: v01
  omr:
    enabled: false
    shutdown_requested: false
    variant: v01
  ramp:
    enabled: true
    shutdown_requested: false
    variant: v11
"""

LIST_UNIT_FILES_OUTPUT = (
    "homeguard-multi.service enabled disabled\n"
    "homeguard-cscm.service enabled enabled\n"
    "homeguard-gateway.service enabled enabled\n"
)

SHOW_OUTPUT = """Id=homeguard-multi.service
ActiveState=active
SubState=running
NRestarts=0
ActiveEnterTimestamp=Mon 2026-10-05 08:01:12 EDT
MemoryCurrent=536870912
ExecStart={ path=/v/bin/python ; argv[]=/v/bin/python scripts/trading/run_live_paper_trading.py --strategy ramp --initial-capital 100000 ; ignore_errors=no }

Id=homeguard-cscm.service
ActiveState=active
SubState=running
NRestarts=0
ActiveEnterTimestamp=Mon 2026-10-05 08:01:15 EDT
MemoryCurrent=287309824
ExecStart={ path=/v/bin/python ; argv[]=/v/bin/python scripts/trading/run_cscm_live.py --check-interval 0.0833 ; ignore_errors=no }

Id=homeguard-gateway.service
ActiveState=active
SubState=running
NRestarts=0
ActiveEnterTimestamp=Mon 2026-10-05 08:00:50 EDT
MemoryCurrent=522190848
ExecStart={ path=/opt/ibc/gatewaystart.sh ; argv[]=/opt/ibc/gatewaystart.sh ; ignore_errors=no }
"""


def make_decision_record(strategy, health_passed=True, extra=None):
    def gate(passed, error=None):
        return {"passed": passed, "details": {}, "error": error}

    return {
        "schema_version": 2,
        "decision_id": f"{strategy}-20261005-1555",
        "strategy": strategy,
        "timestamp": "2026-10-05T15:55:04.120000-04:00",
        "trigger": {
            "kind": "scheduled_rebalance",
            "schedule_time": "15:55",
            "actual_fire_time": "2026-10-05T15:55:04.120000-04:00",
            "delay_seconds": 4.12,
            "consecutive_fire_count": 1,
        },
        "preconditions": {
            "all_passed": health_passed,
            "strategy_enabled": gate(True),
            "shutdown_requested": gate(True),
            "execution_lock_acquired": gate(True),
            "health_check": gate(health_passed, None if health_passed else "Insufficient buying power"),
            "data_freshness": gate(True),
            "extra": extra or {},
        },
        "inputs": {},
        "logic_decisions": None,
        "executions": [],
        "post_state": None,
        "error": None,
        "metadata": {},
        "parent_decision_id": None,
    }


@pytest.fixture
def agent_config(tmp_path):
    (tmp_path / "config" / "trading").mkdir(parents=True)
    (tmp_path / "config" / "trading" / "strategy_toggle.yaml").write_text(TOGGLE_YAML)

    latest_dir = tmp_path / "data" / "trading" / "decisions" / "_latest"
    latest_dir.mkdir(parents=True)
    (tmp_path / "data" / "trading" / "strategy_positions.json").write_text(
        json.dumps({"version": 1, "strategies": {}, "execution_lock": None})
    )
    (latest_dir / "ramp.json").write_text(json.dumps(make_decision_record("ramp", health_passed=False)))
    (latest_dir / "cscm.json").write_text(json.dumps(make_decision_record("cscm")))
    (latest_dir / "ramp_position_state.json").write_text(json.dumps({"strategy": "ramp", "positions": {}}))
    (latest_dir / "omr.json.tmp").write_text("{not json")

    snapshot_dir = tmp_path / "snapshots"
    snapshot_dir.mkdir()
    for strategy in ("ramp", "cscm"):
        snapshot = {
            "strategy": strategy,
            "timestamp": 1791230100.5,
            "gauges": {"hg_broker_last_heartbeat_timestamp": {"{}": 1791230098.0}},
            "counters": {"hg_orders_total": {"{}": 11}},
            "histograms": {"hg_decision_seconds": {"{}": {"count": 3}}},
        }
        (snapshot_dir / f"{strategy}_snapshot.json").write_text(json.dumps(snapshot))

    return AgentConfig(repo_root=tmp_path, snapshot_dir=snapshot_dir, operator_login=OPERATOR_LOGIN)


@pytest.fixture
def fake_systemd(monkeypatch):
    calls = []

    def fake_run(args):
        calls.append(args)
        if args[1] == "list-unit-files":
            return LIST_UNIT_FILES_OUTPUT
        if args[1] == "show":
            return SHOW_OUTPUT
        raise AssertionError(f"unexpected command {args}")

    monkeypatch.setattr(units, "run_command", fake_run)
    return calls
