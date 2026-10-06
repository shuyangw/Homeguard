"""Unit tests for systemd parsing and the unit-to-strategy mapping."""

from src.console_agent import units

MULTI_EXEC_START = (
    "{ path=/home/ec2-user/Homeguard/venv/bin/python ; "
    "argv[]=/home/ec2-user/Homeguard/venv/bin/python scripts/trading/run_live_paper_trading.py "
    "--strategy ramp --initial-capital 100000 ; ignore_errors=no ; "
    "start_time=[Mon 2026-10-05 08:01:12 EDT] ; stop_time=[n/a] ; pid=1234 ; code=(null) ; status=0/0 }"
)
CSCM_EXEC_START = (
    "{ path=/home/ec2-user/Homeguard/venv/bin/python ; "
    "argv[]=/home/ec2-user/Homeguard/venv/bin/python scripts/trading/run_cscm_live.py "
    "--check-interval 0.0833 --initial-capital 100000 ; ignore_errors=no ; "
    "start_time=[Mon 2026-10-05 08:01:15 EDT] ; stop_time=[n/a] ; pid=1240 ; code=(null) ; status=0/0 }"
)
GATEWAY_EXEC_START = (
    "{ path=/usr/bin/xvfb-run ; argv[]=/usr/bin/xvfb-run /opt/ibc/gatewaystart.sh ; "
    "ignore_errors=no ; start_time=[n/a] ; stop_time=[n/a] ; pid=0 ; code=(null) ; status=0/0 }"
)

SHOW_OUTPUT = f"""Id=homeguard-multi.service
ActiveState=active
SubState=running
NRestarts=0
ActiveEnterTimestamp=Mon 2026-10-05 08:01:12 EDT
MemoryCurrent=536870912
ExecStart={MULTI_EXEC_START}

Id=homeguard-cscm.service
ActiveState=active
SubState=running
NRestarts=2
ActiveEnterTimestamp=Mon 2026-10-05 08:01:15 EDT
MemoryCurrent=287309824
ExecStart={CSCM_EXEC_START}

Id=loki.service
ActiveState=inactive
SubState=dead
NRestarts=0
ActiveEnterTimestamp=
MemoryCurrent=[not set]
ExecStart={GATEWAY_EXEC_START}
"""


def test_multi_unit_running_ramp_maps_to_ramp():
    assert units.strategy_for_exec_start(MULTI_EXEC_START) == "ramp"


def test_strategy_argument_with_equals_sign_maps():
    exec_start = "argv[]=python scripts/trading/run_live_paper_trading.py --strategy=omr ;"
    assert units.strategy_for_exec_start(exec_start) == "omr"


def test_cscm_runner_maps_to_cscm():
    assert units.strategy_for_exec_start(CSCM_EXEC_START) == "cscm"


def test_unit_that_runs_no_strategy_maps_to_none():
    assert units.strategy_for_exec_start(GATEWAY_EXEC_START) is None
    assert units.strategy_for_exec_start("") is None


def test_parse_show_output_reads_one_state_per_unit_block():
    states = units.parse_show_output(SHOW_OUTPUT)

    assert [state.unit for state in states] == ["homeguard-multi.service", "homeguard-cscm.service", "loki.service"]
    multi = states[0]
    assert multi.active_state == "active"
    assert multi.sub_state == "running"
    assert multi.restarts == 0
    assert multi.active_since == "Mon 2026-10-05 08:01:12 EDT"
    assert multi.memory_bytes == 536870912
    assert multi.strategy == "ramp"
    assert states[1].restarts == 2
    assert states[1].strategy == "cscm"


def test_parse_show_output_treats_unset_values_as_none():
    loki = units.parse_show_output(SHOW_OUTPUT)[2]

    assert loki.memory_bytes is None
    assert loki.active_since is None
    assert loki.strategy is None


def test_max_uint64_memory_reads_as_none():
    output = "Id=x.service\nActiveState=active\nSubState=running\nMemoryCurrent=18446744073709551615\n"
    assert units.parse_show_output(output)[0].memory_bytes is None


def test_list_enabled_homeguard_units_takes_the_first_column(monkeypatch):
    calls = []

    def fake_run(args):
        calls.append(args)
        return "homeguard-multi.service enabled disabled\nhomeguard-cscm.service enabled enabled\n\n"

    monkeypatch.setattr(units, "run_command", fake_run)

    assert units.list_enabled_homeguard_units() == ["homeguard-multi.service", "homeguard-cscm.service"]
    assert calls[0][:3] == ["systemctl", "list-unit-files", "homeguard-*.service"]
    assert "--state=enabled" in calls[0]


def test_show_units_with_no_names_runs_nothing(monkeypatch):
    def fail_run(args):
        raise AssertionError("no command expected")

    monkeypatch.setattr(units, "run_command", fail_run)

    assert units.show_units([]) == []


def test_show_units_passes_names_and_properties(monkeypatch):
    calls = []

    def fake_run(args):
        calls.append(args)
        return SHOW_OUTPUT

    monkeypatch.setattr(units, "run_command", fake_run)

    states = units.show_units(["homeguard-multi.service", "homeguard-cscm.service", "loki.service"])

    assert len(states) == 3
    assert calls[0][:2] == ["systemctl", "show"]
    assert calls[0][2:5] == ["homeguard-multi.service", "homeguard-cscm.service", "loki.service"]
    assert calls[0][-2:] == ["-p", units.SHOW_PROPERTIES]


def test_load_state_tells_a_missing_unit_from_a_stopped_one():
    output = (
        "Id=loki.service\nLoadState=not-found\nActiveState=inactive\nSubState=dead\n\n"
        "Id=promtail.service\nLoadState=loaded\nActiveState=inactive\nSubState=dead\n"
    )
    loki, promtail = units.parse_show_output(output)

    assert loki.load_state == "not-found"
    assert promtail.load_state == "loaded"
    assert "LoadState" in units.SHOW_PROPERTIES.split(",")
