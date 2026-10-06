"""systemd queries for the console agent.

Every subprocess the agent runs goes through run_command, so tests replace that one
function instead of stubbing systemctl on PATH (the tests also run on Windows).
"""
from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass

COMMAND_TIMEOUT_SECONDS = 5
INFRA_UNITS = (
    "homeguard-gateway.service",
    "victoria-metrics.service",
    "loki.service",
    "promtail.service",
    "grafana-server.service",
    "node-exporter.service",
)
SHOW_PROPERTIES = "Id,ActiveState,SubState,NRestarts,ActiveEnterTimestamp,MemoryCurrent,ExecStart"
_STRATEGY_ARG = re.compile(r"run_live_paper_trading\.py\b.*?--strategy[ =]([a-z0-9_]+)")
# systemd reports MemoryCurrent as max uint64 when memory accounting has no value.
_MEMORY_NOT_SET = "18446744073709551615"


@dataclass
class UnitState:
    unit: str
    active_state: str
    sub_state: str
    restarts: int | None
    active_since: str | None
    memory_bytes: int | None
    strategy: str | None


def run_command(args: list[str]) -> str:
    completed = subprocess.run(args, capture_output=True, text=True, timeout=COMMAND_TIMEOUT_SECONDS, check=True)
    return completed.stdout


def strategy_for_exec_start(exec_start: str) -> str | None:
    match = _STRATEGY_ARG.search(exec_start)
    if match:
        return match.group(1)
    if "run_cscm_live.py" in exec_start:
        return "cscm"
    return None


def list_enabled_homeguard_units() -> list[str]:
    output = run_command(
        ["systemctl", "list-unit-files", "homeguard-*.service", "--state=enabled", "--no-legend", "--plain"]
    )
    return [line.split()[0] for line in output.splitlines() if line.strip()]


def show_units(unit_names: list[str]) -> list[UnitState]:
    if not unit_names:
        return []
    output = run_command(["systemctl", "show", *unit_names, "-p", SHOW_PROPERTIES])
    return parse_show_output(output)


def parse_show_output(output: str) -> list[UnitState]:
    states = []
    for block in output.strip().split("\n\n"):
        props = dict(line.split("=", 1) for line in block.splitlines() if "=" in line)
        if "Id" in props:
            states.append(_unit_state(props))
    return states


def _unit_state(props: dict[str, str]) -> UnitState:
    restarts = props.get("NRestarts", "")
    memory = props.get("MemoryCurrent", "")
    return UnitState(
        unit=props["Id"],
        active_state=props.get("ActiveState", "unknown"),
        sub_state=props.get("SubState", "unknown"),
        restarts=int(restarts) if restarts.isdigit() else None,
        active_since=props.get("ActiveEnterTimestamp") or None,
        memory_bytes=int(memory) if memory.isdigit() and memory != _MEMORY_NOT_SET else None,
        strategy=strategy_for_exec_start(props.get("ExecStart", "")),
    )
