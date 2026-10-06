"""Reads the instance's on-disk state into the /status document.

Read-only: it never constructs StrategyStateManager (its __init__ writes a state
backup and can create the toggle file) and never imports src.trading (about 75MB
of broker code, over the agent's MemoryMax). Decision records are read as JSON.
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Mapping

import yaml

from src.console_agent import units

# Errors a malformed file or an unexpected JSON shape can raise while reading it.
READ_ERRORS = (OSError, ValueError, KeyError, TypeError, AttributeError)
TOGGLE_ERRORS = READ_ERRORS + (yaml.YAMLError,)


@dataclass(frozen=True)
class AgentConfig:
    repo_root: Path
    snapshot_dir: Path
    operator_login: str

    @property
    def toggle_path(self) -> Path:
        return self.repo_root / "config" / "trading" / "strategy_toggle.yaml"

    @property
    def state_path(self) -> Path:
        return self.repo_root / "data" / "trading" / "strategy_positions.json"

    @property
    def latest_dir(self) -> Path:
        return self.repo_root / "data" / "trading" / "decisions" / "_latest"


def config_from_env(environ: Mapping[str, str], repo_root: Path, snapshot_dir: Path) -> AgentConfig:
    login = environ.get("CONSOLE_OPERATOR_LOGIN", "").strip()
    if not login:
        raise ValueError("CONSOLE_OPERATOR_LOGIN is not set")
    return AgentConfig(repo_root=repo_root, snapshot_dir=snapshot_dir, operator_login=login)


def read_toggle(toggle_path: Path) -> dict:
    toggle = yaml.safe_load(toggle_path.read_text())
    strategies = toggle.get("strategies") if isinstance(toggle, dict) else None
    if not isinstance(strategies, dict):
        raise ValueError(f"{toggle_path.name} has no 'strategies' mapping")
    return strategies


def read_execution_lock(state_path: Path, now: datetime) -> dict:
    try:
        lock = json.loads(state_path.read_text())["execution_lock"]
    except (OSError, ValueError, KeyError, TypeError) as e:
        return {"state": "error", "error": f"cannot read execution lock: {e!r}"}
    if lock is None:
        return {"state": "free"}
    try:
        expires = datetime.fromisoformat(lock["expires"])
        expired = now > expires
    except (KeyError, TypeError, ValueError) as e:
        return {"state": "error", "error": f"malformed execution lock: {e!r}"}
    return {
        "state": "free" if expired else "held",
        "holder": lock.get("holder"),
        "acquired": lock.get("acquired"),
        "expires": lock["expires"],
    }


def read_snapshot(snapshot_dir: Path, strategy: str) -> dict:
    snapshot = json.loads((snapshot_dir / f"{strategy}_snapshot.json").read_text())
    snapshot.pop("histograms", None)
    return snapshot


def read_latest_decision(latest_dir: Path, strategy: str) -> dict | None:
    path = latest_dir / f"{strategy}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def summarize_decision(record: dict) -> dict:
    preconditions = record["preconditions"]
    gates = {name: gate for name, gate in preconditions.items() if isinstance(gate, dict) and "passed" in gate}
    gates.update(preconditions.get("extra", {}))
    gate_results = {name: {"passed": gate["passed"], "error": gate.get("error")} for name, gate in gates.items()}
    return {
        "decision_id": record["decision_id"],
        "timestamp": record["timestamp"],
        "trigger_kind": record["trigger"]["kind"],
        "schedule_time": record["trigger"].get("schedule_time"),
        "all_passed": preconditions["all_passed"],
        "gates": gate_results,
        "error": record.get("error"),
    }


def build_status(config: AgentConfig, now: datetime) -> dict:
    errors: list[dict] = []
    unit_states = _read_units(errors)
    try:
        toggle = read_toggle(config.toggle_path)
    except TOGGLE_ERRORS as e:
        errors.append({"source": "toggle", "error": repr(e)})
        toggle = {}
    strategies = {}
    for name, settings in toggle.items():
        # YAML turns keys like 2026-01-01 into dates, which JSON cannot use as keys.
        key = str(name)
        strategies[key] = _strategy_entry(config, key, settings, unit_states, errors)
    return {
        "generated_at": now.isoformat(),
        "units": [asdict(state) for state in unit_states],
        "strategies": strategies,
        "execution_lock": read_execution_lock(config.state_path, now),
        "errors": errors,
    }


def _read_units(errors: list[dict]) -> list[units.UnitState]:
    try:
        names = list(dict.fromkeys(units.list_enabled_homeguard_units() + list(units.INFRA_UNITS)))
        return units.show_units(names)
    except (subprocess.SubprocessError, OSError, ValueError) as e:
        errors.append({"source": "systemd", "error": repr(e)})
        return []


def _strategy_entry(
    config: AgentConfig,
    name: str,
    settings: object,
    unit_states: list[units.UnitState],
    errors: list[dict],
) -> dict:
    if not isinstance(settings, dict):
        errors.append({"source": f"toggle:{name}", "error": f"expected a mapping, got {settings!r}"})
        settings = {}
    entry = {
        "enabled": settings.get("enabled"),
        "shutdown_requested": settings.get("shutdown_requested"),
        "variant": settings.get("variant"),
        "units": [state.unit for state in unit_states if state.strategy == name],
        "snapshot": None,
        "last_decision": None,
    }
    try:
        entry["snapshot"] = read_snapshot(config.snapshot_dir, name)
    except FileNotFoundError as e:
        # Only a strategy that some unit runs is expected to write a snapshot.
        if entry["units"]:
            errors.append({"source": f"snapshot:{name}", "error": repr(e)})
    except READ_ERRORS as e:
        errors.append({"source": f"snapshot:{name}", "error": repr(e)})
    try:
        record = read_latest_decision(config.latest_dir, name)
        entry["last_decision"] = summarize_decision(record) if record is not None else None
    except READ_ERRORS as e:
        errors.append({"source": f"decision:{name}", "error": repr(e)})
    return entry
