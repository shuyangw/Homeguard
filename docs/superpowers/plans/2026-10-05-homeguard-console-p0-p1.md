# Homeguard Console Phase 0 + Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make strategy toggle changes survive deploys (Phase 0) and ship a read-only on-instance agent that serves `GET /status` and `GET /decisions` over the tailnet (Phase 1).

**Architecture:** Phase 0 is two small commits: `_load_toggle()` fails closed when the toggle file is missing, then the toggle file is untracked. Phase 1 adds `src/console_agent/`, a stdlib `ThreadingHTTPServer` on 127.0.0.1:8090 that reads systemd state, the toggle YAML, the state JSON, metric snapshots and the latest decision JSON, and returns them as one JSON document; `tailscale serve` publishes it on 8443. Everything is built on main and later cherry-picked onto the deploy branch `ramp-phase4-turnover-regime-research`.

**Tech Stack:** Python stdlib (`http.server`, `subprocess`, `json`, `re`), PyYAML, python-dotenv, pytest, systemd, Tailscale.

**Spec:** `docs/superpowers/specs/2026-10-05-homeguard-console-design.md` (sections "Revision notes", "Phase 0 prerequisites", "Phase 1 detailed design", "Test plan").

## Global Constraints

- Run all Python with `C:\Users\qwqw1\anaconda3\envs\fintech\python.exe` (in Git Bash: `/c/Users/qwqw1/anaconda3/envs/fintech/python.exe`); written below as `$PY`.
- All work happens in the worktree `C:\Users\qwqw1\Dropbox\cs\github\Homeguard\.worktrees\console-p0-p1` on branch `feat/console-p0-p1` (see Setup). Never commit in the main working tree.
- Git: targeted `git add <paths>` only, never `git add -A` or `git add .`; never bare `git status`/`git diff` with no path in the main tree. Test files and implementation files go in separate commits (tests first, after both pass).
- ASCII-only characters in all code, comments, logs and docs. No emojis, no Unicode arrows or dashes.
- Every new Python module starts with `from __future__ import annotations` and uses only Python 3.9-compatible syntax at runtime (no `match`, no runtime `X | Y` unions, no parenthesized `with`), since the instance venv's Python version is unverified.
- The agent never imports anything under `src.trading` (that package's `__init__` pulls in about 75MB of broker code, over `MemoryMax=96M`). Allowed imports: stdlib, `yaml`, `dotenv`, `src.utils.logger`, `src.utils.timezone`, `src.settings`, and `src.console_agent.*`.
- Phase 1 is read-only: the agent never writes a file and never constructs `StrategyStateManager`.
- Agent binds `127.0.0.1:8090`; tailnet port `8443`; auth header `Tailscale-User-Login` must equal `CONSOLE_OPERATOR_LOGIN`; non-GET returns 405.
- Unit limits: `MemoryMax=96M`, `OOMScoreAdjust=500`, not `PartOf=homeguard-trading.target`.
- Logging: `from src.utils.logger import logger`; f-strings only (the Homeguard logger does not support `%s` args).
- No push to `ramp-phase4-turnover-regime-research` and no command on the EC2 instance without the operator's explicit go-ahead at that step.

## Review Focus

- A corrupt or truncated `_latest/<strategy>.json` (a crash mid-write on an older writer) must show up in `errors` for that strategy while the rest of `/status` returns 200. Pinned in Task 4.
- Leftover `.tmp` files and the deploy branch's `ramp_position_state.json` in `_latest/` must never be read as strategies. Pinned in Task 4 (fixture decoys) and Task 5 (tree-unchanged check).
- A future edit that imports `src.trading` (for example to "reuse" `reader.latest`) silently pushes the agent past its memory cap and gets it OOM-killed. Pinned in Task 5 by an import-weight test.
- An execution lock whose `expires` is naive, missing or not a string must read as `error`, never `free`, since Stop will later trust it. Pinned in Task 4.
- A toggle entry that is not a mapping (for example `mp: true`) must produce an `errors` entry, not a 500. Pinned in Task 4.

---

## Setup (controller, once)

- [ ] **Create the worktree from main**

```bash
cd /c/Users/qwqw1/Dropbox/cs/github/Homeguard
git worktree add .worktrees/console-p0-p1 -b feat/console-p0-p1 main
cd .worktrees/console-p0-p1 && ls settings.ini config/trading/strategy_toggle.yaml
```

Expected: both files listed (both are tracked today). If `settings.ini` is missing, copy it from the main tree.

- [ ] **Baseline the state manager tests in the worktree**

```bash
cd /c/Users/qwqw1/Dropbox/cs/github/Homeguard/.worktrees/console-p0-p1
$PY -m pytest tests/trading/test_multi_strategy_coordination.py tests/trading/test_omr_live_adapter.py tests/trading/test_ramp_crash_protection_parity.py tests/trading/test_ramp_live_adapter.py tests/trading/test_ramp_live_adapter_target_execution.py tests/trading/test_run_live_paper_trading_preflight.py tests/trading/test_state_manager_adopt.py tests/trading/test_state_manager_broker_aware.py tests/trading/test_state_manager_migration.py tests/trading/test_state_tracking.py -q -p no:cacheprovider
```

Expected: `167 passed` (measured on main at 11295b6).

---

### Task 1: Toggle file fails closed when missing (Phase 0, commit A)

**Files:**
- Modify: `src/trading/state/strategy_state_manager.py` (the `else:` branch of `_load_toggle`, currently lines 170-181)
- Test: `tests/trading/test_state_manager_toggle_defaults.py` (create)

**Interfaces:**
- Consumes: `StrategyStateManager(state_file: Path, toggle_file: Path)`, `get_enabled_strategies() -> List[str]`, `is_enabled(strategy: str) -> bool` (all existing).
- Produces: no new API. Behavior change only: a missing toggle file regenerates `omr`, `mp`, `ramp`, `cscm` all with `enabled: False`.

- [ ] **Step 1: Write the failing test**

Create `tests/trading/test_state_manager_toggle_defaults.py`:

```python
"""A missing toggle file must fail closed: regenerate with every strategy off."""

import yaml

from src.trading.state.strategy_state_manager import StrategyStateManager


def test_missing_toggle_file_regenerates_with_every_strategy_disabled(tmp_path):
    toggle_file = tmp_path / "strategy_toggle.yaml"
    manager = StrategyStateManager(state_file=tmp_path / "strategy_positions.json", toggle_file=toggle_file)

    assert manager.get_enabled_strategies() == []
    for strategy in ("omr", "mp", "ramp", "cscm"):
        assert manager.is_enabled(strategy) is False


def test_missing_toggle_file_is_rewritten_to_disk_disabled(tmp_path):
    toggle_file = tmp_path / "strategy_toggle.yaml"
    StrategyStateManager(state_file=tmp_path / "strategy_positions.json", toggle_file=toggle_file)

    written = yaml.safe_load(toggle_file.read_text())
    assert set(written["strategies"]) == {"omr", "mp", "ramp", "cscm"}
    assert all(settings["enabled"] is False for settings in written["strategies"].values())
    assert all(settings["shutdown_requested"] is False for settings in written["strategies"].values())


def test_existing_toggle_file_is_left_alone(tmp_path):
    toggle_file = tmp_path / "strategy_toggle.yaml"
    toggle_file.write_text("strategies:\n  ramp:\n    enabled: true\n")
    manager = StrategyStateManager(state_file=tmp_path / "strategy_positions.json", toggle_file=toggle_file)

    assert manager.get_enabled_strategies() == ["ramp"]
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `$PY -m pytest tests/trading/test_state_manager_toggle_defaults.py -v -p no:cacheprovider`
Expected: 2 failed, 1 passed. The first test fails with `assert ['cscm', 'omr', 'ramp'] == []`, the second on the all-disabled assertion; the third passes. (Checked against main at 11295b6.)

- [ ] **Step 3: Change the defaults**

In `src/trading/state/strategy_state_manager.py`, replace this block inside `_load_toggle`:

```python
            else:
                # Create default toggle config
                self._toggle = {
                    'strategies': {
                        'omr': {'enabled': True, 'shutdown_requested': False},
                        'mp': {'enabled': False, 'shutdown_requested': False},
                        'ramp': {'enabled': True, 'shutdown_requested': False},
                        'cscm': {'enabled': True, 'shutdown_requested': False},
                    },
```

with:

```python
            else:
                # Fail closed: a missing file (e.g. deleted by a deploy) must never turn trading on.
                logger.error(
                    f"Toggle file missing at {self.toggle_file}; regenerating with every strategy "
                    f"disabled. Trading stays off until the file is restored."
                )
                self._toggle = {
                    'strategies': {
                        'omr': {'enabled': False, 'shutdown_requested': False},
                        'mp': {'enabled': False, 'shutdown_requested': False},
                        'ramp': {'enabled': False, 'shutdown_requested': False},
                        'cscm': {'enabled': False, 'shutdown_requested': False},
                    },
```

Leave the `'last_modified'`, `'modified_by': 'auto'` lines and the `self._save_toggle()` call unchanged.

- [ ] **Step 4: Run the new test and the baseline suite**

Run: `$PY -m pytest tests/trading/test_state_manager_toggle_defaults.py -v -p no:cacheprovider`
Expected: 3 passed.

Then rerun the baseline command from Setup. Expected: `167 passed`. If any test fails because it relied on the old enabled defaults, stop and report it; do not edit that test without the controller's agreement.

- [ ] **Step 5: Commit (tests, then implementation)**

```bash
git add tests/trading/test_state_manager_toggle_defaults.py
git commit -m "test(state): missing toggle file must regenerate with every strategy off

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git add src/trading/state/strategy_state_manager.py
git commit -m "fix(state): fail closed when strategy_toggle.yaml is missing

A deploy that deletes the toggle file used to regenerate defaults with
OMR, RAMP and CSCM enabled. Regenerate with every strategy disabled and
log an error instead.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Stop tracking the toggle file (Phase 0, commit B)

**Files:**
- Modify: `.gitignore` (comment lines 24-27 above `config/trading/strategy_toggle.yaml`)
- Untrack: `config/trading/strategy_toggle.yaml` (file stays on disk)

**Interfaces:**
- Consumes: Task 1 must already be committed on this branch (order matters on both branches).
- Produces: `config/trading/strategy_toggle.yaml` is no longer in the index.

- [ ] **Step 1: Confirm Task 1 precedes this commit**

Run: `git log --oneline -3`
Expected: the top commit is `fix(state): fail closed when strategy_toggle.yaml is missing`.

- [ ] **Step 2: Update the .gitignore comment**

Replace lines 24-27:

```
# Strategy enable/shutdown toggle: runtime state modified by the bot at every
# rebalance / via /toggle Discord command. Don't track. Auto-generated on
# first run by StrategyStateManager._load_toggle if missing. See template
# at config/trading/strategy_toggle.example.yaml.
```

with:

```
# Strategy enable/shutdown toggle: runtime state, changed on the instance.
# Don't track. If missing, StrategyStateManager._load_toggle regenerates it
# with every strategy DISABLED (fail closed). Template:
# config/trading/strategy_toggle.example.yaml.
```

Leave line 28 (`config/trading/strategy_toggle.yaml`) as is.

- [ ] **Step 3: Untrack the file and verify it stays on disk**

```bash
git rm --cached config/trading/strategy_toggle.yaml
git ls-files config/trading/strategy_toggle.yaml
ls config/trading/strategy_toggle.yaml config/trading/strategy_toggle.example.yaml
git check-ignore -v config/trading/strategy_toggle.yaml
```

Expected: `git ls-files` prints nothing; both files are listed by `ls`; `check-ignore` names `.gitignore:28`.

- [ ] **Step 4: Commit**

```bash
git add .gitignore
git commit -m "fix(config): untrack strategy_toggle.yaml so instance toggles survive deploys

The file was gitignored but still tracked, so the deploy script's
stash + pull replaced instance-side toggle changes with committed
values. Requires the fail-closed defaults from the previous commit,
since pulling this commit deletes the file on the instance.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git show --stat --format= HEAD
```

Expected: the commit shows `.gitignore` modified and `config/trading/strategy_toggle.yaml` deleted (from the index only).

---

### Task 3: systemd queries and unit-to-strategy mapping (`units.py`)

**Files:**
- Create: `src/console_agent/__init__.py`
- Create: `src/console_agent/units.py`
- Create: `tests/console_agent/__init__.py` (empty)
- Test: `tests/console_agent/test_units.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces (used by Task 4 and Task 5):
  - `INFRA_UNITS: tuple[str, ...]`
  - `run_command(args: list[str]) -> str` (the single subprocess seam; tests monkeypatch `units.run_command`)
  - `strategy_for_exec_start(exec_start: str) -> str | None`
  - `list_enabled_homeguard_units() -> list[str]`
  - `show_units(unit_names: list[str]) -> list[UnitState]`
  - `parse_show_output(output: str) -> list[UnitState]`
  - `@dataclass UnitState(unit: str, active_state: str, sub_state: str, restarts: int | None, active_since: str | None, memory_bytes: int | None, strategy: str | None)`

- [ ] **Step 1: Write the failing tests**

Create `tests/console_agent/__init__.py` as an empty file.

Create `tests/console_agent/test_units.py`:

```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `$PY -m pytest tests/console_agent/test_units.py -v -p no:cacheprovider`
Expected: collection ERROR, `ModuleNotFoundError: No module named 'src.console_agent'`.

- [ ] **Step 3: Implement the package and `units.py`**

Create `src/console_agent/__init__.py`:

```python
"""Read-only agent that serves Homeguard instance status to the local console over the tailnet."""
```

Create `src/console_agent/units.py`:

```python
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `$PY -m pytest tests/console_agent/test_units.py -v -p no:cacheprovider`
Expected: 10 passed.

- [ ] **Step 5: Commit (tests, then implementation)**

```bash
git add tests/console_agent/__init__.py tests/console_agent/test_units.py
git commit -m "test(console-agent): systemd parsing and unit-to-strategy mapping

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git add src/console_agent/__init__.py src/console_agent/units.py
git commit -m "feat(console-agent): systemd queries and unit-to-strategy mapping

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: On-disk state readers and the `/status` document (`status.py`)

**Files:**
- Create: `src/console_agent/status.py`
- Create: `tests/console_agent/conftest.py` (fixtures shared with Task 5)
- Test: `tests/console_agent/test_status.py`

**Interfaces:**
- Consumes (Task 3): `units.run_command`, `units.list_enabled_homeguard_units()`, `units.show_units(names)`, `units.INFRA_UNITS`, `units.UnitState`.
- Produces (used by Task 5):
  - `@dataclass(frozen=True) AgentConfig(repo_root: Path, snapshot_dir: Path, operator_login: str)` with properties `toggle_path`, `state_path`, `latest_dir` (all `Path`)
  - `config_from_env(environ: Mapping[str, str], repo_root: Path, snapshot_dir: Path) -> AgentConfig` (raises `ValueError` when `CONSOLE_OPERATOR_LOGIN` is unset or blank)
  - `read_toggle(toggle_path: Path) -> dict` (the `strategies` mapping; raises on a bad file)
  - `read_execution_lock(state_path: Path, now: datetime) -> dict`
  - `read_latest_decision(latest_dir: Path, strategy: str) -> dict | None`
  - `build_status(config: AgentConfig, now: datetime) -> dict` with keys `generated_at`, `units`, `strategies`, `execution_lock`, `errors`
- Fixtures produced in `conftest.py` (used by Task 5): `agent_config`, `fake_systemd`, constant `OPERATOR_LOGIN`.

- [ ] **Step 1: Write the shared fixtures**

Create `tests/console_agent/conftest.py`:

```python
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
```

- [ ] **Step 2: Write the failing tests**

Create `tests/console_agent/test_status.py`:

```python
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
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `$PY -m pytest tests/console_agent/test_status.py -v -p no:cacheprovider`
Expected: collection ERROR, `ModuleNotFoundError` / `ImportError` for `src.console_agent.status`.

- [ ] **Step 4: Implement `status.py`**

Create `src/console_agent/status.py`:

```python
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
_READ_ERRORS = (OSError, ValueError, KeyError, TypeError, AttributeError)
_TOGGLE_ERRORS = _READ_ERRORS + (yaml.YAMLError,)


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
    except _TOGGLE_ERRORS as e:
        errors.append({"source": "toggle", "error": repr(e)})
        toggle = {}
    strategies = {
        name: _strategy_entry(config, name, settings, unit_states, errors) for name, settings in toggle.items()
    }
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
    except _READ_ERRORS as e:
        errors.append({"source": f"snapshot:{name}", "error": repr(e)})
    try:
        record = read_latest_decision(config.latest_dir, name)
        entry["last_decision"] = summarize_decision(record) if record is not None else None
    except _READ_ERRORS as e:
        errors.append({"source": f"decision:{name}", "error": repr(e)})
    return entry
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `$PY -m pytest tests/console_agent -v -p no:cacheprovider`
Expected: all tests in `test_units.py` and `test_status.py` pass (10 + 28).

- [ ] **Step 6: Commit (tests, then implementation)**

```bash
git add tests/console_agent/conftest.py tests/console_agent/test_status.py
git commit -m "test(console-agent): on-disk readers and the /status document

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git add src/console_agent/status.py
git commit -m "feat(console-agent): read toggle, lock, snapshots and decisions into /status

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: HTTP server, auth, routes and entry point (`server.py`, `__main__.py`)

**Files:**
- Create: `src/console_agent/server.py`
- Create: `src/console_agent/__main__.py`
- Test: `tests/console_agent/test_server.py`

**Interfaces:**
- Consumes (Task 4): `status.AgentConfig`, `status.config_from_env`, `status.build_status(config, now)`, `status.read_toggle(path)`, `status.read_latest_decision(latest_dir, strategy)`; fixtures `agent_config`, `fake_systemd`, `OPERATOR_LOGIN` from `tests/console_agent/conftest.py`.
- Produces:
  - `LOGIN_HEADER = "Tailscale-User-Login"`
  - `make_server(config: AgentConfig, host: str = "127.0.0.1", port: int = 8090) -> AgentServer` (an `http.server.ThreadingHTTPServer` subclass carrying `.config`)
  - `python -m src.console_agent` entry point

- [ ] **Step 1: Write the failing tests**

Create `tests/console_agent/test_server.py`:

```python
"""Integration tests: a real ThreadingHTTPServer on an ephemeral port."""

import json
import subprocess
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from src.console_agent.server import LOGIN_HEADER, make_server
from tests.console_agent.conftest import OPERATOR_LOGIN

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def agent_url(agent_config, fake_systemd):
    server = make_server(agent_config, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


def call(url, path, login=OPERATOR_LOGIN, method="GET"):
    headers = {LOGIN_HEADER: login} if login is not None else {}
    data = b"{}" if method != "GET" else None
    request = urllib.request.Request(url + path, headers=headers, method=method, data=data)
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_status_with_the_operator_login_returns_every_strategy(agent_url):
    code, body = call(agent_url, "/status")

    assert code == 200
    assert set(body["strategies"]) == {"cscm", "mp", "omr", "ramp"}
    assert body["strategies"]["mp"]["units"] == []


@pytest.mark.parametrize("login", [None, "", "intruder@example.com", OPERATOR_LOGIN.upper()])
def test_requests_without_the_operator_login_are_forbidden(agent_url, login):
    code, body = call(agent_url, "/status", login=login)

    assert code == 403
    assert "strategies" not in body


@pytest.mark.parametrize("method", ["POST", "PUT", "PATCH", "DELETE"])
def test_write_methods_are_not_allowed(agent_url, method):
    code, _ = call(agent_url, "/strategies/ramp/enabled", method=method)
    assert code == 405


def test_unknown_path_is_not_found(agent_url):
    assert call(agent_url, "/metrics")[0] == 404


def test_decisions_returns_the_full_latest_record(agent_url):
    code, body = call(agent_url, "/decisions?strategy=ramp")

    assert code == 200
    assert body["decision_id"] == "ramp-20261005-1555"
    assert body["preconditions"]["health_check"]["error"] == "Insufficient buying power"


@pytest.mark.parametrize(
    "query",
    ["", "?strategy=", "?strategy=../x", "?strategy=RAMP", "?strategy=ramp%2F..", "?strategy=%C3%A9", "?strategy=" + "x" * 17],
    ids=["no-param", "empty", "path", "upper", "encoded-slash", "non-ascii", "too-long"],
)
def test_decisions_rejects_bad_strategy_names(agent_url, query):
    assert call(agent_url, "/decisions" + query)[0] == 400


def test_decisions_for_a_name_not_in_the_toggle_is_not_found(agent_url):
    assert call(agent_url, "/decisions?strategy=zzz")[0] == 404


def test_decisions_for_a_strategy_with_no_record_is_not_found(agent_url):
    code, body = call(agent_url, "/decisions?strategy=mp")
    assert code == 404
    assert "no decision" in body["error"]


def test_agent_writes_nothing_to_the_tree(agent_config, agent_url):
    def tree_contents():
        return {
            path: path.read_bytes() for path in sorted(agent_config.repo_root.rglob("*")) if path.is_file()
        }

    before = tree_contents()
    call(agent_url, "/status")
    call(agent_url, "/decisions?strategy=ramp")
    call(agent_url, "/decisions?strategy=cscm")

    assert tree_contents() == before


def test_agent_does_not_import_the_trading_stack():
    probe = (
        "import sys\n"
        "import src.console_agent.server, src.console_agent.status, src.console_agent.units\n"
        "import src.console_agent.__main__\n"
        "heavy = sorted(m for m in sys.modules if m.startswith('src.trading')"
        " or m.split('.')[0] in ('pandas', 'numpy', 'alpaca', 'ib_async'))\n"
        "print(heavy)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], cwd=REPO_ROOT, capture_output=True, text=True, timeout=60, check=True
    )
    assert result.stdout.strip().splitlines()[-1] == "[]"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `$PY -m pytest tests/console_agent/test_server.py -v -p no:cacheprovider`
Expected: collection ERROR, `ModuleNotFoundError` for `src.console_agent.server`.

- [ ] **Step 3: Implement `server.py`**

Create `src/console_agent/server.py`:

```python
"""HTTP front end of the console agent: auth, routing and JSON responses."""
from __future__ import annotations

import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

from src.console_agent import status
from src.utils.logger import logger
from src.utils.timezone import tz

LOGIN_HEADER = "Tailscale-User-Login"
_STRATEGY_NAME = re.compile(r"[a-z][a-z0-9_]{0,15}")


class AgentHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        # tailscale serve sets this header for tailnet users; requests without it never came through serve.
        if self.headers.get(LOGIN_HEADER) != self.server.config.operator_login:
            self._send_json(403, {"error": "forbidden"})
            return
        url = urlsplit(self.path)
        try:
            if url.path == "/status":
                self._send_json(200, status.build_status(self.server.config, tz.now()))
            elif url.path == "/decisions":
                self._send_decision(parse_qs(url.query).get("strategy", [""])[0])
            else:
                self._send_json(404, {"error": "not found"})
        except Exception as e:
            logger.error(f"[console-agent] {url.path} failed: {e!r}")
            self._send_json(500, {"error": "internal error"})

    def _method_not_allowed(self) -> None:
        self._send_json(405, {"error": "read-only agent"})

    do_POST = do_PUT = do_PATCH = do_DELETE = _method_not_allowed

    def _send_decision(self, strategy: str) -> None:
        if not _STRATEGY_NAME.fullmatch(strategy):
            self._send_json(400, {"error": "invalid strategy name"})
            return
        if strategy not in status.read_toggle(self.server.config.toggle_path):
            self._send_json(404, {"error": f"unknown strategy {strategy}"})
            return
        record = status.read_latest_decision(self.server.config.latest_dir, strategy)
        if record is None:
            self._send_json(404, {"error": f"no decision recorded for {strategy}"})
            return
        self._send_json(200, record)

    def _send_json(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_request(self, code="-", size="-") -> None:
        pass  # the console polls every 10 s, so per-request lines would flood the journal

    def log_error(self, format: str, *args) -> None:
        logger.error(f"[console-agent] {format % args}")


class AgentServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], config: status.AgentConfig):
        super().__init__(address, AgentHandler)
        self.config = config


def make_server(config: status.AgentConfig, host: str = "127.0.0.1", port: int = 8090) -> AgentServer:
    return AgentServer((host, port), config)
```

- [ ] **Step 4: Implement `__main__.py`**

Create `src/console_agent/__main__.py`:

```python
"""Entry point: python -m src.console_agent (run from the repo root)."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

from src.console_agent.server import make_server
from src.console_agent.status import config_from_env
from src.settings import get_local_storage_dir
from src.utils.logger import logger

REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    load_dotenv(REPO_ROOT / ".env")
    snapshot_dir = Path(get_local_storage_dir()) / "metrics_snapshots"
    try:
        config = config_from_env(os.environ, REPO_ROOT, snapshot_dir)
    except ValueError as e:
        logger.error(f"[console-agent] refusing to start: {e}")
        raise SystemExit(1)
    server = make_server(config)
    host, port = server.server_address[:2]
    logger.info(f"[console-agent] serving {REPO_ROOT} on {host}:{port}, snapshots from {snapshot_dir}")
    server.serve_forever()


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `$PY -m pytest tests/console_agent -v -p no:cacheprovider`
Expected: all pass (10 + 28 + 22). If `test_agent_does_not_import_the_trading_stack` fails, the printed list names the offending module; remove that import path rather than relaxing the test.

- [ ] **Step 6: Smoke-run the entry point locally**

```bash
CONSOLE_OPERATOR_LOGIN= $PY -m src.console_agent; echo "exit=$?"
```

Expected: an error line `refusing to start: CONSOLE_OPERATOR_LOGIN is not set` and `exit=1`. (If the worktree's `.env` defines the variable, `load_dotenv` does not override the empty value already in the environment, so this still exercises the refusal.)

- [ ] **Step 7: Commit (tests, then implementation)**

```bash
git add tests/console_agent/test_server.py
git commit -m "test(console-agent): HTTP auth, routes and import-weight guard

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git add src/console_agent/server.py src/console_agent/__main__.py
git commit -m "feat(console-agent): read-only HTTP server with /status and /decisions

Binds 127.0.0.1:8090, requires Tailscale-User-Login to equal
CONSOLE_OPERATOR_LOGIN, rejects every write method, and refuses to
start when the login is not configured.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Deployment artifacts (unit, installer, env template, package README)

**Files:**
- Create: `infra/ec2/services/homeguard-console-agent.service`
- Create: `infra/ec2/setup/install_console_agent.sh`
- Create: `src/console_agent/README.md`
- Modify: `.env.example` (append a block after the EC2 section, around line 70)

**Interfaces:**
- Consumes: `python -m src.console_agent` (Task 5), port 8090.
- Produces: the unit `homeguard-console-agent.service` and an idempotent installer that the operator runs on the instance.

- [ ] **Step 1: Write the unit file**

Create `infra/ec2/services/homeguard-console-agent.service`:

```ini
[Unit]
Description=Homeguard Console Agent (read-only instance status for the local console)
After=network-online.target
Wants=network-online.target
# StartLimit* must live in [Unit]; systemd ignores them in [Service].
# 600s exceeds 5 x RestartSec, so a crash loop stops after five attempts.
StartLimitIntervalSec=600
StartLimitBurst=5

[Service]
Type=simple
User=ec2-user
WorkingDirectory=/home/ec2-user/Homeguard
Environment="PATH=/home/ec2-user/Homeguard/venv/bin:/usr/local/bin:/usr/bin:/bin"
ExecStart=/home/ec2-user/Homeguard/venv/bin/python -m src.console_agent
Restart=on-failure
RestartSec=10
# Killed before any trading process under memory pressure (homeguard-multi runs at 0).
MemoryMax=96M
OOMScoreAdjust=500
StandardOutput=journal
StandardError=journal
SyslogIdentifier=homeguard-console-agent

[Install]
WantedBy=multi-user.target
```

- [ ] **Step 2: Write the installer**

Create `infra/ec2/setup/install_console_agent.sh`:

```bash
#!/bin/bash
# Idempotent installer for the read-only Homeguard console agent (Phase 1).
# Run ON the instance as ec2-user from the repo root:
#   bash infra/ec2/setup/install_console_agent.sh
set -euo pipefail

REPO_DIR=/home/ec2-user/Homeguard
UNIT=homeguard-console-agent.service
AGENT_PORT=8090
TAILNET_PORT=8443

if ! grep -q '^CONSOLE_OPERATOR_LOGIN=..*' "$REPO_DIR/.env"; then
    echo "[-] CONSOLE_OPERATOR_LOGIN is not set in $REPO_DIR/.env"
    echo "    Add: CONSOLE_OPERATOR_LOGIN=\"<your tailscale login>\""
    exit 1
fi

echo "[+] Installing $UNIT"
sudo cp "$REPO_DIR/infra/ec2/services/$UNIT" "/etc/systemd/system/$UNIT"
sudo systemctl daemon-reload
sudo systemctl enable "$UNIT"
sudo systemctl restart "$UNIT"

for _ in $(seq 1 10); do
    if curl -s -o /dev/null "http://127.0.0.1:$AGENT_PORT/status"; then
        break
    fi
    sleep 1
done
if ! systemctl is-active --quiet "$UNIT"; then
    echo "[-] $UNIT is not active; see: journalctl -u $UNIT -n 50"
    exit 1
fi
echo "  $UNIT active"

echo "[+] Publishing 127.0.0.1:$AGENT_PORT on the tailnet at :$TAILNET_PORT"
tailscale version | head -1
if sudo tailscale serve status 2>/dev/null | grep -q "127.0.0.1:$AGENT_PORT"; then
    echo "  Already published via tailscale serve"
else
    sudo tailscale serve --bg --https="$TAILNET_PORT" "http://127.0.0.1:$AGENT_PORT"
fi
sudo tailscale serve status
```

- [ ] **Step 3: Syntax-check the installer**

Run: `bash -n infra/ec2/setup/install_console_agent.sh && echo ok`
Expected: `ok`.

- [ ] **Step 4: Add the env template entry**

Append to `.env.example`, directly after the `EC2_USER="ec2-user"` line's section:

```
# CONSOLE AGENT (set in the .env on the EC2 instance)
# The Tailscale login allowed to read the console agent; tailscale serve sends it
# as the Tailscale-User-Login header.
CONSOLE_OPERATOR_LOGIN="<YOUR_TAILSCALE_LOGIN>"
```

- [ ] **Step 5: Write the package README**

Create `src/console_agent/README.md`:

````markdown
# Console agent

Read-only HTTP agent on the EC2 instance that serves Homeguard's live state to
the local Homeguard Console. Design: `docs/superpowers/specs/2026-10-05-homeguard-console-design.md`.

## Routes (Phase 1)

| Route | Returns |
| --- | --- |
| `GET /status` | Units (systemd), per-strategy toggle state, units running each strategy, metric snapshot, latest decision gates, execution lock, and an `errors` list for any source that failed to read |
| `GET /decisions?strategy=<name>` | The full latest decision record from `data/trading/decisions/_latest/<name>.json` |

Every request must carry `Tailscale-User-Login` equal to `CONSOLE_OPERATOR_LOGIN`
(403 otherwise). Any write method returns 405.

## Rules

- Never import `src.trading`: its package `__init__` loads about 75MB of broker code,
  over the unit's `MemoryMax=96M`. `tests/console_agent/test_server.py` enforces this.
- Never write files and never construct `StrategyStateManager` (its `__init__` writes).
- Every subprocess goes through `units.run_command`.

## Run

```bash
# On the instance (via the unit):
bash infra/ec2/setup/install_console_agent.sh
curl -s -H "Tailscale-User-Login: $CONSOLE_OPERATOR_LOGIN" http://127.0.0.1:8090/status

# From the operator's machine over the tailnet:
curl -s https://<tailnet-host>:8443/status
```

## Tests

```bash
python -m pytest tests/console_agent -q
```
````

- [ ] **Step 6: Commit (config, then docs)**

```bash
git add infra/ec2/services/homeguard-console-agent.service infra/ec2/setup/install_console_agent.sh .env.example
git commit -m "feat(infra): console agent systemd unit and idempotent installer

MemoryMax=96M and OOMScoreAdjust=500 so the kernel kills the agent
before any trading process; not part of homeguard-trading.target.
The installer publishes 127.0.0.1:8090 on the tailnet at :8443.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git add src/console_agent/README.md
git commit -m "docs(console-agent): package README with routes, rules and run steps

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: Repo docs and merge to main

**Files:**
- Modify: `CLAUDE.md` (the `src/ packages` block, after the `discord_cscm/` line)
- Modify: `docs/architecture/ARCHITECTURE_OVERVIEW.md` (add a short subsection after the Discord bot components block, around line 425)

**Interfaces:**
- Consumes: everything above.
- Produces: `feat/console-p0-p1` merged into `main` and pushed (standing permission in CLAUDE.md).

- [ ] **Step 1: Add the package to CLAUDE.md**

After the line `discord_cscm/    CSCM-specific Discord alerts`, add:

```
console_agent/   Read-only instance agent for the Homeguard Console (/status, /decisions over tailnet :8443)
```

- [ ] **Step 2: Add the architecture subsection**

In `docs/architecture/ARCHITECTURE_OVERVIEW.md`, after the Discord bot "Key Components" list, add:

```markdown
### Console agent

**Key Components** ([src/console_agent/](../../src/console_agent/)):

- A read-only stdlib HTTP server on the instance (127.0.0.1:8090, published on the tailnet at :8443 by `tailscale serve`) that serves systemd unit state, the strategy toggle, the execution lock, metric snapshots and the latest decision records to the local Homeguard Console. It never imports `src.trading` and never writes. Design: `docs/superpowers/specs/2026-10-05-homeguard-console-design.md`.
```

- [ ] **Step 3: Run the full verification**

```bash
$PY -m pytest tests/console_agent tests/trading/test_state_manager_toggle_defaults.py -q -p no:cacheprovider
```

Then the Setup baseline command again. Expected: console agent and toggle tests all pass; baseline `167 passed`. Also run `grep -rnP '[^\x00-\x7F]' src/console_agent tests/console_agent infra/ec2/setup/install_console_agent.sh infra/ec2/services/homeguard-console-agent.service` and expect no output.

- [ ] **Step 4: Commit docs**

```bash
git add CLAUDE.md docs/architecture/ARCHITECTURE_OVERVIEW.md
git commit -m "docs(architecture): register the console agent package

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

- [ ] **Step 5: Record the commits to cherry-pick**

```bash
git log --oneline main..feat/console-p0-p1
```

The deploy branch needs every commit except the final `docs(architecture)` one (Tasks 1-6, in order). Save the list for the rollout.

- [ ] **Step 6: Merge to main (controller, after the whole-branch review is clean)**

From the main working tree (it has unrelated uncommitted files; a fast-forward that does not touch them is fine):

```bash
cd /c/Users/qwqw1/Dropbox/cs/github/Homeguard
git merge --ff-only feat/console-p0-p1
git log --oneline -1
git push origin main
```

Expected: fast-forward succeeds. If it refuses because a local uncommitted file would be overwritten, stop and report; do not stash or discard the user's files.

---

## Operator-gated rollout (controller only, each [operator] step needs an explicit go-ahead)

The instance is stopped outside 08:00-20:00 ET weekdays. Starting it is a power change; do it outside 09:15-16:15 ET on trading days.

- [ ] **R1 [operator]: Cherry-pick onto the deploy branch**

```bash
cd /c/Users/qwqw1/Dropbox/cs/github/Homeguard
git fetch origin ramp-phase4-turnover-regime-research
git branch -f ramp-phase4-turnover-regime-research origin/ramp-phase4-turnover-regime-research
git worktree add .worktrees/deploy-console ramp-phase4-turnover-regime-research
cd .worktrees/deploy-console
git cherry-pick <task1-test> <task1-fix> <task2-untrack> <task3-test> <task3-feat> <task4-test> <task4-feat> <task5-test> <task5-feat> <task6-infra> <task6-readme>
$PY -m pytest tests/console_agent tests/trading/test_state_manager_toggle_defaults.py tests/trading/test_state_manager_adopt.py tests/trading/test_state_manager_migration.py tests/trading/test_state_manager_broker_aware.py -q -p no:cacheprovider
```

Expected: every cherry-pick applies without conflict, and all tests pass on the deploy branch. On any conflict, `git cherry-pick --abort` and report. Then, with go-ahead: `git push origin ramp-phase4-turnover-regime-research`.

- [ ] **R2 [operator]: Check the instance Python and Tailscale versions**

```bash
ssh ec2 '~/Homeguard/venv/bin/python --version; tailscale version | head -1; sudo tailscale serve --help 2>&1 | grep -n -- "--https" | head -3'
```

Expected: Python 3.9 or newer, and `--https` listed as a `serve` flag. If `--https` is missing, stop and adjust the installer's serve line to the installed syntax before R4.

- [ ] **R3 [operator]: One-time toggle untrack on the instance (outside any decision window)**

```bash
ssh ec2 'cd ~/Homeguard && sha256sum config/trading/strategy_toggle.yaml && cp config/trading/strategy_toggle.yaml ~/strategy_toggle.yaml.pre-untrack && bash infra/ec2/instance_update_repo.sh; cp ~/strategy_toggle.yaml.pre-untrack config/trading/strategy_toggle.yaml && sha256sum config/trading/strategy_toggle.yaml; git ls-files --error-unmatch config/trading/strategy_toggle.yaml; echo "ls-files exit=$?"'
```

Expected: both sha256 lines match; `ls-files exit=1` (untracked). Do NOT pass `--restart` (it restarts the legacy homeguard-trading unit). No trading restart is needed; the next 08:00 boot picks up the fail-closed code.

- [ ] **R4 [operator]: Install the agent**

Add `CONSOLE_OPERATOR_LOGIN="<operator tailscale login>"` to `~/Homeguard/.env` on the instance, then:

```bash
ssh ec2 'cd ~/Homeguard && bash infra/ec2/setup/install_console_agent.sh'
```

Expected: `homeguard-console-agent.service active` and a `tailscale serve status` listing `:8443` proxying to `http://127.0.0.1:8090`.

- [ ] **R5: Phase 0 + Phase 1 exit gates**

```bash
# Phase 1: from the operator's machine over the tailnet
curl -s https://<tailnet-host>:8443/status | $PY -c "import json,sys; d=json.load(sys.stdin); print(sorted(d['strategies'])); print('mp units', d['strategies']['mp']['units']); print({k: v['last_decision'] and v['last_decision']['all_passed'] for k, v in d['strategies'].items()}); print('errors', d['errors'])"
# Send a forged header from the operator's device: tailscale serve must replace it with the real login
curl -s -o /dev/null -w "%{http_code}\n" -H "Tailscale-User-Login: someone-else@example.com" https://<tailnet-host>:8443/status
# No header on the loopback port
ssh ec2 'curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:8090/status; systemctl show -p MemoryCurrent homeguard-console-agent'
# Phase 0: the toggle stays untracked and unchanged across a later pull
ssh ec2 'cd ~/Homeguard && sha256sum config/trading/strategy_toggle.yaml && git pull --ff-only && sha256sum config/trading/strategy_toggle.yaml && git ls-files config/trading/strategy_toggle.yaml | wc -l'
```

Expected:
- `['cscm', 'mp', 'omr', 'ramp']`, `mp units []`, decision summaries present for the strategies that have run.
- The forged-header request returns `200`. That proves `tailscale serve` replaced the client's `someone-else@example.com` with the operator's real login (the agent only accepts the operator's login, so the forged value cannot produce a 200). A `403` means serve passed the client's header through (or put it first); in that case stop and add a tailnet ACL restricting :8443 to the operator's devices before relying on the agent.
- Loopback without header: `403`. `MemoryCurrent` well under 100663296 (96M).
- Phase 0: identical sha256 lines and `0` tracked files.

- [ ] **R6: Session log**

Write `docs/progress/20261005_CONSOLE_PHASE0_PHASE1.md` in the CLAUDE.md session-log format (summary, changes, commits with hashes on main and the deploy branch, remaining work: Phase 2 and Phase 3, the open questions from the spec, the stash follow-up), force-add it (`docs/*` is gitignored), commit, and push main.
