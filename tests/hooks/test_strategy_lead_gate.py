"""Tests for the strategy_lead_gate PreToolUse hook decision function.

The hook must fire ONLY when a command actually invokes python/pytest to RUN a
strategy backtest -- never for git commands (commit messages, staging runner
files), read-only tools (grep/cat), or `python -m py_compile`.
"""
import importlib.util
from pathlib import Path

import pytest

_HOOK = Path(__file__).resolve().parents[2] / ".claude" / "hooks" / "strategy_lead_gate.py"
spec = importlib.util.spec_from_file_location("strategy_lead_gate", _HOOK)
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


@pytest.mark.parametrize("cmd,sentinel_exists,expected", [
    # ALLOW: git commit message merely names a runner
    ('git commit -m "fix: run_futures_backtest threads x"', False, False),
    # ALLOW: staging a runner file
    ("git add -f scripts/backtest_scripts/run_fx_walkforward.py", False, False),
    # ALLOW: py_compile only byte-compiles, never executes
    ("PYTHONPATH=/x python -m py_compile scripts/backtest_scripts/run_fx_walkforward.py", False, False),
    # ALLOW: grep is read-only
    ("grep -n run_fx_walkforward scripts/backtest_scripts/run_fx_walkforward.py", False, False),
    # ALLOW: echo is not python
    ("echo run_futures_backtest", False, False),
    # BLOCK: real backtest runner execution
    ("python -m src.backtest_runner --config config/x.yaml", False, True),
    # BLOCK: real walk-forward runner execution
    ("PYTHONPATH=/x python scripts/backtest_scripts/run_fx_walkforward.py", False, True),
    # BLOCK: chained compile then real run
    ("python -m py_compile a.py && PYTHONPATH=/x python scripts/backtest_scripts/run_carver_walkforward.py", False, True),
    # NOT BLOCK when sentinel present (strategy-lead owns the phase)
    ("python -m src.backtest_runner --config x", True, False),
    # NOT BLOCK when no runner-name token at all
    ("python -m pytest tests/foo.py", False, False),
])
def test_should_block(cmd, sentinel_exists, expected):
    assert gate._should_block(cmd, sentinel_exists) is expected
