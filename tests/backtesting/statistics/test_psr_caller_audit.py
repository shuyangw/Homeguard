"""AST audit: no live gate caller may forget `periods_per_year`.

Three call sites survived the 2026-07-25 unit-bug migration because the kwarg
landed on a CONTINUATION line and the search-and-replace was line-based. This
checks the PARSED call, not the text, so that class of miss cannot recur.
"""
import ast
import pathlib

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[3]
_TARGETS = [
    "src/backtesting/walkforward_common.py",
    "src/backtesting/session/session_walkforward.py",
    "src/backtesting/blend/satellite_blend.py",
    "scripts/backtest_scripts/run_fx_walkforward.py",
    "scripts/backtest_scripts/run_fx_spread_walkforward.py",
    "scripts/backtest_scripts/run_carver_walkforward.py",
    "scripts/backtest_scripts/run_fx_carry_seatbelt_walkforward.py",
    "scripts/backtest_scripts/run_fx_london_breakout_walkforward.py",
]


def _offenders(path: pathlib.Path):
    out = []
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        name = fn.id if isinstance(fn, ast.Name) else getattr(fn, "attr", None)
        if name not in ("psr", "dsr"):
            continue
        if not any(k.arg == "periods_per_year" for k in node.keywords):
            out.append(f"{path.name}:{node.lineno}")
    return out


@pytest.mark.parametrize("rel", _TARGETS)
def test_gate_caller_passes_periods_per_year(rel):
    path = _ROOT / rel
    if not path.exists():
        pytest.skip(f"{rel} not present")
    bad = _offenders(path)
    assert not bad, (
        f"psr/dsr called without periods_per_year -> annualized Sharpe against a "
        f"daily n, inflating the statistic by ~sqrt(252): {bad}")
