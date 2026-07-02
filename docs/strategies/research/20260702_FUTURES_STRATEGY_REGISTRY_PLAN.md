# Strategy-Pluggable Futures Runner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `run_futures_backtest` resolve its strategy from config (via the existing `src/strategies/registry.py`) instead of hardcoding `CarverMomentumStrategy`, so a new futures signal is a registry entry + a config `name`, not a runner edit.

**Architecture:** Register Carver in the existing generic registry. Add a small `@runtime_checkable SupportsForecastPanel` protocol inline in the futures engine. Reorder `run_futures_backtest` to resolve + instantiate + validate the strategy BEFORE the data load (so bad names/contracts fail fast), then call `strat.forecast_panel(close)`. Config gains optional `strategy.name` (default `CarverMomentum`) and `strategy.params`.

**Tech Stack:** Python 3.13, pandas, pytest, typing.Protocol. Conda env `fintech`.

## Global Constraints

- **Python execution:** ALWAYS `/c/Users/qwqw1/anaconda3/envs/fintech/python.exe -m pytest <args>`. Never system Python.
- **ASCII only** (Windows cp1252). No `print()` (use `src.utils.logger`, f-strings).
- **Base branch:** `feat/futures-strategy-registry` (already checked out, off `main` @ a4972f0). Do NOT switch.
- **Backward compatibility (binding):** existing futures configs (`carver_tsmom.yaml`, `carver_tsmom_broad.yaml`) have `strategy.universe` but NO `strategy.name` / `strategy.params`. A no-`name` config MUST run `CarverMomentum` and produce an IDENTICAL result to today. The report label and registry `strategy_name` for a no-`name` config MUST stay `"CarverMomentum"`.
- **Isolation:** touch ONLY `src/strategies/registry.py`, `src/backtesting/engine/futures_backtest.py`, and test files. Do NOT change the simulator, sizing, loader, carver strategy, the equity/crypto path, or existing YAML configs.
- **Strategy contract:** a registered futures forecast strategy constructs as `__init__(self, universe, **params)` and exposes `forecast_panel(close_panel: pd.DataFrame) -> pd.DataFrame` (per-root forecast in the +/- cap convention `run_sized` sizes).

---

## Task 1: Register Carver in the strategy registry

**Files:**
- Modify: `src/strategies/registry.py` (add to `_STRATEGY_REGISTRY` and `_DISPLAY_NAME_MAP`)
- Test: `tests/strategies/test_registry_carver.py`

**Interfaces:**
- Produces: `get_strategy_class("CarverMomentum")` (and aliases `"Carver"`, `"Carver TSMOM"`, `"Carver Momentum"`) resolves to `CarverMomentumStrategy`. Consumed by Task 2.

- [ ] **Step 1: Write the failing test**

```python
# tests/strategies/test_registry_carver.py
from src.strategies.registry import get_strategy_class
from src.strategies.advanced.carver_momentum_strategy import CarverMomentumStrategy


def test_carver_registered_by_class_name():
    assert get_strategy_class("CarverMomentum") is CarverMomentumStrategy


def test_carver_registered_by_aliases():
    assert get_strategy_class("Carver") is CarverMomentumStrategy
    assert get_strategy_class("Carver TSMOM") is CarverMomentumStrategy
    assert get_strategy_class("Carver Momentum") is CarverMomentumStrategy
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/c/Users/qwqw1/anaconda3/envs/fintech/python.exe -m pytest tests/strategies/test_registry_carver.py -v`
Expected: FAIL — `ValueError: Unknown strategy: 'CarverMomentum'` (not yet registered).

- [ ] **Step 3: Register Carver**

In `src/strategies/registry.py`, add to `_STRATEGY_REGISTRY` (in the "Advanced/Production strategies" block):

```python
    "CarverMomentum": ("src.strategies.advanced.carver_momentum_strategy", "CarverMomentumStrategy"),
```

And add to `_DISPLAY_NAME_MAP`:

```python
    "Carver": "CarverMomentum",
    "Carver TSMOM": "CarverMomentum",
    "Carver Momentum": "CarverMomentum",
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/c/Users/qwqw1/anaconda3/envs/fintech/python.exe -m pytest tests/strategies/test_registry_carver.py -v`
Expected: PASS (both tests).

- [ ] **Step 5: Commit**

```bash
git add src/strategies/registry.py tests/strategies/test_registry_carver.py
git commit -m "feat(futures): register CarverMomentum in the strategy registry"
```

---

## Task 2: Pluggable strategy resolution in run_futures_backtest

**Files:**
- Modify: `src/backtesting/engine/futures_backtest.py`
- Test: `tests/backtesting/engine/test_futures_backtest_pluggable.py`

**Interfaces:**
- Consumes: `get_strategy_class` (Task 1 registration).
- Produces: `run_futures_backtest` resolves `config["strategy"].get("name", "CarverMomentum")` + `config["strategy"].get("params", {})`; a new inline `@runtime_checkable` Protocol `SupportsForecastPanel` with `forecast_panel(self, close_panel: pd.DataFrame) -> pd.DataFrame`.

**Context — current code (verified, lines 43-91):**
- `strategy_cfg = config.get("strategy", {})`; `universe = list(strategy_cfg["universe"])`.
- Line 58: `forecasts = CarverMomentumStrategy(universe).forecast_panel(close)` (after `load_daily_panel`).
- Line 74: `generate_report(..., "CarverMomentum", universe, ...)`.
- Line 83: `append_run(strategy_name="CarverMomentum", ...)`.
- The direct import `from src.strategies.advanced.carver_momentum_strategy import CarverMomentumStrategy` (top of file) becomes unused after this task and must be removed.

- [ ] **Step 1: Write the failing tests**

```python
# tests/backtesting/engine/test_futures_backtest_pluggable.py
import pandas as pd
import pytest

from src.backtesting.engine.futures_backtest import run_futures_backtest
from src.strategies.registry import register_strategy

_SLICE = {
    "strategy": {"universe": ["6E", "GC"]},
    "dates": {"start": "2022-01-03", "end": "2022-06-30"},
    "backtest": {"initial_capital": 100000, "vol_target_per_instrument": 0.20,
                 "rebalance": "weekly", "cost_mult": 1.0},
}


class _ZeroForecast:
    """Stub futures strategy: forecast 0 everywhere -> no positions, flat equity."""
    def __init__(self, universe, **params):
        self.universe = list(universe)

    def forecast_panel(self, close_panel: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(0.0, index=close_panel.index, columns=close_panel.columns)


class _NoForecast:
    """Stub lacking forecast_panel -> must be rejected."""
    def __init__(self, universe, **params):
        self.universe = list(universe)


def test_unknown_strategy_name_raises_fast():
    cfg = {**_SLICE, "strategy": {"name": "NoSuchStrategy", "universe": ["6E", "GC"]}}
    with pytest.raises(ValueError):
        run_futures_backtest(cfg)


def test_strategy_missing_forecast_panel_raises():
    register_strategy("NoForecastStub", _NoForecast)
    cfg = {**_SLICE, "strategy": {"name": "NoForecastStub", "universe": ["6E", "GC"]}}
    with pytest.raises(ValueError):
        run_futures_backtest(cfg)


def test_stub_strategy_is_actually_used():
    # Zero-forecast stub -> no trades -> equity stays flat at initial capital.
    register_strategy("ZeroForecastStub", _ZeroForecast)
    cfg = {**_SLICE, "strategy": {"name": "ZeroForecastStub", "universe": ["6E", "GC"]}}
    res = run_futures_backtest(cfg)
    eq = res["equity_curve"]
    assert eq, "empty equity curve"
    assert all(abs(v - 100000) < 1e-6 for v in eq), "stub not used (equity moved)"


def test_default_name_runs_carver_backward_compat():
    # No strategy.name -> Carver; must produce a non-flat, finite equity curve.
    res = run_futures_backtest(_SLICE)
    eq = res["equity_curve"]
    assert eq and all(isinstance(v, float) for v in eq)
    assert any(abs(v - 100000) > 1e-6 for v in eq), "Carver produced no trades (unexpected)"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/c/Users/qwqw1/anaconda3/envs/fintech/python.exe -m pytest tests/backtesting/engine/test_futures_backtest_pluggable.py -v`
Expected: FAIL — `test_unknown_strategy_name_raises_fast` and `test_strategy_missing_forecast_panel_raises` fail (no resolution/validation yet; the hardcoded Carver ignores `name`); the stub test fails (Carver runs, not the stub).

- [ ] **Step 3: Add the protocol import + definition**

At the top of `src/backtesting/engine/futures_backtest.py`, add to the typing imports:

```python
from typing import Any, Dict, Protocol, runtime_checkable
```
Add the registry import (and REMOVE the now-unused `from src.strategies.advanced.carver_momentum_strategy import CarverMomentumStrategy`):

```python
from src.strategies.registry import get_strategy_class
```
Add the protocol near the top of the module (after imports, before `run_futures_backtest`):

```python
@runtime_checkable
class SupportsForecastPanel(Protocol):
    def forecast_panel(self, close_panel: pd.DataFrame) -> pd.DataFrame:
        ...
```

- [ ] **Step 4: Resolve + validate the strategy BEFORE the data load**

In `run_futures_backtest`, replace the strategy parse + hardcoded call. After the existing `universe = list(strategy_cfg["universe"])` and param parsing (lines 47-53), insert resolution BEFORE `load_daily_panel` (so bad names/contracts fail fast):

```python
    strategy_name = strategy_cfg.get("name", "CarverMomentum")
    strategy_params = strategy_cfg.get("params", {})
    strategy_cls = get_strategy_class(strategy_name)   # raises ValueError on unknown name
    strategy = strategy_cls(universe, **strategy_params)
    if not isinstance(strategy, SupportsForecastPanel):
        raise ValueError(
            f"Strategy '{strategy_name}' does not implement forecast_panel(close_panel); "
            f"it cannot be used on the futures forecast path."
        )
```
Then change the old line 58 from constructing Carver to using the resolved `strategy`:

```python
    forecasts = strategy.forecast_panel(close)
```
(Keep `load_daily_panel` + `close` extraction exactly where they are, between the resolution block and this line.)

- [ ] **Step 5: Use the resolved name in the report label and registry**

Replace the hardcoded `"CarverMomentum"` at the `generate_report` call (line 74) and the `append_run(strategy_name=...)` call (line 83) with `strategy_name`:

```python
    report = StandardReportGenerator().generate_report(
        res.equity_curve, strategy_name, universe,
        str(start), str(end), capital,
    )
```
```python
        run_id = append_run(
            strategy_name=strategy_name,
            agent_name="futures-harness",
            ...
```
(Because `strategy_name` defaults to `"CarverMomentum"`, a no-`name` config keeps the exact same label -- backward compatible.)

- [ ] **Step 6: Run the pluggable tests to verify they pass**

Run: `/c/Users/qwqw1/anaconda3/envs/fintech/python.exe -m pytest tests/backtesting/engine/test_futures_backtest_pluggable.py -v`
Expected: PASS (all four: unknown-name raises, missing-forecast_panel raises, stub-used flat equity, default runs Carver non-flat).

- [ ] **Step 7: Confirm the existing e2e still passes (equivalence / no regression)**

Run: `/c/Users/qwqw1/anaconda3/envs/fintech/python.exe -m pytest tests/backtesting/engine/test_futures_backtest_e2e.py -v`
Expected: PASS unchanged -- the e2e config has no `strategy.name`, so it defaults to Carver and must yield the same Sharpe as before B (the resolution is behavior-neutral for the default path).

- [ ] **Step 8: Confirm no dead import remains**

Run: `grep -n "CarverMomentumStrategy" src/backtesting/engine/futures_backtest.py`
Expected: NO matches (the direct import and the hardcoded construction are both gone; Carver is reached only via the registry now).

- [ ] **Step 9: Commit**

```bash
git add src/backtesting/engine/futures_backtest.py tests/backtesting/engine/test_futures_backtest_pluggable.py
git commit -m "feat(futures): resolve strategy from config via registry (pluggable runner)"
```

---

## Self-Review

- **Spec coverage:** Task 1 = register Carver + aliases (spec component 1). Task 2 = protocol + config-driven resolution + validation + resolved-name labeling + dead-import removal (spec components 2, 3, error handling). All covered.
- **Placeholder scan:** none -- every edit and test has concrete code.
- **Type consistency:** `SupportsForecastPanel.forecast_panel(self, close_panel) -> pd.DataFrame` matches `CarverMomentumStrategy.forecast_panel` and the stub. `strategy_cls(universe, **strategy_params)` matches the `__init__(self, universe, **params)` contract (Carver's signature `(universe, speeds=None, forecast_cap=20.0, **params)` accepts this).
- **Backward compat:** `strategy_name` defaults to `"CarverMomentum"`; report label + registry name unchanged for name-less configs; `test_default_name_runs_carver_backward_compat` + the e2e (Step 7) prove it.
- **Fail-fast ordering:** resolution/validation placed before `load_daily_panel`, so `test_unknown_strategy_name_raises_fast` and the missing-`forecast_panel` test do not pay the data-load cost.
- **Isolation:** only `registry.py`, `futures_backtest.py`, and two test files change; no simulator/sizing/loader/carver/equity-path/YAML edits.
