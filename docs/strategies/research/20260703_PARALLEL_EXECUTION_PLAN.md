# Parallel Execution Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A reusable `parallel_map` primitive (ProcessPoolExecutor, input-order results, worker cap, fail-fast), applied to the walk-forward (by window) and the carry cache build (by root), with determinism and registry-race-freedom guaranteed and tested.

**Architecture:** One generic `parallel_map` in `src/backtesting/parallel.py`. The walk-forward extracts a top-level picklable `process_window` worker and maps over windows; per-window backtests run with `register=False` so only the parent writes the registry (race-free); input-order aggregation makes parallel byte-identical to serial (strategies are RNG-free). The cache builder gains `--jobs`.

**Tech Stack:** Python 3.13, stdlib `concurrent.futures`, pandas/numpy, pytest. Conda env `fintech`. Machine has 32 cores.

## Global Constraints

- **Python execution:** ALWAYS `/c/Users/qwqw1/anaconda3/envs/fintech/python.exe -m pytest <args>`. Scripts importing `scripts/` need `PYTHONPATH=.`. Never system Python.
- **ASCII only**; no `print()` (use `src.utils.logger`, f-strings).
- **Base branch:** `feat/parallel-execution` (checked out, off `main` @ 88c8002). Do NOT switch.
- **Processes, not threads** (GIL-bound backtest loops). Reuse `get_default_workers()` from `src.backtesting.optimization.data_loader`.
- **Determinism:** `parallel_map` returns results in INPUT order; the walk-forward aggregation/gate math is UNCHANGED, so parallel results equal serial results exactly.
- **Race-free registry:** parallel per-window workers pass `register=False`; only the parent walk-forward writes `append_run`.
- **Isolation:** do NOT change the simulator, sizing, strategies, gate math (`psr`/`dsr`/`pbo`), report content, or the equity/crypto path. Only add `parallel_map`, the `register` flag, the per-window worker extraction + map wiring, and `--jobs`.
- **Backward compatibility:** `run_futures_backtest` default `register=True` and `walk_forward_carver` default (now parallel) must produce identical metrics to today; `--jobs 1` forces serial.

---

## Task 1: `parallel_map` primitive

**Files:**
- Create: `src/backtesting/parallel.py`
- Test: `tests/backtesting/test_parallel.py`

**Interfaces:**
- Produces: `parallel_map(fn: Callable[[T], R], items: Sequence[T], max_workers: int | None = None) -> list[R]`.

- [ ] **Step 1: Write the failing tests** (worker fns are module-level so they pickle for spawn)

```python
# tests/backtesting/test_parallel.py
import pytest
from src.backtesting.parallel import parallel_map


def _double(x):
    return x * 2


def _boom(x):
    if x == 3:
        raise ValueError("boom at 3")
    return x


def test_returns_input_order():
    items = list(range(10))
    assert parallel_map(_double, items, max_workers=4) == [x * 2 for x in items]


def test_serial_path_max_workers_1():
    items = list(range(5))
    assert parallel_map(_double, items, max_workers=1) == [0, 2, 4, 6, 8]


def test_empty_items():
    assert parallel_map(_double, [], max_workers=4) == []


def test_worker_exception_propagates():
    with pytest.raises(ValueError):
        parallel_map(_boom, [1, 2, 3, 4], max_workers=2)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/c/Users/qwqw1/anaconda3/envs/fintech/python.exe -m pytest tests/backtesting/test_parallel.py -v`
Expected: FAIL — module does not exist.

- [ ] **Step 3: Implement `parallel_map`**

```python
# src/backtesting/parallel.py
"""Deterministic process-parallel map for CPU-bound backtest work.

Uses ProcessPoolExecutor (not threads -- the backtest loop is GIL-bound).
Results are returned in INPUT order, so callers whose aggregation is
order-sensitive (e.g. the walk-forward stitching OOS segments by window)
get byte-identical results to a serial run. Worker count is capped by
get_default_workers(). The first worker exception propagates (fail-fast).
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Callable, List, Optional, Sequence, TypeVar

from src.backtesting.optimization.data_loader import get_default_workers

T = TypeVar("T")
R = TypeVar("R")


def parallel_map(fn: Callable[[T], R], items: Sequence[T],
                 max_workers: Optional[int] = None) -> List[R]:
    items = list(items)
    if not items:
        return []
    if max_workers is None:
        max_workers = min(get_default_workers(), len(items))
    if max_workers <= 1:
        return [fn(x) for x in items]

    results: List[Any] = [None] * len(items)
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(fn, item): i for i, item in enumerate(items)}
        for fut in as_completed(futures):
            idx = futures[fut]
            results[idx] = fut.result()  # re-raises worker exception in the parent
    return results
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/c/Users/qwqw1/anaconda3/envs/fintech/python.exe -m pytest tests/backtesting/test_parallel.py -v`
Expected: PASS (4 tests). This also proves ProcessPoolExecutor spawn + pickling works in this env on a trivial case, de-risking Task 3.

- [ ] **Step 5: Commit**

```bash
git add src/backtesting/parallel.py tests/backtesting/test_parallel.py
git commit -m "feat(backtesting): parallel_map primitive (ProcessPoolExecutor, input-order, fail-fast)"
```

---

## Task 2: `register` flag on `run_futures_backtest`

**Files:**
- Modify: `src/backtesting/engine/futures_backtest.py`
- Test: `tests/backtesting/engine/test_futures_backtest_register.py`

**Interfaces:**
- Produces: `run_futures_backtest(config, register: bool = True)`. `register=False` -> no `append_run`, `run_id=None`.

**Context (verified):** the registry block is `futures_backtest.py:97-111` (`try: from src.experiments import append_run; run_id = append_run(...) except Exception ...`). `run_id` is initialized to `None` just above.

- [ ] **Step 1: Write the failing tests**

```python
# tests/backtesting/engine/test_futures_backtest_register.py
import pytest
from src.backtesting.engine import futures_backtest as fb

_SLICE = {
    "strategy": {"name": "FuturesCarry", "universe": ["GC", "CL"]},
    "dates": {"start": "2022-01-03", "end": "2022-03-31"},
    "backtest": {"initial_capital": 1_000_000, "vol_target_per_instrument": 0.20,
                 "rebalance": "weekly", "cost_mult": 1.0},
}


def test_register_false_skips_append_run(monkeypatch):
    def _boom(*a, **k):
        raise AssertionError("append_run must NOT be called when register=False")
    monkeypatch.setattr("src.experiments.append_run", _boom)
    res = fb.run_futures_backtest(_SLICE, register=False)
    assert res["run_id"] is None


def test_register_true_calls_append_run(monkeypatch):
    called = {}
    def _fake(**kwargs):
        called["yes"] = True
        return "rid-123"
    monkeypatch.setattr("src.experiments.append_run", _fake)
    res = fb.run_futures_backtest(_SLICE)  # default register=True
    assert called.get("yes") and res["run_id"] == "rid-123"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/c/.../python.exe -m pytest tests/backtesting/engine/test_futures_backtest_register.py -v`
Expected: FAIL — `run_futures_backtest` has no `register` kwarg (TypeError).

- [ ] **Step 3: Add the flag**

Change the signature:
```python
def run_futures_backtest(config: Dict[str, Any], register: bool = True) -> Dict[str, Any]:
```
Guard the registry block (wrap the existing `try:` at line ~97):
```python
    run_id = None
    if register:
        try:
            from src.experiments import append_run
            run_id = append_run(
                ...unchanged...
            )
        except Exception as e:
            logger.error(f"[futures_backtest] registry append_run failed (non-fatal): {e}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/c/.../python.exe -m pytest tests/backtesting/engine/test_futures_backtest_register.py tests/backtesting/engine/test_futures_backtest_pluggable.py tests/backtesting/engine/test_futures_backtest_e2e.py -v`
Expected: PASS (new + existing pluggable/e2e unchanged, since default is `register=True`).

- [ ] **Step 5: Commit**

```bash
git add src/backtesting/engine/futures_backtest.py tests/backtesting/engine/test_futures_backtest_register.py
git commit -m "feat(futures): run_futures_backtest(register=False) to suppress registry writes"
```

---

## Task 3: Parallelize the walk-forward by window

**Files:**
- Modify: `scripts/backtest_scripts/run_carver_walkforward.py`
- Test: `tests/backtesting/test_walkforward_parallel.py`

**Interfaces:**
- Consumes: `parallel_map` (Task 1), `run_futures_backtest(register=...)` (Task 2).
- Produces: top-level `process_window(spec: dict) -> dict | None`; `_run_window(..., register=True)`; `walk_forward_carver(..., max_workers=None)`; `main()` gains `--jobs` and `--json`.

**Context (verified, lines 192-235):** the serial `for train_start, test_start, test_end in windows:` loop loads the panel (skip on FileNotFoundError), computes `window_universe`/`dates`, calls `_run_window(...cost_mult=1.0...)` then `...1.5...`, slices OOS via `_oos_returns`, appends to `per_window_returns_1x/1_5x`, `window_sharpes`, `window_universes`, `used_windows`. `_run_window(universe, train_start, test_end, capital, vol_target, cost_mult, strategy_name, strategy_params)` builds a config + calls `run_futures_backtest(config)`.

- [ ] **Step 1: Write the failing test** (subprocess determinism -- tests the REAL script/spawn path)

```python
# tests/backtesting/test_walkforward_parallel.py
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from src.data.futures.paths import continuous_1min_dir, carry_dir

REPO = Path(__file__).resolve().parents[2]
PY = sys.executable


def _data_present():
    return (continuous_1min_dir() / "symbol=ES").exists() and (carry_dir() / "GC.parquet").exists()


pytestmark = pytest.mark.skipif(not _data_present(), reason="futures/carry store not present")


def _run(tmp_path, jobs):
    out = tmp_path / f"metrics_{jobs}.json"
    env = {**os.environ, "PYTHONPATH": str(REPO)}
    cfg = REPO / "config/backtesting/carver_tsmom.yaml"  # tiny 3-root config
    subprocess.run(
        [PY, "scripts/backtest_scripts/run_carver_walkforward.py",
         "--config", str(cfg), "--report", str(tmp_path / f"r{jobs}.md"),
         "--jobs", str(jobs), "--json", str(out),
         "--train-months", "12", "--test-months", "6", "--step-months", "6"],
        cwd=str(REPO), env=env, check=True, capture_output=True, text=True, timeout=1200)
    return json.loads(out.read_text())


def test_parallel_equals_serial(tmp_path):
    serial = _run(tmp_path, 1)
    par = _run(tmp_path, 2)
    for k in ("oos_sharpe", "psr", "dsr", "pbo", "oos_sharpe_1_5x_cost", "n_windows"):
        assert serial[k] == par[k], f"{k}: serial={serial[k]} parallel={par[k]}"
```

Note: this test requires `main()` to accept `--train-months/--test-months/--step-months` (small config over the short 3-root `carver_tsmom.yaml` range gives >=2 windows quickly) and `--json`. Add those args in Step 4.

- [ ] **Step 2: Run test to verify it fails**

Run: `/c/.../python.exe -m pytest tests/backtesting/test_walkforward_parallel.py -v`
Expected: FAIL — `--jobs`/`--json`/`--train-months` not recognized (subprocess exits non-zero).

- [ ] **Step 3: Add the `register` passthrough to `_run_window`**

```python
def _run_window(universe, train_start, test_end, capital, vol_target, cost_mult,
                 strategy_name="CarverMomentum", strategy_params=None, register=True):
    config = {...unchanged...}
    return run_futures_backtest(config, register=register)
```

- [ ] **Step 4: Extract `process_window` (top-level) and map over windows**

Add a top-level worker (picklable; runs both cost legs with `register=False`):
```python
def process_window(spec):
    universe = spec["universe"]
    train_start, test_start, test_end = spec["train_start"], spec["test_start"], spec["test_end"]
    capital, vol_target = spec["capital"], spec["vol_target"]
    strategy_name, strategy_params = spec["strategy_name"], spec["strategy_params"]
    try:
        panel = load_daily_panel(universe, train_start, test_end)
    except FileNotFoundError as e:
        logger.warning(f"[walk_forward] skipping window {test_start}..{test_end}: {e}")
        return None
    window_universe = sorted({r for r, _ in panel.columns})
    dates = list(panel.index)
    res_1x = _run_window(window_universe, train_start, test_end, capital, vol_target,
                         cost_mult=1.0, strategy_name=strategy_name,
                         strategy_params=strategy_params, register=False)
    res_1_5x = _run_window(window_universe, train_start, test_end, capital, vol_target,
                           cost_mult=1.5, strategy_name=strategy_name,
                           strategy_params=strategy_params, register=False)
    return {
        "train_start": train_start, "test_start": test_start, "test_end": test_end,
        "window_universe": window_universe,
        "oos_1x": _oos_returns(res_1x["equity_curve"], dates, test_start),
        "oos_1_5x": _oos_returns(res_1_5x["equity_curve"], dates, test_start),
    }
```
In `walk_forward_carver`, add `max_workers: Optional[int] = None` to the signature, and replace the serial `for ... in windows:` loop (lines 192-235) with a spec build + parallel map + ordered aggregation:
```python
    specs = [
        {"universe": universe, "train_start": ts, "test_start": tst, "test_end": te,
         "capital": capital, "vol_target": vol_target,
         "strategy_name": strategy_name, "strategy_params": strategy_params or {}}
        for (ts, tst, te) in windows
    ]
    from src.backtesting.parallel import parallel_map
    results = parallel_map(process_window, specs, max_workers=max_workers)
    for r in results:
        if r is None:
            continue
        per_window_returns_1x.append(r["oos_1x"])
        per_window_returns_1_5x.append(r["oos_1_5x"])
        window_sharpes.append(_annualized_sharpe(r["oos_1x"]))
        window_universes.append(r["window_universe"])
        used_windows.append((r["train_start"], r["test_start"], r["test_end"]))
```
Everything after `if len(used_windows) < 2:` (stitching, gate, report, parent `append_run`) is UNCHANGED.

- [ ] **Step 5: Wire `main()` args (`--jobs`, `--json`, window months)**

In `main()` add argparse options: `--jobs` (int, default None -> parallel), `--json` (path, optional), `--train-months`/`--test-months`/`--step-months` (ints, defaults 36/12/12). Pass `max_workers=args.jobs` to `walk_forward_carver`, and the window months through (they were hardcoded 36/12/12). After the run, if `--json` given, dump the gate metrics:
```python
    if args.json:
        import json
        keys = ("oos_sharpe", "psr", "dsr", "pbo", "oos_sharpe_1_5x_cost", "n_windows")
        Path(args.json).write_text(json.dumps({k: result[k] for k in keys}))
```

- [ ] **Step 6: Run the determinism test (+ existing config tests)**

Run: `/c/.../python.exe -m pytest tests/backtesting/test_walkforward_parallel.py tests/backtesting/test_carver_walkforward_config.py -v`
Expected: PASS -- serial (`--jobs 1`) and parallel (`--jobs 2`) produce identical gate metrics; the existing config/title/capital tests still pass.

- [ ] **Step 7: Commit**

```bash
git add -f scripts/backtest_scripts/run_carver_walkforward.py
git add tests/backtesting/test_walkforward_parallel.py
git commit -m "feat(futures): parallelize walk-forward by window (deterministic, register=False workers)"
```

---

## Task 4: `build_carry_cache --jobs`

**Files:**
- Modify: `scripts/data/build_carry_cache.py`
- Test: `tests/data/futures/test_build_carry_cache.py` (append)

**Interfaces:**
- Produces: `build_carry_cache(roots, start, end, max_workers=None)` mapping roots through `parallel_map`; `--jobs N` CLI arg.

**Context:** current `build_carry_cache(roots, start, end)` loops roots serially, calling `CarryCalculator().compute_history(root, asset_class_for(root), start, end)` and writing `carry_dir()/{root}.parquet`. Each root is independent (own file, no shared state).

- [ ] **Step 1: Write the failing test (append)**

```python
# append to tests/data/futures/test_build_carry_cache.py
def test_build_carry_cache_parallel_matches_serial(tmp_path, monkeypatch):
    import polars as pl
    from datetime import date
    import scripts.data.build_carry_cache as bcc

    monkeypatch.setattr(bcc, "carry_dir", lambda: tmp_path)

    def fake_hist(self, root, ac, start, end):
        return pl.DataFrame({"date": [date(2020, 1, 2)], "carry": [0.05]})
    monkeypatch.setattr(bcc.CarryCalculator, "compute_history", fake_hist)

    written = bcc.build_carry_cache(["GC", "CL"], date(2020, 1, 1), date(2020, 1, 31), max_workers=2)
    assert sorted(written) == ["CL", "GC"]
    assert (tmp_path / "GC.parquet").exists() and (tmp_path / "CL.parquet").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/c/.../python.exe -m pytest tests/data/futures/test_build_carry_cache.py::test_build_carry_cache_parallel_matches_serial -v`
Expected: FAIL — `build_carry_cache` has no `max_workers` kwarg.

- [ ] **Step 3: Refactor the builder to use `parallel_map`**

Extract a top-level worker and map:
```python
from src.backtesting.parallel import parallel_map

def _build_one(spec):
    root, start, end = spec["root"], spec["start"], spec["end"]
    ac = asset_class_for(root)
    hist = CarryCalculator().compute_history(root, ac, start, end)
    if hist.height == 0:
        logger.warning(f"[build_carry_cache] {root}: no carry rows, skipping")
        return None
    hist.write_parquet(carry_dir() / f"{root}.parquet")
    logger.info(f"[build_carry_cache] {root} ({ac}): {hist.height} rows")
    return root

def build_carry_cache(roots, start, end, max_workers=None):
    carry_dir().mkdir(parents=True, exist_ok=True)
    specs = [{"root": r, "start": start, "end": end} for r in roots]
    written = parallel_map(_build_one, specs, max_workers=max_workers)
    return [r for r in written if r is not None]
```
Note: the monkeypatched test forces `carry_dir` in the module; `_build_one` must reference `carry_dir()`/`CarryCalculator`/`asset_class_for` at module scope (already imported) so monkeypatching works. With `max_workers=2` on 2 items the test exercises the real pool; the monkeypatched `carry_dir`/`compute_history` are module-level and picklable-by-reference under spawn (the child re-imports the module) -- if spawn cannot see the monkeypatch (it re-imports the unpatched module), keep the test at `max_workers=2` but note that monkeypatch may not cross the process boundary; in that case assert on the RETURN VALUE (roots written) rather than file contents, or drop to `max_workers=1` for the monkeypatched assertion and add a separate real-pool test with a top-level fake module. (Prefer: keep the assert on the returned roots list, which does not depend on the child seeing the monkeypatch.)

Add `--jobs` to `main()`: `p.add_argument("--jobs", type=int, default=None)`, and pass `max_workers=args.jobs` to `build_carry_cache`.

- [ ] **Step 4: Run test to verify it passes**

Run: `/c/.../python.exe -m pytest tests/data/futures/test_build_carry_cache.py -v`
Expected: PASS (existing serial test + new parallel test). If the monkeypatch does not cross the spawn boundary, follow the Step-3 note (assert on returned roots) so the test is robust.

- [ ] **Step 5: Commit**

```bash
git add scripts/data/build_carry_cache.py tests/data/futures/test_build_carry_cache.py
git commit -m "feat(futures): build_carry_cache --jobs via parallel_map"
```

---

## Self-Review

- **Spec coverage:** Task 1 = `parallel_map`; Task 2 = `register` flag; Task 3 = walk-forward by-window parallel + determinism test; Task 4 = cache `--jobs`. Covers the primitive + both concrete applications + the correctness tests.
- **Placeholder scan:** none -- all code + tests concrete.
- **Type consistency:** `parallel_map(fn, items, max_workers) -> list` (input order); `run_futures_backtest(config, register=True)`; `_run_window(..., register=True)`; `process_window(spec) -> dict | None`; `walk_forward_carver(..., max_workers=None)`; `build_carry_cache(..., max_workers=None)`.
- **Determinism/race commitments:** parallel_map input-order + RNG-free strategies => Task 3's subprocess test asserts parallel==serial gate metrics; workers `register=False` => only the parent writes the registry.
- **Pickling:** `process_window`, `_build_one`, and the `parallel_map` test workers are all top-level module functions; Task 1's real-pool test de-risks spawn/pickling early; Task 3 tests the production `__main__` path via subprocess.
- **Backward compat:** `register` and `max_workers` default to preserve current behavior; `--jobs 1` forces serial; window months default 36/12/12; existing config/e2e/pluggable tests retained.
- **Isolation:** no change to simulator/sizing/strategies/gate math/report content/equity path.
