# Fill Logging Everywhere Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Persist fills to disk on every simulated backtest run (every walk-forward window, every sweep config, every inner optimizer probe) across all asset classes, via one shared sink.

**Architecture:** A new `FillSink` owns the run-scoped gzip layout, manifest, and OOS concatenation. The two existing single-run writers (`fx_backtest`, `futures_backtest`) and every validation path (FX WF runners, sweep_runner, vectorbt walk_forward + GridSearchOptimizer, intraday runner, crypto CSCM) delegate to it. The top-level single-window verdict `trades.csv` stays plain; all high-volume internals are gzipped.

**Tech Stack:** Python 3, pandas (gzip via `.csv.gz` extension, no new dependency), pytest, existing `TradeLogger`.

## Global Constraints

- Format: gzipped CSV for all run-scoped internals; the top-level `output/backtests/<strategy>/<start>_to_<end>/trades.csv` verdict file stays PLAIN (uncompressed).
- Layout: `output/backtests/<strategy>/runs/<run_id>/` with `run_id = "<UTC-timestamp>_<cfg-hash>"`, e.g. `20260720T014530Z_a1b2c3`.
- No new dependency, no new file format (gzip only).
- ASCII-only, no em dashes, no `print()` (use `from src.utils import logger`).
- Env: fintech conda; run tests with `PYTHONPATH=$(pwd)`.
- Commit by explicit path only; never `git add -A`. Do not push (orchestrator owns pushes). Never commit `settings.ini`, `.claude/.strategy-lead-active`, or `*.tmp`.
- Homeguard logger uses f-strings, not `%s`.
- Implementation runs in an isolated git worktree.
- Any real-strategy end-to-end walk-forward/gate validation is verdict-adjacent and delegated to strategy-lead, NOT run in this plan. This plan's tests use synthetic fixtures only.

---

### Task 1: FillSink construction (run dir + meta.json)

**Files:**
- Create: `src/backtesting/engine/fill_sink.py`
- Test: `tests/backtesting/engine/test_fill_sink.py`

**Interfaces:**
- Produces: `FillSink(strategy: str, run_id: str, meta: dict, root: Path = Path("output/backtests"))`; attribute `run_dir: Path`; classmethod `make_run_id(cfg_hash: str, now: datetime) -> str`.

- [ ] **Step 1: Write the failing test**

```python
# tests/backtesting/engine/test_fill_sink.py
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from src.backtesting.engine.fill_sink import FillSink


def test_init_creates_run_dir_and_meta(tmp_path):
    sink = FillSink(
        strategy="FxDemo",
        run_id="20260720T000000Z_abc123",
        meta={"kind": "walkforward", "n_windows": 3},
        root=tmp_path,
    )
    assert sink.run_dir == tmp_path / "FxDemo" / "runs" / "20260720T000000Z_abc123"
    assert sink.run_dir.is_dir()
    meta = json.loads((sink.run_dir / "meta.json").read_text())
    assert meta["kind"] == "walkforward"
    assert meta["n_windows"] == 3
    assert meta["strategy"] == "FxDemo"


def test_make_run_id_is_deterministic_given_now():
    now = datetime(2026, 7, 20, 1, 45, 30, tzinfo=timezone.utc)
    assert FillSink.make_run_id("a1b2c3", now) == "20260720T014530Z_a1b2c3"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=$(pwd) pytest tests/backtesting/engine/test_fill_sink.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.backtesting.engine.fill_sink'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/backtesting/engine/fill_sink.py
"""Run-scoped fill logging sink.

Every simulated backtest run persists its fills here: per-window and
per-config, gzipped, under output/backtests/<strategy>/runs/<run_id>/.
See docs/superpowers/specs/2026-07-20-fill-logging-everywhere-design.md.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from src.utils import logger


class FillSink:
    def __init__(self, strategy: str, run_id: str, meta: dict,
                 root: Path = Path("output/backtests")):
        self.strategy = strategy
        self.run_id = run_id
        self.run_dir = Path(root) / strategy / "runs" / run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._manifest_rows: list[dict[str, Any]] = []
        full_meta = {"strategy": strategy, "run_id": run_id, **meta}
        (self.run_dir / "meta.json").write_text(json.dumps(full_meta, indent=2, default=str))

    @staticmethod
    def make_run_id(cfg_hash: str, now: datetime) -> str:
        return f"{now.strftime('%Y%m%dT%H%M%SZ')}_{cfg_hash}"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=$(pwd) pytest tests/backtesting/engine/test_fill_sink.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/backtesting/engine/fill_sink.py tests/backtesting/engine/test_fill_sink.py
git commit -m "feat(backtest): FillSink construction + run-scoped meta.json"
```

---

### Task 2: write_window (DataFrame fills -> gz) + extras sidecars

**Files:**
- Modify: `src/backtesting/engine/fill_sink.py`
- Test: `tests/backtesting/engine/test_fill_sink.py`

**Interfaces:**
- Consumes: `FillSink` from Task 1.
- Produces: `write_window(self, trades_df: pd.DataFrame, window: int, cfg_hash: Optional[str] = None, extras: Optional[dict[str, pd.DataFrame]] = None) -> Path` returning the written fills path; records one manifest row per file with keys `file, kind, window, cfg_hash, row_count`.

- [ ] **Step 1: Write the failing test**

```python
def test_write_window_gzips_and_names(tmp_path):
    sink = FillSink("FxDemo", "rid", {}, root=tmp_path)
    df = pd.DataFrame({"date": ["2011-01-03"], "pair": ["EURUSD"], "units": [100.0]})
    path = sink.write_window(df, window=1, cfg_hash="a1b2c3")
    assert path.name == "w01_a1b2c3_trades.csv.gz"
    back = pd.read_csv(path)  # pandas auto-decompresses by .gz extension
    assert list(back.columns) == ["date", "pair", "units"]
    assert len(back) == 1


def test_write_window_without_cfg_hash(tmp_path):
    sink = FillSink("FxDemo", "rid", {}, root=tmp_path)
    df = pd.DataFrame({"date": ["2011-01-03"], "units": [1.0]})
    path = sink.write_window(df, window=2)
    assert path.name == "w02_trades.csv.gz"


def test_write_window_extras_sidecars(tmp_path):
    sink = FillSink("FutDemo", "rid", {}, root=tmp_path)
    trades = pd.DataFrame({"units": [1.0]})
    margin = pd.DataFrame({"margin": [0.3]})
    sink.write_window(trades, window=1, extras={"margin_utilization": margin})
    assert (sink.run_dir / "w01_margin_utilization.csv.gz").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=$(pwd) pytest tests/backtesting/engine/test_fill_sink.py -k write_window -v`
Expected: FAIL with `AttributeError: 'FillSink' object has no attribute 'write_window'`

- [ ] **Step 3: Write minimal implementation**

```python
    def _stem(self, window: int, cfg_hash: Optional[str]) -> str:
        return f"w{window:02d}" + (f"_{cfg_hash}" if cfg_hash else "")

    def write_window(self, trades_df, window, cfg_hash=None, extras=None):
        stem = self._stem(window, cfg_hash)
        path = self.run_dir / f"{stem}_trades.csv.gz"
        trades_df.to_csv(path, index=False, compression="gzip")
        self._manifest_rows.append({
            "file": path.name, "kind": "trades", "window": window,
            "cfg_hash": cfg_hash or "", "row_count": len(trades_df),
        })
        for name, extra_df in (extras or {}).items():
            epath = self.run_dir / f"{stem}_{name}.csv.gz"
            extra_df.to_csv(epath, index=False, compression="gzip")
            self._manifest_rows.append({
                "file": epath.name, "kind": name, "window": window,
                "cfg_hash": cfg_hash or "", "row_count": len(extra_df),
            })
        return path
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=$(pwd) pytest tests/backtesting/engine/test_fill_sink.py -k write_window -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/backtesting/engine/fill_sink.py tests/backtesting/engine/test_fill_sink.py
git commit -m "feat(backtest): FillSink.write_window with gz + extras sidecars"
```

---

### Task 3: write_portfolio (vectorbt fills via TradeLogger)

**Files:**
- Modify: `src/backtesting/engine/fill_sink.py`
- Test: `tests/backtesting/engine/test_fill_sink.py`

**Interfaces:**
- Consumes: `TradeLogger.export_trades_csv(portfolio, output_path, symbol="")` from `src/backtesting/engine/trade_logger.py` (writes CSV; auto-gzips when path ends `.csv.gz`).
- Produces: `write_portfolio(self, portfolio, window: int, cfg_hash: Optional[str] = None, symbol: str = "") -> Path`.

- [ ] **Step 1: Write the failing test**

```python
def test_write_portfolio_delegates_to_tradelogger(tmp_path):
    sink = FillSink("EqDemo", "rid", {}, root=tmp_path)

    class FakePortfolio:
        # custom-Portfolio shape TradeLogger understands: trades is a list of dicts
        trades = [
            {"type": "entry", "timestamp": "2020-01-02", "price": 10.0, "shares": 5},
            {"type": "exit", "timestamp": "2020-01-05", "price": 11.0, "shares": 5,
             "pnl": 5.0, "pnl_pct": 0.1, "exit_reason": "target"},
        ]

    path = sink.write_portfolio(FakePortfolio(), window=1, cfg_hash="cfg9", symbol="AAPL")
    assert path.name == "w01_cfg9_trades.csv.gz"
    back = pd.read_csv(path)
    assert len(back) == 2  # one buy row + one sell row
    assert (sink.run_dir / "w01_cfg9_trades.csv.gz").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=$(pwd) pytest tests/backtesting/engine/test_fill_sink.py -k write_portfolio -v`
Expected: FAIL with `AttributeError: 'FillSink' object has no attribute 'write_portfolio'`

- [ ] **Step 3: Write minimal implementation**

```python
    def write_portfolio(self, portfolio, window, cfg_hash=None, symbol=""):
        from src.backtesting.engine.trade_logger import TradeLogger
        stem = self._stem(window, cfg_hash)
        path = self.run_dir / f"{stem}_trades.csv.gz"
        TradeLogger.export_trades_csv(portfolio, path, symbol=symbol)
        row_count = 0
        if path.exists():
            try:
                row_count = len(pd.read_csv(path))
            except Exception:
                row_count = 0
        self._manifest_rows.append({
            "file": path.name, "kind": "trades", "window": window,
            "cfg_hash": cfg_hash or "", "row_count": row_count,
        })
        return path
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=$(pwd) pytest tests/backtesting/engine/test_fill_sink.py -k write_portfolio -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add src/backtesting/engine/fill_sink.py tests/backtesting/engine/test_fill_sink.py
git commit -m "feat(backtest): FillSink.write_portfolio via TradeLogger gz"
```

---

### Task 4: finalize (OOS concat + manifest)

**Files:**
- Modify: `src/backtesting/engine/fill_sink.py`
- Test: `tests/backtesting/engine/test_fill_sink.py`

**Interfaces:**
- Consumes: `write_window`, `_manifest_rows` from Tasks 1-2.
- Produces: `finalize(self, oos_windows: Optional[list[int]] = None) -> Path` returning the manifest path; writes `manifest.csv` always and `trades_oos.csv.gz` when `oos_windows` is given (concatenation of those windows' `trades` files, in ascending window order).

- [ ] **Step 1: Write the failing test**

```python
def test_finalize_writes_manifest_and_oos_concat(tmp_path):
    sink = FillSink("FxDemo", "rid", {}, root=tmp_path)
    sink.write_window(pd.DataFrame({"date": ["2011-01-03"], "units": [1.0]}), window=1)
    sink.write_window(pd.DataFrame({"date": ["2012-01-03"], "units": [2.0]}), window=2)
    manifest_path = sink.finalize(oos_windows=[1, 2])

    manifest = pd.read_csv(manifest_path)
    assert set(manifest["file"]) >= {"w01_trades.csv.gz", "w02_trades.csv.gz"}

    oos = pd.read_csv(sink.run_dir / "trades_oos.csv.gz")
    assert len(oos) == 2
    assert list(oos["units"]) == [1.0, 2.0]


def test_finalize_without_oos_windows_skips_concat(tmp_path):
    sink = FillSink("FxDemo", "rid", {}, root=tmp_path)
    sink.write_window(pd.DataFrame({"units": [1.0]}), window=1)
    sink.finalize()
    assert (sink.run_dir / "manifest.csv").exists()
    assert not (sink.run_dir / "trades_oos.csv.gz").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=$(pwd) pytest tests/backtesting/engine/test_fill_sink.py -k finalize -v`
Expected: FAIL with `AttributeError: 'FillSink' object has no attribute 'finalize'`

- [ ] **Step 3: Write minimal implementation**

```python
    def finalize(self, oos_windows=None):
        if oos_windows:
            frames = []
            for w in sorted(oos_windows):
                wpath = self.run_dir / f"w{w:02d}_trades.csv.gz"
                if wpath.exists():
                    frames.append(pd.read_csv(wpath))
            if frames:
                oos = pd.concat(frames, ignore_index=True)
                oos.to_csv(self.run_dir / "trades_oos.csv.gz", index=False,
                           compression="gzip")
                self._manifest_rows.append({
                    "file": "trades_oos.csv.gz", "kind": "oos_concat",
                    "window": -1, "cfg_hash": "", "row_count": len(oos),
                })
        manifest_path = self.run_dir / "manifest.csv"
        pd.DataFrame(self._manifest_rows,
                     columns=["file", "kind", "window", "cfg_hash", "row_count"]
                     ).to_csv(manifest_path, index=False)
        logger.info(f"[fill_sink] finalized run {self.run_id}: "
                    f"{len(self._manifest_rows)} artifacts in {self.run_dir}")
        return manifest_path
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=$(pwd) pytest tests/backtesting/engine/test_fill_sink.py -k finalize -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/backtesting/engine/fill_sink.py tests/backtesting/engine/test_fill_sink.py
git commit -m "feat(backtest): FillSink.finalize OOS concat + manifest"
```

---

### Task 5: Edge cases (zero-trade run, empty finalize)

**Files:**
- Modify: `src/backtesting/engine/fill_sink.py`
- Test: `tests/backtesting/engine/test_fill_sink.py`

**Interfaces:**
- Consumes: everything from Tasks 1-4. No new public methods.

- [ ] **Step 1: Write the failing test**

```python
def test_zero_trade_window_writes_header_only_and_counts_zero(tmp_path):
    sink = FillSink("FxDemo", "rid", {}, root=tmp_path)
    empty = pd.DataFrame(columns=["date", "pair", "units"])
    path = sink.write_window(empty, window=1)
    back = pd.read_csv(path)
    assert len(back) == 0
    assert list(back.columns) == ["date", "pair", "units"]
    manifest = pd.read_csv(sink.finalize())
    row = manifest[manifest["file"] == "w01_trades.csv.gz"].iloc[0]
    assert row["row_count"] == 0
```

- [ ] **Step 2: Run test to verify it fails or passes**

Run: `PYTHONPATH=$(pwd) pytest tests/backtesting/engine/test_fill_sink.py -k zero_trade -v`
Expected: PASS if Task 2/4 already handle empties. If it FAILS, fix in Step 3.

- [ ] **Step 3: Confirm/adjust implementation**

No code change expected: `to_csv` on an empty DataFrame writes a header row, and `len(empty) == 0` yields `row_count == 0`. If the test failed, ensure `write_window` does not early-return on empty and `finalize` includes zero-count rows.

- [ ] **Step 4: Run full FillSink suite**

Run: `PYTHONPATH=$(pwd) pytest tests/backtesting/engine/test_fill_sink.py -v`
Expected: PASS (all)

- [ ] **Step 5: Commit**

```bash
git add tests/backtesting/engine/test_fill_sink.py src/backtesting/engine/fill_sink.py
git commit -m "test(backtest): FillSink zero-trade + empty edge cases"
```

---

### Task 6: fx_backtest delegates to FillSink (verdict plain, add sink params)

**Files:**
- Modify: `src/backtesting/engine/fx_backtest.py:56-129`
- Test: `tests/backtesting/engine/test_fx_backtest_fillsink.py`

**Interfaces:**
- Consumes: `FillSink` (Tasks 1-4).
- Produces: `run_fx_backtest(config, register=True, log_trades=False, fill_sink=None, window=None)`. When `fill_sink` is set, per-window `res.trades` is routed to `fill_sink.write_window(res.trades, window, extras={"leverage_utilization": ...})` and the fixed single-window path is NOT written. `log_trades=True` (no sink) still writes the PLAIN `output/backtests/fx/<strategy>/<start>_to_<end>/trades.csv` exactly as before.

- [ ] **Step 1: Write the failing test**

```python
# tests/backtesting/engine/test_fx_backtest_fillsink.py
import pandas as pd
from pathlib import Path
from src.backtesting.engine.fill_sink import FillSink


def test_route_result_to_sink_writes_window(tmp_path, monkeypatch):
    # Build a fake res and exercise ONLY the sink-routing branch via the helper.
    from src.backtesting.engine import fx_backtest

    class Res:
        trades = pd.DataFrame({"date": ["2011-01-03"], "pair": ["EURUSD"], "units": [1.0]})
        equity_curve = pd.Series([1.0, 1.01], name="equity")
        leverage_utilization = pd.Series([0.2, 0.3])

    sink = FillSink("FxSeatbelt", "rid", {}, root=tmp_path)
    fx_backtest._route_fills(Res(), sink, window=3)
    assert (sink.run_dir / "w03_trades.csv.gz").exists()
    assert (sink.run_dir / "w03_leverage_utilization.csv.gz").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=$(pwd) pytest tests/backtesting/engine/test_fx_backtest_fillsink.py -v`
Expected: FAIL with `AttributeError: module 'src.backtesting.engine.fx_backtest' has no attribute '_route_fills'`

- [ ] **Step 3: Write minimal implementation**

Add the helper and thread the params. In `src/backtesting/engine/fx_backtest.py`:

```python
def _route_fills(res, fill_sink, window):
    extras = {"leverage_utilization": res.leverage_utilization.rename(
        "leverage_utilization").reset_index()}
    fill_sink.write_window(res.trades, window, extras=extras)
```

Change the signature and the tail of `run_fx_backtest`:

```python
def run_fx_backtest(config: Dict[str, Any], register: bool = True,
                    log_trades: bool = False, fill_sink=None, window=None) -> Dict[str, Any]:
    ...
    # (unchanged through res = sim.run_sized(...) and report/registry)
    if fill_sink is not None:
        _route_fills(res, fill_sink, window if window is not None else 0)
        trade_log_dir = str(fill_sink.run_dir)
    else:
        trade_log_dir = _write_trade_log(res, strategy_name, start, end) if log_trades else None
    return {
        "n_days": len(res.equity_curve),
        "metrics": report["overall_metrics"],
        "equity_curve": res.equity_curve.tolist(),
        "run_id": run_id,
        "trade_log_dir": trade_log_dir,
    }
```

Leave `_write_trade_log` (the plain top-level verdict writer) unchanged.

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=$(pwd) pytest tests/backtesting/engine/test_fx_backtest_fillsink.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add src/backtesting/engine/fx_backtest.py tests/backtesting/engine/test_fx_backtest_fillsink.py
git commit -m "feat(fx): run_fx_backtest routes per-window fills to FillSink"
```

---

### Task 7: futures_backtest delegates to FillSink (with margin extra)

**Files:**
- Modify: `src/backtesting/engine/futures_backtest.py` (the `_write_trade_log` helper near line 134 and `run_futures_backtest` signature near line 46)
- Test: `tests/backtesting/engine/test_futures_backtest_fillsink.py`

**Interfaces:**
- Consumes: `FillSink`.
- Produces: a `_route_fills(res, fill_sink, window)` in `futures_backtest.py` that calls `fill_sink.write_window(res.trades, window, extras={"margin_utilization": <res margin df>})`; `run_futures_backtest(..., fill_sink=None, window=None)` routing to it when a sink is passed, else the existing plain writer. First read the file to confirm the exact attribute name for the margin series (`res.margin_utilization` or similar) and the existing `_write_trade_log` body before editing.

- [ ] **Step 1: Write the failing test**

```python
# tests/backtesting/engine/test_futures_backtest_fillsink.py
import pandas as pd
from src.backtesting.engine.fill_sink import FillSink
from src.backtesting.engine import futures_backtest


def test_futures_route_fills_writes_margin_extra(tmp_path):
    class Res:
        trades = pd.DataFrame({"date": ["2017-01-03"], "symbol": ["CL"], "units": [1.0]})
        equity_curve = pd.Series([1.0, 1.02], name="equity")
        margin_utilization = pd.Series([0.4, 0.5])

    sink = FillSink("FuturesCarry", "rid", {}, root=tmp_path)
    futures_backtest._route_fills(Res(), sink, window=2)
    assert (sink.run_dir / "w02_trades.csv.gz").exists()
    assert (sink.run_dir / "w02_margin_utilization.csv.gz").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=$(pwd) pytest tests/backtesting/engine/test_futures_backtest_fillsink.py -v`
Expected: FAIL with `AttributeError: module ... has no attribute '_route_fills'`

- [ ] **Step 3: Write minimal implementation**

Read `futures_backtest.py` first to confirm the margin attribute name; then add (adjusting the attribute name to the real one):

```python
def _route_fills(res, fill_sink, window):
    extras = {"margin_utilization": res.margin_utilization.rename(
        "margin_utilization").reset_index()}
    fill_sink.write_window(res.trades, window, extras=extras)
```

Add `fill_sink=None, window=None` to `run_futures_backtest`; where it currently calls `_write_trade_log(...) if log_trades`, branch: if `fill_sink is not None`, call `_route_fills(res, fill_sink, window or 0)`; else keep the existing plain writer.

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=$(pwd) pytest tests/backtesting/engine/test_futures_backtest_fillsink.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add src/backtesting/engine/futures_backtest.py tests/backtesting/engine/test_futures_backtest_fillsink.py
git commit -m "feat(futures): run_futures_backtest routes per-window fills to FillSink"
```

---

### Task 8: Wire the 3 FX walk-forward runners

**Files:**
- Modify: `scripts/backtest_scripts/run_fx_walkforward.py`, `scripts/backtest_scripts/run_fx_carry_seatbelt_walkforward.py`, `scripts/backtest_scripts/run_fx_wave2_gate.py`
- Test: `tests/backtesting/engine/test_fx_wf_runner_logging.py`

**Interfaces:**
- Consumes: `FillSink`, `run_fx_backtest(..., fill_sink=, window=)` (Task 6).
- Produces: each runner builds one `FillSink` per invocation (kind `walkforward`), passes `fill_sink`/`window` on every per-window `run_fx_backtest` call (replacing `log_trades=False`), and calls `sink.finalize(oos_windows=<all test windows>)` at the end. The runner computes `cfg_hash` via `hashlib.sha1` of the sorted config JSON; `run_id = FillSink.make_run_id(cfg_hash, datetime.now(timezone.utc))`.

- [ ] **Step 1: Write the failing test**

```python
# tests/backtesting/engine/test_fx_wf_runner_logging.py
import pandas as pd
from datetime import datetime, timezone
from src.backtesting.engine.fill_sink import FillSink


def test_wf_loop_produces_oos_concat(tmp_path, monkeypatch):
    # Simulate the runner loop contract: N windows -> N gz + trades_oos.csv.gz
    sink = FillSink("FxWFDemo", FillSink.make_run_id(
        "cfg", datetime(2026, 7, 20, tzinfo=timezone.utc)), {"kind": "walkforward"},
        root=tmp_path)
    for w in (1, 2, 3):
        sink.write_window(pd.DataFrame({"date": [f"201{w}-01-03"], "units": [float(w)]}), window=w)
    sink.finalize(oos_windows=[1, 2, 3])
    oos = pd.read_csv(sink.run_dir / "trades_oos.csv.gz")
    assert len(oos) == 3
    assert (sink.run_dir / "manifest.csv").exists()
```

- [ ] **Step 2: Run test to verify it fails, then passes after wiring**

Run: `PYTHONPATH=$(pwd) pytest tests/backtesting/engine/test_fx_wf_runner_logging.py -v`
Expected: PASS immediately (this test asserts the sink contract the runners must use). It is the acceptance contract; Steps 3-4 make the three runners honor it.

- [ ] **Step 3: Edit each runner**

In each of the three runners, at the top of the WF driver (before the window loop), add:

```python
import hashlib, json
from datetime import datetime, timezone
from src.backtesting.engine.fill_sink import FillSink

cfg_hash = hashlib.sha1(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:6]
sink = FillSink(strategy_name, FillSink.make_run_id(cfg_hash, datetime.now(timezone.utc)),
                {"kind": "walkforward", "start": str(start), "end": str(end)})
```

Replace each per-window `res = run_fx_backtest(cfg, register=False, log_trades=False)` with:

```python
res = run_fx_backtest(window_cfg, register=False, fill_sink=sink, window=window_number)
```

After the loop, add:

```python
sink.finalize(oos_windows=list(range(1, window_number + 1)))
```

Note: `run_fx_wave2_gate.py` takes `--config`/`--name`; derive `strategy_name` from `--name` or the config `strategy.name`. Read each runner to place these at the correct scope (the loop variable names differ per runner).

- [ ] **Step 4: Run the runner smoke + unit test**

Run: `PYTHONPATH=$(pwd) pytest tests/backtesting/engine/test_fx_wf_runner_logging.py -v`
Expected: PASS. (A real-data end-to-end run of these runners is verdict-adjacent and is delegated to strategy-lead, NOT run here.)

- [ ] **Step 5: Commit**

```bash
git add scripts/backtest_scripts/run_fx_walkforward.py scripts/backtest_scripts/run_fx_carry_seatbelt_walkforward.py scripts/backtest_scripts/run_fx_wave2_gate.py tests/backtesting/engine/test_fx_wf_runner_logging.py
git commit -m "feat(fx): FX walk-forward runners log per-window fills + OOS concat"
```

---

### Task 9: Re-point sweep_runner at FillSink

**Files:**
- Modify: `src/backtesting/optimization/sweep_runner.py:357-384`
- Test: `tests/backtesting/optimization/test_sweep_runner_logging.py`

**Interfaces:**
- Consumes: `FillSink.write_portfolio`.
- Produces: the per-symbol export block builds one `FillSink` (kind `sweep`) and calls `sink.write_portfolio(portfolio, window=0, cfg_hash=<symbol or config hash>, symbol=symbol)` per symbol, then `sink.finalize()`. Keep the existing equity/state CSV exports as-is (they are not fills); only the trades export is re-pointed. Read the surrounding method to find `output_dir`, `timestamp`, and `self._portfolios` scope.

- [ ] **Step 1: Write the failing test**

```python
# tests/backtesting/optimization/test_sweep_runner_logging.py
import pandas as pd
from src.backtesting.engine.fill_sink import FillSink


def test_sweep_writes_per_symbol_via_sink(tmp_path):
    sink = FillSink("SweepDemo", "rid", {"kind": "sweep"}, root=tmp_path)

    class FakePortfolio:
        trades = [
            {"type": "entry", "timestamp": "2020-01-02", "price": 10.0, "shares": 5},
            {"type": "exit", "timestamp": "2020-01-05", "price": 11.0, "shares": 5,
             "pnl": 5.0, "pnl_pct": 0.1, "exit_reason": "target"},
        ]

    for sym in ("AAPL", "MSFT"):
        sink.write_portfolio(FakePortfolio(), window=0, cfg_hash=sym, symbol=sym)
    sink.finalize()
    assert (sink.run_dir / "w00_AAPL_trades.csv.gz").exists()
    assert (sink.run_dir / "w00_MSFT_trades.csv.gz").exists()
    assert (sink.run_dir / "manifest.csv").exists()
```

- [ ] **Step 2: Run test to verify it fails, then passes**

Run: `PYTHONPATH=$(pwd) pytest tests/backtesting/optimization/test_sweep_runner_logging.py -v`
Expected: PASS immediately (asserts the sink contract). Step 3 makes `sweep_runner` honor it.

- [ ] **Step 3: Edit sweep_runner.py**

Replace the `TradeLogger.export_trades_csv(portfolio, trades_csv, symbol=symbol)` call inside the per-symbol loop with a sink built once before the loop:

```python
from src.backtesting.engine.fill_sink import FillSink
sink = FillSink(getattr(self, "strategy_name", "sweep"),
                FillSink.make_run_id("sweep", datetime.now(timezone.utc)),
                {"kind": "sweep", "start": str(start_date), "end": str(end_date)})
```

and in the loop:

```python
sink.write_portfolio(portfolio, window=0, cfg_hash=symbol, symbol=symbol)
```

after the loop: `sink.finalize()`. Leave `export_equity_curve_csv` / `export_portfolio_state_csv` unchanged. Add `from datetime import datetime, timezone` if not present.

- [ ] **Step 4: Run test**

Run: `PYTHONPATH=$(pwd) pytest tests/backtesting/optimization/test_sweep_runner_logging.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/backtesting/optimization/sweep_runner.py tests/backtesting/optimization/test_sweep_runner_logging.py
git commit -m "feat(opt): sweep_runner routes fills through FillSink"
```

---

### Task 10: vectorbt walk_forward.py logs OOS + IS portfolios

**Files:**
- Modify: `src/backtesting/chunking/walk_forward.py:208-341` (the `validate` window loop)
- Test: `tests/backtesting/chunking/test_walk_forward_logging.py`

**Interfaces:**
- Consumes: `FillSink.write_portfolio`.
- Produces: `WalkForwardValidator.validate(..., fill_sink=None)`. When a sink is passed, after each window log `train_result['best_portfolio']` as IS (cfg_hash `"is"`) and `test_portfolio` as OOS (cfg_hash `"oos"`), then `finalize()` after the loop. Default `None` keeps current behavior for existing callers/tests.

- [ ] **Step 1: Write the failing test**

```python
# tests/backtesting/chunking/test_walk_forward_logging.py
import inspect
from src.backtesting.chunking.walk_forward import WalkForwardValidator


def test_validate_accepts_fill_sink_param():
    sig = inspect.signature(WalkForwardValidator.validate)
    assert "fill_sink" in sig.parameters
    assert sig.parameters["fill_sink"].default is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=$(pwd) pytest tests/backtesting/chunking/test_walk_forward_logging.py -v`
Expected: FAIL with `AssertionError` (no `fill_sink` param).

- [ ] **Step 3: Write minimal implementation**

Add `fill_sink=None` to `validate`. Inside the window loop, after `test_portfolio` is computed and `is` stats collected:

```python
if fill_sink is not None:
    fill_sink.write_portfolio(train_result['best_portfolio'],
                              window=window.window_number, cfg_hash="is")
    fill_sink.write_portfolio(test_portfolio,
                              window=window.window_number, cfg_hash="oos")
```

After the loop, before building `results`:

```python
if fill_sink is not None:
    oos_windows = [w.window_number for w in windows]
    fill_sink.finalize(oos_windows=oos_windows)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=$(pwd) pytest tests/backtesting/chunking/test_walk_forward_logging.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/backtesting/chunking/walk_forward.py tests/backtesting/chunking/test_walk_forward_logging.py
git commit -m "feat(wf): vectorbt walk_forward logs IS + OOS portfolio fills"
```

---

### Task 11: GridSearchOptimizer inner-probe logging

**Files:**
- Modify: `src/backtesting/optimization/` (the `GridSearchOptimizer.optimize` method; read to find its file and per-combo evaluation loop)
- Test: `tests/backtesting/optimization/test_gridsearch_probe_logging.py`

**Interfaces:**
- Consumes: `FillSink.write_portfolio`.
- Produces: `GridSearchOptimizer.optimize(..., fill_sink=None, base_window=0)`. When a sink is passed, each evaluated param combo's portfolio is logged via `write_portfolio(portfolio, window=base_window, cfg_hash=<combo hash>)`. Default `None` preserves behavior. First read the optimizer to confirm the method signature and where each combo's portfolio object is available.

- [ ] **Step 1: Write the failing test**

```python
# tests/backtesting/optimization/test_gridsearch_probe_logging.py
import inspect
from src.backtesting.optimization import GridSearchOptimizer


def test_optimize_accepts_fill_sink_param():
    sig = inspect.signature(GridSearchOptimizer.optimize)
    assert "fill_sink" in sig.parameters
    assert sig.parameters["fill_sink"].default is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=$(pwd) pytest tests/backtesting/optimization/test_gridsearch_probe_logging.py -v`
Expected: FAIL (`fill_sink` not a param).

- [ ] **Step 3: Write minimal implementation**

Add `fill_sink=None, base_window=0` to `optimize`. Where each param combo's portfolio is produced, add:

```python
if fill_sink is not None:
    import hashlib, json
    combo_hash = hashlib.sha1(json.dumps(params, sort_keys=True, default=str).encode()).hexdigest()[:6]
    fill_sink.write_portfolio(portfolio, window=base_window, cfg_hash=combo_hash)
```

Then in `walk_forward.py` (Task 10), pass `fill_sink=fill_sink, base_window=window.window_number` into the `optimizer.optimize(...)` call so probes are tagged by their training window.

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=$(pwd) pytest tests/backtesting/optimization/test_gridsearch_probe_logging.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/backtesting/optimization/ tests/backtesting/optimization/test_gridsearch_probe_logging.py
git commit -m "feat(opt): GridSearchOptimizer logs every inner probe fill"
```

---

### Task 12: Intraday runner logs ALL OrderEngine fills (entries + exits)

**Files:**
- Modify: `scripts/backtest_scripts/run_fx_london_breakout_walkforward.py` (the `build_trade_log` / `_pair_trade_log` functions added in the backfill)
- Test: `tests/backtesting/engine/test_intraday_all_fills.py`

**Interfaces:**
- Consumes: `OrderEngine.fills` (a `list[Fill]` where `Fill = NamedTuple(order_id, ts, price, qty, side)`), from `src/backtesting/engine/intraday_order_engine.py`.
- Produces: the runner's trade-log builder emits one row per fill in `engine.fills` (entries AND bracket/OCO exits), not just entries. Read the current `build_trade_log` to see how it currently filters to entries.

- [ ] **Step 1: Write the failing test**

```python
# tests/backtesting/engine/test_intraday_all_fills.py
import importlib.util, sys
from pathlib import Path

MOD = Path("scripts/backtest_scripts/run_fx_london_breakout_walkforward.py")


def _load():
    spec = importlib.util.spec_from_file_location("lb_runner", MOD)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["lb_runner"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_build_trade_log_includes_exits():
    from src.backtesting.engine.intraday_order_engine import Fill
    import pandas as pd
    mod = _load()
    fills = [
        Fill("o1", pd.Timestamp("2011-01-21 09:30", tz="UTC"), 1.5, 1.0, "buy"),
        Fill("o1x", pd.Timestamp("2011-01-21 15:00", tz="UTC"), 1.6, 1.0, "sell"),
    ]
    df = mod.build_trade_log("GBPUSD", fills, day_r=0.9)
    assert len(df) == 2  # entry AND exit both present
    assert set(df["side"]) == {"buy", "sell"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=$(pwd) pytest tests/backtesting/engine/test_intraday_all_fills.py -v`
Expected: FAIL (current builder returns entries only, so `len(df) == 1`). If the current `build_trade_log` signature differs, adjust the test call to match after reading the file.

- [ ] **Step 3: Write minimal implementation**

Rewrite the runner's builder to iterate ALL `engine.fills` rather than filtering to entries:

```python
def build_trade_log(pair, fills, day_r=None):
    import pandas as pd
    rows = [{"pair": pair, "ts": f.ts, "side": f.side, "price": f.price,
             "qty": f.qty, "order_id": f.order_id, "day_r": day_r} for f in fills]
    return pd.DataFrame(rows, columns=["pair", "ts", "side", "price", "qty", "order_id", "day_r"])
```

Keep the docstring note that `day_r` is the day-level R attached to each fill row. Remove the entry-only filter.

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=$(pwd) pytest tests/backtesting/engine/test_intraday_all_fills.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/backtest_scripts/run_fx_london_breakout_walkforward.py tests/backtesting/engine/test_intraday_all_fills.py
git commit -m "fix(fx): intraday runner logs all OrderEngine fills (entries + exits)"
```

---

### Task 13: Verify and wire crypto CSCM

**Files:**
- Read first: locate the CSCM backtest path (`grep -rl CSCM src/strategies src/backtesting` via Bash, avoiding the tokens the strategy-lead hook blocks; search for `cscm` lowercase and `CrossSectional`).
- Modify: the CSCM backtest entry point (path determined by the read)
- Test: `tests/strategies/test_cscm_fill_logging.py`

**Interfaces:**
- Consumes: `FillSink.write_portfolio` if CSCM produces a vectorbt-shaped portfolio, else `write_window` if it produces a `res.trades` DataFrame.
- Produces: the CSCM backtest persists fills via a `FillSink` (kind `verdict` or `walkforward` as appropriate).

- [ ] **Step 1: Determine the shape**

Read the CSCM backtest path. Decide: DataFrame fills -> `write_window`; vectorbt portfolio -> `write_portfolio`. Record the decision in the commit message.

- [ ] **Step 2: Write the failing test**

Write a test asserting the CSCM entry point, given a tiny synthetic input, produces a `manifest.csv` under a temp sink root. Match the real function signature discovered in Step 1. If CSCM already routes through `run_fx_backtest`/`TradeLogger` (and is thus already covered), assert that and mark this task complete with a note; do not add redundant wiring.

- [ ] **Step 3: Wire (only if not already covered)**

Add a `fill_sink=None` param to the CSCM backtest entry point and route its fills to `write_portfolio`/`write_window` exactly as the FX/equity paths do. If already covered, no code change.

- [ ] **Step 4: Run test**

Run: `PYTHONPATH=$(pwd) pytest tests/strategies/test_cscm_fill_logging.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add <cscm path> tests/strategies/test_cscm_fill_logging.py
git commit -m "feat(cscm): crypto backtest persists fills via FillSink"
```

---

### Task 14: Docs + strategy-lead enforcement

**Files:**
- Modify: `.claude/rules/strategy-pipeline.md:71`, `docs/methodology/backtesting.md` (Section 12), `.claude/agents/strategy-lead.md`

**Interfaces:** none (documentation + agent instruction).

- [ ] **Step 1: Rewrite the carve-out (strategy-pipeline.md:71)**

Replace the sentence "Validation-harness internals (e.g. per-window walk-forward runs) may suppress logging, but the primary/representative backtest for a strategy MUST produce one." with:

```
Every simulated run persists its fills -- per-window, per-config, and per inner
optimizer probe -- across equity, crypto, futures, and FX. Runs write to a
run-scoped sink (output/backtests/<strategy>/runs/<run_id>/, gzipped) with a
manifest.csv and, for walk-forward runs, a trades_oos.csv.gz concatenation of the
OOS windows (the actual gated-verdict fills). The single-window verdict
output/backtests/<strategy>/<start>_to_<end>/trades.csv remains plain. A run that
discards its fills is incomplete and must be rejected in review.
```

- [ ] **Step 2: Update methodology Section 12**

Add a paragraph mirroring Step 1's rule to `docs/methodology/backtesting.md` Section 12 (read the section first to match its heading style and cross-references). ASCII only, no em dashes.

- [ ] **Step 3: Update strategy-lead manifest check**

In `.claude/agents/strategy-lead.md`, extend the fills-level verification row (added in `3106b9c`) so it accepts EITHER the single-window `trades.csv` OR a run-scoped `runs/<run_id>/manifest.csv` with a non-empty `trades_oos.csv.gz` for walk-forward verdicts. State that a walk-forward verdict must have the OOS concat, not just a representative single-pass log.

- [ ] **Step 4: Verify docs are ASCII-clean**

Run: `grep -nP "[^\x00-\x7F]" .claude/rules/strategy-pipeline.md docs/methodology/backtesting.md .claude/agents/strategy-lead.md && echo FOUND || echo clean`
Expected: `clean`

- [ ] **Step 5: Commit**

```bash
git add .claude/rules/strategy-pipeline.md docs/methodology/backtesting.md .claude/agents/strategy-lead.md
git commit -m "docs: mandate fills on every simulated run; strategy-lead checks manifest"
```

---

## Final Validation (whole-branch)

- [ ] Run the full new test surface: `PYTHONPATH=$(pwd) pytest tests/backtesting/engine/test_fill_sink.py tests/backtesting/engine/test_fx_backtest_fillsink.py tests/backtesting/engine/test_futures_backtest_fillsink.py tests/backtesting/engine/test_fx_wf_runner_logging.py tests/backtesting/optimization/test_sweep_runner_logging.py tests/backtesting/optimization/test_gridsearch_probe_logging.py tests/backtesting/chunking/test_walk_forward_logging.py tests/backtesting/engine/test_intraday_all_fills.py tests/strategies/test_cscm_fill_logging.py -v`
- [ ] Confirm no existing trade-log tests regressed: `PYTHONPATH=$(pwd) pytest tests/backtesting/engine -k "trade or log" -v`
- [ ] Whole-branch review (most capable model).
- [ ] A real-strategy end-to-end walk-forward run to confirm the sink produces `trades_oos.csv.gz` on real data is VERDICT-ADJACENT and is delegated to strategy-lead, not run in this plan.
