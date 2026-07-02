# Futures Carry Strategy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an absolute (Carver-style) futures carry strategy that plugs into the pluggable runner via `forecast_panel`, backed by a precomputed carry cache, and produce a gate-checked broad-basket walk-forward result.

**Architecture:** `forecast = clip(EWMA(carry)/annualized_price_vol * carry_scalar, -cap, cap)` per instrument, sourced from `CarryCalculator` via a per-root parquet cache. Reuses the pluggable runner (B) and the config-driven walk-forward; generalizes the walk-forward to pass strategy name/params (the piece deferred from B).

**Tech Stack:** Python 3.13, pandas, numpy, polars, pytest. Conda env `fintech`.

## Global Constraints

- **Python execution:** ALWAYS `/c/Users/qwqw1/anaconda3/envs/fintech/python.exe -m pytest <args>`. `PYTHONPATH=.` for scripts importing `scripts/`. Never system Python.
- **ASCII only** (Windows cp1252). No `print()` (use `src.utils.logger`, f-strings).
- **Base branch:** `feat/futures-carry-strategy` (already checked out, off `main` @ eec3ea9). Do NOT switch.
- **Parameter-free discipline:** `carry_scalar` (default 30.0) and `ewma_span` (default 10) are FIXED doctrine constants -- never optimized. Walk-forward trial_count stays 1.
- **Isolation:** do NOT change the simulator, sizing, loader, Carver strategy, the equity/crypto path, or `CarryCalculator` itself. Reuse `CarryCalculator` as-is.
- **Contract:** `FuturesCarryStrategy` constructs as `__init__(self, universe, **params)` and exposes `forecast_panel(close_panel: pd.DataFrame) -> pd.DataFrame` (the `SupportsForecastPanel` contract the runner validates).
- **Universe (33 roots), verbatim** (same as the broad basket): ES, NQ, YM, ZT, ZF, ZN, TN, ZB, UB, 6E, 6J, 6B, 6A, 6C, 6S, 6M, 6N, CL, BZ, NG, HO, RB, GC, SI, HG, PL, ZC, ZW, ZS, ZL, ZM, LE, HE.

---

## Task 1: `carry_dir()` path helper

**Files:**
- Modify: `src/data/futures/paths.py`
- Test: `tests/data/futures/test_carry_dir.py`

**Interfaces:**
- Produces: `carry_dir() -> Path` == `_futures_root() / "carry"`. Consumed by Tasks 3, 4.

- [ ] **Step 1: Write the failing test**

```python
# tests/data/futures/test_carry_dir.py
from src.data.futures.paths import carry_dir, roll_calendar_dir


def test_carry_dir_sibling_of_roll_calendar():
    assert carry_dir().name == "carry"
    assert carry_dir().parent == roll_calendar_dir().parent
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/c/Users/qwqw1/anaconda3/envs/fintech/python.exe -m pytest tests/data/futures/test_carry_dir.py -v`
Expected: FAIL — `ImportError: cannot import name 'carry_dir'`.

- [ ] **Step 3: Add `carry_dir`**

In `src/data/futures/paths.py`, mirror `roll_calendar_dir` (which is `return _futures_root() / "roll_calendar"`):

```python
def carry_dir() -> Path:
    return _futures_root() / "carry"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/c/Users/qwqw1/anaconda3/envs/fintech/python.exe -m pytest tests/data/futures/test_carry_dir.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/data/futures/paths.py tests/data/futures/test_carry_dir.py
git commit -m "feat(futures): add carry_dir() path helper"
```

---

## Task 2: Root -> asset_class map

**Files:**
- Create: `src/data/futures/asset_class.py`
- Test: `tests/data/futures/test_asset_class.py`

**Interfaces:**
- Produces: `ASSET_CLASS: dict[str, str]` and `asset_class_for(root: str) -> str` (raises `KeyError` on an unmapped root). Values in {`"equity_index"`, `"fx"`, `"bond"`, `"commodity"`} -- exactly the strings `CarryCalculator.compute` accepts. Consumed by Tasks 3 (builder) and used indirectly by Task 7 (universe).

- [ ] **Step 1: Write the failing test**

```python
# tests/data/futures/test_asset_class.py
import pytest
from src.data.futures.asset_class import ASSET_CLASS, asset_class_for

BROAD = ["ES", "NQ", "YM", "ZT", "ZF", "ZN", "TN", "ZB", "UB",
         "6E", "6J", "6B", "6A", "6C", "6S", "6M", "6N",
         "CL", "BZ", "NG", "HO", "RB", "GC", "SI", "HG", "PL",
         "ZC", "ZW", "ZS", "ZL", "ZM", "LE", "HE"]
VALID = {"equity_index", "fx", "bond", "commodity"}


def test_every_broad_root_mapped_to_valid_class():
    for r in BROAD:
        assert asset_class_for(r) in VALID


def test_spot_check_classes():
    assert asset_class_for("ES") == "equity_index"
    assert asset_class_for("6E") == "fx"
    assert asset_class_for("ZN") == "bond"
    assert asset_class_for("CL") == "commodity"
    assert asset_class_for("GC") == "commodity"


def test_unmapped_root_raises():
    with pytest.raises(KeyError):
        asset_class_for("NOPE")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/c/Users/qwqw1/anaconda3/envs/fintech/python.exe -m pytest tests/data/futures/test_asset_class.py -v`
Expected: FAIL — module does not exist.

- [ ] **Step 3: Create the map**

```python
# src/data/futures/asset_class.py
"""Root -> asset_class map for carry (the strings CarryCalculator.compute accepts)."""

ASSET_CLASS: dict[str, str] = {
    # equity_index
    "ES": "equity_index", "NQ": "equity_index", "YM": "equity_index", "RTY": "equity_index",
    "M2K": "equity_index", "MES": "equity_index", "MNQ": "equity_index", "MYM": "equity_index",
    # fx
    "6A": "fx", "6B": "fx", "6C": "fx", "6E": "fx", "6J": "fx", "6M": "fx", "6N": "fx", "6S": "fx",
    # bond
    "ZT": "bond", "ZF": "bond", "ZN": "bond", "TN": "bond", "ZB": "bond", "UB": "bond",
    "10Y": "bond", "2YY": "bond", "5YY": "bond", "30Y": "bond", "SR1": "bond", "SR3": "bond",
    # commodity
    "CL": "commodity", "BZ": "commodity", "NG": "commodity", "HO": "commodity", "RB": "commodity",
    "MCL": "commodity", "MNG": "commodity", "GC": "commodity", "SI": "commodity", "HG": "commodity",
    "PL": "commodity", "MGC": "commodity", "SIL": "commodity", "MET": "commodity",
    "ZC": "commodity", "ZW": "commodity", "ZS": "commodity", "ZL": "commodity", "ZM": "commodity",
    "KE": "commodity", "LE": "commodity", "HE": "commodity",
}


def asset_class_for(root: str) -> str:
    """Return the carry asset_class for `root`; raise KeyError if unmapped."""
    return ASSET_CLASS[root]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/c/Users/qwqw1/anaconda3/envs/fintech/python.exe -m pytest tests/data/futures/test_asset_class.py -v`
Expected: PASS (all three).

- [ ] **Step 5: Commit**

```bash
git add src/data/futures/asset_class.py tests/data/futures/test_asset_class.py
git commit -m "feat(futures): root->asset_class map for carry"
```

---

## Task 3: Carry cache builder

**Files:**
- Create: `scripts/data/build_carry_cache.py`
- Test: `tests/data/futures/test_build_carry_cache.py`

**Interfaces:**
- Consumes: `asset_class_for` (Task 2), `carry_dir` (Task 1), `CarryCalculator.compute_history`.
- Produces: `build_carry_cache(roots: list[str], start: date, end: date) -> list[str]` (returns roots written); writes `carry_dir()/{root}.parquet` `[date, carry]`. A `main()` with argparse `--roots --start --end`, mirroring `build_roll_calendar.py`.

- [ ] **Step 1: Write the failing test**

```python
# tests/data/futures/test_build_carry_cache.py
from datetime import date
import polars as pl


def test_build_carry_cache_writes_parquet(tmp_path, monkeypatch):
    import scripts.data.build_carry_cache as bcc

    monkeypatch.setattr(bcc, "carry_dir", lambda: tmp_path)

    def fake_compute_history(self, root, asset_class, start, end):
        return pl.DataFrame({"date": [date(2020, 1, 2), date(2020, 1, 3)],
                             "carry": [0.05, 0.06]})
    monkeypatch.setattr(bcc.CarryCalculator, "compute_history", fake_compute_history)

    written = bcc.build_carry_cache(["GC"], date(2020, 1, 1), date(2020, 1, 31))
    assert written == ["GC"]
    out = tmp_path / "GC.parquet"
    assert out.exists()
    df = pl.read_parquet(out)
    assert df.columns == ["date", "carry"]
    assert df.height == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/c/Users/qwqw1/anaconda3/envs/fintech/python.exe -m pytest tests/data/futures/test_build_carry_cache.py -v`
Expected: FAIL — module does not exist.

- [ ] **Step 3: Create the builder** (mirror `build_roll_calendar.py`'s argparse `main`)

```python
# scripts/data/build_carry_cache.py
"""Precompute per-root carry series to futures/carry/{root}.parquet.

For each root: CarryCalculator.compute_history(root, asset_class_for(root), start, end)
-> carry_dir()/{root}.parquet [date, carry]. Mirrors build_roll_calendar.py.

Usage:
    python scripts/data/build_carry_cache.py --roots GC CL ES --start 2010-06-07 --end 2026-02-20
"""
from __future__ import annotations

import argparse
from datetime import date, datetime

from src.data.carry_calculator import CarryCalculator
from src.data.futures.asset_class import asset_class_for
from src.data.futures.paths import carry_dir
from src.utils.logger import get_logger

logger = get_logger(__name__)


def build_carry_cache(roots: list[str], start: date, end: date) -> list[str]:
    out_dir = carry_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    calc = CarryCalculator()
    written: list[str] = []
    for root in roots:
        ac = asset_class_for(root)
        hist = calc.compute_history(root, ac, start, end)
        if hist.height == 0:
            logger.warning(f"[build_carry_cache] {root}: no carry rows, skipping")
            continue
        hist.write_parquet(out_dir / f"{root}.parquet")
        written.append(root)
        logger.info(f"[build_carry_cache] {root} ({ac}): {hist.height} rows")
    return written


def _as_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def main() -> None:
    p = argparse.ArgumentParser(description="Build per-root carry cache")
    p.add_argument("--roots", nargs="+", required=True)
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    args = p.parse_args()
    written = build_carry_cache(args.roots, _as_date(args.start), _as_date(args.end))
    logger.info(f"[build_carry_cache] wrote {len(written)}/{len(args.roots)} roots to {carry_dir()}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/c/Users/qwqw1/anaconda3/envs/fintech/python.exe -m pytest tests/data/futures/test_build_carry_cache.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/data/build_carry_cache.py tests/data/futures/test_build_carry_cache.py
git commit -m "feat(futures): carry cache builder (per-root compute_history -> parquet)"
```

---

## Task 4: FuturesCarryStrategy + registry entry

**Files:**
- Create: `src/strategies/advanced/futures_carry_strategy.py`
- Modify: `src/strategies/registry.py`
- Test: `tests/strategies/test_futures_carry_strategy.py`

**Interfaces:**
- Consumes: `carry_dir` (Task 1), `close_to_close_rv`.
- Produces: `FuturesCarryStrategy(universe, carry_scalar=30.0, ewma_span=10, cap=20.0, **params)` with `forecast_panel(close_panel) -> pd.DataFrame`; registry name `"FuturesCarry"` (+ aliases `"Carry"`, `"Futures Carry"`).

- [ ] **Step 1: Write the failing tests**

```python
# tests/strategies/test_futures_carry_strategy.py
import numpy as np
import pandas as pd
import pytest

from src.strategies.advanced.futures_carry_strategy import FuturesCarryStrategy
from src.strategies.registry import get_strategy_class


def _close(n=60):
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    # gently trending prices so vol is finite and non-zero
    return pd.DataFrame({"GC": np.linspace(1800, 1900, n), "CL": np.linspace(60, 70, n)}, index=idx)


def test_registered():
    assert get_strategy_class("FuturesCarry") is FuturesCarryStrategy
    assert get_strategy_class("Carry") is FuturesCarryStrategy


def test_forecast_shape_and_cap(monkeypatch):
    close = _close()
    # constant carry 0.05 for every root
    monkeypatch.setattr(FuturesCarryStrategy, "_load_carry",
                        lambda self, root: pd.Series(0.05, index=close.index))
    strat = FuturesCarryStrategy(["GC", "CL"])
    fc = strat.forecast_panel(close)
    assert list(fc.columns) == ["GC", "CL"]
    assert fc.index.equals(close.index)
    valid = fc.dropna()
    assert ((valid >= -20.0) & (valid <= 20.0)).all().all()


def test_missing_cache_gives_nan_column(monkeypatch):
    close = _close()
    monkeypatch.setattr(FuturesCarryStrategy, "_load_carry", lambda self, root: None)
    strat = FuturesCarryStrategy(["GC", "CL"])
    fc = strat.forecast_panel(close)
    assert fc["GC"].isna().all() and fc["CL"].isna().all()


def test_forecast_sign_follows_carry(monkeypatch):
    close = _close()
    monkeypatch.setattr(FuturesCarryStrategy, "_load_carry",
                        lambda self, root: pd.Series(0.05, index=close.index))
    fc = FuturesCarryStrategy(["GC"]).forecast_panel(close).dropna()
    assert (fc["GC"] > 0).all()  # positive carry -> positive (long) forecast
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/c/Users/qwqw1/anaconda3/envs/fintech/python.exe -m pytest tests/strategies/test_futures_carry_strategy.py -v`
Expected: FAIL — module/registry entry missing.

- [ ] **Step 3: Write the strategy** (plain class implementing the pluggable contract)

```python
# src/strategies/advanced/futures_carry_strategy.py
"""Absolute (Carver-style) futures carry strategy.

forecast = clip(EWMA(carry) / annualized_price_vol * carry_scalar, -cap, cap),
per instrument, sourced from the per-root carry cache (build_carry_cache.py).
carry_scalar and ewma_span are FIXED doctrine constants -- never optimized.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl

from src.data.futures.paths import carry_dir
from src.features.volatility import close_to_close_rv

_SQRT252 = np.sqrt(252.0)


class FuturesCarryStrategy:
    def __init__(self, universe, carry_scalar: float = 30.0, ewma_span: int = 10,
                 cap: float = 20.0, **params):
        self.universe = list(universe)
        self.carry_scalar = float(carry_scalar)
        self.ewma_span = int(ewma_span)
        self.cap = float(cap)

    def _load_carry(self, root: str):
        fp = carry_dir() / f"{root}.parquet"
        if not fp.exists():
            return None
        df = pl.read_parquet(fp).to_pandas()
        return pd.Series(df["carry"].to_numpy(),
                         index=pd.to_datetime(df["date"]))

    def _forecast_root(self, close: pd.Series, carry: pd.Series) -> pd.Series:
        carry = carry.reindex(close.index).ffill()
        carry_sm = carry.ewm(span=self.ewma_span, adjust=False).mean()
        rets = close.pct_change(fill_method=None)
        ann_vol = close_to_close_rv(rets, 25, annualization_factor=1) * _SQRT252
        fc = (carry_sm / ann_vol) * self.carry_scalar
        return fc.clip(-self.cap, self.cap)

    def forecast_panel(self, close_panel: pd.DataFrame) -> pd.DataFrame:
        out: dict[str, pd.Series] = {}
        for root in self.universe:
            if root not in close_panel.columns:
                out[root] = pd.Series(np.nan, index=close_panel.index)
                continue
            carry = self._load_carry(root)
            if carry is None:
                out[root] = pd.Series(np.nan, index=close_panel.index)
                continue
            out[root] = self._forecast_root(close_panel[root], carry)
        return pd.DataFrame(out).reindex(columns=self.universe)
```

- [ ] **Step 4: Register in `src/strategies/registry.py`**

Add to `_STRATEGY_REGISTRY`:
```python
    "FuturesCarry": ("src.strategies.advanced.futures_carry_strategy", "FuturesCarryStrategy"),
```
Add to `_DISPLAY_NAME_MAP`:
```python
    "Carry": "FuturesCarry",
    "Futures Carry": "FuturesCarry",
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `/c/Users/qwqw1/anaconda3/envs/fintech/python.exe -m pytest tests/strategies/test_futures_carry_strategy.py -v`
Expected: PASS (all four).

- [ ] **Step 6: Commit**

```bash
git add src/strategies/advanced/futures_carry_strategy.py src/strategies/registry.py tests/strategies/test_futures_carry_strategy.py
git commit -m "feat(futures): FuturesCarryStrategy (absolute carry forecast) + registry entry"
```

---

## Task 5: Generalize the walk-forward to any strategy

**Files:**
- Modify: `scripts/backtest_scripts/run_carver_walkforward.py`
- Test: `tests/backtesting/test_carver_walkforward_config.py` (append)

**Interfaces:**
- Consumes: config `strategy.name`/`params`.
- Produces: `_config_to_kwargs` gains `strategy_name`/`strategy_params`; `walk_forward_carver` + `_run_window` accept and pass them; the readiness-report title is parametrized by strategy name. Default (`no name`) stays `"CarverMomentum"`.

**Context (verified):** `_config_to_kwargs` returns `{universe, capital, vol_target, start, end}`; `_run_window(universe, train_start, test_end, capital, vol_target, cost_mult)` builds `config = {"strategy": {"universe": ...}, "dates": {...}, "backtest": {...}}`; `walk_forward_carver(train, test, step, start, end, universe=None, capital=..., vol_target=...)`; `_write_readiness_report(result, ...)` title is `# Carver TSMOM Walk-Forward Readiness Report`; `main()` reads `--config`/`--report`.

- [ ] **Step 1: Write the failing tests (append)**

```python
# append to tests/backtesting/test_carver_walkforward_config.py
def test_config_to_kwargs_reads_strategy_name_and_params():
    cfg = {"strategy": {"name": "FuturesCarry", "universe": ["ES"],
                        "params": {"carry_scalar": 25.0}},
           "dates": {"start": "2010-06-07", "end": "2026-02-20"},
           "backtest": {"initial_capital": 10_000_000, "vol_target_per_instrument": 0.20}}
    kw = wf._config_to_kwargs(cfg)
    assert kw["strategy_name"] == "FuturesCarry"
    assert kw["strategy_params"] == {"carry_scalar": 25.0}


def test_config_to_kwargs_defaults_strategy_name():
    cfg = {"strategy": {"universe": ["ES"]}, "dates": {"start": "2010-06-07", "end": "2026-02-20"},
           "backtest": {}}
    kw = wf._config_to_kwargs(cfg)
    assert kw["strategy_name"] == "CarverMomentum"
    assert kw["strategy_params"] == {}


def test_report_title_reflects_strategy(tmp_path):
    result = {
        "oos_sharpe": 0.3, "psr": 1.0, "dsr": 1.0, "pbo": 0.25,
        "oos_sharpe_1_5x_cost": 0.2, "n_windows": 2, "n_oos_days": 500,
        "window_sharpes": [0.3, 0.4], "trial_count": 1, "skew": -0.2, "kurtosis_pearson": 5.0,
        "universe": ["ES"], "window_universes": [["ES"], ["ES"]],
        "window_start": __import__("datetime").date(2013, 6, 7),
        "window_end": __import__("datetime").date(2026, 2, 20),
        "capital": 10_000_000, "vol_target": 0.20, "strategy_name": "FuturesCarry",
    }
    out = tmp_path / "CARRY.md"
    wf._write_readiness_report(result, 36, 12, 12, "2010-06-07", "2026-02-20",
                               report_path=str(out))
    assert "FuturesCarry" in out.read_text()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/c/Users/qwqw1/anaconda3/envs/fintech/python.exe -m pytest tests/backtesting/test_carver_walkforward_config.py -v -k "strategy or title"`
Expected: FAIL — no `strategy_name` key; title not parametrized.

- [ ] **Step 3: Extend `_config_to_kwargs`**

Add two keys to the returned dict:
```python
        "strategy_name": strat.get("name", "CarverMomentum"),
        "strategy_params": strat.get("params", {}),
```

- [ ] **Step 4: Thread name/params through `_run_window` and `walk_forward_carver`**

`_run_window` signature + config:
```python
def _run_window(universe, train_start, test_end, capital, vol_target, cost_mult,
                 strategy_name="CarverMomentum", strategy_params=None):
    config = {
        "strategy": {"name": strategy_name, "universe": list(universe),
                     "params": strategy_params or {}},
        "dates": {"start": str(train_start), "end": str(test_end)},
        "backtest": {"initial_capital": capital, "vol_target_per_instrument": vol_target,
                     "rebalance": "weekly", "cost_mult": cost_mult},
    }
    return run_futures_backtest(config)
```
`walk_forward_carver`: add params `strategy_name="CarverMomentum", strategy_params=None` and forward them to BOTH `_run_window` calls (the 1x and 1.5x cost runs). Add `"strategy_name": strategy_name` to the returned `result` dict.

- [ ] **Step 5: Parametrize the report title**

In `_write_readiness_report`, change the title line from the hardcoded `# Carver TSMOM Walk-Forward Readiness Report` to use the strategy name:
```python
    title = result.get("strategy_name", "CarverMomentum")
```
and start the report content with:
```python
    content = f"""# {title} Walk-Forward Readiness Report
```

- [ ] **Step 6: Wire `main()`**

In `main()`, pass the new kwargs. For the `--config` branch use `kw["strategy_name"]`/`kw["strategy_params"]`; for the no-config default branch use `"CarverMomentum"`/`{}`:
```python
    result = walk_forward_carver(
        train_months=36, test_months=12, step_months=12,
        start=kw["start"], end=kw["end"], universe=kw["universe"],
        capital=kw["capital"], vol_target=kw["vol_target"],
        strategy_name=kw.get("strategy_name", "CarverMomentum"),
        strategy_params=kw.get("strategy_params", {}),
    )
```
(Ensure the no-config `kw` dict built in `main()` also carries `strategy_name`/`strategy_params` defaults, mirroring the `--config` path.)

- [ ] **Step 7: Run tests + confirm Carver default unchanged**

Run: `/c/Users/qwqw1/anaconda3/envs/fintech/python.exe -m pytest tests/backtesting/test_carver_walkforward_config.py -v`
Expected: PASS (existing + 3 new). The pre-existing `test_report_interpolates_actual_capital_and_count` still passes (default title is `CarverMomentum`, capital interpolation unchanged).

- [ ] **Step 8: Commit**

```bash
git add scripts/backtest_scripts/run_carver_walkforward.py tests/backtesting/test_carver_walkforward_config.py
git commit -m "feat(futures): thread strategy name/params through the walk-forward (any strategy)"
```

---

## Task 6: Broad-basket carry config

**Files:**
- Create: `config/backtesting/carry_broad.yaml`
- Test: `tests/backtesting/config/test_carry_broad_config.py`

**Interfaces:** consumed by `src.backtest_runner` and `run_carver_walkforward.py --config`.

- [ ] **Step 1: Write the failing test**

```python
# tests/backtesting/config/test_carry_broad_config.py
from pathlib import Path
import yaml
from src.data.futures.asset_class import asset_class_for

CONFIG = Path("config/backtesting/carry_broad.yaml")


def test_carry_broad_config():
    cfg = yaml.safe_load(CONFIG.read_text())
    assert cfg["asset_class"] == "futures"
    assert cfg["strategy"]["name"] == "FuturesCarry"
    u = cfg["strategy"]["universe"]
    assert len(u) == 33
    for r in u:
        assert asset_class_for(r)  # every root is carry-mappable
    assert cfg["backtest"]["initial_capital"] == 10_000_000
    assert cfg["dates"]["start"] == "2010-06-07"
    assert cfg["dates"]["end"] == "2026-02-20"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/c/Users/qwqw1/anaconda3/envs/fintech/python.exe -m pytest tests/backtesting/config/test_carry_broad_config.py -v`
Expected: FAIL — config missing.

- [ ] **Step 3: Create the config** (copy `carver_tsmom_broad.yaml`'s universe block; add `name: FuturesCarry`)

```yaml
# config/backtesting/carry_broad.yaml
# Absolute (Carver-style) futures carry over the 33-root broad basket.
# Requires the carry cache: python scripts/data/build_carry_cache.py --roots <...> --start 2010-06-07 --end 2026-02-20
# Walk-forward: python scripts/backtest_scripts/run_carver_walkforward.py \
#   --config config/backtesting/carry_broad.yaml --report docs/reports/futures/CARRY_BROAD_READINESS.md
asset_class: futures

strategy:
  name: FuturesCarry
  universe:
    - ES
    - NQ
    - YM
    - ZT
    - ZF
    - ZN
    - TN
    - ZB
    - UB
    - 6E
    - 6J
    - 6B
    - 6A
    - 6C
    - 6S
    - 6M
    - 6N
    - CL
    - BZ
    - NG
    - HO
    - RB
    - GC
    - SI
    - HG
    - PL
    - ZC
    - ZW
    - ZS
    - ZL
    - ZM
    - LE
    - HE

dates:
  start: "2010-06-07"
  end: "2026-02-20"

backtest:
  initial_capital: 10000000
  vol_target_per_instrument: 0.20
  rebalance: weekly
  cost_mult: 1.0
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/c/Users/qwqw1/anaconda3/envs/fintech/python.exe -m pytest tests/backtesting/config/test_carry_broad_config.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add config/backtesting/carry_broad.yaml tests/backtesting/config/test_carry_broad_config.py
git commit -m "feat(futures): broad-basket carry config (FuturesCarry, 33 roots, 10M)"
```

---

## Task 7: Execution and Acceptance (CONTROLLER-run, not a TDD/subagent task)

Controller-run after Tasks 1-6 are complete and committed. Builds the cache, runs the walk-forward, records the verdict. Multi-hour possible for the cache build (per-day per-contract reads over 33 roots x ~15.7y).

- [ ] **Step 1: Build the carry cache for the 33 roots** (background)

```bash
cd "C:/Users/qwqw1/Dropbox/cs/github/Homeguard"
PYTHONPATH=. /c/Users/qwqw1/anaconda3/envs/fintech/python.exe scripts/data/build_carry_cache.py \
  --roots ES NQ YM ZT ZF ZN TN ZB UB 6E 6J 6B 6A 6C 6S 6M 6N CL BZ NG HO RB GC SI HG PL ZC ZW ZS ZL ZM LE HE \
  --start 2010-06-07 --end 2026-02-20 > .superpowers/sdd/build_carry_cache.log 2>&1
```
Run in background. On completion verify: `carry_dir()/{root}.parquet` exists for the roots that had data; the log reports per-root row counts; bonds with no SOFR/duration may have short/zero series (documented caveat).

- [ ] **Step 2: Run the broad-basket carry walk-forward** (background)

```bash
PYTHONPATH=. /c/Users/qwqw1/anaconda3/envs/fintech/python.exe \
  scripts/backtest_scripts/run_carver_walkforward.py \
  --config config/backtesting/carry_broad.yaml \
  --report docs/reports/futures/CARRY_BROAD_READINESS.md \
  > .superpowers/sdd/carry_walkforward.log 2>&1
```

- [ ] **Step 3: Verify + record verdict**

Confirm the log ends with the metrics line; `CARRY_BROAD_READINESS.md` exists with title "FuturesCarry ...", metrics + per-window tables, sane tail stats; the Carver baseline reports are untouched. Summarize the gate outcome (OOS Sharpe, PSR/DSR/PBO, 1.5x cost, verdict) for the user. Clears gate -> viable momentum-uncorrelated strategy; still WEAK -> documented.

- [ ] **Step 4: Commit the report**

```bash
git add -f docs/reports/futures/CARRY_BROAD_READINESS.md
git commit -m "docs(futures): broad-basket carry walk-forward results"
```

---

## Self-Review

- **Spec coverage:** Task 1 = `carry_dir()`; Task 2 = asset_class map; Task 3 = cache builder; Task 4 = strategy + registry; Task 5 = walk-forward generalization (B-deferred); Task 6 = config; Task 7 = acceptance run. All spec components covered.
- **Placeholder scan:** none -- every code + test step is concrete.
- **Type consistency:** `asset_class_for(root) -> str`; `_load_carry(root) -> pd.Series | None`; `forecast_panel(close) -> pd.DataFrame` (matches `SupportsForecastPanel`); `_config_to_kwargs` adds `strategy_name: str`, `strategy_params: dict`; `_run_window`/`walk_forward_carver` accept them; result dict + report read `strategy_name`.
- **Parameter-free:** `carry_scalar`/`ewma_span` are constructor defaults, never swept; walk-forward trial_count stays 1.
- **Isolation:** no edits to the simulator/sizing/loader/Carver strategy/CarryCalculator/equity path; registry + walk-forward changes are additive and default-preserving (Carver title/behavior unchanged, proven by the retained `test_report_interpolates_actual_capital_and_count`).
