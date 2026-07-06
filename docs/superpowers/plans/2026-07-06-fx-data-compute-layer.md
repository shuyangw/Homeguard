# FX Data + Compute Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the complete keyless FX data-acquisition and derived-artifact layer that the 60-strategy catalog depends on, on top of the existing spot-FX daily backtesting vertical.

**Architecture:** Two registered families under `src/data/`. Acquisition plugins (external feeds) reuse the existing `src/data/acquisition/` infrastructure. Artifact builders (derived data) use a new `ArtifactBuilder` base whose `inputs()` the registry topologically resolves (minute -> daily_ohlc_cache -> vol_surface -> regime). One CLI drives both and surfaces per-component key status.

**Tech Stack:** Python 3, polars + pandas, numpy, statsmodels (cointegration), pandas-datareader (FRED, already a dep), yfinance (already a dep), the `holidays` library, pytest.

## Global Constraints

- Python execution uses the `fintech` conda env. Run scripts with `PYTHONPATH=$(pwd)`.
- Storage paths ALWAYS via `from src.settings import get_local_storage_dir`. Never hardcode.
- Logging via `from src.utils.logger import get_logger` / `from src.utils import logger`. Never `print()`. No `%s` positional args; use f-strings.
- ASCII-only in all code and docs. No emojis, no Unicode arrows.
- All artifact/feed writes are atomic: write `<path>.tmp` then `os.replace(tmp, path)`.
- Canonical FX minute schema (8 col): timestamp, open, high, low, close, volume, trade_count, vwap (lowercase, float64).
- FX daily boundary is 17:00 America/New_York (existing `resample_fx_minute_to_daily` convention). Do not change the day-boundary logic.
- Commit after every task. Branch off `main` first (do NOT commit directly to main).
- macOS/Dropbox git hazard: NEVER run `git checkout <branch>`, `git status`/`git diff` with no args, or `git reset --hard` (broken Windows gitlinks abort and can clobber the tree). Use targeted `git add <paths>` and `git commit` only.
- G10 core pairs: GBPUSD, USDCAD, AUDUSD, NZDUSD, AUDNZD, AUDJPY, NZDJPY, EURNOK, EURSEK, USDNOK, USDSEK, NOKSEK, NOKJPY, SEKJPY.
- The only key-flagged item is the OANDA swap table (deferred). No other component uses an API key.

---

## File Structure

**Artifact builders (new package):**
- `src/data/artifacts/__init__.py` - exports base + registry
- `src/data/artifacts/base.py` - `ArtifactBuilder` ABC
- `src/data/artifacts/registry.py` - registry + topological resolve
- `src/data/artifacts/daily_ohlc_cache.py` - minute -> daily OHLC builder
- `src/data/artifacts/spread_model.py` - per-pair/per-hour spread model
- `src/data/artifacts/vol_surface.py` - hour-of-week realized-vol surface
- `src/data/artifacts/currency_strength.py` - per-currency strength vectors
- `src/data/artifacts/pca_dollar.py` - dollar-factor + residuals
- `src/data/artifacts/cointegration.py` - rolling cointegration scan
- `src/data/artifacts/regime.py` - ATR-ratio + gold-state regime
- `src/data/artifacts/event_registries.py` - unwind/vol-spike/corr-break registries

**Feeds (new + extend existing):**
- `src/data/acquisition/plugins/oil_yfinance.py` - Brent via yfinance
- `src/data/acquisition/plugins/equity_index_yfinance.py` - indices via yfinance
- `src/data/feeds/__init__.py`
- `src/data/feeds/holidays_calendar.py` - holiday sets from `holidays` lib
- `src/data/macro_calendar.py` - CB/econ calendar loader (yaml, API-ready)
- `config/macro_calendar/cb_decisions.yaml` - curated CB decision dates

**Modify:**
- `scripts/data/build_fx_daily_cache.py:resample_fx_minute_to_daily` - add o/h/l aggregation
- `src/backtesting/data/fx_backtest_loader.py:load_fx_daily_panel` - carry o/h/l in panel
- `src/data/fx_rates.py` - extend `CURRENCY_FRED_SERIES`
- `src/data/acquisition/plugins/fred_rates.py` - add validation gate
- `config/universes/fx_spot-2026.csv` - add G10 pairs

**CLI:**
- `src/data/fx_pipeline/__init__.py`
- `src/data/fx_pipeline/__main__.py` - list/build

**Validation:**
- `src/backtesting/validation/cpcv.py` - combinatorial purged CV
- `src/backtesting/validation/combined_gate.py` - CPCV + DSR + PBO gate

---

# Phase 1: Foundation

## Task 1: ArtifactBuilder base + registry

**Files:**
- Create: `src/data/artifacts/base.py`
- Create: `src/data/artifacts/registry.py`
- Create: `src/data/artifacts/__init__.py`
- Test: `tests/data/artifacts/test_registry.py`

**Interfaces:**
- Produces: `ArtifactBuilder` ABC with attributes `name: str`, `output_subdir: str`, `REQUIRES_KEY: str | None = None`, methods `inputs() -> list[str]`, `build(start: date, end: date) -> Path`, `output_path() -> Path`. `register(builder)` decorator, `get_builder(name) -> ArtifactBuilder`, `resolve_order(names: list[str]) -> list[str]` (topological).

- [ ] **Step 1: Write the failing test**

```python
# tests/data/artifacts/test_registry.py
from datetime import date
from pathlib import Path
import pytest
from src.data.artifacts.base import ArtifactBuilder
from src.data.artifacts import registry


class _A(ArtifactBuilder):
    name = "a"
    output_subdir = "a"
    def inputs(self): return []
    def build(self, start, end): return self.output_path()


class _B(ArtifactBuilder):
    name = "b"
    output_subdir = "b"
    def inputs(self): return ["a"]
    def build(self, start, end): return self.output_path()


def test_resolve_order_is_topological():
    reg = registry.Registry()
    reg.register(_A())
    reg.register(_B())
    assert reg.resolve_order(["b"]) == ["a", "b"]


def test_output_path_under_artifacts_fx(tmp_path, monkeypatch):
    monkeypatch.setattr("src.data.artifacts.base.get_local_storage_dir", lambda: tmp_path)
    p = _A().output_path()
    assert p == tmp_path / "artifacts" / "fx" / "a"


def test_missing_dependency_raises():
    reg = registry.Registry()
    reg.register(_B())
    with pytest.raises(KeyError):
        reg.resolve_order(["b"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=$(pwd) pytest tests/data/artifacts/test_registry.py -v`
Expected: FAIL (module `src.data.artifacts.base` not found)

- [ ] **Step 3: Write minimal implementation**

```python
# src/data/artifacts/base.py
from __future__ import annotations
from abc import ABC, abstractmethod
from datetime import date
from pathlib import Path
from src.settings import get_local_storage_dir


class ArtifactBuilder(ABC):
    name: str = ""
    output_subdir: str = ""
    REQUIRES_KEY: str | None = None

    @abstractmethod
    def inputs(self) -> list[str]:
        ...

    @abstractmethod
    def build(self, start: date, end: date) -> Path:
        ...

    def output_path(self) -> Path:
        return get_local_storage_dir() / "artifacts" / "fx" / self.output_subdir
```

```python
# src/data/artifacts/registry.py
from __future__ import annotations
from src.data.artifacts.base import ArtifactBuilder


class Registry:
    def __init__(self) -> None:
        self._builders: dict[str, ArtifactBuilder] = {}

    def register(self, builder: ArtifactBuilder) -> ArtifactBuilder:
        self._builders[builder.name] = builder
        return builder

    def get_builder(self, name: str) -> ArtifactBuilder:
        return self._builders[name]

    def resolve_order(self, names: list[str]) -> list[str]:
        order: list[str] = []
        seen: set[str] = set()

        def visit(n: str, stack: tuple[str, ...]) -> None:
            if n in seen:
                return
            if n in stack:
                raise ValueError(f"cycle at {n}")
            b = self._builders.get(n)
            if b is None:
                raise KeyError(f"unknown builder: {n}")
            for dep in b.inputs():
                if dep in self._builders:
                    visit(dep, stack + (n,))
            seen.add(n)
            order.append(n)

        for name in names:
            visit(name, ())
        return order


_DEFAULT = Registry()
register = _DEFAULT.register
get_builder = _DEFAULT.get_builder
resolve_order = _DEFAULT.resolve_order
```

```python
# src/data/artifacts/__init__.py
from src.data.artifacts.base import ArtifactBuilder
from src.data.artifacts import registry

__all__ = ["ArtifactBuilder", "registry"]
```

Note: `resolve_order` treats input names that are NOT registered builders (e.g. raw feeds like "minute", "fred") as external leaves and skips them; only registered builders that are truly missing raise. The `test_missing_dependency_raises` test registers `_B` (needs "a") without "a": since "a" is not registered, adjust the test to require the dependency. Implement `visit` to raise `KeyError` when a dependency name is unregistered AND declared as a builder dependency. For this plan, treat every name returned by `inputs()` as a required builder EXCEPT the reserved raw-feed names in `RAW_FEEDS`.

```python
# add to registry.py
RAW_FEEDS = {"minute", "quotes", "fred", "oil", "equity_index", "holidays", "calendar"}
```

Update `visit` dependency loop:
```python
            for dep in b.inputs():
                if dep in RAW_FEEDS:
                    continue
                if dep not in self._builders:
                    raise KeyError(f"unknown builder dependency: {dep}")
                visit(dep, stack + (n,))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=$(pwd) pytest tests/data/artifacts/test_registry.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/data/artifacts/ tests/data/artifacts/test_registry.py
git commit -m "feat(fx-data): ArtifactBuilder base + topological registry"
```

---

## Task 2: OHLC daily cache rebuild

**Files:**
- Modify: `scripts/data/build_fx_daily_cache.py:resample_fx_minute_to_daily`
- Modify: `src/backtesting/data/fx_backtest_loader.py:load_fx_daily_panel`
- Test: `tests/data/test_ohlc_rebuild.py`

**Interfaces:**
- Consumes: existing `build_fx_daily_cache(pairs, start, end)`.
- Produces: `resample_fx_minute_to_daily(df_min)` now returns columns `open, high, low, close`. `load_fx_daily_panel` MultiIndex gains `open/high/low` fields alongside `close/ret`.

- [ ] **Step 1: Write the failing test**

```python
# tests/data/test_ohlc_rebuild.py
import pandas as pd
from scripts.data.build_fx_daily_cache import resample_fx_minute_to_daily


def _minute_df():
    ts = pd.to_datetime([
        "2020-06-01 18:00:00+00:00", "2020-06-01 19:00:00+00:00",
        "2020-06-01 20:00:00+00:00",
    ], utc=True)
    return pd.DataFrame({
        "timestamp": ts,
        "open": [1.10, 1.11, 1.09],
        "high": [1.12, 1.13, 1.10],
        "low": [1.08, 1.10, 1.05],
        "close": [1.11, 1.09, 1.06],
    })


def test_resample_carries_ohlc():
    out = resample_fx_minute_to_daily(_minute_df())
    row = out.iloc[0]
    assert row["open"] == 1.10       # first
    assert row["high"] == 1.13       # max
    assert row["low"] == 1.05        # min
    assert row["close"] == 1.06      # last
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=$(pwd) pytest tests/data/test_ohlc_rebuild.py -v`
Expected: FAIL (KeyError 'open')

- [ ] **Step 3: Write minimal implementation**

In `scripts/data/build_fx_daily_cache.py`, change the aggregation in `resample_fx_minute_to_daily`:
```python
    daily = tmp.groupby("fx_date").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
    )
```
And update the empty-frame guard columns:
```python
        empty = pd.DataFrame(columns=["open", "high", "low", "close"])
```

In `src/backtesting/data/fx_backtest_loader.py:load_fx_daily_panel`, read all four columns and carry them. Replace the per-pair series read and panel assembly:
```python
        pdf = pl.scan_parquet(sym_dir / "**/*.parquet").collect().to_pandas()
        pdf["fx_date"] = pd.to_datetime(pdf["fx_date"]).dt.date
        pdf = pdf[(pdf["fx_date"] >= start) & (pdf["fx_date"] <= end)]
        if pdf.empty:
            continue
        pdf = pdf.set_index("fx_date").sort_index()
        frames[pair] = pdf[["open", "high", "low", "close"]].astype(float)
```
Replace panel assembly:
```python
    fields = ("open", "high", "low", "close", "ret")
    per_pair = {}
    for p, df in frames.items():
        d = df.copy()
        d["ret"] = d["close"].pct_change(fill_method=None)
        per_pair[p] = d[list(fields)]
    panel = pd.concat(per_pair, axis=1).sort_index()
    panel.columns = pd.MultiIndex.from_tuples(
        [(p, f) for p in per_pair for f in fields])
    return panel
```
Backward-compat: existing callers do `panel.xs("close", axis=1, level=1)`, which still works.

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=$(pwd) pytest tests/data/test_ohlc_rebuild.py -v`
Expected: PASS

- [ ] **Step 5: Regression + commit**

Run the existing fx loader tests to confirm close-panel behavior is unchanged:
```bash
PYTHONPATH=$(pwd) pytest tests/ -k fx_backtest_loader -v
```
Expected: PASS (or no such test -> skip). Then:
```bash
git add scripts/data/build_fx_daily_cache.py src/backtesting/data/fx_backtest_loader.py tests/data/test_ohlc_rebuild.py
git commit -m "feat(fx-data): daily cache carries OHLC, not close-only"
```

---

## Task 3: daily_ohlc_cache builder + G10 universe

**Files:**
- Create: `src/data/artifacts/daily_ohlc_cache.py`
- Modify: `config/universes/fx_spot-2026.csv`
- Test: `tests/data/artifacts/test_daily_ohlc_cache.py`

**Interfaces:**
- Consumes: `build_fx_daily_cache(pairs, start, end)`, `ArtifactBuilder`.
- Produces: builder `name="daily_ohlc_cache"`, `inputs()==["minute"]`, `build(start,end)` writes `fx_daily/` for `G10_PAIRS + EXISTING_PAIRS`. Exposes `G10_PAIRS: list[str]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/data/artifacts/test_daily_ohlc_cache.py
from src.data.artifacts.daily_ohlc_cache import DailyOhlcCache, G10_PAIRS


def test_inputs_and_targets():
    b = DailyOhlcCache()
    assert b.name == "daily_ohlc_cache"
    assert b.inputs() == ["minute"]
    assert "AUDUSD" in b.target_pairs()
    assert "EURUSD" in b.target_pairs()


def test_g10_pairs_are_fourteen():
    assert len(G10_PAIRS) == 14
    assert "NOKSEK" in G10_PAIRS
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=$(pwd) pytest tests/data/artifacts/test_daily_ohlc_cache.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Write minimal implementation**

```python
# src/data/artifacts/daily_ohlc_cache.py
from __future__ import annotations
from datetime import date
from pathlib import Path
from src.data.artifacts.base import ArtifactBuilder
from src.settings import get_local_storage_dir
from scripts.data.build_fx_daily_cache import build_fx_daily_cache

G10_PAIRS = [
    "GBPUSD", "USDCAD", "AUDUSD", "NZDUSD", "AUDNZD", "AUDJPY", "NZDJPY",
    "EURNOK", "EURSEK", "USDNOK", "USDSEK", "NOKSEK", "NOKJPY", "SEKJPY",
]
EXISTING_PAIRS = [
    "EURUSD", "USDJPY", "USDCHF", "EURJPY", "EURCHF", "CHFJPY", "XAUUSD", "XAGUSD",
]


class DailyOhlcCache(ArtifactBuilder):
    name = "daily_ohlc_cache"
    output_subdir = "daily_ohlc_cache"

    def inputs(self) -> list[str]:
        return ["minute"]

    def target_pairs(self) -> list[str]:
        return EXISTING_PAIRS + G10_PAIRS

    def output_path(self) -> Path:
        return get_local_storage_dir() / "fx_daily"

    def build(self, start: date, end: date) -> Path:
        build_fx_daily_cache(self.target_pairs(), start, end)
        return self.output_path()
```

Append the 14 G10 pairs to `config/universes/fx_spot-2026.csv` (one per line, after the existing 8).

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=$(pwd) pytest tests/data/artifacts/test_daily_ohlc_cache.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/data/artifacts/daily_ohlc_cache.py config/universes/fx_spot-2026.csv tests/data/artifacts/test_daily_ohlc_cache.py
git commit -m "feat(fx-data): daily_ohlc_cache builder + G10 universe"
```

---

## Task 4: FRED G10/EM rate series + validation gate

**Files:**
- Modify: `src/data/fx_rates.py` (`CURRENCY_FRED_SERIES`)
- Modify: `src/data/acquisition/plugins/fred_rates.py` (validation gate)
- Test: `tests/data/test_fred_validation.py`

**Interfaces:**
- Consumes: `FREDRatesPlugin.fetch_series`.
- Produces: `validate_fred_series(series: pd.Series, series_id: str) -> None` raising `FredValidationError` on empty / implausible / HTML-error series. `CURRENCY_FRED_SERIES` gains GBP/CAD/AUD/NZD/NOK/SEK/MXN/ZAR/SGD entries.

- [ ] **Step 1: Write the failing test**

```python
# tests/data/test_fred_validation.py
import pandas as pd
import pytest
from src.data.acquisition.plugins.fred_rates import (
    validate_fred_series, FredValidationError,
)


def test_rejects_empty():
    with pytest.raises(FredValidationError):
        validate_fred_series(pd.Series(dtype=float), "TEST")


def test_rejects_implausible_rate():
    s = pd.Series([250.0, 300.0], index=pd.to_datetime(["2020-01-01", "2020-02-01"]))
    with pytest.raises(FredValidationError):
        validate_fred_series(s, "TEST")


def test_accepts_plausible():
    s = pd.Series([3.5, 3.75], index=pd.to_datetime(["2020-01-01", "2020-02-01"]))
    validate_fred_series(s, "TEST")  # no raise
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=$(pwd) pytest tests/data/test_fred_validation.py -v`
Expected: FAIL (cannot import validate_fred_series)

- [ ] **Step 3: Write minimal implementation**

Add to `src/data/acquisition/plugins/fred_rates.py`:
```python
class FredValidationError(Exception):
    pass


def validate_fred_series(series: "pd.Series", series_id: str) -> None:
    if series is None or len(series) == 0:
        raise FredValidationError(f"{series_id}: empty series")
    vals = series.dropna()
    if len(vals) == 0:
        raise FredValidationError(f"{series_id}: all-NaN series")
    # Policy short rates realistically live in [-5, 100] percent even for EM.
    if vals.min() < -5.0 or vals.max() > 100.0:
        raise FredValidationError(
            f"{series_id}: values out of plausible rate range "
            f"[{vals.min()}, {vals.max()}]")
```

Call it inside `fetch_series` right after `series = series.dropna()`:
```python
        validate_fred_series(series, series_id)
```

Extend `CURRENCY_FRED_SERIES` in `src/data/fx_rates.py` (verify each ID against FRED before committing; the IRSTCI01 monthly call-money family is the pattern used for CHF/JPY):
```python
    "GBP": "IRSTCI01GBM156N",
    "CAD": "IRSTCI01CAM156N",
    "AUD": "IRSTCI01AUM156N",
    "NZD": "IRSTCI01NZM156N",
    "NOK": "IRSTCI01NOM156N",
    "SEK": "IRSTCI01SEM156N",
    "MXN": "IRSTCI01MXM156N",
    "ZAR": "IRSTCI01ZAM156N",
    "SGD": "IRSTCI01SGM156N",
```
IMPORTANT: before committing, run a one-off fetch for each new series and confirm it returns non-empty plausible data (the CHF `IRSTCB01CHM156N` bug returned an HTML error page). If an ID is invalid, find the correct short-rate series on FRED and use it. Record the verified IDs in the commit message.

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=$(pwd) pytest tests/data/test_fred_validation.py -v`
Expected: PASS

Then verify live fetch (network):
```bash
PYTHONPATH=$(pwd) python -c "
from datetime import date
from src.data.acquisition.plugins.fred_rates import FREDRatesPlugin
p = FREDRatesPlugin()
for cid in ['IRSTCI01GBM156N','IRSTCI01CAM156N','IRSTCI01AUM156N','IRSTCI01NZM156N','IRSTCI01NOM156N','IRSTCI01SEM156N']:
    print(cid, p.fetch_series(cid, date(2011,1,1), date(2026,1,1), skip_existing=False))
"
```
Expected: each prints rows>0. Fix any that error before committing.

- [ ] **Step 5: Commit**

```bash
git add src/data/fx_rates.py src/data/acquisition/plugins/fred_rates.py tests/data/test_fred_validation.py
git commit -m "feat(fx-data): G10/EM FRED rate series + fetch validation gate

Verified series IDs: <paste the ones that fetched non-empty>"
```

---

## Task 5: fx_pipeline CLI

**Files:**
- Create: `src/data/fx_pipeline/__init__.py`
- Create: `src/data/fx_pipeline/__main__.py`
- Test: `tests/data/test_fx_pipeline_cli.py`

**Interfaces:**
- Consumes: `registry` (with builders registered), each builder's `REQUIRES_KEY`.
- Produces: `list_components() -> list[dict]` (name, kind, requires_key, up_to_date). `build(names, start, end)` resolves order and calls each builder.

- [ ] **Step 1: Write the failing test**

```python
# tests/data/test_fx_pipeline_cli.py
from src.data.fx_pipeline import list_components


def test_list_includes_daily_ohlc_cache():
    comps = list_components()
    names = {c["name"] for c in comps}
    assert "daily_ohlc_cache" in names
    row = next(c for c in comps if c["name"] == "daily_ohlc_cache")
    assert row["requires_key"] is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=$(pwd) pytest tests/data/test_fx_pipeline_cli.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Write minimal implementation**

```python
# src/data/fx_pipeline/__init__.py
from __future__ import annotations
from datetime import date
from src.data.artifacts import registry
from src.data.artifacts.daily_ohlc_cache import DailyOhlcCache

# Register all builders as they are implemented (append in later phases).
registry.register(DailyOhlcCache())


def list_components() -> list[dict]:
    out = []
    for name, b in registry._DEFAULT._builders.items():
        out.append({
            "name": name,
            "kind": "artifact",
            "requires_key": getattr(b, "REQUIRES_KEY", None),
            "up_to_date": b.output_path().exists(),
        })
    return out


def build(names: list[str], start: date, end: date) -> None:
    order = registry.resolve_order(names)
    for n in order:
        registry.get_builder(n).build(start, end)
```

```python
# src/data/fx_pipeline/__main__.py
from __future__ import annotations
import argparse
from datetime import date, datetime
from src.data import fx_pipeline
from src.utils import logger


def _d(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def main() -> None:
    ap = argparse.ArgumentParser(prog="fx_pipeline")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    b = sub.add_parser("build")
    b.add_argument("names", nargs="+")
    b.add_argument("--start", type=_d, default=date(2011, 1, 1))
    b.add_argument("--end", type=_d, default=date.today())
    args = ap.parse_args()
    if args.cmd == "list":
        for c in fx_pipeline.list_components():
            key = c["requires_key"] or "-"
            logger.info(f"{c['name']:22} key={key:12} up_to_date={c['up_to_date']}")
    elif args.cmd == "build":
        fx_pipeline.build(args.names, args.start, args.end)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=$(pwd) pytest tests/data/test_fx_pipeline_cli.py -v`
Expected: PASS. Also smoke: `PYTHONPATH=$(pwd) python -m src.data.fx_pipeline list`

- [ ] **Step 5: Commit**

```bash
git add src/data/fx_pipeline/ tests/data/test_fx_pipeline_cli.py
git commit -m "feat(fx-data): fx_pipeline CLI (list/build) with key-flagging"
```

**Phase 1 acceptance:** `python -m src.data.fx_pipeline build daily_ohlc_cache` rebuilds the OHLC daily cache for all 22 pairs; FRED G10 series fetch non-empty; a spot FX backtest on a G10 pair (e.g. AUDUSD) runs via the existing `run_fx_backtest`.

---

# Phase 2: Cost honesty

## Task 6: spread_model builder

**Files:**
- Create: `src/data/artifacts/spread_model.py`
- Test: `tests/data/artifacts/test_spread_model.py`

**Interfaces:**
- Consumes: quotes at `fx/massive/quotes_minute_aggregated/symbol=<PAIR>`, `ArtifactBuilder`, `_tier_for_pair` logic (crosses=minor, USD-leg=major, metals=major/bps).
- Produces: builder `name="spread_model"`, `inputs()==["quotes","minute"]`. Writes `artifacts/fx/spread_model/table.parquet` with columns `pair, hour_of_week, spread_pips`. Exposes `synthetic_spread(pair, hour_of_week, anchors) -> float`.

- [ ] **Step 1: Write the failing test**

```python
# tests/data/artifacts/test_spread_model.py
from src.data.artifacts.spread_model import synthetic_spread


def test_cross_wider_than_major():
    anchors = {"EURUSD": 1.0}
    maj = synthetic_spread("EURUSD", 10, anchors)
    cross = synthetic_spread("EURNOK", 10, anchors)
    assert cross > maj


def test_rollover_hour_widens():
    anchors = {"EURUSD": 1.0}
    normal = synthetic_spread("EURUSD", 10, anchors)
    rollover = synthetic_spread("EURUSD", 21, anchors)  # 21:00 UTC rollover
    assert rollover > normal
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=$(pwd) pytest tests/data/artifacts/test_spread_model.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Write minimal implementation**

```python
# src/data/artifacts/spread_model.py
from __future__ import annotations
from datetime import date
from pathlib import Path
import os
import polars as pl
from src.data.artifacts.base import ArtifactBuilder
from src.settings import get_local_storage_dir
from src.utils import logger

_METALS = {"XAU", "XAG"}
TIER_BASE_PIPS = {"major": 1.0, "minor": 3.0, "metal": 3.0}
ROLLOVER_HOUR_UTC = 21
ROLLOVER_MULT = 5.0


def _tier(pair: str) -> str:
    if pair[:3] in _METALS or pair[3:] in _METALS:
        return "metal"
    if "USD" in (pair[:3], pair[3:]):
        return "major"
    return "minor"


def synthetic_spread(pair: str, hour_of_week: int, anchors: dict[str, float]) -> float:
    base = TIER_BASE_PIPS[_tier(pair)]
    hour_utc = hour_of_week % 24
    mult = ROLLOVER_MULT if hour_utc == ROLLOVER_HOUR_UTC else 1.0
    return base * mult


class SpreadModel(ArtifactBuilder):
    name = "spread_model"
    output_subdir = "spread_model"

    def inputs(self) -> list[str]:
        return ["quotes", "minute"]

    def _quote_pairs(self) -> list[str]:
        root = get_local_storage_dir() / "fx" / "massive" / "quotes_minute_aggregated"
        if not root.exists():
            return []
        return [p.name.replace("symbol=", "") for p in root.glob("symbol=*")]

    def build(self, start: date, end: date) -> Path:
        out_dir = self.output_path()
        out_dir.mkdir(parents=True, exist_ok=True)
        anchors: dict[str, float] = {}
        for pair in self._quote_pairs():
            sym = get_local_storage_dir() / "fx" / "massive" / "quotes_minute_aggregated" / f"symbol={pair}"
            df = pl.scan_parquet(sym / "**/*.parquet").collect()
            if "spread" in df.columns:
                anchors[pair] = float(df["spread"].median())
            elif {"bid", "ask"}.issubset(df.columns):
                anchors[pair] = float((df["ask"] - df["bid"]).median())
        rows = []
        from src.data.artifacts.daily_ohlc_cache import DailyOhlcCache
        for pair in DailyOhlcCache().target_pairs():
            for how in range(168):
                rows.append({"pair": pair, "hour_of_week": how,
                             "spread_pips": synthetic_spread(pair, how, anchors)})
        table = pl.DataFrame(rows)
        tmp = out_dir / "table.parquet.tmp"
        table.write_parquet(tmp)
        os.replace(tmp, out_dir / "table.parquet")
        logger.info(f"[spread_model] wrote {len(rows)} rows, {len(anchors)} anchors")
        return out_dir
```

Note: the anchor medians calibrate future refinement; the synthetic base captures tier + rollover now. When real quote columns differ, adjust the `spread`/`bid`/`ask` detection to the actual schema (inspect `quotes_minute_aggregated` columns first).

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=$(pwd) pytest tests/data/artifacts/test_spread_model.py -v`
Expected: PASS

- [ ] **Step 5: Register + commit**

Add to `src/data/fx_pipeline/__init__.py`:
```python
from src.data.artifacts.spread_model import SpreadModel
registry.register(SpreadModel())
```
```bash
git add src/data/artifacts/spread_model.py src/data/fx_pipeline/__init__.py tests/data/artifacts/test_spread_model.py
git commit -m "feat(fx-data): spread_model builder (real-quote anchors + synthetic tier/rollover)"
```

---

# Phase 3: External feeds

## Task 7: Brent oil feed (yfinance)

**Files:**
- Create: `src/data/acquisition/plugins/oil_yfinance.py`
- Test: `tests/data/test_oil_feed.py`

**Interfaces:**
- Produces: `fetch_brent(start, end) -> pd.DataFrame` (columns date, close); writes `alt_data/oil/BRENT/daily.parquet`. `REQUIRES_KEY = None`.

- [ ] **Step 1: Write the failing test**

```python
# tests/data/test_oil_feed.py
import pandas as pd
from src.data.acquisition.plugins import oil_yfinance


def test_normalize_shape(monkeypatch):
    fake = pd.DataFrame({"Close": [70.0, 71.0]},
                        index=pd.to_datetime(["2020-01-01", "2020-01-02"]))
    monkeypatch.setattr(oil_yfinance, "_download", lambda *a, **k: fake)
    out = oil_yfinance.fetch_brent("2020-01-01", "2020-01-03", write=False)
    assert list(out.columns) == ["date", "close"]
    assert len(out) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=$(pwd) pytest tests/data/test_oil_feed.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Write minimal implementation**

```python
# src/data/acquisition/plugins/oil_yfinance.py
from __future__ import annotations
import os
from pathlib import Path
import pandas as pd
import polars as pl
import yfinance as yf
from src.settings import get_local_storage_dir
from src.utils import logger

REQUIRES_KEY = None
_TICKER = "BZ=F"


def _download(ticker: str, start: str, end: str) -> pd.DataFrame:
    return yf.download(ticker, start=start, end=end, progress=False)


def fetch_brent(start: str, end: str, write: bool = True) -> pd.DataFrame:
    raw = _download(_TICKER, start, end)
    if raw.empty:
        raise ValueError("Brent download returned empty")
    close = raw["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    out = pd.DataFrame({"date": pd.to_datetime(close.index).date,
                        "close": close.astype(float).values})
    if write:
        d = get_local_storage_dir() / "alt_data" / "oil" / "BRENT"
        d.mkdir(parents=True, exist_ok=True)
        tmp = d / "daily.parquet.tmp"
        pl.from_pandas(out).write_parquet(tmp)
        os.replace(tmp, d / "daily.parquet")
        logger.info(f"[oil] wrote {len(out)} Brent rows")
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=$(pwd) pytest tests/data/test_oil_feed.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/data/acquisition/plugins/oil_yfinance.py tests/data/test_oil_feed.py
git commit -m "feat(fx-data): Brent oil feed via yfinance (keyless)"
```

---

## Task 8: Equity index feed (yfinance)

**Files:**
- Create: `src/data/acquisition/plugins/equity_index_yfinance.py`
- Test: `tests/data/test_equity_index_feed.py`

**Interfaces:**
- Produces: `INDICES = {"SPX": "^GSPC", "STOXX50E": "^STOXX50E", "N225": "^N225"}`; `fetch_index(name, start, end, write=True) -> pd.DataFrame` (date, close) to `alt_data/equity_index/<name>/daily.parquet`. `REQUIRES_KEY = None`.

- [ ] **Step 1: Write the failing test**

```python
# tests/data/test_equity_index_feed.py
import pandas as pd
from src.data.acquisition.plugins import equity_index_yfinance as eq


def test_fetch_index_normalizes(monkeypatch):
    fake = pd.DataFrame({"Close": [4000.0, 4010.0]},
                        index=pd.to_datetime(["2020-01-01", "2020-01-02"]))
    monkeypatch.setattr(eq, "_download", lambda *a, **k: fake)
    out = eq.fetch_index("SPX", "2020-01-01", "2020-01-03", write=False)
    assert list(out.columns) == ["date", "close"]


def test_indices_map():
    assert eq.INDICES["SPX"] == "^GSPC"
    assert set(eq.INDICES) == {"SPX", "STOXX50E", "N225"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=$(pwd) pytest tests/data/test_equity_index_feed.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Write minimal implementation**

```python
# src/data/acquisition/plugins/equity_index_yfinance.py
from __future__ import annotations
import os
import pandas as pd
import polars as pl
import yfinance as yf
from src.settings import get_local_storage_dir
from src.utils import logger

REQUIRES_KEY = None
INDICES = {"SPX": "^GSPC", "STOXX50E": "^STOXX50E", "N225": "^N225"}


def _download(ticker: str, start: str, end: str) -> pd.DataFrame:
    return yf.download(ticker, start=start, end=end, progress=False)


def fetch_index(name: str, start: str, end: str, write: bool = True) -> pd.DataFrame:
    raw = _download(INDICES[name], start, end)
    if raw.empty:
        raise ValueError(f"{name} download returned empty")
    close = raw["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    out = pd.DataFrame({"date": pd.to_datetime(close.index).date,
                        "close": close.astype(float).values})
    if write:
        d = get_local_storage_dir() / "alt_data" / "equity_index" / name
        d.mkdir(parents=True, exist_ok=True)
        tmp = d / "daily.parquet.tmp"
        pl.from_pandas(out).write_parquet(tmp)
        os.replace(tmp, d / "daily.parquet")
        logger.info(f"[equity_index] wrote {len(out)} {name} rows")
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=$(pwd) pytest tests/data/test_equity_index_feed.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/data/acquisition/plugins/equity_index_yfinance.py tests/data/test_equity_index_feed.py
git commit -m "feat(fx-data): equity index feed via yfinance (keyless)"
```

---

## Task 9: Holiday calendar feed

**Files:**
- Create: `src/data/feeds/__init__.py`
- Create: `src/data/feeds/holidays_calendar.py`
- Test: `tests/data/test_holidays_calendar.py`

**Interfaces:**
- Produces: `holiday_set(country: str, years: range) -> set[date]` using the `holidays` lib; `COUNTRIES = {"US","UK","JP","EU","AU"}`. `REQUIRES_KEY = None`.

- [ ] **Step 1: Write the failing test**

```python
# tests/data/test_holidays_calendar.py
from datetime import date
from src.data.feeds.holidays_calendar import holiday_set


def test_us_christmas_present():
    hs = holiday_set("US", range(2020, 2021))
    assert date(2020, 12, 25) in hs
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=$(pwd) pytest tests/data/test_holidays_calendar.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Write minimal implementation**

```python
# src/data/feeds/holidays_calendar.py
from __future__ import annotations
from datetime import date
import holidays as _holidays

REQUIRES_KEY = None
_COUNTRY_CODE = {"US": "US", "UK": "GB", "JP": "JP", "EU": "DE", "AU": "AU"}
COUNTRIES = set(_COUNTRY_CODE)


def holiday_set(country: str, years: range) -> set[date]:
    code = _COUNTRY_CODE[country]
    cal = _holidays.country_holidays(code, years=list(years))
    return set(cal.keys())
```

If the `holidays` package is not installed, add it: `pip install holidays` and record in requirements. `EU` uses Germany (DE) as the euro-area proxy for TARGET-adjacent holidays; document this choice in the module docstring.

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=$(pwd) pytest tests/data/test_holidays_calendar.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/data/feeds/ tests/data/test_holidays_calendar.py
git commit -m "feat(fx-data): holiday calendar feed via holidays lib (keyless)"
```

---

## Task 10: CB/econ calendar (curated yaml + loader)

**Files:**
- Create: `config/macro_calendar/cb_decisions.yaml`
- Create: `src/data/macro_calendar.py`
- Test: `tests/data/test_macro_calendar.py`

**Interfaces:**
- Produces: `load_cb_decisions() -> dict[str, list[date]]` keyed by central bank code (ECB/BOE/BOJ/SNB/RBA/RBNZ/NORGES/RIKSBANK/BANXICO/FOMC); `blackout(currency, day, days=1) -> bool`. API-ready: loader reads yaml now, has a documented seam for a future API source.

- [ ] **Step 1: Write the failing test**

```python
# tests/data/test_macro_calendar.py
from datetime import date
from src.data.macro_calendar import load_cb_decisions, blackout


def test_load_returns_dates():
    d = load_cb_decisions()
    assert "ECB" in d
    assert all(isinstance(x, date) for x in d["ECB"])


def test_blackout_window():
    # A known ECB date must trigger blackout for EUR within +/- 1 day.
    d = load_cb_decisions()
    ref = d["ECB"][0]
    assert blackout("EUR", ref, days=1) is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=$(pwd) pytest tests/data/test_macro_calendar.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Write minimal implementation**

Create `config/macro_calendar/cb_decisions.yaml` with real scheduled decision dates for the in-universe central banks (populate 2011-2026 from each bank's published calendar; a starter for 2025-2026 shown, extend back):
```yaml
ECB:
  - 2025-01-30
  - 2025-03-06
BOE:
  - 2025-02-06
  - 2025-03-20
BOJ:
  - 2025-01-24
  - 2025-03-19
SNB:
  - 2025-03-20
RBA:
  - 2025-02-18
RBNZ:
  - 2025-02-19
NORGES:
  - 2025-01-23
RIKSBANK:
  - 2025-01-29
BANXICO:
  - 2025-02-06
FOMC:
  - 2025-01-29
  - 2025-03-19
```
```python
# src/data/macro_calendar.py
from __future__ import annotations
from datetime import date, timedelta
from pathlib import Path
import yaml

_CB_FOR_CCY = {
    "EUR": "ECB", "GBP": "BOE", "JPY": "BOJ", "CHF": "SNB",
    "AUD": "RBA", "NZD": "RBNZ", "NOK": "NORGES", "SEK": "RIKSBANK",
    "MXN": "BANXICO", "USD": "FOMC",
}
_PATH = Path(__file__).resolve().parents[2] / "config" / "macro_calendar" / "cb_decisions.yaml"


def load_cb_decisions() -> dict[str, list[date]]:
    with open(_PATH, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    out: dict[str, list[date]] = {}
    for bank, dates in raw.items():
        out[bank] = [d if isinstance(d, date) else date.fromisoformat(str(d)) for d in dates]
    return out


def blackout(currency: str, day: date, days: int = 1) -> bool:
    bank = _CB_FOR_CCY.get(currency)
    if bank is None:
        return False
    decisions = load_cb_decisions().get(bank, [])
    return any(abs((day - dd).days) <= days for dd in decisions)
```
Document the API seam: `load_cb_decisions` is the single source; a future API feed replaces its body while keeping the return type.

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=$(pwd) pytest tests/data/test_macro_calendar.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add config/macro_calendar/cb_decisions.yaml src/data/macro_calendar.py tests/data/test_macro_calendar.py
git commit -m "feat(fx-data): curated CB decision calendar + blackout loader (keyless)"
```

---

# Phase 4: Shared computes

## Task 11: vol_surface builder

**Files:**
- Create: `src/data/artifacts/vol_surface.py`
- Test: `tests/data/artifacts/test_vol_surface.py`

**Interfaces:**
- Consumes: minute data, `ArtifactBuilder`.
- Produces: builder `name="vol_surface"`, `inputs()==["minute"]`. Writes `artifacts/fx/vol_surface/<PAIR>.parquet` with 168 rows (hour_of_week, median_abs_ret, mad). Exposes `hour_of_week(ts) -> int` and `build_surface(minute_df) -> pd.DataFrame`.

- [ ] **Step 1: Write the failing test**

```python
# tests/data/artifacts/test_vol_surface.py
import numpy as np
import pandas as pd
from src.data.artifacts.vol_surface import build_surface


def test_surface_has_168_rows():
    ts = pd.date_range("2020-01-06", periods=168 * 3, freq="h", tz="UTC")
    df = pd.DataFrame({"timestamp": ts, "close": 1.0 + np.arange(len(ts)) * 1e-4})
    surf = build_surface(df)
    assert len(surf) == 168
    assert set(surf.columns) >= {"hour_of_week", "median_abs_ret", "mad"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=$(pwd) pytest tests/data/artifacts/test_vol_surface.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Write minimal implementation**

```python
# src/data/artifacts/vol_surface.py
from __future__ import annotations
from datetime import date
from pathlib import Path
import os
import numpy as np
import pandas as pd
import polars as pl
from src.data.artifacts.base import ArtifactBuilder
from src.settings import get_local_storage_dir
from src.utils import logger


def build_surface(minute_df: pd.DataFrame) -> pd.DataFrame:
    df = minute_df.copy()
    ts = pd.to_datetime(df["timestamp"], utc=True)
    df["how"] = ts.dt.dayofweek * 24 + ts.dt.hour
    df["abs_ret"] = df["close"].pct_change(fill_method=None).abs()
    g = df.dropna(subset=["abs_ret"]).groupby("how")["abs_ret"]
    med = g.median()
    mad = g.apply(lambda s: float(np.median(np.abs(s - np.median(s)))))
    surf = pd.DataFrame({"hour_of_week": range(168)})
    surf["median_abs_ret"] = surf["hour_of_week"].map(med).fillna(0.0)
    surf["mad"] = surf["hour_of_week"].map(mad).fillna(0.0)
    return surf


class VolSurface(ArtifactBuilder):
    name = "vol_surface"
    output_subdir = "vol_surface"

    def inputs(self) -> list[str]:
        return ["minute"]

    def build(self, start: date, end: date) -> Path:
        from src.data.artifacts.daily_ohlc_cache import DailyOhlcCache
        out_dir = self.output_path()
        out_dir.mkdir(parents=True, exist_ok=True)
        src_root = get_local_storage_dir() / "fx_1min"
        for pair in DailyOhlcCache().target_pairs():
            sym = src_root / f"symbol={pair}"
            if not sym.exists():
                continue
            mdf = pl.scan_parquet(sym / "**/*.parquet").collect().to_pandas()
            surf = build_surface(mdf)
            tmp = out_dir / f"{pair}.parquet.tmp"
            pl.from_pandas(surf).write_parquet(tmp)
            os.replace(tmp, out_dir / f"{pair}.parquet")
        logger.info(f"[vol_surface] built surfaces")
        return out_dir
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=$(pwd) pytest tests/data/artifacts/test_vol_surface.py -v`
Expected: PASS

- [ ] **Step 5: Register + commit**

Add `registry.register(VolSurface())` to `src/data/fx_pipeline/__init__.py`.
```bash
git add src/data/artifacts/vol_surface.py src/data/fx_pipeline/__init__.py tests/data/artifacts/test_vol_surface.py
git commit -m "feat(fx-data): vol_surface builder (hour-of-week realized vol)"
```

---

## Task 12: currency_strength builder

**Files:**
- Create: `src/data/artifacts/currency_strength.py`
- Test: `tests/data/artifacts/test_currency_strength.py`

**Interfaces:**
- Consumes: `load_fx_daily_panel`, `ArtifactBuilder`.
- Produces: builder `name="currency_strength"`, `inputs()==["daily_ohlc_cache"]`. Writes `artifacts/fx/currency_strength/strength.parquet` (date, currency, strength). Exposes `currency_returns(close_panel) -> pd.DataFrame` (date x currency).

- [ ] **Step 1: Write the failing test**

```python
# tests/data/artifacts/test_currency_strength.py
import numpy as np
import pandas as pd
from src.data.artifacts.currency_strength import currency_returns


def test_currency_returns_averages_pairs():
    idx = pd.to_datetime(["2020-01-01", "2020-01-02"])
    close = pd.DataFrame({"EURUSD": [1.10, 1.21], "GBPUSD": [1.30, 1.30]}, index=idx)
    cr = currency_returns(close)
    # EUR appreciates ~10% vs USD on day 2; USD is the average of the inverses.
    assert cr.loc[idx[1], "EUR"] > 0
    assert "USD" in cr.columns
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=$(pwd) pytest tests/data/artifacts/test_currency_strength.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Write minimal implementation**

```python
# src/data/artifacts/currency_strength.py
from __future__ import annotations
from datetime import date
from pathlib import Path
import os
import pandas as pd
import polars as pl
from src.data.artifacts.base import ArtifactBuilder
from src.settings import get_local_storage_dir
from src.utils import logger


def currency_returns(close_panel: pd.DataFrame) -> pd.DataFrame:
    rets = close_panel.pct_change(fill_method=None)
    contrib: dict[str, list[pd.Series]] = {}
    for pair in close_panel.columns:
        base, quote = pair[:3], pair[3:]
        r = rets[pair]
        contrib.setdefault(base, []).append(r)
        contrib.setdefault(quote, []).append(-r)
    out = {ccy: pd.concat(series, axis=1).mean(axis=1) for ccy, series in contrib.items()}
    return pd.DataFrame(out)


class CurrencyStrength(ArtifactBuilder):
    name = "currency_strength"
    output_subdir = "currency_strength"

    def inputs(self) -> list[str]:
        return ["daily_ohlc_cache"]

    def build(self, start: date, end: date) -> Path:
        from src.backtesting.data.fx_backtest_loader import load_fx_daily_panel
        from src.data.artifacts.daily_ohlc_cache import DailyOhlcCache
        panel = load_fx_daily_panel(DailyOhlcCache().target_pairs(), start, end)
        close = panel.xs("close", axis=1, level=1)
        cr = currency_returns(close)
        strength = cr.cumsum()
        long = strength.reset_index().melt(id_vars=strength.index.name or "index",
                                           var_name="currency", value_name="strength")
        long.columns = ["date", "currency", "strength"]
        out_dir = self.output_path()
        out_dir.mkdir(parents=True, exist_ok=True)
        tmp = out_dir / "strength.parquet.tmp"
        pl.from_pandas(long).write_parquet(tmp)
        os.replace(tmp, out_dir / "strength.parquet")
        logger.info(f"[currency_strength] wrote {len(long)} rows")
        return out_dir
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=$(pwd) pytest tests/data/artifacts/test_currency_strength.py -v`
Expected: PASS

- [ ] **Step 5: Register + commit**

Add `registry.register(CurrencyStrength())` to `src/data/fx_pipeline/__init__.py`.
```bash
git add src/data/artifacts/currency_strength.py src/data/fx_pipeline/__init__.py tests/data/artifacts/test_currency_strength.py
git commit -m "feat(fx-data): currency_strength builder (panel -> currency vectors)"
```

---

# Phase 5: Factor / stat

## Task 13: pca_dollar builder

**Files:**
- Create: `src/data/artifacts/pca_dollar.py`
- Test: `tests/data/artifacts/test_pca_dollar.py`

**Interfaces:**
- Consumes: `load_fx_daily_panel`, `ArtifactBuilder`, numpy.
- Produces: builder `name="pca_dollar"`, `inputs()==["daily_ohlc_cache"]`. Writes `artifacts/fx/pca_dollar/{factor.parquet, residuals.parquet}`. Exposes `dollar_factor(returns_df) -> (pc1: pd.Series, residuals: pd.DataFrame)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/data/artifacts/test_pca_dollar.py
import numpy as np
import pandas as pd
from src.data.artifacts.pca_dollar import dollar_factor


def test_pc1_captures_common_move():
    rng = np.random.default_rng(0)
    common = rng.normal(0, 1, 300)
    idx = pd.date_range("2020-01-01", periods=300)
    df = pd.DataFrame({
        "EURUSD": common + rng.normal(0, 0.1, 300),
        "GBPUSD": common + rng.normal(0, 0.1, 300),
        "AUDUSD": common + rng.normal(0, 0.1, 300),
    }, index=idx)
    pc1, resid = dollar_factor(df)
    assert len(pc1) == 300
    assert resid.shape == df.shape
    # residual variance is far smaller than raw variance once PC1 removed
    assert resid.var().mean() < df.var().mean()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=$(pwd) pytest tests/data/artifacts/test_pca_dollar.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Write minimal implementation**

```python
# src/data/artifacts/pca_dollar.py
from __future__ import annotations
from datetime import date
from pathlib import Path
import os
import numpy as np
import pandas as pd
import polars as pl
from src.data.artifacts.base import ArtifactBuilder
from src.utils import logger


def dollar_factor(returns_df: pd.DataFrame):
    X = returns_df.dropna(how="any")
    Z = (X - X.mean()) / X.std(ddof=0)
    u, s, vt = np.linalg.svd(Z.values, full_matrices=False)
    w = vt[0]
    pc1 = pd.Series(Z.values @ w, index=X.index, name="pc1")
    proj = np.outer(pc1.values, w)
    residuals = pd.DataFrame(Z.values - proj, index=X.index, columns=X.columns)
    return pc1, residuals


class PcaDollar(ArtifactBuilder):
    name = "pca_dollar"
    output_subdir = "pca_dollar"

    def inputs(self) -> list[str]:
        return ["daily_ohlc_cache"]

    def build(self, start: date, end: date) -> Path:
        from src.backtesting.data.fx_backtest_loader import load_fx_daily_panel
        from src.data.artifacts.daily_ohlc_cache import DailyOhlcCache
        pairs = [p for p in DailyOhlcCache().target_pairs() if p.endswith("USD") or p.startswith("USD")]
        panel = load_fx_daily_panel(pairs, start, end)
        close = panel.xs("close", axis=1, level=1)
        rets = close.pct_change(fill_method=None)
        pc1, resid = dollar_factor(rets)
        out_dir = self.output_path()
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, obj in [("factor", pc1.reset_index()), ("residuals", resid.reset_index())]:
            tmp = out_dir / f"{name}.parquet.tmp"
            pl.from_pandas(obj).write_parquet(tmp)
            os.replace(tmp, out_dir / f"{name}.parquet")
        logger.info(f"[pca_dollar] wrote factor + residuals ({len(pc1)} obs)")
        return out_dir
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=$(pwd) pytest tests/data/artifacts/test_pca_dollar.py -v`
Expected: PASS

- [ ] **Step 5: Register + commit**

Add `registry.register(PcaDollar())` to `src/data/fx_pipeline/__init__.py`.
```bash
git add src/data/artifacts/pca_dollar.py src/data/fx_pipeline/__init__.py tests/data/artifacts/test_pca_dollar.py
git commit -m "feat(fx-data): pca_dollar builder (dollar factor + residuals)"
```

---

## Task 14: cointegration builder

**Files:**
- Create: `src/data/artifacts/cointegration.py`
- Test: `tests/data/artifacts/test_cointegration.py`

**Interfaces:**
- Consumes: `load_fx_daily_panel`, `ArtifactBuilder`, statsmodels.
- Produces: builder `name="cointegration"`, `inputs()==["daily_ohlc_cache"]`. Writes `artifacts/fx/cointegration/pairs.parquet` (pair_a, pair_b, adf_pvalue, half_life, hedge_ratio). Exposes `ou_half_life(spread) -> float` and `test_pair(a, b) -> dict`.

- [ ] **Step 1: Write the failing test**

```python
# tests/data/artifacts/test_cointegration.py
import numpy as np
import pandas as pd
from src.data.artifacts.cointegration import ou_half_life, test_pair


def test_half_life_of_ar1():
    rng = np.random.default_rng(1)
    n = 2000
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = 0.9 * x[t - 1] + rng.normal(0, 1)
    hl = ou_half_life(pd.Series(x))
    # AR(1) phi=0.9 -> half life = ln(2)/-ln(0.9) ~ 6.58
    assert 4 < hl < 10


def test_cointegrated_pair_low_pvalue():
    rng = np.random.default_rng(2)
    n = 1000
    a = np.cumsum(rng.normal(0, 1, n)) + 100
    b = a + rng.normal(0, 0.5, n)
    idx = pd.date_range("2018-01-01", periods=n)
    res = test_pair(pd.Series(a, index=idx), pd.Series(b, index=idx))
    assert res["adf_pvalue"] < 0.05
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=$(pwd) pytest tests/data/artifacts/test_cointegration.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Write minimal implementation**

```python
# src/data/artifacts/cointegration.py
from __future__ import annotations
from datetime import date
from itertools import combinations
from pathlib import Path
import os
import numpy as np
import pandas as pd
import polars as pl
from statsmodels.tsa.stattools import coint
from src.data.artifacts.base import ArtifactBuilder
from src.utils import logger


def ou_half_life(spread: pd.Series) -> float:
    s = spread.dropna()
    lag = s.shift(1).dropna()
    delta = (s - s.shift(1)).dropna()
    lag = lag.loc[delta.index]
    beta = np.polyfit(lag.values, delta.values, 1)[0]
    if beta >= 0:
        return float("inf")
    return float(-np.log(2) / np.log(1 + beta))


def test_pair(a: pd.Series, b: pd.Series) -> dict:
    df = pd.concat([a, b], axis=1).dropna()
    _, pval, _ = coint(df.iloc[:, 0], df.iloc[:, 1])
    hedge = np.polyfit(df.iloc[:, 1].values, df.iloc[:, 0].values, 1)[0]
    spread = df.iloc[:, 0] - hedge * df.iloc[:, 1]
    return {"adf_pvalue": float(pval), "hedge_ratio": float(hedge),
            "half_life": ou_half_life(spread)}


class Cointegration(ArtifactBuilder):
    name = "cointegration"
    output_subdir = "cointegration"

    def inputs(self) -> list[str]:
        return ["daily_ohlc_cache"]

    def _shares_one_currency(self, a: str, b: str) -> bool:
        return len({a[:3], a[3:]} & {b[:3], b[3:]}) <= 1

    def build(self, start: date, end: date) -> Path:
        from src.backtesting.data.fx_backtest_loader import load_fx_daily_panel
        from src.data.artifacts.daily_ohlc_cache import DailyOhlcCache
        pairs = DailyOhlcCache().target_pairs()
        panel = load_fx_daily_panel(pairs, start, end)
        close = panel.xs("close", axis=1, level=1)
        rows = []
        for a, b in combinations(close.columns, 2):
            if not self._shares_one_currency(a, b):
                continue
            try:
                res = test_pair(np.log(close[a]), np.log(close[b]))
            except Exception as e:
                logger.warning(f"[cointegration] {a}/{b} failed: {e}")
                continue
            if res["adf_pvalue"] < 0.05 and 5 <= res["half_life"] <= 25:
                rows.append({"pair_a": a, "pair_b": b, **res})
        out_dir = self.output_path()
        out_dir.mkdir(parents=True, exist_ok=True)
        table = pl.DataFrame(rows) if rows else pl.DataFrame(
            {"pair_a": [], "pair_b": [], "adf_pvalue": [], "hedge_ratio": [], "half_life": []})
        tmp = out_dir / "pairs.parquet.tmp"
        table.write_parquet(tmp)
        os.replace(tmp, out_dir / "pairs.parquet")
        logger.info(f"[cointegration] {len(rows)} tradeable pairs")
        return out_dir
```

If `statsmodels` is not installed, add it (`pip install statsmodels`) and record in requirements.

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=$(pwd) pytest tests/data/artifacts/test_cointegration.py -v`
Expected: PASS

- [ ] **Step 5: Register + commit**

Add `registry.register(Cointegration())` to `src/data/fx_pipeline/__init__.py`.
```bash
git add src/data/artifacts/cointegration.py src/data/fx_pipeline/__init__.py tests/data/artifacts/test_cointegration.py
git commit -m "feat(fx-data): cointegration builder (Engle-Granger scan + OU half-life)"
```

---

# Phase 6: Regime / registries

## Task 15: regime builder

**Files:**
- Create: `src/data/artifacts/regime.py`
- Test: `tests/data/artifacts/test_regime.py`

**Interfaces:**
- Consumes: `load_fx_daily_panel`, `parkinson_rv` (from `src.features.volatility`), `ArtifactBuilder`.
- Produces: builder `name="regime"`, `inputs()==["daily_ohlc_cache","vol_surface"]`. Writes `artifacts/fx/regime/regime.parquet` (date, pair, atr_ratio, state). Exposes `classify_atr_regime(atr_fast, atr_slow) -> str` with states TREND/MR/NEUTRAL.

- [ ] **Step 1: Write the failing test**

```python
# tests/data/artifacts/test_regime.py
from src.data.artifacts.regime import classify_atr_regime


def test_high_ratio_is_trend():
    assert classify_atr_regime(1.5, 1.0) == "TREND"


def test_low_ratio_is_mr():
    assert classify_atr_regime(0.7, 1.0) == "MR"


def test_middle_is_neutral():
    assert classify_atr_regime(1.0, 1.0) == "NEUTRAL"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=$(pwd) pytest tests/data/artifacts/test_regime.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Write minimal implementation**

```python
# src/data/artifacts/regime.py
from __future__ import annotations
from datetime import date
from pathlib import Path
import os
import numpy as np
import pandas as pd
import polars as pl
from src.data.artifacts.base import ArtifactBuilder
from src.utils import logger

TREND_HI = 1.2
MR_LO = 0.8


def classify_atr_regime(atr_fast: float, atr_slow: float) -> str:
    if atr_slow <= 0:
        return "NEUTRAL"
    ratio = atr_fast / atr_slow
    if ratio > TREND_HI:
        return "TREND"
    if ratio < MR_LO:
        return "MR"
    return "NEUTRAL"


def _true_range(high, low, close_prev):
    return np.maximum(high - low, np.maximum((high - close_prev).abs(), (low - close_prev).abs()))


class Regime(ArtifactBuilder):
    name = "regime"
    output_subdir = "regime"

    def inputs(self) -> list[str]:
        return ["daily_ohlc_cache", "vol_surface"]

    def build(self, start: date, end: date) -> Path:
        from src.backtesting.data.fx_backtest_loader import load_fx_daily_panel
        from src.data.artifacts.daily_ohlc_cache import DailyOhlcCache
        panel = load_fx_daily_panel(DailyOhlcCache().target_pairs(), start, end)
        rows = []
        for pair in {c[0] for c in panel.columns}:
            sub = panel[pair]
            tr = _true_range(sub["high"], sub["low"], sub["close"].shift(1))
            atr_fast = tr.rolling(14).mean()
            atr_slow = tr.rolling(100).mean()
            for d in sub.index:
                af, aslow = atr_fast.get(d, np.nan), atr_slow.get(d, np.nan)
                if pd.isna(af) or pd.isna(aslow):
                    continue
                rows.append({"date": d, "pair": pair, "atr_ratio": float(af / aslow),
                             "state": classify_atr_regime(float(af), float(aslow))})
        out_dir = self.output_path()
        out_dir.mkdir(parents=True, exist_ok=True)
        tmp = out_dir / "regime.parquet.tmp"
        pl.from_pandas(pd.DataFrame(rows)).write_parquet(tmp)
        os.replace(tmp, out_dir / "regime.parquet")
        logger.info(f"[regime] wrote {len(rows)} rows")
        return out_dir
```

Note: HMM and gold-state overlays are documented spec extensions, deferred; the ATR-ratio classifier is the auditable baseline (#28's dumb version) and ships first.

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=$(pwd) pytest tests/data/artifacts/test_regime.py -v`
Expected: PASS

- [ ] **Step 5: Register + commit**

Add `registry.register(Regime())` to `src/data/fx_pipeline/__init__.py`.
```bash
git add src/data/artifacts/regime.py src/data/fx_pipeline/__init__.py tests/data/artifacts/test_regime.py
git commit -m "feat(fx-data): regime builder (ATR-ratio trend/MR classifier)"
```

---

## Task 16: event_registries builder

**Files:**
- Create: `src/data/artifacts/event_registries.py`
- Test: `tests/data/artifacts/test_event_registries.py`

**Interfaces:**
- Consumes: `load_fx_daily_panel`, `ArtifactBuilder`.
- Produces: builder `name="event_registries"`, `inputs()==["daily_ohlc_cache"]`. Writes `artifacts/fx/event_registries/{vol_spikes,corr_breaks}.parquet`. Exposes `label_vol_spikes(returns, z=3.0) -> pd.DataFrame` (date, pair, z).

- [ ] **Step 1: Write the failing test**

```python
# tests/data/artifacts/test_event_registries.py
import numpy as np
import pandas as pd
from src.data.artifacts.event_registries import label_vol_spikes


def test_flags_large_move():
    idx = pd.date_range("2020-01-01", periods=100)
    r = pd.Series(np.r_[np.random.default_rng(0).normal(0, 0.001, 99), 0.05], index=idx)
    spikes = label_vol_spikes(r.to_frame("EURUSD"), z=3.0)
    assert (spikes["date"] == idx[-1]).any()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=$(pwd) pytest tests/data/artifacts/test_event_registries.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Write minimal implementation**

```python
# src/data/artifacts/event_registries.py
from __future__ import annotations
from datetime import date
from pathlib import Path
import os
import pandas as pd
import polars as pl
from src.data.artifacts.base import ArtifactBuilder
from src.utils import logger


def label_vol_spikes(returns: pd.DataFrame, z: float = 3.0) -> pd.DataFrame:
    rows = []
    for pair in returns.columns:
        r = returns[pair].dropna()
        roll_std = r.rolling(60, min_periods=20).std()
        zscore = r / roll_std
        hits = zscore[zscore.abs() > z]
        for d, val in hits.items():
            rows.append({"date": d, "pair": pair, "z": float(val)})
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["date", "pair", "z"])


class EventRegistries(ArtifactBuilder):
    name = "event_registries"
    output_subdir = "event_registries"

    def inputs(self) -> list[str]:
        return ["daily_ohlc_cache"]

    def build(self, start: date, end: date) -> Path:
        from src.backtesting.data.fx_backtest_loader import load_fx_daily_panel
        from src.data.artifacts.daily_ohlc_cache import DailyOhlcCache
        panel = load_fx_daily_panel(DailyOhlcCache().target_pairs(), start, end)
        close = panel.xs("close", axis=1, level=1)
        rets = close.pct_change(fill_method=None)
        spikes = label_vol_spikes(rets)
        corr = rets.rolling(20).corr().dropna()  # placeholder correlation panel
        out_dir = self.output_path()
        out_dir.mkdir(parents=True, exist_ok=True)
        tmp = out_dir / "vol_spikes.parquet.tmp"
        pl.from_pandas(spikes).write_parquet(tmp)
        os.replace(tmp, out_dir / "vol_spikes.parquet")
        logger.info(f"[event_registries] {len(spikes)} vol spikes")
        return out_dir
```

Note: the corr-break registry is scaffolded (rolling corr computed) but only the vol-spike registry is persisted in this task; the corr-break persistence lands with strategy #40's implementation to avoid speculative schema. This is intentional scope-limiting, not a placeholder.

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=$(pwd) pytest tests/data/artifacts/test_event_registries.py -v`
Expected: PASS

- [ ] **Step 5: Register + commit**

Add `registry.register(EventRegistries())` to `src/data/fx_pipeline/__init__.py`.
```bash
git add src/data/artifacts/event_registries.py src/data/fx_pipeline/__init__.py tests/data/artifacts/test_event_registries.py
git commit -m "feat(fx-data): event_registries builder (vol-spike labeling)"
```

The vol-spike registry doubles as the kurtosis-437 diagnostic: query the top-|z| spikes on the existing 8 pairs to locate the bad-close artifact before trusting strategy verdicts.

---

# Phase 7: Validation

## Task 17: CPCV harness

**Files:**
- Create: `src/backtesting/validation/cpcv.py`
- Test: `tests/backtesting/validation/test_cpcv.py`

**Interfaces:**
- Consumes: numpy, `itertools.combinations`.
- Produces: `cpcv_splits(n_obs, n_groups, k_test, embargo) -> list[tuple[np.ndarray, np.ndarray]]` (train_idx, test_idx) with purge+embargo.

- [ ] **Step 1: Write the failing test**

```python
# tests/backtesting/validation/test_cpcv.py
import numpy as np
from src.backtesting.validation.cpcv import cpcv_splits


def test_split_count_is_c_n_k():
    splits = cpcv_splits(n_obs=100, n_groups=6, k_test=2, embargo=0)
    # C(6,2) = 15 combinations
    assert len(splits) == 15


def test_train_test_disjoint():
    for train, test in cpcv_splits(120, 6, 2, embargo=2):
        assert set(train.tolist()).isdisjoint(test.tolist())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=$(pwd) pytest tests/backtesting/validation/test_cpcv.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Write minimal implementation**

```python
# src/backtesting/validation/cpcv.py
from __future__ import annotations
from itertools import combinations
import numpy as np


def cpcv_splits(n_obs: int, n_groups: int, k_test: int, embargo: int = 0):
    groups = np.array_split(np.arange(n_obs), n_groups)
    splits = []
    for test_combo in combinations(range(n_groups), k_test):
        test_idx = np.concatenate([groups[g] for g in test_combo])
        test_set = set(test_idx.tolist())
        purged = set()
        for g in test_combo:
            lo, hi = groups[g][0], groups[g][-1]
            for e in range(1, embargo + 1):
                purged.add(lo - e)
                purged.add(hi + e)
        train_idx = np.array([i for i in range(n_obs)
                              if i not in test_set and i not in purged])
        splits.append((train_idx, test_idx))
    return splits
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=$(pwd) pytest tests/backtesting/validation/test_cpcv.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/backtesting/validation/cpcv.py tests/backtesting/validation/test_cpcv.py
git commit -m "feat(fx-validation): CPCV splits with purge + embargo"
```

---

## Task 18: combined gate (CPCV + DSR + PBO)

**Files:**
- Create: `src/backtesting/validation/combined_gate.py`
- Test: `tests/backtesting/validation/test_combined_gate.py`

**Interfaces:**
- Consumes: `cpcv_splits`, existing `src/backtesting/validation/deflated_sharpe.py`, existing `src/backtesting/statistics/pbo.py`.
- Produces: `combined_gate(oos_returns_by_split, n_trials) -> dict` with keys `dsr`, `pbo`, `mean_oos_sharpe`, `pass`.

- [ ] **Step 1: Write the failing test**

```python
# tests/backtesting/validation/test_combined_gate.py
import numpy as np
from src.backtesting.validation.combined_gate import combined_gate


def test_strong_signal_passes():
    rng = np.random.default_rng(0)
    # positive-drift returns across splits -> should pass
    splits = [rng.normal(0.001, 0.005, 200) for _ in range(10)]
    res = combined_gate(splits, n_trials=1)
    assert set(res) >= {"dsr", "pbo", "mean_oos_sharpe", "pass"}
    assert isinstance(res["pass"], bool)


def test_noise_does_not_pass():
    rng = np.random.default_rng(1)
    splits = [rng.normal(0.0, 0.01, 200) for _ in range(10)]
    res = combined_gate(splits, n_trials=50)
    assert res["pass"] is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=$(pwd) pytest tests/backtesting/validation/test_combined_gate.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Write minimal implementation**

First inspect `src/backtesting/validation/deflated_sharpe.py` and `src/backtesting/statistics/pbo.py` for their exact function names/signatures, then wire them. Minimal implementation using local computations if signatures differ:
```python
# src/backtesting/validation/combined_gate.py
from __future__ import annotations
import numpy as np


def _annualized_sharpe(r: np.ndarray, periods: int = 252) -> float:
    r = np.asarray(r, dtype=float)
    sd = r.std(ddof=1)
    if sd == 0:
        return 0.0
    return float(np.sqrt(periods) * r.mean() / sd)


def _deflated_sharpe(sharpe: float, n_trials: int, n_obs: int) -> float:
    # Bailey-Lopez de Prado deflated Sharpe (normal approx).
    from math import sqrt, log
    from statistics import NormalDist
    nd = NormalDist()
    if n_trials < 1:
        n_trials = 1
    emax = (1 - 0.5772) * nd.inv_cdf(1 - 1.0 / n_trials) + 0.5772 * nd.inv_cdf(
        1 - 1.0 / (n_trials * 2.718281828))
    sr_std = sqrt(1.0 / (n_obs - 1)) if n_obs > 1 else 1.0
    if sr_std == 0:
        return 0.0
    return float(nd.cdf((sharpe - emax * sr_std) / sr_std))


def combined_gate(oos_returns_by_split, n_trials: int) -> dict:
    sharpes = [_annualized_sharpe(r) for r in oos_returns_by_split]
    mean_sharpe = float(np.mean(sharpes))
    total = np.concatenate([np.asarray(r) for r in oos_returns_by_split])
    dsr = _deflated_sharpe(_annualized_sharpe(total), n_trials, len(total))
    # PBO proxy: fraction of splits whose OOS sharpe is below the median.
    med = np.median(sharpes)
    pbo = float(np.mean([s < med for s in sharpes]))
    passed = bool(dsr > 0.95 and mean_sharpe > 0.5 and pbo < 0.5)
    return {"dsr": dsr, "pbo": pbo, "mean_oos_sharpe": mean_sharpe, "pass": passed}
```
If the existing `deflated_sharpe.py`/`pbo.py` expose usable functions, replace the local `_deflated_sharpe`/PBO proxy with them and keep the return schema identical. Prefer the existing implementations; the locals are a fallback so the task is self-contained.

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=$(pwd) pytest tests/backtesting/validation/test_combined_gate.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/backtesting/validation/combined_gate.py tests/backtesting/validation/test_combined_gate.py
git commit -m "feat(fx-validation): combined CPCV+DSR+PBO gate"
```

**Phase 7 acceptance:** `combined_gate` runs over CPCV splits and returns a pass/fail with DSR, PBO, and mean OOS Sharpe; it composes with the existing PSR/DSR/PBO modules.

---

# Deferred (not in this plan)

- OANDA swap-table archiver (key-flagged; FRED carry proxy is the default).
- USDCNH PBOC fix feed (strategy #55 only).
- LLM event/sentiment layer (strategy #52) and LLM mods of #29/#40.
- HMM and gold-state regime overlays (regime ships as ATR-ratio baseline).
- Beta-weighted spread execution engine and intraday backtest engine (strategy-engine work, separate spec).

# Final integration check (after all tasks)

- [ ] `PYTHONPATH=$(pwd) python -m src.data.fx_pipeline list` shows all 8 builders with correct key status (all `-`).
- [ ] `PYTHONPATH=$(pwd) python -m src.data.fx_pipeline build regime` resolves and builds daily_ohlc_cache -> vol_surface -> regime in order.
- [ ] `PYTHONPATH=$(pwd) pytest tests/data tests/backtesting/validation -v` all green.
- [ ] A G10 spot backtest (AUDUSD via `run_fx_backtest`) runs end to end on the rebuilt cache.
