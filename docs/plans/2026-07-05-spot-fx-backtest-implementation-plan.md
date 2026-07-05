# Spot FX Backtesting Platform Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reusable spot-FX daily backtesting vertical (loader, USD-conversion + rate panels, carry-accruing simulator, notional sizing, pip/bps costs, walk-forward gate) with trend + value reference strategies, routed by `asset_class: fx`.

**Architecture:** A dedicated asset-class path structurally parallel to the futures vertical, routed off the raw YAML `asset_class` key in `src/backtest_runner.py`. Reuses all asset-agnostic infrastructure (walk-forward harness shape, PSR/DSR/PBO gate, volatility features, Carver/Asness forecast logic, StandardReportGenerator, experiment registry, RunStatus, parallel_map). Adds only FX-native pieces: 17:00-ET daily bars, a per-currency USD-conversion panel, daily carry accrual, notional vol-target sizing, and a pip/bps cost model. The futures path is not modified.

**Tech Stack:** Python 3 (conda `fintech` env), pandas, polars, numpy, pyarrow, pytest. FRED data already on disk under `alt_data/fred/`; spot FX minute data under `fx_1min/`.

## Global Constraints

- All Python runs in the conda `fintech` environment.
- ASCII-only in all code, comments, logs, and docs (no emojis, no Unicode arrows/symbols).
- Never use `print()` for output; use `from src.utils import logger`. The Homeguard logger does NOT support `%s` positional args -- use f-strings.
- Never hardcode storage paths; use `from src.settings import get_local_storage_dir`.
- The futures vertical (`futures_backtest.py`, `futures_portfolio_simulator.py`, `futures_backtest_loader.py`, `position_sizer_futures.py`) MUST remain functionally unchanged. The only shared file edited is `src/backtesting/utils/idm_weights.py`, and that edit MUST be backward-compatible (default arg preserves exact existing behavior; existing futures tests must still pass).
- TDD: write the failing test first, watch it fail, implement minimal code, watch it pass, commit. One logical unit per commit.
- Symbol convention: every FX symbol is 6 chars, `base = sym[:3]`, `quote = sym[3:]` (e.g. `EURUSD` -> base EUR quote USD; `USDJPY` -> base USD quote JPY; `XAUUSD` -> base XAU quote USD).
- Carry sign convention: long the base currency earns `+(r_base - r_quote)` on USD notional.
- Trade-log persistence is mandatory for the representative run (methodology Section 12); the walk-forward per-window runs suppress it.

---

## File Structure

**New files:**
- `scripts/data/build_fx_daily_cache.py` -- resample `fx_1min/` to 17:00-ET daily bars, cache to `fx_daily/`. Contains the pure, testable `resample_fx_minute_to_daily`.
- `src/backtesting/data/fx_backtest_loader.py` -- `load_fx_daily_panel`, `build_quote_usd_panel`.
- `src/data/fx/__init__.py` -- package marker.
- `src/data/fx/clusters.py` -- `fx_cluster_for(pair)` for IDM.
- `src/data/fx_rates.py` -- `CURRENCY_FRED_SERIES`, `load_fx_rate_panel`, `build_rate_diff_panel`.
- `src/backtesting/utils/position_sizer_fx.py` -- `size_from_forecast_fx`.
- `src/backtesting/engine/fx_spot_portfolio_simulator.py` -- `FxSpotPortfolioSimulator`, `FxBacktestResult`.
- `src/backtesting/engine/fx_backtest.py` -- `run_fx_backtest`.
- `src/strategies/advanced/fx_strategies.py` -- `FxTrendStrategy`, `FxValueStrategy`.
- `src/backtesting/walkforward_common.py` -- pure walk-forward helpers (window building, OOS slicing, annualized Sharpe, PBO, verdict) shared by the futures and FX walk-forward scripts.
- `scripts/backtest_scripts/run_fx_walkforward.py` -- `walk_forward_fx`, report writer (imports helpers from `walkforward_common`).
- `config/backtesting/fx_trend.yaml`, `config/backtesting/fx_value.yaml`
- `config/universes/fx_spot-2026.csv`
- Tests mirroring each of the above under `tests/`.

**Modified files (additive):**
- `src/backtesting/costs/fx.py` -- add `fx_round_trip_usd` (keep `fx_round_trip_pips`).
- `src/backtesting/utils/idm_weights.py` -- add optional `cluster_fn` param.
- `src/strategies/registry.py` -- register `FxTrend`, `FxValue`.
- `src/backtest_runner.py` -- add `asset_class == 'fx'` routing branch.
- `scripts/backtest_scripts/run_carver_walkforward.py` -- move its pure helpers to `walkforward_common.py` and import them back (Task 10; futures walk-forward test is the regression gate).

---

# Milestone 1 -- Data layer

### Task 1: FX daily cache builder (17:00-ET resample)

**Files:**
- Create: `scripts/data/build_fx_daily_cache.py`
- Test: `tests/data/test_build_fx_daily_cache.py`

**Interfaces:**
- Produces: `resample_fx_minute_to_daily(df_min: pd.DataFrame) -> pd.DataFrame` where `df_min` has columns `timestamp` (tz-aware UTC datetime) and `close` (float); returns a DataFrame indexed by `fx_date` (python `date`) with a single `close` column = last minute close of each FX trading day (day boundary 17:00 America/New_York). Also `build_fx_daily_cache(pairs: list[str], start: date, end: date, src_root: Path | None = None, out_root: Path | None = None) -> list[str]` writing `fx_daily/symbol={SYM}/year={YYYY}/month={M}/data.parquet`, returning the list of pairs written.

- [ ] **Step 1: Write the failing test**

```python
# tests/data/test_build_fx_daily_cache.py
import pandas as pd
from scripts.data.build_fx_daily_cache import resample_fx_minute_to_daily


def _ts(*parts):
    # UTC timestamp helper
    return pd.Timestamp(*parts, tz="UTC")


def test_1700_et_boundary_splits_days():
    # 2024-06-03 is a Monday. 17:00 ET = 21:00 UTC (EDT, UTC-4).
    # A bar at 20:59 UTC (16:59 ET Mon) belongs to FX-day Monday.
    # A bar at 21:01 UTC (17:01 ET Mon) belongs to FX-day Tuesday.
    df = pd.DataFrame(
        {
            "timestamp": [
                _ts("2024-06-03 20:58", tz="UTC"),
                _ts("2024-06-03 20:59", tz="UTC"),
                _ts("2024-06-03 21:01", tz="UTC"),
            ],
            "close": [1.10, 1.11, 1.12],
        }
    )
    daily = resample_fx_minute_to_daily(df)
    import datetime as dt

    assert daily.loc[dt.date(2024, 6, 3), "close"] == 1.11  # last before 17:00 ET
    assert daily.loc[dt.date(2024, 6, 4), "close"] == 1.12  # rolled into Tuesday
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n fintech pytest tests/data/test_build_fx_daily_cache.py -v`
Expected: FAIL with `ModuleNotFoundError` / `ImportError` (function not defined).

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/data/build_fx_daily_cache.py
"""Resample fx_1min/ to 17:00-ET-anchored daily bars, cached to fx_daily/.

FX trades 24/5; the market-convention day boundary is 17:00 America/New_York
(Sunday 17:00 -> Friday 17:00). Each minute is assigned to the FX trading day
whose (prev-day 17:00, this-day 17:00] window contains it; the daily close is
the last minute close inside that window. A +7h wall-clock shift after tz
conversion (24 - 17 = 7) maps each ET timestamp onto its FX date.

DST is handled by the America/New_York tz conversion; the +7h shift is applied
in local wall-clock time, so the boundary tracks 17:00 ET across DST changes.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import polars as pl

from src.settings import get_local_storage_dir
from src.utils import logger


def resample_fx_minute_to_daily(df_min: pd.DataFrame) -> pd.DataFrame:
    if df_min.empty:
        return pd.DataFrame(columns=["close"])
    ts_et = df_min["timestamp"].dt.tz_convert("America/New_York")
    fx_date = (ts_et + pd.Timedelta(hours=7)).dt.date
    tmp = df_min.assign(fx_date=fx_date).sort_values("timestamp")
    daily = tmp.groupby("fx_date").agg(close=("close", "last"))
    daily.index.name = "fx_date"
    return daily


def build_fx_daily_cache(pairs: list[str], start: date, end: date,
                         src_root: Path | None = None,
                         out_root: Path | None = None) -> list[str]:
    base = Path(get_local_storage_dir())
    src_root = src_root or (base / "fx_1min")
    out_root = out_root or (base / "fx_daily")
    written: list[str] = []
    for pair in pairs:
        sym_dir = src_root / f"symbol={pair}"
        if not sym_dir.exists():
            logger.warning(f"[build_fx_daily_cache] no source data for {pair}")
            continue
        lf = pl.scan_parquet(sym_dir / "**/*.parquet").select(["timestamp", "close"])
        pdf = lf.collect().to_pandas()
        pdf["timestamp"] = pd.to_datetime(pdf["timestamp"], utc=True)
        pdf = pdf[(pdf["timestamp"].dt.date >= start) & (pdf["timestamp"].dt.date <= end)]
        daily = resample_fx_minute_to_daily(pdf)
        if daily.empty:
            continue
        out = daily.reset_index()
        out["year"] = pd.to_datetime(out["fx_date"]).dt.year
        out["month"] = pd.to_datetime(out["fx_date"]).dt.month
        for (yr, mo), grp in out.groupby(["year", "month"]):
            dst = out_root / f"symbol={pair}" / f"year={yr}" / f"month={mo}"
            dst.mkdir(parents=True, exist_ok=True)
            pl.from_pandas(grp[["fx_date", "close"]]).write_parquet(dst / "data.parquet")
        written.append(pair)
        logger.info(f"[build_fx_daily_cache] wrote {pair}: {len(daily)} daily bars")
    return written


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Build fx_daily/ cache from fx_1min/")
    p.add_argument("--csv", required=True, help="Universe CSV with a 'symbol' column")
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    args = p.parse_args()
    pairs = pd.read_csv(args.csv)["symbol"].tolist()
    written = build_fx_daily_cache(
        pairs, date.fromisoformat(args.start), date.fromisoformat(args.end))
    logger.success(f"[build_fx_daily_cache] wrote {len(written)} pairs")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n fintech pytest tests/data/test_build_fx_daily_cache.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/data/build_fx_daily_cache.py tests/data/test_build_fx_daily_cache.py
git commit -m "feat(fx): 17:00-ET daily cache builder for spot FX"
```

---

### Task 2: FX daily loader + USD-conversion panel

**Files:**
- Create: `src/backtesting/data/fx_backtest_loader.py`
- Test: `tests/backtesting/data/test_fx_backtest_loader.py`

**Interfaces:**
- Consumes: `fx_daily/` parquet layout from Task 1 (`symbol={SYM}/.../data.parquet` with columns `fx_date`, `close`).
- Produces:
  - `load_fx_daily_panel(pairs: list[str], start: date, end: date) -> pd.DataFrame` -- MultiIndex columns `(pair, field)` with `field in {"close", "ret"}`, index = FX dates. Silently excludes (WARNING) a pair with no data in-window; raises `FileNotFoundError` only if NO pair has data.
  - `build_quote_usd_panel(close_panel: pd.DataFrame, pairs: list[str]) -> pd.DataFrame` -- columns = pairs, values = the quote-currency -> USD rate for that pair on each date. Raises `ValueError` if a required USD leg is missing.

- [ ] **Step 1: Write the failing test**

```python
# tests/backtesting/data/test_fx_backtest_loader.py
import datetime as dt

import numpy as np
import pandas as pd
import pytest

from src.backtesting.data.fx_backtest_loader import build_quote_usd_panel


def _close_panel(data: dict) -> pd.DataFrame:
    idx = pd.Index([dt.date(2024, 1, 2), dt.date(2024, 1, 3)], name="fx_date")
    frames = {}
    for pair, closes in data.items():
        frames[(pair, "close")] = pd.Series(closes, index=idx)
        frames[(pair, "ret")] = pd.Series(closes, index=idx).pct_change()
    df = pd.DataFrame(frames)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df


def test_usd_quote_pair_is_identity():
    panel = _close_panel({"EURUSD": [1.10, 1.11]})
    q = build_quote_usd_panel(panel, ["EURUSD"])
    # quote is USD -> USD->USD = 1.0
    assert (q["EURUSD"] == 1.0).all()


def test_usd_base_pair_inverts():
    panel = _close_panel({"USDJPY": [150.0, 160.0]})
    q = build_quote_usd_panel(panel, ["USDJPY"])
    # quote is JPY -> JPY->USD = 1/USDJPY
    assert q["USDJPY"].iloc[0] == pytest.approx(1 / 150.0)
    assert q["USDJPY"].iloc[1] == pytest.approx(1 / 160.0)


def test_true_cross_uses_usd_leg():
    # EURGBP: quote GBP -> USD via GBPUSD
    panel = _close_panel({"EURGBP": [0.85, 0.86], "GBPUSD": [1.25, 1.26]})
    q = build_quote_usd_panel(panel, ["EURGBP", "GBPUSD"])
    assert q["EURGBP"].iloc[0] == pytest.approx(1.25)
    assert q["EURGBP"].iloc[1] == pytest.approx(1.26)


def test_missing_usd_leg_raises():
    panel = _close_panel({"EURGBP": [0.85, 0.86]})  # no GBPUSD, no USDGBP
    with pytest.raises(ValueError):
        build_quote_usd_panel(panel, ["EURGBP"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n fintech pytest tests/backtesting/data/test_fx_backtest_loader.py -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/backtesting/data/fx_backtest_loader.py
"""Daily spot-FX panel + per-currency USD-conversion panel.

load_fx_daily_panel builds a (pair, {close,ret}) MultiIndex daily panel from the
fx_daily/ cache. build_quote_usd_panel derives, for each pair, the daily rate
that converts its QUOTE currency into USD -- USD legs are read directly
(EURUSD), inverted (USDJPY -> 1/rate), or sourced from another pair's USD leg
for true crosses (EURGBP -> GBPUSD). A missing USD leg is a hard error: silent
mis-conversion is never acceptable.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import polars as pl

from src.settings import get_local_storage_dir
from src.utils import logger


def load_fx_daily_panel(pairs: list[str], start: date, end: date) -> pd.DataFrame:
    base = Path(get_local_storage_dir()) / "fx_daily"
    frames: dict[str, pd.Series] = {}
    for pair in pairs:
        sym_dir = base / f"symbol={pair}"
        if not sym_dir.exists():
            logger.warning(f"[load_fx_daily_panel] no fx_daily data for {pair}")
            continue
        pdf = pl.scan_parquet(sym_dir / "**/*.parquet").collect().to_pandas()
        pdf["fx_date"] = pd.to_datetime(pdf["fx_date"]).dt.date
        pdf = pdf[(pdf["fx_date"] >= start) & (pdf["fx_date"] <= end)]
        if pdf.empty:
            continue
        s = pdf.set_index("fx_date").sort_index()["close"].astype(float)
        frames[pair] = s
    if not frames:
        raise FileNotFoundError(f"no fx_daily data for pairs {pairs} in {start}..{end}")
    close = pd.DataFrame(frames).sort_index()
    ret = close.pct_change(fill_method=None)
    panel = pd.concat(
        {p: pd.DataFrame({"close": close[p], "ret": ret[p]}) for p in close.columns}, axis=1)
    panel.columns = pd.MultiIndex.from_tuples(
        [(p, f) for p in close.columns for f in ("close", "ret")])
    return panel


def _currency_to_usd(currency: str, close_panel: pd.DataFrame) -> pd.Series:
    if currency == "USD":
        idx = close_panel.index
        return pd.Series(1.0, index=idx)
    pairs = {c[0] for c in close_panel.columns}
    direct = f"{currency}USD"
    inverse = f"USD{currency}"
    if direct in pairs:
        return close_panel[(direct, "close")].astype(float)
    if inverse in pairs:
        return 1.0 / close_panel[(inverse, "close")].astype(float)
    raise ValueError(
        f"cannot convert {currency} to USD: neither {direct} nor {inverse} in panel")


def build_quote_usd_panel(close_panel: pd.DataFrame, pairs: list[str]) -> pd.DataFrame:
    out: dict[str, pd.Series] = {}
    for pair in pairs:
        quote = pair[3:]
        out[pair] = _currency_to_usd(quote, close_panel)
    return pd.DataFrame(out)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n fintech pytest tests/backtesting/data/test_fx_backtest_loader.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/backtesting/data/fx_backtest_loader.py tests/backtesting/data/test_fx_backtest_loader.py
git commit -m "feat(fx): daily panel loader + quote-currency USD-conversion panel"
```

---

### Task 3: FRED rate panel + rate-differential panel

**Files:**
- Create: `src/data/fx/__init__.py` (empty package marker)
- Create: `src/data/fx_rates.py`
- Test: `tests/data/test_fx_rates.py`

**Interfaces:**
- Consumes: `alt_data/fred/{series_id}/daily.parquet` (columns `date`, `value`; `value` is a percent, e.g. `5.33` meaning 5.33%).
- Produces:
  - `CURRENCY_FRED_SERIES: dict[str, str]` mapping currency -> FRED short-rate series id.
  - `load_fx_rate_panel(currencies: list[str], index: pd.Index) -> pd.DataFrame` -- columns = currencies, values = decimal annual rate (e.g. 0.0533), forward-filled and reindexed to `index` (the panel's FX dates). Metals (XAU, XAG) map to a constant 0.0.
  - `build_rate_diff_panel(pairs: list[str], rate_panel: pd.DataFrame) -> pd.DataFrame` -- columns = pairs, values = `r_base - r_quote`.

- [ ] **Step 1: Write the failing test**

```python
# tests/data/test_fx_rates.py
import datetime as dt

import pandas as pd
import pytest

from src.data.fx_rates import build_rate_diff_panel


def test_rate_diff_base_minus_quote():
    idx = pd.Index([dt.date(2024, 1, 2), dt.date(2024, 1, 3)])
    rates = pd.DataFrame(
        {"EUR": [0.04, 0.04], "USD": [0.053, 0.053], "JPY": [0.001, 0.001]}, index=idx)
    diff = build_rate_diff_panel(["EURUSD", "USDJPY"], rates)
    # EURUSD: r_EUR - r_USD = 0.04 - 0.053
    assert diff["EURUSD"].iloc[0] == pytest.approx(0.04 - 0.053)
    # USDJPY: r_USD - r_JPY = 0.053 - 0.001
    assert diff["USDJPY"].iloc[0] == pytest.approx(0.053 - 0.001)


def test_metals_base_rate_zero():
    idx = pd.Index([dt.date(2024, 1, 2)])
    rates = pd.DataFrame({"XAU": [0.0], "USD": [0.053]}, index=idx)
    diff = build_rate_diff_panel(["XAUUSD"], rates)
    # gold carry = 0 - r_USD (pure USD funding)
    assert diff["XAUUSD"].iloc[0] == pytest.approx(-0.053)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n fintech pytest tests/data/test_fx_rates.py -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/data/fx/__init__.py
```

```python
# src/data/fx_rates.py
"""Currency short-rate panel + FX carry rate differentials from FRED.

Carry accrual on spot FX is the overnight interest-rate differential
(r_base - r_quote). This module maps each currency to a FRED short-rate series
(policy or short-tenor bill rate), builds a daily decimal-rate panel aligned to
the backtest's FX dates, and computes per-pair rate differentials. Metals
(XAU/XAG) have no interest rate -> base rate 0.0, so gold carry is pure USD
funding.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.settings import get_local_storage_dir
from src.utils import logger

# Currency -> FRED daily short-rate series id (percent units in FRED).
# v1 covers ONLY the currencies whose rate series are on disk in alt_data/fred/
# (verified against config/universes/fred_series-2026.csv). GBP/CAD/AUD/NZD have
# no short-rate series downloaded, so the v1 universe is restricted to
# USD/EUR/CHF/JPY pairs + metals -- see the universe CSV in Task 9. Any currency
# absent from this map falls back to 0.0 with a WARNING (graceful, not fatal).
CURRENCY_FRED_SERIES: dict[str, str] = {
    "USD": "DFF",              # Effective Federal Funds Rate
    "EUR": "ECBDFR",           # ECB Deposit Facility Rate
    "CHF": "IRSTCB01CHM156N",  # Switzerland short-term rate
    "JPY": "IRLTLT01JPM156N",  # Japan LONG-term rate used as a proxy: no short
                               # series is on disk; JP rates were ~0 across the
                               # sample so the magnitude error is small. Documented
                               # caveat, not an oversight.
}
_METALS = {"XAU", "XAG"}


def load_fx_rate_panel(currencies: list[str], index: pd.Index) -> pd.DataFrame:
    base = Path(get_local_storage_dir()) / "alt_data" / "fred"
    out: dict[str, pd.Series] = {}
    idx_dt = pd.to_datetime(pd.Index(index))
    for ccy in currencies:
        if ccy in _METALS:
            out[ccy] = pd.Series(0.0, index=index)
            continue
        series_id = CURRENCY_FRED_SERIES.get(ccy)
        if series_id is None:
            logger.warning(f"[load_fx_rate_panel] no FRED series for {ccy}; rate=0")
            out[ccy] = pd.Series(0.0, index=index)
            continue
        fp = base / series_id / "daily.parquet"
        if not fp.exists():
            logger.warning(f"[load_fx_rate_panel] FRED file missing for {ccy} ({series_id}); rate=0")
            out[ccy] = pd.Series(0.0, index=index)
            continue
        raw = pd.read_parquet(fp)
        s = pd.Series(raw["value"].values, index=pd.to_datetime(raw["date"].values)) / 100.0
        s = s.sort_index().reindex(idx_dt.union(s.index)).ffill().reindex(idx_dt)
        s.index = index
        out[ccy] = s
    return pd.DataFrame(out)


def build_rate_diff_panel(pairs: list[str], rate_panel: pd.DataFrame) -> pd.DataFrame:
    out: dict[str, pd.Series] = {}
    for pair in pairs:
        base_ccy, quote_ccy = pair[:3], pair[3:]
        out[pair] = rate_panel[base_ccy] - rate_panel[quote_ccy]
    return pd.DataFrame(out)


def currencies_for_pairs(pairs: list[str]) -> list[str]:
    ccys: set[str] = set()
    for pair in pairs:
        ccys.add(pair[:3])
        ccys.add(pair[3:])
    return sorted(ccys)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n fintech pytest tests/data/test_fx_rates.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/data/fx/__init__.py src/data/fx_rates.py tests/data/test_fx_rates.py
git commit -m "feat(fx): FRED currency rate panel + carry rate differentials"
```

---

# Milestone 2 -- Simulator, sizing, costs

### Task 4: FX cluster map + generalize compute_div_mult

**Files:**
- Create: `src/data/fx/clusters.py`
- Modify: `src/backtesting/utils/idm_weights.py`
- Test: `tests/backtesting/utils/test_idm_weights_fx.py`

**Interfaces:**
- Consumes: `compute_div_mult` from `src/backtesting/utils/idm_weights.py`.
- Produces:
  - `fx_cluster_for(pair: str) -> str` in `src/data/fx/clusters.py`.
  - `compute_div_mult(universe, per_instrument_cap=None, cluster_fn=cluster_for)` -- new optional `cluster_fn` param; default preserves exact existing (futures) behavior.

- [ ] **Step 1: Write the failing test**

```python
# tests/backtesting/utils/test_idm_weights_fx.py
from src.backtesting.utils.idm_weights import compute_div_mult
from src.data.fx.clusters import fx_cluster_for


def test_fx_cluster_assignments():
    assert fx_cluster_for("EURUSD") == "usd_major"
    assert fx_cluster_for("USDJPY") == "usd_major"
    assert fx_cluster_for("EURGBP") == "eur_cross"
    assert fx_cluster_for("XAUUSD") == "metal"


def test_compute_div_mult_accepts_fx_cluster_fn():
    universe = ["EURUSD", "USDJPY", "XAUUSD"]
    dm = compute_div_mult(universe, cluster_fn=fx_cluster_for)
    assert set(dm) == set(universe)
    assert all(v > 0 for v in dm.values())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n fintech pytest tests/backtesting/utils/test_idm_weights_fx.py -v`
Expected: FAIL (`ImportError` for `fx_cluster_for`, and `compute_div_mult` rejects `cluster_fn`).

- [ ] **Step 3a: Write the FX cluster map**

```python
# src/data/fx/clusters.py
"""Coarse FX clusters for IDM diversification weighting.

Groups pairs by dominant risk driver so the Instrument Diversification
Multiplier gives each cluster an equal risk budget. Deterministic, data-free.
"""
from __future__ import annotations

_METALS = {"XAU", "XAG"}
_EM = {"BRL", "CNH", "CLP", "CZK", "HKD", "HUF", "ILS", "INR", "KRW",
       "MXN", "PLN", "RUB", "TRY", "ZAR", "SGD"}


def fx_cluster_for(pair: str) -> str:
    base, quote = pair[:3], pair[3:]
    if base in _METALS or quote in _METALS:
        return "metal"
    if base in _EM or quote in _EM:
        return "em"
    if "USD" in (base, quote):
        return "usd_major"
    if "EUR" in (base, quote):
        return "eur_cross"
    if "JPY" in (base, quote):
        return "jpy_cross"
    return "other_cross"
```

- [ ] **Step 3b: Generalize compute_div_mult (backward-compatible)**

In `src/backtesting/utils/idm_weights.py`, change the import and signature. Replace the top import block and the function signature/first line:

```python
# near the top, keep the existing import and alias it as the default
from src.data.futures.asset_class import cluster_for as _default_cluster_for
```

```python
def compute_div_mult(
    universe: list[str],
    per_instrument_cap: float | None = None,
    cluster_fn=_default_cluster_for,
) -> dict[str, float]:
    """Return {symbol: div_mult} for every symbol in `universe`.

    `cluster_fn` maps a symbol to its cluster label (default: the futures
    asset-class map, preserving existing behavior). Raises KeyError if any
    symbol is unmapped.

    `per_instrument_cap`, if given, clips each symbol's div_mult to at most
    that value. Default None reproduces the uncapped output exactly.
    """
    clusters = [cluster_fn(sym) for sym in universe]
```

Leave the rest of the function body unchanged (it already operates on the local `clusters` list).

- [ ] **Step 4: Run tests to verify pass (new + futures regression)**

Run: `conda run -n fintech pytest tests/backtesting/utils/test_idm_weights_fx.py tests/backtesting/utils/ -k idm -v`
Expected: PASS for the new tests AND all existing IDM tests (backward compatibility).

- [ ] **Step 5: Commit**

```bash
git add src/data/fx/clusters.py src/backtesting/utils/idm_weights.py tests/backtesting/utils/test_idm_weights_fx.py
git commit -m "feat(fx): FX cluster map; make compute_div_mult cluster_fn-pluggable"
```

---

### Task 5: FX notional position sizer

**Files:**
- Create: `src/backtesting/utils/position_sizer_fx.py`
- Test: `tests/backtesting/utils/test_position_sizer_fx.py`

**Interfaces:**
- Produces: `size_from_forecast_fx(forecast: float, capital: float, vol_target: float, base_to_usd: float, daily_vol: float, div_mult: float = 1.0) -> float` -- signed float number of base-currency units. `daily_vol` is daily return stdev; annualized via `sqrt(252)`. Returns 0.0 when `daily_vol <= 0`, `base_to_usd <= 0`, or `vol_target <= 0`. No integer rounding, no hard cap (leverage cap is applied at the book level in the simulator).

- [ ] **Step 1: Write the failing test**

```python
# tests/backtesting/utils/test_position_sizer_fx.py
import pytest

from src.backtesting.utils.position_sizer_fx import size_from_forecast_fx


def test_sizer_formula():
    # forecast 10 -> forecast/10 = 1.0
    # units = 1.0 * capital * vol_target * div_mult / (base_to_usd * ann_vol)
    units = size_from_forecast_fx(
        forecast=10.0, capital=100_000.0, vol_target=0.2,
        base_to_usd=1.10, daily_vol=0.01)
    ann_vol = 0.01 * (252 ** 0.5)
    expected = 1.0 * 100_000.0 * 0.2 / (1.10 * ann_vol)
    assert units == pytest.approx(expected)


def test_negative_forecast_gives_short():
    units = size_from_forecast_fx(-10.0, 100_000.0, 0.2, 1.10, 0.01)
    assert units < 0


def test_zero_vol_returns_zero():
    assert size_from_forecast_fx(10.0, 100_000.0, 0.2, 1.10, 0.0) == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n fintech pytest tests/backtesting/utils/test_position_sizer_fx.py -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/backtesting/utils/position_sizer_fx.py
"""Notional (base-currency-unit) FX position sizer.

Carver forecast -> vol-targeted notional, in units of the base currency. The
risk of holding one base unit is base_to_usd * annualized_return_vol (its USD
standard deviation), so dividing the USD risk budget by that term targets equal
USD risk per instrument. Unlike the futures sizer there is no contract
multiplier or integer/contract cap -- FX trades in continuous notional and the
leverage cap is enforced at the portfolio level.
"""
from __future__ import annotations


def size_from_forecast_fx(forecast: float, capital: float, vol_target: float,
                          base_to_usd: float, daily_vol: float,
                          div_mult: float = 1.0) -> float:
    if daily_vol <= 0 or base_to_usd <= 0 or vol_target <= 0:
        return 0.0
    ann_vol = daily_vol * (252 ** 0.5)
    denom = base_to_usd * ann_vol
    if denom <= 0:
        return 0.0
    return (forecast / 10.0) * capital * vol_target * div_mult / denom
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n fintech pytest tests/backtesting/utils/test_position_sizer_fx.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/backtesting/utils/position_sizer_fx.py tests/backtesting/utils/test_position_sizer_fx.py
git commit -m "feat(fx): notional vol-target position sizer"
```

---

### Task 6: FX cost model (pip + metals bps) in USD

**Files:**
- Modify: `src/backtesting/costs/fx.py`
- Test: `tests/backtesting/costs/test_fx_round_trip_usd.py`

**Interfaces:**
- Consumes: existing `fx_round_trip_pips(tier, session, override_pips)` in the same module.
- Produces: `fx_round_trip_usd(pair: str, units_traded: float, price: float, quote_to_usd: float, tier: str = "major", session: str = "ny", metals_bps: float = 4.0) -> float` -- total round-trip USD cost for trading `abs(units_traded)` base units. Currency pairs: `round_trip_pips * pip_size(pair) * abs(units_traded) * quote_to_usd`, with `pip_size = 0.01` for JPY-quoted pairs else `0.0001`. Metals (base in {XAU, XAG}): `abs(units_traded) * price * quote_to_usd * metals_bps / 10_000`.

- [ ] **Step 1: Write the failing test**

```python
# tests/backtesting/costs/test_fx_round_trip_usd.py
import pytest

from src.backtesting.costs.fx import fx_round_trip_pips, fx_round_trip_usd


def test_major_usd_quote_cost():
    # EURUSD, quote USD -> quote_to_usd = 1.0, pip_size 0.0001
    units = 100_000.0
    cost = fx_round_trip_usd("EURUSD", units, price=1.10, quote_to_usd=1.0,
                             tier="major", session="ny")
    rt_pips = fx_round_trip_pips("major", "ny")
    assert cost == pytest.approx(rt_pips * 0.0001 * units * 1.0)


def test_jpy_quote_uses_2dp_pip():
    units = 100_000.0
    cost = fx_round_trip_usd("USDJPY", units, price=150.0, quote_to_usd=1 / 150.0,
                             tier="major", session="ny")
    rt_pips = fx_round_trip_pips("major", "ny")
    assert cost == pytest.approx(rt_pips * 0.01 * units * (1 / 150.0))


def test_metals_use_bps_of_notional():
    units = 100.0  # 100 oz gold
    cost = fx_round_trip_usd("XAUUSD", units, price=2000.0, quote_to_usd=1.0,
                             metals_bps=4.0)
    notional = 100.0 * 2000.0 * 1.0
    assert cost == pytest.approx(notional * 4.0 / 10_000)


def test_cost_uses_absolute_units():
    a = fx_round_trip_usd("EURUSD", 50_000.0, 1.1, 1.0)
    b = fx_round_trip_usd("EURUSD", -50_000.0, 1.1, 1.0)
    assert a == pytest.approx(b)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n fintech pytest tests/backtesting/costs/test_fx_round_trip_usd.py -v`
Expected: FAIL with `ImportError` for `fx_round_trip_usd`.

- [ ] **Step 3: Append implementation to `src/backtesting/costs/fx.py`**

```python
# append to src/backtesting/costs/fx.py

_METALS_BASES = {"XAU", "XAG"}


def _pip_size(pair: str) -> float:
    """0.01 for JPY-quoted pairs, 0.0001 otherwise."""
    return 0.01 if pair[3:] == "JPY" else 0.0001


def fx_round_trip_usd(pair: str, units_traded: float, price: float,
                      quote_to_usd: float, tier: FxTier = "major",
                      session: Session = "ny", metals_bps: float = 4.0) -> float:
    """Total round-trip USD cost for trading abs(units_traded) base units.

    Currency pairs: spread (pips) x pip_size x units x quote->USD. Metals
    (XAU/XAG) have no standard pip -> priced as metals_bps of USD notional,
    which is scale-invariant and how metal spreads are actually quoted.
    """
    qty = abs(units_traded)
    if pair[:3] in _METALS_BASES:
        notional_usd = qty * price * quote_to_usd
        return notional_usd * metals_bps / 10_000.0
    rt_pips = fx_round_trip_pips(tier, session)
    return rt_pips * _pip_size(pair) * qty * quote_to_usd
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n fintech pytest tests/backtesting/costs/test_fx_round_trip_usd.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/backtesting/costs/fx.py tests/backtesting/costs/test_fx_round_trip_usd.py
git commit -m "feat(fx): pip/metals-bps round-trip cost in USD"
```

---

### Task 7: FxSpotPortfolioSimulator (MTM + carry accrual + leverage cap + floor)

**Files:**
- Create: `src/backtesting/engine/fx_spot_portfolio_simulator.py`
- Test: `tests/backtesting/engine/test_fx_spot_portfolio_simulator.py`

**Interfaces:**
- Consumes: `size_from_forecast_fx` (Task 5); a `cost_fn(pair, units_traded, price, quote_to_usd) -> float` closure supplied by the caller.
- Produces:
  - `FxBacktestResult` dataclass: `equity_curve: pd.Series`, `trades: pd.DataFrame` (columns `date, pair, units, cost`), `leverage_utilization: pd.Series`.
  - `FxSpotPortfolioSimulator(initial_capital, cost_fn, rebalance="daily", cost_mult=1.0, leverage_cap=10.0)` with `run_sized(close_panel, forecast_panel, daily_vol_panel, vol_target, quote_usd_panel, rate_diff_panel, div_mult=1.0) -> FxBacktestResult`. `close_panel`, `forecast_panel`, `daily_vol_panel`, `quote_usd_panel`, `rate_diff_panel` are all single-level DataFrames indexed by FX date with pair columns.

- [ ] **Step 1: Write the failing tests (golden carry + PnL conversion + floor)**

```python
# tests/backtesting/engine/test_fx_spot_portfolio_simulator.py
import datetime as dt

import numpy as np
import pandas as pd
import pytest

from src.backtesting.engine.fx_spot_portfolio_simulator import FxSpotPortfolioSimulator


def _dates(n):
    return pd.Index([dt.date(2024, 1, 1) + dt.timedelta(days=i) for i in range(n)])


def _flat(pair, price, n):
    idx = _dates(n)
    return pd.DataFrame({pair: [price] * n}, index=idx)


def _zero_cost(pair, units_traded, price, quote_to_usd):
    return 0.0


def test_golden_carry_flat_price():
    # Hold long EURUSD one "year" (365 daily steps) at flat price with a +2%
    # rate differential and no costs. Carry should compound the equity by ~2%
    # of the held USD notional. We fix units directly via a huge forecast + the
    # sizer, so instead we drive units deterministically: use forecast that
    # yields a known notional by setting daily_vol so units are round.
    n = 366
    idx = _dates(n)
    price = 1.00
    close = pd.DataFrame({"EURUSD": [price] * n}, index=idx)
    quote_usd = pd.DataFrame({"EURUSD": [1.0] * n}, index=idx)
    rate_diff = pd.DataFrame({"EURUSD": [0.02] * n}, index=idx)
    # daily_vol chosen so units come out to exactly 100_000 for forecast 10:
    # units = 1.0 * capital * vol_target / (base_to_usd * daily_vol*sqrt(252))
    capital, vol_target = 1_000_000.0, 0.2
    base_to_usd = price * 1.0
    target_units = 100_000.0
    daily_vol = (1.0 * capital * vol_target) / (base_to_usd * target_units * np.sqrt(252))
    forecast = pd.DataFrame({"EURUSD": [10.0] * n}, index=idx)
    dvol = pd.DataFrame({"EURUSD": [daily_vol] * n}, index=idx)

    sim = FxSpotPortfolioSimulator(capital, _zero_cost, rebalance="daily", leverage_cap=100.0)
    res = sim.run_sized(close, forecast, dvol, vol_target, quote_usd, rate_diff)
    # ~365 days of carry at 2%/yr on 100k notional ~= 100k * 0.02 = ~2000 USD
    gain = res.equity_curve.iloc[-1] - capital
    assert gain == pytest.approx(100_000.0 * 0.02, rel=0.05)


def test_usd_base_pnl_conversion():
    # Long USDJPY: price rises 150 -> 151, quote is JPY. PnL in JPY converted
    # to USD via 1/price. With units of USD held, PnL_usd = units*(dPx)*quote_to_usd.
    idx = _dates(2)
    close = pd.DataFrame({"USDJPY": [150.0, 151.0]}, index=idx)
    quote_usd = pd.DataFrame({"USDJPY": [1 / 150.0, 1 / 151.0]}, index=idx)
    rate_diff = pd.DataFrame({"USDJPY": [0.0, 0.0]}, index=idx)
    # Force a fixed long position via forecast/vol on day 1.
    forecast = pd.DataFrame({"USDJPY": [10.0, 10.0]}, index=idx)
    capital, vol_target = 1_000_000.0, 0.2
    target_units = 10_000.0
    base_to_usd = 150.0 * (1 / 150.0)  # = 1.0
    daily_vol = (capital * vol_target) / (base_to_usd * target_units * np.sqrt(252))
    dvol = pd.DataFrame({"USDJPY": [daily_vol, daily_vol]}, index=idx)
    sim = FxSpotPortfolioSimulator(capital, _zero_cost, rebalance="daily", leverage_cap=100.0)
    res = sim.run_sized(close, forecast, dvol, vol_target, quote_usd, rate_diff)
    # Day 2 PnL = units * (151-150) * quote_to_usd(day2) = 10_000 * 1 * (1/151)
    day2_pnl = res.equity_curve.iloc[1] - res.equity_curve.iloc[0]
    assert day2_pnl == pytest.approx(10_000.0 * 1.0 * (1 / 151.0), rel=1e-6)


def test_bankruptcy_floor():
    # Massive adverse move with high leverage floors equity at 0, never negative.
    idx = _dates(2)
    close = pd.DataFrame({"EURUSD": [1.00, 0.01]}, index=idx)
    quote_usd = pd.DataFrame({"EURUSD": [1.0, 1.0]}, index=idx)
    rate_diff = pd.DataFrame({"EURUSD": [0.0, 0.0]}, index=idx)
    forecast = pd.DataFrame({"EURUSD": [10.0, 10.0]}, index=idx)
    dvol = pd.DataFrame({"EURUSD": [1e-6, 1e-6]}, index=idx)  # tiny vol -> huge units
    sim = FxSpotPortfolioSimulator(100_000.0, _zero_cost, rebalance="daily", leverage_cap=100.0)
    res = sim.run_sized(close, forecast, dvol, 0.2, quote_usd, rate_diff)
    assert (res.equity_curve >= 0).all()
    assert res.equity_curve.iloc[-1] == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda run -n fintech pytest tests/backtesting/engine/test_fx_spot_portfolio_simulator.py -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/backtesting/engine/fx_spot_portfolio_simulator.py
"""Daily spot-FX backtest simulator with carry accrual.

Separate from the equity and futures simulators. Positions are signed notionals
in units of each pair's base currency. Each day: mark-to-market in USD, accrue
the interest-rate differential (carry) on held USD notional, debit costs on
rebalance days, then enforce a gross-notional leverage cap and a bankruptcy
floor (equity is provably non-negative).

    base_to_usd  = price * quote_to_usd            # 1 base unit in USD
    mtm_usd      = sum_p units_p * (px_t - px_{t-1}) * quote_to_usd_t
    carry_usd    = sum_p (units_p * base_to_usd_t) * rate_diff_t / 365
    equity_t     = equity_{t-1} + mtm_usd + carry_usd - costs_t
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd

from src.backtesting.utils.position_sizer_fx import size_from_forecast_fx

CostFn = Callable[[str, float, float, float], float]  # (pair, units_traded, price, quote_to_usd)


@dataclass
class FxBacktestResult:
    equity_curve: pd.Series
    trades: pd.DataFrame
    leverage_utilization: pd.Series


class FxSpotPortfolioSimulator:
    def __init__(self, initial_capital: float, cost_fn: CostFn,
                 rebalance: str = "daily", cost_mult: float = 1.0,
                 leverage_cap: float = 10.0):
        self.initial_capital = float(initial_capital)
        self.cost_fn = cost_fn
        self.rebalance = rebalance
        self.cost_mult = float(cost_mult)
        self.leverage_cap = float(leverage_cap)

    def _is_rebalance(self, d, prev_d) -> bool:
        if self.rebalance == "daily" or prev_d is None:
            return True
        if self.rebalance == "weekly":
            return d.isocalendar()[1] != prev_d.isocalendar()[1]
        if self.rebalance == "monthly":
            return d.month != prev_d.month
        return True

    def _scale_to_leverage(self, targets: dict, base_to_usd: dict, equity: float) -> dict:
        gross = sum(abs(u * base_to_usd[p]) for p, u in targets.items())
        cap_notional = self.leverage_cap * equity
        if gross <= cap_notional or gross <= 0:
            return targets
        scale = cap_notional / gross
        return {p: u * scale for p, u in targets.items()}

    def run_sized(self, close_panel: pd.DataFrame, forecast_panel: pd.DataFrame,
                  daily_vol_panel: pd.DataFrame, vol_target: float,
                  quote_usd_panel: pd.DataFrame, rate_diff_panel: pd.DataFrame,
                  div_mult: float | dict = 1.0) -> FxBacktestResult:
        pairs = list(close_panel.columns)
        dates = list(close_panel.index)
        equity_val = self.initial_capital
        current = {p: 0.0 for p in pairs}
        equity, util, trade_rows = [], [], []
        prev_close = None
        prev_d = None
        blown = False

        def dm(p):
            return div_mult if isinstance(div_mult, (int, float)) else div_mult.get(p, 1.0)

        for d in dates:
            row_close = close_panel.loc[d]
            row_q = quote_usd_panel.loc[d]
            row_rd = rate_diff_panel.loc[d]

            if blown:
                util.append(0.0)
                equity.append(0.0)
                prev_close, prev_d = row_close, d
                continue

            # 1. MTM + carry on existing positions
            if prev_close is not None:
                pnl = 0.0
                for p in pairs:
                    u = current[p]
                    if u == 0.0:
                        continue
                    px, ppx, q = row_close[p], prev_close[p], row_q[p]
                    if pd.notna(px) and pd.notna(ppx) and pd.notna(q):
                        pnl += u * (px - ppx) * q
                    if pd.notna(px) and pd.notna(q) and pd.notna(row_rd[p]):
                        pnl += (u * px * q) * row_rd[p] / 365.0
                equity_val += pnl

            if equity_val <= 0:
                current = {p: 0.0 for p in pairs}
                equity_val, blown = 0.0, True
                util.append(0.0)
                equity.append(0.0)
                prev_close, prev_d = row_close, d
                continue

            # 2. Rebalance -> target notionals, leverage-capped
            if self._is_rebalance(d, prev_d):
                base_to_usd = {}
                targets = {}
                for p in pairs:
                    px, q = row_close[p], row_q[p]
                    f = forecast_panel.loc[d, p] if d in forecast_panel.index else float("nan")
                    v = daily_vol_panel.loc[d, p] if d in daily_vol_panel.index else float("nan")
                    if pd.isna(px) or pd.isna(q) or pd.isna(f) or pd.isna(v):
                        base_to_usd[p] = 0.0
                        targets[p] = 0.0
                        continue
                    b2u = px * q
                    base_to_usd[p] = b2u
                    targets[p] = size_from_forecast_fx(
                        float(f), equity_val, vol_target, b2u, float(v), dm(p))
                targets = self._scale_to_leverage(targets, base_to_usd, equity_val)
                for p in pairs:
                    want = targets[p]
                    diff = want - current[p]
                    if diff != 0.0:
                        c = self.cost_fn(p, diff, float(row_close[p]), float(row_q[p])) * self.cost_mult
                        equity_val -= c
                        trade_rows.append({"date": d, "pair": p, "units": diff, "cost": c})
                        current[p] = want

            if not blown and equity_val <= 0:
                current = {p: 0.0 for p in pairs}
                equity_val, blown = 0.0, True

            gross = sum(abs(current[p] * (row_close[p] * row_q[p]))
                        for p in pairs if pd.notna(row_close[p]) and pd.notna(row_q[p]))
            util.append(gross / equity_val if equity_val > 0 else 0.0)
            equity.append(equity_val)
            prev_close, prev_d = row_close, d

        eq = pd.Series(equity, index=dates, name="equity")
        lu = pd.Series(util, index=dates, name="leverage_utilization")
        trades = pd.DataFrame(trade_rows) if trade_rows else pd.DataFrame(
            columns=["date", "pair", "units", "cost"])
        return FxBacktestResult(equity_curve=eq, trades=trades, leverage_utilization=lu)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda run -n fintech pytest tests/backtesting/engine/test_fx_spot_portfolio_simulator.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/backtesting/engine/fx_spot_portfolio_simulator.py tests/backtesting/engine/test_fx_spot_portfolio_simulator.py
git commit -m "feat(fx): spot simulator with carry accrual, leverage cap, floor"
```

---

# Milestone 3 -- Runner, strategies, config

### Task 8: FX reference strategies + registry entries

**Files:**
- Create: `src/strategies/advanced/fx_strategies.py`
- Modify: `src/strategies/registry.py`
- Test: `tests/strategies/test_fx_strategies.py`

**Interfaces:**
- Consumes: `CarverMomentumStrategy`, `FuturesValueStrategy` (both expose `forecast_panel(close_panel)` on a single-level pair-columned close DataFrame).
- Produces: `FxTrendStrategy(universe, **params)` and `FxValueStrategy(universe, **params)` in `fx_strategies.py`; registry names `"FxTrend"` and `"FxValue"` resolvable via `get_strategy_class`.

- [ ] **Step 1: Write the failing test**

```python
# tests/strategies/test_fx_strategies.py
import numpy as np
import pandas as pd

from src.strategies.registry import get_strategy_class


def _price_panel(pairs, n=400):
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    rng = np.random.default_rng(0)
    data = {p: 1.0 + np.cumsum(rng.normal(0, 0.001, n)) for p in pairs}
    return pd.DataFrame(data, index=idx)


def test_fx_trend_registered_and_forecasts():
    cls = get_strategy_class("FxTrend")
    strat = cls(["EURUSD", "USDJPY"])
    fc = strat.forecast_panel(_price_panel(["EURUSD", "USDJPY"]))
    assert list(fc.columns) == ["EURUSD", "USDJPY"]
    assert fc.abs().max().max() <= 20.0  # forecast cap


def test_fx_value_registered_and_forecasts():
    cls = get_strategy_class("FxValue")
    strat = cls(["EURUSD", "USDJPY"])
    fc = strat.forecast_panel(_price_panel(["EURUSD", "USDJPY"], n=1400))
    assert list(fc.columns) == ["EURUSD", "USDJPY"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n fintech pytest tests/strategies/test_fx_strategies.py -v`
Expected: FAIL (`ValueError` unknown strategy name).

- [ ] **Step 3a: Create the FX strategy subclasses**

```python
# src/strategies/advanced/fx_strategies.py
"""Spot-FX reference strategies.

Both are price-only forecast_panel strategies, so they reuse the futures
forecast logic unchanged: FX trend = Carver multi-speed EWMAC; FX value =
Asness nominal 5yr-to-1yr reversal. Thin subclasses keep the FX names distinct
in the registry and leave room to diverge (e.g. a future PPP value signal).
"""
from __future__ import annotations

from src.strategies.advanced.carver_momentum_strategy import CarverMomentumStrategy
from src.strategies.advanced.futures_value_strategy import FuturesValueStrategy


class FxTrendStrategy(CarverMomentumStrategy):
    pass


class FxValueStrategy(FuturesValueStrategy):
    pass
```

- [ ] **Step 3b: Register the FX strategies**

In `src/strategies/registry.py`, add two entries to the registry dict (right after the `FuturesValue` entry, before the closing brace at line 64):

```python
    "FxTrend": ("src.strategies.advanced.fx_strategies", "FxTrendStrategy"),
    "FxValue": ("src.strategies.advanced.fx_strategies", "FxValueStrategy"),
```

And add display aliases to `_DISPLAY_NAME_MAP` (after the `"Futures Value"` entry):

```python
    "FX Trend": "FxTrend",
    "FX Value": "FxValue",
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n fintech pytest tests/strategies/test_fx_strategies.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/strategies/advanced/fx_strategies.py src/strategies/registry.py tests/strategies/test_fx_strategies.py
git commit -m "feat(fx): trend + value reference strategies and registry entries"
```

---

### Task 9: run_fx_backtest orchestration + routing + config + universe

**Files:**
- Create: `src/backtesting/engine/fx_backtest.py`
- Modify: `src/backtest_runner.py` (add routing branch after the futures branch, ~line 1218)
- Create: `config/universes/fx_spot-2026.csv`
- Create: `config/backtesting/fx_trend.yaml`, `config/backtesting/fx_value.yaml`
- Test: `tests/backtesting/engine/test_fx_backtest.py`

**Interfaces:**
- Consumes: `load_fx_daily_panel`, `build_quote_usd_panel` (Task 2), `load_fx_rate_panel`, `build_rate_diff_panel`, `currencies_for_pairs` (Task 3), `FxSpotPortfolioSimulator` (Task 7), `fx_round_trip_usd` (Task 6), `compute_div_mult` + `fx_cluster_for` (Task 4), `close_to_close_rv` (existing), `StandardReportGenerator` (existing), `get_strategy_class` (existing), `append_run` (existing).
- Produces: `run_fx_backtest(config: dict, register: bool = True, log_trades: bool = False) -> dict` with keys `n_days`, `metrics`, `equity_curve` (list), `run_id`, `trade_log_dir`. Trade log dir: `output/backtests/fx/<strategy>/<start>_to_<end>/`.

- [ ] **Step 1: Write the failing test (dependency-injected panels, no disk I/O)**

```python
# tests/backtesting/engine/test_fx_backtest.py
import datetime as dt

import numpy as np
import pandas as pd

from src.backtesting.engine import fx_backtest


def _fake_panel(pairs, n=400):
    idx = pd.Index([dt.date(2020, 1, 1) + dt.timedelta(days=i) for i in range(n)])
    rng = np.random.default_rng(1)
    frames = {}
    for p in pairs:
        close = 1.0 + np.cumsum(rng.normal(0, 0.001, n))
        frames[(p, "close")] = pd.Series(close, index=idx)
        frames[(p, "ret")] = pd.Series(close, index=idx).pct_change()
    df = pd.DataFrame(frames)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df


def test_run_fx_backtest_end_to_end(monkeypatch, tmp_path):
    pairs = ["EURUSD", "USDJPY", "EURGBP", "GBPUSD"]
    panel = _fake_panel(pairs)
    monkeypatch.setattr(fx_backtest, "load_fx_daily_panel", lambda p, s, e: panel)

    def _fake_rates(currencies, index):
        return pd.DataFrame({c: [0.02] * len(index) for c in currencies}, index=index)

    monkeypatch.setattr(fx_backtest, "load_fx_rate_panel", _fake_rates)
    monkeypatch.chdir(tmp_path)

    config = {
        "asset_class": "fx",
        "strategy": {"name": "FxTrend", "universe": pairs, "params": {}},
        "dates": {"start": "2020-01-01", "end": "2021-02-01"},
        "backtest": {"initial_capital": 100_000.0, "vol_target_per_instrument": 0.2,
                     "rebalance": "weekly", "leverage_cap": 10.0, "tier": "major"},
    }
    result = fx_backtest.run_fx_backtest(config, register=False, log_trades=True)
    assert result["n_days"] == len(panel)
    assert "sharpe_ratio" in result["metrics"]
    assert result["trade_log_dir"] is not None
    import os
    assert os.path.exists(os.path.join(result["trade_log_dir"], "trades.csv"))
    assert os.path.exists(os.path.join(result["trade_log_dir"], "equity.csv"))
    assert os.path.exists(os.path.join(result["trade_log_dir"], "leverage_utilization.csv"))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n fintech pytest tests/backtesting/engine/test_fx_backtest.py -v`
Expected: FAIL with `ImportError` / `AttributeError`.

- [ ] **Step 3a: Write run_fx_backtest**

```python
# src/backtesting/engine/fx_backtest.py
"""Config-driven spot-FX backtest orchestration.

Assembles: daily FX panel -> USD-conversion + rate-diff panels -> strategy
forecast -> close-to-close vol -> FxSpotPortfolioSimulator (carry-accruing) ->
standard report -> experiment registry. The FX counterpart to
futures_backtest.py; kept separate because spot-FX PnL/carry/notional math does
not fit the futures contract/margin abstractions.
"""
from __future__ import annotations

from datetime import date, datetime
from functools import partial
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from src.backtesting.costs.fx import fx_round_trip_usd
from src.backtesting.data.fx_backtest_loader import load_fx_daily_panel, build_quote_usd_panel
from src.backtesting.engine.fx_spot_portfolio_simulator import FxSpotPortfolioSimulator
from src.backtesting.reporting.standard_report import StandardReportGenerator
from src.backtesting.utils.idm_weights import compute_div_mult
from src.data.fx.clusters import fx_cluster_for
from src.data.fx_rates import load_fx_rate_panel, build_rate_diff_panel, currencies_for_pairs
from src.features.volatility import close_to_close_rv
from src.strategies.registry import get_strategy_class
from src.utils import logger

_DEFAULT_CAPITAL = 100_000.0
_DEFAULT_VOL_TARGET = 0.20
_DEFAULT_REBALANCE = "weekly"
_DEFAULT_LEVERAGE_CAP = 10.0


def _as_date(value: Any) -> date:
    if isinstance(value, date):
        return value
    return datetime.strptime(str(value), "%Y-%m-%d").date()


def _cost_fn_factory(tier: str, session: str = "ny"):
    def cost_fn(pair, units_traded, price, quote_to_usd):
        return fx_round_trip_usd(pair, units_traded, price, quote_to_usd,
                                 tier=tier, session=session)
    return cost_fn


def run_fx_backtest(config: Dict[str, Any], register: bool = True,
                    log_trades: bool = False) -> Dict[str, Any]:
    strat_cfg = config.get("strategy", {})
    dates_cfg = config.get("dates", {})
    bt = config.get("backtest", {})

    universe = list(strat_cfg["universe"])
    start = _as_date(dates_cfg["start"])
    end = _as_date(dates_cfg["end"])
    capital = float(bt.get("initial_capital", _DEFAULT_CAPITAL))
    vol_target = float(bt.get("vol_target_per_instrument", _DEFAULT_VOL_TARGET))
    rebalance = bt.get("rebalance", _DEFAULT_REBALANCE)
    cost_mult = float(bt.get("cost_mult", 1.0))
    leverage_cap = float(bt.get("leverage_cap", _DEFAULT_LEVERAGE_CAP))
    tier = bt.get("tier", "major")
    use_idm = bool(bt.get("idm", False))
    idm_cap = bt.get("idm_cap", None)

    strategy_name = strat_cfg.get("name", "FxTrend")
    strategy = get_strategy_class(strategy_name)(universe, **strat_cfg.get("params", {}))

    panel = load_fx_daily_panel(universe, start, end)
    present = [p for p in universe if p in {c[0] for c in panel.columns}]
    close = panel.xs("close", axis=1, level=1)[present]

    quote_usd = build_quote_usd_panel(panel, present)
    rate_panel = load_fx_rate_panel(currencies_for_pairs(present), close.index)
    rate_diff = build_rate_diff_panel(present, rate_panel)

    forecasts = strategy.forecast_panel(close)[present]
    returns = close.pct_change(fill_method=None)
    daily_vol = returns.apply(lambda col: close_to_close_rv(col, 25, annualization_factor=1), axis=0)

    div_mult = compute_div_mult(present, per_instrument_cap=idm_cap,
                                cluster_fn=fx_cluster_for) if use_idm else 1.0

    sim = FxSpotPortfolioSimulator(capital, _cost_fn_factory(tier), rebalance=rebalance,
                                   cost_mult=cost_mult, leverage_cap=leverage_cap)
    res = sim.run_sized(close, forecasts, daily_vol, vol_target, quote_usd, rate_diff, div_mult)

    report = StandardReportGenerator().generate_report(
        res.equity_curve, strategy_name, present, str(start), str(end), capital)

    run_id = None
    if register:
        try:
            from src.experiments import append_run
            run_id = append_run(
                strategy_name=strategy_name, agent_name="fx-harness",
                metrics=report["overall_metrics"], asset_class="fx",
                data_frequency="daily", params=config,
                window_start=start, window_end=end)
        except Exception as e:
            logger.error(f"[fx_backtest] registry append_run failed (non-fatal): {e}")

    trade_log_dir = _write_trade_log(res, strategy_name, start, end) if log_trades else None
    return {
        "n_days": len(res.equity_curve),
        "metrics": report["overall_metrics"],
        "equity_curve": res.equity_curve.tolist(),
        "run_id": run_id,
        "trade_log_dir": trade_log_dir,
    }


def _write_trade_log(res, strategy_name: str, start, end) -> str:
    out = Path("output") / "backtests" / "fx" / strategy_name / f"{start}_to_{end}"
    out.mkdir(parents=True, exist_ok=True)
    res.trades.to_csv(out / "trades.csv", index=False)
    res.equity_curve.rename("equity").to_frame().to_csv(out / "equity.csv", index_label="date")
    res.leverage_utilization.rename("leverage_utilization").to_frame().to_csv(
        out / "leverage_utilization.csv", index_label="date")
    logger.info(f"[fx_backtest] wrote trade log ({len(res.trades)} fills) to {out}")
    return str(out)
```

- [ ] **Step 3b: Add the routing branch in `src/backtest_runner.py`**

Immediately after the futures routing block (after `return` at line 1217, before the `try: config = load_config(...)` at line 1219), insert:

```python
        if raw_config.get('asset_class') == 'fx':
            from src.backtesting.engine.fx_backtest import run_fx_backtest

            logger.info(f"Running config-driven spot-FX backtest: {args.config}")
            save_trades = raw_config.get('output', {}).get('save_trades', True)
            result = run_fx_backtest(raw_config, log_trades=save_trades)
            logger.success(
                f"FX backtest complete: n_days={result['n_days']}, "
                f"sharpe_ratio={result['metrics'].get('sharpe_ratio')}, "
                f"trade_log={result.get('trade_log_dir')}"
            )
            if result.get('run_id'):
                logger.info(f"[registry] appended run_id={result['run_id']}")
            return
```

- [ ] **Step 3c: Create the universe CSV**

```csv
# config/universes/fx_spot-2026.csv
symbol
EURUSD
USDJPY
USDCHF
EURJPY
EURCHF
CHFJPY
XAUUSD
XAGUSD
```

v1 is restricted to USD/EUR/CHF/JPY-legged pairs + metals because only those currencies have carry rates on disk (Task 3). GBP/CAD/AUD/NZD are deferred until their FRED short rates are pulled. All USD-conversion legs (USDJPY, USDCHF, EURUSD) are in-universe, so every cross resolves.

- [ ] **Step 3d: Create the config YAMLs**

```yaml
# config/backtesting/fx_trend.yaml
asset_class: fx
strategy:
  name: FxTrend
  universe: [EURUSD, USDJPY, USDCHF, EURJPY, EURCHF, CHFJPY, XAUUSD, XAGUSD]
  params: {}
dates:
  start: "2011-01-01"
  end: "2025-12-31"
backtest:
  initial_capital: 100000.0
  vol_target_per_instrument: 0.20
  rebalance: weekly
  leverage_cap: 10.0
  tier: major
  idm: true
  idm_cap: 2.5
output:
  save_trades: true
```

```yaml
# config/backtesting/fx_value.yaml
asset_class: fx
strategy:
  name: FxValue
  universe: [EURUSD, USDJPY, USDCHF, EURJPY, EURCHF, CHFJPY, XAUUSD, XAGUSD]
  params: {}
dates:
  start: "2011-01-01"
  end: "2025-12-31"
backtest:
  initial_capital: 100000.0
  vol_target_per_instrument: 0.20
  rebalance: weekly
  leverage_cap: 10.0
  tier: major
  idm: true
  idm_cap: 2.5
output:
  save_trades: true
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n fintech pytest tests/backtesting/engine/test_fx_backtest.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/backtesting/engine/fx_backtest.py src/backtest_runner.py config/universes/fx_spot-2026.csv config/backtesting/fx_trend.yaml config/backtesting/fx_value.yaml tests/backtesting/engine/test_fx_backtest.py
git commit -m "feat(fx): run_fx_backtest orchestration, routing, config, universe"
```

---

# Milestone 4 -- Evaluation

### Task 10: FX walk-forward + statistical gate

**Files:**
- Create: `scripts/backtest_scripts/run_fx_walkforward.py`
- Test: `tests/backtesting/test_fx_walkforward.py`

**Interfaces:**
- Consumes: `run_fx_backtest` (Task 9), `load_fx_daily_panel` (Task 2), `psr`, `dsr`, `pbo` (existing `src/backtesting/statistics/`), `parallel_map` (existing), `RunStatus` (existing), `append_run` (existing).
- Produces: `walk_forward_fx(train_months, test_months, step_months, start, end, universe, capital=100000.0, vol_target=0.20, strategy_name="FxTrend", tier="major", idm=False, idm_cap=None, max_workers=None) -> dict` with keys `oos_sharpe`, `psr`, `dsr`, `pbo`, `oos_sharpe_1_5x_cost`, `n_windows`, `n_oos_days`, `window_sharpes`, `trial_count`, `run_id`; and `_verdict_fx(result) -> str`.

Extract the pure window-building, OOS-slicing, and gate helpers into a NEW shared module `src/backtesting/walkforward_common.py` (they operate only on dates/equity-curves/return-arrays and carry no futures concepts), and have BOTH `run_carver_walkforward.py` and the new `run_fx_walkforward.py` import them -- no duplication, single source of truth. Only the FX per-window runner and aggregator are new. Because trend/value are parameter-free, `trial_count = 1`.

This modifies `run_carver_walkforward.py` (delete its local helper defs, import them from the shared module instead). The existing futures walk-forward test (`tests/backtesting/test_futures_walkforward.py`) MUST still pass unchanged after the move -- run it as a regression gate.

Note (annualization): `_annualized_sharpe` uses 252; FX daily bars are ~260/yr. The ~1.5% difference is immaterial and matches the futures convention, so 252 is kept deliberately.

- [ ] **Step 1: Write the failing test**

```python
# tests/backtesting/test_fx_walkforward.py
import numpy as np

from scripts.backtest_scripts import run_fx_walkforward as wf


def test_build_windows_non_overlapping():
    import datetime as dt
    windows = wf._build_windows(36, 12, 12, dt.date(2011, 1, 1), dt.date(2020, 1, 1))
    assert len(windows) >= 2
    # OOS windows are non-overlapping and ordered
    for (ts1, tst1, te1), (ts2, tst2, te2) in zip(windows, windows[1:]):
        assert tst2 >= te1


def test_verdict_reject_on_nonpositive_sharpe():
    result = {"psr": 0.5, "dsr": 0.5, "pbo": 0.3, "oos_sharpe": -0.1,
              "oos_sharpe_1_5x_cost": -0.2}
    assert wf._verdict_fx(result).startswith("REJECT")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n fintech pytest tests/backtesting/test_fx_walkforward.py -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Write the implementation**

Step A -- create the shared module. Move these functions VERBATIM out of `scripts/backtest_scripts/run_carver_walkforward.py` into a new `src/backtesting/walkforward_common.py`: `_as_date`, `_add_months`, `_build_windows`, `_oos_returns_dated`, `_oos_returns`, `_annualized_sharpe`, `_compute_pbo`, `_verdict`, plus the constants `TRIAL_COUNT_PARAMETER_FREE` and `_TRADING_DAYS_PER_YEAR`. Keep the exact names (leading underscores). Then edit `run_carver_walkforward.py` to `from src.backtesting.walkforward_common import (...)` those names and DELETE its now-moved local defs (leave its FX-agnostic futures-specific functions -- `_config_to_kwargs`, `_run_window`, `process_window`, `walk_forward_carver`, `_write_readiness_report`, `main` -- in place). Run `conda run -n fintech pytest tests/backtesting/test_futures_walkforward.py -v` as a REGRESSION GATE -- it MUST pass unchanged.

Step B -- create `scripts/backtest_scripts/run_fx_walkforward.py` importing from the shared module:

```python
from src.backtesting.walkforward_common import (
    _as_date, _build_windows, _oos_returns, _annualized_sharpe,
    _compute_pbo, _verdict as _verdict_fx,
)
```

Then define the FX per-window runner and aggregator:

```python
"""Spot-FX walk-forward + statistical gate + readiness report.

FX trend and value are PARAMETER-FREE (fixed forecast scalars/speeds), so this
rolls non-overlapping OOS windows, runs run_fx_backtest once per window per cost
leg (1x and 1.5x), stitches the OOS-dated return series, and evaluates the
Sharpe/PSR/DSR/PBO gate. Trial count = 1.
"""
from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from src.backtesting.data.fx_backtest_loader import load_fx_daily_panel
from src.backtesting.engine.fx_backtest import run_fx_backtest
from src.backtesting.statistics.dsr import dsr
from src.backtesting.statistics.pbo import pbo
from src.backtesting.statistics.psr import psr
from src.utils import logger

_DEFAULT_CAPITAL = 100_000.0
_DEFAULT_VOL_TARGET = 0.20
TRIAL_COUNT_PARAMETER_FREE = 1
_REPORT_PATH = "docs/reports/fx/FX_WALK_FORWARD.md"
_TRADING_DAYS_PER_YEAR = 252
```

With the helpers imported (above), define the FX per-window runner + aggregator:

```python
def _run_window_fx(universe, train_start, test_end, capital, vol_target,
                   cost_mult, strategy_name, tier, idm, idm_cap):
    config = {
        "asset_class": "fx",
        "strategy": {"name": strategy_name, "universe": list(universe), "params": {}},
        "dates": {"start": str(train_start), "end": str(test_end)},
        "backtest": {"initial_capital": capital, "vol_target_per_instrument": vol_target,
                     "rebalance": "weekly", "cost_mult": cost_mult, "leverage_cap": 10.0,
                     "tier": tier, "idm": idm, "idm_cap": idm_cap},
    }
    return run_fx_backtest(config, register=False, log_trades=False)


def process_window(spec: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    universe = spec["universe"]
    train_start, test_start, test_end = spec["train_start"], spec["test_start"], spec["test_end"]
    try:
        panel = load_fx_daily_panel(universe, train_start, test_end)
    except FileNotFoundError as e:
        logger.warning(f"[fx_walk_forward] skipping window {test_start}..{test_end}: {e}")
        return None
    window_universe = sorted({p for p, _ in panel.columns})
    dates = list(panel.index)
    res_1x = _run_window_fx(window_universe, train_start, test_end, spec["capital"],
                            spec["vol_target"], 1.0, spec["strategy_name"], spec["tier"],
                            spec["idm"], spec["idm_cap"])
    res_1_5x = _run_window_fx(window_universe, train_start, test_end, spec["capital"],
                              spec["vol_target"], 1.5, spec["strategy_name"], spec["tier"],
                              spec["idm"], spec["idm_cap"])
    return {
        "train_start": train_start, "test_start": test_start, "test_end": test_end,
        "window_universe": window_universe,
        "oos_1x": _oos_returns(res_1x["equity_curve"], dates, test_start),
        "oos_1_5x": _oos_returns(res_1_5x["equity_curve"], dates, test_start),
    }
```

Define `walk_forward_fx(...)` mirroring `walk_forward_carver` (build specs, `parallel_map(process_window, specs)`, stitch, compute `psr`/`dsr`/`pbo`, `append_run(asset_class="fx", ...)`); `_verdict_fx` is the imported `_verdict`. Add a `main()` wrapping the run in `RunStatus("fx_walkforward", ...)` and writing `_REPORT_PATH`, following `run_carver_walkforward.py::main` (config-driven via `--config`, `--train-months/--test-months/--step-months`, `--json`). The default `--train-months` is 36 for trend; the value walk-forward MUST be invoked with `--train-months 72` because FxValue needs a 5yr (1260-day) lookback and a 3yr warmup leaves its first OOS window signal-starved.

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n fintech pytest tests/backtesting/test_fx_walkforward.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add scripts/backtest_scripts/run_fx_walkforward.py tests/backtesting/test_fx_walkforward.py
git commit -m "feat(fx): walk-forward + PSR/DSR/PBO statistical gate"
```

---

### Task 11: Real-data validation run (build cache, run backtest + walk-forward)

**Files:**
- No new source; produces `fx_daily/` cache, `output/backtests/fx/...`, and `docs/reports/fx/FX_WALK_FORWARD.md`.

**Interfaces:**
- Consumes everything above. This task is the acceptance run, not a code change.

- [ ] **Step 1: Build the daily cache for the v1 universe**

Run:
```bash
conda run -n fintech python scripts/data/build_fx_daily_cache.py \
  --csv config/universes/fx_spot-2026.csv --start 2010-01-01 --end 2026-05-31
```
Expected: `wrote N pairs` log line; `fx_daily/symbol=EURUSD/...` parquet exists.

- [ ] **Step 2: Run a single representative backtest (produces trade log)**

Run:
```bash
conda run -n fintech python -m src.backtest_runner --config config/backtesting/fx_trend.yaml
```
Expected: `FX backtest complete: n_days=..., sharpe_ratio=..., trade_log=output/backtests/fx/FxTrend/...`. Verify `trades.csv`, `equity.csv`, `leverage_utilization.csv` exist and `trades.csv` is non-empty.

- [ ] **Step 3: Run the walk-forward gate for trend and value**

Run:
```bash
conda run -n fintech python scripts/backtest_scripts/run_fx_walkforward.py \
  --config config/backtesting/fx_trend.yaml --json output/fx_trend_gate.json
conda run -n fintech python scripts/backtest_scripts/run_fx_walkforward.py \
  --config config/backtesting/fx_value.yaml --train-months 72 \
  --report docs/reports/fx/FX_VALUE_WALK_FORWARD.md \
  --json output/fx_value_gate.json
```
Expected: each writes a readiness report with `OOS Sharpe`, `PSR`, `DSR`, `PBO`, and a verdict line. Record the numbers.

- [ ] **Step 4: Sanity-check carry impact**

Confirm the trend equity curve differs materially from a carry-off run (temporarily set all rates equal is not needed -- instead confirm `docs/reports/fx/FX_WALK_FORWARD.md` reports non-degenerate stats and the equity curve is monotonic-plausible). Record OOS Sharpe with carry ON as the headline number.

- [ ] **Step 5: Commit the reports**

```bash
git add -f docs/reports/fx/FX_WALK_FORWARD.md docs/reports/fx/FX_VALUE_WALK_FORWARD.md
git commit -m "docs(fx): first spot-FX walk-forward readiness reports (trend + value)"
```

---

## Self-Review

**Spec coverage:**
- 17:00-ET daily loader -> Task 1, 2. USD-conversion panel -> Task 2. FRED rate/carry panels -> Task 3. Dedicated FxSpotPortfolioSimulator with carry accrual + leverage cap + bankruptcy floor -> Task 7. Notional vol-target sizing -> Task 5. pip/bps cost model with metals -> Task 6. IDM (cluster_fn generalization + FX cluster map) -> Task 4. Trend + value reference strategies (price-only reuse) -> Task 8. Runner + `asset_class: fx` routing + config + universe + trade log -> Task 9. Walk-forward + PSR/DSR/PBO gate + registry + report -> Task 10. Real-data acceptance -> Task 11. All spec sections covered.
- Gap surfaced vs spec: the spec said `compute_div_mult` is reused "as-is"; it is NOT fully asset-agnostic (depends on futures `cluster_for`). Task 4 corrects this with a backward-compatible `cluster_fn` param. Documented.
- Out-of-scope items (PPP/CPI, empirical spreads, intraday calendar, live FX) are correctly absent; the `context` extension seam for PPP is preserved by keeping strategies on the `forecast_panel(close)` contract.

**Placeholder scan:** No TBD/TODO. Task 10 extracts the pure walk-forward helpers into `src/backtesting/walkforward_common.py` (single source of truth) and has both the futures and FX walk-forward scripts import them -- no duplication; the futures walk-forward test is the regression gate for the move.

**Type consistency:**
- `load_fx_daily_panel(pairs, start, end) -> MultiIndex (pair,{close,ret})` consumed by `run_fx_backtest` via `panel.xs("close", axis=1, level=1)` and by `build_quote_usd_panel(panel, pairs)` -- consistent.
- `build_quote_usd_panel` returns pair-columned DataFrame; `FxSpotPortfolioSimulator.run_sized` consumes it as `quote_usd_panel` -- consistent.
- `size_from_forecast_fx(forecast, capital, vol_target, base_to_usd, daily_vol, div_mult)` signature matches the call in the simulator -- consistent.
- `cost_fn(pair, units_traded, price, quote_to_usd)` produced by `_cost_fn_factory` and typed `CostFn` in the simulator -- consistent.
- `fx_round_trip_usd(pair, units_traded, price, quote_to_usd, tier, session, metals_bps)` matches both the factory call and its tests -- consistent.
- `compute_div_mult(..., cluster_fn=fx_cluster_for)` matches the Task 4 signature -- consistent.

No issues found requiring further change.
