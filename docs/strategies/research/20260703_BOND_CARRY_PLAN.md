# Bond Carry Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the 6 price-traded bond futures (ZT/ZF/ZN/TN/ZB/UB) real full-history carry via `carry = duration * (FRED CMT yield - DFF funding) / 100`, replacing the `return 0.0` v1 fallback, so futures carry runs on all 33 roots.

**Architecture:** A small FRED reader over the on-disk `alt_data/fred/{series}/daily.parquet` files, plus a tenor map, wired into `CarryCalculator.compute`'s bond branch. The micro-yield path (2YY/5YY/10Y/30Y) is untouched.

**Tech Stack:** Python 3.13, polars, pytest. Conda env `fintech`.

## Global Constraints

- **Python execution:** ALWAYS `/c/Users/qwqw1/anaconda3/envs/fintech/python.exe -m pytest <args>`. Never system Python.
- **ASCII only**; no `print()` (use `src.utils.logger`).
- **Base branch:** `feat/bond-carry` (checked out, off `main` @ 3de106a). Do NOT switch.
- **Point-in-time:** the FRED reader returns the latest value with `date <= d` (causal forward-fill); it must NEVER return a value dated after `d`. FRED DGS/DFF are as-of-date and unrevised.
- **Parameter-free:** durations (`DURATION_BY_ROOT`), the tenor map, and the funding series (`DFF`) are FIXED doctrine -- no fitting. Carry stays `trial_count=1`.
- **Isolation:** add the reader + edit ONLY the price-traded bond branch of `CarryCalculator.compute` (line 133). Do NOT change the micro-yield branch, other asset classes, the simulator, sizing, or the gate.
- **Data on disk (verified):** `alt_data/fred/{DGS2,DGS5,DGS10,DGS30,DFF}/daily.parquet`, schema `date,value`, all 1995-2026.

---

## Task 1: FRED series reader

**Files:**
- Create: `src/data/rates/__init__.py` (empty) and `src/data/rates/fred_reader.py`
- Test: `tests/data/rates/test_fred_reader.py`

**Interfaces:**
- Produces: `get_fred_series(series_id: str, d: date) -> float` -- the value of `series_id` as of the latest date `<= d`. Raises `FileNotFoundError` if the series parquet is missing, `ValueError` if `d` precedes the series start.

- [ ] **Step 1: Write the failing tests**

```python
# tests/data/rates/test_fred_reader.py
from datetime import date
import polars as pl
import pytest
from src.data.rates.fred_reader import get_fred_series


def test_reads_value_on_exact_date():
    # DGS10 has a print for a known trading day; value is a plausible yield.
    v = get_fred_series("DGS10", date(2024, 6, 3))
    assert 0.0 < v < 20.0


def test_forward_fills_weekend_causally():
    # A Sunday has no print -> returns the prior Friday's value (latest <= d).
    sun = get_fred_series("DGS10", date(2024, 6, 2))   # Sunday
    fri = get_fred_series("DGS10", date(2024, 5, 31))  # Friday
    assert sun == fri


def test_raises_before_series_start():
    with pytest.raises(ValueError):
        get_fred_series("DGS10", date(1990, 1, 1))  # series starts 1995


def test_missing_series_raises(monkeypatch, tmp_path):
    import src.data.rates.fred_reader as fr
    monkeypatch.setattr(fr, "get_local_storage_dir", lambda: tmp_path)
    with pytest.raises(FileNotFoundError):
        get_fred_series("NOPE", date(2024, 1, 1))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/c/.../python.exe -m pytest tests/data/rates/test_fred_reader.py -v`
Expected: FAIL -- module does not exist.

- [ ] **Step 3: Implement the reader**

```python
# src/data/rates/fred_reader.py
"""Point-in-time reader for downloaded FRED series (alt_data/fred/{id}/daily.parquet)."""
from __future__ import annotations

from datetime import date

import polars as pl

from src.settings import get_local_storage_dir


def get_fred_series(series_id: str, d: date) -> float:
    """Value of `series_id` as of the latest date <= d (causal forward-fill).

    Raises FileNotFoundError if the series is not downloaded, ValueError if
    `d` precedes the series' first observation.
    """
    fp = get_local_storage_dir() / "alt_data" / "fred" / series_id / "daily.parquet"
    if not fp.exists():
        raise FileNotFoundError(f"FRED series not downloaded: {fp}")
    df = pl.read_parquet(fp, columns=["date", "value"]).filter(pl.col("date") <= d)
    if df.height == 0:
        raise ValueError(f"no {series_id} observation on or before {d}")
    return float(df.sort("date")["value"][-1])
```
(Empty `src/data/rates/__init__.py`.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `/c/.../python.exe -m pytest tests/data/rates/test_fred_reader.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/data/rates/__init__.py src/data/rates/fred_reader.py tests/data/rates/test_fred_reader.py
git commit -m "feat(data): point-in-time FRED series reader (alt_data/fred)"
```

---

## Task 2: Bond carry from CMT yield in CarryCalculator

**Files:**
- Modify: `src/data/carry_calculator.py`
- Test: `tests/data/test_carry_calculator_bond.py`

**Interfaces:**
- Consumes: `get_fred_series` (Task 1).
- Produces: `CarryCalculator.compute(root, "bond", d)` returns `duration * (CMT_yield - DFF) / 100` for ZT/ZF/ZN/TN/ZB/UB (was `0.0`). New module const `_BOND_CMT_TENOR`.

**Context (verified, lines 121-133):** the bond branch computes `duration = DURATION_BY_ROOT.get(root)`, handles `MICRO_YIELD_ROOTS` via `derive_sofr`, then `return 0.0` for price-traded roots. `compute` computes `front/second/months` at the top (lines 109-113) for all classes; the bond branch does not use them, but they must resolve (the inert cache proves ZT/ZF/ZN/TN/ZB/UB have front/second data on ~4000+ days).

- [ ] **Step 1: Write the failing tests**

```python
# tests/data/test_carry_calculator_bond.py
from datetime import date
import pytest
import src.data.carry_calculator as cc
from src.data.carry_calculator import CarryCalculator


def _patch_contracts(monkeypatch):
    # avoid needing real per-contract data: bond branch ignores front/second,
    # but compute() resolves them up top. Return a valid 3-month gap.
    monkeypatch.setattr(
        CarryCalculator, "_find_front_second_close",
        lambda self, root, d: (f"{root}H4", 100.0, f"{root}M4", 100.0))
    monkeypatch.setattr(CarryCalculator, "_months_between", lambda self, a, b, r: 3)


def test_zn_bond_carry_uses_cmt_minus_funding(monkeypatch):
    _patch_contracts(monkeypatch)
    monkeypatch.setattr(cc, "get_fred_series",
                        lambda sid, d: {"DGS10": 4.2, "DFF": 5.3}[sid])
    # ZN duration 9: 9*(4.2-5.3)/100 = -0.099
    got = CarryCalculator().compute("ZN", "bond", date(2024, 1, 3))
    assert abs(got - (9.0 * (4.2 - 5.3) / 100.0)) < 1e-9
    assert got < 0  # inverted curve -> negative carry (short)


def test_positive_curve_gives_positive_carry(monkeypatch):
    _patch_contracts(monkeypatch)
    monkeypatch.setattr(cc, "get_fred_series",
                        lambda sid, d: {"DGS30": 3.5, "DFF": 0.1}[sid])
    got = CarryCalculator().compute("ZB", "bond", date(2013, 6, 3))  # ZB->DGS30, dur 17
    assert got > 0 and abs(got - (17.0 * (3.5 - 0.1) / 100.0)) < 1e-9


def test_tenor_map_covers_all_price_traded(monkeypatch):
    _patch_contracts(monkeypatch)
    seen = {}
    monkeypatch.setattr(cc, "get_fred_series",
                        lambda sid, d: seen.setdefault(sid, 4.0) or 4.0)
    for root in ["ZT", "ZF", "ZN", "TN", "ZB", "UB"]:
        v = CarryCalculator().compute(root, "bond", date(2024, 1, 3))
        assert v is not None
    assert {"DGS2", "DGS5", "DGS10", "DGS30", "DFF"} <= set(seen)


def test_micro_yield_path_unchanged(monkeypatch):
    # 10Y is a MICRO_YIELD_ROOT -> must use derive_sofr, NOT get_fred_series.
    _patch_contracts(monkeypatch)
    def _boom(*a, **k):
        raise AssertionError("micro-yield path must not call get_fred_series")
    monkeypatch.setattr(cc, "get_fred_series", _boom)
    monkeypatch.setattr(cc, "derive_sofr", lambda d: 5.0)
    # front close (100.0 from _patch) treated as the yield for micro roots
    got = CarryCalculator().compute("10Y", "bond", date(2024, 1, 3))
    assert got == 9.0 * (100.0 - 5.0) / 100.0  # duration 9 for 10Y
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/c/.../python.exe -m pytest tests/data/test_carry_calculator_bond.py -v`
Expected: FAIL -- price-traded roots return `0.0` (not the CMT formula); `get_fred_series` not imported.

- [ ] **Step 3: Add the tenor map + import**

Near `DURATION_BY_ROOT` in `carry_calculator.py`:
```python
_BOND_CMT_TENOR = {"ZT": "DGS2", "ZF": "DGS5", "ZN": "DGS10",
                   "TN": "DGS10", "ZB": "DGS30", "UB": "DGS30"}
_BOND_FUNDING_SERIES = "DFF"
```
Add the import:
```python
from src.data.rates.fred_reader import get_fred_series
```

- [ ] **Step 4: Replace the price-traded fallback**

Replace lines 130-133 (the `# ZT/ZF/... v1 fallback: return 0.0` block) with:
```python
            # ZT/ZF/ZN/TN/ZB/UB: price-traded bond futures. Yield is not in the
            # futures price; use the FRED constant-maturity yield for the tenor
            # minus the funding rate, scaled by duration (Carver bond carry).
            cmt_yield = get_fred_series(_BOND_CMT_TENOR[root], d)
            funding = get_fred_series(_BOND_FUNDING_SERIES, d)
            return duration * (cmt_yield - funding) / 100.0
```
(Keep the `MICRO_YIELD_ROOTS` branch above unchanged.)

- [ ] **Step 5: Run tests to verify they pass**

Run: `/c/.../python.exe -m pytest tests/data/test_carry_calculator_bond.py tests/data/test_carry_calculator.py -v`
Expected: PASS (4 new + the existing carry-calculator tests, unchanged).

- [ ] **Step 6: Integration check -- all 6 bonds compute nonzero on real data**

Run:
```bash
PYTHONPATH=. /c/.../python.exe -c "
from src.data.carry_calculator import CarryCalculator
from datetime import date
c=CarryCalculator()
for r in ['ZT','ZF','ZN','TN','ZB','UB']:
    h=c.compute_history(r,'bond',date(2024,1,1),date(2024,3,31))
    nz=int((h['carry']!=0).sum()) if h.height else 0
    print(f'{r}: {h.height} rows, nonzero={nz}')
    assert nz>0, f'{r} still inert'
print('ALL 6 BONDS NONZERO')
"
```
Expected: each bond prints nonzero rows + "ALL 6 BONDS NONZERO".

- [ ] **Step 7: Commit**

```bash
git add src/data/carry_calculator.py tests/data/test_carry_calculator_bond.py
git commit -m "feat(futures): real bond carry from FRED CMT yield - funding (6 price-traded roots)"
```

---

## Task 3: Acceptance -- rebuild cache + re-baseline (CONTROLLER-run, not TDD)

Controller-run after Tasks 1-2 are merged-ready and the GC/CL rebuild (already running) is done.

- [ ] **Step 1: Rebuild the 6 bond carry caches (+ confirm GC/CL)** (background)

```bash
cd "C:/Users/qwqw1/Dropbox/cs/github/Homeguard"
PYTHONPATH=. /c/.../python.exe scripts/data/build_carry_cache.py \
  --roots ZT ZF ZN TN ZB UB --start 2010-06-07 --end 2026-02-20 --jobs 6 \
  > .superpowers/sdd/rebuild_bonds.log 2>&1
```
On completion, audit all 33 carry parquets: every root non-zero, ~4000+ rows (reuse the audit snippet). Confirm GC/CL are now full-range (the parallel GC/CL rebuild finished).

- [ ] **Step 2: Re-baseline the carry walk-forward on the complete 33-root cache** (background)

```bash
PYTHONPATH=. /c/.../python.exe scripts/backtest_scripts/run_carver_walkforward.py \
  --config config/backtesting/carry_broad.yaml \
  --report docs/reports/futures/CARRY_BROAD_READINESS.md \
  --jobs <cores> > .superpowers/sdd/carry_rebaseline.log 2>&1
```

- [ ] **Step 3: Compare + diagnose**

Compare the corrected metrics to the 27-root baseline (OOS Sharpe 0.88, PBO 0.63, kurt 33.5).
Quantify: how much did completing the data (GC/CL + 6 bonds) move PBO / kurtosis / Sharpe --
i.e. how much concentration was a DATA HOLE vs intrinsic. Check whether the `nan` windows
W11/W12 (2023-25) are now populated (were they caused by the cache gaps?). Summarize for the
user; this corrected number is the baseline the XS-carry / IDM / combine variants are measured
against.

- [ ] **Step 4: Commit the corrected report**

```bash
git add -f docs/reports/futures/CARRY_BROAD_READINESS.md
git commit -m "docs(futures): re-baseline carry on complete 33-root cache (bonds + GC/CL)"
```

---

## Self-Review

- **Spec coverage:** Task 1 = FRED reader; Task 2 = bond-branch CMT carry + tenor map; Task 3 = rebuild + re-baseline. Covers the design.
- **Placeholder scan:** none -- reader, tenor map, branch edit, and tests are all concrete.
- **Type consistency:** `get_fred_series(series_id: str, d: date) -> float`; `_BOND_CMT_TENOR: dict[str,str]`; bond branch returns float `duration*(cmt-funding)/100`.
- **Point-in-time:** reader filters `date <= d` then takes the last -> causal; a test asserts weekend forward-fill equals the prior print; a test asserts raise-before-start.
- **Isolation / parameter-free:** only the price-traded bond branch changes; micro-yield path proven unchanged by `test_micro_yield_path_unchanged`; durations/map/funding are fixed doctrine (trial_count unaffected).
- **Integration:** Task 2 Step 6 proves all 6 bonds go nonzero on real data before the expensive rebuild.
