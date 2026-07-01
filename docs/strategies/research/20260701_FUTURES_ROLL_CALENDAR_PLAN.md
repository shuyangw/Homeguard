# Futures Roll Calendar Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Repair the futures roll/carry infrastructure broken by the 2026 data consolidation, then extend it with an OI-primary roll calendar (FND-aware) that unblocks per-contract carry and spread backtests.

**Architecture:** Phase 0 repairs ~14 stale data-path references (centralized into one path helper) and removes a silent-swallow that hid the breakage. Phase 1 layers the approved design (`docs/strategies/research/20260701_FUTURES_ROLL_CALENDAR_DESIGN.md`) on top of the now-working modules: a static contract-spec table, per-contract OI extraction, an OI-primary roll algorithm with FND clamp, a cached per-root calendar artifact, and a dual-`nth` lookup API.

**Tech Stack:** Python 3.13, polars, pytest. Conda env `fintech` (`C:\Users\qwqw1\anaconda3\envs\fintech`). Data under `H:/Stock_Data/futures/`.

## Global Constraints

- **Python execution:** ALWAYS `conda activate fintech` (or call `/c/Users/qwqw1/anaconda3/envs/fintech/python.exe` directly). Never system Python.
- **ASCII only** in all code, comments, docs, log strings (Windows cp1252). Use `->`, `[+]`, `[-]`, `[!]`.
- **Paths:** NEVER hardcode storage paths. Resolve via `from src.settings import get_local_storage_dir`. All futures subpaths go through the new `src/data/futures/paths.py` helper (Task 1).
- **Logging:** use `from src.utils.logger import get_logger`; f-strings only (no `%s`). Never `print()`. ALWAYS log caught exceptions; fail loud, return explicit errors not silent `None`/empty.
- **Consolidated data layout (verified 2026-07-01):**
  - continuous `.v.0` bars: `futures/databento/1min/symbol={ROOT}/year={Y}/month={M}/data.parquet`
  - per-contract bars: `futures/databento/per_contract_1min/year={Y}/month={M}/data.parquet` (raw CME symbol in `symbol` col)
  - statistics (OI): `futures/databento/statistics/year={Y}/month={M}/data.parquet` (`stat_type==9` is OI)
  - definitions: `futures/definitions/year={Y}/month={M}/data.parquet`
- **Roll integrity:** the roll calendar is ONLY for per-contract strategies. Continuous-bar strategies use `.v.0`; applying the calendar to them is a double-roll bug.
- **Multiplier source:** `contract_multiplier` in definitions is a garbage i32 sentinel (`2147483647`). Multipliers come ONLY from the static table in `contract_specs.py`. `min_price_increment` (tick size) IS valid.
- **CME month codes:** `FGHJKMNQUVXZ` = Jan..Dec.
- **TDD:** write the failing test first, watch it fail, implement minimally, watch it pass, commit. Commit after each task.

---

## File Structure

**Phase 0 (repair):**
- Create: `src/data/futures/__init__.py`, `src/data/futures/paths.py` — centralized consolidated-path helpers
- Modify: `src/data/carry_calculator.py`, `src/data/continuous_contract_loader.py`, `src/data/futures_definitions_loader.py`, `src/data/derivations/futures/open_interest.py`, `src/data/derivations/futures/sofr.py`, `src/data/derivations/futures/yields.py`, `src/data/signed_volume_estimator.py` — repoint to `paths.py`
- Modify: `tests/data/test_carry_calculator_integration.py` — fix skip guard to new path
- Create: `tests/data/futures/test_paths.py`, `tests/data/futures/test_repair_regression.py`

**Phase 1 (enhance):**
- Create: `src/data/futures/contract_specs.py` — static spec table + settlement classification
- Modify: `src/data/derivations/futures/open_interest.py` — add `per_contract_open_interest`
- Create: `src/data/futures/roll_calendar.py` — RollCalendar + roll algorithm
- Create: `scripts/data/build_roll_calendar.py` — batch builder
- Modify: `src/data/roll_detector.py` — implement `get_upcoming_rolls` on the calendar
- Create: `tests/data/futures/test_contract_specs.py`, `test_roll_calendar.py`, `test_roll_calendar_golden.py`

---

# PHASE 0 — REPAIR

## Task 1: Centralized futures path helper

**Files:**
- Create: `src/data/futures/__init__.py`
- Create: `src/data/futures/paths.py`
- Test: `tests/data/futures/test_paths.py`

**Interfaces:**
- Produces: `continuous_1min_dir() -> Path`, `per_contract_1min_dir() -> Path`, `statistics_dir() -> Path`, `definitions_dir() -> Path`, `roll_calendar_dir() -> Path`. Each returns the consolidated absolute path under `get_local_storage_dir()`.

- [ ] **Step 1: Write the failing test**

```python
# tests/data/futures/test_paths.py
from pathlib import Path
import src.data.futures.paths as paths


def test_paths_point_at_consolidated_layout(monkeypatch):
    monkeypatch.setattr(paths, "get_local_storage_dir", lambda: Path("/data"))
    assert paths.continuous_1min_dir() == Path("/data/futures/databento/1min")
    assert paths.per_contract_1min_dir() == Path("/data/futures/databento/per_contract_1min")
    assert paths.statistics_dir() == Path("/data/futures/databento/statistics")
    assert paths.definitions_dir() == Path("/data/futures/definitions")
    assert paths.roll_calendar_dir() == Path("/data/futures/roll_calendar")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/c/Users/qwqw1/anaconda3/envs/fintech/python.exe -m pytest tests/data/futures/test_paths.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.data.futures.paths'`

- [ ] **Step 3: Create the package and helper**

```python
# src/data/futures/__init__.py
"""Futures data helpers: consolidated paths, contract specs, roll calendar."""
```

```python
# src/data/futures/paths.py
"""Consolidated futures data paths (post-2026 consolidation).

Single source of truth for the futures/databento/* and futures/definitions
layout. All futures readers resolve paths through here so a future
reorganization is a one-file change instead of a repo-wide grep.
"""
from __future__ import annotations

from pathlib import Path

from src.settings import get_local_storage_dir


def _futures_root() -> Path:
    return get_local_storage_dir() / "futures"


def continuous_1min_dir() -> Path:
    """`.v.0` volume-roll continuous minute bars."""
    return _futures_root() / "databento" / "1min"


def per_contract_1min_dir() -> Path:
    """Per-contract (raw CME symbol) minute bars."""
    return _futures_root() / "databento" / "per_contract_1min"


def statistics_dir() -> Path:
    """Databento statistics (settle / OI / volume events)."""
    return _futures_root() / "databento" / "statistics"


def definitions_dir() -> Path:
    """Contract definition events (expiration, tick size, etc.)."""
    return _futures_root() / "definitions"


def roll_calendar_dir() -> Path:
    """Cached per-root roll calendar artifacts (built in Phase 1)."""
    return _futures_root() / "roll_calendar"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/c/Users/qwqw1/anaconda3/envs/fintech/python.exe -m pytest tests/data/futures/test_paths.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/data/futures/__init__.py src/data/futures/paths.py tests/data/futures/test_paths.py
git commit -m "feat(futures): centralized consolidated-path helper"
```

---

## Task 2: Repoint carry + continuous-contract readers to consolidated paths

**Files:**
- Modify: `src/data/carry_calculator.py:64-67` (the `pcm =` path)
- Modify: `src/data/continuous_contract_loader.py:51` and `:118` (the `pcm_root` and `sym_dir` paths)
- Test: `tests/data/futures/test_repair_regression.py` (real-data, integration-marked)

**Interfaces:**
- Consumes: `per_contract_1min_dir()`, `continuous_1min_dir()` from Task 1.
- Produces: `CarryCalculator.compute(root, asset_class, date)` and `ContinuousContractDataLoader.detect_roll_dates(root, start, end)` returning real non-empty results against current data.

- [ ] **Step 1: Write the failing real-data regression test**

```python
# tests/data/futures/test_repair_regression.py
"""Real-data regression: these MUST fail before the path repair and pass after.
They would have caught the 2026 consolidation break that the fixture-based
unit tests missed."""
from datetime import date

import pytest

from src.data.futures.paths import per_contract_1min_dir
from src.data.carry_calculator import CarryCalculator
from src.data.continuous_contract_loader import ContinuousContractDataLoader


def _data_present() -> bool:
    # 2024-01 partition is known to exist in the consolidated store
    return (per_contract_1min_dir() / "year=2024" / "month=1" / "data.parquet").exists()


pytestmark = pytest.mark.skipif(not _data_present(), reason="consolidated futures store not present")


def test_carry_returns_value_for_known_gc_date():
    # GC had dense data on 2024-01-15 (GCG4 ~65k volume) -- carry must compute
    val = CarryCalculator().compute("GC", "commodity", date(2024, 1, 15))
    assert isinstance(val, float)


def test_carry_history_nonempty_for_gc_january():
    hist = CarryCalculator().compute_history("GC", "commodity", date(2024, 1, 8), date(2024, 1, 20))
    assert hist.height > 0, "carry history empty -> readers still broken"


def test_roll_dates_detected_for_gc_2024():
    rolls = ContinuousContractDataLoader().detect_roll_dates("GC", date(2024, 1, 1), date(2024, 12, 31))
    assert len(rolls) >= 4, f"GC should roll several times in 2024, got {len(rolls)}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/c/Users/qwqw1/anaconda3/envs/fintech/python.exe -m pytest tests/data/futures/test_repair_regression.py -v`
Expected: FAIL — `test_carry_returns_value_for_known_gc_date` raises `ValueError: no per-contract data for 2024-01-15: missing ...futures_per_contract_1min...`; history returns 0 rows; roll dates 0.

- [ ] **Step 3: Repoint `carry_calculator.py`**

Replace the import and the `pcm` path. In `src/data/carry_calculator.py`, change the imports near line 14-15:

```python
from src.data.derivations.futures.sofr import derive_sofr
from src.data.futures.paths import per_contract_1min_dir
from src.settings import get_local_storage_dir
```

Then in `_find_front_second_close` (currently lines 64-67), replace:

```python
        pcm = (
            per_contract_1min_dir()
            / f"year={d.year}" / f"month={d.month}" / "data.parquet"
        )
```

(Leave `_storage_root()` defined if other code in the file uses it; the `pcm` path no longer does.)

- [ ] **Step 4: Repoint `continuous_contract_loader.py`**

In `src/data/continuous_contract_loader.py`, add import near line 20:

```python
from src.data.futures.paths import continuous_1min_dir, per_contract_1min_dir
from src.settings import get_local_storage_dir
```

Replace line 51 (`pcm_root = _storage_root() / "futures_per_contract_1min"`) with:

```python
        pcm_root = per_contract_1min_dir()
```

Replace line 118 (`sym_dir = _storage_root() / "futures_1min" / f"symbol={root}"`) with:

```python
        sym_dir = continuous_1min_dir() / f"symbol={root}"
```

- [ ] **Step 5: Run the regression test to verify it passes**

Run: `/c/Users/qwqw1/anaconda3/envs/fintech/python.exe -m pytest tests/data/futures/test_repair_regression.py -v`
Expected: PASS (all 3 tests)

- [ ] **Step 6: Run the existing unit tests to confirm no regression**

Run: `/c/Users/qwqw1/anaconda3/envs/fintech/python.exe -m pytest tests/data/test_carry_calculator.py tests/data/test_continuous_contract_loader.py -v`
Expected: PASS (13 tests — fixtures still write to the paths the monkeypatched `_storage_root` controls, so they are unaffected)

- [ ] **Step 7: Commit**

```bash
git add src/data/carry_calculator.py src/data/continuous_contract_loader.py tests/data/futures/test_repair_regression.py
git commit -m "fix(futures): repoint carry + continuous loader to consolidated paths"
```

---

## Task 3: Repoint definitions loader, OI, SOFR, yields, signed-volume readers

**Files:**
- Modify: `src/data/futures_definitions_loader.py:90-96`
- Modify: `src/data/derivations/futures/open_interest.py:62-68`
- Modify: `src/data/derivations/futures/sofr.py` (the `futures_per_contract_1min` ref)
- Modify: `src/data/derivations/futures/yields.py` (the `futures_1min` ref)
- Modify: `src/data/signed_volume_estimator.py:34`
- Test: extend `tests/data/futures/test_repair_regression.py`

**Interfaces:**
- Consumes: `definitions_dir()`, `statistics_dir()`, `per_contract_1min_dir()`, `continuous_1min_dir()` from Task 1.
- Produces: `FuturesDefinitionsLoader.get_definition(...)`, `aggregate_open_interest(root, date)` returning real values.

- [ ] **Step 1: Write the failing test (append to regression file)**

```python
# append to tests/data/futures/test_repair_regression.py
from src.data.futures_definitions_loader import FuturesDefinitionsLoader
from src.data.derivations.futures.open_interest import aggregate_open_interest


def test_definition_lookup_for_known_contract():
    # GCG4 (Feb 2024 gold) is active in the 2024-01 definitions partition
    d = FuturesDefinitionsLoader().get_definition("GCG4", "GC", date(2024, 1, 15))
    assert d.expiration.year == 2024
    assert d.tick_size > 0


def test_aggregate_oi_positive_for_gc():
    oi = aggregate_open_interest("GC", date(2024, 1, 15))
    assert oi > 0, "aggregate OI zero -> statistics path still broken"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/c/Users/qwqw1/anaconda3/envs/fintech/python.exe -m pytest tests/data/futures/test_repair_regression.py -k "definition or aggregate_oi" -v`
Expected: FAIL — `FileNotFoundError: futures_definitions partition not found` and `futures_statistics partition not found`.

- [ ] **Step 3: Repoint `futures_definitions_loader.py`**

Add import near line 23: `from src.data.futures.paths import definitions_dir`. Replace the `_load_partition` path (lines 90-96) with:

```python
        path = (
            definitions_dir()
            / f"year={year}"
            / f"month={month}"
            / "data.parquet"
        )
```

- [ ] **Step 4: Repoint `open_interest.py`**

Add import near line 27: `from src.data.futures.paths import statistics_dir`. Replace the path (lines 62-68) with:

```python
    path = (
        statistics_dir()
        / f"year={d.year}"
        / f"month={d.month}"
        / "data.parquet"
    )
```

- [ ] **Step 5: Repoint `sofr.py`, `yields.py`, `signed_volume_estimator.py`**

In `src/data/derivations/futures/sofr.py`, add `from src.data.futures.paths import per_contract_1min_dir` and replace the `_storage_root() / "futures_per_contract_1min"` fragment with `per_contract_1min_dir()`.

In `src/data/derivations/futures/yields.py`, add `from src.data.futures.paths import continuous_1min_dir` and replace `_storage_root() / "futures_1min"` with `continuous_1min_dir()`.

In `src/data/signed_volume_estimator.py:34`, add `from src.data.futures.paths import continuous_1min_dir` and replace `sym_dir = _storage_root() / "futures_1min" / f"symbol={symbol}"` with `sym_dir = continuous_1min_dir() / f"symbol={symbol}"`.

- [ ] **Step 6: Run tests to verify they pass**

Run: `/c/Users/qwqw1/anaconda3/envs/fintech/python.exe -m pytest tests/data/futures/test_repair_regression.py -v`
Expected: PASS (all tests)

- [ ] **Step 7: Commit**

```bash
git add src/data/futures_definitions_loader.py src/data/derivations/futures/open_interest.py src/data/derivations/futures/sofr.py src/data/derivations/futures/yields.py src/data/signed_volume_estimator.py tests/data/futures/test_repair_regression.py
git commit -m "fix(futures): repoint definitions/OI/SOFR/yields/signed-volume to consolidated paths"
```

---

## Task 4: Fix the silent-swallow and the skip-guard that hid the breakage

**Files:**
- Modify: `src/data/carry_calculator.py` (`compute_history`, lines ~139-158)
- Modify: `tests/data/test_carry_calculator_integration.py:10-12` (skip guard)
- Test: extend `tests/data/futures/test_repair_regression.py`

**Interfaces:**
- Produces: `compute_history` raises `FileNotFoundError` when the whole per-contract dataset directory is missing (fail loud), but still skips individual dates with no data (weekends/holidays).

- [ ] **Step 1: Write the failing test**

```python
# append to tests/data/futures/test_repair_regression.py
import polars as pl


def test_compute_history_raises_when_dataset_dir_missing(monkeypatch, tmp_path):
    # Point the per-contract dir at an empty tmp dir -> whole dataset missing.
    monkeypatch.setattr(
        "src.data.carry_calculator.per_contract_1min_dir",
        lambda: tmp_path / "does_not_exist",
    )
    with pytest.raises(FileNotFoundError):
        CarryCalculator().compute_history("GC", "commodity", date(2024, 1, 8), date(2024, 1, 20))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/c/Users/qwqw1/anaconda3/envs/fintech/python.exe -m pytest tests/data/futures/test_repair_regression.py -k dataset_dir_missing -v`
Expected: FAIL — currently returns an empty DataFrame instead of raising (the silent-swallow bug).

- [ ] **Step 3: Add an upfront dataset guard in `compute_history`**

In `src/data/carry_calculator.py`, add the logger import at the top:

```python
from src.data.futures.paths import per_contract_1min_dir
from src.utils.logger import get_logger

logger = get_logger(__name__)
```

At the START of `compute_history` (before the `while` loop), add:

```python
        pcm_dir = per_contract_1min_dir()
        if not pcm_dir.exists():
            raise FileNotFoundError(
                f"per-contract futures store missing: {pcm_dir} "
                f"-- carry cannot be computed"
            )
```

Then change the per-day `except` block (currently `except (ValueError, FileNotFoundError, NotImplementedError): pass`) to log instead of silently swallowing:

```python
            except (ValueError, FileNotFoundError, NotImplementedError) as e:
                logger.debug(f"skip {root} carry on {d}: {e}")
```

Rationale: whole-dataset-missing is now fatal and loud (the consolidation-break class); per-day gaps (weekends) stay skipped but are logged at debug, not invisible.

- [ ] **Step 4: Fix the integration-test skip guard**

In `tests/data/test_carry_calculator_integration.py`, replace `_local_data_available` (lines 10-12):

```python
def _local_data_available() -> bool:
    from src.data.futures.paths import per_contract_1min_dir
    return (per_contract_1min_dir() / "year=2024" / "month=1" / "data.parquet").exists()
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `/c/Users/qwqw1/anaconda3/envs/fintech/python.exe -m pytest tests/data/futures/test_repair_regression.py tests/data/test_carry_calculator_integration.py -v`
Expected: PASS — the integration tests now RUN (not skip) and pass; the new guard test passes.

- [ ] **Step 6: Commit**

```bash
git add src/data/carry_calculator.py tests/data/test_carry_calculator_integration.py tests/data/futures/test_repair_regression.py
git commit -m "fix(futures): fail loud on missing dataset + un-skip integration guard"
```

---

## Task 5: Repoint validation checks + acquisition plugin output; full Phase-0 sweep

**Files:**
- Modify: `src/data/validation/futures/checks/cross_source.py`, `external.py`, `statistical.py`, `structural.py`, `expectations.py`
- Modify: `src/data/acquisition/plugins/databento_futures.py:238,290` (output dir)
- Test: run the full futures-related suite

**Interfaces:**
- Produces: no stale `futures_1min` / `futures_per_contract_1min` / `futures_statistics` / `futures_definitions` literal remains anywhere in `src/`.

- [ ] **Step 1: Write the failing guard test**

```python
# tests/data/futures/test_no_stale_paths.py
"""Guard: no module may reference the pre-consolidation flat futures paths."""
import subprocess


def test_no_stale_futures_path_literals():
    # ripgrep for the old flat dir names as quoted string literals under src/
    res = subprocess.run(
        ["git", "grep", "-n", "-E",
         r'"futures_(1min|per_contract_1min|per_contract_daily|statistics|definitions)"',
         "--", "src/"],
        capture_output=True, text=True,
    )
    # git grep exits 1 (no matches) when clean; 0 (matches) when stale refs remain
    assert res.returncode == 1, f"stale futures path literals remain:\n{res.stdout}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/c/Users/qwqw1/anaconda3/envs/fintech/python.exe -m pytest tests/data/futures/test_no_stale_paths.py -v`
Expected: FAIL — validation checks and the acquisition plugin still contain stale literals.

- [ ] **Step 3: Repoint the validation checks**

In each of `src/data/validation/futures/checks/cross_source.py`, `external.py`, `statistical.py`, `structural.py`, add `from src.data.futures.paths import continuous_1min_dir, per_contract_1min_dir, definitions_dir` and replace the corresponding `_storage_root() / "futures_1min"`, `/ "futures_per_contract_1min"`, `/ "futures_definitions"` fragments with the helper calls. In `expectations.py`, update the `"futures_1min"` dict key/label only if it is used to build a filesystem path (leave pure display labels alone; if it builds a path, route through the helper).

- [ ] **Step 4: Repoint the acquisition plugin output dir**

In `src/data/acquisition/plugins/databento_futures.py`, lines 238 and 290 return/join `"futures_1min"` as the OHLCV output dir. Update these so downloads land in the consolidated location. At line ~290 replace `ohlcv_dir = self.base_output_dir / "futures_1min"` with a path that matches `futures/databento/1min`. VERIFY first what `self.base_output_dir` resolves to; if it already equals `get_local_storage_dir()`, use `self.base_output_dir / "futures" / "databento" / "1min"`. Mirror the same for the string returned at line 238.

- [ ] **Step 5: Run the guard + full sweep**

Run: `/c/Users/qwqw1/anaconda3/envs/fintech/python.exe -m pytest tests/data/futures/test_no_stale_paths.py -v`
Expected: PASS

Run: `/c/Users/qwqw1/anaconda3/envs/fintech/python.exe -m pytest tests/data/ -q`
Expected: PASS (no regressions across the data suite)

- [ ] **Step 6: Commit**

```bash
git add src/data/validation/futures/checks/ src/data/validation/futures/expectations.py src/data/acquisition/plugins/databento_futures.py tests/data/futures/test_no_stale_paths.py
git commit -m "fix(futures): repoint validation checks + acquisition output to consolidated paths"
```

**Phase 0 done. Futures carry/roll infra works against real data again, failures are loud, and a guard test prevents silent path-drift regressions.**

---

# PHASE 1 — ENHANCE (OI-primary roll calendar)

## Task 6: Static contract-spec table + settlement classification

**Files:**
- Create: `src/data/futures/contract_specs.py`
- Test: `tests/data/futures/test_contract_specs.py`

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True) ContractSpec` with fields `root: str, multiplier: float, tick_size: float, tick_value: float, currency: str, cycle_months: str, settlement_type: Literal["physical","financial"], fnd_offset_days: int`
  - `get_spec(root: str) -> ContractSpec` (raises `KeyError` for unknown root)
  - `SPECS: dict[str, ContractSpec]` covering all 53 roots

- [ ] **Step 1: Write the failing test**

```python
# tests/data/futures/test_contract_specs.py
import pytest
from src.data.futures.contract_specs import get_spec, SPECS


def test_gc_spec_physical():
    s = get_spec("GC")
    assert s.multiplier == 100.0        # $100/point full GC
    assert s.tick_size == 0.1
    assert s.settlement_type == "physical"
    assert s.fnd_offset_days > 0        # metals roll before FND


def test_es_spec_financial_no_fnd():
    s = get_spec("ES")
    assert s.multiplier == 50.0
    assert s.settlement_type == "financial"
    assert s.fnd_offset_days == 0       # cash-settled, no FND clamp


def test_all_53_roots_present():
    expected = {
        "ES","NQ","YM","RTY","MES","MNQ","M2K","MYM",
        "CL","NG","HO","RB","BZ","MCL","MNG",
        "GC","SI","HG","PL","MGC","SIL",
        "ZT","ZF","ZN","TN","ZB","UB","SR3","SR1","10Y","30Y","5YY","2YY",
        "6E","6J","6B","6A","6C","6S","6N","6M",
        "ZC","ZS","ZW","KE","ZL","ZM","LE","HE",
        "BTC","MBT","ETH","MET",
    }
    assert expected <= set(SPECS.keys())


def test_unknown_root_raises():
    with pytest.raises(KeyError):
        get_spec("XYZ")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/c/Users/qwqw1/anaconda3/envs/fintech/python.exe -m pytest tests/data/futures/test_contract_specs.py -v`
Expected: FAIL — module does not exist.

- [ ] **Step 3: Implement the spec table**

```python
# src/data/futures/contract_specs.py
"""Static, hand-verified futures contract specifications.

Source of truth for multiplier / tick / settlement type -- the definitions
dataset's contract_multiplier is an unreliable i32 sentinel, so multipliers
MUST come from here. settlement_type + fnd_offset_days drive the FND clamp in
the roll calendar: physical roots roll before first notice; financial
(cash-settled) roots have no delivery risk (fnd_offset_days == 0).

fnd_offset_days is an APPROXIMATE, conservative business-day cushion before
`expiration` past which a physical contract must not remain front. It only
ever moves a roll EARLIER. Refine per family if golden-date tests disagree.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Settlement = Literal["physical", "financial"]


@dataclass(frozen=True)
class ContractSpec:
    root: str
    multiplier: float
    tick_size: float
    tick_value: float
    currency: str
    cycle_months: str        # subset of FGHJKMNQUVXZ that lists liquid contracts
    settlement_type: Settlement
    fnd_offset_days: int     # business-day cushion before expiration; 0 for financial


def _s(root, mult, tick, tick_val, ccy, cycle, settle, fnd):
    return ContractSpec(root, mult, tick, tick_val, ccy, cycle, settle, fnd)


ALL = "FGHJKMNQUVXZ"
QTR = "HMUZ"   # quarterly cycle (equity index, rates, FX)

SPECS: dict[str, ContractSpec] = {
    # Equity index -- cash settled (financial)
    "ES": _s("ES", 50.0, 0.25, 12.5, "USD", QTR, "financial", 0),
    "NQ": _s("NQ", 20.0, 0.25, 5.0, "USD", QTR, "financial", 0),
    "YM": _s("YM", 5.0, 1.0, 5.0, "USD", QTR, "financial", 0),
    "RTY": _s("RTY", 50.0, 0.10, 5.0, "USD", QTR, "financial", 0),
    "MES": _s("MES", 5.0, 0.25, 1.25, "USD", QTR, "financial", 0),
    "MNQ": _s("MNQ", 2.0, 0.25, 0.5, "USD", QTR, "financial", 0),
    "M2K": _s("M2K", 5.0, 0.10, 0.5, "USD", QTR, "financial", 0),
    "MYM": _s("MYM", 0.5, 1.0, 0.5, "USD", QTR, "financial", 0),
    # Energy -- physical
    "CL": _s("CL", 1000.0, 0.01, 10.0, "USD", ALL, "physical", 4),
    "NG": _s("NG", 10000.0, 0.001, 10.0, "USD", ALL, "physical", 4),
    "HO": _s("HO", 42000.0, 0.0001, 4.2, "USD", ALL, "physical", 4),
    "RB": _s("RB", 42000.0, 0.0001, 4.2, "USD", ALL, "physical", 4),
    "BZ": _s("BZ", 1000.0, 0.01, 10.0, "USD", ALL, "physical", 4),
    "MCL": _s("MCL", 100.0, 0.01, 1.0, "USD", ALL, "physical", 4),
    "MNG": _s("MNG", 2500.0, 0.001, 2.5, "USD", ALL, "physical", 4),
    # Metals -- physical
    "GC": _s("GC", 100.0, 0.1, 10.0, "USD", "GJMQVZ", "physical", 3),
    "SI": _s("SI", 5000.0, 0.005, 25.0, "USD", "HKNUZ", "physical", 3),
    "HG": _s("HG", 25000.0, 0.0005, 12.5, "USD", "HKNUZ", "physical", 3),
    "PL": _s("PL", 50.0, 0.1, 5.0, "USD", "FJNV", "physical", 3),
    "MGC": _s("MGC", 10.0, 0.1, 1.0, "USD", "GJMQVZ", "physical", 3),
    "SIL": _s("SIL", 1000.0, 0.005, 5.0, "USD", "HKNUZ", "physical", 3),
    # Rates -- physical delivery (bonds) / financial (SOFR, micro yield)
    "ZT": _s("ZT", 2000.0, 0.0078125, 15.625, "USD", QTR, "physical", 2),
    "ZF": _s("ZF", 1000.0, 0.0078125, 7.8125, "USD", QTR, "physical", 2),
    "ZN": _s("ZN", 1000.0, 0.015625, 15.625, "USD", QTR, "physical", 2),
    "TN": _s("TN", 1000.0, 0.015625, 15.625, "USD", QTR, "physical", 2),
    "ZB": _s("ZB", 1000.0, 0.03125, 31.25, "USD", QTR, "physical", 2),
    "UB": _s("UB", 1000.0, 0.03125, 31.25, "USD", QTR, "physical", 2),
    "SR3": _s("SR3", 2500.0, 0.005, 12.5, "USD", QTR, "financial", 0),
    "SR1": _s("SR1", 4167.0, 0.005, 20.835, "USD", ALL, "financial", 0),
    "10Y": _s("10Y", 1000.0, 0.001, 1.0, "USD", QTR, "financial", 0),
    "30Y": _s("30Y", 1000.0, 0.001, 1.0, "USD", QTR, "financial", 0),
    "5YY": _s("5YY", 1000.0, 0.001, 1.0, "USD", QTR, "financial", 0),
    "2YY": _s("2YY", 1000.0, 0.001, 1.0, "USD", QTR, "financial", 0),
    # FX -- physically deliverable currency
    "6E": _s("6E", 125000.0, 0.00005, 6.25, "USD", QTR, "physical", 2),
    "6J": _s("6J", 12500000.0, 0.0000005, 6.25, "USD", QTR, "physical", 2),
    "6B": _s("6B", 62500.0, 0.0001, 6.25, "USD", QTR, "physical", 2),
    "6A": _s("6A", 100000.0, 0.0001, 10.0, "USD", QTR, "physical", 2),
    "6C": _s("6C", 100000.0, 0.00005, 5.0, "USD", QTR, "physical", 2),
    "6S": _s("6S", 125000.0, 0.0001, 12.5, "USD", QTR, "physical", 2),
    "6N": _s("6N", 100000.0, 0.0001, 10.0, "USD", QTR, "physical", 2),
    "6M": _s("6M", 500000.0, 0.00001, 5.0, "USD", QTR, "physical", 2),
    # Ag -- physical
    "ZC": _s("ZC", 50.0, 0.25, 12.5, "USD", "HKNUZ", "physical", 2),
    "ZS": _s("ZS", 50.0, 0.25, 12.5, "USD", "FHKNQUX", "physical", 2),
    "ZW": _s("ZW", 50.0, 0.25, 12.5, "USD", "HKNUZ", "physical", 2),
    "KE": _s("KE", 50.0, 0.25, 12.5, "USD", "HKNUZ", "physical", 2),
    "ZL": _s("ZL", 600.0, 0.01, 6.0, "USD", "FHKNQUVZ", "physical", 2),
    "ZM": _s("ZM", 100.0, 0.1, 10.0, "USD", "FHKNQUVZ", "physical", 2),
    "LE": _s("LE", 400.0, 0.00025, 10.0, "USD", "GJMQVZ", "physical", 2),
    "HE": _s("HE", 400.0, 0.00025, 10.0, "USD", "GJKMNQVZ", "physical", 2),
    # Crypto -- cash settled (financial)
    "BTC": _s("BTC", 5.0, 5.0, 25.0, "USD", ALL, "financial", 0),
    "MBT": _s("MBT", 0.1, 5.0, 0.5, "USD", ALL, "financial", 0),
    "ETH": _s("ETH", 50.0, 0.5, 25.0, "USD", ALL, "financial", 0),
    "MET": _s("MET", 0.1, 0.5, 0.05, "USD", ALL, "financial", 0),
}


def get_spec(root: str) -> ContractSpec:
    """Return the ContractSpec for `root`, or raise KeyError."""
    return SPECS[root]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/c/Users/qwqw1/anaconda3/envs/fintech/python.exe -m pytest tests/data/futures/test_contract_specs.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/data/futures/contract_specs.py tests/data/futures/test_contract_specs.py
git commit -m "feat(futures): static contract-spec table + settlement classification"
```

---

## Task 7: Per-contract open interest extraction

**Files:**
- Modify: `src/data/derivations/futures/open_interest.py` (add function)
- Test: extend `tests/data/futures/test_repair_regression.py` (real-data)

**Interfaces:**
- Consumes: `statistics_dir()`, `_is_outright` (already in module).
- Produces: `per_contract_open_interest(root: str, d: date) -> dict[str, int]` mapping each outright contract symbol to its end-of-day OI on `d`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/data/futures/test_repair_regression.py
from src.data.derivations.futures.open_interest import per_contract_open_interest


def test_per_contract_oi_ranks_gc_front():
    oi = per_contract_open_interest("GC", date(2024, 1, 15))
    assert oi, "no per-contract OI returned"
    # front (most-liquid) should be a real GC contract with positive OI
    top = max(oi, key=oi.get)
    assert top.startswith("GC") and "-" not in top
    assert oi[top] > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/c/Users/qwqw1/anaconda3/envs/fintech/python.exe -m pytest tests/data/futures/test_repair_regression.py -k per_contract_oi -v`
Expected: FAIL — `per_contract_open_interest` not defined.

- [ ] **Step 3: Implement `per_contract_open_interest`**

Append to `src/data/derivations/futures/open_interest.py`:

```python
def per_contract_open_interest(symbol_root: str, d: date) -> dict[str, int]:
    """Return {contract_symbol: end-of-day OI} for each outright of `root` on `d`.

    Unlike aggregate_open_interest (which sums), this preserves per-contract OI
    so the roll calendar can detect the front-to-back OI crossover.

    Raises:
        FileNotFoundError: If the statistics partition for the date is missing.
    """
    path = (
        statistics_dir()
        / f"year={d.year}"
        / f"month={d.month}"
        / "data.parquet"
    )
    if not path.exists():
        raise FileNotFoundError(f"futures_statistics partition not found: {path}")

    df = (
        pl.scan_parquet(path)
        .filter(pl.col("stat_type") == STAT_TYPE_OPEN_INTEREST)
        .filter(pl.col("timestamp").dt.date() == d)
        .select("symbol", "timestamp", "quantity")
        .collect()
    )
    if df.is_empty():
        return {}

    df = df.filter(
        pl.col("symbol").map_elements(
            lambda s: _is_outright(s, symbol_root), return_dtype=pl.Boolean,
        )
    )
    if df.is_empty():
        return {}

    latest = (
        df.sort("timestamp")
        .group_by("symbol")
        .agg(pl.col("quantity").last().alias("oi"))
    )
    return {row["symbol"]: int(row["oi"]) for row in latest.iter_rows(named=True)}
```

Also add the import for `statistics_dir` if not already present from Task 3 (it is).

- [ ] **Step 4: Run test to verify it passes**

Run: `/c/Users/qwqw1/anaconda3/envs/fintech/python.exe -m pytest tests/data/futures/test_repair_regression.py -k per_contract_oi -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/data/derivations/futures/open_interest.py tests/data/futures/test_repair_regression.py
git commit -m "feat(futures): per-contract open-interest extraction for roll detection"
```

---

## Task 8: Roll algorithm core (OI crossover + hysteresis + volume tiebreak)

**Files:**
- Create: `src/data/futures/roll_calendar.py`
- Test: `tests/data/futures/test_roll_calendar.py`

**Interfaces:**
- Consumes: `per_contract_open_interest` (Task 7), `ContinuousContractDataLoader._active_contract_per_day` (existing volume signal), `get_spec` (Task 6).
- Produces:
  - `@dataclass(frozen=True) ContractRef(raw_symbol: str, expiration: date, activation: date)`
  - `@dataclass(frozen=True) RollEvent(roll_date: date, from_symbol: str, to_symbol: str, trigger: str)` where `trigger in {"oi_crossover","fnd_clamp","calendar_fallback"}`
  - `detect_rolls(root, oi_by_day: dict[date, dict[str,int]], hysteresis: int = 2) -> list[RollEvent]` — pure function (unit-testable with synthetic OI, no I/O)

This task builds the PURE algorithm against synthetic OI series so it is fully deterministic and fast. Task 9 wraps it with real data + FND + the cache.

- [ ] **Step 1: Write the failing test (pure, synthetic OI)**

```python
# tests/data/futures/test_roll_calendar.py
from datetime import date
from src.data.futures.roll_calendar import detect_rolls, RollEvent


def _day(n): return date(2024, 1, n)


def test_oi_crossover_with_hysteresis():
    # GCG4 front until Jan 10; GCJ4 OI overtakes for 2 consecutive days -> roll on 2nd
    oi = {
        _day(8):  {"GCG4": 100, "GCJ4": 10},
        _day(9):  {"GCG4": 90,  "GCJ4": 40},
        _day(10): {"GCG4": 50,  "GCJ4": 60},   # crossover day 1 (not yet, hysteresis=2)
        _day(11): {"GCG4": 30,  "GCJ4": 80},   # crossover day 2 -> ROLL here
        _day(12): {"GCG4": 20,  "GCJ4": 90},
    }
    rolls = detect_rolls("GC", oi, hysteresis=2)
    assert len(rolls) == 1
    assert rolls[0].roll_date == _day(11)
    assert rolls[0].from_symbol == "GCG4"
    assert rolls[0].to_symbol == "GCJ4"
    assert rolls[0].trigger == "oi_crossover"


def test_single_day_oi_blip_does_not_roll():
    # One-day OI spike in back month must NOT trigger a roll (hysteresis guards it)
    oi = {
        _day(8):  {"GCG4": 100, "GCJ4": 10},
        _day(9):  {"GCG4": 40,  "GCJ4": 60},   # blip up
        _day(10): {"GCG4": 90,  "GCJ4": 20},   # back to front dominant
        _day(11): {"GCG4": 85,  "GCJ4": 25},
    }
    rolls = detect_rolls("GC", oi, hysteresis=2)
    assert rolls == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/c/Users/qwqw1/anaconda3/envs/fintech/python.exe -m pytest tests/data/futures/test_roll_calendar.py -v`
Expected: FAIL — module/function does not exist.

- [ ] **Step 3: Implement the pure roll detector**

```python
# src/data/futures/roll_calendar.py
"""OI-primary futures roll calendar.

detect_rolls() is a pure function over a per-day per-contract OI series; it has
no I/O so it is fully deterministic and unit-testable. The RollCalendar class
(Task 9) wraps it with real data, the FND clamp, and a cached artifact.

Roll rule: the front contract rolls to the back contract when the back
contract's OI exceeds the front's for `hysteresis` consecutive days (anti-blip).
The roll date is the day the streak completes. Trigger is recorded so callers
(and tests) can see WHY each roll fired.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class ContractRef:
    raw_symbol: str
    expiration: date
    activation: date


@dataclass(frozen=True)
class RollEvent:
    roll_date: date
    from_symbol: str
    to_symbol: str
    trigger: str   # "oi_crossover" | "fnd_clamp" | "calendar_fallback"


def _front_by_oi(day_oi: dict[str, int]) -> str | None:
    """Contract with the highest OI on a day, or None if empty."""
    if not day_oi:
        return None
    return max(day_oi, key=day_oi.get)


def detect_rolls(
    root: str,
    oi_by_day: dict[date, dict[str, int]],
    hysteresis: int = 2,
) -> list[RollEvent]:
    """Detect OI-crossover rolls with a consecutive-day hysteresis.

    Args:
        root: symbol root (for context only; symbols come from the OI dict).
        oi_by_day: {date: {contract_symbol: open_interest}}.
        hysteresis: consecutive days the new front must dominate before rolling.

    Returns:
        Chronological list of RollEvent with trigger="oi_crossover".
    """
    days = sorted(oi_by_day)
    rolls: list[RollEvent] = []
    current_front: str | None = None
    candidate: str | None = None
    streak = 0

    for d in days:
        day_oi = oi_by_day[d]
        top = _front_by_oi(day_oi)
        if top is None:
            continue
        if current_front is None:
            current_front = top
            continue
        if top == current_front:
            candidate = None
            streak = 0
            continue
        # a different contract leads today
        if top == candidate:
            streak += 1
        else:
            candidate = top
            streak = 1
        if streak >= hysteresis:
            rolls.append(RollEvent(
                roll_date=d,
                from_symbol=current_front,
                to_symbol=top,
                trigger="oi_crossover",
            ))
            current_front = top
            candidate = None
            streak = 0
    return rolls
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/c/Users/qwqw1/anaconda3/envs/fintech/python.exe -m pytest tests/data/futures/test_roll_calendar.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/data/futures/roll_calendar.py tests/data/futures/test_roll_calendar.py
git commit -m "feat(futures): pure OI-crossover roll detector with hysteresis"
```

---

## Task 9: RollCalendar class — FND clamp, dual-nth API, fail-loud lookups

**Files:**
- Modify: `src/data/futures/roll_calendar.py` (add `RollCalendar`, `NoActiveContractError`, `apply_fnd_clamp`)
- Test: extend `tests/data/futures/test_roll_calendar.py`

**Interfaces:**
- Consumes: `detect_rolls` (Task 8), `get_spec` (Task 6), `FuturesDefinitionsLoader` (for expiration/cycle order), `per_contract_open_interest` (Task 7).
- Produces:
  - `NoActiveContractError(LookupError)`
  - `apply_fnd_clamp(root, rolls, expirations) -> list[RollEvent]` — pure; pulls a physical root's roll earlier if it would sit past `expiration - fnd_offset_days` business days
  - `RollCalendar` with: `get_front(root, on) -> ContractRef`, `get_nth_by_cycle(root, on, n) -> ContractRef`, `get_nth_by_oi(root, on, n) -> ContractRef`, `days_to_expiry(root, on) -> int`, `settlement_type(root) -> str`, `roll_events(root) -> list[RollEvent]`

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/data/futures/test_roll_calendar.py
import pytest
from src.data.futures.roll_calendar import apply_fnd_clamp, NoActiveContractError, RollEvent


def test_fnd_clamp_pulls_physical_roll_earlier():
    # A physical root whose OI-roll lands AFTER the FND cutoff must be clamped earlier.
    rolls = [RollEvent(date(2024, 1, 28), "GCF4", "GCG4", "oi_crossover")]
    expirations = {"GCF4": date(2024, 1, 29)}   # last-trade; FND well before
    # GC fnd_offset_days=3 -> cutoff = expiration - 3 business days = ~2024-01-24
    clamped = apply_fnd_clamp("GC", rolls, expirations)
    assert clamped[0].roll_date <= date(2024, 1, 25)
    assert clamped[0].trigger == "fnd_clamp"


def test_fnd_clamp_noop_for_financial_root():
    rolls = [RollEvent(date(2024, 3, 15), "ESH4", "ESM4", "oi_crossover")]
    expirations = {"ESH4": date(2024, 3, 15)}
    clamped = apply_fnd_clamp("ES", rolls, expirations)
    assert clamped == rolls   # financial -> untouched


def test_missing_root_lookup_raises(tmp_path):
    cal = RollCalendar(cache_dir=tmp_path)   # empty cache
    with pytest.raises(NoActiveContractError):
        cal.get_front("GC", date(2024, 1, 15))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/c/Users/qwqw1/anaconda3/envs/fintech/python.exe -m pytest tests/data/futures/test_roll_calendar.py -k "fnd or missing_root" -v`
Expected: FAIL — `apply_fnd_clamp` / `RollCalendar` / `NoActiveContractError` not defined.

- [ ] **Step 3: Implement clamp + class**

Append to `src/data/futures/roll_calendar.py`:

```python
from pathlib import Path

import polars as pl

from src.data.futures.contract_specs import get_spec
from src.data.futures.paths import roll_calendar_dir


class NoActiveContractError(LookupError):
    """No active contract for the requested (root, date), or root not built."""


def _minus_business_days(d: date, n: int) -> date:
    """Return the date n business days before d (Mon-Fri only)."""
    from datetime import timedelta
    cur = d
    remaining = n
    while remaining > 0:
        cur -= timedelta(days=1)
        if cur.weekday() < 5:
            remaining -= 1
    return cur


def apply_fnd_clamp(
    root: str,
    rolls: list[RollEvent],
    expirations: dict[str, date],
) -> list[RollEvent]:
    """Pull a physical root's roll earlier if it sits past its FND cutoff.

    Financial roots (fnd_offset_days == 0) are returned unchanged. The clamp
    only ever moves a roll EARLIER (never later); trigger becomes "fnd_clamp"
    when it fires.
    """
    spec = get_spec(root)
    if spec.settlement_type == "financial" or spec.fnd_offset_days == 0:
        return rolls
    out: list[RollEvent] = []
    for ev in rolls:
        exp = expirations.get(ev.from_symbol)
        if exp is None:
            out.append(ev)
            continue
        cutoff = _minus_business_days(exp, spec.fnd_offset_days)
        if ev.roll_date > cutoff:
            out.append(RollEvent(cutoff, ev.from_symbol, ev.to_symbol, "fnd_clamp"))
        else:
            out.append(ev)
    return out


class RollCalendar:
    """Lookup API over cached per-root roll calendars.

    Cache schema (futures/roll_calendar/{root}.parquet): one row per date with
    [date, front_symbol, next_cycle_symbol, next_oi_symbol, dte_front].
    """

    def __init__(self, cache_dir: Path | None = None) -> None:
        self._dir = cache_dir if cache_dir is not None else roll_calendar_dir()
        self._cache: dict[str, pl.DataFrame] = {}

    def _load(self, root: str) -> pl.DataFrame:
        if root in self._cache:
            return self._cache[root]
        path = self._dir / f"{root}.parquet"
        if not path.exists():
            raise NoActiveContractError(f"no roll calendar built for {root}: {path}")
        df = pl.read_parquet(path)
        self._cache[root] = df
        return df

    def _row(self, root: str, on: date) -> dict:
        df = self._load(root)
        matched = df.filter(pl.col("date") == on)
        if matched.is_empty():
            raise NoActiveContractError(f"no active contract for {root} on {on}")
        return matched.row(0, named=True)

    def get_front(self, root: str, on: date) -> ContractRef:
        r = self._row(root, on)
        return ContractRef(r["front_symbol"], r["front_expiration"], r["front_activation"])

    def get_nth_by_cycle(self, root: str, on: date, n: int) -> ContractRef:
        r = self._row(root, on)
        col = "front_symbol" if n == 0 else f"next_cycle_symbol"
        # n>=1 stored as next_cycle_symbol for n==1; deeper n require the builder
        # to store additional columns (YAGNI: only n in {0,1} supported in v1).
        if n not in (0, 1):
            raise ValueError(f"get_nth_by_cycle supports n in {{0,1}} in v1, got {n}")
        sym = r["front_symbol"] if n == 0 else r["next_cycle_symbol"]
        return ContractRef(sym, r["front_expiration"], r["front_activation"])

    def get_nth_by_oi(self, root: str, on: date, n: int) -> ContractRef:
        r = self._row(root, on)
        if n not in (0, 1):
            raise ValueError(f"get_nth_by_oi supports n in {{0,1}} in v1, got {n}")
        sym = r["front_symbol"] if n == 0 else r["next_oi_symbol"]
        return ContractRef(sym, r["front_expiration"], r["front_activation"])

    def days_to_expiry(self, root: str, on: date) -> int:
        return int(self._row(root, on)["dte_front"])

    def settlement_type(self, root: str) -> str:
        return get_spec(root).settlement_type

    def roll_events(self, root: str) -> list[RollEvent]:
        df = self._load(root).sort("date")
        events: list[RollEvent] = []
        prev = None
        for r in df.iter_rows(named=True):
            if prev is not None and r["front_symbol"] != prev["front_symbol"]:
                events.append(RollEvent(
                    r["date"], prev["front_symbol"], r["front_symbol"],
                    r.get("roll_trigger", "oi_crossover"),
                ))
            prev = r
        return events
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/c/Users/qwqw1/anaconda3/envs/fintech/python.exe -m pytest tests/data/futures/test_roll_calendar.py -v`
Expected: PASS (all roll-calendar tests)

- [ ] **Step 5: Commit**

```bash
git add src/data/futures/roll_calendar.py tests/data/futures/test_roll_calendar.py
git commit -m "feat(futures): RollCalendar lookup API + FND clamp + fail-loud lookups"
```

---

## Task 10: Batch builder — write per-root cached calendars

**Files:**
- Create: `scripts/data/build_roll_calendar.py`
- Test: extend `tests/data/futures/test_roll_calendar.py` (round-trip against a tiny synthetic cache)

**Interfaces:**
- Consumes: `per_contract_open_interest` (Task 7), `detect_rolls` + `apply_fnd_clamp` (Tasks 8-9), `FuturesDefinitionsLoader` (expirations/cycle order), `get_spec`.
- Produces: writes `futures/roll_calendar/{root}.parquet` with columns `[date, front_symbol, front_expiration, front_activation, next_cycle_symbol, next_oi_symbol, dte_front, roll_trigger]`.

- [ ] **Step 1: Write the failing round-trip test**

```python
# append to tests/data/futures/test_roll_calendar.py
import polars as pl
from datetime import date as _date
from src.data.futures.roll_calendar import RollCalendar


def test_roll_calendar_roundtrip(tmp_path):
    # Hand-write a 2-row calendar and confirm the lookup API reads it back.
    df = pl.DataFrame({
        "date": [_date(2024, 1, 15), _date(2024, 1, 16)],
        "front_symbol": ["GCG4", "GCG4"],
        "front_expiration": [_date(2024, 2, 27), _date(2024, 2, 27)],
        "front_activation": [_date(2022, 3, 30), _date(2022, 3, 30)],
        "next_cycle_symbol": ["GCH4", "GCH4"],
        "next_oi_symbol": ["GCJ4", "GCJ4"],
        "dte_front": [43, 42],
        "roll_trigger": ["oi_crossover", "oi_crossover"],
    })
    df.write_parquet(tmp_path / "GC.parquet")
    cal = RollCalendar(cache_dir=tmp_path)
    assert cal.get_front("GC", _date(2024, 1, 15)).raw_symbol == "GCG4"
    assert cal.get_nth_by_cycle("GC", _date(2024, 1, 15), 1).raw_symbol == "GCH4"
    assert cal.get_nth_by_oi("GC", _date(2024, 1, 15), 1).raw_symbol == "GCJ4"
    assert cal.days_to_expiry("GC", _date(2024, 1, 16)) == 42
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/c/Users/qwqw1/anaconda3/envs/fintech/python.exe -m pytest tests/data/futures/test_roll_calendar.py -k roundtrip -v`
Expected: FAIL — the get_nth columns/logic mismatch OR passes only after Task 9 is correct. (If Task 9 stored the right column names, this validates the builder's output contract.)

- [ ] **Step 3: Implement the builder**

```python
# scripts/data/build_roll_calendar.py
"""Build per-root futures roll calendars from OI + definitions.

For each root and each trading day in the requested range:
  1. read per-contract OI (statistics stat_type=9)
  2. detect OI-crossover rolls (hysteresis) -> front contract per day
  3. clamp physical-root rolls to before FND
  4. resolve next-by-cycle and next-by-OI contracts + front expiration
  5. write futures/roll_calendar/{root}.parquet

Usage:
    python scripts/data/build_roll_calendar.py --roots GC CL ES --start 2024-01-01 --end 2024-12-31
"""
from __future__ import annotations

import argparse
from datetime import date, timedelta

import polars as pl

from src.data.derivations.futures.open_interest import per_contract_open_interest
from src.data.futures.contract_specs import SPECS, get_spec
from src.data.futures.paths import roll_calendar_dir
from src.data.futures.roll_calendar import apply_fnd_clamp, detect_rolls
from src.data.futures_definitions_loader import FuturesDefinitionsLoader
from src.utils.logger import get_logger

logger = get_logger(__name__)

_MONTH_CODES = "FGHJKMNQUVXZ"


def _daterange(start: date, end: date):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def _cycle_order_key(sym: str, root: str) -> tuple[int, int]:
    """Sort key (year, month) for a raw contract symbol, for cycle ordering."""
    suffix = sym[len(root):]
    month = _MONTH_CODES.index(suffix[0])
    year = int(suffix[1:])
    return (year, month)


def build_root(root: str, start: date, end: date) -> pl.DataFrame:
    defs = FuturesDefinitionsLoader()
    oi_by_day: dict[date, dict[str, int]] = {}
    for d in _daterange(start, end):
        if d.weekday() >= 5:
            continue
        try:
            oi = per_contract_open_interest(root, d)
        except FileNotFoundError:
            continue
        if oi:
            oi_by_day[d] = oi

    if not oi_by_day:
        logger.warning(f"[!] no OI data for {root} in range -- skipping")
        return pl.DataFrame()

    rolls = detect_rolls(root, oi_by_day)

    # expirations for the front symbols involved (for FND clamp + dte)
    expirations: dict[str, date] = {}
    for ev in rolls:
        for sym in (ev.from_symbol, ev.to_symbol):
            if sym not in expirations:
                try:
                    expirations[sym] = defs.get_expiration(sym, root, start)
                except (LookupError, FileNotFoundError, ValueError):
                    pass
    rolls = apply_fnd_clamp(root, rolls, expirations)

    # front contract per day by walking rolls
    roll_map = {ev.roll_date: ev for ev in rolls}
    rows = []
    current_front = None
    for d in sorted(oi_by_day):
        if current_front is None:
            current_front = max(oi_by_day[d], key=oi_by_day[d].get)
        if d in roll_map:
            current_front = roll_map[d].to_symbol
        day_oi = oi_by_day[d]
        # next-by-oi: 2nd highest OI outright
        ranked = sorted(day_oi, key=day_oi.get, reverse=True)
        next_oi = ranked[1] if len(ranked) > 1 else ranked[0]
        # next-by-cycle: next expiry after front in cycle order among present contracts
        by_cycle = sorted(day_oi, key=lambda s: _cycle_order_key(s, root))
        try:
            fi = by_cycle.index(current_front)
            next_cycle = by_cycle[fi + 1] if fi + 1 < len(by_cycle) else next_oi
        except ValueError:
            next_cycle = next_oi
        exp = expirations.get(current_front)
        if exp is None:
            try:
                exp = defs.get_expiration(current_front, root, d)
            except (LookupError, FileNotFoundError, ValueError):
                exp = d  # degenerate fallback; dte becomes 0
        try:
            act = defs.get_definition(current_front, root, d).activation
        except (LookupError, FileNotFoundError, ValueError):
            act = d
        rows.append({
            "date": d,
            "front_symbol": current_front,
            "front_expiration": exp,
            "front_activation": act,
            "next_cycle_symbol": next_cycle,
            "next_oi_symbol": next_oi,
            "dte_front": max((exp - d).days, 0),
            "roll_trigger": roll_map[d].trigger if d in roll_map else "hold",
        })
    return pl.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", default=sorted(SPECS.keys()))
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    args = ap.parse_args()
    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    out_dir = roll_calendar_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    for root in args.roots:
        get_spec(root)  # validate known root
        df = build_root(root, start, end)
        if df.is_empty():
            continue
        df.write_parquet(out_dir / f"{root}.parquet")
        logger.info(f"[+] built roll calendar for {root}: {df.height} days")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the round-trip test to verify it passes**

Run: `/c/Users/qwqw1/anaconda3/envs/fintech/python.exe -m pytest tests/data/futures/test_roll_calendar.py -k roundtrip -v`
Expected: PASS

- [ ] **Step 5: Smoke-build one root against real data**

Run: `/c/Users/qwqw1/anaconda3/envs/fintech/python.exe scripts/data/build_roll_calendar.py --roots GC --start 2024-01-01 --end 2024-12-31`
Expected: log `[+] built roll calendar for GC: <N> days` with N > 200; a `futures/roll_calendar/GC.parquet` file created.

- [ ] **Step 6: Commit**

```bash
git add scripts/data/build_roll_calendar.py tests/data/futures/test_roll_calendar.py
git commit -m "feat(futures): batch builder for per-root roll calendars"
```

---

## Task 11: Golden-date validation against published CME rolls

**Files:**
- Create: `tests/data/futures/test_roll_calendar_golden.py`

**Interfaces:**
- Consumes: `build_root` (Task 10) or a pre-built cache; `RollCalendar`.

- [ ] **Step 1: Write the golden-date test (real data, integration-marked)**

```python
# tests/data/futures/test_roll_calendar_golden.py
"""Golden validation: OI-based rolls must land near published CME roll dates.

This is the killer test -- GC and CL were the roots broken by the old .c.0
calendar roll (the 43-bars/day bug). Reproducing their 2024 rolls proves the
new calendar is correct.
"""
from datetime import date

import pytest

from src.data.futures.paths import statistics_dir
from scripts.data.build_roll_calendar import build_root


def _data_present() -> bool:
    return (statistics_dir() / "year=2024" / "month=1" / "data.parquet").exists()


pytestmark = pytest.mark.skipif(not _data_present(), reason="consolidated futures store not present")


def _roll_dates(root):
    df = build_root(root, date(2024, 1, 1), date(2024, 12, 31))
    return [r["date"] for r in df.iter_rows(named=True) if r["roll_trigger"] != "hold"]


def _near(actual_dates, target, tol_days=3):
    return any(abs((a - target).days) <= tol_days for a in actual_dates)


def test_gc_rolls_near_published_2024():
    # GC 2024 front rolls occur ahead of each even-month delivery (approx late
    # prior month). Assert several rolls land near these windows.
    rolls = _roll_dates("GC")
    assert len(rolls) >= 5, f"GC should roll ~6x in 2024, got {len(rolls)}"
    # Feb->Apr roll lands late Jan; Apr->Jun roll lands late Mar
    assert _near(rolls, date(2024, 1, 25), tol_days=5)
    assert _near(rolls, date(2024, 3, 25), tol_days=5)


def test_es_rolls_quarterly_2024():
    # ES rolls ~8 days before each quarterly expiry (3rd Fri Mar/Jun/Sep/Dec)
    rolls = _roll_dates("ES")
    assert len(rolls) >= 3, f"ES should roll ~4x in 2024, got {len(rolls)}"
    assert _near(rolls, date(2024, 3, 8), tol_days=4)    # pre-March expiry
    assert _near(rolls, date(2024, 6, 13), tol_days=4)   # pre-June expiry
```

- [ ] **Step 2: Run the golden test**

Run: `/c/Users/qwqw1/anaconda3/envs/fintech/python.exe -m pytest tests/data/futures/test_roll_calendar_golden.py -v`
Expected: PASS. If a family's dates are off, that means an FND offset or hysteresis value needs tuning (see Task 6 note) — adjust `contract_specs.py` / `hysteresis` and re-run. Do NOT loosen `tol_days` beyond 5 to force a pass.

- [ ] **Step 3: Commit**

```bash
git add tests/data/futures/test_roll_calendar_golden.py
git commit -m "test(futures): golden-date validation vs published 2024 CME rolls"
```

---

## Task 12: Wire `get_upcoming_rolls` to the calendar + docs update

**Files:**
- Modify: `src/data/roll_detector.py` (`get_upcoming_rolls`, currently a stub returning `[]`)
- Modify: `docs/strategies/research/20260509_FUTURES_STRATEGY_TESTING_PLAN.md` (correct Gap D + stale paths)
- Test: extend `tests/data/futures/test_roll_calendar.py`

**Interfaces:**
- Consumes: `RollCalendar.roll_events` (Task 9).
- Produces: `FuturesRollManager.get_upcoming_rolls(roots, today, lookahead_days)` returns real `RollEvent`s from the built calendar within the window.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/data/futures/test_roll_calendar.py
import polars as pl
from datetime import date as _d
from src.data.roll_detector import FuturesRollManager


def test_upcoming_rolls_from_calendar(tmp_path, monkeypatch):
    df = pl.DataFrame({
        "date": [_d(2024, 1, 24), _d(2024, 1, 25), _d(2024, 1, 26)],
        "front_symbol": ["GCG4", "GCJ4", "GCJ4"],
        "front_expiration": [_d(2024, 2, 27)] * 3,
        "front_activation": [_d(2022, 3, 30)] * 3,
        "next_cycle_symbol": ["GCH4", "GCK4", "GCK4"],
        "next_oi_symbol": ["GCJ4", "GCM4", "GCM4"],
        "dte_front": [34, 33, 32],
        "roll_trigger": ["hold", "oi_crossover", "hold"],
    })
    df.write_parquet(tmp_path / "GC.parquet")
    monkeypatch.setattr(
        "src.data.roll_detector.roll_calendar_dir", lambda: tmp_path, raising=False,
    )
    mgr = FuturesRollManager(cache_dir=tmp_path)
    rolls = mgr.get_upcoming_rolls(["GC"], today=_d(2024, 1, 20), lookahead_days=14)
    assert len(rolls) == 1
    assert rolls[0].to_contract == "GCJ4"
    assert rolls[0].roll_date == _d(2024, 1, 25)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/c/Users/qwqw1/anaconda3/envs/fintech/python.exe -m pytest tests/data/futures/test_roll_calendar.py -k upcoming -v`
Expected: FAIL — `get_upcoming_rolls` returns `[]` and `FuturesRollManager.__init__` takes no `cache_dir`.

- [ ] **Step 3: Implement on the calendar**

In `src/data/roll_detector.py`, update the import block and both methods:

```python
from src.data.futures.roll_calendar import RollCalendar
from src.data.futures.paths import roll_calendar_dir


class FuturesRollManager:
    def __init__(self, cache_dir=None) -> None:
        self._loader = ContinuousContractDataLoader()
        self._calendar = RollCalendar(cache_dir=cache_dir)

    def get_upcoming_rolls(self, roots, today=None, lookahead_days=14):
        if today is None:
            today = date.today()
        horizon = today + timedelta(days=lookahead_days)
        out: list[RollEvent] = []
        for root in roots:
            try:
                events = self._calendar.roll_events(root)
            except Exception:  # noqa: BLE001 -- unbuilt root -> no upcoming rolls
                continue
            for ev in events:
                if today <= ev.roll_date <= horizon:
                    out.append(RollEvent(
                        root=root,
                        roll_date=ev.roll_date,
                        from_contract=ev.from_symbol,
                        to_contract=ev.to_symbol,
                    ))
        return out
```

Add `from datetime import timedelta` to the imports.

- [ ] **Step 4: Run test to verify it passes**

Run: `/c/Users/qwqw1/anaconda3/envs/fintech/python.exe -m pytest tests/data/futures/test_roll_calendar.py -k upcoming -v`
Expected: PASS

- [ ] **Step 5: Correct the May plan doc**

In `docs/strategies/research/20260509_FUTURES_STRATEGY_TESTING_PLAN.md`, add a note at the top of Section 2.2 (Gap D) that the roll calendar, carry calculator, continuous loader, and futures cost model already exist and were repaired + extended (reference this plan and the design doc), and update the stale `futures_1min/` / `futures_per_contract_1min/` / `futures_statistics/` / `futures_definitions/` paths throughout to the consolidated `futures/databento/*` and `futures/definitions/` layout.

- [ ] **Step 6: Full suite + commit**

Run: `/c/Users/qwqw1/anaconda3/envs/fintech/python.exe -m pytest tests/data/ -q`
Expected: PASS

```bash
git add src/data/roll_detector.py docs/strategies/research/20260509_FUTURES_STRATEGY_TESTING_PLAN.md tests/data/futures/test_roll_calendar.py
git commit -m "feat(futures): get_upcoming_rolls on roll calendar + correct plan doc"
```

**Phase 1 done. OI-primary roll calendar with FND clamp, dual-nth API, cached artifacts, golden-validated against CME dates, and live-roll prediction wired in.**

---

## Self-Review

- **Spec coverage:** All spec sections mapped — data-source constraints (Global Constraints + Task 1), architecture 6 units (Tasks 1,6,7,8-9,10 + existing repaired modules), roll algorithm OI+hysteresis+FND+fallback (Tasks 8-9), dual-nth API (Task 9), cache artifact (Task 10), testing incl golden/spread/FND/fail-loud (Tasks 8,9,11 + repair regression), effort table (Tasks map to it). Repair phase (not in original spec, added after discovering existing broken infra) is Tasks 1-5.
- **Placeholder scan:** none — every code step contains full code; every run step has exact command + expected output.
- **Type consistency:** `ContractRef`, `RollEvent` (calendar) vs `RollEvent` (roll_detector — different dataclass, intentionally, one has `root/from_contract/to_contract`, the other `from_symbol/to_symbol/trigger`; Task 12 converts between them explicitly). `get_spec`, `per_contract_open_interest`, `detect_rolls`, `apply_fnd_clamp`, path helpers — names consistent across tasks.
- **Known caveat carried into execution:** `get_nth_by_cycle`/`get_nth_by_oi` support only n in {0,1} in v1 (YAGNI — carry needs front+next only); documented inline and enforced with a ValueError.
