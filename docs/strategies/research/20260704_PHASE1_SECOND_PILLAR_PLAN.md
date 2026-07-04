# Phase 1 (Second-Pillar Hunt) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test whether VALUE and CRYPTO-CARRY each qualify as a diversifying second pillar for the IDM-carry book (0.76 Sharpe), by building crypto CME-calendar carry, running standalone walk-forwards, and measuring carry-return correlation against the two-tier inclusion bar.

**Architecture:** Crypto carry reuses the existing `CarryCalculator` / `build_carry_cache` / `FuturesCarryStrategy` pipeline -- add a `crypto` asset_class branch (roll-yield) + map entries for BTC/ETH, build their carry cache, and run the SAME FuturesCarry strategy on a crypto universe. Value is already built (`FuturesValueStrategy`, commit 4854afa). Standalone walk-forwards produce OOS Sharpe/PBO; a small correlation tool runs single-shot full-period backtests of each candidate vs carry and computes daily-return Pearson correlation. No combiner (Phase 0) or combination (Phase 4) here -- those are later plans gated on a pillar qualifying.

**Tech Stack:** Python (fintech conda env), polars/pandas, existing futures walk-forward harness.

## Global Constraints
- Python: run tests ONLY with `/c/Users/qwqw1/anaconda3/envs/fintech/python.exe -m pytest <files>`. `PYTHONPATH=.` for scripts.
- ASCII only; no `print()`; no Unicode. Branch `feat/futures-sharpe-uplift` -- do NOT switch.
- Parameter-free: crypto carry uses the commodity roll-yield formula unchanged; no swept constants.
- Every walk-forward / cache build: 8-thread cap (`POLARS_MAX_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 ... --jobs 8`), ONE run per background job (under ~60min cap), RunStatus-tracked, trade-logged, registered in `output/experiments.duckdb`.
- NOTHING merged/pushed without user approval. **Controller-run tasks (2, 5, 6, 7) launch ONLY after the user gives an explicit go.**
- Two-tier inclusion bar (spec Sec 4): standalone PBO < 0.35 AND Sharpe > 0.35, correlation-tiered vs carry returns -- `|rho|<0.3` full weight; `0.3<=|rho|<0.5` include only if SR>=0.45; `|rho|>=0.5` exclude.
- Value warmup (spec Sec 6): value-inclusive walk-forwards use `--train-months 61` (>= 5yr lookback) so the 1260-day signal is real, not truncated.
- Update the DSR trial-count ledger (spec Sec 9) after every run.

---

## Task 1: Crypto carry branch + cluster/asset_class map (BTC/ETH)

**Files:**
- Modify: `src/data/futures/asset_class.py` (add BTC/ETH to ASSET_CLASS + CLUSTER)
- Modify: `src/data/carry_calculator.py:113-142` (add `crypto` branch to `compute`)
- Test: `tests/data/futures/test_crypto_carry.py`

**Interfaces:**
- Produces: `asset_class_for("BTC") == "crypto"`, `cluster_for("BTC") == "crypto"` (same for ETH); `CarryCalculator().compute("BTC", "crypto", d)` returns annualized roll yield.

- [ ] **Step 1: Write the failing test**
```python
# tests/data/futures/test_crypto_carry.py
from datetime import date
import pytest
from src.data.futures.asset_class import asset_class_for, cluster_for

def test_crypto_maps():
    for r in ("BTC", "ETH"):
        assert asset_class_for(r) == "crypto"
        assert cluster_for(r) == "crypto"

def test_crypto_carry_is_annualized_roll_yield(monkeypatch):
    from src.data import carry_calculator as cc
    calc = cc.CarryCalculator()
    # front=BTCF4 (Jan), second=BTCG4 (Feb): 1 month apart, days_to_second=30
    monkeypatch.setattr(calc, "_find_front_second_close",
                        lambda root, d: ("BTCF4", 100.0, "BTCG4", 102.0))
    val = calc.compute("BTC", "crypto", date(2024, 1, 15))
    # commodity-style roll yield: (second-front)/front * 365/days = 0.02 * 365/30
    assert val == pytest.approx((102.0 - 100.0) / 100.0 * (365.0 / 30.0))

def test_unknown_still_raises():
    from src.data import carry_calculator as cc
    calc = cc.CarryCalculator()
    calc2 = cc.CarryCalculator()
    import pytest as _p
    with _p.raises(ValueError):
        # asset_class 'bogus' still unknown
        object.__setattr__(calc2, "_find_front_second_close",
                           lambda root, d: ("XF4", 1.0, "XG4", 1.0))
        calc2.compute("X", "bogus", date(2024, 1, 15))
```

- [ ] **Step 2: Run to verify it fails**

Run: `/c/Users/qwqw1/anaconda3/envs/fintech/python.exe -m pytest tests/data/futures/test_crypto_carry.py -v`
Expected: FAIL (KeyError on BTC in ASSET_CLASS / unknown asset_class 'crypto').

- [ ] **Step 3: Add BTC/ETH to the maps** in `src/data/futures/asset_class.py`

In `ASSET_CLASS`, append after the commodity block:
```python
    # crypto (CME futures)
    "BTC": "crypto", "ETH": "crypto",
```
In `CLUSTER`, append after the meats line:
```python
    "BTC": "crypto", "ETH": "crypto",
```

- [ ] **Step 4: Add the `crypto` branch** to `CarryCalculator.compute` in `src/data/carry_calculator.py`, immediately after the `if asset_class == "commodity":` block (line ~122), before `equity_index`:
```python
        if asset_class == "crypto":
            # CME crypto futures: annualized calendar roll yield (same convention
            # as commodity). Short history + regime risk -- flagged in the report.
            return (second_c - front_c) / front_c * (365.0 / days_to_second)
```

- [ ] **Step 5: Run to verify pass**

Run: `/c/Users/qwqw1/anaconda3/envs/fintech/python.exe -m pytest tests/data/futures/test_crypto_carry.py tests/data/futures/test_cluster_map.py -v`
Expected: PASS (new crypto tests + existing cluster-map tests unaffected).

- [ ] **Step 6: Commit**
```bash
git add src/data/futures/asset_class.py src/data/carry_calculator.py tests/data/futures/test_crypto_carry.py
git commit -m "feat(futures): crypto CME calendar carry branch + BTC/ETH cluster/asset_class map"
```

---

## Task 2: Build crypto carry cache (CONTROLLER-RUN, gated on user go)

**Files:** writes `<storage>/futures/carry/{BTC,ETH}.parquet`. No code change.

- [ ] **Step 1 (gated on go):** build the crypto carry cache, 8-thread capped:
```bash
POLARS_MAX_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
PYTHONPATH=. /c/Users/qwqw1/anaconda3/envs/fintech/python.exe scripts/data/build_carry_cache.py \
  --roots BTC ETH --start 2017-01-01 --end 2026-02-20 --jobs 2
```
- [ ] **Step 2: verify** both parquets exist and carry is non-inert (nonzero, sane range):
```bash
PYTHONPATH=. /c/Users/qwqw1/anaconda3/envs/fintech/python.exe -c "
import polars as pl; from src.data.futures.paths import carry_dir
for r in ('BTC','ETH'):
    df = pl.read_parquet(carry_dir()/f'{r}.parquet')
    print(r, 'rows', df.height, 'carry mean', round(df['carry'].mean(),4), 'first', df['date'].min(), 'last', df['date'].max())
    assert df.height > 100 and abs(df['carry'].mean()) > 1e-9"
```
Expected: BTC ~2017+, ETH ~2021+, nonzero mean carry. If a root has < ~100 rows or all-zero carry, STOP and report (crypto per-contract data may be missing).

---

## Task 3: Standalone walk-forward configs (value + crypto)

**Files:**
- Create: `config/backtesting/crypto_carry_broad.yaml`
- Verify: `config/backtesting/value_broad.yaml` (already exists, commit 34c9b33)
- Test: `tests/backtesting/config/test_phase1_configs.py`

**Interfaces:** two configs runnable by `run_carver_walkforward.py`.

- [ ] **Step 1: Write the failing test**
```python
# tests/backtesting/config/test_phase1_configs.py
from pathlib import Path
import yaml

def _load(name):
    return yaml.safe_load((Path("config/backtesting") / name).read_text())

def test_value_config():
    c = _load("value_broad.yaml")
    assert c["strategy"]["name"] == "FuturesValue"
    assert len(c["strategy"]["universe"]) == 33

def test_crypto_carry_config():
    c = _load("crypto_carry_broad.yaml")
    assert c["strategy"]["name"] == "FuturesCarry"
    assert c["strategy"]["universe"] == ["BTC", "ETH"]
    assert str(c["dates"]["start"]) == "2017-01-01"
```

- [ ] **Step 2: Run to verify it fails**

Run: `/c/Users/qwqw1/anaconda3/envs/fintech/python.exe -m pytest tests/backtesting/config/test_phase1_configs.py -v`
Expected: FAIL (crypto_carry_broad.yaml missing).

- [ ] **Step 3: Create `config/backtesting/crypto_carry_broad.yaml`**
```yaml
# config/backtesting/crypto_carry_broad.yaml -- standalone crypto CME calendar carry (BTC/ETH)
asset_class: futures

strategy:
  name: FuturesCarry
  universe:
    - BTC
    - ETH

dates:
  start: "2017-01-01"
  end: "2026-02-20"

backtest:
  initial_capital: 10000000
  vol_target_per_instrument: 0.20
  rebalance: weekly
  cost_mult: 1.0
```

- [ ] **Step 4: Run to verify pass**

Run: `/c/Users/qwqw1/anaconda3/envs/fintech/python.exe -m pytest tests/backtesting/config/test_phase1_configs.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**
```bash
git add config/backtesting/crypto_carry_broad.yaml tests/backtesting/config/test_phase1_configs.py
git commit -m "feat(futures): phase-1 standalone configs (crypto carry BTC/ETH) + test"
```

---

## Task 4: Pillar correlation tool

**Files:**
- Create: `scripts/backtest_scripts/pillar_correlation.py`
- Test: `tests/backtest_scripts/test_pillar_correlation.py`

**Interfaces:**
- Produces: `daily_return_correlation(equity_a: list[float], equity_b: list[float], dates_a, dates_b) -> float` -- Pearson correlation of the two daily-return series on their common dates; NaN-safe.

- [ ] **Step 1: Write the failing test**
```python
# tests/backtest_scripts/test_pillar_correlation.py
from datetime import date
import numpy as np
from scripts.backtest_scripts.pillar_correlation import daily_return_correlation

def test_perfectly_correlated():
    dts = [date(2020,1,d) for d in range(1,8)]
    eq = [100,101,102,101,103,104,103]
    assert daily_return_correlation(eq, eq, dts, dts) > 0.999

def test_common_dates_only():
    a_d = [date(2020,1,d) for d in range(1,6)]
    b_d = [date(2020,1,d) for d in range(3,8)]  # overlap 3,4,5
    a_e = [100,110,100,110,100]
    b_e = [50,55,50,55,50]  # on overlap, returns move together
    r = daily_return_correlation(a_e, b_e, a_d, b_d)
    assert -1.0 <= r <= 1.0
```

- [ ] **Step 2: Run to verify it fails**

Run: `/c/Users/qwqw1/anaconda3/envs/fintech/python.exe -m pytest tests/backtest_scripts/test_pillar_correlation.py -v`
Expected: FAIL (module missing).

- [ ] **Step 3: Implement `scripts/backtest_scripts/pillar_correlation.py`**
```python
"""Pillar correlation: run single-shot full-period backtests of two configs and
correlate their daily returns on common dates. Supplementary to the standalone
walk-forward Sharpe (which is the OOS metric used for the inclusion bar)."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.backtesting.engine.futures_backtest import run_futures_backtest


def daily_return_correlation(equity_a, equity_b, dates_a, dates_b) -> float:
    sa = pd.Series(equity_a, index=pd.DatetimeIndex(dates_a)).pct_change()
    sb = pd.Series(equity_b, index=pd.DatetimeIndex(dates_b)).pct_change()
    joined = pd.concat([sa, sb], axis=1, join="inner").dropna()
    if len(joined) < 3:
        return float("nan")
    return float(joined.iloc[:, 0].corr(joined.iloc[:, 1]))


def _run(config_path: str):
    cfg = yaml.safe_load(Path(config_path).read_text())
    res = run_futures_backtest(cfg, register=False)
    return res["equity_curve"], res["dates"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--a", required=True, help="candidate config (value/crypto)")
    p.add_argument("--b", required=True, help="carry baseline config")
    args = p.parse_args()
    ea, da = _run(args.a)
    eb, db = _run(args.b)
    rho = daily_return_correlation(ea, eb, da, db)
    from src.utils.logger import get_logger
    get_logger(__name__).info(f"[pillar_correlation] rho({args.a} , {args.b}) = {rho:.4f}")


if __name__ == "__main__":
    main()
```
NOTE for implementer: confirm `run_futures_backtest` returns a dict containing `equity_curve` and a matching `dates` key (inspect `src/backtesting/engine/futures_backtest.py`); if the date key differs, adapt `_run` and say so in the report. Do NOT invent a key.

- [ ] **Step 4: Run to verify pass**

Run: `/c/Users/qwqw1/anaconda3/envs/fintech/python.exe -m pytest tests/backtest_scripts/test_pillar_correlation.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**
```bash
git add scripts/backtest_scripts/pillar_correlation.py tests/backtest_scripts/test_pillar_correlation.py
git commit -m "feat(futures): pillar correlation tool (daily-return corr vs carry)"
```

---

## Task 5: Run VALUE standalone walk-forward (CONTROLLER-RUN, gated on go)

- [ ] **Step 1 (gated on go):** run, 8-thread capped, `--train-months 61` (value 5yr warmup), own bg job:
```bash
POLARS_MAX_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
PYTHONPATH=. /c/Users/qwqw1/anaconda3/envs/fintech/python.exe scripts/backtest_scripts/run_carver_walkforward.py \
  --config config/backtesting/value_broad.yaml \
  --report docs/reports/futures/VALUE_BROAD_READINESS.md \
  --json output/deconcentration/value.json --train-months 61 --jobs 8
```
- [ ] **Step 2:** read the readiness report; record OOS Sharpe (1x/1.5x), PBO, PSR, DSR, skew, kurt into the trial ledger (spec Sec 9) + `docs/progress/20260704_OVERNIGHT_RESULTS.md`.

---

## Task 6: Run CRYPTO carry standalone walk-forward (CONTROLLER-RUN, gated on go)

- [ ] **Step 1 (gated on go, AFTER Task 2 cache built):** run, 8-thread capped, own bg job:
```bash
POLARS_MAX_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
PYTHONPATH=. /c/Users/qwqw1/anaconda3/envs/fintech/python.exe scripts/backtest_scripts/run_carver_walkforward.py \
  --config config/backtesting/crypto_carry_broad.yaml \
  --report docs/reports/futures/CRYPTO_CARRY_READINESS.md \
  --json output/deconcentration/crypto_carry.json --jobs 8
```
NOTE: with only 2 roots and short history the walk-forward may yield few windows; if it errors with "< 2 usable OOS windows", reduce `--train-months`/`--test-months` (e.g. 24/12) and record that the crypto result is low-confidence (thin sample).
- [ ] **Step 2:** record metrics into the trial ledger + results doc, with an explicit low-confidence caveat (2 roots, regime-heavy).

---

## Task 7: Correlation + inclusion verdict (CONTROLLER-RUN, gated on go)

- [ ] **Step 1 (gated on go):** compute each candidate's daily-return correlation vs the carry book:
```bash
PYTHONPATH=. /c/Users/qwqw1/anaconda3/envs/fintech/python.exe scripts/backtest_scripts/pillar_correlation.py \
  --a config/backtesting/value_broad.yaml --b config/backtesting/carry_idm_broad.yaml
PYTHONPATH=. /c/Users/qwqw1/anaconda3/envs/fintech/python.exe scripts/backtest_scripts/pillar_correlation.py \
  --a config/backtesting/crypto_carry_broad.yaml --b config/backtesting/carry_idm_broad.yaml
```
- [ ] **Step 2:** apply the two-tier inclusion bar (Global Constraints) to each candidate using its standalone WF Sharpe/PBO (Tasks 5/6) + this correlation. Record the INCLUDE / EXCLUDE verdict and reason per candidate in `docs/progress/20260704_OVERNIGHT_RESULTS.md` and the trial ledger. Report ALL outcomes (spec Sec 6 -- no survivorship).
- [ ] **Step 3:** summarize for the user: which (if any) pillar qualifies, and the recommended next plan (Phase 0 combiner + Phase 4 combination IF a pillar qualified; else the honest fallback -- carry + breadth/buffering only).

---

## Self-Review
- **Spec coverage:** crypto carry branch+maps (T1), crypto cache (T2), value+crypto configs (T3), correlation tool (T4), value WF with 61m warmup (T5), crypto WF (T6), two-tier inclusion verdict + report-all (T7). Value already built (spec Sec 5). Combiner/combination explicitly deferred (spec Phase 0/4). Covered.
- **Placeholders:** none -- crypto formula, map entries, configs, correlation code all concrete. T4 carries an explicit "confirm the dates key" instruction (a verification, not a placeholder).
- **Types:** `asset_class_for/cluster_for(str)->str`; `compute(root,asset_class,d)->float`; `daily_return_correlation(list,list,dates,dates)->float`; configs -> dict. Consistent across tasks.
- **Gating:** Tasks 2,5,6,7 are controller-run and launch only on explicit user go; build tasks (1,3,4) are TDD and safe to implement immediately. 8-thread cap + one-run-per-job + trade-log + ledger on every run.
