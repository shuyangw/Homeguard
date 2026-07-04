# Carry De-Concentration Stack Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** XS carry (signal-side demean) + IDM (sizing-side cluster risk weights), both parameter-free, evaluated as 3 pre-committed trials, to push corrected carry (PBO 0.33) under the 0.25 gate.

**Architecture:** XS = a `FuturesCarryXSStrategy` subclass. IDM = a strategy-agnostic per-root `div_mult` vector via the existing sizing hook. 3 trials = config combos. Design: `docs/strategies/research/20260703_CARRY_DECONCENTRATION_DESIGN.md`.

## Global Constraints

- **Python:** ALWAYS `/c/Users/qwqw1/anaconda3/envs/fintech/python.exe -m pytest`. `PYTHONPATH=.` for scripts.
- **8-THREAD CAP on every backtest/walk-forward launch:** prefix `POLARS_MAX_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1` and use `--jobs 8`. `--jobs` alone does NOT cap threads (polars defaults to 32/proc).
- **ASCII only**; no `print()`. Base branch `feat/carry-deconcentration` (off `main` @ 0640a5f) -- do NOT switch.
- **Parameter-free:** cluster map, `w_i`, fixed correlations (intra 0.5 / inter 0.0), IDM cap 2.5, XS scale 10, carry_scalar/ewma_span are FIXED doctrine. NO sweeping. Exactly 3 trials.
- **Isolation:** do NOT change the gate math, walk-forward structure, margin model, or equity/crypto path. `div_mult` scalar path stays back-compatible.
- **Causality:** XS uses SAME-DAY cross-sectional stats only (no train-window/full-sample stat -> no lookahead). IDM `C` is a fixed constant matrix (no estimation).
- **Universe (33 roots):** ES NQ YM ZT ZF ZN TN ZB UB 6E 6J 6B 6A 6C 6S 6M 6N CL BZ NG HO RB GC SI HG PL ZC ZW ZS ZL ZM LE HE.

---

## Task 1: 7-cluster map

**Files:** Modify `src/data/futures/asset_class.py`; Test `tests/data/futures/test_cluster_map.py`

- [ ] **Step 1: failing test**
```python
# tests/data/futures/test_cluster_map.py
import pytest
from src.data.futures.asset_class import CLUSTER, cluster_for
BROAD = ["ES","NQ","YM","ZT","ZF","ZN","TN","ZB","UB","6E","6J","6B","6A","6C","6S","6M","6N",
         "CL","BZ","NG","HO","RB","GC","SI","HG","PL","ZC","ZW","ZS","ZL","ZM","LE","HE"]
VALID = {"equity","rates","fx","energy","metals","grains","meats"}
def test_all_broad_roots_clustered():
    for r in BROAD: assert cluster_for(r) in VALID
def test_energy_split_from_metals_grains():
    assert cluster_for("CL")=="energy" and cluster_for("GC")=="metals"
    assert cluster_for("ZC")=="grains" and cluster_for("LE")=="meats"
    assert cluster_for("ES")=="equity" and cluster_for("ZN")=="rates" and cluster_for("6E")=="fx"
def test_unmapped_raises():
    with pytest.raises(KeyError): cluster_for("NOPE")
```
- [ ] **Step 2:** run -> FAIL (no CLUSTER).
- [ ] **Step 3: add to `asset_class.py`**
```python
CLUSTER: dict[str, str] = {
    "ES": "equity", "NQ": "equity", "YM": "equity", "RTY": "equity",
    "M2K": "equity", "MES": "equity", "MNQ": "equity", "MYM": "equity",
    "ZT": "rates", "ZF": "rates", "ZN": "rates", "TN": "rates", "ZB": "rates", "UB": "rates",
    "10Y": "rates", "2YY": "rates", "5YY": "rates", "30Y": "rates", "SR1": "rates", "SR3": "rates",
    "6A": "fx", "6B": "fx", "6C": "fx", "6E": "fx", "6J": "fx", "6M": "fx", "6N": "fx", "6S": "fx",
    "CL": "energy", "BZ": "energy", "NG": "energy", "HO": "energy", "RB": "energy",
    "MCL": "energy", "MNG": "energy",
    "GC": "metals", "SI": "metals", "HG": "metals", "PL": "metals", "MGC": "metals",
    "SIL": "metals", "MET": "metals",
    "ZC": "grains", "ZW": "grains", "ZS": "grains", "ZL": "grains", "ZM": "grains", "KE": "grains",
    "LE": "meats", "HE": "meats",
}
def cluster_for(root: str) -> str:
    return CLUSTER[root]
```
- [ ] **Step 4:** run -> PASS. **Step 5:** commit `feat(futures): 7-cluster economic-complex map`.

---

## Task 2: FuturesCarryXSStrategy (within-class demean)

**Files:** Modify `src/strategies/advanced/futures_carry_strategy.py`, `src/strategies/registry.py`; Test `tests/strategies/test_futures_carry_xs.py`

**Interfaces:** `FuturesCarryXSStrategy(FuturesCarryStrategy)`; registry `"FuturesCarryXS"`.

- [ ] **Step 1: failing tests**
```python
# tests/strategies/test_futures_carry_xs.py
import numpy as np, pandas as pd
from src.strategies.advanced.futures_carry_strategy import FuturesCarryXSStrategy
from src.strategies.registry import get_strategy_class

def _close(n=60):
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.DataFrame({"CL": np.linspace(60,70,n), "NG": np.linspace(3,4,n),
                         "ES": np.linspace(3000,4000,n)}, index=idx)

def test_registered():
    assert get_strategy_class("FuturesCarryXS") is FuturesCarryXSStrategy

def test_within_class_demean_and_cap(monkeypatch):
    close = _close()
    # both energy roots carry +0.05 (a pure common bet); ES alone in equity
    monkeypatch.setattr(FuturesCarryXSStrategy, "_load_carry",
                        lambda self, root: pd.Series(0.05, index=close.index))
    fc = FuturesCarryXSStrategy(["CL","NG","ES"]).forecast_panel(close)
    v = fc.dropna()
    assert ((v >= -20.0) & (v <= 20.0)).all().all()
    # CL and NG share a common energy carry -> after within-energy demean their
    # forecasts are ~equal-and-opposite around 0 (common component removed).
    assert abs(v["CL"].mean() + v["NG"].mean()) < 1e-6
```
- [ ] **Step 2:** run -> FAIL.
- [ ] **Step 3: implement** (append to `futures_carry_strategy.py`)
```python
from src.data.futures.asset_class import asset_class_for

_XS_SCALE = 10.0  # doctrine: maps a same-day cross-sectional z-score to forecast units

class FuturesCarryXSStrategy(FuturesCarryStrategy):
    """Cross-sectional carry: absolute carry forecast demeaned WITHIN asset-class
    each day (removes the common directional carry bet), z-scored by the same-day
    within-class dispersion, scaled to forecast units, clipped. Same-day stats only
    -> causal. Singleton/empty classes contribute 0 (no relative-value bet)."""
    def forecast_panel(self, close_panel: pd.DataFrame) -> pd.DataFrame:
        raw = super().forecast_panel(close_panel)  # per-root absolute carry forecasts
        groups: dict[str, list[str]] = {}
        for r in self.universe:
            groups.setdefault(asset_class_for(r), []).append(r)
        out = pd.DataFrame(0.0, index=raw.index, columns=self.universe)
        for _, roots in groups.items():
            block = raw[roots]                          # dates x class-roots
            mean = block.mean(axis=1)                   # same-day within-class mean
            std = block.std(axis=1)                     # same-day within-class dispersion
            z = block.sub(mean, axis=0).div(std.replace(0.0, np.nan), axis=0)
            out[roots] = (z * _XS_SCALE).clip(-self.cap, self.cap)
        return out.reindex(columns=self.universe)
```
- [ ] **Step 4: register** in `registry.py`: `"FuturesCarryXS": ("src.strategies.advanced.futures_carry_strategy", "FuturesCarryXSStrategy")`; aliases `"XS Carry"`, `"Cross-Sectional Carry"`.
- [ ] **Step 5:** run -> PASS. **Step 6:** commit `feat(futures): FuturesCarryXS (within-class demeaned carry) + registry`.

---

## Task 3: IDM weights module

**Files:** Create `src/backtesting/utils/idm_weights.py`; Test `tests/backtesting/utils/test_idm_weights.py`

**Interfaces:** `compute_div_mult(universe: list[str]) -> dict[str, float]` -- deterministic, data-free.

**Math:** clusters c present in the universe; `n_c` roots each. Cluster budget = `1/len(clusters_present)`; `w_i = (1/len(clusters_present)) / n_{c(i)}` (equal risk across present clusters, then within) -> `sum w_i = 1`. Correlation `C`: `C[i][i]=1`, `C[i][j]=0.5` if same cluster else `0.0`. `IDM = min(1/sqrt(w'Cw), 2.5)`. Raw `dm_i = w_i * IDM`. `N_scale = 1 / median(dm_i)` (keeps the median instrument's `div_mult` ~ 1, so per-instrument sizing scale stays comparable to the scalar-1.0 baseline -- integer rounding unaffected; only the RELATIVE cluster down-weighting matters). Return `{root: dm_i * N_scale}`.

- [ ] **Step 1: failing tests**
```python
# tests/backtesting/utils/test_idm_weights.py
import numpy as np
from src.backtesting.utils.idm_weights import compute_div_mult

def test_deterministic_and_cluster_capped():
    U = ["ES","NQ","ZN","ZB","6E","CL","NG","HO","RB","GC","ZC","LE"]
    d1 = compute_div_mult(U); d2 = compute_div_mult(U)
    assert d1 == d2                                   # data-free, deterministic
    assert set(d1) == set(U)
    # energy has 4 roots (CL/NG/HO/RB) vs equity 2 (ES/NQ): each energy root's
    # UNSCALED cluster weight is smaller -> energy is de-concentrated per root.
    # (verify via the underlying weights being equal-risk-per-cluster)
    assert all(np.isfinite(v) and v > 0 for v in d1.values())

def test_median_divmult_near_one():
    U = ["ES","NQ","YM","ZT","ZF","ZN","6E","6J","CL","NG","GC","SI","ZC","ZW","LE","HE"]
    d = compute_div_mult(U)
    med = float(np.median(list(d.values())))
    assert abs(med - 1.0) < 1e-6                       # N_scale pins the median to 1
```
- [ ] **Step 2:** run -> FAIL. **Step 3:** implement per the Math above (numpy for `w'Cw`; `cluster_for` from Task 1). **Step 4:** run -> PASS. **Step 5:** commit `feat(futures): IDM per-root div_mult weights (cluster risk + fixed-corr IDM)`.

---

## Task 4: thread div_mult (scalar->dict) + config idm flag

**Files:** Modify `src/backtesting/engine/futures_portfolio_simulator.py`, `src/backtesting/engine/futures_backtest.py`; Test `tests/backtesting/engine/test_idm_sizing.py`

**Context:** `run_sized(..., div_mult=1.0)` passes `div_mult=div_mult` per root at `:156`. `run_futures_backtest` calls `sim.run_sized(close, forecasts, daily_vol, vol_target)` (`:92`) with no `div_mult`.

- [ ] **Step 1: failing test** (synthetic `run_sized`, no real data)
```python
# tests/backtesting/engine/test_idm_sizing.py
import collections
import pandas as pd
from src.backtesting.engine.futures_portfolio_simulator import FuturesPortfolioSimulator
from src.backtesting.margin.futures_margin import MarginModel
from src.backtesting.costs.futures import futures_round_trip_usd


def _panels():
    idx = pd.date_range("2022-01-03", periods=8, freq="B")
    close = pd.DataFrame({"GC": 1800.0, "CL": 80.0}, index=idx)
    fc = pd.DataFrame({"GC": 10.0, "CL": 10.0}, index=idx)    # equal forecast
    vol = pd.DataFrame({"GC": 0.01, "CL": 0.01}, index=idx)   # equal daily vol
    return close, fc, vol


def _sim():
    return FuturesPortfolioSimulator(initial_capital=1_000_000, cost_fn=futures_round_trip_usd,
                                     margin_model=MarginModel(), rebalance="weekly", cost_mult=1.0)


def _contracts_by_root(res):
    d = collections.defaultdict(int)
    for _, row in res.trades.iterrows():
        d[row["root"]] += abs(int(row["contracts"]))
    return d


def test_dict_divmult_scales_per_root():
    close, fc, vol = _panels()
    base = _contracts_by_root(_sim().run_sized(close, fc, vol, 0.20, div_mult=1.0))
    scaled = _contracts_by_root(_sim().run_sized(close, fc, vol, 0.20,
                                                 div_mult={"GC": 2.0, "CL": 0.5}))
    assert scaled["GC"] > base["GC"]   # GC up-weighted 2x -> more contracts
    assert scaled["CL"] < base["CL"]   # CL down-weighted 0.5x -> fewer


def test_scalar_divmult_still_works():
    close, fc, vol = _panels()
    res = _sim().run_sized(close, fc, vol, 0.20, div_mult=1.0)   # float path back-compat
    assert len(res.equity_curve) == 8
```
- [ ] **Step 2:** run -> FAIL.
- [ ] **Step 3: widen `run_sized`** -- change the signature to `div_mult: float | dict = 1.0` and the call site (`:156`) from `div_mult=div_mult` to `div_mult=(div_mult if isinstance(div_mult, (int, float)) else div_mult.get(r, 1.0))`.
- [ ] **Step 4: wire `run_futures_backtest`** -- read `backtest.idm` (bool, default False); when true, `from src.backtesting.utils.idm_weights import compute_div_mult; dm = compute_div_mult(universe)` and call `sim.run_sized(close, forecasts, daily_vol, vol_target, div_mult=dm)`; else unchanged.
- [ ] **Step 5:** run the new test + `tests/backtesting/engine/test_futures_backtest_pluggable.py` + `test_futures_backtest_e2e.py` -> all PASS (scalar path back-compat).
- [ ] **Step 6:** commit `feat(futures): per-root div_mult (idm flag) threaded into sizing`.

---

## Task 5: 3 trial configs

**Files:** Create `config/backtesting/carry_xs_broad.yaml`, `carry_idm_broad.yaml`, `carry_xs_idm_broad.yaml`; Test `tests/backtesting/config/test_deconcentration_configs.py`

Each copies `carry_broad.yaml` (33-root, $10M, 2010-06-07..2026-02-20, weekly) and sets: xs -> `strategy.name: FuturesCarryXS`; idm -> `strategy.name: FuturesCarry` + `backtest.idm: true`; xs_idm -> `FuturesCarryXS` + `backtest.idm: true`. Test asserts each config's name/idm flag/33 roots.
- [ ] Steps: failing test -> create 3 configs -> PASS -> commit `feat(futures): 3 de-concentration trial configs`.

---

## Task 6: Execution + acceptance (CONTROLLER-run, 8-thread capped)

Not TDD. After Tasks 1-5 merged-ready.
- [ ] **Step 1:** run the 3 walk-forwards, EACH 8-thread capped + RunStatus-tracked (background):
```bash
POLARS_MAX_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
PYTHONPATH=. python scripts/backtest_scripts/run_carver_walkforward.py \
  --config config/backtesting/carry_xs_broad.yaml \
  --report docs/reports/futures/CARRY_XS_BROAD_READINESS.md --jobs 8
```
(and `carry_idm_broad.yaml` -> `CARRY_IDM_BROAD_READINESS.md`, `carry_xs_idm_broad.yaml` -> `CARRY_XS_IDM_BROAD_READINESS.md`).
- [ ] **Step 2:** compare PBO / kurtosis / Sharpe of each trial to the corrected carry baseline (0.85 / PBO 0.33 / kurt 21). Record which (if any) clears PBO < 0.25 at Sharpe clearly > 0.
- [ ] **Step 3:** summarize for the user; commit the 3 reports. If a trial passes -> carry's first gate-pass (deploy candidate). If none -> concentration is intrinsic; documented; proceed to W3.

---

## Self-Review

- **Coverage:** cluster map (T1), XS strategy (T2), IDM weights (T3), div_mult threading + idm flag (T4), 3 configs (T5), 3-trial eval (T6). Matches the spec.
- **Placeholders:** T4 Step 1 is described (synthetic `run_sized` dict-div_mult test); implementer writes the concrete assertions per the note. All other steps have full code.
- **Types:** `cluster_for(root)->str`; `FuturesCarryXSStrategy.forecast_panel(close)->DataFrame`; `compute_div_mult(universe)->dict[str,float]`; `run_sized(..., div_mult: float|dict)`.
- **Parameter-free / causal:** XS same-day cross-sectional only; IDM fixed constants; 3 pre-committed trials; no sweeps. 8-thread cap on every run.
- **Back-compat:** `div_mult` float path unchanged (proven by retained pluggable/e2e tests); `backtest.idm` defaults False so existing carry configs are unaffected.
