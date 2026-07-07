# FxCarrySeatbelt (#16 + #19) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `FxCarrySeatbelt`, an enhanced spot-FX carry strategy (research #16 momentum+swap filter + #19 unwind detector) that replaces the naive `FxCarry` which failed the walk-forward gate, and validate it against an S&P 500 Sharpe benchmark.

**Architecture:** A single `forecast_panel(close)` daily strategy over the existing 22-pair G10 cache. It emits a filtered long-carry book (long only when rate-differential > +2% AND price momentum agrees), zeroes those longs when a reusable carry-unwind composite score signals risk-off, and adds half-size shorts on AUDJPY/NZDJPY when the score signals a cascade. Validation reuses the existing FX walk-forward window machinery plus a new S&P benchmark comparison.

**Tech Stack:** Python 3.13 (conda env `fintech`), pandas, numpy, polars, the existing FX backtest engine (`src/backtesting/engine/fx_backtest.py`, `FxSpotPortfolioSimulator`), FRED rates (`src/data/fx_rates.py`), yfinance for the S&P index (keyless), pytest.

## Global Constraints

- Run all Python via the `fintech` conda env with `PYTHONPATH=$(pwd)`. Test command prefix: `source ~/anaconda3/etc/profile.d/conda.sh; conda activate fintech; PYTHONPATH=$(pwd) python -m pytest ...`.
- ASCII-only in all code and docs. No em dashes, no emojis, no Unicode arrows or symbols. Use `->`, `~`, `x`, `[+]/[-]/[!]`.
- No `print()`; use `from src.utils import logger`. Homeguard logger takes f-strings, not `%s`.
- Forecasts are on the Carver scale: 10.0 = full 1x vol-target position. Full long forecast = +10.0; half-size short = -5.0.
- Every signal must be CAUSAL: the value at date t may use only data with index <= t. No lookahead. This is the single most important correctness property; every test includes a causality check.
- Sizing config mirrors the other FX strategies verbatim: `vol_target_per_instrument: 0.03`, `leverage_cap: 4.0`, `idm: true`, `idm_cap: 2.5`, `initial_capital: 100000.0`, `save_trades: true`.
- Universe (22 pairs, exact order): `[EURUSD, USDJPY, USDCHF, EURJPY, EURCHF, CHFJPY, XAUUSD, XAGUSD, GBPUSD, USDCAD, AUDUSD, NZDUSD, AUDNZD, AUDJPY, NZDJPY, EURNOK, EURSEK, USDNOK, USDSEK, NOKSEK, NOKJPY, SEKJPY]`.
- Currency-strength sign convention (load-bearing): per-currency strength rises when that currency APPRECIATES. In an unwind, JPY and CHF appreciate, so the JPY and CHF strength-delta terms enter the score with a POSITIVE sign (no extra inversion in our convention). Do not negate them.
- Git hazard (macOS/Dropbox): never `git checkout <branch>`, never bare `git status`/`git diff`, never `git reset --hard`. Use only `git add <paths>`, `git commit`, `git log`. Work happens directly on `main` in this repo (standing authorization); commit incrementally.
- Success bar is RELATIVE: strategy OOS Sharpe must beat the S&P 500 Sharpe over the same OOS dates. DSR/PSR/PBO are computed and reported as diagnostics only, they do NOT gate the decision.

---

### Task 1: Pre-registration note (locked before any result exists)

Writing the success criterion to disk and committing it BEFORE any backtest runs is what makes it a pre-registration rather than a post-hoc rationalization. This task has no code dependencies and must be committed first.

**Files:**
- Create: `docs/reports/fx/20260706_carry_seatbelt_prereg.md`

- [ ] **Step 1: Write the pre-registration note**

Create `docs/reports/fx/20260706_carry_seatbelt_prereg.md` with exactly this content:

```markdown
# FxCarrySeatbelt Pre-Registration - 2026-07-06

Written and committed BEFORE any FxCarrySeatbelt backtest was run. Records the
success criterion so it cannot be moved after seeing results.

## Strategy
FxCarrySeatbelt (research #16 Carry-Momentum Double Filter + #19 Carry-Unwind
Detector). Spec: docs/superpowers/specs/2026-07-06-fx-carry-seatbelt-design.md.

## Success criterion (primary, relative)
Run the existing FX walk-forward (36m train / 12m test / 12m step, purge +
embargo, both 1.0x and 1.5x cost legs) on BOTH the daily and weekly rebalance
configs. The strategy PASSES if its stitched OOS Sharpe (1.0x cost) exceeds the
S&P 500 buy-and-hold annualized Sharpe computed over the exact same stitched OOS
dates (rf = 0, same convention), on at least one cadence.

## Diagnostics (reported, NOT gating)
PSR, DSR (using the cumulative project-wide trial count), PBO, trade count,
IS/OOS Sharpe ratio, OOS Sharpe under 1.5x cost, correlation to the S&P over the
OOS dates, information ratio vs the S&P. Plus per-episode P&L attribution for the
Aug 2024 yen-carry unwind and the Mar 2020 COVID unwind, reported as existence
proofs (N is too small to be statistics).

## No absolute kill threshold
There is no pre-committed DSR/Sharpe floor that abandons the carry idea. A form
that fails the S&P bar is a failed variant; whether to iterate (the one deferred
variant: #16 mod-a 12-month TSMOM momentum leg or mod-b graded sizing) or shelve
is decided after seeing the result and the diagnostics.

## Known limitations accepted going in
1. Swap = FRED policy-rate differential proxy (no broker swap tables); an
   optimism bias in the carry gate, reported not hidden.
2. Offensive short rests on ~4-6 unwind events; existence proof, not statistics.
```

- [ ] **Step 2: Commit**

```bash
git add docs/reports/fx/20260706_carry_seatbelt_prereg.md
git commit -m "docs(fx): pre-register FxCarrySeatbelt success criterion (beat S&P)"
```

---

### Task 2: Carry-unwind composite score module

**Files:**
- Create: `src/backtesting/signals/__init__.py`
- Create: `src/backtesting/signals/carry_unwind.py`
- Test: `tests/backtesting/signals/__init__.py`, `tests/backtesting/signals/test_carry_unwind.py`

**Interfaces:**
- Consumes: nothing (pure functions on a close panel).
- Produces:
  - `compute_unwind_score(close_panel: pd.DataFrame, z_window: int = 252) -> pd.Series` (index = close_panel.index, higher = more risk-off, causal, NaN-free).
  - `_trailing_zscore(s: pd.Series, window: int) -> pd.Series` (causal rolling z, NaN -> 0.0).
  - `currency_strength(close_panel: pd.DataFrame) -> pd.DataFrame` (per-currency cumulative strength; appreciation -> rises).

- [ ] **Step 1: Write the failing tests**

Create `tests/backtesting/signals/__init__.py` (empty file), then `tests/backtesting/signals/test_carry_unwind.py`:

```python
import numpy as np
import pandas as pd

from src.backtesting.signals.carry_unwind import (
    compute_unwind_score, _trailing_zscore, currency_strength)


def _calm_panel(n=400):
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    rng = np.random.default_rng(0)
    pairs = ["USDJPY", "EURJPY", "AUDJPY", "CHFJPY", "XAUUSD", "NZDJPY"]
    return pd.DataFrame(
        {p: 100.0 + np.cumsum(rng.normal(0, 0.05, n)) for p in pairs}, index=idx)


def test_trailing_zscore_is_causal_and_nan_free():
    s = pd.Series(np.arange(300, dtype=float))
    z = _trailing_zscore(s, 100)
    assert not z.isna().any()
    # truncating the future must not change past z-values (causality)
    z_trunc = _trailing_zscore(s.iloc[:200], 100)
    pd.testing.assert_series_equal(z.iloc[:200], z_trunc, check_names=False)


def test_currency_strength_rises_when_currency_appreciates():
    idx = pd.date_range("2020-01-01", periods=50, freq="D")
    # AUDJPY rising = AUD appreciating vs JPY -> AUD strength up, JPY down
    panel = pd.DataFrame({"AUDJPY": np.linspace(80, 90, 50)}, index=idx)
    strength = currency_strength(panel)
    assert strength["AUD"].iloc[-1] > strength["AUD"].iloc[0]
    assert strength["JPY"].iloc[-1] < strength["JPY"].iloc[0]


def test_score_is_high_on_a_risk_off_day():
    panel = _calm_panel(400)
    # Engineer a risk-off shock in the last 3 days: JPY and CHF appreciate
    # (their crosses fall), AUDJPY vol spikes, gold jumps.
    for p in ["USDJPY", "EURJPY", "AUDJPY", "CHFJPY", "NZDJPY"]:
        panel.iloc[-3:, panel.columns.get_loc(p)] *= 0.90  # crosses crash -> JPY/CHF up
    panel.iloc[-3:, panel.columns.get_loc("XAUUSD")] *= 1.08  # gold bid
    score = compute_unwind_score(panel)
    assert score.iloc[-1] > score.iloc[:-10].mean() + 2.0


def test_score_is_causal_and_nan_free():
    panel = _calm_panel(400)
    score = compute_unwind_score(panel)
    assert not score.isna().any()
    score_trunc = compute_unwind_score(panel.iloc[:250])
    pd.testing.assert_series_equal(
        score.iloc[:250], score_trunc, check_names=False)


def test_score_handles_missing_inputs():
    # No XAUUSD, no CHF crosses -> those terms degrade to 0, no crash.
    idx = pd.date_range("2020-01-01", periods=300, freq="D")
    panel = pd.DataFrame({"AUDJPY": 80.0 + np.arange(300) * 0.01}, index=idx)
    score = compute_unwind_score(panel)
    assert not score.isna().any()
    assert len(score) == 300
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `source ~/anaconda3/etc/profile.d/conda.sh; conda activate fintech; PYTHONPATH=$(pwd) python -m pytest tests/backtesting/signals/test_carry_unwind.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.backtesting.signals'`.

- [ ] **Step 3: Create the package and implementation**

Create `src/backtesting/signals/__init__.py` (empty file).

Create `src/backtesting/signals/carry_unwind.py`:

```python
"""Carry-unwind composite risk-off score (research #19).

A single daily score, higher = more cascade-like. Built from four causal,
trailing-z-scored terms: JPY strength change, CHF strength change (both funding
currencies that appreciate in an unwind), AUDJPY short-horizon realized vol, and
XAUUSD 3-day return (gold bid). Designed as a shared risk-off brain reusable by
#15/#16/#18/#42; kept dependency-free (pure functions on a close panel) so any
strategy can call it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _trailing_zscore(s: pd.Series, window: int) -> pd.Series:
    mean = s.rolling(window, min_periods=max(window // 2, 2)).mean()
    std = s.rolling(window, min_periods=max(window // 2, 2)).std()
    z = (s - mean) / std.replace(0.0, np.nan)
    return z.fillna(0.0)


def currency_strength(close_panel: pd.DataFrame) -> pd.DataFrame:
    rets = close_panel.pct_change(fill_method=None)
    contrib: dict[str, list[pd.Series]] = {}
    for pair in rets.columns:
        base, quote = pair[:3], pair[3:]
        contrib.setdefault(base, []).append(rets[pair])
        contrib.setdefault(quote, []).append(-rets[pair])
    strength = {ccy: pd.concat(series, axis=1).mean(axis=1).cumsum()
                for ccy, series in contrib.items()}
    return pd.DataFrame(strength)


def compute_unwind_score(close_panel: pd.DataFrame, z_window: int = 252) -> pd.Series:
    idx = close_panel.index
    strength = currency_strength(close_panel)

    def delta_strength(ccy: str) -> pd.Series:
        if ccy not in strength.columns:
            return pd.Series(0.0, index=idx)
        return strength[ccy].diff(3).fillna(0.0)

    # Our strength convention: appreciation -> strength rises. JPY/CHF appreciate
    # in an unwind, so their positive strength-delta enters with a POSITIVE sign.
    jpy_term = _trailing_zscore(delta_strength("JPY"), z_window)
    chf_term = _trailing_zscore(delta_strength("CHF"), z_window)

    if "AUDJPY" in close_panel.columns:
        audjpy_vol = (close_panel["AUDJPY"].pct_change(fill_method=None)
                      .rolling(5, min_periods=3).std().fillna(0.0))
    else:
        audjpy_vol = pd.Series(0.0, index=idx)
    vol_term = _trailing_zscore(audjpy_vol, z_window)

    if "XAUUSD" in close_panel.columns:
        gold_ret = close_panel["XAUUSD"].pct_change(3, fill_method=None).fillna(0.0)
    else:
        gold_ret = pd.Series(0.0, index=idx)
    gold_term = _trailing_zscore(gold_ret, z_window)

    score = jpy_term + chf_term + vol_term + gold_term
    return score.fillna(0.0)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `source ~/anaconda3/etc/profile.d/conda.sh; conda activate fintech; PYTHONPATH=$(pwd) python -m pytest tests/backtesting/signals/test_carry_unwind.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/backtesting/signals/__init__.py src/backtesting/signals/carry_unwind.py tests/backtesting/signals/
git commit -m "feat(fx): reusable carry-unwind composite score (#19)"
```

---

### Task 3: FxCarrySeatbelt strategy + registry

**Files:**
- Create: `src/strategies/advanced/fx_carry_seatbelt.py`
- Modify: `src/strategies/registry.py` (add one entry near line 68, after `FxXSectMom`)
- Test: `tests/strategies/test_fx_carry_seatbelt.py`

**Interfaces:**
- Consumes: `compute_unwind_score` from Task 2; `load_fx_rate_panel`, `build_rate_diff_panel`, `currencies_for_pairs` from `src.data.fx_rates`.
- Produces: `FxCarrySeatbelt(universe, **params)` with `forecast_panel(close_panel: pd.DataFrame) -> pd.DataFrame` returning a daily forecast (values in [-5.0, +10.0]); registered under name `"FxCarrySeatbelt"`.

- [ ] **Step 1: Write the failing tests**

Create `tests/strategies/test_fx_carry_seatbelt.py`:

```python
import numpy as np
import pandas as pd

import src.data.fx_rates as fx_rates
from src.strategies.registry import get_strategy_class


def _panel(pairs, n=400, drift=0.0005):
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    t = np.arange(n)
    return pd.DataFrame({p: 100.0 * (1.0 + drift) ** t for p in pairs}, index=idx)


def _patch_rates(monkeypatch, rate_map):
    def fake(currencies, index):
        return pd.DataFrame(
            {c: pd.Series(rate_map.get(c, 0.0), index=index) for c in currencies})
    monkeypatch.setattr(fx_rates, "load_fx_rate_panel", fake)


def test_long_only_when_carry_and_momentum_agree(monkeypatch):
    # AUD 5%, JPY 0% -> AUDJPY carry +5% > 2% gate; uptrend -> long.
    _patch_rates(monkeypatch, {"AUD": 0.05, "JPY": 0.0})
    strat = get_strategy_class("FxCarrySeatbelt")(["AUDJPY"])
    fc = strat.forecast_panel(_panel(["AUDJPY"], drift=0.0008))
    assert fc["AUDJPY"].iloc[-1] == 10.0


def test_flat_when_carry_fails(monkeypatch):
    # AUD 1% -> carry +1% < 2% gate -> flat despite uptrend.
    _patch_rates(monkeypatch, {"AUD": 0.01, "JPY": 0.0})
    strat = get_strategy_class("FxCarrySeatbelt")(["AUDJPY"])
    fc = strat.forecast_panel(_panel(["AUDJPY"], drift=0.0008))
    assert fc["AUDJPY"].iloc[-1] == 0.0


def test_flat_when_momentum_fails(monkeypatch):
    # Good carry but downtrend -> flat (never short for carry).
    _patch_rates(monkeypatch, {"AUD": 0.05, "JPY": 0.0})
    strat = get_strategy_class("FxCarrySeatbelt")(["AUDJPY"])
    fc = strat.forecast_panel(_panel(["AUDJPY"], drift=-0.0008))
    assert fc["AUDJPY"].iloc[-1] == 0.0


def test_veto_zeroes_longs_on_risk_off(monkeypatch):
    _patch_rates(monkeypatch, {"AUD": 0.05, "JPY": 0.0, "CHF": 0.0})
    pairs = ["AUDJPY", "USDJPY", "EURJPY", "CHFJPY", "XAUUSD", "NZDJPY"]
    panel = _panel(pairs, drift=0.0008)
    # risk-off shock at the end -> veto engages -> AUDJPY long zeroed
    for p in ["USDJPY", "EURJPY", "AUDJPY", "CHFJPY", "NZDJPY"]:
        panel.iloc[-3:, panel.columns.get_loc(p)] *= 0.90
    panel.iloc[-3:, panel.columns.get_loc("XAUUSD")] *= 1.08
    strat = get_strategy_class("FxCarrySeatbelt")(pairs)
    fc = strat.forecast_panel(panel)
    assert fc["AUDJPY"].iloc[-1] <= 0.0  # long flattened (and maybe shorted)


def test_forecast_is_causal_and_bounded(monkeypatch):
    _patch_rates(monkeypatch, {"AUD": 0.05, "JPY": 0.0, "CHF": 0.0})
    pairs = ["AUDJPY", "USDJPY", "EURJPY", "CHFJPY", "XAUUSD", "NZDJPY"]
    panel = _panel(pairs, drift=0.0006)
    strat = get_strategy_class("FxCarrySeatbelt")(pairs)
    fc = strat.forecast_panel(panel)
    assert fc.abs().max().max() <= 10.0
    assert not fc.isna().any().any()
    fc_trunc = strat.forecast_panel(panel.iloc[:250])
    pd.testing.assert_frame_equal(fc.iloc[:250], fc_trunc, check_names=False)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `source ~/anaconda3/etc/profile.d/conda.sh; conda activate fintech; PYTHONPATH=$(pwd) python -m pytest tests/strategies/test_fx_carry_seatbelt.py -v`
Expected: FAIL (strategy `FxCarrySeatbelt` not registered / import error).

- [ ] **Step 3: Write the strategy**

Create `src/strategies/advanced/fx_carry_seatbelt.py`:

```python
"""FxCarrySeatbelt: filtered long-carry book + carry-unwind veto/short (#16 + #19).

Replaces the naive FxCarry (held every pair through every crash, failed the
gate). Long a pair only when its rate-differential carry proxy exceeds +2%
annualized AND price momentum agrees (close > EMA(50) with positive 10-day EMA
slope); flat otherwise (never short for carry). A reusable carry-unwind composite
score zeroes all longs on risk-off days (defensive veto) and adds half-size
shorts on AUDJPY/NZDJPY during a detected cascade (offensive leg). All signals
are causal. Forecasts are Carver-scaled (10 = 1x vol-target).
"""
from __future__ import annotations

import pandas as pd

from src.backtesting.signals.carry_unwind import compute_unwind_score


class FxCarrySeatbelt:
    def __init__(self, universe, carry_gate: float = 0.02, ema_span: int = 50,
                 slope_lookback: int = 10, veto_threshold: float = 1.0,
                 veto_clear_days: int = 3, short_threshold: float = 2.5,
                 short_low_lookback: int = 20, full_forecast: float = 10.0,
                 short_forecast: float = 5.0, z_window: int = 252, **params):
        self.universe = list(universe)
        self.carry_gate = float(carry_gate)
        self.ema_span = int(ema_span)
        self.slope_lookback = int(slope_lookback)
        self.veto_threshold = float(veto_threshold)
        self.veto_clear_days = int(veto_clear_days)
        self.short_threshold = float(short_threshold)
        self.short_low_lookback = int(short_low_lookback)
        self.full_forecast = float(full_forecast)
        self.short_forecast = float(short_forecast)
        self.z_window = int(z_window)

    def forecast_panel(self, close_panel: pd.DataFrame) -> pd.DataFrame:
        from src.data.fx_rates import (load_fx_rate_panel, build_rate_diff_panel,
                                        currencies_for_pairs)
        present = [p for p in self.universe if p in close_panel.columns]
        rate_panel = load_fx_rate_panel(currencies_for_pairs(present), close_panel.index)
        rate_diff = build_rate_diff_panel(present, rate_panel)

        close = close_panel[present].astype(float)
        ema = close.ewm(span=self.ema_span, adjust=False).mean()
        momentum_ok = (close > ema) & (ema > ema.shift(self.slope_lookback))
        carry_ok = rate_diff[present] > self.carry_gate
        longs = (carry_ok & momentum_ok).astype(float) * self.full_forecast

        score = compute_unwind_score(close_panel, z_window=self.z_window)
        veto = self._veto_mask(score)
        longs.loc[veto.values, :] = 0.0

        out = longs
        cascade = score > self.short_threshold
        for pair in ("AUDJPY", "NZDJPY"):
            if pair in out.columns:
                prior_low = close_panel[pair].rolling(self.short_low_lookback).min().shift(1)
                fire = cascade & (close_panel[pair] < prior_low)
                out.loc[fire.values, pair] = -self.short_forecast
        return out.fillna(0.0)

    def _veto_mask(self, score: pd.Series) -> pd.Series:
        engaged, run_below, mask = False, 0, []
        for v in score.values:
            if v >= self.veto_threshold:
                engaged, run_below = True, 0
            else:
                run_below += 1
                if run_below >= self.veto_clear_days:
                    engaged = False
            mask.append(engaged)
        return pd.Series(mask, index=score.index)
```

- [ ] **Step 4: Register the strategy**

In `src/strategies/registry.py`, add this line immediately after the `"FxXSectMom"` entry (currently line 68):

```python
    "FxCarrySeatbelt": ("src.strategies.advanced.fx_carry_seatbelt", "FxCarrySeatbelt"),
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `source ~/anaconda3/etc/profile.d/conda.sh; conda activate fintech; PYTHONPATH=$(pwd) python -m pytest tests/strategies/test_fx_carry_seatbelt.py -v`
Expected: 5 passed.

- [ ] **Step 6: Commit**

```bash
git add src/strategies/advanced/fx_carry_seatbelt.py src/strategies/registry.py tests/strategies/test_fx_carry_seatbelt.py
git commit -m "feat(fx): FxCarrySeatbelt strategy (#16 filter + #19 veto/short)"
```

---

### Task 4: Daily + weekly configs and end-to-end smoke

**Files:**
- Create: `config/backtesting/fx_carry_seatbelt_daily.yaml`
- Create: `config/backtesting/fx_carry_seatbelt_weekly.yaml`
- Test: `tests/backtesting/test_fx_carry_seatbelt_configs.py`

**Interfaces:**
- Consumes: `FxCarrySeatbelt` (Task 3), `run_fx_backtest` from `src.backtesting.engine.fx_backtest`.
- Produces: two runnable YAML configs; confirmation the strategy runs through the real engine end-to-end.

- [ ] **Step 1: Write the two configs**

Create `config/backtesting/fx_carry_seatbelt_daily.yaml`:

```yaml
asset_class: fx
strategy:
  name: FxCarrySeatbelt
  universe: [EURUSD, USDJPY, USDCHF, EURJPY, EURCHF, CHFJPY, XAUUSD, XAGUSD, GBPUSD, USDCAD, AUDUSD, NZDUSD, AUDNZD, AUDJPY, NZDJPY, EURNOK, EURSEK, USDNOK, USDSEK, NOKSEK, NOKJPY, SEKJPY]
  params: {}
dates:
  start: "2011-01-01"
  end: "2026-04-01"
backtest:
  initial_capital: 100000.0
  vol_target_per_instrument: 0.03
  rebalance: daily
  leverage_cap: 4.0
  idm: true
  idm_cap: 2.5
output:
  save_trades: true
```

Create `config/backtesting/fx_carry_seatbelt_weekly.yaml` identical except `rebalance: weekly`.

- [ ] **Step 2: Write the smoke test**

Create `tests/backtesting/test_fx_carry_seatbelt_configs.py`:

```python
import yaml
from pathlib import Path

import pytest

from src.backtesting.engine.fx_backtest import run_fx_backtest


@pytest.mark.parametrize("cadence", ["daily", "weekly"])
def test_config_runs_end_to_end(cadence):
    cfg = yaml.safe_load(
        Path(f"config/backtesting/fx_carry_seatbelt_{cadence}.yaml").read_text())
    # short window keeps the smoke test fast; full run happens in the harness
    cfg["dates"] = {"start": "2019-01-01", "end": "2021-01-01"}
    res = run_fx_backtest(cfg, register=False, log_trades=False)
    assert res["n_days"] > 100
    assert len(res["equity_curve"]) == res["n_days"]
```

- [ ] **Step 3: Run the smoke test**

Run: `source ~/anaconda3/etc/profile.d/conda.sh; conda activate fintech; PYTHONPATH=$(pwd) python -m pytest tests/backtesting/test_fx_carry_seatbelt_configs.py -v`
Expected: 2 passed. (This exercises the real engine, FRED rates, and the score end-to-end. If FRED parquet files are missing for some currencies the loader warns and uses 0.0, which is acceptable for a smoke test.)

- [ ] **Step 4: Commit**

```bash
git add config/backtesting/fx_carry_seatbelt_daily.yaml config/backtesting/fx_carry_seatbelt_weekly.yaml tests/backtesting/test_fx_carry_seatbelt_configs.py
git commit -m "feat(fx): FxCarrySeatbelt daily/weekly configs + end-to-end smoke"
```

---

### Task 5: S&P 500 benchmark module

**Files:**
- Create: `src/backtesting/benchmark.py`
- Test: `tests/backtesting/test_benchmark.py`

**Interfaces:**
- Consumes: `get_local_storage_dir` from `src.settings`; the SPX cache at `alt_data/equity_index/SPX/daily.parquet` (populated in Step 5 below).
- Produces:
  - `load_sp500_daily_returns() -> pd.Series` (datetime-indexed daily returns).
  - `sp500_sharpe_over_dates(dates, sp_returns=None) -> float` (annualized, rf=0).
  - `correlation_over_dates(strat_returns: pd.Series, sp_returns=None) -> float`.
  - `information_ratio_vs_sp500(strat_returns: pd.Series, sp_returns=None) -> float`.

- [ ] **Step 1: Write the failing tests**

Create `tests/backtesting/test_benchmark.py`:

```python
import numpy as np
import pandas as pd

from src.backtesting.benchmark import (
    sp500_sharpe_over_dates, correlation_over_dates, information_ratio_vs_sp500)


def _sp_returns(n=500):
    idx = pd.date_range("2015-01-01", periods=n, freq="B")
    rng = np.random.default_rng(3)
    return pd.Series(rng.normal(0.0004, 0.01, n), index=idx)


def test_sharpe_over_dates_uses_only_given_dates():
    sp = _sp_returns()
    subset = sp.index[100:200]
    got = sp500_sharpe_over_dates(subset, sp_returns=sp)
    expected = sp.reindex(pd.to_datetime(subset)).dropna()
    exp_sharpe = expected.mean() / expected.std(ddof=1) * np.sqrt(252)
    assert abs(got - exp_sharpe) < 1e-9


def test_correlation_of_series_with_itself_is_one():
    sp = _sp_returns()
    assert abs(correlation_over_dates(sp, sp_returns=sp) - 1.0) < 1e-9


def test_information_ratio_is_zero_against_itself():
    sp = _sp_returns()
    # active return (sp - sp) is all zeros -> IR is nan (zero std), handled
    ir = information_ratio_vs_sp500(sp, sp_returns=sp)
    assert np.isnan(ir)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `source ~/anaconda3/etc/profile.d/conda.sh; conda activate fintech; PYTHONPATH=$(pwd) python -m pytest tests/backtesting/test_benchmark.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.backtesting.benchmark'`.

- [ ] **Step 3: Write the implementation**

Create `src/backtesting/benchmark.py`:

```python
"""S&P 500 benchmark helpers for the relative success criterion.

The FX strategy passes only if its OOS Sharpe beats the S&P's over the SAME OOS
dates. These helpers load the cached SPX daily series (from the keyless
equity_index_yfinance plugin) and compute Sharpe / correlation / information
ratio over an arbitrary date index. All accept an injected `sp_returns` so tests
run without I/O.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.settings import get_local_storage_dir

_TRADING_DAYS = 252


def load_sp500_daily_returns() -> pd.Series:
    fp = get_local_storage_dir() / "alt_data" / "equity_index" / "SPX" / "daily.parquet"
    if not fp.exists():
        raise FileNotFoundError(
            f"S&P benchmark parquet missing at {fp}; populate it via "
            f"src.data.acquisition.plugins.equity_index_yfinance.fetch_index('SPX', ...)")
    df = pd.read_parquet(fp)
    s = pd.Series(df["close"].values,
                  index=pd.to_datetime(df["date"].values)).sort_index()
    return s.pct_change().dropna()


def _annualized_sharpe(returns: pd.Series) -> float:
    if returns.size < 2:
        return float("nan")
    std = float(returns.std(ddof=1))
    if std == 0.0 or np.isnan(std):
        return float("nan")
    return float(returns.mean()) / std * np.sqrt(_TRADING_DAYS)


def sp500_sharpe_over_dates(dates, sp_returns=None) -> float:
    if sp_returns is None:
        sp_returns = load_sp500_daily_returns()
    aligned = sp_returns.reindex(pd.to_datetime(pd.Index(dates))).dropna()
    return _annualized_sharpe(aligned)


def correlation_over_dates(strat_returns: pd.Series, sp_returns=None) -> float:
    if sp_returns is None:
        sp_returns = load_sp500_daily_returns()
    joined = pd.concat([strat_returns.rename("s"), sp_returns.rename("b")],
                       axis=1).dropna()
    if len(joined) < 2:
        return float("nan")
    return float(joined["s"].corr(joined["b"]))


def information_ratio_vs_sp500(strat_returns: pd.Series, sp_returns=None) -> float:
    if sp_returns is None:
        sp_returns = load_sp500_daily_returns()
    joined = pd.concat([strat_returns.rename("s"), sp_returns.rename("b")],
                       axis=1).dropna()
    if len(joined) < 2:
        return float("nan")
    active = joined["s"] - joined["b"]
    std = float(active.std(ddof=1))
    if std == 0.0 or np.isnan(std):
        return float("nan")
    return float(active.mean()) / std * np.sqrt(_TRADING_DAYS)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `source ~/anaconda3/etc/profile.d/conda.sh; conda activate fintech; PYTHONPATH=$(pwd) python -m pytest tests/backtesting/test_benchmark.py -v`
Expected: 3 passed.

- [ ] **Step 5: Populate the SPX cache (one-time, keyless)**

Run:
```bash
source ~/anaconda3/etc/profile.d/conda.sh; conda activate fintech
PYTHONPATH=$(pwd) python -c "from src.data.acquisition.plugins.equity_index_yfinance import fetch_index; d=fetch_index('SPX','2010-01-01','2026-04-02'); print('SPX rows:', len(d), d['date'].min(), d['date'].max())"
```
Expected: prints a few thousand rows spanning 2010 to 2026, and writes `alt_data/equity_index/SPX/daily.parquet`. If yfinance is rate-limited, retry once; the parquet must exist before Task 6 runs.

- [ ] **Step 6: Commit**

```bash
git add src/backtesting/benchmark.py tests/backtesting/test_benchmark.py
git commit -m "feat(fx): S&P 500 benchmark helpers (Sharpe/corr/IR over OOS dates)"
```

---

### Task 6: Walk-forward runner with S&P gate, diagnostics, and episode attribution

**Files:**
- Create: `scripts/backtest_scripts/run_fx_carry_seatbelt_walkforward.py`
- Modify: none (self-contained; reuses pure helpers from `walkforward_common`)

**Interfaces:**
- Consumes: `_build_windows`, `_as_date`, `_annualized_sharpe`, `_compute_pbo`, `_oos_returns_dated` from `src.backtesting.walkforward_common`; `run_fx_backtest` from `src.backtesting.engine.fx_backtest`; `load_fx_daily_panel` from `src.backtesting.data.fx_backtest_loader`; `psr`, `dsr` from `src.backtesting.statistics`; the Task 5 benchmark helpers; `RunStatus` from `src.utils.run_status`; `n_trials_project_wide` from `src.experiments`.
- Produces: a readiness report at `docs/reports/fx/FX_CARRY_SEATBELT_WALK_FORWARD.md` and prints the PASS/FAIL verdict.

**Design note (why self-contained):** the existing `run_fx_walkforward.py::process_window` returns bare numpy OOS arrays, discarding dates. This runner needs the dated OOS series (to align the S&P over the same dates and to slice named episodes), so it defines its own dated per-window worker rather than modifying the shared runner (zero regression risk to the FxTrend/FxValue path). The ~20-line duplication is the deliberate, lower-risk choice.

- [ ] **Step 1: Write the runner**

Create `scripts/backtest_scripts/run_fx_carry_seatbelt_walkforward.py`:

```python
"""FxCarrySeatbelt walk-forward with the S&P relative gate.

Rolls the standard 36/12/12 OOS windows on a given cadence config, stitches the
DATED OOS return series (both cost legs), and evaluates the primary criterion:
strategy OOS Sharpe > S&P Sharpe over the same dates. PSR/DSR/PBO, correlation,
IR, and per-episode attribution are computed and reported as diagnostics only.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.backtesting.benchmark import (
    load_sp500_daily_returns, sp500_sharpe_over_dates,
    correlation_over_dates, information_ratio_vs_sp500)
from src.backtesting.data.fx_backtest_loader import load_fx_daily_panel
from src.backtesting.engine.fx_backtest import run_fx_backtest
from src.backtesting.statistics.dsr import dsr
from src.backtesting.statistics.psr import psr
from src.backtesting.walkforward_common import (
    _as_date, _build_windows, _annualized_sharpe, _compute_pbo, _oos_returns_dated)
from src.utils import logger

_REPORT_PATH = "docs/reports/fx/FX_CARRY_SEATBELT_WALK_FORWARD.md"
# Named unwind episodes for existence-proof attribution (start, end inclusive).
_EPISODES = {
    "Aug 2024 yen-carry unwind": ("2024-07-15", "2024-08-15"),
    "Mar 2020 COVID unwind": ("2020-02-20", "2020-03-31"),
    "Jan 2019 flash": ("2019-01-01", "2019-01-10"),
}


def _run_window(spec: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    universe = spec["universe"]
    train_start, test_start, test_end = spec["train_start"], spec["test_start"], spec["test_end"]
    try:
        panel = load_fx_daily_panel(universe, train_start, test_end)
    except FileNotFoundError as e:
        logger.warning(f"[seatbelt_wf] skip {test_start}..{test_end}: {e}")
        return None
    window_universe = sorted({p for p, _ in panel.columns})
    dates = list(panel.index)

    def one(cost_mult: float):
        cfg = {"asset_class": "fx",
               "strategy": {"name": "FxCarrySeatbelt", "universe": window_universe, "params": {}},
               "dates": {"start": str(train_start), "end": str(test_end)},
               "backtest": {"initial_capital": spec["capital"],
                            "vol_target_per_instrument": spec["vol_target"],
                            "rebalance": spec["rebalance"], "cost_mult": cost_mult,
                            "leverage_cap": spec["leverage_cap"], "idm": spec["idm"],
                            "idm_cap": spec["idm_cap"]}}
        res = run_fx_backtest(cfg, register=False, log_trades=False)
        eq = pd.Series(res["equity_curve"], index=pd.Index(dates))
        oos = _oos_returns_dated(res["equity_curve"], dates, test_start)
        is_ret = eq[eq.index < pd.Timestamp(test_start)].pct_change().dropna()
        return oos, is_ret

    oos_1x, is_1x = one(1.0)
    oos_1_5x, _ = one(1.5)
    return {"oos_1x": oos_1x, "oos_1_5x": oos_1_5x, "is_1x": is_1x}


def run(config_path: str, cadence_label: str, trial_count: int,
        train_months: int = 36, test_months: int = 12, step_months: int = 12) -> Dict[str, Any]:
    import yaml
    cfg = yaml.safe_load(Path(config_path).read_text())
    strat, bt, dts = cfg["strategy"], cfg["backtest"], cfg["dates"]
    universe = list(strat["universe"])
    start_d, end_d = _as_date(dts["start"]), _as_date(dts["end"])
    windows = _build_windows(train_months, test_months, step_months, start_d, end_d)

    specs = [{"universe": universe, "train_start": ts, "test_start": tst, "test_end": te,
              "capital": float(bt["initial_capital"]),
              "vol_target": float(bt["vol_target_per_instrument"]),
              "rebalance": bt.get("rebalance", "daily"),
              "leverage_cap": float(bt.get("leverage_cap", 4.0)),
              "idm": bool(bt.get("idm", True)), "idm_cap": bt.get("idm_cap")}
             for (ts, tst, te) in windows]

    from src.backtesting.parallel import parallel_map
    results = [r for r in parallel_map(_run_window, specs) if r is not None]
    if len(results) < 2:
        raise ValueError(f"need >=2 usable OOS windows, got {len(results)}")

    oos_1x = pd.concat([r["oos_1x"] for r in results]).sort_index()
    oos_1_5x = pd.concat([r["oos_1_5x"] for r in results]).sort_index()
    per_window_1x = [r["oos_1x"].to_numpy(dtype=float) for r in results]

    arr = oos_1x.to_numpy(dtype=float)
    n = int(arr.size)
    sharpe = _annualized_sharpe(arr)
    sharpe_1_5x = _annualized_sharpe(oos_1_5x.to_numpy(dtype=float))
    # IS/OOS overfit diagnostic: mean of per-window in-sample Sharpes (avoids
    # double-counting the heavily-overlapping train segments).
    per_window_is = [_annualized_sharpe(r["is_1x"].to_numpy(dtype=float))
                     for r in results if r["is_1x"].size > 1]
    is_sharpe = float(np.nanmean(per_window_is)) if per_window_is else float("nan")
    is_oos_ratio = (is_sharpe / sharpe
                    if sharpe not in (0.0,) and not np.isnan(sharpe) else float("nan"))
    ser = pd.Series(arr)
    skew = float(ser.skew()) if n > 2 else 0.0
    kurt = float(ser.kurtosis()) + 3.0 if n > 3 else 3.0
    psr_val = psr(sharpe, 0.0, n, skew, kurt)
    dsr_val = dsr(sharpe, [sharpe], n, skew, kurt, n_trials_project=trial_count)
    pbo_val = _compute_pbo(per_window_1x)

    sp = load_sp500_daily_returns()
    sp_sharpe = sp500_sharpe_over_dates(oos_1x.index, sp_returns=sp)
    corr = correlation_over_dates(oos_1x, sp_returns=sp)
    ir = information_ratio_vs_sp500(oos_1x, sp_returns=sp)
    beats = bool(sharpe > sp_sharpe)

    episodes = {}
    for name, (s, e) in _EPISODES.items():
        seg = oos_1x[(oos_1x.index >= pd.Timestamp(s)) & (oos_1x.index <= pd.Timestamp(e))]
        episodes[name] = float((1.0 + seg).prod() - 1.0) if len(seg) else float("nan")

    return {"cadence": cadence_label, "n_oos_days": n, "n_windows": len(results),
            "oos_sharpe": sharpe, "oos_sharpe_1_5x": sharpe_1_5x,
            "is_sharpe": is_sharpe, "is_oos_ratio": is_oos_ratio,
            "sp500_sharpe": sp_sharpe, "beats_sp500": beats,
            "correlation_sp500": corr, "information_ratio_sp500": ir,
            "psr": psr_val, "dsr": dsr_val, "pbo": pbo_val, "skew": skew,
            "kurtosis_pearson": kurt, "trial_count": trial_count,
            "oos_start": str(oos_1x.index.min().date()),
            "oos_end": str(oos_1x.index.max().date()), "episodes": episodes}


def _write_report(results: List[Dict[str, Any]], path: str = _REPORT_PATH) -> str:
    lines = ["# FxCarrySeatbelt Walk-Forward Readiness Report", "",
             "Generated by `scripts/backtest_scripts/run_fx_carry_seatbelt_walkforward.py`.",
             "Primary gate: OOS Sharpe (1x cost) > S&P 500 Sharpe over the same OOS dates.",
             "PSR/DSR/PBO are diagnostics, not gates (see the 2026-07-06 pre-registration).", ""]
    for r in results:
        verdict = "PASS" if r["beats_sp500"] else "FAIL"
        lines += [f"## Cadence: {r['cadence']} -- {verdict}", "",
                  "| Metric | Value |", "|---|---|",
                  f"| OOS Sharpe (1x) | {r['oos_sharpe']:.4f} |",
                  f"| S&P Sharpe (same dates) | {r['sp500_sharpe']:.4f} |",
                  f"| Beats S&P | {r['beats_sp500']} |",
                  f"| OOS Sharpe (1.5x cost) | {r['oos_sharpe_1_5x']:.4f} |",
                  f"| IS Sharpe (1x, mean per-window) | {r['is_sharpe']:.4f} |",
                  f"| IS/OOS Sharpe ratio | {r['is_oos_ratio']:.4f} |",
                  f"| Correlation to S&P | {r['correlation_sp500']:.4f} |",
                  f"| Information ratio vs S&P | {r['information_ratio_sp500']:.4f} |",
                  f"| PSR (diag) | {r['psr']:.4f} |",
                  f"| DSR (diag, trials={r['trial_count']}) | {r['dsr']:.4f} |",
                  f"| PBO (diag) | {r['pbo']:.4f} |",
                  f"| n_windows / n_oos_days | {r['n_windows']} / {r['n_oos_days']} |",
                  f"| OOS window | {r['oos_start']} .. {r['oos_end']} |", "",
                  "### Episode attribution (existence proof, not statistics)", "",
                  "| Episode | Strategy OOS return |", "|---|---|"]
        for name, val in r["episodes"].items():
            shown = "n/a (outside OOS)" if np.isnan(val) else f"{val:+.2%}"
            lines.append(f"| {name} | {shown} |")
        lines += ["",
                  "Limitations: carry gate uses the FRED policy-rate differential as a swap",
                  "proxy (optimism bias); the offensive short rests on ~4-6 events.", ""]
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    return str(out)


def main() -> None:
    import argparse
    from src.utils.run_status import RunStatus
    try:
        from src.experiments import n_trials_project_wide
        base_trials = int(n_trials_project_wide())
    except Exception:
        base_trials = 0

    parser = argparse.ArgumentParser(description="FxCarrySeatbelt walk-forward + S&P gate")
    parser.add_argument("--report", default=_REPORT_PATH)
    args = parser.parse_args()

    configs = [("config/backtesting/fx_carry_seatbelt_daily.yaml", "daily"),
               ("config/backtesting/fx_carry_seatbelt_weekly.yaml", "weekly")]
    # Two new configs increment the project-wide trial count for the DSR diagnostic.
    trial_count = base_trials + len(configs)

    with RunStatus("fx_carry_seatbelt_walkforward", meta={"configs": [c for c, _ in configs]}):
        results = [run(path, label, trial_count) for path, label in configs]
        report_path = _write_report(results, args.report)

    for r in results:
        logger.info(f"[seatbelt_wf] {r['cadence']}: oos_sharpe={r['oos_sharpe']:.4f} "
                    f"sp500={r['sp500_sharpe']:.4f} beats={r['beats_sp500']} "
                    f"dsr={r['dsr']:.4f} pbo={r['pbo']:.4f}")
    logger.info(f"[seatbelt_wf] wrote {report_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the walk-forward**

Run:
```bash
source ~/anaconda3/etc/profile.d/conda.sh; conda activate fintech
PYTHONPATH=$(pwd) python scripts/backtest_scripts/run_fx_carry_seatbelt_walkforward.py
```
Expected: logs a `daily` and a `weekly` line each with `oos_sharpe`, `sp500`, `beats`, and writes `docs/reports/fx/FX_CARRY_SEATBELT_WALK_FORWARD.md`. Runtime is a few minutes (two cadences x ~13 windows x 2 cost legs, parallelized).

- [ ] **Step 3: Verify the report exists and record the verdict**

Run: `cat docs/reports/fx/FX_CARRY_SEATBELT_WALK_FORWARD.md`
Confirm both cadence sections rendered with a PASS/FAIL verdict, the S&P comparison, the diagnostics, and the episode table.

- [ ] **Step 4: Commit**

```bash
git add scripts/backtest_scripts/run_fx_carry_seatbelt_walkforward.py docs/reports/fx/FX_CARRY_SEATBELT_WALK_FORWARD.md
git commit -m "feat(fx): FxCarrySeatbelt walk-forward runner + S&P gate report"
```

---

## Post-implementation (orchestrator, after all tasks)

- Update `docs/strategies/FX_60_CATALOG_TRACKER.md`: mark #16 and #19 with the seatbelt verdict (PASS/FAIL vs S&P per cadence), linking the readiness report. This is the 5th gated strategy of the 60.
- Per the pre-registration: if neither cadence beats the S&P, decide (with the diagnostics in hand) whether to run the single deferred variant (#16 mod-a 12-month TSMOM momentum leg or mod-b graded sizing) or shelve the carry idea. Do not silently start a second variant without recording that decision.
```
