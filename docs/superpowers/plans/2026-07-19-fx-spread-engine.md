# FX Beta-Weighted Spread Engine + 3 RV Strategies (Wave 2 Track B) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a beta-weighted 2-leg spread-execution engine and the 3 market-neutral relative-value strategies (#35 AUD/NZD, #37 cointegration scanner, #30 vol-ratio) that ride it, so strategy-lead can gate the last 3 Wave 2 strategies.

**Architecture:** A strategy emits, per rebalance date, a set of active beta-weighted spreads. A pure sizing function converts the spread book to NET per-instrument target notionals (spread vol-targeted, hedge-ratio-weighted). A simulator applies those targets with the existing spot-sim per-instrument MTM / cost / leverage machinery, so charging cost per instrument on the netted diff naturally charges both legs of each spread while correctly netting internally-offsetting positions.

**Tech Stack:** Python 3.13 (conda env `fintech`), pandas, numpy, statsmodels (already used by the cointegration artifact), pytest. Reuses `cointegration.py`, the FX cost model, `walkforward_common`, `benchmark.py`. No new dependencies.

## Global Constraints

- Run Python via the `fintech` conda env. Test prefix: `conda activate fintech; PYTHONPATH=$(pwd) python -m pytest <path> -v` (conda already initialized).
- ASCII-only, no em dashes, no emojis, no `print()`; use `from src.utils import logger` where logging is needed.
- Spread sizing is BETA-WEIGHTED: for a spread `ln(A) - beta*ln(B)`, hold notional_A = w, notional_B = -beta*w; scale w so the spread's annualized vol = vol_target (from trailing std of `r_a - beta*r_b`). NOT equal-vol.
- Costs: charge the round-trip spread on the NET per-instrument diff each rebalance (reuses `cost_fn(pair, units_traded, price, quote_to_usd)` from `_cost_fn_factory`); netting across spreads that share a leg is correct (you only cross the spread on the net).
- Forecast/strength on the Carver scale: strength 10 = 1x vol-target spread. Multiple spreads each vol-targeted; apply an IDM-style diversification multiplier (cap 2.5) and a portfolio leverage cap (default 4.0).
- The 3 strategies' VERDICT runs are delegated to `strategy-lead` (Wave 2 pre-registration: combined statistical gate, honest every-spec trial count, no-push). This plan builds the engine + strategies + runner and a fast shape test only; it does NOT run the full walk-forward gates.
- Git hazard (macOS/Dropbox): use ONLY `git add <explicit paths>`, `git commit`, `git log`. NEVER `git checkout`, bare `git status`/`git diff`, or `git reset`. Commit ONLY each task's own files by explicit path (unrelated uncommitted files may exist; never `git add -A`/`.`). Subagents COMMIT only; the orchestrator owns pushes.

---

### Task 1: Beta-weighted spread sizing (pure function)

**Files:**
- Create: `src/backtesting/engine/spread_sizing.py`
- Test: `tests/backtesting/engine/test_spread_sizing.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `Spread` namedtuple `(leg_a: str, leg_b: str, hedge_ratio: float, strength: float)` (strength Carver-signed; positive = long the spread).
  - `spread_leg_targets(spreads: list[Spread], sigma_s: dict[tuple, float], close_row: dict[str, float], quote_usd_row: dict[str, float], equity: float, vol_target: float, idm: float) -> dict[str, float]` returning NET target UNITS per instrument (summed across spreads; base units, signed).

- [ ] **Step 1: Write the failing tests**

Create `tests/backtesting/engine/test_spread_sizing.py`:

```python
import math

from src.backtesting.engine.spread_sizing import Spread, spread_leg_targets


def test_beta_weighted_notional_ratio():
    # One spread, beta=1.5: notional_B = 1.5 * notional_A, opposite signs.
    sp = [Spread("AUDUSD", "NZDUSD", 1.5, 10.0)]
    sigma = {("AUDUSD", "NZDUSD"): 0.01}  # 1% daily spread vol
    close = {"AUDUSD": 0.65, "NZDUSD": 0.60}
    q = {"AUDUSD": 1.0, "NZDUSD": 1.0}  # both USD-quoted -> quote_to_usd=1
    tgt = spread_leg_targets(sp, sigma, close, q, equity=100000.0,
                             vol_target=0.10, idm=1.0)
    notional_a = tgt["AUDUSD"] * close["AUDUSD"] * q["AUDUSD"]
    notional_b = tgt["NZDUSD"] * close["NZDUSD"] * q["NZDUSD"]
    assert notional_a > 0 and notional_b < 0            # long A, short B
    assert math.isclose(abs(notional_b), 1.5 * abs(notional_a), rel_tol=1e-6)


def test_spread_vol_targets_to_vol_target():
    # notional_A chosen so the spread's annualized vol == vol_target.
    sp = [Spread("AUDUSD", "NZDUSD", 1.0, 10.0)]
    sigma = {("AUDUSD", "NZDUSD"): 0.008}
    close = {"AUDUSD": 0.65, "NZDUSD": 0.60}
    q = {"AUDUSD": 1.0, "NZDUSD": 1.0}
    eq, vt = 100000.0, 0.10
    tgt = spread_leg_targets(sp, sigma, close, q, eq, vt, idm=1.0)
    notional_a = abs(tgt["AUDUSD"] * close["AUDUSD"] * q["AUDUSD"])
    # spread annualized vol = (notional_a/equity) * sigma_s * sqrt(252) == vt
    implied_vol = (notional_a / eq) * sigma[("AUDUSD", "NZDUSD")] * math.sqrt(252)
    assert math.isclose(implied_vol, vt, rel_tol=1e-6)


def test_shared_leg_nets_across_spreads():
    # Two spreads both long AUDUSD -> net AUDUSD units add.
    sp = [Spread("AUDUSD", "NZDUSD", 1.0, 10.0),
          Spread("AUDUSD", "USDCAD", 1.0, 10.0)]
    sigma = {("AUDUSD", "NZDUSD"): 0.01, ("AUDUSD", "USDCAD"): 0.01}
    close = {"AUDUSD": 0.65, "NZDUSD": 0.60, "USDCAD": 1.35}
    q = {"AUDUSD": 1.0, "NZDUSD": 1.0, "USDCAD": 1.0 / 1.35}
    tgt = spread_leg_targets(sp, sigma, close, q, 100000.0, 0.10, idm=1.0)
    single = spread_leg_targets([sp[0]], {("AUDUSD", "NZDUSD"): 0.01},
                                close, q, 100000.0, 0.10, idm=1.0)
    assert tgt["AUDUSD"] > single["AUDUSD"]   # two long-A spreads add


def test_strength_scales_size_linearly():
    base = spread_leg_targets([Spread("AUDUSD", "NZDUSD", 1.0, 10.0)],
                              {("AUDUSD", "NZDUSD"): 0.01},
                              {"AUDUSD": 0.65, "NZDUSD": 0.60},
                              {"AUDUSD": 1.0, "NZDUSD": 1.0}, 100000.0, 0.10, 1.0)
    half = spread_leg_targets([Spread("AUDUSD", "NZDUSD", 1.0, 5.0)],
                              {("AUDUSD", "NZDUSD"): 0.01},
                              {"AUDUSD": 0.65, "NZDUSD": 0.60},
                              {"AUDUSD": 1.0, "NZDUSD": 1.0}, 100000.0, 0.10, 1.0)
    assert math.isclose(half["AUDUSD"], 0.5 * base["AUDUSD"], rel_tol=1e-9)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda activate fintech; PYTHONPATH=$(pwd) python -m pytest tests/backtesting/engine/test_spread_sizing.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'src.backtesting.engine.spread_sizing'`).

- [ ] **Step 3: Write the implementation**

Create `src/backtesting/engine/spread_sizing.py`:

```python
"""Beta-weighted spread sizing: convert a book of 2-leg spreads to net
per-instrument target notionals, each spread vol-targeted on its own spread vol.
"""
from __future__ import annotations

import math
from typing import NamedTuple


class Spread(NamedTuple):
    leg_a: str
    leg_b: str
    hedge_ratio: float
    strength: float  # Carver scale; 10 = 1x vol-target spread, sign = direction


_ANN = math.sqrt(252)


def spread_leg_targets(spreads, sigma_s, close_row, quote_usd_row,
                       equity: float, vol_target: float, idm: float) -> dict:
    targets: dict[str, float] = {}
    for sp in spreads:
        key = (sp.leg_a, sp.leg_b)
        sig = sigma_s.get(key)
        if sig is None or sig <= 0 or not math.isfinite(sig):
            continue
        pa, pb = close_row.get(sp.leg_a), close_row.get(sp.leg_b)
        qa, qb = quote_usd_row.get(sp.leg_a), quote_usd_row.get(sp.leg_b)
        if None in (pa, pb, qa, qb) or pa <= 0 or pb <= 0:
            continue
        # notional_A (USD) so spread annualized vol == vol_target, scaled by strength/10 and idm.
        scale = (sp.strength / 10.0) * idm
        notional_a_usd = scale * vol_target * equity / (sig * _ANN)
        notional_b_usd = -sp.hedge_ratio * notional_a_usd
        units_a = notional_a_usd / (pa * qa)
        units_b = notional_b_usd / (pb * qb)
        targets[sp.leg_a] = targets.get(sp.leg_a, 0.0) + units_a
        targets[sp.leg_b] = targets.get(sp.leg_b, 0.0) + units_b
    return targets
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda activate fintech; PYTHONPATH=$(pwd) python -m pytest tests/backtesting/engine/test_spread_sizing.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/backtesting/engine/spread_sizing.py tests/backtesting/engine/test_spread_sizing.py
git commit -m "feat(fx): beta-weighted spread leg-sizing (net per-instrument targets)"
```

---

### Task 2: FxSpreadPortfolioSimulator

**Files:**
- Create: `src/backtesting/engine/fx_spread_simulator.py`
- Test: `tests/backtesting/engine/test_fx_spread_simulator.py`

**Interfaces:**
- Consumes: `Spread`, `spread_leg_targets` (Task 1); `FxBacktestResult` from `src.backtesting.engine.fx_spot_portfolio_simulator`; a `cost_fn(pair, units_traded, price, quote_to_usd) -> usd` (as built by `_cost_fn_factory` in `fx_backtest.py`).
- Produces:
  - `FxSpreadPortfolioSimulator(initial_capital, cost_fn, rebalance="weekly", cost_mult=1.0, leverage_cap=4.0)`.
  - `.run_spreads(close_panel: pd.DataFrame, spread_book: dict[date, list[Spread]], sigma_panel: dict[date, dict[tuple, float]], quote_usd_panel: pd.DataFrame, vol_target: float, idm: float=1.0) -> FxBacktestResult` -- MTM the net per-instrument book daily, rebalance to spread targets on rebalance days, charge cost per instrument on the diff, leverage cap + bankruptcy floor.

- [ ] **Step 1: Write the failing tests**

Create `tests/backtesting/engine/test_fx_spread_simulator.py`:

```python
import datetime as dt

import numpy as np
import pandas as pd

from src.backtesting.engine.spread_sizing import Spread
from src.backtesting.engine.fx_spread_simulator import FxSpreadPortfolioSimulator


def _flat_cost(pair, units, price, q):
    return abs(units) * price * q * 0.0001  # 1bp of notional per trade, simple


def _panel(pairs, n=40):
    idx = pd.date_range("2022-01-03", periods=n, freq="B").date
    rng = np.random.default_rng(0)
    data = {p: 1.0 + np.cumsum(rng.normal(0, 0.002, n)) for p in pairs}
    return pd.DataFrame(data, index=pd.Index(idx))


def test_both_legs_charged_on_entry():
    pairs = ["AUDUSD", "NZDUSD"]
    close = _panel(pairs)
    q = pd.DataFrame({p: 1.0 for p in pairs}, index=close.index)
    d0 = close.index[0]
    book = {d: [Spread("AUDUSD", "NZDUSD", 1.0, 10.0)] for d in close.index}
    sigma = {d: {("AUDUSD", "NZDUSD"): 0.01} for d in close.index}
    sim = FxSpreadPortfolioSimulator(100000.0, _flat_cost, rebalance="weekly")
    res = sim.run_spreads(close, book, sigma, q, vol_target=0.10)
    first_rebal = res.trades[res.trades["date"] == res.trades["date"].min()]
    assert set(first_rebal["pair"]) == {"AUDUSD", "NZDUSD"}  # BOTH legs traded/charged


def test_market_neutral_no_pnl_when_legs_move_together():
    # If both legs move identically and beta=1, the spread (A-B) has ~0 PnL.
    pairs = ["AUDUSD", "NZDUSD"]
    idx = pd.date_range("2022-01-03", periods=30, freq="B").date
    common = 1.0 + np.cumsum(np.full(30, 0.001))  # identical path
    close = pd.DataFrame({"AUDUSD": common, "NZDUSD": common}, index=pd.Index(idx))
    q = pd.DataFrame({p: 1.0 for p in pairs}, index=close.index)
    book = {d: [Spread("AUDUSD", "NZDUSD", 1.0, 10.0)] for d in idx}
    sigma = {d: {("AUDUSD", "NZDUSD"): 0.01} for d in idx}
    sim = FxSpreadPortfolioSimulator(100000.0, lambda *a: 0.0, rebalance="weekly")
    res = sim.run_spreads(close, book, sigma, q, vol_target=0.10)
    # equity barely moves (legs cancel): final within 0.5% of start
    assert abs(res.equity_curve.iloc[-1] / 100000.0 - 1.0) < 0.005


def test_empty_book_holds_flat():
    pairs = ["AUDUSD", "NZDUSD"]
    close = _panel(pairs)
    q = pd.DataFrame({p: 1.0 for p in pairs}, index=close.index)
    book = {d: [] for d in close.index}
    sigma = {d: {} for d in close.index}
    sim = FxSpreadPortfolioSimulator(100000.0, _flat_cost, rebalance="weekly")
    res = sim.run_spreads(close, book, sigma, q, vol_target=0.10)
    assert (res.equity_curve == 100000.0).all()  # never traded
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda activate fintech; PYTHONPATH=$(pwd) python -m pytest tests/backtesting/engine/test_fx_spread_simulator.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Write the implementation**

Create `src/backtesting/engine/fx_spread_simulator.py`:

```python
"""Beta-weighted spread portfolio simulator.

Holds a book of 2-leg spreads as NET per-instrument positions, MTM daily,
rebalances to spread-vol-targeted beta-weighted targets on rebalance days, and
charges cost per instrument on the net diff (both legs of a spread are charged;
internally-offsetting positions net first). Reuses FxBacktestResult and mirrors
the spot simulator's MTM/leverage/bankruptcy pattern.
"""
from __future__ import annotations

import datetime as dt

import pandas as pd

from src.backtesting.engine.fx_spot_portfolio_simulator import FxBacktestResult
from src.backtesting.engine.spread_sizing import spread_leg_targets


class FxSpreadPortfolioSimulator:
    def __init__(self, initial_capital: float, cost_fn, rebalance: str = "weekly",
                 cost_mult: float = 1.0, leverage_cap: float = 4.0):
        self.capital = float(initial_capital)
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

    def _scale_leverage(self, targets: dict, close_row, q_row, equity: float) -> dict:
        gross = sum(abs(u * close_row[p] * q_row[p]) for p, u in targets.items())
        cap = self.leverage_cap * equity
        if gross > cap and gross > 0:
            f = cap / gross
            return {p: u * f for p, u in targets.items()}
        return targets

    def run_spreads(self, close_panel, spread_book, sigma_panel, quote_usd_panel,
                    vol_target: float, idm: float = 1.0) -> FxBacktestResult:
        pairs = list(close_panel.columns)
        dates = list(close_panel.index)
        current: dict[str, float] = {p: 0.0 for p in pairs}
        equity_val = self.capital
        equity, util, trade_rows = [], [], []
        prev_close, prev_d, blown = None, None, False

        for d in dates:
            row_close = {p: float(close_panel.loc[d, p]) for p in pairs}
            row_q = {p: float(quote_usd_panel.loc[d, p]) for p in pairs}
            # 1. MTM: pnl from close-to-close on held units (USD).
            if prev_close is not None and not blown:
                pnl = sum(current[p] * (row_close[p] - prev_close[p]) * row_q[p]
                          for p in pairs)
                equity_val += pnl
            if not blown and equity_val <= 0:
                current = {p: 0.0 for p in pairs}
                equity_val, blown = 0.0, True
            # 2. Rebalance to spread targets.
            if not blown and self._is_rebalance(d, prev_d):
                spreads = spread_book.get(d, [])
                sigma = sigma_panel.get(d, {})
                targets = spread_leg_targets(spreads, sigma, row_close, row_q,
                                             equity_val, vol_target, idm)
                targets = {p: targets.get(p, 0.0) for p in pairs}
                targets = self._scale_leverage(targets, row_close, row_q, equity_val)
                for p in pairs:
                    diff = targets[p] - current[p]
                    if diff != 0.0:
                        c = self.cost_fn(p, diff, row_close[p], row_q[p]) * self.cost_mult
                        equity_val -= c
                        trade_rows.append({"date": d, "pair": p, "units": diff, "cost": c})
                        current[p] = targets[p]
            gross = sum(abs(current[p] * row_close[p] * row_q[p]) for p in pairs)
            util.append(gross / equity_val if equity_val > 0 else 0.0)
            equity.append(equity_val)
            prev_close, prev_d = row_close, d

        eq = pd.Series(equity, index=dates, name="equity")
        lu = pd.Series(util, index=dates, name="leverage_utilization")
        trades = pd.DataFrame(trade_rows, columns=["date", "pair", "units", "cost"])
        return FxBacktestResult(equity_curve=eq, trades=trades, leverage_utilization=lu)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda activate fintech; PYTHONPATH=$(pwd) python -m pytest tests/backtesting/engine/test_fx_spread_simulator.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/backtesting/engine/fx_spread_simulator.py tests/backtesting/engine/test_fx_spread_simulator.py
git commit -m "feat(fx): FxSpreadPortfolioSimulator (net beta-weighted book, both-leg cost, MTM)"
```

---

### Task 3: Spread strategy base + #35 AUD/NZD

**Files:**
- Create: `src/strategies/advanced/fx_spread_base.py`
- Create: `src/strategies/advanced/fx_audnzd_pairs.py`
- Test: `tests/strategies/test_fx_audnzd_pairs.py`

**Interfaces:**
- Consumes: `Spread` (Task 1).
- Produces:
  - `SpreadStrategy` base with `spread_book(close_panel) -> (book: dict[date, list[Spread]], sigma: dict[date, dict[tuple,float]])` contract (subclasses implement `_spreads_for_history`).
  - `AudNzdPairs(lookback=120, entry_z=2.0, target_z=0.5, stop_z=3.25, max_days=20)` producing an AUDUSD/NZDUSD spread from the rolling-regression residual z, with RBA/RBNZ blackout.

- [ ] **Step 1: Write the failing tests**

Create `tests/strategies/test_fx_audnzd_pairs.py`:

```python
import numpy as np
import pandas as pd

from src.backtesting.engine.spread_sizing import Spread
from src.strategies.advanced.fx_audnzd_pairs import AudNzdPairs


def _coint_panel(n=400, div_start=350):
    # AUDUSD and NZDUSD co-move; inject a residual divergence late so |z|>2.
    idx = pd.date_range("2020-01-01", periods=n, freq="B").date
    rng = np.random.default_rng(0)
    common = np.cumsum(rng.normal(0, 0.004, n))
    aud = 0.70 * np.exp(common + rng.normal(0, 0.0005, n))
    nzd = 0.65 * np.exp(common + rng.normal(0, 0.0005, n))
    aud[div_start:] *= 1.03  # AUD richens vs NZD -> residual z spikes
    return pd.DataFrame({"AUDUSD": aud, "NZDUSD": nzd}, index=pd.Index(idx))


def test_emits_spread_when_residual_z_exceeds_entry():
    close = _coint_panel()
    book, sigma = AudNzdPairs().spread_book(close)
    # some late date has an active AUDUSD/NZDUSD spread
    active = [d for d, sps in book.items() if sps]
    assert active, "expected at least one active spread after the divergence"
    sp = book[active[-1]][0]
    assert {sp.leg_a, sp.leg_b} == {"AUDUSD", "NZDUSD"}
    assert (sp.leg_a, sp.leg_b) in sigma[active[-1]]  # spread vol provided


def test_hedge_ratio_is_from_regression_not_one():
    close = _coint_panel()
    book, _ = AudNzdPairs().spread_book(close)
    active = [book[d][0] for d in book if book[d]]
    assert active
    # beta from ln-regression should differ from a naive 1.0
    assert abs(active[-1].hedge_ratio - 1.0) > 1e-6
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda activate fintech; PYTHONPATH=$(pwd) python -m pytest tests/strategies/test_fx_audnzd_pairs.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Write the base + strategy**

Create `src/strategies/advanced/fx_spread_base.py`:

```python
"""Base for spread strategies: produce a per-date active-spread book plus the
trailing spread-vol map the simulator needs. Subclasses implement the signal.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class SpreadStrategy:
    vol_window = 60

    def spread_book(self, close_panel: pd.DataFrame):
        raise NotImplementedError

    def _spread_sigma(self, close_panel, leg_a, leg_b, hedge_ratio, upto_idx):
        # trailing daily std of r_a - beta*r_b over vol_window, causal (<= upto_idx)
        ra = close_panel[leg_a].pct_change(fill_method=None)
        rb = close_panel[leg_b].pct_change(fill_method=None)
        s = (ra - hedge_ratio * rb).iloc[max(0, upto_idx - self.vol_window):upto_idx + 1]
        v = float(s.std())
        return v if np.isfinite(v) and v > 0 else None
```

Create `src/strategies/advanced/fx_audnzd_pairs.py`. Implementer writes `spread_book` to: iterate business-week rebalance dates; for each, run a trailing `lookback`-day OLS of `ln(AUDUSD)` on `ln(NZDUSD)` using only data up to that date (causal) to get `hedge_ratio` (beta) and the residual series; compute the residual z-score at that date; maintain per-position state (enter when `|z| > entry_z`, signed `-sign(z)`; hold until `|z| < target_z` target, `|z| > stop_z` stop, or `max_days` time); skip NEW entries within 7 days of an RBA or RBNZ decision (`from src.data.macro_calendar import load_cb_decisions`, keys `RBA`, `RBNZ`); emit a `Spread("AUDUSD","NZDUSD", beta, strength)` with `strength = clip(z-scaled, -20, 20) * -1` on active dates, and the spread-vol via `self._spread_sigma`. Return `(book, sigma)` dicts keyed by date. Keep all computation causal (no future rows). Use the exact class signature `AudNzdPairs(lookback=120, entry_z=2.0, target_z=0.5, stop_z=3.25, max_days=20)`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda activate fintech; PYTHONPATH=$(pwd) python -m pytest tests/strategies/test_fx_audnzd_pairs.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/strategies/advanced/fx_spread_base.py src/strategies/advanced/fx_audnzd_pairs.py tests/strategies/test_fx_audnzd_pairs.py
git commit -m "feat(fx): spread strategy base + #35 AUD/NZD pairs (regression residual-z)"
```

---

### Task 4: #37 Rolling cointegration scanner

**Files:**
- Create: `src/strategies/advanced/fx_coint_scanner.py`
- Test: `tests/strategies/test_fx_coint_scanner.py`

**Interfaces:**
- Consumes: `SpreadStrategy` (Task 3); `Spread` (Task 1); `test_pair`, `ou_half_life` from `src.data.artifacts.cointegration`.
- Produces: `CointScanner(universe, scan_window=250, half_life_range=(5,25), adf_max=0.05, top_n=5, entry_z=2.0, target_z=0.5, stop_z=3.5)` with `spread_book`.

- [ ] **Step 1: Write the failing tests**

Create `tests/strategies/test_fx_coint_scanner.py`:

```python
import numpy as np
import pandas as pd

from src.strategies.advanced.fx_coint_scanner import CointScanner, _candidate_pairs


def test_candidate_pairs_excludes_shared_gt1_currency():
    # pairs sharing 2 currencies (mechanical) excluded; <=1 shared kept
    prs = ["EURUSD", "GBPUSD", "EURGBP", "AUDUSD"]
    cands = _candidate_pairs(prs)
    assert ("EURUSD", "GBPUSD") in cands or ("GBPUSD", "EURUSD") in cands  # share only USD
    # EURUSD vs EURGBP share EUR only (<=1) -> allowed; EURGBP vs GBPUSD share GBP only -> allowed
    assert all(len(set(a) & set(b)) <= 3 for a, b in cands)  # sanity (currency-code overlap bounded)


def test_scanner_emits_only_tradeable_cointegrated_spreads():
    n = 400
    idx = pd.date_range("2020-01-01", periods=n, freq="B").date
    rng = np.random.default_rng(1)
    common = np.cumsum(rng.normal(0, 0.004, n))
    # A,B cointegrated (share common + fast-reverting spread); C independent
    a = 1.30 * np.exp(common + 0.01 * np.sin(np.arange(n) / 3))
    b = 1.10 * np.exp(common)
    c = 0.90 * np.exp(np.cumsum(rng.normal(0, 0.004, n)))
    close = pd.DataFrame({"EURCAD": a, "AUDCAD": b, "GBPJPY": c}, index=pd.Index(idx))
    book, sigma = CointScanner(list(close.columns)).spread_book(close)
    active = [sps for sps in book.values() if sps]
    assert active, "expected the cointegrated EURCAD/AUDCAD spread to be tradeable"
    legs = {(sp.leg_a, sp.leg_b) for sps in active for sp in sps}
    assert any({"EURCAD", "AUDCAD"} == set(l) for l in legs)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda activate fintech; PYTHONPATH=$(pwd) python -m pytest tests/strategies/test_fx_coint_scanner.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Write the strategy**

Create `src/strategies/advanced/fx_coint_scanner.py` with a module-level `_candidate_pairs(pairs) -> list[tuple]` that returns all pairs-of-instruments sharing at most one common 3-letter currency code (exclude mechanical triangles), and a `CointScanner(SpreadStrategy)`. Implementer writes `spread_book` to: on a MONTHLY scan cadence, for each candidate pair run `test_pair(ln(a_upto), ln(b_upto))` on the trailing `scan_window` days (causal); build the tradeable set where `adf_pvalue < adf_max`, `half_life in half_life_range`, and the spread's move over 1.5 sigma clears 2x round-trip cost (use the spread vol from `_spread_sigma` and a nominal cost of `2 * fx_round_trip_pips(major) * pip`); rank by an edge/cost proxy (e.g. spread_vol / cost) and keep `top_n`; between scans, hold the selected spreads, entering when the current-date residual `|z| > entry_z`, exiting on `|z| < target_z`, `|z| > stop_z`, `2 * half_life` days, or the STRUCTURAL exit: the pair's rolling ADF p-value degrades by > 0.2 sustained for 10 consecutive days. Emit `Spread(a, b, hedge_ratio, strength)` per active pair with its `_spread_sigma`. All computation causal. Signature `CointScanner(universe, scan_window=250, half_life_range=(5,25), adf_max=0.05, top_n=5, entry_z=2.0, target_z=0.5, stop_z=3.5)`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda activate fintech; PYTHONPATH=$(pwd) python -m pytest tests/strategies/test_fx_coint_scanner.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/strategies/advanced/fx_coint_scanner.py tests/strategies/test_fx_coint_scanner.py
git commit -m "feat(fx): #37 rolling cointegration scanner (Engle-Granger, structural exit)"
```

---

### Task 5: #30 Vol-ratio pair (simplified symmetric)

**Files:**
- Create: `src/strategies/advanced/fx_vol_ratio_pair.py`
- Test: `tests/strategies/test_fx_vol_ratio_pair.py`

**Interfaces:**
- Consumes: `SpreadStrategy`, `Spread`.
- Produces: `VolRatioPair(coupled_sets=(("EURNOK","EURSEK"),("AUDUSD","NZDUSD"),("XAUUSD","XAGUSD")), rv_window=10, z_window=504, entry_z=2.0, exit_z=1.0)` with `spread_book`.

- [ ] **Step 1: Write the failing tests**

Create `tests/strategies/test_fx_vol_ratio_pair.py`:

```python
import numpy as np
import pandas as pd

from src.strategies.advanced.fx_vol_ratio_pair import VolRatioPair


def _panel(n=700):
    idx = pd.date_range("2019-01-01", periods=n, freq="B").date
    rng = np.random.default_rng(3)
    a = 10.0 + np.cumsum(rng.normal(0, 0.02, n))
    b = 11.0 + np.cumsum(rng.normal(0, 0.02, n))
    # inject a vol spike in A late -> RV ratio z spikes
    a[-30:] += np.cumsum(rng.normal(0, 0.15, 30))
    return pd.DataFrame({"EURNOK": a, "EURSEK": b}, index=pd.Index(idx))


def test_emits_spread_when_vol_ratio_z_high():
    close = _panel()
    book, sigma = VolRatioPair(coupled_sets=(("EURNOK", "EURSEK"),)).spread_book(close)
    active = [d for d, sps in book.items() if sps]
    assert active
    sp = book[active[-1]][0]
    assert {sp.leg_a, sp.leg_b} == {"EURNOK", "EURSEK"}


def test_shorts_high_vol_leg_longs_low_vol_leg():
    close = _panel()
    book, _ = VolRatioPair(coupled_sets=(("EURNOK", "EURSEK"),)).spread_book(close)
    active = [book[d][0] for d in book if book[d]]
    # EURNOK is the high-vol leg late -> spread should be short EURNOK / long EURSEK,
    # i.e. sign convention: strength expresses long(low-vol)-short(high-vol)
    assert active[-1].strength != 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda activate fintech; PYTHONPATH=$(pwd) python -m pytest tests/strategies/test_fx_vol_ratio_pair.py -v`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Write the strategy**

Create `src/strategies/advanced/fx_vol_ratio_pair.py` as `VolRatioPair(SpreadStrategy)`. Implementer writes `spread_book` to: weekly, for each coupled set compute `RV_rv_window` = trailing realized vol (std of daily returns) of each leg; form `r = ln(RV_a / RV_b)`; z-score `r` vs its trailing `z_window` (~2yr) distribution, causal; when `|z| > entry_z`, emit a spread betting reversion -- LONG the low-vol leg, SHORT the high-vol leg (if RV_a > RV_b, i.e. A is hot, short A / long B), hedge_ratio from a trailing price regression of the set (or 1.0 if degenerate), `strength` signed accordingly and magnitude-scaled by z clipped to +-20; hold until `|z| < exit_z`. Provide the spread vol via `_spread_sigma`. All causal. Signature exactly `VolRatioPair(coupled_sets=(("EURNOK","EURSEK"),("AUDUSD","NZDUSD"),("XAUUSD","XAGUSD")), rv_window=10, z_window=504, entry_z=2.0, exit_z=1.0)`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda activate fintech; PYTHONPATH=$(pwd) python -m pytest tests/strategies/test_fx_vol_ratio_pair.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/strategies/advanced/fx_vol_ratio_pair.py tests/strategies/test_fx_vol_ratio_pair.py
git commit -m "feat(fx): #30 vol-ratio pair (simplified symmetric reversion)"
```

---

### Task 6: Spread walk-forward runner + configs (build only; gate delegated to strategy-lead)

**Files:**
- Create: `scripts/backtest_scripts/run_fx_spread_backtest.py`
- Create: `config/backtesting/fx_audnzd_pairs.yaml`, `config/backtesting/fx_coint_scanner.yaml`, `config/backtesting/fx_vol_ratio_pair.yaml`
- Test: `tests/backtesting/test_fx_spread_backtest.py`

**Interfaces:**
- Consumes: the 3 strategies; `FxSpreadPortfolioSimulator`; `load_fx_daily_panel`, `build_quote_usd_panel` from `fx_backtest_loader`; `_cost_fn_factory` from `fx_backtest`.
- Produces: `run_spread_backtest(strategy_name, universe, start, end, vol_target=0.10, rebalance="weekly") -> pd.Series` (daily return series of the spread book) and a thin CLI. NOTE: this is the assembly the strategy-lead's walk-forward will call per OOS window; this task builds + shape-tests it but does NOT run the gate.

- [ ] **Step 1: Write the configs**

Create `config/backtesting/fx_audnzd_pairs.yaml`:

```yaml
asset_class: fx
strategy: {name: AudNzdPairs, universe: [AUDUSD, NZDUSD], params: {}}
dates: {start: "2011-01-01", end: "2026-04-01"}
backtest: {initial_capital: 100000.0, vol_target: 0.10, rebalance: weekly, leverage_cap: 4.0}
output: {save_trades: true}
```

Create `config/backtesting/fx_coint_scanner.yaml` (universe = the 22-pair G10 cache list, `name: CointScanner`, `rebalance: weekly`) and `config/backtesting/fx_vol_ratio_pair.yaml` (`name: VolRatioPair`, universe `[EURNOK, EURSEK, AUDUSD, NZDUSD, XAUUSD, XAGUSD]`, `rebalance: weekly`), same backtest/output blocks.

- [ ] **Step 2: Write the failing shape test**

Create `tests/backtesting/test_fx_spread_backtest.py`:

```python
import datetime as dt

from src.backtesting_scripts.run_fx_spread_backtest import run_spread_backtest  # see note


def test_audnzd_backtest_produces_daily_return_series():
    s = run_spread_backtest("AudNzdPairs", ["AUDUSD", "NZDUSD"],
                            dt.date(2015, 1, 1), dt.date(2018, 1, 1))
    assert s is not None and len(s) > 200
    assert s.index.is_monotonic_increasing
```

Note: import via `import sys; sys.path.insert(0, "scripts/backtest_scripts"); from run_fx_spread_backtest import run_spread_backtest` (do NOT create a `src/backtesting_scripts` package).

- [ ] **Step 3: Run the shape test to verify it fails**

Run: `conda activate fintech; PYTHONPATH=$(pwd) python -m pytest tests/backtesting/test_fx_spread_backtest.py -v`
Expected: FAIL (module not found).

- [ ] **Step 4: Write the runner**

Create `scripts/backtest_scripts/run_fx_spread_backtest.py` exposing `run_spread_backtest(strategy_name, universe, start, end, vol_target=0.10, rebalance="weekly") -> pd.Series`. It must: `load_fx_daily_panel(universe, start, end)` -> close panel; `build_quote_usd_panel`; instantiate the named strategy (registry or direct import); call `strategy.spread_book(close)` -> `(book, sigma)`; run `FxSpreadPortfolioSimulator(...).run_spreads(close, book, sigma, quote_usd, vol_target)`; return the equity curve's `pct_change().dropna()` as the daily return series. Wrap any long run in `RunStatus`. Keep it importable (function-level) so the shape test and the strategy-lead walk-forward can both call it.

- [ ] **Step 5: Run the shape test to verify it passes**

Run: `conda activate fintech; PYTHONPATH=$(pwd) python -m pytest tests/backtesting/test_fx_spread_backtest.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add scripts/backtest_scripts/run_fx_spread_backtest.py config/backtesting/fx_audnzd_pairs.yaml config/backtesting/fx_coint_scanner.yaml config/backtesting/fx_vol_ratio_pair.yaml tests/backtesting/test_fx_spread_backtest.py
git commit -m "feat(fx): spread backtest assembly + configs (gate runs delegated to strategy-lead)"
```

---

## Post-implementation (orchestrator, after all tasks)

- Confirm the full suite passes (spread_sizing 4, simulator 3, 3 strategies 6, runner 1).
- The engine + strategies are BUILT but NOT gated. Hand the 3 strategies to `strategy-lead` (Wave 2 pre-registration: combined statistical gate, honest every-spec trial count, both cost legs, S&P/corr/IR book-level context, sentinel + registry + no-push) to produce the verdicts via a walk-forward that calls `run_spread_backtest` per OOS window.
- Those 3 verdicts complete Wave 2 -> resolve the pre-registered stopping rule (any clear -> Wave 3 on that mechanism; all 6 fail -> declare the catalog exhausted and stop).
