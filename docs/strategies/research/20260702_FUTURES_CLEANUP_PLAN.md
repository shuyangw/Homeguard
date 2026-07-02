# Futures Harness Post-Merge Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the two non-blocking accuracy items the final whole-branch review flagged on the merged futures harness: the `pct_change` FutureWarning and the stale/self-contradictory skew-kurtosis prose in the Carver readiness report.

**Architecture:** Two independent fixes. (1) Pass `fill_method=None` to the three `pct_change()` call sites — silences the pandas FutureWarning AND is more correct (no forward-filling returns across gaps). (2) Rewrite the readiness report's "Concern" section to reflect the RESOLVED state after the equity-feedback + bankruptcy-floor fix.

**Tech Stack:** Python 3.13, pandas, pytest. Conda env `fintech`.

## Global Constraints

- **Python execution:** ALWAYS `/c/Users/qwqw1/anaconda3/envs/fintech/python.exe -m pytest <args>`. Never system Python.
- **ASCII only** in all code/docs (Windows cp1252).
- **Base:** branch off `main` (currently `0e8f222`) — do NOT commit directly to main.
- **Context:** the harness is already merged. The earlier statistics contamination (skew -30.5, kurtosis 1332, 1.5x-cost Sharpe > 1x) was caused by the simulator letting equity go negative and `pct_change` exploding on a zero-crossing equity curve. It was FIXED by equity-feedback sizing + a bankruptcy floor; the regenerated readiness report shows clean stats (skew -0.39, kurtosis 8.7, 1.5x-cost Sharpe 0.0798 correctly below the 1x 0.1088). The Carver verdict is WEAK (does not clear the gate), which is a valid finding.

---

## Task 1: Silence the pct_change FutureWarning (and be gap-correct)

**Files:**
- Modify: `src/backtesting/data/futures_backtest_loader.py:45`
- Modify: `src/strategies/advanced/carver_momentum_strategy.py:34`
- Modify: `src/backtesting/engine/futures_backtest.py:60`
- Test: `tests/strategies/test_carver_momentum_strategy.py` (append)

**Interfaces:**
- No signature changes. `close.pct_change()` -> `close.pct_change(fill_method=None)` at all three sites.

**Context:** pandas emits `FutureWarning: The default fill_method='pad' in ... pct_change is deprecated`. Passing `fill_method=None` silences it and is more correct — it does NOT forward-fill a stale price before differencing, so a gap yields NaN (correctly excluded) rather than a fake 0% return.

- [ ] **Step 1: Write the failing test (append)**

```python
# append to tests/strategies/test_carver_momentum_strategy.py
def test_forecast_panel_no_future_warning():
    import warnings
    import numpy as np
    import pandas as pd
    from src.strategies.advanced.carver_momentum_strategy import CarverMomentumStrategy
    dates = pd.date_range("2020-01-01", periods=400, freq="B")
    close = pd.DataFrame({
        "MES": np.linspace(3000, 4000, 400),
        "MGC": np.linspace(1800, 1700, 400),
    }, index=dates)
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)  # any FutureWarning becomes an error
        CarverMomentumStrategy(universe=["MES", "MGC"]).forecast_panel(close)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/c/Users/qwqw1/anaconda3/envs/fintech/python.exe -m pytest tests/strategies/test_carver_momentum_strategy.py::test_forecast_panel_no_future_warning -v`
Expected: FAIL — a `FutureWarning` about `pct_change` default `fill_method='pad'` is raised (promoted to error).

- [ ] **Step 3: Apply `fill_method=None` at all three sites**

In `src/strategies/advanced/carver_momentum_strategy.py` line 34:
```python
            rets = close.pct_change(fill_method=None)
```
In `src/backtesting/data/futures_backtest_loader.py` line 45:
```python
    ret = close.pct_change(fill_method=None)
```
In `src/backtesting/engine/futures_backtest.py` line 60:
```python
    returns = close.pct_change(fill_method=None)
```
(Match on the code content — line numbers may drift by a line or two.)

- [ ] **Step 4: Run test to verify it passes**

Run: `/c/Users/qwqw1/anaconda3/envs/fintech/python.exe -m pytest tests/strategies/test_carver_momentum_strategy.py -v`
Expected: PASS (the new test + the existing forecast tests).

- [ ] **Step 5: Confirm the real-data e2e still passes (the other two sites)**

Run: `/c/Users/qwqw1/anaconda3/envs/fintech/python.exe -m pytest tests/backtesting/engine/test_futures_backtest_e2e.py tests/backtesting/data/test_futures_backtest_loader.py -v`
Expected: PASS. Report the e2e Sharpe — it should be essentially unchanged from -0.399 (a clean daily continuous series has no gaps to forward-fill, so `fill_method=None` is numerically identical here; a tiny change is acceptable, a large one means investigate).

- [ ] **Step 6: Commit**

```bash
git add src/backtesting/data/futures_backtest_loader.py src/strategies/advanced/carver_momentum_strategy.py src/backtesting/engine/futures_backtest.py tests/strategies/test_carver_momentum_strategy.py
git commit -m "fix(futures): pct_change(fill_method=None) - silence FutureWarning, gap-correct returns"
```

---

## Task 2: Rewrite the stale skew/kurtosis "Concern" section in the readiness report

**Files:**
- Modify: `docs/reports/futures/CARVER_TSMOM_READINESS.md:94-114`

**Interfaces:** Doc-only. No code, no test.

**Context:** The `## Concern: extreme skew/kurtosis` section (lines 94-114) was written for the ORIGINAL contaminated run (skew -30.5, kurtosis 1332) but the report was regenerated with the FIXED numbers substituted into the same alarmist template. It now reads self-contradictorily: it states "skew -0.4 and Pearson kurtosis 8.7" then claims "Both are far outside what is plausible ... (a well-behaved vol-targeted series would typically show single- to low-double-digit kurtosis, not four digits)" — but 8.7 IS single-to-low-double-digit and there are no four-digit values anymore. The anomaly was root-caused and FIXED (negative equity + `pct_change` explosion -> equity-feedback sizing + bankruptcy floor). The section must reflect resolution, not a live concern.

- [ ] **Step 1: Read the current section**

Run: `sed -n '94,114p' docs/reports/futures/CARVER_TSMOM_READINESS.md`
Expected: the stale "extreme skew/kurtosis" prose.

- [ ] **Step 2: Replace lines 94-114 with the resolved-state version**

Replace the entire `## Concern: extreme skew/kurtosis in the stitched OOS series` section (from the `## Concern` heading through the end of that section, ~line 94 to ~114) with:

```markdown
## Note: tail statistics (resolved)

An earlier version of this run showed extreme tail statistics (skew -30.5,
Pearson kurtosis ~1332, and a 1.5x-cost OOS Sharpe *above* the 1x Sharpe --
physically backwards). Root cause: the `FuturesPortfolioSimulator` allowed
account equity to cross zero (no bankruptcy floor), and OOS returns were
computed via `pct_change` on a zero-crossing equity curve, which explodes near
the crossing.

This was fixed before merge: the simulator now (a) sizes each rebalance against
LIVE equity (equity-feedback sizing), so position sizes shrink in a drawdown,
and (b) floors equity at zero after both mark-to-market and cost debits
(bankruptcy floor), guaranteeing a non-negative equity curve. After the fix,
the regenerated stitched OOS series is well-behaved: skew -0.39, Pearson
kurtosis 8.7 (mild for a daily futures book), and the 1.5x-cost Sharpe (0.0798)
is correctly below the 1x Sharpe (0.1088). PSR/DSR are therefore reliable here.

The WEAK verdict (OOS Sharpe 0.1088; PBO 0.438 -- near-coin-flip) stands on the
clean statistics: Carver multi-speed TSMOM on this basket does not clear the
combined gate. Follow-up (fidelity, not correctness): the combined forecast
omits Carver's Forecast Diversification Multiplier, so forecasts are
systematically under-scaled -- worth adding before a fair head-to-head, though
it will not flip the WEAK verdict.
```

Keep the surrounding sections (the metrics table, per-window table, verdict, PBO-interpretation note) unchanged. Ensure ASCII-only.

- [ ] **Step 3: Verify the report no longer contradicts itself**

Run: `grep -n -iE "four digit|far outside|1332|30.5|extreme skew" docs/reports/futures/CARVER_TSMOM_READINESS.md`
Expected: no matches in a live-concern framing — the only mentions of the old -30.5/1332 values are in the past-tense "an earlier version ... was fixed" note.

- [ ] **Step 4: Commit**

```bash
git add docs/reports/futures/CARVER_TSMOM_READINESS.md
git commit -m "docs(futures): correct readiness report tail-stats note (anomaly resolved by equity-feedback fix)"
```

---

## Self-Review

- **Spec coverage:** both flagged items covered — Task 1 (pct_change FutureWarning, 3 sites), Task 2 (stale readiness prose). No other cleanup requested.
- **Placeholder scan:** none — Task 1 has the exact one-line edits + a real test; Task 2 has the full replacement prose.
- **Type consistency:** no new symbols introduced; `fill_method=None` is a pandas kwarg. The Task-2 replacement references only values already in the report (0.1088, 0.0798, 0.438, -0.39, 8.7) and the real fix (equity-feedback + floor).
- **Note:** Task 1 Step 5 flags that `fill_method=None` is numerically identical on a gap-free daily series, so the e2e Sharpe should be unchanged; a material change is a signal to investigate (not expected).
