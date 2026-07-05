# B4: Crypto Satellite Blend (core-satellite, 15% crypto risk weight) - Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Steps use `- [ ]`.

**Goal:** Harvest crypto's rho=-0.065 diversification WITHOUT the naive IDM over-allocation, by
blending the IDM-carry book (core, 85% risk) and the crypto-carry book (satellite, 15% risk) at
the RETURN-STREAM level, and measuring the blended walk-forward Sharpe + PBO vs the 0.76 incumbent.

**Architecture:** Both books run the SAME walk-forward window schedule (start 2010-06-07; crypto
contributes 0 in pre-2018 windows where it has no data). We expose per-window DATED OOS returns
from `walk_forward_carver`, then blend per-window by date: `combined = w_c*(carry_ret/sig_c) +
w_k*(crypto_ret/sig_k)` with `w_c=0.85, w_k=0.15` (PRE-REGISTERED) and `sig_c, sig_k` = each
book's full-sample OOS daily vol (disclosed normalization; causal-vol is a later refinement).
Gate (Sharpe/PSR/DSR/PBO) computed on the blended per-window returns via the existing functions.

## Global Constraints
- Python tests: `/c/Users/qwqw1/anaconda3/envs/fintech/python.exe -m pytest <files> -q`. `PYTHONPATH=.` scripts.
- ASCII only, no `print()`. Branch `feat/crypto-satellite-blend` -- do NOT switch.
- **crypto weight 0.15 is PRE-REGISTERED** (chosen before results). No weight sweeping this task.
- 8-thread cap on runs (A1 cache makes this fast/light). Runs are CONTROLLER-run, on explicit go.
- Do NOT change existing walk_forward_carver behavior when the new opt-in flag is off (back-compat).
- Both books already exist: carry `carry_idm_broad.yaml` (33-root, idm), crypto `crypto_carry_broad.yaml` (BTC/ETH).

## Files
- Modify: `scripts/backtest_scripts/run_carver_walkforward.py` (opt-in dated per-window returns)
- Create: `src/backtesting/blend/satellite_blend.py` (blend + gate)
- Create: `scripts/backtest_scripts/run_satellite_blend.py` (runner)
- Tests: `tests/backtesting/blend/test_satellite_blend.py`, `tests/backtest_scripts/test_walkforward_dated_returns.py`

---

## Task 1: expose per-window DATED OOS returns from the walk-forward

**Files:** Modify `scripts/backtest_scripts/run_carver_walkforward.py`; Test `tests/backtest_scripts/test_walkforward_dated_returns.py`

Currently `process_window` returns `oos_1x` as a bare `np.ndarray` (dates dropped by `_oos_returns`).
`walk_forward_carver` stitches these but does not return them. Add an opt-in that carries dates.

- [ ] Step 1: failing test -- monkeypatch a tiny 2-window run (or a unit test on a helper): assert that
  with `return_window_returns=True`, `walk_forward_carver(...)` result includes key `per_window_oos`:
  a list of `pd.Series` (one per used window), each indexed by OOS date, and that concatenating them
  reproduces the existing `stitched_1x` values (same numbers, now dated). Keep it lightweight (can test
  the new `_oos_returns_dated` helper directly: given an equity list + dates + test_start, returns a
  dated `pd.Series` whose `.to_numpy()` equals the existing `_oos_returns` output).
- [ ] Step 2: run -> FAIL.
- [ ] Step 3: add `_oos_returns_dated(equity_curve, dates, test_start) -> pd.Series` (same slice logic as
  `_oos_returns` but returns the dated Series, not `.to_numpy()`). In `process_window`, when
  `spec.get("return_dated")` is set, also include `"oos_1x_dated"` (the Series). In `walk_forward_carver`,
  add param `return_window_returns: bool = False`; thread `return_dated` into specs; when true, collect the
  per-window dated Series into `result["per_window_oos"]`. Default False -> result unchanged (back-compat).
- [ ] Step 4: run -> PASS. Also run `tests/backtest_scripts/test_walkforward_idm_threading.py` (no regression).
- [ ] Step 5: commit `feat(futures): opt-in dated per-window OOS returns from walk_forward_carver`.

---

## Task 2: satellite blend + gate

**Files:** Create `src/backtesting/blend/satellite_blend.py`; Test `tests/backtesting/blend/test_satellite_blend.py`

**Interface:** `blend_books(core_windows: list[pd.Series], sat_windows: list[pd.Series],
sat_weight: float, core_vol: float | None = None, sat_vol: float | None = None) -> dict` returns a
gate dict `{oos_sharpe, pbo, psr, dsr, n_windows, n_oos_days, skew, kurtosis_pearson,
core_vol, sat_vol, sat_weight}`.

Logic: (1) if vols not given, compute each book's full-sample OOS daily vol from the concatenation of
its window returns (std, ddof=1). (2) Per window, align core and satellite dated Series on the union of
dates; satellite missing dates -> 0 contribution (pre-2018). blended = `(1-sat_weight)*(core/core_vol) +
sat_weight*(sat/sat_vol)`. (3) Concatenate blended windows -> stitched; compute `_annualized_sharpe`,
skew, Pearson kurt, `psr`, `dsr` (n_trials=1), `_compute_pbo(blended_per_window)` -- REUSE the functions
from `run_carver_walkforward` (import them). Weights sum to 1 by construction.

- [ ] Step 1: failing test:
  - `test_blend_reduces_to_core_at_zero_weight`: with `sat_weight=0.0`, blended gate `oos_sharpe` equals the
    core-only stitched Sharpe (to the float).
  - `test_blend_math_two_windows`: two hand-built windows for core and satellite (known values), sat_weight
    0.15, assert the blended stitched returns equal the hand-computed `0.85*core/cv + 0.15*sat/sv` and the
    Sharpe matches a hand calc.
  - `test_satellite_missing_dates_contribute_zero`: a window where the satellite Series covers only half the
    core dates -> the other half is pure scaled core (sat term 0), asserted.
- [ ] Step 2: run -> FAIL. Step 3: implement. Step 4: run -> PASS.
- [ ] Step 5: commit `feat(futures): satellite_blend (core-satellite return-stream blend + gate)`.

---

## Task 3: runner + run the 15% blend (CONTROLLER-RUN)

**Files:** Create `scripts/backtest_scripts/run_satellite_blend.py`

- [ ] Step 1: implement runner: `walk_forward_carver(carry 33-root config, return_window_returns=True)` and
  `walk_forward_carver(crypto BTC/ETH config, same start/train/test schedule, return_window_returns=True)`;
  call `blend_books(core.per_window_oos, sat.per_window_oos, sat_weight=0.15)`; log/emit the blended gate
  vs the core-only gate. `--sat-weight` arg default 0.15; `--core-config`/`--sat-config`. Include a
  `if __name__ == "__main__":` guard (workers re-import on Windows spawn).
- [ ] Step 2 (gated on user go): run at 8-thread cap:
  `... run_satellite_blend.py --core-config config/backtesting/carry_idm_broad.yaml
   --sat-config config/backtesting/crypto_carry_broad.yaml --sat-weight 0.15 --jobs 8`.
- [ ] Step 3: record blended Sharpe/PBO/skew/kurt vs incumbent 0.76/0.19 in the results log + trial ledger.
  Verdict: does the 15% blend clear the gate AND beat 0.76? Report honestly (estimate was ~0.86; real may differ).

---

## Self-Review
- Coverage: dated per-window returns (T1), blend+gate (T2), runner+run (T3). Weighting pre-registered 0.15.
- Placeholders: none. Types: `_oos_returns_dated(...)->pd.Series`; `blend_books(...)->dict`.
- Back-compat: `return_window_returns` defaults False -> walk_forward_carver unchanged; blend reuses existing
  gate functions (no gate-math change). Vol normalization = full-sample (disclosed; causal-vol is a refinement).
