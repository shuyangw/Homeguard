# Carry De-Concentration Stack (XS carry + IDM) - Design

**Date:** 2026-07-03 · **Status:** approved, pre-plan · **Depends on:** corrected carry baseline (`main` @ 0640a5f: OOS Sharpe 0.85, PBO 0.33, kurt 21.0).

## Goal

Push corrected carry under the PBO 0.25 gate via two orthogonal, parameter-free
de-concentration levers, evaluated as 3 pre-committed trials. Detailed design in the
research briefs `docs/strategies/research/20260703_EXPAND_item2_xs_carry.md` (XS) and
`20260703_EXPAND_item1_carry_idm.md` (IDM); this spec is the build contract.

## Context (verified integration points)

- `FuturesCarryStrategy.forecast_panel` (`src/strategies/advanced/futures_carry_strategy.py`)
  builds `out[root]` per-root then `return pd.DataFrame(out).reindex(columns=self.universe)`.
- Sizing hooks ALREADY thread `div_mult`: `size_from_forecast(..., div_mult=1.0)`
  (`position_sizer_futures.py:53,66`) and `run_sized(..., div_mult=1.0)`
  (`futures_portfolio_simulator.py:142`) which passes `div_mult=div_mult` per root at the
  call site (`:156`). But `run_futures_backtest` calls
  `sim.run_sized(close, forecasts, daily_vol, vol_target)` (`futures_backtest.py:92`) with
  NO `div_mult` -> every instrument targets a FULL vol budget, no correlation term, clusters
  over-risked (the source of the kurtosis).
- Root->asset_class map: `src/data/futures/asset_class.py` (equity_index/fx/bond/commodity).

## Architecture -- two independent, stackable levers

### Lever 1: XS carry (signal-side)
`FuturesCarryXSStrategy(FuturesCarryStrategy)` overrides `forecast_panel`: build the
per-root risk-adjusted carry forecasts (reuse the parent), then per day **demean within
asset-class**, renormalize by the within-class cross-sectional dispersion, clip +/-20.
Removes the common directional carry bet. **demean, not rank; within-class, not global.**
Register as `"FuturesCarryXS"`. No sizing/runner change.

### Lever 2: IDM (sizing-side, strategy-agnostic)
A per-root `div_mult` vector replaces the implicit global `1.0`:
`div_mult_i = w_i * IDM * N_scale`, where
- **`w_i`** = handcrafted cluster risk weights: equal risk across **7 clusters**
  (equity / rates / FX / energy / metals / grains / meats -- energy split from commodity),
  then equal within cluster; `sum_i w_i = 1`. No complex exceeds its 1/7 cluster share
  regardless of how many correlated roots it holds (the cluster cap).
- **`IDM`** = `min(1 / sqrt(w' C w), 2.5)`, with `C` a **fixed, handcrafted correlation
  matrix** using two doctrine constants: **intra-cluster rho = 0.5, inter-cluster rho = 0.0**
  (diagonal 1.0). Data-free: no estimation, no lookahead -> parameter-free, `trial_count`
  unaffected. (Empirical expanding-window `C`, and any tuning of these rho values, is
  deferred as a separate, explicitly-logged future trial -- they are NOT swept here.)
- **`N_scale`** = a single scalar constant chosen once (not fit) so the IDM-weighted book
  targets a comparable overall risk to the scalar-`div_mult=1` baseline rather than being
  silently de-risked toward zero (a `w_i`-weighted book has each `w_i << 1`). Verified by a
  test that the simulated book's realized vol stays in a sane band. Concentrated clusters'
  per-root allocations are cut hardest -- the mechanism that compresses skew/kurt.

## Components

1. **7-cluster map** in `src/data/futures/asset_class.py`: add `CLUSTER: dict[str,str]` +
   `cluster_for(root)` mapping the 33 roots to the 7 economic complexes (equity_index->equity,
   bond->rates, fx->fx, commodity split into energy/metals/grains/meats). KeyError on unmapped.
2. **`FuturesCarryXSStrategy`** (subclass in `futures_carry_strategy.py`) + registry entry
   `"FuturesCarryXS"` (+ alias `"XS Carry"`).
3. **`src/backtesting/utils/idm_weights.py`**: `compute_div_mult(universe) -> dict[str,float]`
   -- the doctrine `w_i`, fixed-`C` `IDM`, and `N_scale`. Pure, deterministic, data-free.
4. **`div_mult` scalar -> Series/dict**: widen `run_sized` to accept `float | dict[str,float]`
   (scalar preserved for back-compat; per-root looked up at the `:156` call site). Add a
   `backtest.idm: bool` config flag; when true, `run_futures_backtest` computes
   `compute_div_mult(universe)` and passes it to `run_sized`.
5. **3 configs** (each a pre-committed trial): `config/backtesting/carry_xs_broad.yaml`
   (name FuturesCarryXS, idm off), `carry_idm_broad.yaml` (FuturesCarry, idm on),
   `carry_xs_idm_broad.yaml` (FuturesCarryXS, idm on) -- all 33-root, $10M, else identical
   to `carry_broad.yaml`.

## The 3 pre-committed trials

| Trial | strategy `name` | `backtest.idm` |
|---|---|---|
| XS-alone | FuturesCarryXS | false |
| IDM-alone | FuturesCarry | true |
| XS+IDM | FuturesCarryXS | true |

Each walk-forward appends to `output/experiments.duckdb` (project-wide trial count += 3).

## Data Flow

`config -> run_futures_backtest`: resolve strategy (XS or not) -> `forecast_panel`; if
`backtest.idm`, `compute_div_mult(universe)` -> `run_sized(..., div_mult=<dict>)` sizes each
root by `w_i*IDM*N_scale`. Walk-forward -> per-trial readiness report.

## Testing

- XS: within-class demean gives ~0 within-class mean per day; forecasts stay in +/-20;
  missing-cache root -> NaN column (parent behavior preserved).
- Cluster map covers all 33 roots.
- IDM weights: `sum w_i = 1`; each cluster's summed weight = 1/7 (cluster cap binds);
  `IDM <= 2.5`; deterministic (same input -> same output, no randomness).
- `div_mult` dict flows to sizing: a concentrated cluster's per-root contracts SHRINK vs
  the scalar-1.0 baseline (known-value); scalar `div_mult` path unchanged (back-compat).
- Book-risk normalization: with IDM on, the simulated book's realized vol is in a sane band
  (not de-risked to ~0), confirming `N_scale`.
- No change to gate math / walk-forward / equity path.

## Execution (controller-run, 8-thread capped)

Run the 3 walk-forwards with the **8-thread cap** (`POLARS_MAX_THREADS=1 OMP_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONPATH=. python
scripts/backtest_scripts/run_carver_walkforward.py --config <trial>.yaml --report <...> --jobs 8`),
each RunStatus-tracked. Compare PBO/kurt/Sharpe to the 0.33 baseline; record which (if any)
clears PBO < 0.25 at Sharpe meaningfully > 0.

## Success / Scope

- **Success:** any trial with PBO < 0.25 and Sharpe clearly > 0 -> carry's first gate-pass ->
  deploy candidate. All still WEAK -> concentration is intrinsic; documented; move to W3.
- **Parameter-free:** cluster map, `w_i`, fixed `C`, cap 2.5, EWMA/scalar are fixed doctrine.
  No sweeping. Exactly 3 pre-committed trials.
- **Out of scope:** empirical-estimated `C` (future logged trial); carry+trend combine (item 3,
  separate); any change to the gate, walk-forward structure, or equity/crypto path.
