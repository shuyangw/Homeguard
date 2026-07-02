# Futures Carry Strategy - Design (Option C)

**Date:** 2026-07-02 · **Status:** approved, pre-plan · **Depends on:** pluggable runner (`main` @ eec3ea9), `CarryCalculator`, the roll/per-contract data layer.

## Goal

Add an absolute (time-series) carry strategy that plugs into the pluggable futures
runner via `forecast_panel`, and produce a gate-checked walk-forward result on the
broad basket. Carry is the least-correlated diversifier to the WEAK Carver momentum
result; if it clears the gate it is a deploy candidate and a future combine-with-momentum
building block.

## Context (verified)

- `src/data/carry_calculator.py::CarryCalculator` already computes annualized carry:
  `compute(root, asset_class, d) -> float` and
  `compute_history(root, asset_class, start, end) -> polars DataFrame[date, carry]`
  (front/second by volume ranking from `per_contract_1min`, outrights only, spreads
  filtered). Per-class conventions: commodity/fx `(second-front)/front * 365/days`,
  equity_index `(front-second)/second * 365/days`, bond `duration*(yield-funding)/100`
  (SOFR funding via `derive_sofr`; some bonds fall back to 0 in v1). It takes
  `asset_class` as a caller-supplied argument.
- `ContractSpec` does NOT carry `asset_class`; no root->asset_class map exists anywhere.
  C is the first production consumer of `CarryCalculator`.
- The pluggable runner (B) resolves `strategy.name`/`params` and validates
  `forecast_panel(close) -> DataFrame`. `run_carver_walkforward.py::_run_window` still
  reconstructs a Carver-only config (universe only) -- generalizing that is the piece
  deferred from B to C.
- `src/data/futures/paths.py` exposes `roll_calendar_dir()` (= `_futures_root()/"roll_calendar"`);
  C mirrors it with `carry_dir()` (= `_futures_root()/"carry"`).

## Architecture

Carver-style ABSOLUTE carry. Per instrument, per day:
`forecast = EWMA_span10(raw_carry) / annualized_price_vol * carry_scalar`, capped +/-20.
This is a per-instrument forecast in the same +/-cap convention Carver produces, so it
flows through the existing `forecast_panel -> run_sized` vol-target sizing unchanged and
is directly comparable to (and later combinable with) Carver momentum. Carry is
self-sourced from `CarryCalculator` via a precomputed per-root cache.

`annualized_price_vol = daily_price_vol * sqrt(252)`, with `daily_price_vol` computed from
`close` exactly as the harness sizing does (`close_to_close_rv(returns, 25,
annualization_factor=1)`). Since raw carry is already annualized, `carry / annualized_vol`
is a risk-adjusted (Sharpe-like) carry; `carry_scalar` (Carver doctrine ~30) scales the
average |forecast| toward ~10 before the +/-20 cap.

**Parameter-free discipline:** `carry_scalar` (~30) and the EWMA span (~10 days) are FIXED
doctrine constants, NOT fit to our data. Carry therefore stays a single non-selected
configuration -> DSR trial_count = 1 (as with Carver). They must never be optimized.

## Components

1. **`carry_dir()` in `src/data/futures/paths.py`** -- `_futures_root()/"carry"`, mirroring
   `roll_calendar_dir()`.

2. **Root->asset_class map** (`src/strategies/advanced/futures_carry_strategy.py`,
   strategy-local): the ~33 broad-basket roots (and their micros where relevant) mapped to
   {`equity_index`, `fx`, `bond`, `commodity`}. equity_index: ES/NQ/YM/RTY/M2K/MES/MNQ/MYM;
   fx: 6A/6B/6C/6E/6J/6M/6N/6S; bond: ZT/ZF/ZN/TN/ZB/UB/10Y/2YY/5YY/30Y/SR1/SR3;
   commodity: CL/BZ/NG/HO/RB/MCL/MNG/GC/SI/HG/PL/MGC/SIL/MET/ZC/ZW/ZS/ZL/ZM/KE/LE/HE. A
   `KeyError`-raising accessor so an unmapped root fails loud. (Future: consolidate into
   `ContractSpec` -- out of scope.)

3. **Carry cache builder** (`scripts/data/build_carry_cache.py`): for each root, call
   `CarryCalculator().compute_history(root, asset_class, start, end)` and write
   `carry_dir()/{root}.parquet` `[date, carry]`. Mirrors `build_roll_calendar.py`. Rationale:
   the walk-forward runs ~14 overlapping windows; without a cache each recomputes the same
   per-day per-contract volume rankings, making runtime intractable. The cache makes it
   tractable and reproducible.

4. **`FuturesCarryStrategy`** (`src/strategies/advanced/futures_carry_strategy.py`):
   `__init__(self, universe, carry_scalar=30.0, ewma_span=10, cap=20.0, **params)`;
   `forecast_panel(close_panel) -> DataFrame`. For each root: read cached carry
   (`carry_dir()/{root}.parquet`), convert to a pandas Series, reindex to `close.index`
   (forward-fill small gaps), EWMA-smooth (span), divide by `annualized_price_vol` from
   `close`, multiply by `carry_scalar`, clip to +/-cap. Missing cache for a root -> that
   root's forecast column is all-NaN (harness sizes NaN -> 0 contracts, consistent with
   Carver's missing-data handling). Returns forecasts indexed by `close.index`, columns =
   universe.

5. **Register `"FuturesCarry"`** in `src/strategies/registry.py` (+ aliases `"Carry"`,
   `"Futures Carry"`).

6. **Walk-forward generalization** (`scripts/backtest_scripts/run_carver_walkforward.py`):
   thread `strategy_name` (default `"CarverMomentum"`) and `strategy_params` through
   `_config_to_kwargs` -> `walk_forward_carver` -> `_run_window` (which currently hardcodes
   only `strategy.universe`), and parametrize the readiness-report title/label by the
   strategy name. No-`name`/no-config default behavior stays byte-equivalent (Carver).

7. **`config/backtesting/carry_broad.yaml`** -- `name: FuturesCarry`, the 33-root broad
   basket, $10M, weekly, 2010-06-07..2026-02-20.

## Data Flow

`build_carry_cache` (one-time) -> `carry_dir()/{root}.parquet`; then
`FuturesCarryStrategy.forecast_panel(close)` reads cache + close -> capped forecast panel
-> `run_sized`. Walk-forward (`--config carry_broad.yaml`) ->
`docs/reports/futures/CARRY_BROAD_READINESS.md`.

## Success Criteria

- A trustworthy OOS Sharpe for carry on the broad basket with clean tail stats.
- Clears the combined gate (PSR/DSR/PBO, 1.5x cost) -> viable momentum-uncorrelated
  strategy (deploy candidate; future combine-with-Carver). Still WEAK -> documented; the
  two canonical Carver signals are then fairly tested.

## Error Handling

- Unmapped root in the asset_class map -> `KeyError`/`ValueError` (fail loud).
- Missing per-contract store when building the cache -> `CarryCalculator` already raises
  `FileNotFoundError`.
- Missing carry cache for a root at strategy time -> NaN forecast column (sized to 0), logged.

## Testing

- asset_class map covers every root in the broad-basket universe (no `KeyError`).
- `FuturesCarryStrategy.forecast_panel` with a MOCKED constant carry gives the deterministic
  forecast `clip(carry/annvol*scalar, -cap, cap)`; shape + index match `close`; all values
  within +/-cap.
- Missing-cache root -> all-NaN column (no crash).
- Registry resolves `"FuturesCarry"` and an alias.
- End-to-end `run_futures_backtest` with `name: FuturesCarry` on a SMALL cached slice runs
  and returns a finite equity curve.
- Walk-forward: `_run_window` passes `strategy.name`/`params` (a fast unit assertion on the
  built config dict) and the report title reflects the strategy; Carver default unchanged.

## Scope / Caveats (documented, non-blocking)

- No IDM/FDM (same as Carver).
- Bond carry uses `CarryCalculator`'s SOFR-funding path; some bonds fall back to 0 (its v1).
- Absolute carry only (cross-sectional deferred).
- asset_class map is strategy-local (future: consolidate into `ContractSpec`).
- Carry cache is a point-in-time snapshot of `compute_history`; rebuild when the
  per-contract store changes.
