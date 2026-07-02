# Broad-Basket Carver TSMOM Walk-Forward - Design (Option A)

**Date:** 2026-07-02 · **Status:** approved, pre-plan · **Depends on:** merged futures harness (`main` @ a855ae2)

## Goal

Give Carver multi-speed TSMOM a fair walk-forward test on a properly diversified,
institutional-scale universe, replacing the earlier 3-instrument (MES/MGC/6E) WEAK
result with a trustworthy, gate-checked number on ~33 full-size markets. Carver's
edge is diversification across many low-correlation markets; the prior test could
not express that. This test decides whether naive Carver is viable or whether we
move on to the B/C follow-ons.

## Context

- Harness is merged and isolated (dedicated futures path; equity-feedback sizing +
  bankruptcy floor guarantee clean stats). Single-backtest path is config-driven via
  `python -m src.backtest_runner --config <yaml>` (routes `asset_class: futures` to
  `src.backtesting.engine.futures_backtest.run_futures_backtest`).
- Walk-forward + statistical gate lives in
  `scripts/backtest_scripts/run_carver_walkforward.py`
  (`walk_forward_carver(train_months, test_months, step_months, start, end)` + a
  `main()` that currently HARDCODES its basket).
- Data coverage (from roll-calendar parquets): all full-size liquid roots start
  2010-06-07 and run to 2026-02-20 (~3990 daily bars, ~15.7y). Micros start 2019
  (rejected: too short/narrow). TN phases in 2010-10-26.

## Universe (~33 full-size roots, 2010-06-07 start)

| Class | Roots |
|---|---|
| Equity | ES, NQ, YM |
| Rates | ZT, ZF, ZN, TN, ZB, UB |
| FX | 6E, 6J, 6B, 6A, 6C, 6S, 6M, 6N |
| Energy | CL, BZ, NG, HO, RB |
| Metals | GC, SI, HG, PL |
| Grains | ZC, ZW, ZS, ZL, ZM |
| Meats | LE, HE |

Total 33. The harness's per-window data-availability filter handles late starters
(e.g. TN) by skipping roots without data for a given window; no special handling
needed here.

## Parameters

| Param | Value | Rationale |
|---|---|---|
| initial_capital | 10_000_000 | Integer-contract rounding rarely binds at ~$300k/instrument (capital/N); non-zero positions across the basket including ES/NQ/ZB. |
| vol_target_per_instrument | 0.20 | Carver default; matches existing config. Sharpe is scale-invariant so exact value is not load-bearing. |
| rebalance | weekly | Cost control; matches existing. |
| cost_mult | 1.0 | Base; the walk-forward also reports the 1.5x cost-sensitivity Sharpe (methodology Section 4 gate). |
| start / end | 2010-06-07 / 2026-02-20 | Full available full-size history. |
| walk-forward | train 36m, test 12m, step 12m | Existing script defaults; ~14 non-overlapping OOS windows from ~2013.5. Parameter-free strategy -> trial_count = 1 (DSR reduces to PSR). |

## Components

1. **`config/backtesting/carver_tsmom_broad.yaml`** (new) - `asset_class: futures`, the
   33-root universe, $10M, 0.20 vol-target, weekly, cost 1.0, dates above. Runnable
   immediately via `python -m src.backtest_runner --config ...` for a single-pass
   backtest + standard report.

2. **Parametrize the walk-forward universe** - `run_carver_walkforward.py` must run the
   broad basket through the PSR/DSR/PBO gate. Add an optional `--config <yaml>` CLI arg;
   when supplied, the script reads `strategy.universe`, `dates.start/end`, and the
   `backtest.*` params from that YAML (the SAME file the single-backtest path consumes),
   making the config the single source of truth. When omitted, preserve the existing
   hardcoded default behavior exactly. Script-level change only; does NOT touch harness
   core (keeps A isolated from B, the pluggable runner).

3. **Broad-basket readiness report** - to preserve the 3-instrument baseline
   (`docs/reports/futures/CARVER_TSMOM_READINESS.md`), the broad-basket walk-forward
   writes a SIBLING file `docs/reports/futures/CARVER_TSMOM_BROAD_READINESS.md` with full
   gate metrics + per-window OOS Sharpe + roots-with-data-per-window column. The output
   path is derived from the config (or a CLI `--report` arg) so the baseline is never
   clobbered.

## Success Criteria

- A trustworthy OOS Sharpe on ~33 markets with clean tail statistics (skew/kurtosis
  sane, 1.5x-cost Sharpe below 1x - the fix already guarantees this).
- Gate decision per methodology Section 2: clears combined gate (PSR/DSR/PBO) ->
  Carver is viable, candidate for paper deployment. Still WEAK -> naive Carver fairly
  exhausted; proceed to B/C.
- Report is internally consistent (no stale prose), per-window table present.

## Known Caveats (documented, non-blocking)

- **No IDM** (Instrument Diversification Multiplier): rates (6) + FX (8) clusters are
  over-weighted by raw count in the equal-forecast-average; portfolio leans into those
  classes. Motivates adding IDM/FDM before a fair cross-strategy comparison (B).
- **No FDM** (Forecast Diversification Multiplier): combined forecast mildly
  under-scaled; does NOT move Sharpe (vol-targeting normalizes uniform scaling), but
  can push low-forecast positions to round to zero.
- **Integer-contract rounding** coarse on the highest-notional instruments (ES/NQ/ZB)
  even at $10M; finer on grains/FX.
- **Runtime multi-hour** (33 roots x 15.7y x ~14 windows); run in background.

## B/C Tie-In

- The `carver_tsmom_broad.yaml` universe becomes the reusable standard futures basket
  for B (strategy-pluggable runner) and C (carry strategy).
- The IDM/FDM gap this test highlights is the natural first harness enhancement once we
  begin comparing strategies head-to-head.

## Out of Scope

- No IDM/FDM implementation (documented as caveats; separate enhancement).
- No changes to harness-core (`run_futures_backtest`, simulator, sizing, loader).
- No pluggable-strategy registry (that is B) - the walk-forward change is a narrow
  universe-parametrization of one script, not a strategy-routing layer.
