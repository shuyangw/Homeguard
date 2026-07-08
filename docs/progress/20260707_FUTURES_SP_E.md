# Futures SP-E: External Data Feeds + Consuming Strategies - 2026-07-07

## Summary
Shipped sub-project E of the futures strategy-testability campaign: four external
data feeds plus three consuming strategies, all reusing existing patterns with NO
changes to `load_daily_panel`, the futures cost model, or the margin model. Built
via subagent-driven development in an ISOLATED git worktree (`.worktrees/futures-sp-e`)
so a concurrent FX session on `main` could not collide (it did in SP-A). Final Opus
whole-subproject review: MERGE-READY.

## Context
Campaign umbrella: `docs/superpowers/specs/2026-07-07-futures-strategy-testability-campaign-design.md`.
SP-E design: `docs/superpowers/specs/2026-07-07-futures-sp-e-external-data-design.md`.
SP-E plan: `docs/superpowers/plans/2026-07-07-futures-sp-e-external-data.md`.
Exploration finding that reshaped SP-E: event calendars (FOMC/NFP/CPI) were ALREADY
built (`load_macro_calendar`), and CoT was already built for FX only -- so SP-E only
added what was genuinely missing.

## Changes Made
- **E1 EIA calendar** (data-only): `"eia"` added to `VALID_EVENT_TYPES`;
  `scripts/data/generate_eia_calendar.py` emits `config/macro_calendar/eia.yaml`
  (Wed release, shift to Thu on holiday weeks; 887 dates 2010-2026).
- **E2 CoT broad-universe extension**: `cftc_cot.py` gained a Legacy Futures-Only
  report path (`parse_legacy_csv`, `COT_LEGACY_INSTRUMENTS`, `fetch_all_legacy`) ->
  `alt_data/cot/<root>/legacy_weekly.parquet`. 22 roots; all from 2015, 20/22 from 2010.
- **E3 Binance perp funding** (`binance_funding.py`): USDT-M funding history ->
  `alt_data/funding/<root>/funding.parquet` (daily-annualized). LIVE fetch geo-blocked.
- **E4 Cboe VIX** (`cboe_vix.py`): per-contract VX settlement -> front/second curve
  `alt_data/vix/vx_curve.parquet` (real 3304-row curve, 2013-05..2026-07).
- **C1 #49 funding carry** (`futures_funding_strategy.py`): forecast_panel on BTC/ETH,
  self-loads funding. Registered `FuturesFundingCarry`.
- **C2 #37 CoT tilt** (`futures_cot_strategy.py`): forecast_panel on the broad universe,
  self-loads CoT net positioning, LAGGED to the CFTC Friday publication (holiday-aware).
  Registered `FuturesCoTTilt`.
- **C3 #26/#27 VIX roll-down** (`src/backtesting/vix/vix_rolldown_eval.py`): return stream
  (short-VX-in-contango + backwardation kill-switch, roll-day-excluded) gated via the
  shared PSR/DSR/PBO helpers. SP-E trial ledger at `docs/strategies/research/20260707_FUTURES_SP_E_TRIALS.md`.

## Commits (on main, fast-forward 281771e..34dc345)
- `60f2b6d` EIA calendar; `6c297d3` CoT Legacy extension; `04b255a` Binance funding;
  `a9adc41` Cboe VIX; `4af55cc` #49 funding carry; `b00c7fe` #37 CoT tilt;
  `8df6c50` fix CoT whitespace-null; `044030f` fix CoT guard; `7ca6f06` #26/#27 VIX;
  `dd6bbb6` fix VIX roll-jump; `34dc345` fix CoT holiday-week lag + VIX import.

## Known Issues / Remaining Work
- **Binance funding geo-blocked (HTTP 451)**: no funding data on disk. #49 is unit-tested
  (monkeypatched) but has no real-data smoke. To finish #49: fetch funding from an
  unblocked environment, then (a) `_FUNDING_SCALAR=2.0` is under-calibrated (forecasts
  ~0.25-1.0 vs the ~10 convention) -> a Carver avg-abs-forecast-10 sanity pass (fixed
  convention, NOT fit to results), and (b) run the correlated-re-expression check vs the
  deployed CME-basis satellite (#48).
- **No walk-forward gated verdicts recorded yet**: SP-E built the capability + one
  confirmation run each. Real results so far (pre-registered, not yet ledger verdicts):
  #37 CoT tilt OOS Sharpe -0.341 (contested [C], likely sub-gate); #26 VIX roll-down
  +0.564 (positive VRP, with real crash tail: skew -2.50, kurt 20.4, pre-cost).
- CoT CL/HO miss 2010-2014 (comma-in-name misalignment in malformed early archives;
  fail-safe drop, no wrong data). EIA dates are formula-generated (validate vs EIA's
  official schedule before #41 in SP-B). VIX PBO is NaN on a single-config return stream
  (inherited `_compute_pbo` limitation). macro_calendar docstrings not updated for "eia".
- **Next campaign sub-projects**: B (intraday engine -- the one genuinely new simulator;
  note the FX session-clock work on main is groundwork), C (spread engine), D (options-IV).

## Validation
- SDD per-task reviews caught THREE real bugs the offline tests missed, all fixed +
  re-reviewed: (1) CoT Legacy parser silently nulled whitespace-padded CFTC numbers ->
  the entire CoT cache was null (caught by the #37 smoke; fixed + cache rebuilt +
  regression test); (2) VIX roll-down counted monthly roll gaps as returns -> OOS Sharpe
  -0.854 CONTAMINATED, corrected to +0.564 after excluding roll-day returns; (3) CoT
  publication lag ignored CFTC holiday-week delays -> a bounded ~1-day lookahead, fixed
  to a holiday-aware monotone-forward lag.
- Tests: 13 feed/vix + 31 futures-strategy (SP-A regression intact). 2 pre-existing
  unrelated collection errors (`test_dukascopy_fx.py`, `test_holidays_calendar.py`) noted,
  not SP-E's.
- External-data discipline: every feed's offline canned-fixture test is the acceptance
  gate; live fetches confirm/adapt; a geo-block (funding) is reported, never fabricated.
- Worktree isolation worked -- zero collision with the concurrent FX session on main
  (contrast SP-A). Merged as a clean fast-forward; no force-push.
