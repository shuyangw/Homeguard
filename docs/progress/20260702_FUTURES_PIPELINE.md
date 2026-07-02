# Futures Pipeline: Roll Calendar + Backtest Harness - 2026-07-02

## Summary

Built out the futures pipeline from data primitives through a runnable backtest harness across two subagent-driven efforts, both merged to `main` and pushed. First: repaired the silently-broken futures data/roll/carry layer and shipped an OI-primary roll calendar (Gap D). Second: built the futures backtest execution layer (Gaps A/B/C) - a dedicated daily multi-instrument simulator with per-contract mark-to-market, SPAN-style margin, equity-feedback vol-target sizing, and a walk-forward statistical gate - proven end-to-end with Carver multi-speed TSMOM (honest result: WEAK, does not clear the gate).

## Changes Made

### Effort 1 - Roll calendar + data-layer repair (merged at `6c1a4fc`)
- **Discovery that reframed the work:** the roll/carry/OI/definitions modules already existed but were silently broken by the 2026 data consolidation (reading old flat `futures_1min/` paths that no longer exist, failing closed to empty results; fixture-only tests masked it). Proven by measuring against `origin/main` (39 pass -> 15 fail on affected files).
- **Phase 0 (repair):** centralized paths in `src/data/futures/paths.py`; repointed ~9 modules to the consolidated `futures/databento/*` + `futures/definitions/` layout; removed a silent-swallow in `CarryCalculator.compute_history` (now fails loud on missing dataset dir); restored the `FuturesDefinitionsLoader` `storage_root` DI point; retargeted 16 downstream test monkeypatches (dead `_storage_root` -> `paths.get_local_storage_dir`); added a `git grep` guard test against stale-path regressions.
- **Phase 1 (enhance):** OI-primary roll calendar - `src/data/futures/contract_specs.py` (53-root static spec table: multiplier/tick/settlement/cycle), `per_contract_open_interest`, pure `detect_rolls` (OI crossover + hysteresis), `RollCalendar` (FND clamp, dual-`nth` API, fail-loud lookups), a batch builder, and a golden-date acceptance gate anchored on independent CME 2024 expiry/first-notice facts. `get_upcoming_rolls` wired to the calendar.
- **Cache:** built the roll-calendar cache for all 53 roots (`futures/roll_calendar/{root}.parquet`).

### Effort 2 - Backtest harness (merged at `ec93dfa`, 19 commits)
- **Cost model** `src/backtesting/costs/futures.py`: extended 9 -> 53 roots, single commission source (tick values from `contract_specs`); original 9 round-trip values byte-identical.
- **Margin:** added `initial_margin`/`maintenance_margin` to `contract_specs` (maintenance = round(0.9*initial), structurally guaranteed); `src/backtesting/margin/futures_margin.py` - `MarginModel` (scan-range margin + default-on inter-commodity offsets ES/NQ, ZN/ZB + BP cap + utilization).
- **Sizing:** `size_from_forecast` in `position_sizer_futures.py` (forecast/vol-target -> signed integer contracts, margin-capped); added `max_contracts` to `contract_specs`.
- **Data:** `src/backtesting/data/futures_backtest_loader.py::load_daily_panel` (ratio-adjusted continuous daily panel).
- **Strategy:** `src/strategies/advanced/carver_indicators.py` (EWMAC forecasts, parameter-free doctrine) + `carver_momentum_strategy.py` (`CarverMomentumStrategy.forecast_panel`).
- **Simulator:** `src/backtesting/engine/futures_portfolio_simulator.py` - separate from the equity/crypto `PortfolioSimulator`; per-contract daily MTM into cash; per-contract dollar costs on contracts traded; **equity-feedback sizing** (`run_sized` sizes against live equity each rebalance) + **bankruptcy floor** (equity provably >= 0 after both MTM and cost debits).
- **Runner:** `src/backtesting/engine/futures_backtest.py::run_futures_backtest` + `config/backtesting/carver_tsmom.yaml`; `src/backtest_runner.py` routes `asset_class: futures` (no-op for existing equity/crypto configs).
- **Acceptance:** `scripts/backtest_scripts/run_carver_walkforward.py` + `docs/reports/futures/CARVER_TSMOM_READINESS.md` (walk-forward, PSR/DSR/PBO, 1.5x cost, trial-count=1 for a parameter-free strategy).
- **Shared-loader hardening** (surfaced by the first full 12-root/15-year run): deterministic roll tie-break (fixed intermittent SIL `KeyError`), vectorized outright filter, bounded per-(root,year) volume cache (was ballooning >40GB), graceful empty-root skip. Behavior-preserving (golden gate + roll/carry tests green).

## Commits
- Roll calendar: range `c5c65f7..6c1a4fc` on `main` (design spec, plan, Phase-0 repair, Phase-1 calendar, follow-up fixes).
- Harness: range `e2d7dbd..ec93dfa` on `main` (design spec, plan, Tasks 1-10, equity-feedback + floor fix, readiness regeneration). Key: `8bd8aaf` simulator, `fedf991` end-to-end runner, `0d4c4de`/`cf72250` equity-feedback + floor, `da6ad5e` walk-forward, `ec93dfa` clean readiness report.

## Known Issues / Remaining Work
- **Carver TSMOM = WEAK** (OOS Sharpe 0.11, PBO 0.44, does not clear the combined gate). A trustworthy "don't deploy this naive version" finding, not a harness failure.
- **Follow-ups (non-blocking, from final review):**
  - `combined_forecast` omits Carver's Forecast Diversification Multiplier (~1.1-1.5) -> forecasts systematically under-scaled (doctrine-fidelity gap; won't flip the WEAK verdict).
  - Readiness report "Notes" carry stale alarmist skew/kurtosis prose contradicting the now-clean table (-0.39 / 8.7) - being fixed in a planned cleanup.
  - `pct_change` FutureWarning (pass `fill_method=None`) - being fixed in the same cleanup.
  - Approximate margins (SR1/SR3, micro-yield placeholders); uniform `max_contracts=100`; `forecast_panel` raises `KeyError` on a missing root (walk-forward pre-filters, so shipped paths safe); `MarginModel.utilization` returns `inf` on blown days; `append_run` non-fatal wrap (mild methodology 9.3 conflict).
- **Pre-existing / environmental:** 3 failures in `tests/backtesting/engine/test_rolling_mode.py` (missing local AAPL 2024-01 parquet) - equity path, unrelated to this work.
- **Next strategies:** MOP TSMOM, commodity/FX/rates carry, Donchian, pairs, seasonality - each its own spec on the harness (per `20260509_FUTURES_STRATEGY_TESTING_PLAN.md`).

## Validation
- Roll calendar: golden-date gate reproduces independent CME 2024 rolls (ES ~2d before quarterly expiries; GC ~8-10d before even-month first-notice), 2/2 pass; full 53-root cache built (0 skipped).
- Harness: full futures suite 655 passed / 3 skipped / 3 env-only failures (final whole-branch review on opus). Simulator MTM/cost hand-traced to exact numbers; bankruptcy floor RED->GREEN (equity >= 0 after MTM and cost); Task-8 known-answer numbers (25500, 24994) preserved through the equity-feedback refactor.
- Acceptance run confirmed the equity-feedback + floor fix resolved the earlier statistics contamination: skew -30.5 -> -0.39, kurtosis 1332 -> 8.7, 1.5x-cost Sharpe now correctly below 1x.
- Every task went through a fresh implementer + independent reviewer; ~9 real defects caught and fixed pre-merge (shadowed commission table, offsets disabled, pd.NA dtype corruption, vacuous margin test, inline-vol rule violation, negative-equity contamination, cost-driven floor gap, and others).
- Both efforts merged to `main` via fast-forward and pushed to `origin/main` (`ec93dfa`).
