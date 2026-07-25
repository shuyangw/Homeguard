# Session Handoff: Futures Pipeline (Roll Calendar + Backtest Harness)

**Date:** 2026-07-02 · **Working dir:** `C:\Users\qwqw1\Dropbox\cs\github\Homeguard` · **Env:** conda `fintech` (`/c/Users/qwqw1/anaconda3/envs/fintech/python.exe`) · **Model:** Opus 4.8 (1M)

## Resume Here (read this first)
- **Goal:** Stand up a trustworthy futures backtest pipeline so strategy ideas can be tested. Two efforts done this session: (1) OI-primary roll calendar + data-layer repair; (2) the futures backtest execution harness proven with Carver TSMOM.
- **Status:** BOTH efforts complete, merged to `main`, and pushed. `main == origin/main == a855ae2`. Working tree clean, on branch `main`. Full futures test suite green (655 passed; only failures are a pre-existing missing-AAPL-parquet env issue on the untouched equity path).
- **Next steps (pick up here):**
  1. If continuing futures strategy work: each next strategy (MOP TSMOM, commodity/FX/rates carry, Donchian, pairs, seasonality) is its own brainstorm->spec->plan->build cycle ON the harness, per `docs/strategies/research/20260509_FUTURES_STRATEGY_TESTING_PLAN.md`.
  2. Optional harness fidelity fix (won't flip verdicts, non-blocking): add Carver's Forecast Diversification Multiplier (~1.1-1.5) to `combined_forecast` in `src/strategies/advanced/carver_indicators.py` (forecasts currently under-scaled).
  3. Other documented non-blocking follow-ups listed under "Key Takeaways".
- **Blockers / open questions:** None blocking. Carver TSMOM verdict is WEAK/REJECT (a valid finding, not a bug).
- **To resume, you need:** conda `fintech`; data at `H:/Stock_Data/futures/` (consolidated layout); run futures tests with `PYTHONPATH=.` when a test imports `scripts/`. Live-trading (IBKR) is NOT involved in this work.

## Original Task
Progression of asks this session: (a) "Diagnose our futures data" (GC sparsity) -> (b) analyze two external strategy-proposal docs vs our infra -> (c) "write a new comprehensive doc on how we can test each strategy with our infra" -> (d) fix a `/doctor` settings issue -> (e) "what's needed to implement the roll calendar" -> brainstorm+plan+BUILD it -> (f) "make sure everything in our futures pipeline is good to start idea testing" (audit revealed the backtest execution layer did not exist) -> brainstorm+plan+BUILD the harness -> (g) merge to main, push, write session log, plan+solve cleanup #2 -> (h) this handoff.

## Subtasks & Progress
- [x] Futures data diagnosis — GC `.c.0` calendar-roll was broken (~43 bars/day); already fixed earlier via `.v.0` volume-roll rebuild. Data consolidated under `H:/Stock_Data/futures/databento/*` + `futures/definitions/`.
- [x] Strategy-doc analysis — `FUTURES_STRATEGIES_PROPOSAL.md` (25 strats) and the compass doc (13 families) assessed vs our data; produced `docs/strategies/research/20260509_FUTURES_STRATEGY_TESTING_PLAN.md` (per-strategy testability + 4 infra gaps A/B/C/D). NOTE: this doc's paths are the OLD flat layout (stale); the actual data moved to `futures/databento/*`.
- [x] `/doctor` fix — user's `~/.claude/settings.json` had invalid `permissions.allow: ["Bash:*"]`. (Interrupted before applied; NOT resolved this session — see Open. Low priority; `dangerouslySkipPermissions:true` is set so the allow rule is moot anyway.)
- [x] Roll calendar (Gap D) + Phase-0 data-layer repair — MERGED to main, PUSHED (was `6c1a4fc`). 53-root cache built.
- [x] Backtest harness (Gaps A/B/C) — MERGED to main, PUSHED. Carver TSMOM = WEAK/REJECT.
- [x] Session log — `docs/progress/20260702_FUTURES_PIPELINE.md` (committed+pushed).
- [x] Cleanup #2 — pct_change FutureWarning + stale readiness prose; planned, subagent-executed, merged, pushed (`a855ae2`).
- [ ] `/doctor` settings fix — still open (out of futures scope; `Bash:*` is invalid allow-rule syntax; fix = `"Bash"` or a scoped `"Bash(git *)"`).

## Key Decisions & Tradeoffs
- **Roll signal = OI-primary + volume tiebreak + calendar fallback, with FND clamp.** Why: matches Databento `.n.0` / vendor standard; OI empirically respects first-notice. Tradeoff: FND cannot be read from data (no field), so encoded as approximate per-family offset rules (metals/energy/grains/etc.); golden-date test guards it.
- **"next" contract exposed BOTH by-cycle and by-OI.** Why: carry literature uses varying conventions; let backtests A/B.
- **Harness = dedicated futures path, NOT retrofit of the equity/crypto PortfolioSimulator.** Why: correct futures semantics (per-contract $ costs, daily MTM via multiplier, margin) + zero risk to OMR/RAMP/CSCM. Tradeoff: more net-new code.
- **Margin = SPAN-style approximation (scan-range + offset matrix), replaceable module.** Why: true CME SPAN needs historical parameter files we don't have + is a huge subsystem, and is irrelevant to TSMOM (outrights). Offsets default-ON (ES/NQ 0.75, ZN/ZB 0.70).
- **First strategy = Carver multi-speed TSMOM.** Why: parameter-free (speeds (4,16)(16,64)(64,256) + cap 20 are doctrine, NEVER optimized) -> no overfit surface -> clean methodology read.
- **Config-driven + registry-integrated** (`asset_class: futures` routes in `backtest_runner.py`; no-op for existing configs).
- **Post-acceptance fix chosen = option 1 (equity-feedback sizing + bankruptcy floor).** Why: the first walk-forward exposed negative equity + `pct_change` explosion contaminating stats; user chose the proper fix over a quick floor-only. Result: stats cleaned (see below).

## Discussion Summary
- The pivotal discovery (harness effort): the roll/carry/OI/definitions modules ALREADY existed but were silently broken by the 2026 data consolidation (read old flat paths, failed closed to empty; fixture-only tests hid it). Proven by measuring against `origin/main` (39 pass -> 15 fail on affected files). Repaired in Phase 0.
- The audit for "ready to test ideas" found the DATA/signal layer ready but the BACKTEST EXECUTION layer (Gaps A/B/C) nonexistent: `StreamingDataLoader` equities/crypto-only, cost model 9/53 roots and wired to nothing, no futures simulator, no strategies/config/runner. That framed the harness build.
- First walk-forward (Task 10) gave contaminated stats: OOS Sharpe -0.45, skew -30.5, kurt 1332, and 1.5x-cost Sharpe (0.24) ABOVE 1x (backwards). Root-caused: simulator let equity go negative (no floor); `pct_change` on a zero-crossing equity curve explodes. Fixed (Task 11) with equity-feedback sizing (size vs live equity each rebalance) + bankruptcy floor (equity >= 0 after BOTH MTM and cost). Re-run: skew -0.39, kurt 8.7, 1.5x Sharpe 0.0798 correctly BELOW 1x 0.1088. Verdict WEAK stands on clean stats.
- Execution model was subagent-driven-development: fresh implementer + independent reviewer per task, controller (me) adjudicating. ~9 real defects caught pre-merge (shadowed commission table; offsets accidentally disabled; pd.NA dtype corruption; vacuous margin test; inline-vol rule violation; negative-equity contamination; cost-driven floor gap; LE/HE 100x multiplier authoring error; mis-specified golden-date constants). Reviewers repeatedly verified the shared-loader changes did NOT regress the merged roll/carry/golden paths.
- Recurring operational gotcha: background implementer/review subagents sometimes stalled by backgrounding a slow pytest and "waiting"; resumed via SendMessage instructing foreground execution.

## Commands & Outputs
```
# final sync state
$ git rev-parse --short main origin/main   -> a855ae2 / a855ae2 (in sync)

# harness acceptance (post equity-feedback fix), full walk-forward:
oos_sharpe=0.1088 psr=1.0000 dsr=1.0000 pbo=0.4377 oos_sharpe_1_5x_cost=0.0798 n_windows=12
# verdict: WEAK (does not clear combined gate; PBO ~0.44 near coin-flip)

# final whole-branch review (opus) full futures suite:
3 failed, 655 passed, 3 skipped   # 3 failures = missing-AAPL-parquet in tests/backtesting/engine/test_rolling_mode.py (equity path, pre-existing/env)
# golden roll-calendar gate: 2 passed

# 53-root roll-calendar cache build: 53 of 53 built, 0 skipped
```

## Files Touched (high-signal; all on main @ a855ae2)
Roll calendar / data layer:
- `src/data/futures/paths.py` (NEW) — consolidated-path single source of truth.
- `src/data/futures/contract_specs.py` (NEW) — 53-root specs: multiplier, tick, tick_value, currency, cycle_months, settlement_type, fnd_offset_days, initial_margin, maintenance_margin, max_contracts. Multipliers come ONLY from here (definitions `contract_multiplier` is a garbage i32 sentinel).
- `src/data/futures/roll_calendar.py` (NEW) — `detect_rolls` (OI crossover + hysteresis), `apply_fnd_clamp`, `RollCalendar` (get_front / get_nth_by_cycle / get_nth_by_oi / roll_events; fail-loud `NoActiveContractError`). Cache cols: date, front_symbol, front_expiration, front_activation, next_cycle_symbol, next_oi_symbol, dte_front, roll_trigger.
- `src/data/derivations/futures/open_interest.py` — added `per_contract_open_interest` (stat_type=9).
- `src/data/continuous_contract_loader.py` — repointed to consolidated paths; deterministic roll tie-break (`sort ["date","vol","symbol"]`), vectorized outright filter, bounded per-(root,year) volume cache (was >40GB), graceful empty-root skip.
- `src/data/carry_calculator.py`, `futures_definitions_loader.py`, `derivations/futures/{sofr,yields}.py`, `signed_volume_estimator.py`, `validation/futures/checks/*` — repointed to consolidated paths; `compute_history` fails loud on missing dataset dir.
- `scripts/data/build_roll_calendar.py` (NEW) — batch builder -> `futures/roll_calendar/{root}.parquet`.
- `src/data/roll_detector.py` — `get_upcoming_rolls` wired to RollCalendar.

Harness:
- `src/backtesting/costs/futures.py` — 53-root cost model, single commission source.
- `src/backtesting/margin/futures_margin.py` (NEW) — `MarginModel` (requirement / check_and_scale / utilization; default-on offsets).
- `src/backtesting/utils/position_sizer_futures.py` — `size_from_forecast` (forecast/vol-target -> signed int contracts, margin-capped).
- `src/backtesting/data/futures_backtest_loader.py` (NEW) — `load_daily_panel` (ratio-adjusted continuous daily, MultiIndex (root, {close,ret})).
- `src/strategies/advanced/carver_indicators.py` (NEW) — `ewmac_forecast`, `combined_forecast`, `FORECAST_SCALARS` (Table 19: 10.6/6.49/3.75). Uses `np.nan` (NOT pd.NA) for zero-vol.
- `src/strategies/advanced/carver_momentum_strategy.py` (NEW) — `CarverMomentumStrategy.forecast_panel`.
- `src/backtesting/engine/futures_portfolio_simulator.py` (NEW) — `_simulate(close, target_provider)` core loop; `run(close, target_contracts)` (static targets, Task-8 known-answer tests: equity [25000,25500,25500], final 24994); `run_sized(close, forecasts, daily_vol_panel, vol_target, div_mult)` (equity-feedback); bankruptcy floor after MTM AND cost. Isolated from equity/crypto PortfolioSimulator.
- `src/backtesting/engine/futures_backtest.py` (NEW) — `run_futures_backtest(config)` orchestration.
- `config/backtesting/carver_tsmom.yaml` (NEW).
- `src/backtest_runner.py` — routes `asset_class: futures`.
- `scripts/backtest_scripts/run_carver_walkforward.py` (NEW) — walk-forward + PSR/DSR/PBO + 1.5x cost; trial_count=1 (parameter-free).
- `docs/reports/futures/CARVER_TSMOM_READINESS.md` — acceptance report (WEAK verdict; "tail statistics (resolved)" note).

Docs (planning/spec, tracked under `docs/strategies/research/`; note `docs/superpowers/` is gitignored so specs/plans live here):
- `20260701_FUTURES_ROLL_CALENDAR_DESIGN.md` + `_PLAN.md`
- `20260701_FUTURES_BACKTEST_HARNESS_DESIGN.md` + `_PLAN.md`
- `20260702_FUTURES_CLEANUP_PLAN.md`
- `20260509_FUTURES_STRATEGY_TESTING_PLAN.md` (STALE PATHS — pre-consolidation flat layout)
- `docs/progress/20260702_FUTURES_PIPELINE.md` (session log)

## Key Takeaways & Gotchas
- **Data is at `H:/Stock_Data/futures/databento/{1min, per_contract_1min, statistics, options_1min, ...}` and `futures/definitions/`.** The 20260509 doc's `futures_1min/` etc. paths are STALE.
- **Multipliers/margins ONLY from `contract_specs.SPECS`.** Definitions `contract_multiplier` = i32 sentinel (2147483647). Guard test: `tick_value == multiplier*tick_size`.
- **Carver is parameter-free by design** — never expose speeds/cap to optimization.
- **Return basis = ratio-adjusted continuous close** (`.v.0` volume-roll already removes roll discontinuities; no separate roll-P&L term). Use `pct_change(fill_method=None)`.
- **Roll calendar is ONLY for per-contract strategies** (carry/spreads). Applying it to continuous-bar strategies is a double-roll bug.
- **The equity-feedback + bankruptcy-floor pattern is now the correct sizing/accounting model** for any futures strategy on this harness (avoids the negative-equity/pct_change contamination). `run_sized` is the path to use; `run(explicit targets)` is the low-level primitive.
- **subagent stall gotcha:** if a background implementer/reviewer "waits for a background job", SendMessage it to run pytest FOREGROUND and finalize.
- **Non-blocking follow-ups (documented):** Carver FDM missing (forecasts under-scaled); approximate margins for SR1/SR3 + micro-yield (10Y/30Y/5YY/2YY) placeholders; uniform max_contracts=100 across 53 roots -> make per-root; `forecast_panel` raises KeyError on a missing root (walk-forward pre-filters so shipped paths safe); `MarginModel.utilization` returns inf on blown days; `append_run` non-fatal wrap (mild methodology 9.3 conflict); definitions-loader `storage_root` DI restored.
- **`.superpowers/sdd/` is gitignored scratch** (ledgers, task briefs, per-task reports). Recovery map for the SDD runs if needed.

## References
- Repo: https://github.com/shuyangw/Homeguard (branch `main` @ a855ae2)
- Methodology: `docs/methodology/backtesting.md` (statistical gates §2, cost §4, stopping §5, registry §9)
- Strategy pipeline rules: `.claude/rules/strategy-pipeline.md`
- Experiment registry: `output/experiments.duckdb` via `src/experiments/registry.py::append_run`
