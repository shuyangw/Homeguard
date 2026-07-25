# Session Handoff: Futures Carry Campaign (bond carry, integrity fixes, de-concentration stack)

**Date:** 2026-07-03 · **Working dir:** `C:\Users\qwqw1\Dropbox\cs\github\Homeguard` · **Env:** conda `fintech` (`/c/Users/qwqw1/anaconda3/envs/fintech/python.exe`) · **Model:** Opus 4.8 (1M)

## Resume Here (read this first)
- **Goal:** Get a futures strategy through the PSR/DSR/PBO gate. Carry is the lead (corrected OOS Sharpe 0.85, PBO 0.33, just over the 0.25 gate). Next: build the XS-carry + IDM de-concentration stack to push PBO < 0.25.
- **Status:** **JUST ABOUT TO START EXECUTION.** Design + plan for the de-concentration stack are written, committed, and approved. Branch `feat/carry-deconcentration` (off `main` @ 0640a5f) holds the design + plan docs; NO implementation started yet. The user chose subagent-driven vs inline execution as the immediately pending answer.
- **Next steps:**
  1. User picks execution mode (subagent-driven recommended vs inline) for `docs/strategies/research/20260703_CARRY_DECONCENTRATION_PLAN.md` (6 tasks).
  2. Execute Tasks 1-5 (TDD: cluster map -> FuturesCarryXS -> idm_weights -> div_mult threading -> 3 configs), each via fresh implementer + reviewer if subagent-driven.
  3. Task 6 (controller-run): run the 3 walk-forwards (XS-alone / IDM-alone / XS+IDM), EACH 8-thread capped + RunStatus-tracked, compare PBO/kurt/Sharpe to the 0.33 baseline.
  4. If any trial clears PBO < 0.25 at Sharpe clearly > 0 -> carry is the first futures gate-pass (deploy candidate). Else concentration is intrinsic -> proceed to W3 (more signals).
- **Blockers / open questions:** none. Awaiting the execution-mode choice.
- **To resume, you need:** conda `fintech`; data at `H:/Stock_Data/futures/` (continuous_1min, per_contract_1min, carry cache all 33 roots complete); FRED data at `H:/Stock_Data/alt_data/fred/`; branch `feat/carry-deconcentration`. **ALWAYS cap runs at 8 threads:** `POLARS_MAX_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONPATH=. python ... --jobs 8` (see Gotchas).

## Original Task (this session's arc)
Continued a long futures-pipeline session. This stretch: (a) "what else can we test" -> strategy-lead built a 20-item futures backlog; (b) deep-dive expansions on the 2 tested strats + backlog items 1-3 (5 parallel strategy-lead agents); (c) user asked "do we log trades" -> found + fixed a futures trade-logging gap; (d) "why was the run killed / did we log why" -> built run-status logging; (e) found + fixed a loader bug silently shrinking the basket; (f) implemented bond carry (FRED); (g) re-baselined carry on the complete 33-root cache; (h) designed + planned the carry de-concentration stack (XS + IDM). Verbatim last instruction: user said "proceed" to writing-plans, then chose to do a session handoff "just abt to start execution."

## Subtasks & Progress (this session, chronological)
- [x] **Futures strategy backlog** -- `docs/strategies/research/20260703_FUTURES_STRATEGY_BACKLOG.md` (20 items, 5 tiers, ranked; thesis = attack concentration not signal-discovery). Committed on main.
- [x] **5 parallel strategy-lead deep-dives** -- `docs/strategies/research/20260703_EXPAND_{momentum,carry,item1_carry_idm,item2_xs_carry,item3_carry_trend_combine}.md`. Committed on main. Key: momentum dead-standalone (keep as diversifier, corr -0.29 w/ carry); carry's concentration is fixable; XS + IDM orthogonal + stackable; combine is tail-insurance not a standalone pass.
- [x] **Trade logging for ALL asset classes** -- futures harness bypassed TradeLogger; `run_futures_backtest` computed `res.trades` and discarded it. Fixed: `run_futures_backtest(..., log_trades=True)` writes `output/backtests/futures/<strategy>/<start>_to_<end>/{trades,equity,margin_utilization}.csv`; `backtest_runner` on by default via `output.save_trades` (default True; equity/crypto already logged). Enforced in methodology Section 12.0 + backtest-driver agent + strategy-pipeline rules. MERGED main (commit `1e73776`).
- [x] **Run-status logging (survives SIGKILL)** -- `src/utils/run_status.py::RunStatus` (JSON status file + heartbeat thread under `output/run_status/`; killed run -> stale RUNNING + last heartbeat). Wired into walk-forward `main()`. Enforced in strategy-pipeline rules + backtest-driver. MERGED main (commit `e20e6e8`).
- [x] **Loader silent-basket-shrink bug** -- `continuous_contract_loader.py:221` `close_map[roll_date]` KeyError'd when a roll date had no bar; `load_daily_panel`'s over-broad `except Exception` then dropped the WHOLE root. Fixed: snap roll-date lookup to nearest trading day on-or-after; narrow except to FileNotFoundError only. Had been shrinking the basket in EVERY prior futures run. MERGED main (commit `71eefa9`).
- [x] **Bond carry (FRED CMT - DFF)** -- the 6 price-traded bonds (ZT/ZF/ZN/TN/ZB/UB) hit `CarryCalculator`'s `return 0.0` v1 fallback (inert). Now `duration * (FRED DGS{2,5,10,30} - DFF) / 100`. New `src/data/rates/fred_reader.py::get_fred_series` (point-in-time). MERGED main (commit `5da7b97` after rebase).
- [x] **Cache completed + carry re-baselined** -- GC/CL were 27-row stubs (missed in the parallel build), rebuilt full-range; 6 bonds rebuilt with real carry. All 33 roots complete. Re-baseline on the complete cache: **OOS Sharpe 0.85, PBO 0.33 (was 0.63!), kurt 21 (was 33.5), skew 1.25, nan windows W11/W12 resolved, every window 33 roots.** Most of carry's "concentration" was a data hole + the loader bug. Report `docs/reports/futures/CARRY_BROAD_READINESS.md`; 27-root preserved at `..._27root_prebond.md`. MERGED main (commit `0640a5f`).
- [x] **De-concentration DESIGN + PLAN** -- `docs/strategies/research/20260703_CARRY_DECONCENTRATION_DESIGN.md` + `_PLAN.md`. Committed on branch `feat/carry-deconcentration` (@ `e74d44f`). Approved by user.
- [ ] **EXECUTE the de-concentration plan** -- NOT STARTED. 6 tasks (see plan). This is the resume point.

## Key Decisions & Tradeoffs
- **Carry de-concentration = XS carry + IDM, both parameter-free, 3 pre-committed trials.** XS-alone / IDM-alone / XS+IDM. Why: attack the demonstrated failure mode (concentration); both are cheap and orthogonal (signal-side vs sizing-side). Risk: may still be WEAK if concentration is intrinsic.
- **IDM uses FIXED handcrafted correlations (intra 0.5 / inter 0.0), NOT estimated.** Why: data-free -> no lookahead, `trial_count` stays 1 (parameter-free). Empirical-estimated C deferred as a future logged trial.
- **IDM is a strategy-agnostic SIZING flag (`backtest.idm`), via the existing `div_mult` hook** (currently hardwired 1.0, never passed). Promote scalar -> per-root dict `div_mult_i = w_i*IDM*N_scale`. Minimal surface (no signal code change).
- **XS demeans WITHIN asset-class (4-class `asset_class` map), same-day cross-sectional z-score** (causal, no lookahead), scale 10, clip +/-20. IDM uses the 7-cluster map (energy split from commodity).
- **8-thread cap is a HARD user preference** (see Gotchas). Applied to all Task-6 runs.
- **Bond carry method = FRED CMT yield - DFF funding** (Approach A). Why: FRED DGS/DFF on disk full-history (1995-2026); theoretically correct; CTD-from-price infeasible; micro-yield siblings only 2021+.

## Commands & Outputs (load-bearing)
```
# corrected carry re-baseline (complete 33-root cache + loader fix):
oos_sharpe=0.8520 psr=1.0 dsr=1.0 pbo=0.3306 oos_sharpe_1_5x_cost=0.8225 n_windows=13
# vs 27-root pre-fix: 0.8818 / PBO 0.6319 / kurt 33.5 / skew 1.85
# per-window Sharpes now: 0.05 1.52 0.95 0.69 -0.16 1.73 2.54 1.42 -1.12 1.05 1.89 1.08 0.96 (all 33 roots)

# bond carry sanity (post-fix), 2024-03-15, correct sign (inverted curve) + duration-monotonic:
ZT -0.012  ZF -0.05  ZN -0.092  TN -0.092  ZB -0.153  UB -0.198

# 8-thread cap discovery:
pl.thread_pool_size() == 32   # polars uses 32 threads PER process by default; --jobs caps procs only

# loader bug repro (pre-fix): load_daily_panel(['TN','BZ','ES'],2010-2014) -> roots ['ES'] (TN/BZ dropped)
# post-fix -> ['BZ','ES','TN']
```

## Files Touched (merged to main this session)
- `src/backtesting/engine/futures_backtest.py` -- `register` flag (earlier) + `log_trades` (trade logging).
- `src/utils/run_status.py` (NEW) -- RunStatus.
- `src/data/continuous_contract_loader.py` -- roll-date nearest-snap fix.
- `src/backtesting/data/futures_backtest_loader.py` -- narrowed except to FileNotFoundError.
- `src/data/carry_calculator.py` -- FRED bond carry branch + `_BOND_CMT_TENOR`.
- `src/data/rates/fred_reader.py` (NEW) -- `get_fred_series`.
- `scripts/backtest_scripts/run_carver_walkforward.py` -- RunStatus wrap in main().
- `.claude/rules/strategy-pipeline.md`, `.claude/agents/backtest-driver.md`, `docs/methodology/backtesting.md` (Section 12.0) -- trade-logging + run-status requirements.
- `docs/reports/futures/CARRY_BROAD_READINESS.md` (corrected) + `_27root_prebond.md`.

## Files to Create/Modify NEXT (the de-concentration plan, NOT yet done)
- `src/data/futures/asset_class.py` -- add `CLUSTER` map + `cluster_for` (Task 1).
- `src/strategies/advanced/futures_carry_strategy.py` -- `FuturesCarryXSStrategy` subclass (Task 2).
- `src/strategies/registry.py` -- register `FuturesCarryXS` (Task 2).
- `src/backtesting/utils/idm_weights.py` (NEW) -- `compute_div_mult` (Task 3).
- `src/backtesting/engine/futures_portfolio_simulator.py` (`run_sized` div_mult scalar->dict at line ~142/156) + `src/backtesting/engine/futures_backtest.py` (`backtest.idm` flag at the `run_sized` call ~line 92) (Task 4).
- `config/backtesting/{carry_xs_broad,carry_idm_broad,carry_xs_idm_broad}.yaml` (Task 5).
- Reports out: `docs/reports/futures/CARRY_{XS,IDM,XS_IDM}_BROAD_READINESS.md` (Task 6).

## Key Takeaways & Gotchas
- **8-THREAD CAP (hard user rule, corrected me twice):** `--jobs N` caps PROCESSES only; each polars/numpy worker defaults to 32 threads. To honor "8 threads total": `POLARS_MAX_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 ... --jobs 8` (8 single-threaded workers). Saved as memory `feedback_parallel_thread_cap.md`.
- **Never mutate the working tree / switch git branches while a long background run is in flight** -- its ProcessPoolExecutor workers re-import code from disk on spawn. The 2026-07-03 carry re-baseline was killed (status `killed`, NOT OOM -- ruled out via Windows event log, 61.5 GB RAM) most likely because the harness reaped the background job when branches were switched mid-run.
- **A SIGKILL'd run cannot self-log its death** -- use RunStatus (`output/run_status/`) and read it BEFORE guessing a cause.
- **`--jobs` on the walk-forward parallelizes WINDOWS** (13), deterministic (input-order aggregation), workers use `register=False` (no registry races), parent writes one append_run.
- **Every new backtest engine MUST persist a trade log + wrap long runs in RunStatus** -- now enforced in methodology Section 12.0 + the agents.
- **Contract multipliers ONLY from `contract_specs.SPECS`** (definitions `contract_multiplier` is a garbage i32 sentinel). Carry/Carver are parameter-free (never optimize speeds/scalars/caps).
- **Prior-session context:** the RAMP equity campaign (separate track) found the same lesson -- momentum-signal tinkering fails walk-forward on concentration/BEAR fragility; structural diversification is the fix. Reinforces the carry de-concentration direction.

## References
- Repo: https://github.com/shuyangw/Homeguard (branch `main` @ 0640a5f; work branch `feat/carry-deconcentration` @ e74d44f)
- Methodology: `docs/methodology/backtesting.md` (gates Sec 2, cost Sec 4, trade-log Sec 12.0, registry Sec 9)
- Design/plan under review: `docs/strategies/research/20260703_CARRY_DECONCENTRATION_{DESIGN,PLAN}.md`
- Research briefs (detailed design): `docs/strategies/research/20260703_EXPAND_item1_carry_idm.md`, `..._item2_xs_carry.md`
- Registry: `output/experiments.duckdb` (371+ runs incl. CarverMomentum, FuturesCarry)
- Prior session log: `docs/progress/20260702_FUTURES_PIPELINE.md`, `SESSION-HANDOFF-futures-pipeline-2026-07-02.md`
