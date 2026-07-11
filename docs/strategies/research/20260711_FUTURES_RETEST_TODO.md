# Strategy Pipeline TODO -- Futures Campaign Comprehensive Retest (SP-A .. SP-E)

> Status: `[ ]` pending - `[~]` in progress - `[x]` done - `[!]` failed - `[-]` skipped
> Run: `claude --agent strategy-lead` (point it at THIS file, not the root TODO.md)
> Resume: `claude --agent strategy-lead --continue`
> Orchestrator: read THIS file FIRST on every session start. Rebuild context only from
> output files (docs/reports/futures/, docs/agent-learnings/<strategy>/,
> output/optimization/<strategy>/), never from session memory.
> SENTINEL: FIRST action `touch .claude/.strategy-lead-active`; LAST action
> `rm -f .claude/.strategy-lead-active`. Without it the backtest hook blocks every run.
> Mark `[~]` BEFORE starting a phase; mark `[x]` only AFTER verifying the output file exists.
> Never overwrite the iterations table -- increment run numbers.

## Why this retest exists (context)

The futures testability campaign (SP-A/B/C/D/E) built and gated ~19 of ~55 catalog
strategies, but ALL of it ran through superpowers + general-purpose subagents with
verdicts run directly by the controller -- NOT through strategy-lead. This TODO drives
the correction: re-validate EVERY built strategy through strategy-lead's full integrity
pipeline, with HONEST, UNIFORM deflation.

Three reasons this is the first honest evaluation, not busywork:
1. **Deflation was never uniform.** SP-D's honest DSR deflation (SR_zero = 0.733) only
   reached the `gate_return_stream` path (#26, #28, SP-C spreads). The carver gate
   (SP-A, #37) and session gate (SP-B) still pass a single-element trial-Sharpe list ->
   DSR == PSR, un-deflated. Gate 0 fixes this repo-wide.
2. **Broken/stale verdicts:** #16 mis-sampled (daily signal on a weekly runner -> verdict
   "unreliable"); some SP-E/SP-C "PBO NaN" predate SP-D's PBO 2*s fix; #36's mandatory
   book-correlation check never ran; #31 NG REJECT provisional (volume-rank F1/F2).
3. **No committed drivers** for the Path-2 sleeves (the sp_* smokes never existed in-tree).

HONEST EXPECTATION: SR_zero 0.733 exceeds every strategy's OOS Sharpe, so this will very
likely CONFIRM the negative -- nothing clears DSR >= 0.95, including the carry incumbent.
The value is rigor, governance, uniform honest deflation, fixing the broken verdicts, and
committed durable drivers. Surfacing that cleanly IS the completed objective (North Star:
surfacing a failure is success). Do NOT engineer around a FAIL.

## Incumbent to beat

carry (`FuturesCarry`) is the best DEPLOYABLE book (real cash-and-carry mechanism, passes
PBO 0.093 -- not overfit). But under honest deflation even carry FAILs DSR: OOS Sharpe
0.588 (this-run equity) / 0.765 (walk-forward) both AT/BELOW SR_zero 0.733. So "beats
carry" is NOT the bar; the bar is the combined statistical gate below, and the honest
finding is that the 40-trial (and growing) search cannot distinguish any edge from the
best-of-N-under-the-null with 95% confidence.

## Acceptance bar (read before running anything)

A strategy is VIABLE for live only if ALL hold (methodology Sections 2.5, 4, 12, 11):
1. PSR(0) > 0.95 OOS.
2. DSR > 0.95 using the PROJECT-WIDE cumulative trial count AND the real trial-Sharpe
   distribution (Gate 0.2 -- sourced from output/experiments.duckdb, growing as this
   retest appends runs). This is the binding gate; nothing is expected to clear it.
3. PBO < 0.25 (CSCV s=16, windows dropped if < 2*s rows -- the SP-D fix).
4. 1.5x cost gate: sharpe_1.5x > 0 AND (sharpe <= 0 OR sharpe_1.5x >= 0.5*sharpe).
5. Trade count >= 30 OOS; OOS/IS Sharpe ratio >= 0.7.
6. Section 12 operational gates (capacity, regime-transition robustness, information
   ratio, parameter temporal stability) -- where the execution path produces them.
7. Section 11 exit gates (MAE/MFE, exit-logic summary) for exit-bearing strategies
   (the convergence spreads #31-#34).
Parameter discipline: <= 3 tunables, economic rationale each, ±10/±20% neighborhood
Sharpe >= 0.9 of best (STABLE). No post-hoc sign flips.

---

## GATE 0 -- shared prep (MUST complete before any strategy run)

- [ ] **G0.1 Deflation-consistency fix (repo-wide).** Thread `CAMPAIGN_TRIAL_SHARPES`
  (from `src/backtesting/walkforward_common.py`) into the un-deflated gate paths so all
  use SR_zero 0.733, mirroring `gate_return_stream`:
  - `scripts/backtest_scripts/run_carver_walkforward.py` -- its inline
    `dsr(oos_sharpe, [oos_sharpe], ..., n_trials_project=CAMPAIGN_CUMULATIVE_TRIALS)`
    -> pass the real distribution.
  - `src/backtesting/session/session_walkforward.py::gate_session_stream` -- same.
  - `src/backtesting/walkforward_common.py::run_fx_walkforward` and
    `src/backtesting/blend/satellite_blend.py` -- same (consistency).
  - Only LOWERS DSR -> no prior PASS flips. VIX/session/spreads suites must stay green.
- [ ] **G0.2 Honest, growing trial count.** Source the DSR n_trials AND trial-Sharpe
  distribution from `output/experiments.duckdb` (methodology Section 9.4) as runs are
  appended, replacing the static `CAMPAIGN_CUMULATIVE_TRIALS = 40` / hardcoded 29-value
  list. Fallback to the documented constants if the registry read fails. Every run in
  this retest (Section 9.3) + each Phase 6.5 improvement round raises N -> raises SR_zero.
- [ ] **G0.3 Committed sleeve drivers.** Create the missing Path-2 entry points under
  `scripts/backtest_scripts/` (`sp_retest_<sleeve>.py`), each: wrapped in `RunStatus`
  (`src/utils/run_status.py`); calling the sleeve `run_*` fn; writing `returns.csv` +
  `gate.json`; applying `walkforward_common._verdict` so Path-2 emits PASS/WEAK/REJECT.
- [ ] **G0.4 Data + exclusions.** Confirm data per strategy; the EXCLUDED table below is
  authoritative for the two with no data.
- [ ] **G0.5 Run durability.** All long/bg runs in `RunStatus` -> `output/run_status/`;
  mandatory trade-log persistence (Section 12.0). Do not switch branches / mutate the
  tree mid-run.
- [ ] **G0.verify:** on a synthetic positive-Sharpe stream, confirm `run_carver_walkforward`
  and `gate_session_stream` now give DSR < PSR (deflation bites); suites green.

---

## THE WORK-LIST

Each strategy: strategy-lead Phases 5 (backtest-driver) -> 6 (validate) -> [6.5 improve /
7 optimize if marginal-but-real] -> 8 (final validation). Phase 3 (implement) SKIPPED
where code exists; used only for the flagged caveat-fixes. Per-strategy nested checklist:
`[ ] Phase 5 backtest + record row / [ ] Phase 6 validate / [ ] 6.5-7 if marginal / [ ] Phase 8 final`.

### Tier 1 -- Path 1: carver / forecast_panel (config-driven; gate = run_carver_walkforward, Gate-0-deflated)

Command shape (under sentinel, RunStatus-wrapped):
`conda run -n fintech python scripts/backtest_scripts/run_carver_walkforward.py --config config/backtesting/<yaml> --train-months 36 --test-months 12 --step-months 12 --report docs/reports/futures/<STRAT>_READINESS.md --json output/<strat>_gate.json --jobs 8 > logs/backtesting/<strat>.log 2>&1`

- [ ] **#3 FuturesXSMomentum** -- was WEAK (PBO 0.579). config `xs_commodity_momentum.yaml`.
  - [ ] Phase 5 backtest + record / [ ] Phase 6 validate / [ ] Phase 8 final
- [ ] **#10 FuturesCarryXS** -- was WEAK (PBO 0.690; highest raw 0.846). config `curve_slope_xs.yaml`.
  - [ ] Phase 5 / [ ] Phase 6 / [ ] Phase 8
- [ ] **#13 FuturesCarryTrend** -- only SP-A "PASS" (PBO 0.189, 0.357 << carry). config `carry_trend_gate.yaml`.
  Re-gate deflated (expected to fail now) + run the marginal-Sharpe-vs-the-pair re-expression check.
  - [ ] Phase 5 / [ ] Phase 6 / [ ] Phase 8
- [ ] **#15 FuturesSameMonthSeasonality** -- was WEAK (PBO 0.281). config `same_month_seasonality.yaml`.
  - [ ] Phase 5 / [ ] Phase 6 / [ ] Phase 8
- [ ] **#16 FuturesTurnOfMonth** -- REJECT* MIS-SAMPLED. config `turn_of_month.yaml`.
  **CAVEAT-FIX (Phase 3):** it is a DAILY signal run on a WEEKLY-rebalance runner; the
  verdict is unreliable. Rebuild as a daily-rebalance walk-forward BEFORE re-gating.
  - [ ] Phase 3 caveat-fix (daily rebalance) / [ ] Phase 4 review / [ ] Phase 5 / [ ] Phase 6 / [ ] Phase 8
- [ ] **#23 FuturesReversal** -- was WEAK (PBO 0.805). config `index_reversal.yaml`.
  - [ ] Phase 5 / [ ] Phase 6 / [ ] Phase 8
- [ ] **#37 FuturesCoTTilt** -- was REJECT (-0.124). config `cot_tilt.yaml`. Note: prior PBO NaN predates the fix; re-run.
  - [ ] Phase 5 / [ ] Phase 6 / [ ] Phase 8
- [ ] **carry incumbent (FuturesCarry)** -- re-gate the benchmark under the honest deflated carver gate; report DSR/PBO for the portfolio summary.
  - [ ] Phase 5 / [ ] Phase 6

### Tier 2 -- Path 2: return-stream sleeves (run_* via G0.3 drivers; gates deflated)

- [ ] **#26/#27 VIX roll-down** -- `run_vix_rolldown` (`vix/vix_rolldown_eval.py`), gate_return_stream (deflated).
  Already FAIL (DSR 8.9e-06, PBO 0.613, max DD -81.1%). Re-run for the durable record + attach the subperiod/skew tail audit.
  - [ ] Phase 5 (driver sp_retest_vix) / [ ] Phase 6 / [ ] Phase 8
- [ ] **#28 VRP short-VX1** -- `run_vrp` (`vol/vrp_strategy.py`), gate_return_stream (deflated).
  Already FAIL + re-expression of #26 (corr 0.479, marginal Sharpe 0.015). Re-run; attach re-expression stats.
  - [ ] Phase 5 / [ ] Phase 6 / [ ] Phase 8
- [ ] **#21/#25 Overnight drift** -- `run_overnight_drift` (`overnight_drift_strategy.py`), gate_session_stream (NOW Gate-0-deflated).
  Was WEAK (0.792/0.671, PBO 0.513). Re-gate under 0.733.
  - [ ] Phase 5 / [ ] Phase 6 / [ ] Phase 8
- [ ] **#21 Hour-slice** -- `run_hour_slice` (same file), gate_session_stream (deflated). Was REJECT (-0.023).
  - [ ] Phase 5 / [ ] Phase 6
- [ ] **#36 Intermarket (NQ/ES, RTY/ES)** -- `run_intermarket` (`spread_intermarket_strategy.py`), gate_return_stream (deflated).
  Was NQ/ES FAIL (0.329) / RTY/ES REJECT (-0.280). **CAVEAT-FIX:** run the MANDATORY
  book-correlation check vs the equity-momentum sleeve (never run) -- a high positive corr
  means re-expression regardless of Sharpe.
  - [ ] Phase 5 / [ ] Phase 6 (+ book-corr) / [ ] Phase 8
- [ ] **#31 Calendar (CL/NG/ZC/ZS/ZW)** -- `run_calendar` (`spread_calendar_strategy.py`), gate_convergence (deflated).
  REJECT all (roll-masked). **CAVEAT-FIX (optional):** NG REJECT provisional (PBO 0.320);
  try RollCalendar-based F1/F2 instead of volume-rank to stop over-masking. Produce
  Section 11 exit diagnostics (SpreadTrade exits: converge / time / structural).
  - [ ] Phase 5 (+ Section 11 exit diag) / [ ] Phase 6 / [ ] NG RollCalendar caveat-fix / [ ] Phase 8
- [ ] **#32 Crack (RB-CL, HO-CL)** -- `run_processing` (`spread_processing_strategy.py`), gate_convergence (deflated).
  REJECT (negative). Section 11 exit diagnostics apply.
  - [ ] Phase 5 (+ exit diag) / [ ] Phase 6
- [ ] **#33 Crush (ZM+ZL-ZS)** -- `run_processing` (same), gate_convergence (deflated).
  MARGINAL (PBO 0.109 clean, Sharpe 0.136 trivial) -- the ONLY candidate that might reach
  Phase 6.5. Apply the improvement-design discipline (pre-committed hypotheses, trial-count cost).
  - [ ] Phase 5 (+ exit diag) / [ ] Phase 6 / [ ] Phase 6.5 improve (<=2 rounds) / [ ] Phase 7 optimize / [ ] Phase 8
- [ ] **#34 Ratio (GC/SI)** -- `run_ratio` (`spread_ratio_strategy.py`), gate_convergence (deflated).
  REJECT (0.269, PBO 0.674; kurt 109 genuine GC/SI tails). Section 11 exit diagnostics apply.
  - [ ] Phase 5 (+ exit diag) / [ ] Phase 6

### Tier 3 -- ungradeable by the walk-forward gate (document, mark [-] with reason)

- [-] **#39 Pre-FOMC** -- `run_prefomc` (`prefomc_strategy.py`). n_windows=0 (~8 events/yr never fill a
  12-month/10-sample window). Architecturally ungradeable by this gate -- NOT a fixable
  verdict. Record the diagnostic (decay split is small-n noise); skip.
- [-] **#35 Steepener (2s10s/2s5s/5s30s)** -- `run_steepener` (`spread_steepener_strategy.py`).
  n_windows=0 (2YY from ~2021; 3yr z-window leaves ~1.5yr < 48m; 5YY degraded ~440 rows).
  Document; note the ZT/ZN DV01 fallback as a possible future rebuild when the yield-future
  history matures. Skip until then.

---

## EXCLUDED -- cannot retest (no data / not implemented)

| # | Name | Reason | Path to enable |
|---|---|---|---|
| 49 | FuturesFundingCarry | Binance funding geo-blocked (HTTP 451); unit-tested only | fetch funding elsewhere, calibrate `_FUNDING_SCALAR`, + #48 re-expression check |
| 9 | multi-horizon carry blend | never implemented | build a multi-horizon carry cache (adjacent to SP-E data work) |

---

## Iterations table (one row per strategy per cost leg -- NEVER overwrite)

| Run | Strategy | Path | Sharpe 1x | Sharpe 1.5x | PSR | DSR (honest N) | PBO | skew | kurt | CAGR | MaxDD | DDdur | Calmar | Win% | PF | Trades | AvgHold | perWindow OOS (W1..Wn) | regime | capacity | IR | MAE/MFE | window | freq | VERDICT |
|-----|----------|------|-----------|-------------|-----|----------------|-----|------|------|------|-------|-------|--------|------|----|--------|---------|------------------------|--------|----------|----|---------|--------|------|---------|
| (fill per run) | | | | | | | | | | | | | | | | | | | | | | | | | |

Notes: Section 12 operational diagnostics (capacity/regime/IR/param-stability) come from
the Path-1 StandardReport; for Path-2 continuous sleeves record "N/A (continuous return
stream)" -- do NOT fabricate. Section 11 exit diagnostics apply ONLY to the convergence
spreads (#31-#34). Append every run to `output/experiments.duckdb` (Section 9.3) -- this
feeds G0.2's growing trial count.

## The null option

The expected honest outcome is that NO strategy clears DSR >= 0.95 under the growing
honest deflation -- including carry. If so, the portfolio summary states exactly that:
the futures catalog, evaluated with contamination-free, deflation-correct rigor, yields
no statistically-certified sleeve; carry remains the best DEPLOYABLE book but not a
certified one. That is the completed objective.

## File pointers

- Ledgers (prior verdicts): `docs/strategies/research/20260707_FUTURES_SP_A_TRIALS.md`,
  `20260707_FUTURES_SP_E_TRIALS.md`, `20260710_FUTURES_SP_B_TRIALS.md`,
  `20260710_FUTURES_SP_C_TRIALS.md`, `20260711_FUTURES_SP_D_TRIALS.md`.
- Session logs: `docs/progress/20260707_FUTURES_SP_A.md` ... `20260711_FUTURES_SP_D.md`.
- Gate: `src/backtesting/walkforward_common.py` (`gate_return_stream`, `_verdict`,
  `CAMPAIGN_TRIAL_SHARPES`, `CAMPAIGN_CUMULATIVE_TRIALS`); carver
  `scripts/backtest_scripts/run_carver_walkforward.py`; session
  `src/backtesting/session/session_walkforward.py`; convergence
  `src/backtesting/spreads/convergence.py`.
- Strategy code: `src/strategies/advanced/futures_*.py`, `spread_*.py`,
  `overnight_drift_strategy.py`, `prefomc_strategy.py`; `src/backtesting/vix/`,
  `src/backtesting/vol/`, `src/backtesting/spreads/`.
- Configs: `config/backtesting/*.yaml` (Path-1). Reports: `docs/reports/futures/`.
  Run data: `output/backtests/{futures,session}/`. Optimization: `output/optimization/<strategy>/`.
- Methodology: `docs/methodology/backtesting.md` (Sections 2, 4, 9, 11, 12). Rules:
  `.claude/rules/strategy-pipeline.md`. Plan of record: `.claude/plans/ok-lets-discard-that-validated-blum.md`.
