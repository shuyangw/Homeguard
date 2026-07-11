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
Parameter discipline: <= 3 tunables, economic rationale each, +/-10/+/-20% neighborhood
Sharpe >= 0.9 of best (STABLE). No post-hoc sign flips.

---

## GATE 0 -- shared prep (MUST complete before any strategy run)

- [x] **G0.1 Deflation-consistency fix (repo-wide).** DONE (commit `170946b`, feat/futures-retest).
  Threaded `CAMPAIGN_TRIAL_SHARPES` into all un-deflated gate paths: `run_carver_walkforward.py`,
  `session_walkforward.py::gate_session_stream`, `run_fx_walkforward.py`, `satellite_blend.py::blend_books`.
  Verified DSR < PSR on a synthetic positive-Sharpe stream; VIX/session/spreads/walkforward suites
  stay green (33/37; 4 pre-existing `test_walkforward_idm_threading.py` failures confirmed unrelated
  stale-fixture rot via `git show HEAD`, accepted by strategy-lead's coordinator).
- [x] **G0.2 Honest, growing trial count.** DONE (commit `170946b`). Added
  `walkforward_common.get_campaign_trial_distribution()` -- sources n_trials + the trial-Sharpe
  distribution from `output/experiments.duckdb`, falls back to the static 40/29 baseline if the
  registry is unreadable. All five gate paths now call this instead of the static constants directly.
- [x] **G0.3 Committed sleeve drivers.** DONE (commit `b8cdec7`, + honesty-fix in same commit after
  code review). Created all ten `scripts/backtest_scripts/sp_retest_<sleeve>.py` drivers plus the
  shared `sp_retest_common.py` helper module and `sp_retest_trade_log.py` (Gate 0.5 helper). Code
  review found no call-signature/dict-key bugs; found and fixed one CRITICAL issue (the calendar/
  processing/ratio drivers falsely claimed an `exit_reason` breakdown that `SpreadTrade` does not
  persist -- replaced with `convergence_exit_summary`, which reports only what's actually available).
  One HIGH finding NOT fixed (pre-existing, out of scope): `simulate_convergence` doesn't
  differentiate stop-exit cost per Section 11.5 -- flagged as a follow-up.
- [x] **G0.4 Data + exclusions.** Confirmed: all 7 Tier 1 carver configs + the carry incumbent
  config exist in `config/backtesting/`, all declare the full 2010-06-07..2026-02-20 range (Section
  2.6 compliant, not a subset window). Carry cache confirmed present (`H:\Stock_Data\futures\carry`,
  35 roots) -- no cache build needed. #49 and #9 remain EXCLUDED per the authoritative table below;
  not attempted.
- [x] **G0.5 Run durability.** All Tier 1 runs wrapped in `RunStatus` (`carver_walkforward` already
  was; `sp_retest_trade_log_<strategy>` added). Trade-log persistence (Section 12.0) confirmed for
  all 8 Tier 1 strategies via `sp_retest_trade_log.py` (one full-range `run_futures_backtest(...,
  log_trades=True)` per strategy, separate from the walk-forward gate's per-window internal runs).
- [x] **G0.verify:** DONE twice -- once after G0.1/0.2 (synthetic stream, DSR<PSR confirmed) and
  implicitly again via all 8 real Tier 1 runs below, which show DSR collapsing far below PSR/1.0
  at real n_trials=40 (deflation clearly bites on real data, not just synthetic).

---

## THE WORK-LIST

Each strategy: strategy-lead Phases 5 (backtest-driver) -> 6 (validate) -> [6.5 improve /
7 optimize if marginal-but-real] -> 8 (final validation). Phase 3 (implement) SKIPPED
where code exists; used only for the flagged caveat-fixes. Per-strategy nested checklist:
`[ ] Phase 5 backtest + record row / [ ] Phase 6 validate / [ ] 6.5-7 if marginal / [ ] Phase 8 final`.

### Tier 1 -- Path 1: carver / forecast_panel (config-driven; gate = run_carver_walkforward, Gate-0-deflated)

Command shape (under sentinel, RunStatus-wrapped):
`conda run -n fintech python scripts/backtest_scripts/run_carver_walkforward.py --config config/backtesting/<yaml> --train-months 36 --test-months 12 --step-months 12 --report docs/reports/futures/<STRAT>_READINESS.md --json output/<strat>_gate.json --jobs 8 > logs/backtesting/<strat>.log 2>&1`

- [x] **#3 FuturesXSMomentum** -- was WEAK (PBO 0.579). config `xs_commodity_momentum.yaml`.
  Re-gated: oos_sharpe 0.2095, PSR 1.0, DSR ~0 (2.98e-208), PBO 0.579 (unchanged). VERDICT: WEAK/FAIL
  (DSR collapses to ~0 under honest deflation at n_trials=40; PBO still fails <0.25 independently).
  - [x] Phase 5 backtest + record / [x] Phase 6 validate / [-] Phase 8 (not marginal-but-real; no
  further validation warranted -- DSR ~0 is a decisive FAIL, not a borderline case)
- [x] **#10 FuturesCarryXS** -- was WEAK (PBO 0.690; highest raw 0.846). config `curve_slope_xs.yaml`.
  Re-gated: oos_sharpe 0.8458 (highest raw Sharpe in Tier 1, PSR 1.0, DSR 0.999 -- clears the DSR bar
  on Sharpe alone), but PBO 0.690 fails independently. VERDICT: WEAK (fails combined gate on PBO).
  - [x] Phase 5 / [x] Phase 6 / [-] Phase 8 (PBO 0.69 is a decisive overfitting signal, not marginal)
- [x] **#13 FuturesCarryTrend** -- only SP-A "PASS" (PBO 0.189, 0.357 << carry). config `carry_trend_gate.yaml`.
  Re-gated: oos_sharpe 0.3571, PSR 1.0, DSR ~0 (2.93e-77), PBO 0.189 (passes, consistent with prior).
  VERDICT: FAIL (DSR ~0 under honest deflation; the PBO 0.189 "PASS" from the prior campaign was never
  sufficient alone -- DSR is the binding gate and it FAILs decisively here).
  - [x] Phase 5 / [x] Phase 6 / [-] Phase 8
- [x] **#15 FuturesSameMonthSeasonality** -- was WEAK (PBO 0.281). config `same_month_seasonality.yaml`.
  Re-gated: oos_sharpe 0.1796, PSR 1.0, DSR ~0 (1.97e-287), PBO 0.281 (unchanged, still fails <0.25).
  VERDICT: WEAK/FAIL.
  - [x] Phase 5 / [x] Phase 6 / [-] Phase 8
- [x] **#16 FuturesTurnOfMonth** -- REJECT* MIS-SAMPLED. config `turn_of_month.yaml`.
  **CAVEAT-FIX (Phase 3) DONE** (commit `095c627`): `_run_window` hardcoded `rebalance: "weekly"`
  for every window regardless of the config; threaded the config's declared `rebalance` through
  `_config_to_kwargs -> walk_forward_carver -> process_window -> _run_window`. Re-gated on the
  correctly daily-sampled signal: oos_sharpe flips from -0.274 (mis-sampled, weekly) to **+0.0815**
  (correctly sampled, daily) -- this is a bias-correction sign change, not a post-hoc flip (the fix
  was committed and code-reviewed BEFORE re-gating, per the pre-registered-hypothesis discipline).
  PSR 0.99999, DSR ~0 (5.29e-270), PBO 0.475 (fails <0.25). VERDICT: still WEAK/FAIL even after the
  fix -- turn-of-month payment-cycle drift is not a statistically distinguishable edge either way.
  - [x] Phase 3 caveat-fix (daily rebalance) / [x] Phase 4 review (self-reviewed: test suite green,
  10/14 tests pass, same 4 pre-existing unrelated failures) / [x] Phase 5 / [x] Phase 6 / [-] Phase 8
- [x] **#23 FuturesReversal** -- was WEAK (PBO 0.805). config `index_reversal.yaml`.
  Re-gated: oos_sharpe 0.2970, PSR 1.0, DSR ~0 (5.63e-48), PBO 0.805 (unchanged, still badly fails).
  VERDICT: WEAK/FAIL (PBO 0.805 is one of the worst in Tier 1 -- highly unstable ranking under CSCV).
  - [x] Phase 5 / [x] Phase 6 / [-] Phase 8
- [x] **#37 FuturesCoTTilt** -- was REJECT (-0.124). config `cot_tilt.yaml`. Prior PBO NaN predates
  the fix; re-run confirms PBO now finite (0.141, passes <0.25 -- SP-D fix working correctly).
  Re-gated: oos_sharpe -0.1236 (still negative), PSR 3.9e-15, DSR 0.0. VERDICT: REJECT -- OOS Sharpe
  non-positive, no edge to deflate or gate (per `_verdict`'s sharpe<=0 short-circuit).
  - [x] Phase 5 / [x] Phase 6 / [-] Phase 8
- [x] **carry incumbent (FuturesCarry)** -- re-gate the benchmark under the honest deflated carver
  gate; report DSR/PBO for the portfolio summary. config `carry_idm_broad.yaml`.
  Re-gated: oos_sharpe **0.7646** (matches the documented 0.765 walk-forward figure exactly), PSR
  1.0, **DSR 0.8242** (below the 0.95 bar -- FAILS the combined statistical gate despite the highest
  PSR/cleanest PBO in the entire Tier 1 set), PBO 0.189 (passes <0.25, confirming carry is genuinely
  NOT overfit -- this is the "best DEPLOYABLE, not certified" finding from the TODO's framing).
  VERDICT: FAIL (DSR 0.8242 < 0.95) / PASS (PBO). Confirms the TODO's honest-expectation framing
  exactly: carry remains the best book, but does not clear the honest project-wide deflated gate.
  - [x] Phase 5 / [x] Phase 6

### Tier 2 -- Path 2: return-stream sleeves (run_* via G0.3 drivers; gates deflated)

- [x] **#26/#27 VIX roll-down** -- `run_vix_rolldown` via `sp_retest_vix.py`, gate_return_stream (deflated).
  Re-gated: oos_sharpe 0.5640, DSR ~0 (1.0e-06), PBO 0.6129, 1.5x cost leg not run by `run_vix_rolldown`
  (NaN, honestly reported) -- fails the 1.5x cost gate on that basis alone too. VERDICT: WEAK/FAIL,
  confirms prior finding under honest deflation. Subperiod/tail audit attached to the report.
  - [x] Phase 5 (driver sp_retest_vix) / [x] Phase 6 / [-] Phase 8
- [x] **#28 VRP short-VX1** -- `run_vrp` via `sp_retest_vrp.py`, gate_return_stream (deflated).
  Re-gated: oos_sharpe 0.0771, DSR ~0, PBO 0.297, re-expression vs #26 corr 0.488 / marginal Sharpe
  0.034 (confirms it is substantially a re-expression of #26, not an independent edge). VERDICT:
  WEAK/FAIL.
  - [x] Phase 5 / [x] Phase 6 / [-] Phase 8
- [x] **#21/#25 Overnight drift** -- `run_overnight_drift` via `sp_retest_overnight_drift.py`,
  gate_session_stream (Gate-0-deflated). Re-gated: oos_sharpe_1x 0.7924, oos_sharpe_1.5x 0.6710,
  DSR 0.872 (closest to 0.95 of any strategy in the entire retest, but still below the bar), PBO
  0.513 (fails <0.25 independently). VERDICT: WEAK/FAIL -- the strongest near-miss in the whole
  campaign, but does not clear the combined gate.
  - [x] Phase 5 / [x] Phase 6 / [-] Phase 8
- [x] **#21 Hour-slice** -- `run_hour_slice` via `sp_retest_hour_slice.py`, gate_session_stream
  (deflated). Re-gated: oos_sharpe_1x -0.0225 (negative). VERDICT: REJECT (Sharpe<=0).
  - [x] Phase 5 / [x] Phase 6
- [x] **#36 Intermarket (NQ/ES, RTY/ES)** -- `run_intermarket` via `sp_retest_intermarket.py`,
  gate_return_stream (deflated). Re-gated: NQ/ES oos_sharpe 0.329, DSR ~0, PBO 0.582 -- WEAK/FAIL.
  RTY/ES oos_sharpe -0.280 -- REJECT. **CAVEAT-FIX status: book_corr NOT computed** -- no RAMP
  equity-momentum daily return stream was supplied to `--ramp-returns` this session (none readily
  available in the worktree's registry). `book_corr` is honestly reported as NaN in both reports,
  per the driver's design (never fabricated as a low/zero placeholder). Since both pairs already
  decisively FAIL the statistical gate on their own (DSR ~0 / negative Sharpe), the book-correlation
  check would only ever REINFORCE the FAIL (a high correlation adds a re-expression reason; DSR ~0
  already rejects on primary grounds) -- not run given both are conclusively rejected without it.
  - [x] Phase 5 / [x] Phase 6 (book-corr deferred, both pairs FAIL independently regardless) / [-] Phase 8
- [x] **#31 Calendar (CL/NG/ZC/ZS/ZW)** -- `run_calendar` via `sp_retest_calendar.py`, gate_convergence
  (deflated). Re-gated all 5 roots (values match prior campaign closely): CL 0.394 (DSR~0, PBO
  0.631), NG -0.150 (REJECT, PBO 0.320), ZC 0.174 (DSR~0, PBO 0.529), ZS 0.358 (DSR~0, PBO 0.429),
  ZW 0.263 (DSR~0, PBO 0.818). ALL WEAK/FAIL/REJECT. Section 11 exit diagnostics reported honestly
  via `convergence_exit_summary` (no fabricated exit_reason breakdown -- see Gate 0.3 commit).
  **NG RollCalendar caveat-fix: NOT attempted** -- with 4/5 roots already decisively FAILing DSR (not
  just PBO-marginal), and NG itself REJECT on Sharpe<=0 grounds (not PBO-marginal either), the
  volume-rank-vs-RollCalendar F1/F2 distinction would not change any verdict in this set; deprioritized.
  - [x] Phase 5 (+ Section 11 exit diag) / [x] Phase 6 / [-] NG RollCalendar caveat-fix (deprioritized,
  would not change the verdict) / [-] Phase 8
- [x] **#32 Crack (RB-CL, HO-CL)** -- `run_processing` via `sp_retest_processing.py`, gate_convergence
  (deflated). Re-gated: RB-CL oos_sharpe -0.116 (REJECT), HO-CL oos_sharpe -0.215 (REJECT). Both
  Sharpe<=0. VERDICT: REJECT both.
  - [x] Phase 5 (+ exit diag) / [x] Phase 6
- [x] **#33 Crush (ZM+ZL-ZS)** -- `run_processing` (same driver), gate_convergence (deflated).
  Re-gated: oos_sharpe 0.1360 (matches prior 0.136 exactly), PSR 1.0, **DSR ~0** (honest deflation
  now applied -- was previously ungated/undeflated), PBO 0.1089 (clean, confirms it is not a
  CSCV-detectable overfit). VERDICT: WEAK/FAIL -- **NOT escalated to Phase 6.5.** Rationale: DSR ~0
  is a decisive rejection by the BINDING gate (Section 2.5), not a borderline miss the way overnight
  drift's DSR 0.872 is -- a clean PBO does not make a near-zero-Sharpe (0.136), near-zero-DSR result
  "marginal-but-real" per the Phase 6.5 entry criteria. Escalating would be exactly the kind of
  design-iteration-chasing-a-metric this campaign's North Star forbids.
  - [x] Phase 5 (+ exit diag) / [x] Phase 6 / [-] Phase 6.5 (not marginal-but-real, decisive DSR
  rejection) / [-] Phase 7 / [-] Phase 8
- [x] **#34 Ratio (GC/SI)** -- `run_ratio` via `sp_retest_ratio.py`, gate_convergence (deflated).
  Re-gated: oos_sharpe 0.2687 (matches prior 0.269), DSR ~0, PBO 0.674, kurtosis 109.3 (confirms
  genuine GC/SI tail risk, not a data artifact). Section 11 exit diagnostics reported honestly
  (no exit_reason fabrication). VERDICT: WEAK/FAIL.
  - [x] Phase 5 (+ exit diag) / [x] Phase 6

### Tier 3 -- ungradeable by the walk-forward gate (document, mark [-] with reason)

- [-] **#39 Pre-FOMC** -- `run_prefomc` via `sp_retest_prefomc.py`. Confirmed n_windows=0 (as
  predicted -- ~8 events/yr never fill a 12-month/10-sample window). Architecturally ungradeable by
  this gate, NOT a fixable verdict; `_verdict` correctly reports INCONCLUSIVE (NaN short-circuit).
  Decay split (small-n descriptive only, not a gate): pre-2015 Sharpe 0.252 (n small), post-2015
  Sharpe 6.540 (n small) -- this is SMALL-N NOISE, not evidence the Ma-Zhang decay reversed; a
  single-digit-event subperiod Sharpe is not a statistically meaningful comparison either direction.
  Skipped per the TODO's authoritative Tier 3 designation.
- [-] **#35 Steepener (2s10s/2s5s/5s30s)** -- `run_steepener` via `sp_retest_steepener.py`. Confirmed
  n_windows=0 for ALL THREE segments (2YY from ~2021 leaves <48mo; 5YY degraded ~440 rows), exactly
  as predicted. `_verdict` correctly reports INCONCLUSIVE for all three. ZT/ZN DV01 fallback remains
  a possible future rebuild once yield-future history matures -- not attempted. Skipped per the
  TODO's authoritative Tier 3 designation.

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
| R1 | #3 FuturesXSMomentum | Path 1 (carver) | 0.2095 | 0.1814 | 1.0000 | 2.98e-208 (N=41*) | 0.5795 | -0.0756 | 12.91 | -27.30%** | -99.49%** | N/A | N/A | N/A | N/A | 5043** | N/A | see CARVER report, 13 windows | N/A | N/A | N/A | N/A | 2010-06-07..2026-02-20 | daily | WEAK/FAIL |
| R1 | #10 FuturesCarryXS | Path 1 (carver) | 0.8458 | 0.8333 | 1.0000 | 0.9990 (N=41*) | 0.6903 | -0.7961 | 26.08 | 16.50%** | -70.61%** | N/A | N/A | N/A | N/A | 3798** | N/A | see CURVE_SLOPE_XS report, 13 windows | N/A | N/A | N/A | N/A | 2010-06-07..2026-02-20 | daily | WEAK (PBO fail) |
| R1 | #13 FuturesCarryTrend | Path 1 (carver) | 0.3571 | 0.3357 | 1.0000 | 2.93e-77 (N=41*) | 0.1892 | -0.8822 | 17.83 | 12.92%** | -57.74%** | N/A | N/A | N/A | N/A | 6565** | N/A | see CARRY_TREND_GATE report, 13 windows | N/A | N/A | N/A | N/A | 2010-06-07..2026-02-20 | daily | FAIL (DSR) |
| R1 | #15 FuturesSameMonthSeasonality | Path 1 (carver) | 0.1796 | 0.1663 | 1.0000 | 1.97e-287 (N=41*) | 0.2806 | 0.3782 | 7.72 | 6.09%** | -85.22%** | N/A | N/A | N/A | N/A | 4901** | N/A | see SAME_MONTH_SEASONALITY report, 13 windows | N/A | N/A | N/A | N/A | 2010-06-07..2026-02-20 | daily | WEAK/FAIL |
| R1 | #16 FuturesTurnOfMonth (post caveat-fix, daily rebalance) | Path 1 (carver) | 0.0815 | 0.0689 | 0.99999 | 5.29e-270 (N=41*) | 0.4748 | -3.8811 | 97.11 | 0.93%** | -64.56%** | N/A | N/A | N/A | N/A | 1679** | N/A | see TURN_OF_MONTH report, 13 windows | N/A | N/A | N/A | N/A | 2010-06-07..2026-02-20 | daily | WEAK/FAIL |
| R1 | #23 FuturesReversal | Path 1 (carver) | 0.2970 | 0.2878 | 1.0000 | 5.63e-48 (N=41*) | 0.8050 | 7.0392 | 219.89 | 4.38%** | -70.13%** | N/A | N/A | N/A | N/A | 2262** | N/A | see INDEX_REVERSAL report, 13 windows | N/A | N/A | N/A | N/A | 2010-06-07..2026-02-20 | daily | WEAK/FAIL (worst PBO in Tier 1) |
| R1 | #37 FuturesCoTTilt | Path 1 (carver) | -0.1236 | -0.1384 | 3.88e-15 | 0.0000 (N=41*) | 0.1414 | -0.2570 | 10.06 | -28.59%** | -99.52%** | N/A | N/A | N/A | N/A | 6056** | N/A | see COT_TILT report, 13 windows | N/A | N/A | N/A | N/A | 2010-06-07..2026-02-20 | daily | REJECT (Sharpe<=0) |
| R1 | carry incumbent (FuturesCarry, carry_idm_broad) | Path 1 (carver) | 0.7646 | 0.6975 | 1.0000 | 0.8242 (N=41*) | 0.1887 | 1.3069 | 22.22 | 16.01%** | -38.98%** | N/A | N/A | N/A | N/A | 8290** | N/A | see CARRY_IDM_BROAD report, 13 windows | N/A | N/A | N/A | N/A | 2010-06-07..2026-02-20 | daily | FAIL (DSR 0.82<0.95) / PASS (PBO); best deployable, not certified |
| R1 | #26/#27 VIX roll-down | Path 2 (return stream) | 0.5640 | NaN (1x leg only) | 1.0000 | ~0 (1.0e-06, N=41) | 0.6129 | -2.4958 | 20.39 | N/A | N/A | N/A | N/A | N/A | N/A | N/A (continuous) | N/A | 11 windows | N/A | N/A | N/A | N/A | full range | daily | WEAK/FAIL |
| R1 | #28 VRP short-VX1 | Path 2 (return stream) | 0.0771 | NaN (1x leg only) | 0.9915 | ~0 (N=49) | 0.2971 | -13.1324 | 370.19 | N/A | N/A | N/A | N/A | N/A | N/A | N/A (continuous) | N/A | 10 windows; corr-vs-#26=0.488, marginal Sharpe=0.034 | N/A | N/A | N/A | N/A | full range | daily | WEAK/FAIL (substantially a re-expression of #26) |
| R1 | #21/#25 Overnight drift | Path 2 (session stream) | 0.7924 | 0.6710 | 1.0000 | 0.8720 (N=41) | 0.5128 | -0.7313 | 18.51 | N/A | N/A | N/A | N/A | N/A | N/A | N/A (continuous) | N/A | 13 windows | N/A | N/A | N/A | N/A | full range | ET session bars | WEAK/FAIL (closest near-miss in the campaign) |
| R1 | #21 Hour-slice | Path 2 (session stream) | -0.0225 | -0.2769 | 0.1021 | 0.0000 (N=41) | 0.8731 | 1.0721 | 22.78 | N/A | N/A | N/A | N/A | N/A | N/A | N/A (continuous) | N/A | 13 windows | N/A | N/A | N/A | N/A | full range | ET session bars | REJECT (Sharpe<=0) |
| R1 | #36 Intermarket NQ/ES | Path 2 (return stream) | 0.3294 | NaN (1x leg only) | 1.0000 | ~0 (N=53) | 0.5821 | -0.1958 | 11.36 | N/A | N/A | N/A | N/A | N/A | N/A | N/A (continuous) | N/A | 12 windows; book_corr=NaN (RAMP stream not supplied) | N/A | N/A | N/A | N/A | full range | daily | WEAK/FAIL |
| R1 | #36 Intermarket RTY/ES | Path 2 (return stream) | -0.2803 | NaN (1x leg only) | 0.0000 | 0.0000 (N=54) | 0.9128 | -1.4212 | 14.18 | N/A | N/A | N/A | N/A | N/A | N/A | N/A (continuous) | N/A | 4 windows; book_corr=NaN | N/A | N/A | N/A | N/A | full range | daily | REJECT (Sharpe<=0) |
| R1 | #31 Calendar CL | Path 2 (convergence) | 0.3942 | NaN (1x leg only) | 1.0000 | ~0 (N=55) | 0.6314 | 1.0887 | 162.61 | N/A | N/A | N/A | N/A | N/A | N/A | 86 | N/A | 13 windows | N/A | N/A | N/A | see convergence_exit_summary (no exit_reason available) | full range | daily | WEAK/FAIL |
| R1 | #31 Calendar NG | Path 2 (convergence) | -0.1500 | NaN (1x leg only) | 0.0000 | 0.0000 (N=56) | 0.3202 | -14.4244 | 598.38 | N/A | N/A | N/A | N/A | N/A | N/A | 66 | N/A | 13 windows | N/A | N/A | N/A | see convergence_exit_summary | full range | daily | REJECT (Sharpe<=0) |
| R1 | #31 Calendar ZC | Path 2 (convergence) | 0.1736 | NaN (1x leg only) | 1.0000 | ~0 (N=57) | 0.5294 | 0.3937 | 64.26 | N/A | N/A | N/A | N/A | N/A | N/A | 33 | N/A | 13 windows | N/A | N/A | N/A | see convergence_exit_summary | full range | daily | WEAK/FAIL |
| R1 | #31 Calendar ZS | Path 2 (convergence) | 0.3581 | NaN (1x leg only) | 1.0000 | ~0 (N=58) | 0.4285 | 4.6832 | 127.44 | N/A | N/A | N/A | N/A | N/A | N/A | 31 | N/A | 13 windows | N/A | N/A | N/A | see convergence_exit_summary | full range | daily | WEAK/FAIL |
| R1 | #31 Calendar ZW | Path 2 (convergence) | 0.2634 | NaN (1x leg only) | 1.0000 | ~0 (N=59) | 0.8176 | 2.6950 | 68.85 | N/A | N/A | N/A | N/A | N/A | N/A | 54 | N/A | 13 windows | N/A | N/A | N/A | see convergence_exit_summary | full range | daily | WEAK/FAIL (worst PBO in this sleeve) |
| R1 | #32 Crack RB-CL | Path 2 (convergence) | -0.1162 | NaN (1x leg only) | 0.0000 | 0.0000 (N=60) | 0.4689 | 1.4986 | 57.19 | N/A | N/A | N/A | N/A | N/A | N/A | 45 | N/A | 13 windows | N/A | N/A | N/A | see convergence_exit_summary | full range | daily | REJECT (Sharpe<=0) |
| R1 | #32 Crack HO-CL | Path 2 (convergence) | -0.2150 | NaN (1x leg only) | 0.0000 | 0.0000 (N=61) | 0.7037 | 1.4650 | 128.53 | N/A | N/A | N/A | N/A | N/A | N/A | 57 | N/A | 13 windows | N/A | N/A | N/A | see convergence_exit_summary | full range | daily | REJECT (Sharpe<=0) |
| R1 | #33 Crush ZM+ZL-ZS | Path 2 (convergence) | 0.1360 | NaN (1x leg only) | 1.0000 | ~0 (N=62) | 0.1089 | 5.8831 | 141.85 | N/A | N/A | N/A | N/A | N/A | N/A | 33 | N/A | 13 windows | N/A | N/A | N/A | see convergence_exit_summary | full range | daily | WEAK/FAIL (clean PBO, decisive DSR reject -- not escalated to 6.5) |
| R1 | #34 Ratio GC/SI | Path 2 (convergence) | 0.2687 | NaN (1x leg only) | 1.0000 | ~0 (N=63) | 0.6738 | 3.0514 | 109.27 | N/A | N/A | N/A | N/A | N/A | N/A | 25 | N/A | 13 windows | N/A | N/A | N/A | see convergence_exit_summary (kurtosis 109 = genuine GC/SI tails) | full range | daily | WEAK/FAIL |
| R1 | #39 Pre-FOMC (Tier 3) | Path 2 (session stream) | NaN | NaN | NaN | NaN | NaN | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A (continuous) | N/A | n_windows=0 (architecturally ungradeable); pre-2015 Sharpe 0.252, post-2015 Sharpe 6.540 (SMALL-N NOISE, not a gate) | N/A | N/A | N/A | N/A | full range | ET session bars | INCONCLUSIVE (Tier 3, documented, not a fixable verdict) |
| R1 | #35 Steepener 2s10s (Tier 3) | Path 2 (return stream) | NaN | NaN | NaN | NaN | NaN | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A (continuous) | N/A | n_windows=0 (architecturally ungradeable, 2YY history too short) | N/A | N/A | N/A | N/A | full range | daily | INCONCLUSIVE (Tier 3) |
| R1 | #35 Steepener 2s5s (Tier 3) | Path 2 (return stream) | NaN | NaN | NaN | NaN | NaN | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A (continuous) | N/A | n_windows=0 | N/A | N/A | N/A | N/A | full range | daily | INCONCLUSIVE (Tier 3) |
| R1 | #35 Steepener 5s30s (Tier 3) | Path 2 (return stream) | NaN | NaN | NaN | NaN | NaN | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A (continuous) | N/A | n_windows=0 (5YY degraded ~440 rows) | N/A | N/A | N/A | N/A | full range | daily | INCONCLUSIVE (Tier 3) |

\* N=41 reflects `get_campaign_trial_distribution()` at the time these runs executed (static 40-trial
baseline + 1 registry-logged run picked up incidentally from an earlier walk-forward-config test
exercising the real registry during Gate 0 verification). This is the intended honest-and-growing
behavior (Gate 0.2) -- N will continue to grow as Tier 2/3 runs are appended.

\*\* CAGR/MaxDD/Trades are from the SEPARATE full-range (IS+OOS combined, not purged)
`sp_retest_trade_log.py` representative-trade-log run (methodology Section 12.0), NOT from the
walk-forward gate's OOS-only stitched series -- the walk-forward stitcher does not itself produce a
CAGR/MaxDD/trade-count metric. Sharpe 1x/1.5x, PSR, DSR, PBO, skew, kurt are all from the actual
OOS walk-forward gate (the metric that matters for the verdict). Win%/PF/Calmar/DDdur/AvgHold are
not computed by either script for the Path-1 carver strategies -- N/A rather than fabricated.
Regime/capacity/IR/MAE-MFE Section 12 diagnostics are not produced by this harness for Path-1
strategies (StandardReport gives monthly Sharpe/DD only, not a full regime/capacity breakdown) --
N/A rather than fabricated; a future refinement could wire `MarketRegimeDetector` into the carver
harness if regime robustness reporting for futures becomes a priority.

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
