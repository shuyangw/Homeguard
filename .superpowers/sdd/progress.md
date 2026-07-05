# Futures Backtest Harness — SDD Progress Ledger

Branch: feat/futures-backtest-harness (off main)
Merge-base (branch start): 6c1a4fc (main; roll-calendar work already merged+pushed)
Plan: docs/strategies/research/20260701_FUTURES_BACKTEST_HARNESS_PLAN.md
Base for Task 1: e2d7dbd (after spec+plan commits)

(Prior roll-calendar SDD ledger retired -- that work is merged to main + pushed, recoverable via git log.)

## Tasks
- [x] Task 1: extend cost model 9 -> 53 roots (commits e2d7dbd..6b371ef; +fix 6b371ef collapsed to single commission source; 9 originals byte-identical, 25/25)
- [x] Task 2: add margin fields (commit d3f5a81, review clean; maintenance=round(0.9*initial) structurally guaranteed, 37 pass)
- [x] Task 3: MarginModel (commits d3f5a81..9b223b7 + fix ff1f663: restored default-on offsets, non-degenerate tests; 6/6)
- [x] Task 4: forecast sizer (commit f5a6a32, review clean; +max_contracts=100 field on contract_specs, both suites 7/7+8/8)
- [x] Task 5: FuturesBacktestLoader (commit 0221b40, review clean; real ES/GC panel (77,4); .gitignore un-ignored src/backtesting/data + tests/backtesting/data narrowly)
- [x] Task 6: Carver EWMAC indicators (commit e9fc85d + fix 46a2b67: np.nan not pd.NA for zero-vol dtype; 5/5)
- [x] Task 7: CarverMomentumStrategy (commit 8042309, review clean; np.nan carried in, returns-not-prices verified, base wiring OK)
- [x] Task 8: FuturesPortfolioSimulator (commit 8bd8aaf, review clean; MTM+cost hand-traced exact, cost-convention resolved, isolated from equity sim)
- [x] Task 9: runner integration + e2e (commit fedf991 + fix 304ea19 canonical vol; harness WORKS -- Carver MES/MGC/6E 2022-23 Sharpe 0.255; routing guard proven no-op for equity path)
- [x] Task 10: walk-forward + gate (commit da6ad5e, review APPROVED; Carver REJECT OOS Sharpe -0.45; no shared-loader regression 81+golden pass; +4 loader fixes)

## Notes
- Minor findings roll-up (for final review):
  - T2 Minor: margin values approximate (SR1/SR3, micro-yield 10Y/30Y/5YY/2YY low-confidence placeholders); no source-citation URL
  - T4 Minor: uniform max_contracts=100 across 53 roots -> make per-root before real use; daily_vol docstring caveat
  - T5 Minor: redundant panel.columns reassignment (could diverge from concat); missing docstring on outer-join/no-ffill alignment
  - T7 Minor: forecast_panel silently skips missing root (could raise); universe stored twice (attrs + params)
  - T8 Minor: weekly ISO-week year-boundary rebalance edge case (WILL be exercised by Task 9 weekly config); int() truncation of fractional targets; no close/target column-set validation
  - T9 Minor: redundant yaml parse for asset_class peek; run_futures_backtest >30 lines; check_and_scale computed every day (sim uses weekly); pct_change FutureWarning
  - NOTE: 3 pre-existing failures in test_rolling_mode.py (missing local AAPL 2024-01 parquet) -- environmental, NOT branch-introduced (reviewer confirmed via traceback in untouched files)

## ALL 10 TASKS COMPLETE
- T10 KEY FINDING (top pre-merge decision): FuturesPortfolioSimulator lets equity go NEGATIVE (no bankruptcy floor); pct_change on zero-crossable equity -> exploded stats (skew -30.5/kurt 1332, 1.5x>1x Sharpe inversion). REJECT robust, but ANY strategy drawing account near zero gets contaminated stats. Fix location: simulator equity floor OR return-computation handling of non-positive equity.
- T10 Minor: append_run non-fatal wrap (mild methodology 9.3 conflict, matches T9); redundant <2 window guards.
- Carver TSMOM verdict: REJECT (parameter-free, honest first result; a valid finding, not a harness failure).

## POST-ACCEPTANCE FIX (option 1: equity-feedback + bankruptcy floor)
- Task 11: equity-feedback sizing + bankruptcy floor (commit 0d4c4de + fix cf72250 for cost-driven floor). Sim sizes vs LIVE equity per rebalance; equity floored at 0 after BOTH MTM and cost. 5/5 sim tests (Task-8 25500/24994 preserved), e2e passes. New e2e Sharpe -0.399 (live-equity sizing changed the path; expected).
- CONFIRMED: walk-forward re-run CLEAN -- skew -30.5->-0.39, kurt 1332->8.7, 1.5x Sharpe now correctly < 1x. Carver OOS Sharpe 0.11, PBO 0.44 -> WEAK/does-not-clear-gate (trustworthy finding). Anomaly RESOLVED. Readiness report regenerated.
- NEXT: final whole-branch review + merge.
## Broad-Basket Carver (option A) -- SDD ledger
- Task 1: complete (commit 2548e03, config 33 roots, test 1/1, controller-verified data file)
- Task 2: complete (commit fe726c8, config-driven walk-forward + basket-accurate report, unit 2/2, smoke n_windows=2 sharpe=0.2272, default preserved, review Approved)
- Task 3: complete (controller-run; broad walk-forward WEAK oos_sharpe=0.084 pbo=0.35 n_windows=13; commits ad0c5d4 generator-fix+test, 1129488 report; baseline untouched)

## Futures Strategy Registry (option B) -- SDD ledger
- Task 1: complete (commit f6d6b1e, Carver registered + 3 aliases, test 2/2, controller-verified additive)
- Task 2: complete (commit 1165ebd + 41 test-coverage add; pluggable resolution 5/5, e2e unchanged, dead import gone, review Approved)

## Futures Carry Strategy (option C) -- SDD ledger
- Task 1: complete (c43ac48, carry_dir path helper, 1/1, controller-verified)
- Task 2: complete (0a3013b, asset_class map, 3/3, controller-verified)
- Task 3: complete (43fe15b, carry cache builder, 1/1, controller-verified)
- Task 4: complete (0ddde11, FuturesCarryStrategy + registry, 4/4, review Approved; normalization math verified)
- Task 5: complete (fc46da0 + append_run label fix; walk-forward strategy-agnostic, 5/5, review Approved)
- Task 6: complete (6763d48, carry_broad config, 1/1, controller-verified)
- Preflight: FuturesCarry chain verified end-to-end on real data (GC/CL Q1-2022, took positions, report+registry labeled FuturesCarry). Launching full Task 7.
- Task 7: complete (controller-run; carry cache 33/33 parquets built 8-way parallel; walk-forward OOS Sharpe 0.88 PBO 0.63 -> WEAK/concentrated; report + note-generalization committed; baselines untouched)

## Parallel Execution Foundation (P) -- SDD ledger
- Task 1: complete (ddf018e, parallel_map primitive, 4/4, spawn verified 4 PIDs, review Approved; input-order determinism correct)
- Task 2: complete (1d52ae6, register flag on run_futures_backtest, 8 passed [2 new + 6 existing], controller-verified)
- Task 3: complete (0f0fbbd, walk-forward parallel by window, determinism PROVEN parallel==serial byte-identical, 6/6, review Approved; determinism+race-freedom structural)
- Task 4: complete (3de106a, build_carry_cache --jobs via parallel_map, 2/2, controller-verified; real parallel proven earlier in the 33-root build)

## Bond Carry -- SDD ledger
- Task 1: complete (3577362, FRED point-in-time reader, 4/4, controller-verified causal)
- Task 2: complete (a6dd206, bond carry from FRED CMT-DFF, 4/4 new + 6/6 existing; controller-verified: all 6 bonds nonzero, sign+duration-monotonic correct on real data, updated test legitimate)

## Carry De-Concentration (XS + IDM) -- SDD ledger
Branch: feat/carry-deconcentration (off main @ 0640a5f)
Merge-base: 0640a5f
Plan: docs/strategies/research/20260703_CARRY_DECONCENTRATION_PLAN.md
Base for Task 1: e74d44f (after design+plan commits)
Preflight: CLEAN -- parent _load_carry exists (l27), forecast_panel calls it (l49); asset_class_for exists; cluster_for is new (T1). No task contradictions.
- Task 1: complete (commit 8c3974e, review PASS+Approved; 3 passed; 45-symbol CLUSTER map + cluster_for KeyError-on-unmapped). Minor: cluster_for lacks docstring (cosmetic, matches brief snippet) -> roll-up.
- Task 2: complete (commit 044fb13, review PASS+Approved; 2 passed + 914 sibling no-regress; causality=same-day-only CONFIRMED, NaN-contract preserved, scope 3 files). Deviation from brief literal Step-3 (broken: singleton std=NaN, cap-saturated std=0) -> impl skips singletons + zeroes valid-zero-dispersion rows, keeps NaN for missing/warmup. Reviewer ruled deviation legitimate+necessary. Minor: "empty class" docstring wording imprecise; zero-dispersion->0 implicit design choice (sign off w/ strategy-lead before extending XS) -> roll-up.
- Task 3: complete (commit 522e36d, review PASS+Approved zero findings; 2 passed; w & C share one root ordering CONFIRMED, sum(w)=1, IDM cap 2.5 exercised, median pinned to 1.0; data-free/deterministic/parameter-free verified).
- Task 4: complete (commit 96dfe9b, review PASS+Approved zero findings; 8 passed [5 pluggable+1 e2e+2 new]; scalar div_mult path byte-identical, backtest.idm defaults False so existing configs unaffected, compute_div_mult called with correct traded universe; scope 2 engine files + new test; isolation preserved).
- Task 5: complete (commit 0f24d2f, review PASS+Approved zero findings; 3 passed; all 3 configs 33 roots correct order, idm/name correct per variant, no baseline divergence, real YAML-loading test). ALL CODE TASKS (1-5) DONE+REVIEWED.
- Task 5b (FIX, controller-caught in preflight): thread backtest.idm through walk-forward driver (commit 0dfbfb8, review PASS+Approved; 4 passed). Bug: _config_to_kwargs + _run_window rebuilt per-window config and DROPPED idm -> IDM-alone & XS+IDM trials would silently run WITHOUT IDM. Fixed all 5 links (config_to_kwargs->main->walk_forward_carver->specs->process_window->_run_window->backtest.idm), both cost legs. Minor: one default-false test uses non-sentinel capture default (compensated) -> roll-up.
- Task 6 (controller-run, one-trial-per-bg-job after 60min-cap kill of chained run):
  - XS-alone: DONE. WEAK. OOS Sharpe 0.7665 (1.5x 0.7345), PBO 0.4636 (WORSE than 0.33 baseline), kurt 18.3, skew -0.70. Signal-side lever backfired (relative-value carry less window-stable).
  - IDM-alone: DONE. **PASS** (clears combined + 1.5x cost gate). OOS Sharpe 0.7646 (1.5x 0.6975), PBO 0.1887 (<0.25 gate, from 0.33), skew +1.31, kurt 22.2. Sizing-side cluster de-concentration works -> carry's FIRST gate-clear.
  - XS+IDM: RUNNING (bg job b48s7tmm4).
  - NOTE: --json sidecar didn't write for idm run (readiness .md authoritative). Kill cause diagnosed = ~60min bg cap (memory saved).
  - XS+IDM: DONE. WEAK (worst). OOS Sharpe 0.7662 (1.5x 0.7389), PBO 0.5274, skew -0.56, kurt 14.4. XS POISONS IDM (0.19 -> 0.53); levers opposed, not additive.
- Task 6 COMPLETE. WINNER: IDM-alone (PBO 0.19 PASS). Conclusion: de-concentrate at SIZING layer (IDM), never signal (XS). Caveat: best-of-3 pre-committed, DSR n_trials=1; PBO 0.19 has margin under 0.25. NEXT: final whole-branch review -> merge decision -> session log.
- FINAL whole-branch review (opus): MERGE-READY. 0 Critical, 0 Important. Integration continuity (idm flag end-to-end both cost legs), causality (no lookahead), IDM math (w'Cw same-order), back-compat/isolation all PASS. Minors: (1) no direct test of single-shot run_futures_backtest idm branch; (2) back-compat test only asserts len not value-identity; (3) design book-vol sanity check not impl (median-pin proxy); (4) INTERP: IDM concentrates per-instrument risk into small clusters (LE/HE 2.5x) -> kurt can migrate not shrink (matches IDM kurt 22.2>21); (5) singleton asset-class XS->0.0 not NaN (no broad-universe impact). NO fix wave (all Minor). NEXT: session log + merge decision (user).
- Test-gaps (Minor 1+2 closed): commit fbefab3, test-only (no source), inline review Approved. 7 passed no skips. Wrap-through spy on run_sized proves idm:true->dict / absent|false->scalar 1.0; scalar==all-ones-dict identity. NEXT: merge main + push (user authorized).
- MERGED + PUSHED: fast-forward 0640a5f..fbefab3, main -> origin/main confirmed (ls-remote + fetch aligned; trailing update_ref error was cosmetic local tracking-ref lock under Dropbox). Feature surface 25 passed pre-merge. CAMPAIGN COMPLETE.

## Futures Sharpe-Uplift Campaign (target >1) -- SDD ledger
Branch: feat/futures-sharpe-uplift (off main @ fbefab3)
Charter: docs/strategies/research/20260704_FUTURES_SHARPE_UPLIFT_CAMPAIGN.md (commits 6f4bea3, 0cbd5f4)
- Trend diagnosis: -0.45 was STALE pre-fix (equity-explosion bug already fixed); current trend 0.11 WEAK, correctly signed, NOT broken. Trend demoted to crisis-insurance.
- Data scoping: skew BLOCKED (options=ES/NQ only), breadth REDUNDANT (14 addables mostly micros), VALUE the only buildable second pillar (price-only, 47 roots). Revised ceiling ~0.85-0.95; >1 unlikely.
- EXECUTION REORDER: front-load the decisive experiment -- build VALUE standalone, walk-forward it, measure carry-correlation. Gates whether a real 2nd pillar exists BEFORE building Phase-0 combiner plumbing.
- Value construction: Asness 5yr-to-1yr reversal (user-chosen), -(lp.shift(252)-lp.shift(1260)), vol-normalized, fixed scalar. Sharpe+corr are scale-invariant so scalar immaterial to verdict.
- Value signal build: IN PROGRESS.
- [Overnight Phase1] Task 1 crypto carry branch: DONE (4f2a1cc, review clean inline, 5 passed). value WF (bx1cdo5k7) RUNNING. Next: T3 configs, T4 corr tool.
- [Overnight Phase1] Task 3 configs: DONE (7eb100f, 2 passed). Task 4 corr tool: DONE (5ff47e9, 2 passed; adapted _run to read date-indexed equity.csv via log_trades since run_futures_backtest returns no dates key; guard <2). ALL PHASE-1 BUILDS DONE. T2 cache/T6 crypto-WF/T7 corr queued behind value WF (thread cap). value WF (bx1cdo5k7) still RUNNING.
- [Overnight Phase1] CRYPTO PASS (0.61/PBO0.24); rho(crypto,carry)=-0.065 -> FULL-WEIGHT INCLUDE; bound 1.007. VALUE excluded (-0.22). Combination carry_idm_crypto (35-root) WF b4mv003nu KILLED at 15.8min (not 60min cap, abrupt external, no result) -> relaunched bu4ak6bl4.
- [Overnight] Combination CLI (carry_idm_crypto 35-root) KILLED TWICE at exactly ~15.8min, same code point (right after STANDARD_BACKTEST report CSV save, before readiness/registry). REPRODUCIBLE crypto-specific hang in the CLI report phase (33-root carry_idm + 2-root crypto both completed fine). No 35-root registry/report/json. -> computing gate DIRECTLY via walk_forward_carver (bypasses report phase, bbkv4pg71). If it completes -> have combined gate + report-phase bug to fix. If it hangs ~15min -> hang is in walk_forward_carver core.
- [Overnight] ROOT CAUSE of combination crash: PARALLEL-ONLY OOM, not window logic. Serial in-process run did ALL 13 windows OK (crypto=True windows included). BrokenProcessPool = a worker OOM-killed when 8 concurrent workers each load crypto's large 1min data (BTC 2017+/ETH 2021+) on top of 33 macro roots. 33-root fit at jobs=8; +crypto tips over. FIX-FOR-NOW: max_workers=1 (serial, OOM-safe) to get the combined gate (buyidiqrr). REAL FIX (later): cap crypto worker memory / cache daily panels / reduce --jobs for crypto-inclusive runs.

## Daily-Panel Cache (A1 / OOM fix) -- SDD ledger  [branch feat/futures-daily-panel-cache off main 688dff7]
- Task 1: daily_raw_dir + build_daily_raw_cache (c94efc7, ES cache == aggregate_to_daily raw, 1 passed).
- Task 2: ratio_adjust_daily -- EQUIVALENCE EXACT max abs diff 0.0 ES/CL full+sub-window (a0bfd47, 4/4).
- Task 3: aggregate_to_daily reads daily-raw cache when present, fallback intact (89282dd, 2 new + 7/7 regression).
- Task 4 (controller): building cache all 35 roots (binj1ykz2, jobs 4). NEXT: carry_idm re-run must reproduce 0.7646/PBO0.1887 byte-identical (GATE); re-measure RSS (expect sub-GB from 5.6GB); 35-root crypto combo at jobs 8 must run without OOM.
- Task 5 (completing fix): persist per-year roll-volume to disk cache (acd0db7, 9 passed; roll dates BYTE-IDENTICAL cache vs 1min). CAUGHT: A1 daily-close cache alone was correct but incomplete -- detect_roll_dates -> _year_daily_symbol_volume loads a SECOND 1min stream. Now BOTH cached. Concern: no cache invalidation (matches daily_raw pattern; 1min data is static).
- Task 6 (controller): building roll-volume cache all 35 roots (bjgxvy9ib jobs4). THEN re-run carry_verify (both caches -> expect fast + jobs=8 no OOM + 0.7646 byte-identical); mem_probe; 35-root crypto combo jobs=8.
- A1 ACCEPTANCE GATE PASSED: carry_idm via both caches, jobs=8 -> Sharpe 0.7646484/PBO 0.188656 BYTE-IDENTICAL (MATCH=True) in 17s (was ~47min, ~165x faster). Per-window RSS 5.6GB->0.33GB (mem_probe); 8-way 45GB->2.6GB. OOM eliminated. NOTE: my direct verify scripts' jobs=8 BrokenProcessPool was a MISSING __main__ guard (spawn bomb), not OOM -- fixed guard, jobs=8 works. Final proof: 35-root crypto combo via CLI (has guard) at jobs=8 should now complete (was 15.8min OOM).
- A1 COMPLETE + VALIDATED: 35-root carry+crypto via CLI jobs=8 -> 0.4217/0.1019 BYTE-IDENTICAL to serial, 18s (was 15.8min OOM), report+json written. BOTH bugs (parallel OOM + ~15min CLI kill) were memory-driven -> RESOLVED by daily-raw + roll-volume caches. carry_idm 0.7646 byte-identical 17s. Branch feat/futures-daily-panel-cache (Tasks 1-5) ready for review+merge.

## Spot FX Backtesting Platform -- SDD ledger  [branch feat/spot-fx-backtest off main 154245a]
Plan: docs/plans/2026-07-05-spot-fx-backtest-implementation-plan.md
Spec: docs/plans/2026-07-05-spot-fx-backtest-design.md
Base for Task 1: 154245a
- Task 10 DECISION CHANGE (user): do NOT import helpers; EXTRACT to src/backtesting/walkforward_common.py, both futures+FX WF import it. run_carver_walkforward.py modified; test_futures_walkforward.py is the regression gate.
- Task 1: implemented (commit defa58b, 1/1) -- pending review.
- Task 1: COMPLETE (commits defa58b..209b2e7, review Approved after 1 fix wave: FX-date filter bug, DST docstring+test, empty-return contract). Base for Task 2: 209b2e7
- Task 2: COMPLETE (commits bd21983..c39869b, review Approved after 1 fix wave: hardened load_fx_daily_panel no-data exclusion + 3 tests; 7 passed). 
- SCOPE DECISION (user, 2026-07-05): narrow v1 universe to carry-covered currencies USD/EUR/CHF/JPY + metals: EURUSD,USDJPY,USDCHF,EURJPY,EURCHF,CHFJPY,XAUUSD,XAGUSD. GBP/CAD/AUD/NZD deferred (no FRED short rate on disk). CURRENCY_FRED_SERIES restricted to DFF/ECBDFR/IRSTCB01CHM156N/IRLTLT01JPM156N(proxy). ENV: this Mac has no [macos] storage section -> Task 11 real runs must execute on EC2/Windows. Tasks 3-10 buildable+testable here with fixtures.
