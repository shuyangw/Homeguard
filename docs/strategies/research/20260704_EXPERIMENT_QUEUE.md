# Autonomous Experiment Queue - Futures Sharpe-Uplift (2026-07-04, ~6hr unattended)

**READ THIS FIRST if you are a resumed/compacted context.** This drives a self-sustaining
pipeline: each experiment is ONE background walk-forward (~47 min, own bg job, under the
~60-min harness cap). On each completion/killed notification, run the CONTROLLER STEP below.

Branch: `feat/futures-sharpe-uplift`. User is AWAY -- make ALL decisions from the
pre-committed criteria; do NOT ask questions; do NOT merge or push. Commit code per
experiment. Update this file's status table after every experiment.

## CONTROLLER STEP (do this on every invocation)
1. Check the active run: newest `output/run_status/*.json`. If status RUNNING and heartbeat
   fresh (< ~5 min old) -> a run is alive; do nothing but optionally build the NEXT
   experiment's code (aggressive mode). If RUNNING but heartbeat stale (> ~10 min) -> it was
   killed; treat as done-with-whatever-output-exists.
2. If no run active: read the just-finished experiment's `--json` / readiness report, record
   metrics in the status table + registry, apply the criteria, update the incumbent, then
   LAUNCH the next PENDING experiment (8-thread capped, own bg job). Command template:
   `POLARS_MAX_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONPATH=. python scripts/backtest_scripts/run_carver_walkforward.py --config <cfg> --report docs/reports/futures/<NAME>_READINESS.md --json output/deconcentration/<name>.json --jobs 8`
3. If a needed build isn't done: dispatch a subagent to build it TDD, review it, THEN launch.
   If a build fails review twice, mark the experiment SKIPPED and move on. Never stall.
4. When all experiments DONE (or ~6hr elapsed): write a summary to
   `docs/progress/20260704_OVERNIGHT_RESULTS.md`, leave everything on the branch for user review.

## Baseline / incumbent
- Baseline carry (IDM-weighted absolute carry): OOS Sharpe 0.76, PBO 0.19, skew +1.31, kurt 22.2. Config `carry_idm_broad.yaml`.
- Plain carry (no IDM): 0.85 Sharpe, PBO 0.33.
- Trend (Carver TSMOM): 0.11 Sharpe, PBO 0.44 (crisis-insurance only).
- INCUMBENT (compounding, per user "log + keep building"): starts = carry_idm (0.76 / 0.19).

## Pre-committed criteria (autonomous)
- A variant WINS (becomes incumbent) iff PBO < 0.25 AND OOS Sharpe > incumbent Sharpe by > 0.03.
- VALUE is a diversifying pillar iff standalone PBO < 0.35 AND Sharpe > 0.35 AND |corr(value_ret, carry_ret)| < 0.5.
- A COMBINATION helps iff combined OOS Sharpe > best single AND PBO < 0.25.
- No parameter sweeping for Sharpe; each construction pre-committed. Note best-of-N; deflation-check before any deploy claim.

## Queue (status: PENDING / RUNNING / DONE / SKIPPED)

| # | Experiment | Needs build? | Config | Status | Result (Sharpe / PBO / note) |
|---|---|---|---|---|---|
| 1 | Value standalone WF + carry-corr | signal built (4854afa) | value_broad.yaml | RUNNING | -- |
| 2 | IDM per-instrument cap on carry (cap 2.0) | small idm_weights edit | carry_idm_cap20.yaml | PENDING | -- |
| 3 | IDM per-instrument cap on carry (cap 1.5) | (same edit, flag) | carry_idm_cap15.yaml | PENDING | -- |
| 4 | Empirical-C IDM on carry | idm_weights expanding-window C | carry_idm_empC.yaml | PENDING | -- |
| 5 | Phase-0 combiner + FDM | BUILD (forecast_combine.py) | (infra) | PENDING | -- |
| 6 | Multi-horizon carry | needs #5 | carry_multihorizon.yaml | PENDING | -- |
| 7 | Combine carry+value (if #1 pillar) else carry+trend | needs #5 | combine_*.yaml | PENDING | -- |
| 8 | Buffering on best carry variant | BUILD (position buffer) | <best>_buffered.yaml | PENDING | -- |
| 9 | Diversifying universe (RTY/KE + crypto carry) | BUILD (crypto carry branch + maps) | carry_idm_ext.yaml | PENDING | -- |

Aggressive mode (user-chosen): build #5, #8, #9 as needed; each TDD + reviewed before its run.
Order may adapt: Tier-1 (#1-4, existing infra) first; build #5 during a Tier-1 run; then #6-9.

## Decision log (append per experiment)
- 2026-07-04: value signal built (4854afa). Launching #1 (value standalone WF). Incumbent = carry_idm 0.76/0.19.
