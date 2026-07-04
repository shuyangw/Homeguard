# Overnight Autonomous Run - Phase 1 Second-Pillar Hunt (2026-07-04)

**Controller doc + results log. READ THIS + the plan if resumed/compacted.**
Plan: `docs/strategies/research/20260704_PHASE1_SECOND_PILLAR_PLAN.md`.
Spec: `docs/strategies/research/20260704_SHARPE_UPLIFT_CAMPAIGN_SPEC.md`.
Branch `feat/futures-sharpe-uplift`. User AWAY (authorized autonomous execution incl. runs).
Rules: 8-thread cap, ONE walk-forward per bg job (~60min cap), RunStatus, trade-log, register,
commit each task, NOTHING merged/pushed. Chain via completion notifications; controller step
(plan Sec, spec Sec 7) on each invocation. Report ALL outcomes (no survivorship).

## Incumbent / baselines
- carry_idm (incumbent): OOS Sharpe 0.76, PBO 0.19. carry (no IDM): 0.85 / 0.33. trend: 0.11 / 0.44.

## Two-tier inclusion bar (for value & crypto)
standalone PBO<0.35 AND Sharpe>0.35; corr vs carry: <0.3 full / 0.3-0.5 only if SR>=0.45 / >=0.5 exclude.

## Task status
| Task | What | Status |
|---|---|---|
| 1 | crypto carry branch + BTC/ETH maps | DONE 4f2a1cc |
| 2 | build crypto carry cache (BTC/ETH) | DONE (BTC 2527r/ETH 1554r, mean carry +0.059, non-inert) |
| 3 | value + crypto standalone configs | DONE 7eb100f |
| 4 | pillar correlation tool | DONE 5ff47e9 |
| 5 | VALUE standalone WF (train 61m) | DONE -> REJECT/EXCLUDE (-0.22 Sharpe, PBO 0.64) |
| 6 | CRYPTO carry standalone WF | DONE -> PASS (0.61 Sharpe, PBO 0.24) |
| 7 | correlation + inclusion verdict | DONE: crypto INCLUDE full weight (rho -0.065); value EXCLUDE |

**Controller next step:** when value WF (Task 5) completes -> record its metrics -> run Task 2
(crypto cache: `build_carry_cache.py --roots BTC ETH --start 2017-01-01 --end 2026-02-20 --jobs 2`,
8-thread capped) -> Task 6 (crypto WF, own bg job) -> Task 7 (pillar_correlation for value & crypto
vs carry_idm_broad + inclusion verdicts). Serial due to 8-thread cap (one CPU job at a time).

## DSR trial-count ledger (project-wide N)
| exp | combinations_in_run | cumulative N | note |
|---|---|---|---|
| (incumbent carry_idm) | 1 | (parameter-free) | seed |
| value WF | 1 | +1 | parameter-free |
| crypto WF | 1 | +1 | parameter-free |

## Results (append per experiment; Sharpe 1x/1.5x, PBO, PSR, DSR, skew, kurt, corr, verdict)
- **VALUE (Asness 5yr-1yr reversal, train 61m):** OOS Sharpe -0.2162 (1.5x -0.2273), PBO 0.6420,
  PSR/DSR 0.0, skew 0.51, kurt 8.30, n_windows 11, n_oos 3316. **Verdict REJECT -> EXCLUDE**
  (fails inclusion Sharpe floor 0.35 AND PBO floor 0.35; correlation moot). Reversal anti-worked
  on this basket 2015-2026 (long-horizon momentum persisted; raw signal ~+0.22). Sign NOT flipped
  post-hoc (pre-committed construction; flipping after seeing results = data-snooping, forbidden).
  Legitimate negative finding (unit tests confirmed causal + correct reversal sign). N += 1.
- **CRYPTO CARRY (CME BTC/ETH calendar roll-yield):** OOS Sharpe 0.6130 (1.5x 0.6107 -- near-zero
  cost drag, slow/low-turnover), PBO 0.2433, PSR/DSR 1.0, skew +0.57, kurt 11.30, n_windows 7,
  n_oos 1915. **Verdict PASS** (clears Sec 2.5 + cost gate); clears inclusion Sharpe floor 0.35 AND
  PBO floor 0.35. Correlation vs carry PENDING (bphjrc7lk) -> decides weight tier. N += 1 (now 3).
  CAVEATS (low-confidence): only 2 roots / 7 windows / ~7.6yr, PBO 0.24 is JUST under 0.25, crypto
  2019-2026 regime-heavy (2020-21 trends, 2022 crash) -> strong result may be partly regime luck.
  Best-of-2 pillar trials (value rejected, crypto passed) -- mild multiple-comparison.

- **COMBINED carry+crypto (35-root naive IDM equal-cluster, serial):** OOS Sharpe 0.4217
  (1.5x 0.4038), PBO 0.1019, PSR/DSR 1.0, skew 0.82, kurt 27.60, n_windows 13, n_oos 3964.
  **DOES NOT WIN** (Sharpe 0.42 << incumbent 0.76). Clears the full stat gate (PBO 0.10 -- crypto
  diversification HELPED robustness) but Sharpe CRATERED. ROOT CAUSE: IDM equal-cluster-risk gives
  the 2-root crypto sleeve a full 1/8 risk budget = same as the 8-root fx cluster; crypto's high vol
  + 2022 crash dominate the book (kurt 22.2->27.6), dragging Sharpe. Captured only 42% of the 1.007
  optimal-weight bound -- naive weighting massively over-allocates crypto. Incumbent stays carry_idm.
  NEXT: per-instrument div_mult cap (Minor-4 pre-committed refinement, ONE ex-ante value) to constrain
  crypto over-allocation; re-run combination (max_workers=1, OOM-safe).
  KNOWN BUGS (real fixes, logged): (1) parallel OOM on crypto 1min data at jobs=8; (2) CLI report-phase
  hang on crypto-inclusive run.

## After Phase 1
If a pillar qualifies -> next plan = Phase 0 combiner + Phase 4 combination.
If neither -> honest fallback: carry + breadth/buffering only; write summary. Then (time permitting,
per spec) continue to Phase 2 IDM refinements (per-instrument cap, empirical-C) -- buildable without
the combiner.
