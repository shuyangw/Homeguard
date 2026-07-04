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
| 2 | build crypto carry cache (BTC/ETH) | PENDING (run after value WF frees threads) |
| 3 | value + crypto standalone configs | DONE 7eb100f |
| 4 | pillar correlation tool | DONE 5ff47e9 |
| 5 | VALUE standalone WF (train 61m) | RUNNING (bx1cdo5k7, cfg value_broad) |
| 6 | CRYPTO carry standalone WF | PENDING (needs T2 cache; run after value WF) |
| 7 | correlation + inclusion verdict | PENDING (needs T5,T6) |

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

## After Phase 1
If a pillar qualifies -> next plan = Phase 0 combiner + Phase 4 combination.
If neither -> honest fallback: carry + breadth/buffering only; write summary. Then (time permitting,
per spec) continue to Phase 2 IDM refinements (per-instrument cap, empirical-C) -- buildable without
the combiner.
