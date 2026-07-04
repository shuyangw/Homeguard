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
| 1 | crypto carry branch + BTC/ETH maps | PENDING (building) |
| 2 | build crypto carry cache (BTC/ETH) | PENDING (needs T1) |
| 3 | value + crypto standalone configs | PENDING |
| 4 | pillar correlation tool | PENDING |
| 5 | VALUE standalone WF (train 61m) | LAUNCHED |
| 6 | CRYPTO carry standalone WF | PENDING (needs T1-3 + T2 cache) |
| 7 | correlation + inclusion verdict | PENDING (needs T5,T6) |

## DSR trial-count ledger (project-wide N)
| exp | combinations_in_run | cumulative N | note |
|---|---|---|---|
| (incumbent carry_idm) | 1 | (parameter-free) | seed |
| value WF | 1 | +1 | parameter-free |
| crypto WF | 1 | +1 | parameter-free |

## Results (append per experiment; Sharpe 1x/1.5x, PBO, PSR, DSR, skew, kurt, corr, verdict)
- (pending)

## After Phase 1
If a pillar qualifies -> next plan = Phase 0 combiner + Phase 4 combination.
If neither -> honest fallback: carry + breadth/buffering only; write summary. Then (time permitting,
per spec) continue to Phase 2 IDM refinements (per-instrument cap, empirical-C) -- buildable without
the combiner.
