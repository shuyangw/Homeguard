# Futures Sharpe-Uplift Campaign - Charter

**Date:** 2026-07-04 - **Status:** active, Phase 1 scoping - **Branch:** feat/futures-sharpe-uplift

## Goal
Lift the futures book from the current best single-signal result (IDM-weighted absolute
carry, OOS Sharpe 0.76, PBO 0.19 PASS) toward Sharpe > 1.0, WITHOUT sacrificing the
robustness gate (PBO < 0.25).

## Honest premise (read first)
- PBO 0.19 says carry is NOT overfit; it says nothing about the edge being big enough.
  0.76 single-signal Sharpe is respectable but not production-ready.
- Sharpe > 1 sustained OOS on a broad futures basket is a genuine stretch. Production CTAs
  live at ~0.7-1.0. Our one strong signal is already at 0.76.
- **Trend is demoted to crisis-insurance, NOT a Sharpe driver.** Diagnosis (2026-07-04):
  the old -0.45 was a stale pre-fix artifact of the negative-equity pct_change explosion
  (already fixed). Corrected trend is OOS Sharpe 0.11 / PBO 0.44 (WEAK) -- honestly weak,
  consistent with the 2011-2019 trend drought, correctly signed (no bug).
- **Combination arithmetic:** equal-risk combined Sharpe ~ (S1+S2)/sqrt(2(1+rho)).
  Carry 0.76 + trend 0.11 at rho=0 -> ~0.62 < 0.76. A weak second signal DRAGS Sharpe down;
  diversification lifts Sharpe only when signals are comparable in strength. So the path to
  >1 is (a) strengthen carry itself, and (b) find a second signal COMPARABLE to carry
  (~0.5+), low-correlated. Trend stays only for tail/regime cushioning.

## Phases (sequential, each gated; do not build on an ungated phase)

| Phase | Work | Go/No-Go Gate |
|---|---|---|
| 0 | Forecast-combination infra + Carver Forecast Diversification Multiplier (FDM). The FDM is a genuine gap (trend report notes forecasts are under-scaled without it). | Combiner reproduces a single signal exactly at N=1 (back-compat); FDM math verified. |
| 1 | Carry breadth: expand universe 33 -> as many liquid futures as data allows, and add multiple carry horizons combined via the FDM combiner. Build on our one strong signal. | Carry OOS Sharpe > 0.76 AND PBO < 0.25. |
| 2 | Turnover/cost buffering (Carver position buffer) on the carry book. | Net Sharpe uplift, turnover down, no PBO regression. |
| 3 | Second-pillar hunt: skew / vol-carry and value, each a pre-committed walk-forward trial. | Standalone OOS Sharpe >= ~0.4 AND low correlation to carry, else EXCLUDED from the book. |
| 4 | Combine carry(+horizons) + any qualifying second pillars + trend (small crisis weight) via the FDM combiner + IDM sizing. | Combined OOS Sharpe > best single-signal AND PBO < 0.25. The run at >1. |
| 5 | Productionize the winner toward IBKR paper: trade-log/capacity/turnover realism (methodology Section 12), sizing at real capital, broker routing. | Section 12 diagnostics clean; deploy candidate. |

Dependencies: 0 -> 1 -> 2 -> 3 -> 4 -> 5. Phases 2 and 3 may parallelize after 1 passes.
Binding constraint on 1 and 3: DATA availability (instrument price + carry/vol history).

## Doctrine (unchanged from carry/carver work)
- Parameter-free where possible: Carver scalars/speeds/caps, IDM constants, carry_scalar are
  FIXED doctrine, not swept. New signals get pre-committed constructions, not tuned.
- Every backtest: 8-thread cap (POLARS_MAX_THREADS=1 ... --jobs 8), RunStatus-tracked,
  ONE long run per background job (~60min harness cap), trade-logged, registered in
  output/experiments.duckdb. Walk-forward with the existing purge/embargo structure.
- Multiple-comparisons honesty: track cumulative trial count; any "best of N" selection is
  disclosed and deflation-checked before a deploy claim.

## Status log
- 2026-07-04: charter written. Trend diagnosis complete (0.11 WEAK, not broken). Next:
  Phase 1 data/universe scoping (what liquid futures beyond the 33 have usable history).

## Realistic outcome
Credible ceiling with breadth + buffering alone: ~0.9-1.0. A hard, sustained >1.0 likely
requires a genuine second pillar (Phase 3) that we do not yet have. Each gate will report
honestly whether we are clearing the bar or curve-fitting toward it.
