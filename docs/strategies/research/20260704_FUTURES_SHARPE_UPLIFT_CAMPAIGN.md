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

## Data scoping verdict (2026-07-04) -- PRUNES the campaign

Inventory of `H:\Stock_Data\futures\databento` (62 symbol partitions):
- **Universe expansion is weak.** 47 roots reachable with data+specs+carry, but the 14
  addables are mostly REDUNDANT micros (MES/MNQ/MYM/M2K, MGC/MCL/MNG, SIL = smaller copies of
  roots already in the basket -> zero new diversification; IDM would just down-weight the
  crowded clusters). Genuinely diversifying adds: crypto (BTC/ETH -- needs a carry branch +
  cluster/class mapping in code), and marginally RTY/KE. -> DROP the redundant micros.
- **Skew / vol-carry: BLOCKED.** The only futures options on disk cover ES + NQ. No broad
  skew signal buildable. (This was the best carry-COMPARABLE second-pillar candidate.)
- **Value: buildable** across all 47 roots (price-only). The one viable second pillar --
  but value and carry are cousins, so measure its carry-correlation before counting on it.

### Pruned scope (what we actually build)
- Phase 1 breadth: NOT broad micros. Only (a) multiple carry HORIZONS via the combiner, and
  (b) genuinely-diversifying instruments = crypto carry (BTC/ETH, needs code) + RTY/KE.
- Phase 3 second pillar: VALUE only (skew dead). Gate additionally on low carry-correlation.
- Phases 2 (buffering), 4 (combine), 5 (productionize) unchanged.

### Revised realistic outcome
Skew (blocked) and broad breadth (redundant) were the two biggest expected levers; both are
largely gone. Honest ceiling now ~0.85-0.95; a hard sustained >1.0 looks UNLIKELY on this
data unless value surprises or crypto carry pulls real weight. Proceeding anyway per user
direction, pruned to what's buildable; every gate reports honestly.

## Phase 0 design -- forecast combiner + FDM

New `src/backtesting/utils/forecast_combine.py`:
`combine_forecasts(forecasts: dict[str, pd.DataFrame], weights: dict[str, float] | None,
inter_corr: float, cap: float, fdm_cap: float) -> pd.DataFrame`.
- Align the per-signal forecast panels (same dates x roots); weighted sum per cell
  (default equal weights summing to 1).
- FDM = min(1 / sqrt(w' C w), fdm_cap) with C a FIXED constant correlation matrix
  (diagonal 1, off-diagonal `inter_corr` -- doctrine constant, NOT swept; data-free ->
  parameter-free, causal). Same math family as IDM.
- Combined = (weighted_sum * FDM).clip(-cap, cap).
- Back-compat identity: a SINGLE signal (N=1) -> FDM=1 -> returns that forecast unchanged.
- NaN contract: a cell missing in any contributing signal stays NaN (no fabricated forecast);
  or combine over available signals with renormalized weights -- decide + test explicitly.
This is scaling plumbing (FDM does not change Sharpe); Sharpe comes from weights + signals.

## Status log
- 2026-07-04: charter written; trend diagnosed (0.11 WEAK, not broken); data scoped
  (skew blocked, breadth redundant, value the only pillar). Building Phase 0 (combiner+FDM).
