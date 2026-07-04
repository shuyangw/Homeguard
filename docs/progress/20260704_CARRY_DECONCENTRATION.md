# Carry De-Concentration Stack (XS + IDM) - 2026-07-04

## Summary
Built and evaluated two orthogonal, parameter-free de-concentration levers for the
33-root broad futures carry strategy, as 3 pre-committed walk-forward trials, to push
the corrected carry baseline (OOS Sharpe 0.85, PBO 0.33) under the PBO 0.25 gate.
Result: the SIZING-side lever (IDM) PASSES -- PBO 0.33 -> 0.19 -- and is carry's first
gate-clear. The SIGNAL-side lever (XS) backfires alone and poisons IDM when stacked.
Conclusion: de-concentrate carry at the sizing layer, never touch the signal.

## Trial Results (33-root broad, 2010-06-07..2026-02-20, 13 OOS windows, weekly)

| Trial | Lever | OOS Sharpe (1x / 1.5x) | PBO | Skew | Kurt | Verdict |
|---|---|---|---|---|---|---|
| Baseline | absolute carry | 0.85 / -- | 0.33 | -- | 21.0 | just over gate |
| XS-alone | signal-side | 0.7665 / 0.7345 | 0.4636 | -0.70 | 18.3 | WEAK |
| **IDM-alone** | **sizing-side** | **0.7646 / 0.6975** | **0.1887** | **+1.31** | 22.2 | **PASS** |
| XS+IDM | both | 0.7662 / 0.7389 | 0.5274 | -0.56 | 14.4 | WEAK (worst) |

PSR/DSR = 1.0 for every trial (parameter-free, trial_count=1). All clear the 1.5x cost
gate on Sharpe. Reports (gitignored): `docs/reports/futures/CARRY_{XS,IDM,XS_IDM}_BROAD_READINESS.md`.
Registered in `output/experiments.duckdb` (project-wide trial count += 3).

## Interpretation
- **IDM works as designed.** Cluster risk-weighting cut PBO nearly in half for a ~0.09
  Sharpe give-up and flipped skew positive (+1.31). Textbook robustness-for-return trade.
- **XS backfires, consistently.** Within-class demeaning converts absolute carry into
  relative-value carry -- a less window-stable signal. It raised PBO alone (0.46) and
  dragged IDM back below the gate when stacked (0.53, the worst outcome). The levers are
  opposed, not additive; IDM does NOT dominate XS.
- **Caveat (best-of-3).** IDM-alone is the best of 3 pre-committed trials; DSR was computed
  at trial_count=1 (parameter-free doctrine). PBO 0.19 sits below 0.25 with margin and the
  pass is mechanistically justified, so a mild 3-trial deflation would not flip it -- but it
  is a best-of-3 selection, not a pristine single-hypothesis pass.
- **Caveat (risk redistribution, final-review Minor 4).** Equal-cluster-risk + median-pin
  gives the 2-root meats cluster (LE/HE) div_mult ~2.5 (2.5x per-instrument vol target) vs
  ~0.625 for the 8-root fx cluster. So cluster-level de-concentration CONCENTRATES
  per-instrument risk into small clusters; kurtosis can migrate to LE/HE rather than shrink
  (consistent with IDM kurt 22.2 > baseline 21). By-design, not a bug -- and a lead for a
  future refinement (per-instrument div_mult cap, or the deferred empirical-C trial).

## Changes Made
- **`src/data/futures/asset_class.py`**: added 7-cluster economic-complex map (`CLUSTER`,
  `cluster_for`) -- equity/rates/fx/energy/metals/grains/meats.
- **`src/strategies/advanced/futures_carry_strategy.py`**: `FuturesCarryXSStrategy` --
  within-asset-class same-day demean + z-score of absolute carry, `_XS_SCALE=10.0`, clip.
  Registered `FuturesCarryXS` in `src/strategies/registry.py`.
- **`src/backtesting/utils/idm_weights.py`**: `compute_div_mult(universe)->dict` -- data-free
  IDM (cluster risk weights, fixed corr intra 0.5 / inter 0.0, cap 2.5, median-pin N_scale).
- **`src/backtesting/engine/futures_portfolio_simulator.py`** + **`futures_backtest.py`**:
  `run_sized` widened to `div_mult: float|dict`; `run_futures_backtest` honors `backtest.idm`
  (default False, back-compatible).
- **`scripts/backtest_scripts/run_carver_walkforward.py`**: threaded the `idm` flag through
  the per-window config (was silently dropped -- would have run the IDM trials WITHOUT IDM).
- **`config/backtesting/carry_{xs,idm,xs_idm}_broad.yaml`**: 3 trial configs. Tests for each.

## Commits (branch `feat/carry-deconcentration`, off `main` @ 0640a5f)
- `8c3974e` feat(futures): 7-cluster economic-complex map
- `044fb13` feat(futures): FuturesCarryXS (within-class demeaned carry) + registry
- `522e36d` feat(futures): IDM per-root div_mult weights (cluster risk + fixed-corr IDM)
- `96dfe9b` feat(futures): per-root div_mult (idm flag) threaded into sizing
- `0f24d2f` feat(futures): 3 de-concentration trial configs + test
- `0dfbfb8` fix(futures): thread backtest.idm flag through carver walk-forward driver
- (design + plan: `c0926e4`, `e74d44f`)

## Known Issues / Remaining Work
- **Merge decision pending** (branch not merged; nothing pushed).
- Final-review Minors (none blocking): (1) no direct test of the single-shot
  `run_futures_backtest` idm branch; (2) back-compat test asserts length not value-identity;
  (3) design's book-level realized-vol sanity check not implemented (median-pin proxy used);
  (5) singleton asset-class under XS yields 0.0 not NaN (no broad-universe impact).
- **Leads for the deploy candidate (IDM-weighted absolute carry):** per-instrument div_mult
  cap to counter the small-cluster risk concentration (Minor 4); deferred empirical-C IDM
  trial; formal best-of-3 deflation if promoting to production.
- Operational: the 3-in-1 background walk-forward was SIGKILLed at the ~60-min harness bg-job
  cap; re-run one-trial-per-job. Cause diagnosed via RunStatus (stale RUNNING) + event log
  (no OOM/sleep). Recorded to durable memory.

## Validation
- Every code task: fresh implementer + independent reviewer subagent, TDD, per-task green.
- Walk-forward idm-threading fix: 3 new threading tests, all 5 flag links traced by reviewer.
- Final whole-branch review (opus): MERGE-READY, 0 Critical / 0 Important; integration
  continuity (both cost legs), causality (no lookahead), IDM math, back-compat all confirmed.
- 3 walk-forwards run 8-thread capped + RunStatus-tracked; preflight verified 33-root carry
  cache, strategy resolution, and IDM weights (median 1.0, fx floor 0.625, meats cap 2.5).
