# Futures Sharpe-Uplift Campaign - Detailed Spec

**Date:** 2026-07-04 - **Status:** approved design, pre-plan - **Branch:** feat/futures-sharpe-uplift
**Supersedes the queue draft** `20260704_EXPERIMENT_QUEUE.md`; builds on charter
`20260704_FUTURES_SHARPE_UPLIFT_CAMPAIGN.md`. Strategy-lead consulted 2026-07-04.

## 1. Goal & success criterion

Lift the futures book from the incumbent (IDM-weighted absolute carry: OOS Sharpe **0.76**,
PBO **0.19**, skew +1.31, kurt 22.2) toward OOS Sharpe **> 1.0**, WITHOUT losing the gate.

**Success is defined as the COMBINED book clearing the full statistical gate**
(methodology Section 2.5: PSR > 0.95, DSR > 0.95 at then-current project-wide N, PBO < 0.25,
trade count >= 30, IS/OOS >= 0.7) **AND the 1.5x cost gate** (Section 5.7: 1.5x Sharpe >= 0.5,
1.5x PSR(0) >= 0.90) -- at whatever Sharpe that requires. Per Section 2.6 the target is
passing the gate, not a Sharpe number. We PUSH for > 1.0 (user directive) by leading with the
crypto-carry lever, but report honestly if the ceiling lands at 0.85-0.95.

**Honest ceiling (spanning math).** For two return streams optimally weighted,
`SR_comb = sqrt((SR1^2 + SR2^2 - 2*rho*SR1*SR2)/(1 - rho^2))`. With carry SR1 = 0.76 fixed:

| pillar SR2 | rho | combined (optimal-weight upper bound) |
|---|---|---|
| 0.40 | 0.50 | ~0.78 |
| 0.50 | 0.20 | ~0.84 |
| 0.60 | 0.10 | ~0.93 |
| 0.76 | 0.00 | ~1.08 |

These are UPPER bounds (in-sample-optimal weights); pre-committed fixed weights fall short.
**> 1.0 requires a pillar with SR >= ~0.55-0.6 AND rho <= ~0.15.** Value's likely profile
(SR ~0.35-0.5, rho ~0.3-0.5, a carry "cousin") reaches only ~0.78-0.84. The only lever with a
plausible path to a strong low-rho pillar is crypto-carry (esp. perp funding), which is also
the highest engineering + regime risk.

## 2. Lever assessment (strategy-lead, expected Sharpe / overfitting risk)

1. Crypto-carry (BTC/ETH) -- highest EV, only plausible > 1.0 path; highest risk (2 roots,
   short/regime-heavy history). CME calendar carry (data on hand) is weaker/noisier than perp
   funding carry (needs data acquisition).
2. Value (Asness reversal) -- buildable now; modest, capped by cousin-correlation.
3. Multi-horizon carry -- same family; +0.02-0.08, low risk (horizon choice pre-committed).
4. Buffering -- +0.02-0.05, mechanical (1.5x cost only cost 0.76->0.70, so headroom small).
5. IDM refinements (per-instrument cap, empirical-C) -- tail/PBO robustness, not raw Sharpe.
6. FDM combiner -- pure plumbing, zero direct Sharpe; enables the rest correctly.

## 3. Phases & gates

| Phase | Work | Gate (methodology ref) |
|---|---|---|
| 0 | Combiner + FDM (`combine_forecasts`): weighted multi-signal sum, pre-registered fixed inter-signal correlation, causal per-cell NaN contract. | Correctness only: N=1 back-compat identity; FDM formula unit-tested; NaN causal. Not a stat gate. |
| 1A | Value standalone WF + carry-return correlation (signal already built, commit 4854afa). | Inclusion bar (Sec 4). |
| 1B | Crypto CME calendar carry: crypto carry branch (front/back roll-yield) + crypto cluster/class map + BTC/ETH; standalone WF + carry-corr. | Inclusion bar (Sec 4) + crypto-specific caveats (Sec 6). |
| 1C | Perp funding-rate data acquisition (parallel sub-project): exchange funding history -> stronger crypto carry. | Data lands -> re-run 1B with funding carry. |
| 2 | Strengthen carry: multi-horizon carry (via combiner); IDM per-instrument cap + empirical-C. | OOS Sharpe > 0.76 AND PBO < 0.25 AND IS/OOS >= 0.7 AND >= 5% DSR-adjusted improvement over prior round, else STOP widening (Sec 5.2/5.3). |
| 3 | Buffering (Carver position buffer) on best carry variant. | Net Sharpe up AND turnover down AND no PBO regression (Sec 4.6). |
| 4 | Combine qualifying pillars via FDM + IDM, PRE-COMMITTED fixed weights. | Combined OOS Sharpe > best single AND full Sec 2.5 gate (then-current N) AND Sec 5.7 cost gate. |
| 5 | Portfolio integration (Sec 6: corr vs OMR/RAMP/CSCM < 0.7 or 1.2x-Sharpe exception, marginal portfolio Sharpe > 0, capacity <= half ceiling) THEN productionize (Sec 12 diagnostics). | Sec 6 + Sec 12. |

Dependencies: 0 -> {1A,1B,1C,2} -> 3 -> 4 -> 5. 1A/1B/2 may run concurrently (thread-budget
permitting); 1C is a parallel data track. Order prioritizes crypto (1B/1C) per the > 1.0 push.

## 4. Decision framework (two-tier; tiered correlation)

**Pillar INCLUSION bar** (does a signal earn a seat in the combined book):
- standalone PBO < 0.35 AND standalone OOS Sharpe > 0.35, AND correlation-tiered vs carry returns:
  - `|rho| < 0.3` -> good diversifier, full pre-committed weight.
  - `0.3 <= |rho| < 0.5` -> weak diversifier; include ONLY if standalone SR >= 0.45 (else near-zero net uplift, exclude).
  - `|rho| >= 0.5` -> EXCLUDE regardless of standalone Sharpe (cousin trap).

**Combined GRADUATION bar** (does the combined book go live): full Sec 2.5 gate + Sec 5.7 cost
gate. Distinct from inclusion: a pillar that clears inclusion but never lets the combined stream
clear graduation is a DOCUMENTED no-go, not a silent failure.

**Weighting is PRE-COMMITTED, never fitted.** Default: equal-risk (inverse-vol) per pillar
(Sec 6.5). Alternative allowed: equal-weight forecasts pre-FDM (combiner default). Whichever is
chosen is written here BEFORE the combination experiment runs; changing it after seeing results
is a new trial requiring its own count.

### 4.1 Kelly criterion -- as a sizing discipline, NOT a Sharpe lever

Kelly optimizes long-run geometric growth (a function of leverage); Sharpe is scale-invariant,
so Kelly does NOT change the gate Sharpe and is not a path to > 1.0. It applies here as sizing
doctrine, in three specific ways:
- **Fractional-Kelly vol target.** Full-Kelly annualized vol target ~= book Sharpe (~76% for
  the 0.76 book) -- unsafe. Run FRACTIONAL Kelly (~1/4 to 1/2). The existing
  `vol_target_per_instrument = 0.20` already sits in this conservative band; any combined-book
  vol target in Phase 4 is chosen as a fixed fractional-Kelly level, not fit for best Sharpe.
- **Kurtosis-aware de-risking.** Full Kelly assumes Gaussian returns and OVERBETS under fat
  tails. The incumbent runs kurt 22.2, so realized kurtosis is an explicit argument to cut the
  Kelly fraction. Wire the per-gate skew/kurtosis report (Sec 6) into the sizing-fraction
  rationale: higher realized kurt -> lower Kelly fraction.
- **NOT for fitting pillar weights.** Full-Kelly pillar weights are the mean-variance optimal
  weights (proportional to inverse-covariance times mean) -- exactly the FITTED weighting Sec 4
  forbids (estimation error, overfitting, own trial count). Equal-risk (our pre-committed choice)
  IS the robust fractional-Kelly stand-in when mu/Sigma are not trusted. Do not use empirical
  Kelly weights; any such run is a separate, deflation-counted trial and is out of scope here.

## 5. Constructions (pre-committed)

- **Value** (built): `-(log P.shift(252) - log P.shift(1260))`, vol-normalized, `_VALUE_SCALAR`
  fixed, clipped +/-20. Causal (all shifts past). Scale-invariant metrics (Sharpe, rho) decide.
- **Crypto CME calendar carry** (Phase 1B): reuse the futures carry mechanism (annualized
  front/back roll yield) on CME BTC/ETH; add a `crypto` cluster to CLUSTER and asset_class map,
  and a crypto branch to `CarryCalculator`. Fixed convention, no sweep. Flag short history
  (BTC 2017, ETH 2021) and regime risk in the readiness report.
- **Perp funding carry** (Phase 1C): once funding data acquired, crypto carry = trailing
  funding rate (annualized), fixed convention. Separate readiness report.
- **FDM combiner** (Phase 0): `combine_forecasts(forecasts: dict[str,DataFrame],
  weights: dict|None, inter_corr: float, cap: float, fdm_cap: float) -> DataFrame`.
  `FDM = min(1/sqrt(w'Cw), fdm_cap)`, C = fixed constant matrix (diag 1, off-diag `inter_corr`).
  **`inter_corr` pre-registered = 0.5** (typical Carver inter-forecast correlation for related
  signals; conservative, avoids over-scaling). Combined = (weighted_sum * FDM).clip(+/-cap).
  N=1 -> FDM=1 -> identity. NaN: a cell missing in any contributing signal is decided per-cell
  from already-shifted forecasts only (no future-referencing availability window).

## 6. Integrity & logging (mandatory)

- **Project-wide DSR trial-count ledger** (a table in this spec's living log, Sec 9): after every
  experiment record (exp #, `combinations_in_run`, cumulative N, resulting DSR benchmark SR0*).
  The Phase-4 gate uses the then-current N, per Sec 2.3/9.4. Parameter-free runs log N=1.
- **Value 5yr warmup**: value-inclusive walk-forwards use train >= 61 months, OR the per-window
  data-availability filter must be confirmed to EXCLUDE (not silently truncate) roots/windows
  lacking the full 1260-day lookback. Stated in the value experiment's config notes.
- **Report skew AND kurtosis at every gate** (incumbent kurt 22.2; reweighting levers can
  migrate tail risk without a Sharpe/PBO red flag).
- **Report ALL experiments** (wins, losses, SKIPPED) in the final summary -- no survivorship.
- **Per-lever stopping caps** (Sec 5.6, applied PER lever, not once for the campaign): <= 3
  optimization rounds, <= 6 hr compute, <= 5000 cumulative configs per lever (IDM cap sweep,
  horizon choice, weighting choice each count separately).
- **Crypto capacity** uses a crypto-specific model (perp OI / exchange depth), not the
  ADV/OI/order-book futures analog (Sec 6.4 does not map directly).
- **Combined-level exposure clip once** at the combined forecast, not re-clipped per signal.

## 7. Execution mechanics

- Every backtest: 8-thread cap (`POLARS_MAX_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
  MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 ... --jobs 8`), ONE walk-forward per background job
  (under the ~60-min harness bg-job cap -- never chain multiple runs in one job), RunStatus-
  tracked, trade-logged, registered in `output/experiments.duckdb`.
- All work committed to `feat/futures-sharpe-uplift`; NOTHING merged or pushed without explicit
  user approval. Each experiment's code + config committed; readiness reports are gitignored
  (findings captured in committed progress/summary docs).
- Compaction-safe: a living queue/ledger doc holds phase status + controller instructions so an
  unattended or context-compacted run resumes cleanly.
- **No experiment launches until the user explicitly says go.**

## 8. Testing (per component, TDD)

- `combine_forecasts`: N=1 identity; weighted-sum correctness; FDM = min(1/sqrt(w'Cw), cap);
  causal NaN per-cell; ASCII-only, parameter-free.
- Crypto carry: front/back roll-yield sign + magnitude on a known case; crypto cluster/class map
  covers BTC/ETH; carry non-inert (nonzero) on real crypto data.
- Value warmup: confirm first ~1260 rows NaN; per-window exclusion or train>=61m verified.
- Each strategy: causality (no lookahead on append), cap, registry resolution.

## 9. Living log (append per experiment; the recovery anchor)

Trial-count ledger + per-experiment results table maintained here and in a companion progress
doc. Fields: exp #, phase, config, combinations_in_run, cumulative N, OOS Sharpe (1x/1.5x),
PBO, PSR, DSR, skew, kurt, carry-correlation (for pillars), inclusion/graduation verdict, commit.
Seed row: incumbent carry_idm = 0.76 / 0.19 / N so far = 1 (parameter-free).

## 10. Scope / non-goals

- No parameter sweeping for Sharpe; every construction pre-committed. Best-of-N disclosed +
  deflation-checked before any win/deploy claim.
- No merge/push without approval. No live-trading change in this campaign (Phase 5 is the
  productionization DESIGN + paper validation, gated on Sec 6 + Sec 12).
- Skew signal is OUT (data covers only ES/NQ). Redundant micro instruments are OUT (no
  diversification). Trend stays a small crisis-insurance sleeve only (0.11 SR; adding weight
  lowers Sharpe).
