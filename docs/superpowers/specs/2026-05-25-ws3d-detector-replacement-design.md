# WS-3d: Detector Replacement (LightGBM on Leading Indicators) -- Design

**Date**: 2026-05-25
**Status**: Proposed (full implementation spec; track committed from E8 lag-structural decision row)
**Branch**: ws3d-detector-replacement (to be created from v12-bear-to-cash)
**Owner**: Shuyang
**Type**: Full implementation spec; analogous fidelity to V14 rev2
**Builds on**:
- `docs/superpowers/specs/2026-05-24-ws3-detector-intervention-design.md` (track-conditional WS-3 spec; lag-structural row selects WS-3d)
- `docs/reports/ramp/20260525_experiment8_action_convergence.md` (E8 verdict + lag-structural finding)
- `docs/reports/ramp/20260523_regime_detector_diagnostic.md` (H5 confirmed: 14-day SMA lag)
- `docs/reports/ramp/20260524_phase4_v12_readiness.md` (V12 gap_days = -3.42 confirmation)
- V14 rev2 spec (`docs/superpowers/specs/2026-05-24-v14-soft-bear-factorial-design.md`) for spec structure precedent

## Position

The 8-experiment campaign closed with three independent measurements of structural detector lag (V12 gap_days = -3.42, original H5 = 14-day SMA lag, E8 exit-to-SPY-low lag = -8 days). Same finding, three different windows, three different consumption patterns. Consumer-layer fixes (V12/V12c/V13 on argmax; V14a/b/c on soft scores) cannot recover days the detector itself missed.

WS-3d replaces the SMA-based detector with a new architecture trained on **leading indicators** rather than SMA-confirmation features. The new detector is a different strategy in the methodologically relevant sense (new inputs + new scoring + new architecture); it therefore **resets the n_trials chain**, escaping the DSR=36 trap that gates the existing detector family.

This is not an iteration on the existing detector. The v0 detector remains in the codebase for diagnostic continuity (V11/V12/V12c/V13/V14a/b/c keep working). WS-3d's detector lives alongside it as `MarketRegimeDetectorV1` (or similar), consumed by a new variant family (V20+).

## Decision criteria

WS-3d succeeds if **at least one** V20-family variant produces:

- **TIER 1**: all 5 standard gates PASS AND Sharpe(@5 bps near_close) > Sharpe(V11 @ 5 bps near_close on the same window) + 0.10 AND forward OOS Sharpe > 0 over the validation window.
- **TIER 3**: structural gates PASS; absolute-significance gates (PSR, DSR) FAIL on the IS window OR forward OOS Sharpe <= 0. File for extended forward OOS; do not deploy.
- **TIER 4**: any structural gate fails.

Forward OOS validation is **mandatory** for any deployment recommendation regardless of IS gate verdicts (Methodological pre-commitment 2 below). Minimum 1 month forward OOS, ideally aligned with V11's A7 5-session standard.

If WS-3d's primary variant reaches TIER 4 and no V20-family variant clears TIER 1 within the 5-7 week budget, the campaign escalates to halt-or-redirect per the parent WS-3 spec's Appendix.

## Methodological pre-commitments

These are spec-time commitments that bind implementation; relaxing any of them requires a spec amendment.

### Pre-commitment 1: Trial-chain reset justification

WS-3d is a genuinely new strategy, not a variant of the existing detector family. Three sufficient grounds:

1. **New inputs**: VIX/VIX3M ratio, HY OAS (FRED BAMLH0A0HYM2), NYSE A-D breadth (% S&P 500 above 50-day MA), CBOE SKEW. None of these are in the v0 detector's input set (which uses SPY price for SMA + VIX percentile only).
2. **New scoring**: LightGBM classifier outputs P(G1_BEAR | indicators) as a continuous score. Replaces the v0 detector's hand-tuned threshold-and-score formula.
3. **New architecture**: tree-based classifier with purged combinatorial cross-validation, versus the v0 detector's rule-based scoring.

n_trials_project for the WS-3d family starts at **1** (the WS-3d primary variant). Subsequent V20-family variants add +1 each per the standard discipline. The v0-detector family's 36-trial chain is preserved separately in the experiment registry; it does not propagate into the WS-3d family.

### Pre-commitment 2: Forward OOS validation mandate

No deployment recommendation without forward OOS data on the new detector's outputs. Minimum window: 1 month (~21 trading days). Preferred: aligned with V11's A7 5-session standard (currently ~1 calendar month of paper trading).

If WS-3d's IS readiness passes all gates but the forward OOS Sharpe is <= 0 or shows material regime-conditional negative performance, the deployment recommendation is BLOCKED regardless of IS verdicts. Forward OOS Sharpe should be positive on the forward window OR the synthesis explicitly documents why a negative forward Sharpe is acceptable (rare edge case).

### Pre-commitment 3: OMR consumer audit

OMR's current Bayesian filter consumes the production detector's regime labels. WS-3d's new detector is initially used by RAMP only (V20+ family); OMR continues using v0. Before any deployment proposal that would migrate OMR to the new detector:

1. Run OMR's existing backtest harness with the new detector's regime labels.
2. Compare OMR per-day P&L by regime: new vs old detector.
3. If material divergence (Sharpe range/max difference > 30% across any common regime), OMR's adapter requires its own gating before the detector migration proceeds.

For the initial WS-3d landing, no OMR migration. The audit is a deployment-time gate, not an implementation blocker.

### Pre-commitment 4: Backwards-compatibility

The new detector exposes `last_regime_scores: Dict[str, float]` and `classify_regime(spy_data, vix_data, timestamp, ...) -> Tuple[str, float]` with the same signature as the v0 detector. The 5 regime keys (STRONG_BULL, WEAK_BULL, SIDEWAYS, UNPREDICTABLE, BEAR) remain. The score semantics change: v0 outputs hand-tuned scores in [0, 1] with no probabilistic interpretation; WS-3d outputs LightGBM-predicted P(regime | indicators) which IS probabilistic but on a different scale.

V11/V12/V12c/V13/V14a/b/c consumers are NOT migrated to the new detector in the initial WS-3d landing. They continue consuming v0. Schema compatibility means the new detector COULD be swapped in by changing the `_DETECTOR` singleton in `variants.py`, but this swap is NOT part of WS-3d's initial landing.

Breaking changes require a migration plan; non-breaking is preferred. The current spec achieves non-breaking via schema compatibility.

### Pre-commitment 5: Regime diagnostic rerun

The original H1-H5 diagnostic (`docs/reports/ramp/20260523_regime_detector_diagnostic.md`) is rerun against WS-3d's detector outputs on the same 2017-2026 window. Primary success metrics:

1. **H5 lag reduction**: median lag from G1_BEAR ground-truth onset to first detector-BEAR label MUST decrease by >= 30% (from 14d to <= 10d). Indicator-set is dropped from canonical configuration if the reduction is < 30%.
2. **H4 flicker reduction**: median run length for non-SIDEWAYS regimes MUST increase relative to v0. If LightGBM still produces flicker comparable to v0 argmax, add a Schmitt-trigger consumer-side layer (V20 variant) and re-test.
3. **H1-H3 parity**: regime distribution and transition characteristics should not deviate materially from ground-truth labelers (G1_BEAR, G2_forward_window, G3_vol_spike).

The diagnostic rerun is a **gating check** before the readiness orchestrator runs: if H5 lag is not reduced by 30%, the leading indicator set OR the architecture is wrong and we don't proceed to readiness gating.

## Canonical inputs

Pre-committed (Amendment 2):

| Indicator | Source | Frequency | History | Purpose |
|---|---|---|---|---|
| VIX/VIX3M ratio | CBOE direct or yfinance (^VIX, ^VIX3M) | daily | 2017-present | Term-structure inversion as drawdown leading indicator |
| HY OAS | FRED series BAMLH0A0HYM2 | daily | 1997-present | High-yield credit spread; bond market leads equity vol |
| NYSE A-D breadth | yfinance + S&P 500 universe; computed as % of constituents above 50-day MA | daily | 2017-present | Market breadth deterioration as drawdown leading indicator |
| CBOE SKEW index | CBOE direct or yfinance (^SKEW) | daily | 2017-present | Tail-risk options pricing as drawdown leading indicator |
| SPY OHLC | existing data acquisition | daily | 2017-present | Retained as backward-compat input; informs G1_BEAR label generation |
| VIX OHLC | existing data acquisition | daily | 2017-present | Retained as backward-compat input |

These 6 inputs constitute the canonical WS-3d input set. **Alternative input sets are informational sensitivity panels only.** Examples NOT in the canonical set but acceptable as sensitivity:
- DXY 3-month change
- gold/SPY ratio
- 10y-2y Treasury slope
- SPX put/call ratio
- VIX futures basis (front - second month)

Per WS-3 spec rev2 honesty discipline, sensitivity panels do NOT influence the gate verdict; the gate evaluates on the canonical 6-input set only.

## Architecture (pre-committed, Amendment 3)

**LightGBM gradient-boosted classifier** trained on the 6 canonical inputs with **purged combinatorial cross-validation (CPCV)** for hyperparameter selection.

**Targets**: 
- Primary: G1_BEAR binary label (drawdown > 10% from trailing 252-day high; from `scripts/diagnostics/ground_truth_labelers.py::label_g1_drawdown_bear`, locked at commit `9c48245`).
- Secondary (informational): G2_BEAR (forward-window) and G3_vol_spike for output regime mapping.

**Output**: continuous BEAR probability score in [0, 1] from `model.predict_proba(X)[:, 1]`. The 5 regime labels (STRONG_BULL, WEAK_BULL, SIDEWAYS, UNPREDICTABLE, BEAR) are mapped from a combination of BEAR probability + the v0-detector-style indicators retained as features (above_20, above_50, above_200, vix_percentile). This produces compatible `last_regime_scores` for V11+ consumers.

**Hyperparameters** (pre-committed defaults; CPCV sweep is bounded):
- `n_estimators`: 100, sweep {50, 100, 200, 500}
- `max_depth`: 4, sweep {3, 4, 5, 6}
- `learning_rate`: 0.05, sweep {0.01, 0.05, 0.1}
- `subsample`: 0.8, fixed
- `colsample_bytree`: 0.8, fixed
- `objective`: 'binary', fixed
- `metric`: 'binary_logloss', fixed
- Class balancing: `is_unbalance=True` (G1_BEAR is rare positive class)

The CPCV sweep is bounded to AT MOST 4 x 4 x 3 = 48 hyperparameter combinations. Selection methodology is documented to avoid trial-count inflation (the entire sweep counts as 1 trial, not 48).

**CPCV setup**:
- 6 folds with combinatorial assembly
- Purge: 5 trading days (longer than longest label window)
- Embargo: 5 trading days (longer than the longest feature lookback used)
- Reference: Marcos Lopez de Prado, "Advances in Financial Machine Learning" Chapter 7

**Alternative architectures (NOT in canonical; informational sensitivity only)**:

| Architecture | Rationale for exclusion from canonical |
|---|---|
| HMM-based (5-state hidden Markov) | EM training is unstable across rolling fits (mode collapse, label permutation). Would consume the trial-chain reset on methodology decisions. |
| Threshold ensemble (independent rules on each indicator) | Too simple to capture leading-indicator interactions (e.g., VIX term + HY OAS joint behavior in 2018-Q4). Less mathematically principled. |
| Logistic regression with feature engineering | Simpler than LightGBM but cannot capture non-linear indicator interactions documented in 2008/2020 crashes. |

These alternatives can run as sensitivity panels but do not gate.

## Variant family (V20+)

The new detector is consumed by a new RAMP variant family, V20+. The initial primary variant is **V20-rd-bear-cash** (rd = "replaced detector"; bear-cash mirrors V14a's consumption pattern for direct A/B comparison against the v0 family).

| Variant | Description |
|---|---|
| **V20-rd-bear-cash** | V11 base + BEAR_score (from WS-3d detector) Schmitt-trigger consumer, cash on enter. Canonical primary. |
| V20b-rd-bear-spy (reserved) | V11 base + BEAR_score Schmitt, SPY 100% on enter. Mirrors V14b. Only spec'd if V20 primary passes IS readiness. |
| V20c-rd-bear-dampen (reserved) | V11 base + BEAR_score Schmitt, V11*0.5 on enter. Mirrors V14c. Only spec'd if V20 primary passes. |

The choice to start with cash (V20-rd-bear-cash) and defer V20b/c to a conditional sub-spec is deliberate: V14a, V14b, V14c converged within 0.011 Sharpe on the v0 family (E8 finding). If the new detector also produces convergent actions, V20 family stops at V20-rd-bear-cash. If actions DO diverge under the new detector (the failed M2 mechanism finally has bite), V20b/c are spec'd.

n_trials_project for the WS-3d family at the readiness gate: **1** (V20-rd-bear-cash primary). If V20b and V20c land, n_trials grows to 3. Hyperparameter CPCV counts as 1 (per Pre-commitment 1's bounded-sweep discipline).

## Tau pre-registration

Tau threshold for the Schmitt trigger on the new detector's BEAR_score is derived from G1_BEAR median **on the new detector's outputs**, NOT inherited from the v0 family. The v0 `v14_tau_constants.json` (tau_in=0.5556, tau_out=0.4556) does NOT carry over -- the new detector's BEAR_score distribution is different.

Pre-spec script `scripts/diagnostics/compute_tau_in_ws3d.py` (analogous to `compute_tau_in_from_g1.py` but using WS-3d's detector output) runs once at spec time before any backtest. Output: `config/research/v20_tau_constants.json` pinning tau_in (median BEAR_score on G1_BEAR days) and tau_out = tau_in - 0.1.

The G1_BEAR labeler commit (`9c48245`) is pinned in the JSON for reproducibility, same as v14.

## Code structure

| File | Change |
|---|---|
| `src/data/leading_indicators/vix_term.py` (new) | Acquire & maintain VIX/VIX3M ratio. Daily refresh; backfill 2017-present. |
| `src/data/leading_indicators/fred_hy_oas.py` (new) | FRED API client for series BAMLH0A0HYM2. Daily refresh; backfill 1997-present (subset to 2017+). |
| `src/data/leading_indicators/breadth.py` (new) | NYSE A-D breadth computation: % of S&P 500 constituents above 50-day MA. Daily refresh; backfill 2017-present. |
| `src/data/leading_indicators/skew.py` (new) | CBOE SKEW index acquisition. Daily refresh; backfill 2017-present. |
| `src/data/leading_indicators/__init__.py` (new) | Unified loader: `load_leading_indicators(start, end) -> pd.DataFrame`. |
| `src/strategies/advanced/market_regime_detector_v1.py` (new) | `MarketRegimeDetectorV1` class with same public API as v0 but LightGBM-backed scoring. Co-exists with v0. |
| `src/strategies/advanced/market_regime_detector_v1/training.py` (new) | LightGBM training + CPCV pipeline. Outputs trained model artifact. |
| `src/research/ramp_phase4/variants.py` | Add `_variant_v20_rd_bear_cash` + REGISTRY entry. NO changes to existing variants. |
| `src/research/ramp_phase4/config.py` | Add `use_v1_detector: bool = False` field + V20 tau-loading helpers. NO changes to V11-V14 fields. |
| `scripts/diagnostics/compute_tau_in_ws3d.py` (new) | Pre-spec script: derive tau_in from G1_BEAR median on WS-3d detector output. |
| `scripts/diagnostics/regime_detector_v1_replay.py` (new) | Replay the new detector across 2017-2026, parallel to v0 replay. |
| `scripts/diagnostics/regime_detector_v1_diagnostic.py` (new) | Rerun H1-H5 diagnostic on WS-3d outputs; comparison against v0 baseline. |
| `scripts/backtest_scripts/ramp_phase4_v20_readiness.py` (new) | Readiness orchestrator for V20-family. Fresh n_trials chain (starts at 1). |
| `tests/data/leading_indicators/test_*.py` (new) | Unit tests for each indicator acquirer + the unified loader. |
| `tests/strategies/advanced/test_market_regime_detector_v1.py` (new) | Unit tests for the new detector, including a canonical pinning test. |
| `tests/research/ramp_phase4/test_variants.py` (modify) | Add V20-rd-bear-cash unit tests + canonical pinning test. |
| `config/research/v20_tau_constants.json` (new) | Output of compute_tau_in_ws3d.py; pins tau values + G1 labeler commit + WS-3d detector model artifact hash. |
| `docs/strategies/RAMP_VARIANTS.md` (modify) | Add V20+ family section; reserve V21/V22 slots for future detector-replacement variants. |
| `docs/reports/ramp/20260601_ws3d_regime_diagnostic_rerun.md` (new) | H1-H5 diagnostic results on new detector vs v0. |
| `docs/reports/ramp/20260615_ws3d_v20_readiness.md` (new) | V20-rd-bear-cash readiness gate output. |
| `docs/progress/20260615_RAMP_WS3D_V20_READINESS.md` (new) | Session log. |

## Validation gates

In sequence (each gates the next):

### Gate 0: Data ingestion (week 1)

All 4 leading-indicator acquirers produce 2017-present daily series with no gaps > 2 trading days. Daily refresh works. Tests pass.

### Gate 1: Detector training + diagnostic rerun (week 2)

1. LightGBM trains successfully with CPCV; no degenerate folds.
2. Detector replay produces complete 2017-2026 BEAR_score series.
3. H1-H5 diagnostic rerun: **H5 lag must decrease by >= 30%** (from 14d to <= 10d on the v0 baseline) per Pre-commitment 5. If not, the input set or architecture is wrong; STOP and revise spec before proceeding.

### Gate 2: Pre-spec tau registration (week 3)

`compute_tau_in_ws3d.py` runs successfully on the new detector output. `v20_tau_constants.json` committed with G1 labeler commit + model artifact hash.

### Gate 3: V20-rd-bear-cash IS readiness (week 3-4)

5-gate readiness per spec rev4:
- PSR > 0.95
- DSR > 0.95 (with n_trials_project = 1 for the WS-3d family; this is the trial-chain reset payoff)
- PBO < 0.5 (across V01, V11, V12, V14a, V20-rd-bear-cash = 5 variants)
- Gate 4 directional: `(nc - lag) <= max(0.2 * |nc|, 0.1)`
- Gate 5: cost floor > 0.30 AND no-regress vs V11 (Sharpe(@7.5bps lag) >= 0.9 * V11 reference)
- TIER 1 lift: Sharpe(@5bps nc) > V11 + 0.10

### Gate 4: Forward OOS validation (week 5-6)

Per Pre-commitment 2:
- Minimum 21 trading days (~1 month) of forward OOS data captured after model freeze.
- Forward OOS Sharpe must be > 0 OR synthesis documents explicit reason for accepting negative.
- No retraining during forward OOS window.

### Gate 5: OMR consumer audit (week 6, parallel to forward OOS)

Per Pre-commitment 3. Audit only; no OMR migration in this landing.

### Gate 6: Deployment decision (week 7)

Synthesis report aggregates Gates 0-5. Recommendation: deploy / forward OOS extension / TIER 4 close.

## Multi-trial budget

Pre-commitment 1 establishes the fresh chain. Inventory:

**WS-3d family (fresh chain)**:
| Trial | Strategy |
|---|---|
| 1 | V20-rd-bear-cash (primary) |
| 2 (reserved, only if V20 passes) | V20b-rd-bear-spy |
| 3 (reserved, only if V20 passes) | V20c-rd-bear-dampen |

**v0 family (preserved, not reset)**: 36 trials per V14 readiness audit.

The two families have separate `n_trials_project` counters in the experiment registry. Cross-family PBO is computed for diagnostic comparability but does NOT factor into the WS-3d readiness gate (per the reset justification).

## Risks and limitations

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| FRED HY OAS series has gaps or stops being published | Low | High | Backup: substitute with a derivable proxy (HYG yield - 10y Treasury). Documented in `fred_hy_oas.py` fallback path. |
| LightGBM overfits the small G1_BEAR positive class (~371 days in 2017-2026) | Medium | High | CPCV with purge/embargo addresses this directly. Class balancing via `is_unbalance=True`. If validation logloss diverges materially from training, halt and revise. |
| H5 lag does NOT improve by 30% (Gate 1 fail) | Medium | High | Stop and revise spec. If the input set or architecture is wrong, no readiness work is wasted. Choices: (a) revise indicator set, (b) revise architecture (HMM as fallback), (c) escalate to halt-or-redirect per parent WS-3 spec Appendix. |
| Forward OOS Sharpe negative even with IS pass | Medium | High | Per Pre-commitment 2, deployment is BLOCKED. Recommendation becomes extended forward OOS (~2-3 months) or TIER 4 close. |
| V20-rd-bear-cash fails PBO because the variant set {V01, V11, V12, V14a, V20} still has v0-family correlation through V11 | Medium | Medium | The fresh chain says n_trials = 1; PBO is independent of n_trials. If PBO fails, the diagnosis is that V11 + WS-3d's BEAR signal is still correlated with v0's BEAR signal via the underlying SPY/VIX data. Add diagnostic PBO over an orthogonal set {V01, V11, V14a, V20}. |
| Trial-chain reset is challenged on the grounds that V11 base is shared | Low | High | The reset is on the DETECTOR side, not the BASE side. V11's filter logic is unchanged; the detector's regime classification (the source of BEAR_score) is materially different. Document in the spec and in any synthesis. |
| LightGBM model artifact is large or hard to deploy | Low | Low | Persist with joblib or pickle; ~MB-scale; easily fits on EC2. Document the artifact hash in `v20_tau_constants.json`. |
| OMR audit reveals material regime divergence and OMR migration becomes a deployment-time blocker | Medium | Medium | The audit IS a deployment-time gate. If OMR diverges materially, RAMP can still deploy with the new detector (OMR continues on v0). Two-detector world for the deployment phase. |
| Implementation exceeds 5-7 week budget | Medium | Medium | Decision rule: if Gate 1 (diagnostic rerun) hasn't completed by end of week 2, escalate. The diagnostic gate is the cheapest informative gate. |
| LightGBM hyperparameter sweep is interpreted as trial inflation | Low | Medium | Bounded to AT MOST 48 combinations + class-balancing + purge/embargo settings; methodology pre-committed in this spec. The sweep counts as 1 trial, not 48. Document in registry. |

## Validation gates summary (sequential, must pass in order)

1. **Gate 0 (data)**: 4 indicators with full 2017-present coverage.
2. **Gate 1 (diagnostic)**: H5 lag reduction >= 30%.
3. **Gate 2 (tau)**: pre-spec tau constants committed.
4. **Gate 3 (IS readiness)**: 5-gate readiness passes (PSR, DSR, PBO, lag-degradation, cost+no-regress).
5. **Gate 4 (forward OOS)**: 1+ month forward Sharpe > 0.
6. **Gate 5 (OMR audit)**: deployment-time gate; not blocking implementation.
7. **Gate 6 (deployment decision)**: synthesis aggregates all gates.

## Timeline (Amendment 4: realistic)

| Week | Milestone |
|---|---|
| 1 | Data ingestion (4 leading indicators) + tests pass. |
| 2 | Detector training + diagnostic rerun. Gate 1 (H5 lag reduction) checked. STOP if H5 not reduced. |
| 3 | Pre-spec tau + V20-rd-bear-cash implementation + unit tests. |
| 4 | V20 readiness orchestrator + IS gates evaluated. Gate 3 checked. |
| 5 | Forward OOS validation begins (1 month). Gate 4 in progress. |
| 6 | OMR audit (Gate 5) in parallel with forward OOS. |
| 7 | Synthesis + deployment decision (Gate 6). |

**Total: 5-7 weeks** (3-4 weeks implementation + 1-2 weeks forward OOS + 1 week synthesis).

V11 paper validation A7 counter continues throughout. If V11 ships before WS-3d completes, the new detector targets the next iteration (incremental rollout). If V11 fails A7, RAMP has no production strategy during WS-3d implementation; document this risk in the parent ledger.

## What this spec does NOT do

- **Does not migrate V11/V12/V12c/V13/V14a/b/c** to the new detector. They continue consuming v0.
- **Does not deprecate v0 detector**. It remains in the codebase for the existing variant family's continuity.
- **Does not commit V20b/c**. Only V20-rd-bear-cash is the primary; b/c are reserved subject to V20 primary's success.
- **Does not migrate OMR** to the new detector. OMR continues on v0. WS-3d's deployment proposal includes the OMR audit as a deployment-time gate.
- **Does not run forward OOS against forward data not yet collected**. Forward OOS executes after the spec lands and the new detector is implemented.
- **Does not modify the original `MarketRegimeDetector` class**. The new detector is a new class (`MarketRegimeDetectorV1`).
- **Does not change V11's filter logic**. V11 is unchanged; only the upstream detector changes.
- **Does not deploy live**. Even a TIER 1 IS verdict + positive forward OOS produces a deployment RECOMMENDATION subject to the user's go/no-go.

## Open questions to resolve before implementation

1. **Which environment does the LightGBM training run in?** The existing fintech conda env should support LightGBM directly via `lightgbm` PyPI. Confirm with a smoke import in a test. Estimated ~5 minutes.

2. **Is FRED API access (no key needed) sufficient for daily BAMLH0A0HYM2 pulls, or do we need to register a key?** FRED's free API works without a key for low-volume usage. Production deployment may want a key for rate-limit protection. Decision: implement without key initially; switch if rate-limited.

3. **What's the cadence of the leading-indicator daily refresh?** Recommendation: refresh at 8 PM ET (after all daily closes settle), batch with existing data acquisition. Confirm with the data acquisition manager scheduler.

4. **Should the LightGBM model artifact be versioned in git or stored separately?** Recommendation: store artifact in `output/models/v20_detector/<git_sha>/model.lgbm` (gitignored; reproducible from training script + commit SHA + tau JSON pin).

5. **Is purged CPCV implementation already available in the codebase, or does it need to be written?** The codebase has `src/backtesting/validation/` but the existing purged-cv may not be combinatorial. Confirm with code search; if not present, implement in `src/backtesting/validation/cpcv.py` as part of Gate 1.

These are blockers for implementation; resolve in week 1.

## Success criteria

WS-3d as a spec succeeds if:

1. The 6 canonical inputs are pre-committed and the 5 alternative inputs are flagged as informational only.
2. LightGBM + CPCV architecture is pre-committed; HMM and threshold ensemble are alternatives only.
3. Forward OOS validation is mandatory and pre-registered.
4. n_trials_project reset is justified explicitly on three independent grounds.
5. The diagnostic rerun (Pre-commitment 5) is positioned as a gating check before readiness, not after.
6. Backwards-compat is non-breaking; V11-V14 consumers continue unchanged.

WS-3d as a research line succeeds if the V20-rd-bear-cash variant produces:
- A TIER 1 readiness verdict AND positive forward OOS Sharpe -- deploy candidate.
- A TIER 3 verdict on IS that subsequently passes forward OOS -- deployment recommendation with extended monitoring.
- A TIER 4 verdict OR negative forward OOS -- DECISIVE evidence that the regime-aware approach has reached its useful limit for RAMP. The campaign closes regime-detector work. This is itself a useful campaign-level decision.

## Appendix -- Fallback architectures (sensitivity only)

If Gate 1 (H5 lag reduction) fails for LightGBM, the spec can be revised to test an alternative architecture before halting. Order of fallback:

1. **HMM-based**: 5-state hidden Markov model with multivariate Gaussian emissions. Train via EM on the same input set. Output: posterior P(regime | history). Risk: EM instability across rolling fits. Trial counting: model selection (state count, emission distribution) must be pre-committed to avoid trial inflation.

2. **Threshold ensemble**: independent threshold rules on each of the 4 leading indicators, combined via weighted vote. Simpler than HMM; easier to validate. Risk: insufficient capacity to capture indicator interactions.

3. **Logistic regression with feature engineering**: simpler than LightGBM; lower variance. Risk: cannot capture non-linear interactions documented in 2008/2020 crashes.

These are documented here for completeness; they are NOT in scope for the canonical implementation. Activating them requires a spec amendment.
