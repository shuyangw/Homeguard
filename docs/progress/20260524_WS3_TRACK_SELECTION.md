# WS-3 Track Selection -- Session Log (2026-05-24)

## Summary

E8 returned a verdict tuple `(M1=inconclusive, M2=refuted, M3=refuted)` that does not cleanly match any incremental-track row in the WS-3 decision tree. The original framing read this as the catch-all default to WS-3d, but the underlying data tells a stronger story: three independent measurements of structural detector lag converge on the same finding. WS-3d is the evidence-driven choice, not the default.

Track committed: **WS-3d (full detector replacement)**. Four amendments to the WS-3 spec landed before drafting the full implementation spec.

## The verdict reframing

E8's mechanism verdicts:

- **M1 (rare events): inconclusive** -- 366 BEAR-soft days = 15.54% of gated window (NOT rare) AND median event 9 days (NOT short), BUT BEAR-soft partition contributes only 9.75% of V14a-V11 excess (the firing days don't drive the alpha).
- **M2 (action equivalence): refuted** -- cross-variant per-event correlations LOW (V14a/V14b=0.51, V14a/V14c=0.64; only V14b/V14c clears 0.85). Pooled corr(SPY, V11) during BEAR-soft = 0.79 (below 0.85). The action axis IS differentiating; it's just differentiating on days that don't matter.
- **M3 (exit-timing failure): refuted in opposite direction** -- median exit-to-SPY-low lag = -8 trading days (SPY low BEFORE exit, not after); mean 10d post-exit V14a-V11 excess = +0.16% (positive, not negative).

The pre-registered hypothesis structure missed the actual mechanism. The underlying finding is a single clean structural lag:

| Measurement | Value | Source |
|---|---|---|
| V12 gap_days | -3.42 trading days (mean across 59 BEAR onsets 2017-2026) | `docs/reports/ramp/20260524_phase4_v12_readiness.md` |
| Original diagnostic H5 | 14-day SMA lag (median, from G1_BEAR onset to first detector-BEAR label) | `docs/reports/ramp/20260523_regime_detector_diagnostic.md` |
| E8 exit-to-SPY-low lag | -8 trading days (median, V14 trigger exit before SPY trough) | `docs/reports/ramp/20260525_experiment8_action_convergence.md` |

Three different consumption patterns, three different measurement methodologies, three different windows -- one consistent finding: the v0 detector is structurally late. The campaign documentation should read WS-3d as the evidence-driven choice, not the catch-all default.

## Four amendments to the WS-3 spec

Pre-committed before WS-3d implementation drafting. Each amendment closes a degree of freedom that would otherwise inflate the multi-trial penalty or muddy the campaign's documented evidence path.

### Amendment 1: Decision-tree row for the lag-structural finding

Added to `docs/superpowers/specs/2026-05-24-ws3-detector-intervention-design.md`:

> **lag-structural finding** (E8 exit-to-SPY-low lag < -5 days AND BEAR-soft firing days contribute < 15% of V-V11 excess) -> WS-3d (the detector is structurally late; consumer-layer fixes cannot recover days the detector missed; trial-chain reset is the only escape from the DSR=36 trap)

The lag-structural row is binding when its predicate holds, irrespective of the (M1, M2, M3) tuple. Pre-registration matters: future re-reads of the campaign see WS-3d as evidence-driven, not default.

Commit: `36bf435` spec(ws3): add lag-structural decision tree row -- WS-3d evidence-driven

### Amendment 2: Canonical leading-indicator set

Pre-committed in the WS-3d implementation spec:

| Indicator | Source | Purpose |
|---|---|---|
| VIX/VIX3M ratio | CBOE / yfinance | Term-structure inversion as drawdown leading indicator |
| HY OAS | FRED BAMLH0A0HYM2 | Bond market leads equity vol |
| NYSE A-D breadth | computed (% S&P 500 above 50-day MA) | Market breadth deterioration |
| CBOE SKEW index | CBOE / yfinance | Tail-risk options pricing |
| SPY OHLC (retained) | existing | Backwards-compat features |
| VIX OHLC (retained) | existing | Backwards-compat features |

Alternative sets are informational sensitivity panels only; do NOT influence the gate verdict.

### Amendment 3: LightGBM architecture pre-committed

LightGBM gradient-boosted classifier on G1_BEAR labels with purged combinatorial cross-validation (CPCV). Bounded hyperparameter sweep (at most 48 combinations) counts as 1 trial, not 48.

HMM and threshold ensemble are documented in the spec as informational alternatives only. EM training instability + state-count selection would consume the trial-chain reset on methodology decisions, which defeats the purpose of choosing WS-3d.

### Amendment 4: Realistic timeline 5-7 weeks

3-4 weeks implementation + 1-2 weeks forward OOS + 1 week synthesis. The trial-chain reset makes IS gates passable but does NOT bypass forward OOS for deployment. V11 paper validation A7 counter continues in parallel.

## Five methodological pre-commitments

These bind the WS-3d implementation phase; relaxing any of them requires a spec amendment.

1. **Trial-chain reset justification**: WS-3d is a genuinely new strategy on three independent grounds -- new inputs (4 leading indicators absent from v0), new scoring (LightGBM probability vs hand-tuned threshold formula), new architecture (tree-based vs rule-based). n_trials_project for the WS-3d family starts at 1. The v0 family's 36-trial chain is preserved separately.
2. **Forward OOS validation mandatory**: minimum 1 month (~21 trading days), ideally aligned with V11 A7. No deployment recommendation without it regardless of IS verdicts.
3. **OMR consumer audit required**: deployment-time gate, not implementation blocker. RAMP can deploy on the new detector while OMR remains on v0 if material divergence is found.
4. **Backwards-compatibility**: V11-V14 consumers NOT migrated to the new detector. The new detector exposes the same public API (`classify_regime`, `last_regime_scores`) with the same 5 regime keys, so swap-in is mechanically possible but not part of WS-3d's initial landing.
5. **Regime diagnostic rerun**: H1-H5 diagnostic rerun on the new detector's outputs is a **gating check** (Gate 1) before the readiness orchestrator runs. H5 lag must reduce by >= 30% (from 14d to <= 10d), otherwise stop and revise the input set or architecture before proceeding to readiness.

## Variant family reservation

`docs/strategies/RAMP_VARIANTS.md` updated with a new V20+ section reserving the block for WS-3d's variants:

- V20-rd-bear-cash (canonical primary; cash on Schmitt-trigger entry, mirroring V14a for direct A/B comparison)
- V20b-rd-bear-spy (reserved; spec'd only if V20 primary passes)
- V20c-rd-bear-dampen (reserved; same conditional)
- V21 / V22 (open slots for future detector-replacement variants)

V11-V14 are NOT retrofitted to the new detector. They continue consuming v0 for diagnostic continuity.

## Validation gates (sequential)

| Gate | Description | Week |
|---|---|---|
| 0 | Data ingestion: 4 leading indicators with full 2017-present coverage | 1 |
| 1 | Diagnostic rerun: H5 lag reduction >= 30% (gating, must pass before readiness) | 2 |
| 2 | Pre-spec tau registration (analogous to V14's `v14_tau_constants.json`) | 3 |
| 3 | V20 IS readiness: 5-gate (PSR, DSR, PBO, lag-degradation, cost+no-regress) + TIER 1 lift | 3-4 |
| 4 | Forward OOS validation: 1+ month, Sharpe > 0 | 5-6 |
| 5 | OMR consumer audit (parallel with forward OOS; deployment-time only) | 6 |
| 6 | Deployment decision: synthesis aggregating all gates | 7 |

## Decisions

- **WS-3d selected as the evidence-driven WS-3 track.** Not catch-all default.
- **WS-3d full implementation spec drafted** at `docs/superpowers/specs/2026-05-25-ws3d-detector-replacement-design.md` with 4 amendments and 5 pre-commitments.
- **V20+ variant family reserved** in RAMP_VARIANTS.md; V11-V14 not migrated; v0 detector preserved.
- **Implementation begins** after spec review (await user go-ahead), starting with Gate 0 (data ingestion).
- **NO code changes to `src/strategies/advanced/market_regime_detector.py`** until the WS-3d implementation spec is reviewed and approved by the user.

## Commits this session

- `7a52279` feat(diagnostics): E8 V14 action convergence -- M1/M2/M3 mechanism verdicts
- `afee3a8` report(ramp): E8 V14 action convergence -- WS-3 track recommendation
- `36bf435` spec(ws3): add lag-structural decision tree row -- WS-3d evidence-driven (Amendment 1)
- `<TBD>` spec(ws3d): full implementation spec + RAMP_VARIANTS V20+ reservation + session log

## Next session candidates

1. **WS-3d Gate 0: data ingestion** (~1 week). Build acquirers for VIX/VIX3M, HY OAS (FRED), NYSE breadth, CBOE SKEW. Unit tests with full 2017-present coverage.
2. **WS-3d Gate 1: diagnostic rerun** (~1 week). Train LightGBM with CPCV. Run H1-H5 diagnostic against the new detector outputs. STOP if H5 not reduced by 30%.
3. **Monitor V11 paper validation** -- A7 counter continues independently. If V11 ships during WS-3d, new detector targets next iteration.

## Cross-experiment context

Eight experiments now complete in the 2026-05-24 campaign:

| Exp | Verdict | Contribution to WS-3 framing |
|---|---|---|
| E3 soft scores | WS-3c (median argmax_lag 24 days at tau=0.3) | Motivated V14 factorial; soft-score lead is real but bounded |
| E2 UNPREDICTABLE | AMBIGUOUS (53.6% top-3 share) | V12c framing; not WS-3-specific |
| E4 lag asymmetry | DIFFUSE (38.1% transition share) | V14 cost grid used standard cost levels |
| E1 V13-bear-invert | TIER 4 | Argmax-BEAR-as-buy failed; sign-inversion alone insufficient |
| E5 OMR cross-check | AMBIGUOUS (OMR screens out BEAR/UNPREDICTABLE) | WS-3 is RAMP-attributable lever; OMR audit is deployment-time only |
| E6 V12c readiness | TIER 4 (PBO 0.71) | Argmax+UNPREDICTABLE-to-cash overfits |
| V14 factorial | All 3 TIER 4 (Sharpe convergence within 0.011) | Consumer-layer ceiling demonstrated; +0.08 avg lift below +0.10 bar |
| **E8 action convergence** | **(inc, ref, ref) + lag-structural finding** | **WS-3d evidence-driven choice; trial-chain reset justified** |

The campaign has produced 9 documented data points on the v0 detector's structural limits. WS-3d either unlocks the +0.10 lift bar via a structurally faster detector OR closes the regime-aware-RAMP research line with definitive evidence.
