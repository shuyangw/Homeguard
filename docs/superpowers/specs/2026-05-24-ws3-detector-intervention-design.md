# WS-3: Detector Intervention -- Design (track-conditional)

**Date**: 2026-05-24
**Status**: Proposed (track selection conditional on Experiment 8 outcome)
**Branch**: TBD (post-E8 verdict)
**Owner**: Shuyang
**Type**: High-level intervention spec; full implementation specs follow track selection
**Builds on**:
- `docs/reports/ramp/20260526_phase4_v14_factorial_readiness.md` (consumer-layer ceiling demonstrated)
- `docs/reports/ramp/20260523_regime_detector_diagnostic.md` (H4 hysteresis absent, H5 SMA lag confirmed)
- `docs/superpowers/specs/2026-05-24-experiment8-action-convergence-diagnostic.md` (track-selection diagnostic; E8)
- Six-experiment campaign 2026-05-24 (E1-E6 + V14 factorial + E8)

## Position

The 2026-05-24 campaign has produced converging evidence that the binding constraint on RAMP's regime-aware performance lives at or near the production detector, not at the consumer layer. Six pieces of evidence support this reading:

1. **Argmax consumption fails**: V12 BEAR-to-cash, V12c BEAR+UNPREDICTABLE-to-cash, V13 argmax-BEAR-to-SPY all closed TIER 4. The argmax label is too late and too noisy for any consumer-layer use.
2. **Soft-score consumption is materially better but still bounded**: V14a/b/c at Sharpe [0.6035, 0.6146] vs V11 0.5279 is a real +0.08 average lift, but doesn't clear the +0.10 TIER 1 bar at n_trials=36.
3. **Action axis is saturated**: V14a/b/c converged within 0.011 Sharpe. The choice of action on BEAR-soft entry doesn't meaningfully differentiate outcomes.
4. **Multi-trial budget is exhausted on consumer-layer variants**: n_trials_project=36 means any further consumer variant inherits a DSR penalty that requires unrealistic Sharpe lift to overcome.
5. **The original regime diagnostic confirmed structural detector failure modes**: H4 (no hysteresis, median run length <= 2 days for 4 of 5 regimes) and H5 (median 14-day SMA lag to first BEAR label) are both supported.
6. **OMR cross-check was inconclusive** (E5), but the diagnosed failure modes are intrinsic to the detector regardless of how downstream strategies consume it.

WS-3 is the response: an intervention at or near the detector layer. This spec is intentionally track-conditional because the four candidate tracks address different mechanisms with different implementation costs and different multi-trial implications. The diagnostic (E8) returns a mechanism verdict, and this spec's decision tree maps that verdict to a track. Track selection is the only deliverable; full implementation specs are written after track selection.

## Tracks under consideration

Four tracks. Each track is described at sketch fidelity -- enough to compare against the others, not enough to implement. After E8 returns, the selected track receives a full implementation spec (analogous to the V12 rev4 or V14 rev2 specs).

### WS-3a -- Detector-internal hysteresis

**Intervention**: Add a Schmitt-trigger or N-day persistence layer at the detector's output (argmax label and/or soft scores), smoothing flicker between adjacent regimes without changing the underlying classification logic or inputs.

**Mechanism it targets**: H4 (no hysteresis); M2 (action equivalence under flickering trigger).

**Architecture sketch**: Insert a state-machine wrapper around `MarketRegimeDetector.classify_regime` that applies a debouncing layer. Options for the wrapper:
- Symmetric N-day persistence: regime change requires N consecutive days at the candidate regime before output flips.
- Score-threshold hysteresis: entry into a regime requires score > tau_enter, exit requires score < tau_exit (Schmitt analog to V14's consumer-side trigger but operating on the argmax inputs).
- Hybrid: score crossing + minimum dwell time.

**Multi-trial cost**: +1 to existing detector family (the underlying detector is the same; only the wrapper differs). The new detector counts as a single trial on top of the existing 36.

**Implementation cost**: ~3-5 days. Wrapper class, unit tests including a canonical pinning test, plus rerun of the regime diagnostic harness against the new detector for comparability with the v0 results.

**Pros**: smallest detector change; preserves all existing inputs and scoring logic; directly addresses one of two supported hypotheses from the original diagnostic.

**Cons**: does not address H5 (SMA lag). If the signal is already late, smoothing it makes it later -- entry to BEAR-smoothed mode happens after N days of confirmation, where N >= 2 trading days, compounding existing lag. The Schmitt-trigger variant avoids this by being event-driven on score thresholds, but at the cost of being harder to validate.

**When this is the right track**: E8 verdict supports M2 (action equivalence -- trigger right but flickering) and refutes M1 (rare events).

### WS-3b -- Leading indicators

**Intervention**: Augment or replace the SMA-based detector inputs with leading indicators. Candidate inputs:
- VIX term structure (VIX/VIX3M ratio) -- inverts before drawdowns.
- High-yield credit spreads (HY OAS from FRED) -- bond market leads equity vol.
- Market breadth (NYSE advance-decline line, or % of S&P 500 above 50-day MA).
- Options skew (CBOE SKEW index) -- tail-risk premium leads realized drawdowns.
- Cross-asset (DXY 3-month change, gold/SPY ratio).

**Mechanism it targets**: H5 (SMA lag); M1 (rare events -- detector firing too late, too rarely on the right days).

**Architecture sketch**: Two sub-variants:
- **WS-3b.1 augmentation**: add 1-2 leading indicators as additional inputs to the existing score formula. Re-weight scores. Minimal change to detector architecture.
- **WS-3b.2 replacement**: rebuild the BEAR score from leading indicators only, treating SMAs as confirmation rather than primary signal. Larger architectural change.

**Data dependencies**: New. VIX3M from yfinance or CBOE direct, HY OAS from FRED, breadth from Alpaca or yfinance, SKEW from CBOE. All free or already-budgeted sources. Pipeline work: ~1-2 days per indicator for ingestion, cleaning, gap-filling.

**Multi-trial cost**: +1 per indicator combination tested. To avoid trial inflation, pre-register a single canonical indicator set (e.g., "VIX term structure + HY OAS + breadth") and treat that as the one trial. Sensitivity panels on alternative sets are informational only, per the V14 rev2 pattern. Conservative estimate: WS-3b adds 2-3 trials, with subsequent sub-variants paying the standard +1 each.

**Implementation cost**: ~2-3 weeks. Data ingestion, indicator transformation, score formula redesign, validation methodology, regime diagnostic rerun on new detector outputs. Significantly more complex than WS-3a.

**Pros**: addresses the root structural lag identified in the original diagnostic; the only track that can plausibly reduce H5's 14-day lag materially.

**Cons**: highest data dependency; harder to validate (multiple data sources, indicator combinations); inherits the existing multi-trial chain even with discipline; risks adding noise as readily as signal if indicators don't actually lead in the relevant regime.

**When this is the right track**: E8 verdict supports M1 (rare events -- need detector to fire more often) and refutes M2/M3.

### WS-3c.1 -- Consumer-layer exit logic

**Intervention**: Modify V14's exit trigger from "BEAR_score < tau_out" to a forward-looking exit criterion. Candidate exits:
- Exit when SPY closes above its 20-day high (momentum-confirmed recovery).
- Exit when VIX retreats below a percentile threshold (volatility-confirmed normalization).
- Exit when BEAR_score has been below tau_out for K consecutive days (consumer-side debouncing).
- Exit when forward-confirmed drawdown reversal (SPY's trailing 5-day return > 0 AND BEAR_score declining).

**Mechanism it targets**: M3 (exit-timing failure); does not target H4 or H5.

**Architecture sketch**: V14 trigger entry logic unchanged; replace the exit branch of the state machine. The variant remains V14-family (operates on existing detector's soft scores) but introduces a richer exit-side state. The most attractive option is the SPY-recovery-confirmed exit because it's market-state-driven rather than detector-state-driven, breaking the dependency on tau_out tuning.

**Multi-trial cost**: +1 per exit-logic variant. Strict discipline required because the V14 family is already at 36 trials -- adding multiple exit-logic variants quickly exceeds the campaign's coherent trial budget.

**Implementation cost**: ~2-3 days. State machine modification, unit tests including canonical pinning tests, readiness orchestrator rerun. Smallest scope of any WS-3 track.

**Pros**: smallest scope; smallest data dependency (uses existing inputs); preserves V14 architectural investment; testable in a single orchestrator run.

**Cons**: another consumer-layer fix in a campaign that has shown the consumer layer is constrained; doesn't address detector lag or flicker; if M3 isn't actually the binding mechanism, this track produces another TIER 4 verdict with no useful diagnostic information.

**When this is the right track**: E8 verdict supports M3 (exit-timing failure) and refutes M1/M2. Particularly attractive if the counterfactual tau_out sweep in E8's analysis A6 shows V14a's Sharpe materially exceeding the V11+0.10 bar at lower tau_out values.

### WS-3d -- Detector replacement

**Intervention**: Build a new detector from leading indicators with a new scoring architecture, treating it as a separate strategy rather than an iteration on the existing detector. The replacement detector is consumed via V11-family variants or a new strategy family, with a fresh multi-trial chain.

**Architecture sketch**: Several plausible architectures:
- **HMM-based**: 5-state hidden Markov model on SPY returns + VIX + 1-2 leading indicators. Emissions multivariate Gaussian or t-distributed. Outputs posterior P(regime|history), consumed as soft scores. Persistence built into transition matrix diagonal.
- **Threshold ensemble**: independent threshold rules on 3-5 leading indicators (VIX term, HY OAS, breadth, drawdown), combined via weighted vote. Simpler to validate than HMM; less mathematically principled.
- **Continuous regime probability via gradient-boosted classifier**: train LightGBM or XGBoost on leading indicators with G1_BEAR (drawdown > 10%) as the label. Output as probability score, consumed via Schmitt trigger. Requires careful purged cross-validation to avoid label leakage.

**Mechanism it targets**: H4 + H5 + M1 + M2 simultaneously by virtue of replacing the entire detector architecture.

**Multi-trial cost**: **resets the chain**. The new detector is a different strategy with different inputs and different scoring, not a variant of the existing detector. Fresh n_trials_project starts at 1 + however many variants are tested. This is the principal architectural advantage.

**Implementation cost**: ~3-5 weeks. Data ingestion (subset of WS-3b's), new detector implementation, validation methodology, full readiness gating, OMR consumer adapter (since OMR also uses the detector), and migration plan. Significantly the largest scope.

**Pros**: only track that exits the multi-trial trap; addresses root causes structurally; portfolio-level benefit if both RAMP and OMR consume the new detector.

**Cons**: largest scope; highest architectural risk; longer time-to-evidence; requires consumer-side migration work for any strategy currently consuming the old detector.

**When this is the right track**: E8 verdict supports multiple mechanisms simultaneously, OR no clean single-mechanism verdict, OR the analyst judges incremental tracks have reached their useful limit independent of E8.

## Decision tree from E8

The diagnostic returns a verdict tuple `(M1, M2, M3)` where each is in {supported, refuted, inconclusive}. The mapping to tracks is:

| (M1, M2, M3) | Selected track |
|---|---|
| (sup, ref, ref) | WS-3b (leading indicators) |
| (ref, sup, ref) | WS-3a (detector hysteresis) |
| (ref, ref, sup) | WS-3c.1 (consumer exit logic) |
| (sup, sup, ref) | WS-3d (detector replacement) -- two structural targets |
| (ref, sup, sup) | WS-3a + WS-3c.1 in parallel -- smaller scopes, complementary |
| (sup, ref, sup) | WS-3b primary; WS-3c.1 as fallback if WS-3b stalls |
| (sup, sup, sup) | WS-3d -- incremental tracks won't compound enough |
| any 2+ inconclusive | WS-3d with expanded scope |
| (ref, ref, ref) | WS-3d OR halt WS-3 entirely (campaign has hit a wall; explore alternative strategies) |
| **lag-structural finding** (E8 exit-to-SPY-low lag < -5 days AND BEAR-soft firing days contribute < 15% of V-V11 excess) | **WS-3d** (the detector is structurally late; consumer-layer fixes cannot recover days the detector missed; trial-chain reset is the only escape from the DSR=36 trap) |

The lag-structural row is the binding row when its predicate holds, irrespective of the (M1, M2, M3) tuple. It captures the campaign-level finding that three independent measurements (V12 gap_days = -3.42 trading days, original diagnostic H5 = 14-day SMA lag to first BEAR label, E8 exit-to-SPY-low lag = -8 trading days) converge on a structural detector lag. When the predicate holds, WS-3d is selected as the evidence-driven choice rather than as a decision-tree catch-all.

Track selection is committed at the time E8 returns. The selected track receives a full implementation spec; non-selected tracks are filed for potential later consideration but do not block.

## Multi-trial budget considerations

The campaign's n_trials_project is at 36 as of the V14 factorial readiness. Each WS-3 track has different implications:

**WS-3a (hysteresis wrapper)**: n_trials becomes 37 for the wrapper itself, +1 per V11-family variant that consumes the wrapped detector. A clean implementation tests V11-w (V11 with wrapped detector) as a single new variant. Estimated post-WS-3a n_trials: 37-38.

**WS-3b (leading indicators)**: n_trials becomes 37 for the augmented detector + 1 per V11-family variant tested. If WS-3b explores 1-2 indicator combinations, post-WS-3b n_trials: 38-40. Sensitivity panels on alternative combinations must remain informational only.

**WS-3c.1 (consumer exit logic)**: n_trials becomes 37+ for each exit-logic variant. If WS-3c.1 tests 2-3 exit rules, post-WS-3c.1 n_trials: 38-40. This pushes DSR threshold materially.

**WS-3d (detector replacement)**: **resets the trial chain**. Post-WS-3d the new strategy family starts at n_trials = 1 + new variants. This is the dominant reason to consider WS-3d even when E8 supports a single incremental mechanism: the trial-chain reset is itself valuable.

At post-WS-3 n_trials of 38-40 (the incremental tracks), the DSR threshold is binding for any future variant. Forward OOS becomes mandatory for deployment regardless of IS gate verdicts. WS-3d's trial reset is the only escape from this constraint.

## Validation methodology requirements

Regardless of track, any WS-3 deliverable must satisfy:

1. **Regime diagnostic rerun**: the original H1-H5 diagnostic is rerun on the new detector's outputs. Comparisons against v0 detector are reported in the synthesis. Provides direct evidence that the intervention addresses the targeted hypothesis.

2. **Full 5-gate readiness**: PSR, DSR (with correct n_trials), PBO (with appropriate variant set), Gate 4, Gate 5. Same as V12/V14 spec rev4.

3. **Forward OOS requirement**: at least one full month of forward-OOS data on the new detector's outputs before any deployment recommendation. Walked back from V11's A7 5-session timer: WS-3 deployment cannot be on a timer shorter than V11's.

4. **OMR consumer audit**: any change to the production detector (WS-3a or WS-3b augmentation) potentially affects OMR. The WS-3 spec must include an OMR-impact section showing whether OMR's per-day P&L by regime changes materially under the new detector. If yes, OMR's adapter requires its own gating before the detector rolls forward.

5. **Backwards-compatibility audit**: if the new detector changes the `last_regime_scores` schema or `classify_regime` signature, all downstream consumers (V11, V12, V12c, V13, V14a/b/c, OMR adapter, monitoring stack) must be inspected for breaking changes. A non-breaking change is preferred; a breaking change requires a migration plan.

These are spec-level requirements that bind whichever track is selected.

## Risks and limitations

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| E8 returns inconclusive verdicts and the decision tree defaults to WS-3d | Medium | Medium | Pre-acknowledged in the decision tree. WS-3d is the explicit fallback. The campaign can also choose to halt WS-3 if E8 plus prior evidence suggests the regime-detector approach itself is fundamentally limited for RAMP. |
| Selected track produces a TIER 4 readiness verdict | Medium | High | Forward OOS still possible if structural gates pass and lift bar fails. If all gates fail, fall back to next-best track from the decision tree, OR escalate to WS-3d. |
| WS-3d's fresh trial chain is challenged on the grounds that the campaign has been continuous | Low | High | Pre-register the trial-chain reset in the WS-3d spec with explicit justification: new detector, new inputs, new scoring formula, new data dependencies. This is a different strategy in the methodologically relevant sense. |
| OMR consumer adapter work is larger than estimated and delays WS-3 deployment | Medium | Medium | Scope OMR impact in the selected track's spec before implementation begins. If material, OMR work parallels WS-3 implementation rather than blocking it. |
| Multi-trial penalty exceeds plausible alpha for any incremental track (WS-3a, WS-3b, WS-3c.1) | High | Medium | This is the case for considering WS-3d even when E8 supports an incremental mechanism. The trade-off is implementation cost vs trial-chain reset. The analyst's judgment, not a mechanical rule, decides this. |
| WS-3a's hysteresis wrapper compounds existing detector lag | Medium | Medium | The Schmitt-trigger variant (score-based, event-driven) avoids this. If WS-3a is selected, prefer Schmitt over N-day persistence. Pre-register in the WS-3a implementation spec. |
| WS-3b adds noise instead of signal because leading indicators are leading in irrelevant regimes | Medium | Medium | Pre-register validation criteria for each indicator: must reduce H5 lag by at least 30% (e.g., median lag from 14d to <= 10d) on the regime diagnostic rerun, otherwise indicator is dropped from the canonical set. |
| Track-conditional structure delays the actual WS-3 work | Low | Low | Track selection is fast once E8 returns (~half day for the full implementation spec draft). Total delay vs writing a single non-conditional spec is small. |

## What this spec does NOT do

- **Does not commit to a track**. Track selection happens after E8 returns and is committed in the selected track's full implementation spec.
- **Does not write implementation-level pseudocode**. Each track sketch is at design fidelity, not implementation fidelity.
- **Does not pre-commit data dependencies**. WS-3b's indicator set is chosen at full-spec time. WS-3d's architecture is chosen at full-spec time.
- **Does not block V11 paper validation**. A7 counter continues independently. WS-3 implementation runs in parallel.
- **Does not deprecate existing variants**. V11, V12, V12c, V13, V14a/b/c remain in REGISTRY for diagnostic continuity even after WS-3 deployment.
- **Does not promise deployment**. WS-3 may produce a TIER 4 verdict, in which case the recommendation is forward OOS, escalation to WS-3d, or halt -- not deployment.
- **Does not address non-detector RAMP improvements**. Universe expansion, regime-orthogonal features, alternative signal stacks all remain out of scope for this spec.

## Open questions to resolve before track-specific spec

These are deferred to E8's resolution but flagged here:

1. **What is the multi-trial accounting rule for an HMM detector trained via EM on the same window the diagnostic ran on?** WS-3d's HMM variant may count as multiple trials if model selection (state count, emission distribution) is data-driven. Pre-register the selection methodology to avoid trial-count inflation.

2. **Does OMR's consumption of the detector tolerate a soft-score interface change, or is it strictly argmax-coupled?** Determines WS-3 spec's OMR-impact scope. Confirmable in ~1 hour by inspecting OMR's adapter code.

3. **For WS-3b, is FRED's HY OAS data available at daily frequency with sufficient history (2017-2026)?** Affects whether HY OAS is a viable candidate input. ~30 minutes to confirm.

4. **For WS-3d, is the new detector consumed by a new RAMP variant family (V20+) or by retrofitting existing variants?** Architectural decision. Default recommendation: new variant family, since the new detector + retrofitted V11 is still a different strategy than V11+old-detector, and conflating them muddies the trial chain.

5. **What is the canonical leading-indicator set for WS-3b's primary trial?** Pre-registration matters; the analyst's degrees of freedom in selecting indicators is itself a source of multi-trial penalty. Recommendation: defer to WS-3b spec time but commit to a single set before any backtest runs.

## Sequencing

The recommended sequence relative to E8 and V11:

1. **E8 executes** (~half day, in parallel with this spec's drafting).
2. **E8 synthesis report lands** with mechanism verdicts.
3. **WS-3 track committed** based on E8 decision tree (~30 minutes).
4. **Full implementation spec for selected track drafted** (~1-2 days, similar fidelity to V14 rev2).
5. **Implementation begins** per the selected spec.

Total time from this spec's authorship to WS-3 implementation start: ~3-5 days.

V11 paper validation A7 counter continues throughout. If V11 fails A7 during WS-3 implementation, WS-3 priority does not change -- WS-3 addresses a structural detector failure that V11 inherits regardless.

If V11 succeeds A7 (clears 5 clean sessions) before WS-3 implementation completes, V11 enters live deployment with the production detector. WS-3 then targets the live-deployed strategy's underlying detector and the deployment can be rolled forward incrementally.

## Success criteria

This spec succeeds if:

1. The four tracks are described at sufficient fidelity that a knowledgeable reviewer can compare them.
2. The E8 decision tree maps every verdict combination to a track recommendation.
3. The multi-trial budget implications are honest and explicit for each track.
4. The validation methodology requirements are pre-registered before track selection.

The selected track's full implementation spec succeeds on its own criteria (analogous to V14 rev2's success criteria).

WS-3 as a research line succeeds if the selected track produces a TIER 1 readiness verdict OR if its TIER 4 verdict provides definitive evidence that further incremental detector work is unproductive (which is itself a useful campaign-level decision).

## Appendix -- Why not all four tracks in parallel

Tempting alternative: run WS-3a + WS-3b + WS-3c.1 + WS-3d as four parallel work streams, let the best one win. Reasons not to:

1. **Compute and analyst time aren't free**. Even at the spec-and-implementation level, four full specs and four implementations are ~6-10 weeks of work.
2. **Trial-budget interactions**. Running multiple tracks in parallel multiplies the multi-trial penalty. By the time all four complete, n_trials would be ~45-50 across the campaign.
3. **Result interpretation becomes harder**. If three of four tracks produce TIER 3 and one produces TIER 1, the TIER 1 candidate is hard to attribute -- did it succeed because the mechanism was correct, or because the parallel exploration found one path among four?
4. **Sequential learning is faster than parallel discovery**. WS-3a's results inform WS-3b's design (e.g., whether hysteresis-then-add-indicators or indicators-then-add-hysteresis is the right composition). Parallel exploration foregoes this.

The decision-tree approach is the right discipline for this campaign at this stage.

## Appendix -- Halt option

The decision tree's `(ref, ref, ref)` row (all mechanisms refuted by E8) recommends either WS-3d or halting WS-3. The halt option deserves explicit treatment.

If E8 returns no support for any of M1/M2/M3, the most honest reading is that the V14 convergence is not driven by any specific addressable mechanism. The consumer-layer fix produced as much as it could (+0.08 Sharpe), and the action choice doesn't differentiate. Further incremental work on the detector is unlikely to change this materially.

In that case, the right campaign-level decision may not be WS-3d but rather to **halt the regime-detector line of work for RAMP entirely** and redirect to:
- Universe expansion (S&P 500 + NASDAQ-100, Russell 1000) per prior planning.
- Alternative signal stacks (factor moderation, ML-based ensembling).
- Cross-strategy correlation work (RAMP-OMR portfolio construction).
- The Darwinex-inspired FX strategy in the FX expansion roadmap.

WS-3d in this scenario is defensible but expensive; halting is also defensible. The decision is for the analyst, not for this spec.

Halt is not failure. The campaign has produced six experiments worth of evidence about the detector's behavior and consumer-layer ceiling. That is permanent infrastructure for any future regime-aware strategy work.
