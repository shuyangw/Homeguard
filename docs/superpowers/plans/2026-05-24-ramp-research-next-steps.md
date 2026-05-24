# RAMP Research Next Steps -- 2026-05-24

**Date**: 2026-05-24
**Status**: Proposed (pre-execution)
**Owner**: Shuyang
**Type**: Research planning
**Supersedes (partially)**: `docs/superpowers/plans/2026-05-23-ramp-research-roadmap.md` -- WS-3 not yet decomposed
**Builds on**:
- `docs/progress/20260524_RAMP_RESEARCH_STATE.md` (current state synthesis)
- `docs/reports/ramp/20260523_regime_detector_diagnostic.md` (H4/H5 supported)
- `docs/reports/ramp/20260524_phase4_v12_readiness.md` (Tier 3 verdict; lag-asymmetry data)

---

## Position

The state document leaves three queued items: WS-3a (detector hysteresis), WS-3b (leading indicators), and V12c (UNPREDICTABLE-to-cash). These are the right items but the wrong sequence. Several cheap experiments produce information that materially shapes WS-3's design; running WS-3 before them is the highest-risk scheduling error currently available.

This document proposes six additional experiments, sequences them against the existing queue, and identifies the minimum work that should happen before WS-3 spec work begins.

The connecting thread: every V12 finding points at the detector, but the V12 work only measured the detector's *RAMP-strategy-level* impact through one specific consumption pattern (regime-conditional position arrays). The detector has internal state, scales across strategies, and may already contain leading information that's being suppressed by argmax. Until those things are characterized, choosing between hysteresis and leading indicators is partly a guess.

## Six proposed experiments

### Experiment 1 -- BEAR-as-buy signal test

**Hypothesis**: The detector's BEAR onset is empirically a *bottom marker*, not a *top marker*. `gap_days = -3.42` from V12's onset alignment panel means the detector fires ~3.4 trading days *after* the SPY drawdown trough on average across 59 events 2017-2026. If this lag is systematic and not artifactual, going long on BEAR onset should produce positive forward returns.

**Why this matters**: If true, every regime-aware variant we've built has been using the BEAR signal backwards. The fix is not detector improvement; it is sign inversion at the consumption layer. This is the highest-impact result available from the existing data, and it costs almost nothing to test.

**Method**: Backtest a single variant -- V13-bear-invert -- that on BEAR onset *increases* SPY exposure (e.g., position arrays unchanged in non-BEAR regimes, allocate to SPY 100% during BEAR). Run the existing V12 readiness orchestrator unchanged but with this single variant swapped in. The orchestrator already does 5 bps near_close, 5 bps one_day_lag, 7.5 bps stress, and the no-regress vs V11 check. No new infrastructure required.

**Validation discipline**: This was discovered from looking at the V12 onset-alignment panel, which is part of EXT-OOS. There is therefore a multi-trial correction issue -- DSR n_trials should include this as a new trial. PBO must be recomputed including V13.

**Cost**: 1-2 days. Most of the cost is interpretation (running the orchestrator is ~16 min).

**Decision criterion**: If V13 passes the same 5 gates as V12 evaluated against (PSR, DSR with corrected n_trials, PBO, Gate 4 directional, Gate 5 cost floor), and Sharpe exceeds V11 by > 0.10 at 5 bps near_close, declare BEAR-as-buy a real finding and re-evaluate the entire WS-3 roadmap. If V13 fails any gate, file the result and proceed with the rest of this document.

**Risk**: The result is contaminated by the fact that V12's onset-alignment panel was inspected before V13 was specified. The honest framing is that V13 is *not OOS* in the strict sense; it requires later validation on data the analyst has not seen.

### Experiment 2 -- UNPREDICTABLE hand-inspection

**Hypothesis**: V12-up-cash's +0.06 Sharpe over V11 is driven by 1-3 specific events, not by the regime as a whole. UNPREDICTABLE fires only 1.7% of days (14 runs total across 9 years), which is small-N territory where a single 2020-03 event could carry the result.

**Why this matters**: V12c is currently queued for V12.1.0 deployment. If the alpha source is concentrated in 1-3 events, V12c is a fragile fluke and the deployment decision changes. If the alpha is spread evenly across the 14 firings, V12c is a real (if rare-event-dependent) signal worth a readiness gate.

**Method**: Pull the 14 UNPREDICTABLE event windows from the diagnostic's `regime/v0/labels.parquet`. For each event: (a) start and end dates, (b) SPY return during the event, (c) SPY return in the 5/10/20 days following the event end, (d) V12-up-cash's avoided-loss attribution for that event. Tabulate. Compute the cumulative Sharpe contribution of the top-3 events vs the full 14.

**Cost**: Half a day. Pure analysis against existing data; no new backtests.

**Decision criterion**: If top-3 events contribute > 75% of V12c's Sharpe delta vs V11, mark V12c as fragile and *do not* run Experiment 6 (V12c formal readiness). If top-3 events contribute < 50%, V12c is robust enough to test formally, proceed to Experiment 6.

**Risk**: Confirmation bias. The analyst has already seen V12-up-cash beat V11 and may rationalize either outcome. Pre-commit to the decision criterion in writing (in this document) before running the analysis.

### Experiment 3 -- Soft-score extraction from the detector

**Hypothesis**: The detector is a score-based argmax. The argmax lag (H5: median 14 days) is the lag of the *winning* score crossing the *losing* scores. The underlying BEAR score may rise meaningfully earlier -- possibly days or weeks earlier -- before crossing the competing regime scores.

**Why this matters**: This is the single most informative experiment for WS-3 design. If the BEAR soft score leads the argmax by 5+ days on average, then the detector's signal is already there and is being suppressed by argmax; the fix is consumption-layer (use scores not labels), not detector-layer (rewrite the classifier). If the BEAR score is itself flat or lags, the fix must be at the detector inputs (Option D, leading indicators). The answer disambiguates Options B vs D in the diagnostic ranking.

**Method**: Modify `scripts/diagnostics/regime_detector_replay.py` to log all 5 per-regime soft scores (not just the argmax label) alongside the existing per-day record. Re-run the replay over 2017-2026. Produce three plots:
1. Per-event BEAR-score trajectory aligned to drawdown trough date (event-study format) across the 59 BEAR onsets and 5 G4 events.
2. Cross-correlation of BEAR_score with forward SPY drawdown at horizons 1d, 5d, 10d, 20d.
3. Threshold sweep: for each candidate threshold tau in {0.2, 0.3, 0.4, 0.5}, compute the median lag from "BEAR_score crosses tau" to "BEAR becomes argmax" and to "SPY drawdown trough."

**Cost**: Under a day. The driver already exists; this adds 5 columns to its output schema. Analysis is straightforward.

**Decision criterion**:
- If BEAR_score leads argmax by > 3 days at threshold tau = 0.3, declare a soft-score variant viable; spec WS-3c (consume soft scores, not argmax) ahead of WS-3a and WS-3b.
- If BEAR_score is approximately coincident with argmax, the lag is upstream of the classifier; WS-3b (leading indicators) takes priority.
- If BEAR_score is noisy and uncorrelated with forward drawdown, the detector cannot be salvaged without leading-indicator augmentation; WS-3b is necessary; WS-3a is not sufficient.

**Risk**: The soft scores may not be exposed by the current detector API. Phase 0 of the diagnostic established that the detector is score-based but did not document whether scores are returned to callers or only used internally. If only internal, the replay driver needs to call the scoring methods directly, which is a small refactor. Confirm during scoping.

### Experiment 4 -- V12 lag-asymmetry diagnosis

**Hypothesis**: V11 gains +0.052 Sharpe from one_day_lag execution. V12 gains +0.397. The 8× ratio implies V12 is paying same-day-as-regime-flip transaction cost that V11 does not pay. The likely mechanism: V12 transacts to cash on BEAR onset, when SPY is already moving with the regime signal that triggered the detector; one_day_lag lets that price move complete before transacting.

**Why this matters**: This is a deployment-realism issue. Live execution is near_close, not one_day_lag. Any variant that derives its alpha from cash-transition timing (V12, V12c, V13) inherits this exposure. The V12 readiness report passed Gate 4 (directional: nc - lag <= 0.100) because the spec rule permits lag > nc, but the *magnitude* of the asymmetry was not characterized. V12c readiness should not run without this characterization.

**Method**: Restrict V12's near_close vs one_day_lag results to BEAR-onset and BEAR-exit days only. Decompose the +0.397 Sharpe gap into (a) BEAR-onset transitions, (b) BEAR-exit transitions, (c) non-regime-flip days. The expectation is that (a) and (b) together account for nearly all the asymmetry, but quantifying this matters for whether the asymmetry is BEAR-specific or generalizes to all regime transitions.

**Cost**: Half a day. Existing V12 backtest output should contain per-day P&L; subset by transition-day flag and recompute.

**Decision criterion**: If > 80% of the lag asymmetry localizes to regime-transition days, this is a confirmed structural property of cash-transition variants and the V12c readiness gate must include a stricter cost-sensitivity test (e.g., 10 bps stress in addition to 7.5 bps). If the asymmetry is spread across all days, the mechanism is elsewhere and needs deeper diagnosis before any cash-transition variant deploys.

**Risk**: Low. This is pure decomposition.

### Experiment 5 -- OMR cross-check on detector failures

**Hypothesis**: The detector's H4 (flicker) and H5 (lag) failure modes affect RAMP through a specific consumption pattern (regime-conditional parameter switching). OMR consumes the same detector but uses regime as a filter on Bayesian probability buckets. If OMR's per-day P&L by regime shows the same flicker/lag fingerprint that V12 surfaced for RAMP, detector improvement scales across two strategies and possibly RAMP-CSP; if not, the issue is RAMP-specific.

**Why this matters**: Priority ordering. WS-3a and WS-3b are expensive (1-2 weeks each per the diagnostic's remediation ranking). If they help only RAMP, the portfolio-level payoff is smaller than if they help RAMP + OMR + RAMP-CSP. The state document does not currently consider this cross-strategy dimension when prioritizing WS-3.

**Method**: Replay OMR's per-day P&L over 2017-2026, segmented by detector regime. Compute (a) Sharpe within each regime, (b) Sharpe on regime-transition days vs persistent-regime days, (c) cross-correlation of OMR daily P&L with V12's onset-alignment panel BEAR events. If OMR underperforms on regime-transition days the same way RAMP does, detector improvement is a portfolio-level win.

**Cost**: 1-2 days. Requires OMR backtest infrastructure to produce per-day P&L attribution by regime, which may already exist.

**Decision criterion**:
- If OMR's regime-conditional Sharpe variability is comparable to RAMP's (e.g., within 30%), detector improvement scales; WS-3a and WS-3b are higher portfolio-level priority than V12c.
- If OMR is robust to the detector's failure modes, V12c (single-strategy fix) is higher leverage than WS-3a/WS-3b (single-strategy fix at higher cost) for the next research cycle.

**Risk**: OMR's per-day attribution may not currently distinguish regime-driven from non-regime-driven P&L. If so, this experiment first requires instrumenting OMR's logger, which inflates the cost. Confirm OMR's existing P&L attribution schema before committing.

### Experiment 6 -- V12c formal readiness gate

**Hypothesis**: V12-up-cash (UNPREDICTABLE='cash') is a deployable variant. From V12's sensitivity grid: Sharpe 0.586 vs V11's 0.528 at 5 bps near_close.

**Why this matters**: V12-up-cash was found in a sensitivity sweep, not in a pre-registered gate. It has not been subjected to the 5-gate readiness orchestrator. If it passes, it is the next deployment candidate (subject to Experiment 2's fragility check). If it fails, V12c is closed and WS-3 / V13 take priority.

**Method**: Run `scripts/backtest_scripts/ramp_phase4_v12_readiness.py` with `regime_positions[UNPREDICTABLE] = 'cash'` as the only configuration change vs V12 default. Use spec rev4 success criteria unchanged. Critically, the DSR n_trials must include V12c as a new trial -- V12-up-cash was *observed* in V12's sensitivity grid, so the multi-trial correction count increases from 22 to 23.

**Cost**: ~16 minutes for the orchestrator + ~half a day for synthesis and report.

**Decision criterion**: Standard 5-gate readiness. Tier classification per spec rev4 rules.

**Gating prerequisite**: Do NOT run Experiment 6 until Experiment 2 (UNPREDICTABLE hand-inspection) returns "not fragile." Running formal readiness on a fragile signal wastes the readiness gate and pollutes the multi-trial count.

**Risk**: Result honesty discipline. If Experiment 2 returns "fragile" and Experiment 6 is run anyway because the readiness gate is cheap, future research is contaminated by the trial count inflation. The pre-commit gating exists for this reason.

## Critical sequencing question

The state document positions WS-3a and WS-3b as parallel tracks within WS-3. This is premature. The right sequence depends on what Experiment 3 (soft scores) returns:

- If BEAR_score leads argmax by > 3 days, **WS-3c (soft-score consumption) becomes the top WS-3 track** ahead of WS-3a and WS-3b. The detector doesn't need a rewrite; the consumption pattern does.
- If BEAR_score is coincident with argmax but the BEAR_score itself leads forward drawdown, **WS-3a (hysteresis) is correct** -- the detector is producing the right signal at the right time but flickering between adjacent regimes is destroying its usability.
- If BEAR_score does not lead forward drawdown, **WS-3b (leading indicators) is necessary** -- the detector is fundamentally late and no consumption-layer or smoothing fix can rescue it.

These three branches require materially different specs. Starting WS-3 spec work without Experiment 3 means writing a spec that may be discarded.

## Recommended sequence

Three windows of work, ordered by what unblocks what.

**Window 1 -- WS-3 design unblockers (this session through next ~2 sessions):**

1. Experiment 3 (soft scores) -- under a day. Highest information yield. Direct input to WS-3 spec.
2. Experiment 2 (UNPREDICTABLE hand-inspection) -- half a day. Gates Experiment 6.
3. Experiment 4 (lag asymmetry) -- half a day. Gates Experiment 6. Possibly gates V13 (Experiment 1) interpretation.

These three are < 2 days total. They produce the inputs that determine which WS-3 track to spec.

**Window 2 -- conditional follow-ons (next ~3-4 sessions, parallel to Window 1 if capacity):**

4. Experiment 1 (BEAR-as-buy test) -- 1-2 days. Could be a transformative result. Should run regardless of WS-3 direction because the failure mode (it doesn't work) leaves WS-3 priority unchanged.
5. Experiment 6 (V12c readiness) -- ~1 day, gated on Experiments 2 and 4.
6. Experiment 5 (OMR cross-check) -- 1-2 days. Informs portfolio-level priority of WS-3a/WS-3b.

**Window 3 -- WS-3 spec and implementation (depends on Window 1 results):**

7. WS-3 spec write-up (the right track from Experiment 3's decision criterion).
8. WS-3 implementation.

Experiments 1, 5, and 6 can run in parallel with V11's A7 paper-validation counter (independent code paths, independent decision criteria). Experiments 2, 3, and 4 are all analyst-time work that serializes naturally.

## What this document does NOT propose

- **No production code changes** to `src/strategies/advanced/market_regime_detector.py`. All experiments use the diagnostic harness or strategy-level backtests.
- **No V12 default redeployment**. The Tier 3 verdict stands. V12c (UNPREDICTABLE='cash') is a separate variant under separate gates.
- **No early WS-3 spec work** beyond what is needed to scope Experiment 3. WS-3 spec waits on Window 1.
- **No V11 paper-validation interference**. A7 counter continues on its own clock.
- **No new data acquisition**. Every experiment uses existing 2017-2026 SPY/VIX data and existing backtest infrastructure. The FX expansion (Phase A-F) is independent and not impacted.

## Decision points

After Window 1 (~2 sessions):

- **Experiment 3 result** determines WS-3 track (WS-3a vs WS-3b vs WS-3c).
- **Experiment 2 result** determines whether V12c readiness (Experiment 6) is worth running.
- **Experiment 4 result** determines whether V12c readiness needs stricter cost-sensitivity gates.

After Window 2 (~5-6 sessions cumulative):

- **Experiment 1 result** determines whether the entire WS-3 framing is correct, or whether BEAR-as-buy reframes the problem.
- **Experiment 6 result** is a binary deployment decision for V12c.
- **Experiment 5 result** sets the portfolio-level priority of WS-3 work.

After Window 3 (~end of next ~2 weeks):

- WS-3 design spec and implementation, driven by the answers from Windows 1 and 2.

## Risk register

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| Experiment 3 reveals the detector doesn't expose soft scores to callers | Medium | Medium | Refactor scope is small (~few hours). Phase 0 of diagnostic confirmed score-based architecture, just unclear whether scores are returned. |
| Experiment 1 (BEAR-as-buy) produces a too-good-to-be-true result | Medium | High | Honesty discipline: V13 was discovered from EXT-OOS data, so DSR n_trials must include it. Treat as "not OOS in strict sense" and require later validation on unseen data. |
| Experiment 2 returns fragile, but Experiment 6 is run anyway because the orchestrator is cheap | Low | Medium | Pre-committed gating in this document. Honesty checkpoint: if Window 1 is complete and Experiment 2 returned fragile, do not run Experiment 6. |
| Multiple experiments produce noisy or contradictory results | Medium | Low | Each experiment has independent decision criteria. Contradictions are surfaced explicitly rather than averaged out. |
| The full sequence takes longer than V11's A7 5-session timer, and V11 paper validation completes before WS-3 spec exists | Medium | Low | A7 outcome is independent of WS-3. V11 ships or fails on its own merits; WS-3 continues regardless. |
| Experiment 5 (OMR cross-check) requires OMR instrumentation work that wasn't budgeted | Medium | Low | If OMR's existing P&L attribution is insufficient, descope Experiment 5 to a smaller cross-correlation analysis; do not block on full attribution. |

## Success criteria

The proposed sequence succeeds if, after Window 1 (~2 sessions):

1. WS-3 has a defensible track selection (WS-3a, WS-3b, or WS-3c) backed by Experiment 3's quantitative output.
2. V12c readiness has a clear go/no-go decision backed by Experiment 2.
3. V12c readiness has a defined cost-sensitivity gate backed by Experiment 4.

The sequence succeeds at the Window 2 horizon if:

4. The BEAR-as-buy hypothesis has a verdict (real / spurious / inconclusive).
5. V12c has either deployed or been formally closed.
6. WS-3's portfolio-level priority is informed by Experiment 5.

The sequence fails if any of the six experiments are run without producing the information their decision criteria require. Honesty discipline matters: a passed gate that the analyst doesn't believe is worth nothing.

## Appendix -- Files to create

New diagnostic and research code:

- `scripts/diagnostics/regime_score_replay.py` -- extends `regime_detector_replay.py` to log soft scores (Experiment 3)
- `notebooks/research/v12_unpredictable_event_inspection.ipynb` -- Experiment 2
- `notebooks/research/v12_lag_asymmetry_decomposition.ipynb` -- Experiment 4
- `notebooks/research/omr_regime_attribution.ipynb` -- Experiment 5
- `src/research/ramp_phase4/variants/v13_bear_invert.py` -- Experiment 1
- `scripts/backtest_scripts/ramp_phase4_v13_readiness.py` -- Experiment 1 orchestrator

New reports (in `docs/reports/ramp/`):

- `20260525_experiment3_soft_scores.md`
- `20260525_experiment2_unpredictable_inspection.md`
- `20260525_experiment4_lag_asymmetry.md`
- `20260526_experiment1_bear_invert_readiness.md`
- `20260526_experiment5_omr_cross_check.md`
- `20260526_experiment6_v12c_readiness.md` (if Experiment 2 returns not-fragile)

Modified files:

- `src/strategies/advanced/market_regime_detector.py` -- possibly extended to expose soft scores via accessor method, if not already exposed. Pure read API; no logic changes.

All other production code remains untouched until WS-3 spec is approved.
