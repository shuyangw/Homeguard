# WS-3c: Soft-Score BEAR Consumer (V14-soft-bear) -- Design (REV1 -- SUPERSEDED)

> **SUPERSEDED 2026-05-24** by `docs/superpowers/specs/2026-05-24-v14-soft-bear-factorial-design.md` (rev2).
>
> Rev2 expands this single-variant cash-only design into a 3-variant factorial (V14a cash / V14b SPY / V14c dampen) under a single Schmitt-trigger surface, addresses three blocking methodological issues (PBO mitigation language, tau_in selection bias from in-sample optimization, DSR n_trials undercount), and adds infrastructure refinements (`_SentinelPlan` class, explicit detector freshness assertions, warm-up parity, full tau predicate, multi-variant selection rule). See rev2 "Appendix: Differences from rev1" for the full diff.
>
> This rev1 file is preserved for commit-history continuity and as a record of the design choices that were rejected. Implementation work follows rev2.

# WS-3c: Soft-Score BEAR Consumer (V14-soft-bear) -- Design

**Date**: 2026-05-24
**Status**: Superseded by rev2 (factorial spec)
**Branch**: v12-bear-to-cash (continuation of the 2026-05-24 research campaign)
**Builds on**:
- E3 verdict (`docs/reports/ramp/20260525_experiment3_soft_scores.md`): WS-3c, median argmax_lag at tau=0.3 = 24 trading days; Pearson r(BEAR_score, forward 5d drawdown) = -0.198, p ~ 1e-22.
- E1 (`docs/reports/ramp/20260525_phase4_v13_readiness.md`): V13 (argmax-BEAR-as-buy) closed TIER 4 -- argmax is the wrong consumption surface.
- E6 (`docs/reports/ramp/20260526_phase4_v12c_readiness.md`): V12c (argmax-BEAR + UNPREDICTABLE-to-cash) closed TIER 4 -- argmax is the wrong consumption surface.
- V12 spec rev4 (`docs/superpowers/specs/2026-05-23-v12-bear-to-cash-design.md`): readiness gate methodology + state-machine pattern.

## Position

The 6-experiment campaign (2026-05-24) converged on a single diagnosis: the production market regime detector is a score-based argmax classifier whose argmax label lags the underlying BEAR score by a median 24 trading days at tau=0.3. Variants that consume the argmax label (V12 BEAR-to-cash, V12c BEAR+UNPREDICTABLE-to-cash, V13 BEAR-to-SPY) all fail TIER 4 with the same root cause: the trigger is too late and too noisy. The soft scores already contain the leading information.

WS-3c proposes a consumer-layer fix: a new variant V14-soft-bear that reads `detector.last_regime_scores['BEAR']` directly and applies a Schmitt-trigger hysteresis on the score. The detector code itself is NOT modified; the access pattern (last_regime_scores) is already exposed via `src/strategies/advanced/market_regime_detector.py:109`.

## Decision criteria

This spec succeeds if V14-soft-bear's readiness gate produces:

- **TIER 1**: all 5 gates PASS AND Sharpe(V14 @ 5 bps near_close) > Sharpe(V11 @ 5 bps near_close) + 0.10. Deploy candidate.
- **TIER 3**: structural gates (PBO, lag-degradation, cost robustness) PASS; absolute-significance gates (PSR, DSR) FAIL. File for forward-OOS validation; do not deploy.
- **TIER 4**: any structural gate fails. V14-soft-bear closed; reframe WS-3 (the argument that "soft scores are the right surface" fails on this consumption pattern; investigate alternatives).

Honesty discipline (mandatory, not optional):
- tau_in/tau_out defaults are derived from E3's threshold sweep on the 2017-2026 window. This is the test window. The verdict is NOT strict OOS.
- DSR `n_trials_project` increments to **24** (V12c=23 + V14=24).
- PBO recomputed across 8-variant set: {V01, V04, V05, V06, V11, V12, V12c, V14-soft-bear}.
- Sensitivity appendix: tau_in / tau_out sweeps are INFORMATIONAL ONLY; the gate verdict stands on the v14.0.0 default `(tau_in=0.3, tau_out=0.2)`.

## Variant definition

**Name**: V14-soft-bear (variant id `V14-soft-bear`).

**Inputs**:
- BEAR_score: float in [0, 1], read from `detector.last_regime_scores['BEAR']` after the same `classify_regime()` call V11 already makes.
- tau_in: float, default 0.3 (entry threshold).
- tau_out: float, default 0.2 (exit threshold; MUST satisfy tau_out < tau_in for the Schmitt trigger to be well-defined).

**State variable** (new):
- `state.in_bear_soft_mode: bool` -- True while the strategy is in BEAR-soft cash mode.

**State transitions** (Schmitt trigger):
- If `not state.in_bear_soft_mode` and `bear_score >= tau_in`: set `state.in_bear_soft_mode = True`.
- If `state.in_bear_soft_mode` and `bear_score < tau_out`: set `state.in_bear_soft_mode = False`.
- Otherwise: no transition.

**Action**:
- If `state.in_bear_soft_mode`: return `{'__regime__': 'BEAR_SOFT'}` (empty position plan; equivalent to cash via the engine's existing handling of plans with no symbol weights).
- Otherwise: defer to V11 (return the V11 plan unchanged).

The `'BEAR_SOFT'` regime label is a marker for diagnostics (per-day attribution will tag these days). The engine and downstream code MUST handle it as "no exposure" via the same code path as other no-target plans.

**Edge cases**:
- First-day cold-start: `state.in_bear_soft_mode` initializes to False. The first transition can occur on day 1 if BEAR_score >= tau_in.
- BEAR_score NaN (insufficient data, detector raises DataInsufficientError): treat as `state.in_bear_soft_mode = False` (do nothing). V11's existing SAFE_MODE handling takes over upstream.
- tau_out >= tau_in (config error): `__post_init__` raises ValueError. Pre-conditions enforced at config construction time, not at runtime.

## Spec rev4 honesty mapping

| spec rev4 element | V14-soft-bear analog |
|---|---|
| `regime_positions[regime] -> mode` (V12) | NOT used. V14 has a fixed binary action: in_bear_soft_mode -> cash; else V11. No per-regime override surface. |
| `min_regime_days` (V12 symmetric debouncing) | NOT used. Hysteresis is via tau_in/tau_out asymmetry, not via day-count debouncing. The tau_in/tau_out pair plays the analogous role. |
| Pre-variant engine update | YES. The engine updates `state.in_bear_soft_mode` BEFORE the variant reads it (same ordering as V12's `_engine_pre_variant_update` helper). |
| DSR n_trials increment | YES. 23 -> 24. V12c was the prior boundary. |
| PBO 8-variant set | {V01, V04, V05, V06, V11, V12, V12c, V14-soft-bear}. |
| Gate 4 (lag-degradation, directional) | YES. Same rev4 formula: `(nc - lag) <= max(0.2 * |nc|, 0.1)`. |
| Gate 5 (cost floor + no-regress vs V11) | YES. Same rev4-followup: Sharpe(V14 @ 7.5 bps lag) > 0.30 AND >= 0.9 * Sharpe(V11 @ 7.5 bps lag). |
| Canonical pinning test | YES. A canonical unit test (`test_v14_hysteresis_canonical_schmitt`) encodes the state-machine semantics and is the source of truth for the implementation. See "Pinning test" section. |

## Code structure

### New / modified files

| File | Change |
|---|---|
| `src/research/ramp_phase4/config.py` | Add fields `soft_bear_tau_in: float = 0.3` and `soft_bear_tau_out: float = 0.2`. Validate in `__post_init__`: both in [0.0, 1.0]; tau_out < tau_in; raise ValueError otherwise. |
| `src/research/ramp_phase4/engine.py` | Add `state.in_bear_soft_mode: bool = False` to HarnessState. Add module-level helper `_engine_pre_variant_update_soft_bear(state, bear_score, tau_in, tau_out)` that mutates state. |
| `src/research/ramp_phase4/variants.py` | Add `_variant_v14_soft_bear` function. Register in REGISTRY as `'V14-soft-bear'`. |
| `tests/research/ramp_phase4/test_variants.py` | Add ~10 V14 unit tests including the canonical pinning test. |
| `tests/research/ramp_phase4/test_engine.py` | Add tests for `_engine_pre_variant_update_soft_bear` state machine transitions. |
| `scripts/backtest_scripts/ramp_phase4_v14_readiness.py` | New orchestrator, cloned from `ramp_phase4_v12c_readiness.py`. |
| `docs/reports/ramp/20260526_phase4_v14_readiness.md` | Generated by orchestrator. |
| `docs/strategies/RAMP_VARIANTS.md` | Add V14-soft-bear section between V13-bear-invert and V13+ reserved. |
| `docs/progress/20260524_RAMP_V14_READINESS.md` | Session log. |

### `_variant_v14_soft_bear` (sketch)

```python
def _variant_v14_soft_bear(t, state, panel, cfg):
    """V14-soft-bear: consume BEAR_score with Schmitt-trigger hysteresis.

    Reads detector.last_regime_scores['BEAR'] after V11's classify_regime call
    populates it. Transitions state.in_bear_soft_mode on tau_in/tau_out
    crossings. When in_bear_soft_mode is True, returns cash regardless of
    V11's plan; otherwise passes V11 plan through.

    NOT OOS in strict sense -- tau_in/tau_out defaults derived from E3 sweep
    on the 2017-2026 window.
    """
    plan = _variant_v11(t, state, panel, cfg)
    bear_score = _DETECTOR.last_regime_scores.get('BEAR') if _DETECTOR.last_regime_scores else None

    if bear_score is not None:
        _engine_pre_variant_update_soft_bear(
            state, bear_score, cfg.soft_bear_tau_in, cfg.soft_bear_tau_out
        )

    if state.in_bear_soft_mode:
        return {'__regime__': 'BEAR_SOFT'}
    return plan
```

### `_engine_pre_variant_update_soft_bear` (sketch)

```python
def _engine_pre_variant_update_soft_bear(state, bear_score, tau_in, tau_out):
    """Pre-variant state update for V14 Schmitt-trigger hysteresis."""
    if not state.in_bear_soft_mode and bear_score >= tau_in:
        state.in_bear_soft_mode = True
    elif state.in_bear_soft_mode and bear_score < tau_out:
        state.in_bear_soft_mode = False
```

### Pinning test (canonical)

```python
def test_v14_hysteresis_canonical_schmitt():
    """Source of truth for V14 state-machine semantics.

    Encodes Schmitt-trigger behavior: enter cash when BEAR_score >= tau_in,
    exit cash when BEAR_score < tau_out. State sticks in between.

    Sequence (tau_in=0.3, tau_out=0.2):
      Day  Score  Expected state.in_bear_soft_mode after update
       1   0.10   False  (never crossed tau_in)
       2   0.25   False  (in [tau_out, tau_in), no entry)
       3   0.35   True   (>= tau_in, enter)
       4   0.25   True   (in [tau_out, tau_in), stay)
       5   0.31   True   (still above tau_in, stay)
       6   0.18   False  (< tau_out, exit)
       7   0.22   False  (in [tau_out, tau_in), no entry)
       8   0.50   True   (>= tau_in, re-enter)
    """
    # Implementation: instantiate HarnessState, walk through sequence with
    # _engine_pre_variant_update_soft_bear, assert state at each step.
```

This test is the canonical reference. If the implementation contradicts this test, the implementation is wrong.

## Readiness orchestrator

Clone `scripts/backtest_scripts/ramp_phase4_v12c_readiness.py`. Adaptations:

1. **Variant**: V14-soft-bear (with tau_in=0.3, tau_out=0.2 as v14.0.0 defaults).
2. **Cost grid**: 8 backtests {1, 5, 7.5, 10} bps x {near_close, one_day_lag}.
3. **Cross-variants at 5 bps near_close**: V01, V04, V05, V06, V11, V12, V12c. 7 cross-variants + V14 = 8 in PBO.
4. **V11 reference** at 7.5 bps one_day_lag (Gate 5).
5. **DSR n_trials_project = 24** (hard-coded with comment).
6. **PBO matrix**: 2355 days x 8 variants. CSCV s=16.
7. **Gate 4** directional check inherits from V12.
8. **Gate 5** cost floor + no-regress vs V11, same.
9. **Sensitivity appendix** (informational only):
   - tau_in sweep at fixed tau_out=0.2: tau_in in {0.25, 0.30, 0.35, 0.40}. 3 extra runs.
   - tau_out sweep at fixed tau_in=0.3: tau_out in {0.10, 0.20, 0.25}. 2 extra runs (tau_out=0.20 is default, reused).
   - Total sensitivity runs: 5.
10. **Output path**: `docs/reports/ramp/20260526_phase4_v14_readiness.md`.
11. **Total backtests**: 8 cost grid + 7 cross-variants + 1 V11 ref + 5 sensitivity = 21. Wall-clock ~17 min.

## Risks and limitations

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| E3 verdict over-fits to the 2017-2026 sample; soft-score lead is artifactual | Medium | High | Forward OOS validation required regardless of TIER 1 verdict. Pre-register the v14.0.0 defaults; the sensitivity panels are informational. |
| Schmitt-trigger hysteresis still flickers on noisy BEAR_score within [tau_out, tau_in] | Low | Medium | If readiness sensitivity shows tau-pair instability, a follow-up V14.1 with min-persistence could be specced. NOT in v14.0.0 scope. |
| PBO inflates from adding V14 to the variant set | Medium | Medium | V12c's PBO already 0.71 across 7 variants. Adding V14 may push PBO to ~0.75 if V14 is correlated with V12 family. The 5-gate orchestrator surfaces this; if PBO is the binding gate again, the diagnosis is "the WS-3c hypothesis itself is correct but the variant family is too correlated for CSCV to distinguish" -- not a defect of V14 design. |
| tau_in=0.3 default is one of four E3-swept values; choice introduces selection bias | Low | Low | The sensitivity panel includes tau_in in {0.25, 0.30, 0.35, 0.40} for explicit reporting. Honest framing in summary. |
| BEAR_score is reset on every classify_regime call by the detector; access pattern via `_DETECTOR.last_regime_scores` is correct only if V11 has just called classify_regime in the same `_compute_plan_from_panel` invocation | High | Medium | Verified: `_compute_plan_from_panel` (variants.py:37) ALWAYS calls `_DETECTOR.classify_regime(...)` before returning. The score access in V14 happens after V11's plan returns, so the score is fresh. Add a unit test asserting this ordering. |
| Engine state not reset between backtest runs | Low | High | HarnessState is constructed fresh per backtest invocation by `engine.run_variant`. `state.in_bear_soft_mode` initializes to False each run. Add a unit test. |

## Validation gates (before merge)

1. All existing tests pass (~105 currently).
2. New tests: V14 unit tests (~10) + engine state-machine tests (~5) ALL pass.
3. Canonical pinning test passes -- if it doesn't, the implementation does not match this spec, halt.
4. Readiness orchestrator produces a complete `20260526_phase4_v14_readiness.md` with all 5 gates evaluated.
5. RAMP_VARIANTS.md V14 entry written.
6. Session log written.

## What this spec does NOT do

- **No production detector code changes**. `src/strategies/advanced/market_regime_detector.py` stays untouched. V14 only reads the already-exposed `last_regime_scores` accessor.
- **No V11 paper validation interference**. RAMP paper continues on `ramp-phase4-turnover-regime-research`.
- **No V12 / V12c / V13 deprecation**. Those variants remain in the REGISTRY for diagnostic continuity even though all three closed TIER 4.
- **No new data acquisition**. V14 uses the same 2017-2026 panel.
- **No multi-tau parallel search**. v14.0.0 has fixed defaults; tau sweeps are sensitivity panels, not gate-influencing.
- **No UNPREDICTABLE_score consumption** (deferred to a future WS-3c.1 spec after measuring UNPREDICTABLE_score lead-time, which E3 did not measure).
- **No min-persistence filter** (out of v14.0.0 scope; the Schmitt trigger is the chosen noise-suppression mechanism).
