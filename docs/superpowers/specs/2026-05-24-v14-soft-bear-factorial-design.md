# WS-3c: Soft-Score BEAR Consumer Factorial (V14a/b/c) -- Design rev2

**Date**: 2026-05-24
**Status**: Proposed (full revision; supersedes 2026-05-24-ws3c-soft-bear-consumer-design.md rev1)
**Branch**: v12-bear-to-cash (continuation of the 2026-05-24 research campaign)
**Supersedes**: V14-soft-bear single-variant spec rev1 (cash-only consumption action)
**Builds on**:
- E3 verdict (`docs/reports/ramp/20260525_experiment3_soft_scores.md`): WS-3c viable; median argmax_lag at tau=0.3 = 24 trading days; Pearson r(BEAR_score, forward 5d drawdown) = -0.198, p ~ 1e-22.
- E1 (`docs/reports/ramp/20260525_phase4_v13_readiness.md`): V13 (argmax-BEAR-as-buy) closed TIER 4 -- argmax is the wrong consumption surface.
- E6 (`docs/reports/ramp/20260526_phase4_v12c_readiness.md`): V12c (argmax-BEAR + UNPREDICTABLE-to-cash) closed TIER 4 -- argmax is the wrong consumption surface.
- V12 spec rev4 (`docs/superpowers/specs/2026-05-23-v12-bear-to-cash-design.md`): readiness gate methodology + state-machine pattern.

## Why a full revision (not just an amendment)

The rev1 spec (V14-soft-bear, cash-only) conflated two independent design choices into a single variant:

1. **Trigger surface**: argmax label (V12/V12c/V13) vs soft-score Schmitt trigger (V14).
2. **Action**: cash vs SPY vs reduced-leverage.

V13 tested BEAR-to-SPY through the argmax trigger and failed TIER 4. The standard reading of that failure is "BEAR-to-SPY is the wrong action." But the equally consistent reading is "the argmax trigger was so late that the SPY allocation sat out the recovery instead of catching the bottom." Without varying the action under a *working* trigger, the V13 failure does not distinguish trigger-quality from action-quality.

The rev1 spec defaulted to cash because that's what V12 did. But V12's cash choice was made before E3 quantified the argmax lag at 24 days. Now that we know the soft scores lead the argmax by ~24 days at tau=0.3, the bottom-marker interpretation of BEAR onset (per E1's motivation) becomes testable in a way it wasn't under V13: soft-score-triggered BEAR-to-SPY might catch the bottom that argmax-triggered BEAR-to-SPY missed.

The cost of the factorial is ~3x orchestrator runtime (~50 min vs ~17 min). The information yield is categorically larger:

| If passes | If fails | Interpretation |
|---|---|---|
| V14a only | V14b, V14c | Cash is right; the V13 failure was trigger-quality dominating |
| V14b only | V14a, V14c | BEAR-as-buy was right (E1's premise); V13 failed on trigger lag alone |
| V14c only | V14a, V14b | Risk reduction (not regime switch) is the right consumption pattern |
| Multiple | -- | Soft scores rescue the family; pick best on Sharpe/PBO trade-off |
| None | All | WS-3c hypothesis fails on this consumption layer; reframe to WS-3a or WS-3b |

The rev1 spec could only produce the first row of this table. The factorial produces all five.

The rev1 spec also had three blocking methodological issues (PBO mitigation language, tau_in selection bias, DSR n_trials undercount) and six smaller design issues. This rev2 incorporates all nine fixes alongside the factorial restructure.

## Position

The 6-experiment campaign (2026-05-24) converged on a single diagnosis: the production market regime detector is a score-based argmax classifier whose argmax label lags the underlying BEAR score by a median 24 trading days at tau=0.3. Variants that consume the argmax label all fail TIER 4 with the same root cause: the trigger is too late and too noisy. The soft scores already contain the leading information.

WS-3c proposes three parallel consumer-layer variants, all reading `detector.last_regime_scores['BEAR']` directly with Schmitt-trigger hysteresis, differing only in action:

- **V14a-soft-bear-cash**: BEAR_score crosses tau_in -> hold cash.
- **V14b-soft-bear-spy**: BEAR_score crosses tau_in -> hold SPY at V11's equivalent gross.
- **V14c-soft-bear-dampen**: BEAR_score crosses tau_in -> scale V11's positions by 0.5 (configurable).

The detector code is NOT modified (except for an optional read-only idempotency cache; see "Detector freshness"). The access pattern (`last_regime_scores`) is already exposed via `src/strategies/advanced/market_regime_detector.py:109`.

## Decision criteria

The factorial succeeds if at least one of V14a/b/c produces:

- **TIER 1**: all 5 gates PASS AND Sharpe(@5 bps near_close) > Sharpe(V11 @5 bps near_close) + 0.10. Deploy candidate (subject to forward OOS).
- **TIER 3**: structural gates (PBO, lag-degradation, cost robustness) PASS; absolute-significance gates (PSR, DSR) FAIL. File for forward OOS; do not deploy.
- **TIER 4**: any structural gate fails. That variant closed.

If multiple variants reach TIER 1, the deployment candidate is selected by joint Sharpe + PBO (lower PBO breaks Sharpe ties within +/-0.05).

If all three reach TIER 4, the WS-3c hypothesis (soft scores are the right consumption surface) fails on this evidence. Reframe to WS-3a (detector-internal hysteresis) or WS-3b (leading indicators).

### Honesty discipline (mandatory)

This block addresses the three blocking issues from rev1's review.

**1. PBO mitigation -- no goal-shifting.**
If PBO fails for any variant, that variant fails. Period. The rev1 language ("the diagnosis is that the variant family is too correlated for CSCV to distinguish") is removed. If V14a/b/c PBO comes in correlated due to family proximity, the right response is to run PBO over an orthogonal 4-variant set as a *diagnostic*, not as a gate-override.

The PBO gate uses 8 variants (defined below). The diagnostic PBO over 4 orthogonal variants is reported alongside as supplementary information; it does not change the verdict.

**2. tau_in selection bias -- pre-registered tau from independent criterion.**
The rev1 default (tau_in=0.3) was the argmax of E3's lead-time sweep on the same 2017-2026 window the gate evaluates against. This is in-sample optimization.

Rev2 fixes this with a pre-registered tau derived from an independent criterion:

  tau_in = median BEAR_score on G1_BEAR days (drawdown > 10% from 252-day trailing high)

This ties tau_in to the drawdown ground truth (G1), not to lead-time on the gate window. G1 was defined in the diagnostic before E3's sweep ran; using its median BEAR_score is independent of E3's optimization. The value is computed once at spec time and frozen in code as a constant; the spec records both the constant and the computation script.

tau_out is pre-registered as `tau_in - 0.1` (Schmitt hysteresis band of 0.1 absolute score, the minimum that preserves the trigger semantics from the rev1 design). The 0.1 band itself is a free parameter; see sensitivity panel below.

The actual numeric tau_in is determined by the pre-spec script `scripts/diagnostics/compute_tau_in_from_g1.py`, runs ONCE before the orchestrator, writes the result to `config/research/v14_tau_constants.json`, and that JSON is the source of truth that the orchestrator reads. The pre-spec script must run and produce the JSON before any V14 backtest is allowed. The JSON is committed to the branch.

**3. DSR n_trials -- audit and recount.**
Rev1 incremented n_trials from 23 (V12c boundary) to 24. The honest count includes every variant the analyst has evaluated against this data, including:

- V12 base (1)
- V12 sensitivity grid: V12-deb-2, V12-deb-3, V12-deb-5, V12-up-cash (4 trials)
- V12c (1)
- V13 (1)
- V14a (1)
- V14b (1)
- V14c (1)
- tau sensitivity panel: 5 runs at non-default tau pairs (5)
- The pre-spec G1-median computation itself does NOT count -- it is not a strategy, it is a parameter computation against a frozen ground-truth labeler

The rev1 V11 baseline and pre-V12 variants (V01, V04, V05, V06, V11) sit in a prior cohort; whether they count depends on whether the WS-3c campaign treats them as a continuation or a fresh trial cycle. Conservative choice: count them. They were 22 trials at the V11 readiness gate. Adding V12+V12-sensitivity+V12c+V13+V14a+V14b+V14c+tau-sensitivity = 14 new trials.

**DSR n_trials_project = 36.**

If this is too conservative and pushes DSR below the gate threshold for variants that would otherwise pass, the failure mode is honest -- the campaign has accumulated multi-trial penalty. Reducing n_trials artificially to rescue a verdict is the failure mode this revision exists to prevent.

PSR is reported separately and is not multi-trial-adjusted (PSR is single-strategy significance vs SR=0). Both are gated.

## Variant definitions

All three variants share the same trigger logic and state machine; they differ only in the action taken when `state.in_bear_soft_mode` is True.

### Shared inputs

- `BEAR_score`: float in [0, 1], read from `detector.last_regime_scores['BEAR']` after the variant's own explicit `classify_regime()` call (see "Detector freshness" below).
- `soft_bear_tau_in`: float in (0, 1), loaded from `config/research/v14_tau_constants.json` at config construction.
- `soft_bear_tau_out`: float satisfying `0 < tau_out < tau_in`.

### Shared state variable

- `state.in_bear_soft_mode: bool` -- True while the strategy is in BEAR-soft mode.

### Shared state transitions (Schmitt trigger)

```
if not in_bear_soft_mode and bear_score >= tau_in:
    in_bear_soft_mode = True
elif in_bear_soft_mode and bear_score < tau_out:
    in_bear_soft_mode = False
```

Note the strict `<` on the exit: `bear_score == tau_out` does NOT exit (stays in mode). Symmetric with `>=` on the entry (`bear_score == tau_in` enters).

### V14a-soft-bear-cash

- Action when in_bear_soft_mode: return `PLAN_CASH_BEAR_SOFT` sentinel (named constant, see "Plan sentinel" below).
- Otherwise: defer to V11.

### V14b-soft-bear-spy

- Action when in_bear_soft_mode: return a plan allocating 100% of V11's gross to SPY (single-symbol concentration; same gross leverage V11 would have used; broker routing inherits from V11).
- Otherwise: defer to V11.

Rationale: tests whether BEAR onset is a bottom-marker (E1's hypothesis under a working trigger). If V14b > V14a meaningfully, the consumption-layer cash assumption is wrong.

### V14c-soft-bear-dampen

- Action when in_bear_soft_mode: return V11's plan with all symbol weights multiplied by `dampen_factor` (config field, default 0.5).
- Otherwise: defer to V11 unchanged.

Rationale: tests whether the right response to BEAR is risk *reduction* (not switch). Useful if V14a (full cash) is too aggressive and V14b (full SPY) is too directional.

`dampen_factor` is pre-registered at 0.5 -- the midpoint of [0, 1]. The sensitivity panel does NOT sweep it (would expand the trial count further).

### Plan sentinel (replaces rev1's `'__regime__'` key)

Rev1 used `{'__regime__': 'BEAR_SOFT'}` as a marker dict. Rev2 replaces this with a named constant exposed via the plans module:

```python
PLAN_CASH_BEAR_SOFT = _SentinelPlan(reason='BEAR_SOFT_CASH')
```

The engine pattern-matches `isinstance(plan, _SentinelPlan)` and applies the no-exposure path. The reason field flows into per-day attribution logging. Other variants can introduce other sentinels without dunder-key collisions.

The `_SentinelPlan` class lives in `src/research/ramp_phase4/plans.py` (new file). The engine integration is a single dispatch check.

### Edge cases

- **First-day cold-start**: `state.in_bear_soft_mode` initializes to False. Detector warm-up handled separately (see "Warm-up parity").
- **BEAR_score NaN** (insufficient data, detector raised DataInsufficientError caught upstream): treat as no transition (state unchanged). V11's SAFE_MODE handling upstream covers the position side.
- **tau_out >= tau_in or tau_out <= 0 or tau_in >= 1.0**: `__post_init__` raises ValueError. The validation predicate is `0 < tau_out < tau_in < 1.0`.

### Detector freshness

Rev1 relied on the incidental ordering in `_compute_plan_from_panel` to ensure `last_regime_scores` was fresh when the variant read it. This was a fragile coupling: a future refactor could break V14 silently.

Rev2 fixes this by having each V14 variant call `_DETECTOR.classify_regime(...)` explicitly at the top of its function and assert freshness:

```python
def _variant_v14a_soft_bear_cash(t, state, panel, cfg):
    # Explicit classification call -- do not rely on _compute_plan_from_panel ordering
    _DETECTOR.classify_regime(panel.spy_slice(t), panel.vix_slice(t), t)
    assert _DETECTOR.last_classification_timestamp == t, "Detector freshness broken"

    bear_score = _DETECTOR.last_regime_scores.get('BEAR')
    ...
```

The detector's `classify_regime` is idempotent if called with the same inputs (it caches and returns the cached label/scores on a no-op recomputation). The double call is cheap and decouples V14 from upstream ordering.

If the detector's `classify_regime` is NOT idempotent in the current implementation, the spec's first implementation task is to make it so (read-only refactor: add a `(timestamp, input_hash) -> result` cache). This is the only production-code change permitted by this spec.

### Warm-up parity

The detector requires ~200 trading days of SPY/VIX history before producing valid scores. V14's first valid gated day is therefore later than V11's (V11 doesn't depend on BEAR_score).

Rev2 requires that **all variants in the readiness comparison use the same gated window**, starting at the first day all variants have valid signals. This is the V14-warm-start, not the V11-warm-start. V11's reference Sharpe is recomputed on the V14-warm-start window for apples-to-apples comparison.

This is enforced in the orchestrator: compute the V14-warm-start date once, slice all backtest output to `[V14_warm_start, end]` before computing Sharpe. The naive V11 9-year Sharpe (used in prior reports) is *not* used here; the V11 reference Sharpe in this readiness gate is the V14-window V11 Sharpe.

If the V14-warm-start window is materially shorter than the V11 9-year window (e.g., < 7 years), flag this as a robustness limitation in the synthesis report.

## Spec rev4 honesty mapping

| Rev4 element | V14a/b/c analog |
|---|---|
| `regime_positions[regime] -> mode` | Not used. Action is hard-coded per variant; the variant *is* the action choice. |
| `min_regime_days` | Not used. Schmitt trigger via tau_in/tau_out asymmetry replaces day-count debouncing. |
| Pre-variant engine update | Yes -- `_engine_pre_variant_update_soft_bear` mutates `state.in_bear_soft_mode` before the variant reads it. |
| DSR n_trials | 36 (audited count, see honesty discipline section 3). |
| PBO gate set | 8 variants: {V01, V11, V12, V12c, V13, V14a, V14b, V14c}. V04/V05/V06 (minor parameter variations of pre-V11 family) are dropped from the gate; reported in diagnostic PBO only. |
| Diagnostic PBO | 4 orthogonal variants: {V01, V11, V12, V14a}. Reported alongside; not gate-influencing. |
| Gate 4 (lag-degradation, directional) | Same: `(nc - lag) <= max(0.2 * |nc|, 0.1)`. |
| Gate 5 (cost floor + no-regress vs V11) | Same: `Sharpe(@7.5 bps lag) > 0.30 AND >= 0.9 * Sharpe(V11 @7.5 bps lag)`. V11 reference computed on V14-warm-start window. |
| Canonical pinning tests | Three -- one per variant. The state machine is shared; the action diverges. See "Pinning tests". |

## Code structure

### New / modified files

| File | Change |
|---|---|
| `config/research/v14_tau_constants.json` | New. Frozen tau_in/tau_out values produced by pre-spec script. Committed to branch. |
| `scripts/diagnostics/compute_tau_in_from_g1.py` | New. Pre-spec script: reads `diagnostics/regime/v0/labels.parquet` and `diagnostics/regime/v0_scores/labels.parquet`, computes median BEAR_score on G1_BEAR days, writes `v14_tau_constants.json`. Must run before any V14 backtest. |
| `src/research/ramp_phase4/plans.py` | New. `_SentinelPlan` class and `PLAN_CASH_BEAR_SOFT` instance. |
| `src/research/ramp_phase4/config.py` | Add `soft_bear_tau_in: float`, `soft_bear_tau_out: float`, `soft_bear_dampen_factor: float = 0.5`. Validation in `__post_init__`: tau_in/tau_out per predicate `0 < tau_out < tau_in < 1.0`; dampen_factor in [0, 1]. Load tau values from JSON at module import. |
| `src/research/ramp_phase4/engine.py` | Add `state.in_bear_soft_mode: bool = False`. Add `_engine_pre_variant_update_soft_bear(state, bear_score, tau_in, tau_out)`. Add `_SentinelPlan` dispatch in plan->execution path. |
| `src/research/ramp_phase4/variants.py` | Add `_variant_v14a_soft_bear_cash`, `_variant_v14b_soft_bear_spy`, `_variant_v14c_soft_bear_dampen`. Register in REGISTRY. |
| `src/strategies/advanced/market_regime_detector.py` | **Conditional**: only if `classify_regime` is not currently idempotent, add `(timestamp, input_hash)` cache. Read-only refactor; no logic change. |
| `tests/research/ramp_phase4/test_plans.py` | New. `_SentinelPlan` unit tests, dispatch tests. |
| `tests/research/ramp_phase4/test_variants.py` | Add ~30 tests (10 per variant) including the 3 canonical pinning tests. |
| `tests/research/ramp_phase4/test_engine.py` | Add tests for `_engine_pre_variant_update_soft_bear` state machine, including boundary cases. |
| `tests/research/ramp_phase4/test_detector_freshness.py` | New. Asserts `classify_regime` idempotency and freshness assertion behavior. |
| `scripts/backtest_scripts/ramp_phase4_v14_factorial_readiness.py` | New orchestrator. ~35 backtests (see below). |
| `docs/reports/ramp/20260526_phase4_v14_factorial_readiness.md` | Generated by orchestrator. |
| `docs/strategies/RAMP_VARIANTS.md` | Add V14a/b/c entries. |
| `docs/progress/20260524_RAMP_V14_FACTORIAL_READINESS.md` | Session log. |

### Canonical pinning tests (three)

```python
def test_v14a_hysteresis_canonical_schmitt():
    """V14a state-machine pinning test with cash action.

    Sequence with tau_in (from JSON, call it TI) and tau_out (TI - 0.1):
      Day  Score                    Expected in_bear_soft_mode
       1   TI - 0.2 (well below)    False
       2   TI - 0.05 (in band)      False  (never crossed TI)
       3   TI (exactly)             True   (>= TI enters)
       4   TI - 0.05 (in band)      True   (>= tau_out, stay)
       5   TI - 0.1 (= tau_out)     True   (NOT strict <, stay)
       6   TI - 0.1001              False  (strict <, exit)
       7   TI - 0.05 (in band)      False  (no entry)
       8   TI + 0.2 (well above)    True   (re-enter)

    When in_bear_soft_mode: variant returns PLAN_CASH_BEAR_SOFT.
    Otherwise: variant returns V11's plan unchanged.
    """

def test_v14b_hysteresis_canonical_schmitt():
    """V14b state-machine pinning test with SPY action.

    Same state-machine sequence as V14a. When in_bear_soft_mode:
    variant returns plan with 100% V11-gross allocated to SPY.
    Otherwise: returns V11's plan unchanged.

    Asserts: SPY weight in V14b plan during in_bear_soft_mode equals
    sum of |V11's symbol weights| (gross-preserving).
    """

def test_v14c_hysteresis_canonical_schmitt():
    """V14c state-machine pinning test with dampen action.

    Same state-machine sequence. When in_bear_soft_mode: variant
    returns V11's plan with all weights multiplied by dampen_factor=0.5.
    Otherwise: returns V11's plan unchanged.

    Asserts: V14c plan weights during in_bear_soft_mode equal V11
    weights * 0.5 element-wise.
    """
```

These three tests are the canonical source of truth. The boundary cases (`bear_score == tau_in`, `bear_score == tau_out`) are explicitly covered, fixing the gap in rev1's pinning test.

### Plan sentinel implementation

```python
# src/research/ramp_phase4/plans.py
from dataclasses import dataclass, field

@dataclass(frozen=True)
class _SentinelPlan:
    """A plan that signals 'no exposure' or other non-allocation actions.

    Engine pattern-matches via isinstance and dispatches to the matching
    no-trade path. The `reason` field flows into per-day attribution.
    """
    reason: str
    weights: dict = field(default_factory=dict)  # always empty for sentinels

PLAN_CASH_BEAR_SOFT = _SentinelPlan(reason='BEAR_SOFT_CASH')
```

The engine's existing no-target plan handling needs ONE addition:

```python
# src/research/ramp_phase4/engine.py (sketch)
def execute_plan(plan, ...):
    if isinstance(plan, _SentinelPlan):
        log_attribution(reason=plan.reason)
        return zero_target_orders()
    # existing dict-based plan handling continues unchanged
    ...
```

This is non-breaking for V01-V13 (no `_SentinelPlan` ever reaches the engine from those variants).

## Readiness orchestrator

`scripts/backtest_scripts/ramp_phase4_v14_factorial_readiness.py`, cloned from `ramp_phase4_v12c_readiness.py`.

### Backtest grid

For each of V14a, V14b, V14c:
- Cost grid: 4 cost levels {1, 5, 7.5, 10} bps x 2 execution modes {near_close, one_day_lag} = 8 backtests
- Total: 24 backtests across the 3 variants.

Plus references (run once, shared across variants):
- V11 reference: @5 bps near_close, @5 bps one_day_lag, @7.5 bps one_day_lag = 3 runs.
- Cross-variants for PBO: V01, V12, V12c, V13 @ 5 bps near_close = 4 runs.

Plus sensitivity panel (informational only -- runs but does NOT affect gates):
- tau-band sensitivity: at fixed tau_in (from JSON), tau_out in {tau_in - 0.05, tau_in - 0.15} x V14a only = 2 runs.
- For V14c: dampen sensitivity = 0.25, 0.75 = 2 runs.

**Total: 24 + 3 + 4 + 4 = 35 backtests. Wall-clock ~30 minutes (factorial is ~1.8x V12c orchestrator runtime).**

### Reference Sharpe convention

V11 reference for Gate 5 (cost floor + no-regress):
- Run V11 @ 7.5 bps one_day_lag on V14-warm-start window.
- This is NOT V11's 9-year Sharpe from the V11 readiness gate.
- Document the V14-warm-start date and the V11-on-this-window Sharpe value in the readiness report.

### PBO gate vs diagnostic

- **Gate PBO**: 8 variants = {V01, V11, V12, V12c, V13, V14a, V14b, V14c}. CSCV s=16 over the V14-warm-start window. This is the gate.
- **Diagnostic PBO**: 4 orthogonal variants = {V01, V11, V12, V14a}. Same CSCV settings. Reported alongside.

If gate PBO and diagnostic PBO disagree by > 0.15, flag the divergence in the synthesis report. The gate verdict stands either way.

### DSR n_trials

Hard-coded `n_trials_project = 36` per the honesty discipline section 3. Add a comment block in the orchestrator citing the count derivation:

```
V11+pre-V11 cohort:                     22
V12 base + 4-grid sensitivity:           5
V12c base:                               1
V13 base:                                1
V14a, V14b, V14c base:                   3
V14a tau-band sensitivity:               2
V14c dampen sensitivity:                 2
                                       ---
                                        36
```

### Output

`docs/reports/ramp/20260526_phase4_v14_factorial_readiness.md`. Structured with:
- Decision summary (which variants reach which Tier).
- 5-gate evaluation per variant (3 tables).
- PBO gate result + diagnostic PBO + divergence flag.
- Sensitivity panel (informational).
- Selection rationale if multiple variants reach TIER 1.
- V14-warm-start date and V11-on-window reference Sharpe.
- Forward-OOS recommendation regardless of TIER outcome (no live deployment without OOS evidence on data not used in this readiness).

## Risks and limitations

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| E3 verdict over-fits to 2017-2026; soft-score lead is artifactual | Medium | High | Forward OOS validation required regardless of Tier 1 verdict. Pre-registered tau values, not sample-optimized. |
| Schmitt trigger still flickers on noisy BEAR_score within [tau_out, tau_in] | Low | Medium | Sensitivity panel covers tau-band width. If band-sensitive, follow-up spec with min-persistence overlay. NOT in v14.0.0 scope. |
| Gate PBO inflates from V14 family correlation | Medium | Medium | Diagnostic PBO over 4 orthogonal variants reported alongside. If gate PBO fails but diagnostic PBO passes, the synthesis flags the family-correlation interpretation but the gate verdict stands -- no goal-shifting. |
| tau_in derived from G1 median is itself a single-point estimator with variance not characterized | Medium | Medium | The pre-spec script can also report 25th, 50th, 75th percentiles of BEAR_score on G1_BEAR days; the median is the registered value but the IQR informs the sensitivity panel's tau-band range. |
| DSR n_trials = 36 is too conservative and pushes all V14 verdicts to TIER 3 or worse | High | Low | This IS the honest count. If V14 cannot pass DSR at 36 trials, the campaign has consumed its multi-trial budget and no consumer-layer variant can deploy without forward OOS evidence. Reduction of n_trials to rescue verdict is prohibited. |
| Detector `classify_regime` is not idempotent and the read-only cache refactor introduces a regression | Low | High | Test `test_detector_freshness.py` covers idempotency. The refactor is gated on its passing. If the refactor proves invasive, revert to relying on `_compute_plan_from_panel` ordering with an explicit assertion at variant call site (less robust but non-breaking). |
| V14-warm-start window is materially shorter than V11's 9 years | Medium | Low | Flagged in synthesis report as robustness limitation. V14-warm-start should still be 8.4+ years (200-day warm-up against ~2360-day total panel). If < 7 years, escalate. |
| V14b's "100% of V11 gross to SPY" allocation interacts badly with V11's existing leverage caps | Medium | Medium | The variant clamps to V11's leverage cap before returning; explicit unit test for this. |
| Multiple V14 variants reach TIER 1 and the selection rule (Sharpe + PBO tiebreak) is contested | Low | Low | The rule is pre-registered in this spec. If contested, the synthesis report can recommend forward-OOS run-off between top candidates. |
| `_SentinelPlan` engine integration is invasive | Low | Medium | The dispatch is one `isinstance` check in `execute_plan`. Test coverage in `test_plans.py` and `test_engine.py`. |

## Validation gates (before merge)

1. All existing tests pass (~105 currently).
2. Pre-spec script `compute_tau_in_from_g1.py` ran; `v14_tau_constants.json` committed.
3. New tests pass: 3 canonical pinning tests + ~30 variant unit tests + plan sentinel tests + engine state machine tests + detector freshness tests. Approx +50 tests total.
4. Detector `classify_regime` idempotency confirmed (or refactored).
5. Orchestrator produces complete `20260526_phase4_v14_factorial_readiness.md` with all variants gated.
6. RAMP_VARIANTS.md entries written.
7. Session log written.

## What this spec does NOT do

- **No detector logic changes**. The only permitted modification to `src/strategies/advanced/market_regime_detector.py` is a read-only idempotency cache, conditional on the current implementation not already being idempotent. No score formula changes, no threshold changes, no input changes.
- **No V11 paper validation interference**. RAMP paper continues on `ramp-phase4-turnover-regime-research`.
- **No V12 / V12c / V13 deprecation**. They remain in REGISTRY for diagnostic continuity.
- **No new data acquisition**. Uses the same 2017-2026 panel.
- **No multi-tau parallel search as gates**. Sensitivity panels are informational.
- **No UNPREDICTABLE_score consumption** (deferred to a future WS-3c.1 spec after measuring UNPREDICTABLE_score lead-time, which E3 did not measure).
- **No min-persistence filter**. The Schmitt trigger is the chosen noise-suppression mechanism. If it fails, a follow-up spec adds persistence; not in scope.
- **No live deployment**. Even a TIER 1 verdict on V14a/b/c requires forward OOS validation before paper or live promotion.

## Open questions to resolve before implementation

1. **Is `classify_regime` currently idempotent?** Requires inspecting `src/strategies/advanced/market_regime_detector.py`. If yes, the freshness assertion + double-call pattern works as-is. If no, the read-only cache refactor is the first implementation task.

2. **What is the actual V14-warm-start date?** Computable from the panel length and the longest detector lookback. Needs to be confirmed before the orchestrator runs to size the gated window.

3. **Does V11's `position_open_dates` mechanism interact with V14's mid-stream regime entry?** When V14a enters in_bear_soft_mode mid-stream, V11's open positions are forced to cash. V11's exit-cost accounting needs to handle this cleanly. If exit costs are double-counted at the regime transition, V14 Sharpe is artificially depressed at high cost levels. Audit before orchestrator runs.

4. **The "gross-preserving" SPY allocation in V14b -- is V11's gross constant or time-varying?** If V11's gross varies day-to-day (e.g., regime-conditional sizing), V14b's SPY allocation must match V11's same-day gross, not a fixed value. Confirm V11's sizing behavior before locking V14b semantics.

5. **The G1_BEAR labeler used for pre-registered tau -- is it exactly the same as in the diagnostic, or has it been touched since?** Re-running with a modified labeler would invalidate the pre-registration. Lock the labeler version at spec time.

These five are blockers for orchestrator execution; resolve before starting implementation.

## Sequencing

Recommended order:

1. **Resolve 5 open questions** (above). ~1 day.
2. **Run pre-spec script**; commit `v14_tau_constants.json`. ~30 minutes.
3. **Implement plan sentinel** (`plans.py`, engine dispatch). ~half day.
4. **Implement state machine + 3 variants**. ~half day.
5. **Implement tests** (50 new tests). ~1 day.
6. **Implement orchestrator**. ~half day.
7. **Run orchestrator**. ~30 min wall-clock.
8. **Synthesis report**. ~half day.

Total: ~4-5 days of analyst time, plus ~30 min of compute. Parallel to V11 paper validation timer.

## Selection rule if multiple variants reach TIER 1

Pre-registered (rev1's review caught this was undefined):

1. Filter to variants with all 5 gates PASS and Sharpe > V11 + 0.10.
2. Rank by Sharpe(@5 bps near_close) descending.
3. If top two are within 0.05 Sharpe, break tie by lower gate PBO.
4. If still tied, break tie by lower n_trials-adjusted DSR (smaller multi-trial penalty).
5. Recommend forward-OOS run-off if 4 also ties.

The single TIER 1 candidate is the deployment recommendation, subject to forward OOS.

## Appendix: Why three variants, not five or ten

The factorial deliberately stops at three actions {cash, SPY, dampen-0.5}. Tempting expansions:

- BEAR-to-gold (GLD): high prior of being "right" (gold rallies in drawdowns) but adds asset class outside RAMP's universe.
- BEAR-to-cash + add long volatility (VXX): introduces a second instrument and accounting complexity.
- BEAR-to-inverse (SH): leveraged ETF complications + V11 doesn't currently hold short positions.
- BEAR-to-defensive-sector (XLP/XLV): introduces sector selection logic.
- Multiple dampen factors {0.25, 0.5, 0.75}: triples V14c's trial count without disambiguating.

These are all defensible candidates. Including them now would inflate n_trials_project further (currently 36; each addition adds 1-3) without disambiguating WS-3c's core question (does soft-score consumption work). They are deferred to WS-3c.2 if any V14a/b/c reaches TIER 1 and the campaign continues. If all V14a/b/c fail, none of these add value either.

The chosen three span the meaningful action axis: full risk-off (cash), full directional bet (SPY), partial risk reduction (dampen). Anything beyond this is either a refinement of one of these three or a different asset universe; both are out of scope for v14.0.0.

## Appendix: Differences from rev1 (single-variant cash-only spec)

| Rev1 | Rev2 |
|---|---|
| V14-soft-bear (cash only) | V14a (cash) + V14b (SPY) + V14c (dampen) |
| tau_in = 0.3 (E3 sweep argmax) | tau_in = median BEAR_score on G1_BEAR days (pre-registered, independent) |
| DSR n_trials = 24 | DSR n_trials = 36 (full audited count) |
| PBO failure -> "reinterpret as family correlation" | PBO failure -> variant closed; diagnostic PBO over orthogonal set reported alongside |
| `'__regime__': 'BEAR_SOFT'` magic dict | `_SentinelPlan` named class with engine dispatch |
| Relies on `_compute_plan_from_panel` ordering | Variant calls `classify_regime` explicitly + freshness assertion |
| Warm-up parity assumed | Warm-up parity asserted; V11 reference recomputed on V14-warm-start window |
| Pinning test missed boundary cases | Pinning tests include exact-threshold days |
| `tau_out < tau_in` only | `0 < tau_out < tau_in < 1.0` full predicate |
| Multi-variant selection rule undefined | Selection rule pre-registered: Sharpe -> PBO -> DSR -> run-off |
| ~21 backtests, ~17 min | ~35 backtests, ~30 min |
| Single decision matrix row possible | Five-row decision matrix possible |
