# V12 -- Per-Regime Position Override on V11 Base (rev4)

**Date**: 2026-05-23
**Status**: Approved (brainstorming -> spec -> rev2 -> rev3 -> rev4 from Claude re-review)
**Owner**: Shuyang
**Type**: Research strategy variant (no production deploy in this spec; readiness orchestrator decides Phase D candidacy)
**Base**: V11 (`ramp-phase4-turnover-regime-research` at `fc7de60`)
**Related**:
- `docs/reports/ramp/20260523_phase4_v11_readiness.md` (V11 PARTIAL verdict)
- `docs/reports/ramp/20260523_regime_detector_diagnostic.md` (Phase 5 synthesis recommends BEAR-day cash logic)
- `docs/superpowers/plans/2026-05-23-ramp-research-roadmap.md` (WS-2)
- `docs/strategies/production/RAMP_STRATEGY.md` (production reference)

## Revision history

- **rev1 (initial)**: BEAR + UNPREDICTABLE both default to cash; PSR/DSR/PBO/lag (4 gates); no hysteresis; detector-lag flagged as Open Question.
- **rev2**: UNPREDICTABLE default flipped to `normal`; added cost-sensitivity gate (5th); exposed `min_regime_days`; promoted detector-onset alignment; reframed PSR/DSR jointly binding; added Kalman constraint for V13.
- **rev3**: committed to symmetric hysteresis; downgraded A/Bs to sensitivity-only; tightened gates 4 and 5 with direction and lag mode; flagged the "no good hysteresis value" tension.
- **rev4 (this doc)**: prose walk-throughs in rev2 and rev3 kept introducing contradictions in the same neighborhood (tick-ordering of engine state updates relative to variant execution). rev4's central change is structural: a canonical pinning test in real Python becomes the source of truth for hysteresis semantics, and all prose sections explicitly defer to it. Specific fixes:
  1. **Canonical pinning test is now the spec for hysteresis semantics**. Section "Canonical pinning test" below contains 13 ticks of hardcoded expected values for `(regime_streak, last_validated_regime, active_mode)`. If prose and test disagree, the test wins.
  2. **rev3 risk-row 8 contradicted the rev3 design pseudo-code on tick ordering**. Risk row said post-variant update; design said pre-variant. The walk-through table assumed pre-variant. rev4 commits to **pre-variant update** uniformly, removes the conflicting language and the "wait, this needs care" cruft, and points the implementer at the pinning test as the enforcement mechanism.
  3. **Integration test description had self-correcting cruft** ("tick 5 = third BEAR in {0,1,3,4,5}? No -- ...") and stated wrong tick numbers under pre-variant ordering. rev4 cleans this up: BEAR-BEAR-WEAK_BULL-BEAR-BEAR-BEAR with `min_regime_days=3` produces cash on tick 5 (the third consecutive BEAR after the WEAK_BULL reset at tick 2), not "tick 6 onward".
  4. **Test count corrected to 14** (rev3 said 13 in the header and 13 in Decision Gates; both undercounted by one). Decision Gates updated.
  5. **Appendix D terminology corrected**. `min_regime_days` is technically *debouncing* (require N consecutive samples before accepting a state change), not *hysteresis* (different thresholds for entering vs. exiting a state). rev3 Appendix D claimed "control-systems hysteresis is symmetric by definition" -- this is the opposite of true (a thermostat with ON at 68 F and OFF at 72 F has asymmetric thresholds, and that's literally what makes it hysteresis). rev4 fixes the terminology and keeps "hysteresis" as the colloquial name with an explicit note.
  6. **First-tick correctness note completed**. rev3 stopped at ticks 0-1 for `min_regime_days=3`; rev4 extends through tick 2 (validation moment) so the reader sees the full cold-start trajectory.
  7. **Gate 4 floored at 0.1 absolute** to avoid vacuous tightness when `Sharpe(near_close)` is near zero. Not relevant for V12's expected Sharpe range but makes the gate definition robust.

This spec is a working document. The defaults below have been reviewed three times; they remain subject to revision based on readiness orchestrator findings.

---

## Context

V11 cleared the structural gates (PBO 0.126, one-day-lag delta +9.79%) but missed strict significance: PSR 0.944 (just below 0.95), DSR 0.811 (further below). Absolute Sharpe of 0.528 over 9 years is one binding constraint; DSR-under-multi-trial-correction is the other. A small Sharpe lift that doesn't also raise consistency-vs-trial-variance can clear PSR while failing DSR. V12 must lift *both* the point estimate and the effective edge relative to the project's accumulated trial count.

Two independent prior analyses point at the same lever:
- **May 2026 root-cause investigation** found V8 (V0 + BEAR-to-cash) beat V1 (no regime) by ~0.26 Sharpe in EXT-OOS, but V8 failed cost sensitivity at the time (Sharpe -0.714 at 7.5 bps, because non-BEAR daily-rotation costs (~0.10%/day at 5 bps, turnover ~1.0) ate the ~0.045%/day gross edge).
- **2026-05-23 regime detector diagnostic** Phase 5 recommended (c) both tracks in parallel, prioritizing **RAMP BEAR-day cash logic** over detector revision.

V12 is the obvious synthesis: V11's filter base already reduces turnover (rank_buffer + min_hold + delta_rebalance), so non-BEAR cost drag should be lower than V8's. Layering BEAR-to-cash on top tests whether the gross edge minus the (now lower) cost drag clears the cost gate V8 couldn't.

## Goals

1. Variant `V12` registered in `src/research/ramp_phase4/variants.py` such that `cfg.regime_positions` controls per-regime position behavior, with optional `cfg.min_regime_days` debouncing (colloquially "hysteresis"; see Design and Appendix D).
2. Default v12.0.0 config holds cash on BEAR only, defers to V11 logic on STRONG_BULL + WEAK_BULL + SIDEWAYS + UNPREDICTABLE, preserves prior positions on SAFE_MODE, and ships with `min_regime_days=0` (debouncing dormant).
3. Readiness orchestrator re-run with V12 added to the cross-variant PBO set; emit a PSR/DSR/PBO/lag/cost verdict report (5 gates) computed on v12.0.0 alone.
4. Detector-onset alignment analysis emitted as part of the same readiness report -- input to the V12-vs-WS-3 decision.
5. Sensitivity appendix in the readiness report: V12-up-cash and `min_regime_days` in {2, 3, 5}. Informational only; if anything shows clear lift, it becomes input for a V12b/V12c spec, not a v12.0.0 default swap.
6. New canonical glossary doc `docs/strategies/RAMP_VARIANTS.md` documenting V01 through V12.

## Non-goals

- Defensive ticker exposure (SH/TLT/GLD as BEAR-day position). Requires universe extension; deferred to V13. See Appendix C for the Kalman parallel-filter constraint that affects how V13 must be designed.
- Per-regime strategy routing (different strategy class per regime). Requires adapter layer; deferred to V13+.
- Production paper deploy of V12. Gated on readiness verdict. If V12 clears, deploy mirrors V11's path (toggle.yaml `variant: v12`, A7 comparator extended) but only after the IBKR migration paper-comparator framework can run V11 and V12 in parallel.
- Modifying the detector itself. WS-3 (v1 detector with hysteresis) is conditional and out of scope here.
- Changing v12.0.0 defaults post-readiness based on sensitivity-appendix results. If sensitivity says debouncing helps, that's a V12b spec, not an in-place default swap.

## Design

### Variant implementation (rev4 -- pure read on state)

`_variant_v12(t, state, panel, cfg)` in `src/research/ramp_phase4/variants.py`:

```python
def _variant_v12(t, state, panel, cfg):
    # 1. Get V11's plan (computes regime as side effect).
    plan = _variant_v11(t, state, panel, cfg)
    regime = plan['__regime__']

    # 2. Determine the active position mode.
    if cfg.min_regime_days > 0:
        # The engine has already updated state.last_validated_regime
        # PRE-variant on this tick (see "Engine regime-streak tracking"
        # and the canonical pinning test). The variant just reads it.
        if state.last_validated_regime is None:
            active_mode = 'normal'   # cold start; no regime yet validated
        else:
            active_mode = cfg.regime_positions.get(
                state.last_validated_regime, 'normal'
            )
    else:
        active_mode = cfg.regime_positions.get(regime, 'normal')

    # 3. Branch on active mode. Variant is pure: no state mutation.
    if active_mode == 'normal':
        return plan
    elif active_mode == 'cash':
        return {'__regime__': regime}            # engine liquidates
    elif active_mode == 'hold':
        return {'__regime__': 'SAFE_MODE'}       # engine preserves positions
    else:
        raise NotImplementedError(
            f"position_mode '{active_mode}' reserved for V13+"
        )
```

The variant reads `state.last_validated_regime` and does not mutate state. All state updates happen pre-variant in the engine (see next section). The canonical pinning test below is the source of truth for what state values the variant must see on each tick.

### Canonical pinning test (rev4 -- this is the spec for debouncing semantics)

Add to `tests/research/ramp_phase4/test_variants.py`:

```python
def test_v12_hysteresis_symmetric_canonical():
    """
    CANONICAL PINNING TEST -- THE SOURCE OF TRUTH for V12's debouncing
    semantics. If the prose walk-through and this test disagree, the test
    is correct by definition. All other tests in test_variants.py and
    test_engine.py are consistent with this one.

    Semantics enforced:
      (a) Engine updates regime_streak and last_validated_regime BEFORE the
          variant reads them on each tick (pre-variant ordering).
      (b) Debouncing is symmetric: mode changes only when the new regime
          has been observed for min_regime_days consecutive ticks; the
          prior mode persists through transient flips.

    Sequence: 13 ticks (0..12) driving the state machine through cold
    start, validation, transient flip-back (tick 7), and re-validation.
    """
    cfg = HarnessConfig(
        min_regime_days=3,
        regime_positions={'BEAR': 'cash',
                          'WEAK_BULL': 'normal',
                          'STRONG_BULL': 'normal',
                          'SIDEWAYS': 'normal',
                          'UNPREDICTABLE': 'normal',
                          'SAFE_MODE': 'hold'},
    )

    # Each row is the state AFTER the engine's pre-variant update on that
    # tick, plus the active_mode the variant computes from it.
    EXPECTED = [
        # tick, regime,        regime_streak,            last_validated_regime, active_mode
        (0,    'WEAK_BULL',    {'WEAK_BULL': 1},         None,                  'normal'),
        (1,    'WEAK_BULL',    {'WEAK_BULL': 2},         None,                  'normal'),
        (2,    'WEAK_BULL',    {'WEAK_BULL': 3},         'WEAK_BULL',           'normal'),
        (3,    'BEAR',         {'BEAR': 1},              'WEAK_BULL',           'normal'),
        (4,    'BEAR',         {'BEAR': 2},              'WEAK_BULL',           'normal'),
        (5,    'BEAR',         {'BEAR': 3},              'BEAR',                'cash'),     # cash starts HERE
        (6,    'BEAR',         {'BEAR': 4},              'BEAR',                'cash'),
        (7,    'WEAK_BULL',    {'WEAK_BULL': 1},         'BEAR',                'cash'),     # PIN: symmetric stall
        (8,    'BEAR',         {'BEAR': 1},              'BEAR',                'cash'),
        (9,    'BEAR',         {'BEAR': 2},              'BEAR',                'cash'),
        (10,   'WEAK_BULL',    {'WEAK_BULL': 1},         'BEAR',                'cash'),
        (11,   'WEAK_BULL',    {'WEAK_BULL': 2},         'BEAR',                'cash'),
        (12,   'WEAK_BULL',    {'WEAK_BULL': 3},         'WEAK_BULL',           'normal'),   # re-enter
    ]

    state = make_fresh_state()  # last_regime=None, regime_streak={}, last_validated_regime=None
    for tick, regime, exp_streak, exp_lvr, exp_mode in EXPECTED:
        # 1. Engine pre-variant update.
        engine_pre_variant_update(state, regime, cfg.min_regime_days)

        # 2. Check engine state matches expectation BEFORE the variant runs.
        assert state.regime_streak == exp_streak, \
            f"tick {tick}: streak got {state.regime_streak}, expected {exp_streak}"
        assert state.last_validated_regime == exp_lvr, \
            f"tick {tick}: last_validated_regime got {state.last_validated_regime}, expected {exp_lvr}"

        # 3. Variant computes active mode from post-update state.
        plan = _variant_v12(tick, state, _stub_panel(regime), cfg)
        active_mode = _interpret_plan_as_mode(plan)
        assert active_mode == exp_mode, \
            f"tick {tick}: active_mode got {active_mode}, expected {exp_mode}"
```

The helpers (`make_fresh_state`, `engine_pre_variant_update`, `_stub_panel`, `_interpret_plan_as_mode`) are implementation-defined. The expected values are the spec.

Tick 7 is the key row: under asymmetric (entry-only) debouncing, tick 7 would re-enter via V11 because BEAR->WEAK_BULL lifts the gate. Under symmetric debouncing (V12's design), tick 7 stays in cash because WEAK_BULL hasn't been validated. This single row distinguishes the two designs.

### Walk-through table (rev4 -- commentary; the test above is authoritative)

The table below restates the canonical test in human-readable form. If you find a discrepancy, the test is correct.

Column legend: `regime_streak` is the engine's per-regime consecutive-tick counter (resets on regime flip). `last_validated_regime` (abbreviated LVR in inline notes elsewhere) is the most recent regime whose streak reached `min_regime_days`. `active_mode` is what the variant returns: `normal` calls V11, `cash` liquidates, `hold` preserves positions.

| Tick | Regime | regime_streak | last_validated_regime | active_mode | Notes |
|---|---|---|---|---|---|
| 0 | WEAK_BULL | {WB: 1} | None | normal | Cold start; no regime yet validated |
| 1 | WEAK_BULL | {WB: 2} | None | normal | |
| 2 | WEAK_BULL | {WB: 3} | WEAK_BULL | normal | WEAK_BULL validates; mode unchanged since WB maps to normal |
| 3 | BEAR | {B: 1} | WEAK_BULL | normal | Streak resets on flip; LVR unchanged |
| 4 | BEAR | {B: 2} | WEAK_BULL | normal | |
| 5 | BEAR | {B: 3} | BEAR | cash | **BEAR validates; LIQUIDATE** |
| 6 | BEAR | {B: 4} | BEAR | cash | |
| 7 | WEAK_BULL | {WB: 1} | BEAR | cash | **PIN: symmetric stall, WB not validated** |
| 8 | BEAR | {B: 1} | BEAR | cash | Streak resets to BEAR; LVR unchanged |
| 9 | BEAR | {B: 2} | BEAR | cash | |
| 10 | WEAK_BULL | {WB: 1} | BEAR | cash | |
| 11 | WEAK_BULL | {WB: 2} | BEAR | cash | |
| 12 | WEAK_BULL | {WB: 3} | WEAK_BULL | normal | **WB re-validates; RE-ENTER via V11** |

### Config schema

Add to `src/research/ramp_phase4/config.py::HarnessConfig`:

```python
regime_positions: Dict[str, str] = field(default_factory=lambda: {
    'STRONG_BULL':   'normal',
    'WEAK_BULL':     'normal',
    'SIDEWAYS':      'normal',
    'UNPREDICTABLE': 'normal',
    'BEAR':          'cash',
    'SAFE_MODE':     'hold',
})
min_regime_days: int = 0  # debouncing on regime->mode change. 0 = no debouncing (v12.0.0 default).
                          # Semantics: symmetric; see canonical pinning test.
```

Validation in `HarnessConfig.__post_init__`:
- raise `ValueError` if any value in `regime_positions` is not one of `{'normal', 'cash', 'hold'}`
- raise `ValueError` if `min_regime_days < 0`
- Allow unknown KEYS in `regime_positions` (regime names) to fall through to `'normal'` -- future-proofing

### Engine regime-streak tracking (rev4 -- pre-variant ordering, no ambiguity)

Engine state additions:
```python
state.last_regime: Optional[str] = None              # most recent regime classification
state.regime_streak: Dict[str, int] = {}             # consecutive day count for current regime
state.last_validated_regime: Optional[str] = None    # most recent regime whose streak >= min_regime_days
```

Per-tick **pre-variant** processing in `engine.py` (rev4 commits explicitly: engine updates state BEFORE the variant reads it on the same tick):

```python
# Called at the START of each tick, before _variant_v12 runs.
def engine_pre_variant_update(state, regime, min_regime_days):
    # 1. Update regime streak.
    if state.last_regime == regime:
        state.regime_streak[regime] = state.regime_streak.get(regime, 0) + 1
    else:
        # Regime flip: reset streak. First-tick: last_regime is None,
        # which never equals any real regime name, so this branch fires.
        state.regime_streak = {regime: 1}
    state.last_regime = regime

    # 2. Update last_validated_regime if current regime has cleared threshold.
    # With min_regime_days=0, streak >= 1 >= 0 always passes, so
    # last_validated_regime tracks the instantaneous regime -- making this
    # bit-equivalent to no-debouncing behavior.
    if state.regime_streak[regime] >= min_regime_days:
        state.last_validated_regime = regime
```

**First-tick correctness** (rev4 completed): on `t=0`, `state.last_regime is None`. The equality check `None == "BEAR"` is False, so we hit the else branch: `regime_streak = {"BEAR": 1}`. Then `1 >= cfg.min_regime_days` is True iff `min_regime_days <= 1`.
- With default `min_regime_days=0`: `1 >= 0` passes, `last_validated_regime = "BEAR"` on tick 0; variant returns cash on tick 0. Matches no-debouncing behavior.
- With `min_regime_days=3` on a cold BEAR start: tick 0 leaves `last_validated_regime = None` (streak=1 < 3), variant falls through to `active_mode = 'normal'`. Tick 1 same (streak=2 < 3). **Tick 2 has streak=3 = min_regime_days, so the engine sets `last_validated_regime = 'BEAR'` pre-variant; variant returns cash on tick 2.** This matches the canonical pinning test's rows 0-2 (with WEAK_BULL substituted for BEAR; the logic is identical).

Default `min_regime_days=0` makes all of this a no-op for V01-V11; they remain bit-equivalent.

### Engine cash-handling (unchanged from rev2)

Confirm before implementation: when `target_weights == {}` and regime != `'SAFE_MODE'`, does the engine liquidate all positions?

Tracing `src/research/ramp_phase4/engine.py:74-130`: target_weights empty -> `compute_trades` sees all current positions and zero targets -> generates sell trades for each held position. Yes, the engine already does the right thing for empty target_weights.

If the engine actually treats empty as "no-op" instead of "liquidate", that's a contract bug we'd need to fix as a sub-task. The implementer verifies this in `test_v12_bear_day_returns_empty_targets` plus an engine-level test.

### Re-entry semantics

When BEAR is validated and V12 holds cash, then regime flips to non-BEAR:

- **With `min_regime_days = 0` (v12.0.0 default)**: `last_validated_regime` updates instantly to the new regime; the variant on this tick calls V11 (since post-update state has the new regime as validated), which sees `state.positions = {}` and `state.position_open_dates = {}`. V11's `rank_buffer` and `min_hold` both no-op on empty state. V11 returns standard top_n picks. Engine buys them.
- **With `min_regime_days > 0` (sensitivity-only)**: per the canonical test, `last_validated_regime` stays at BEAR until the new regime accumulates `min_regime_days` consecutive ticks. During the stall, V12 remains in cash. After the stall, V11 fires with empty state and the same no-op-then-rebuild path applies.

In both cases, V11's filters degrade gracefully because empty state defaults trigger no protections. No special re-entry code needed.

### Cost realism

Engine already models `cost_bps_per_side` per trade. A BEAR regime onset costs ~5 bps x N positions = ~50 bps round-trip for full liquidation (V11's typical N=10, top_n varies by regime). Re-entry costs another ~50 bps. A single BEAR-then-recover cycle is ~100 bps of friction.

The cost-sensitivity gate at 7.5 bps tests whether this cost is acceptable. V12 must pass it as a **hard requirement** (Gate 5).

## Variants glossary deliverable

New file: `docs/strategies/RAMP_VARIANTS.md`. One-time setup; documents every named variant in the research harness. Subsequent variants add one section.

```markdown
# RAMP Variants Reference

Canonical glossary of every named RAMP variant. Each entry links to:
- code definition in `src/research/ramp_phase4/variants.py`
- spec doc (if any) under `docs/superpowers/specs/`
- readiness report (if any) under `docs/reports/ramp/`
- production status (paper-deployed / archived / research-only)

## V01 -- baseline (fresh portfolio every rebalance)
## V03 -- V01 + planner-correct crash exposure
## V04 -- V01 + rank_buffer
## V05 -- V01 + min_hold
## V06 -- V01 + delta_rebalance_pct threshold
## V11 -- combined turnover-lite (rank_buffer + min_hold + delta_rebalance)
## V12 -- V11 + BEAR-to-cash (symmetric debouncing available; default off)

## V12b / V12c -- reserved
- V12b candidate: V12 with `min_regime_days > 0` if sensitivity appendix motivates
- V12c candidate: V12 with UNPREDICTABLE='cash' if sensitivity appendix motivates

## V13+ -- reserved
- V13 candidate: defensive ticker support (SH/TLT/GLD as BEAR-day position) -- see Kalman constraint, Appendix C
- V14 candidate: per-regime strategy routing (RAMP for bull, OMR for sideways, etc.)
```

For V01-V11 the descriptions are pulled from existing reports + the inline docstrings in `variants.py`. For V12 onwards the entry is written at spec/plan/implementation time.

## Open questions / room to revise

Deferred until readiness orchestrator output is in hand:

1. **Default for SIDEWAYS**: shipped as `'normal'` (V11 logic). If readiness shows V11 SIDEWAYS days are net-negative, that's a finding -- a v12.0.0 default change would be a new spec (V12d), not in-place.
2. **UNPREDICTABLE cash version**: runs as sensitivity-only. If `V12-up-cash` shows clear lift, becomes V12c with own readiness gate.
3. **`min_regime_days` value**: debouncing logic ships, v12.0.0 default is 0. The {2, 3, 5} runs are sensitivity-only. See risk table for the BEAR-run-length tension that makes "no good value" plausible.
4. **Defensive ticker support**: deferred to V13. See Appendix C for the Kalman parallel-filter constraint.
5. **Strategy routing**: deferred to V13+.

## Test plan (rev4 -- 14 unit tests in `test_variants.py`, 5 in `test_engine.py`)

The canonical pinning test above is the centerpiece; the other tests verify specific properties and serve as targeted regression tests.

`tests/research/ramp_phase4/test_variants.py`:

```python
# 1. Basic mode behavior
def test_v12_normal_regime_matches_v11(): ...
def test_v12_bear_day_returns_empty_targets(): ...
def test_v12_unpredictable_day_defaults_to_v11(): ...           # rev2: was cash, now normal
def test_v12_unpredictable_day_returns_cash_when_configured(): ...
def test_v12_sideways_default_matches_v11(): ...
def test_v12_safe_mode_preserves_positions(): ...
def test_v12_bear_then_safe_mode_stays_in_cash(): ...
def test_v12_config_override_sideways_to_cash(): ...

# 2. Debouncing (rev3 added; rev4 makes the canonical one authoritative)
def test_v12_hysteresis_day_0_starts_normal(): ...
def test_v12_hysteresis_validates_after_threshold(): ...
def test_v12_hysteresis_revalidates_on_sustained_flip(): ...
def test_v12_hysteresis_symmetric_canonical(): ...              # rev4: THE SPEC (subsumes the rev3 "holds_cash_through_short_non_bear" pinning test)

# 3. Config validation
def test_harness_config_rejects_unknown_position_value(): ...
def test_harness_config_rejects_negative_min_regime_days(): ...
```

That's 14 tests. (rev4 drops the rev3 `test_v12_hysteresis_symmetric_holds_cash_through_short_non_bear` from the listing because the canonical pinning test asserts the same pin and more.)

`tests/research/ramp_phase4/test_engine.py`:

```python
def test_engine_regime_streak_increments(): ...
def test_engine_regime_streak_resets_on_flip(): ...
def test_engine_last_validated_regime_with_min_zero(): ...      # tracks instantaneous regime
def test_engine_last_validated_regime_with_min_three(): ...     # stays None until tick 2
def test_engine_first_tick_initialization(): ...                # last_regime=None enters flip branch
```

Plus two integration tests:

1. **Basic liquidate-rebuild**: 10-day synthetic panel with regime sequence `[STRONG_BULL]*3 + [BEAR]*3 + [WEAK_BULL]*4`, `min_regime_days=0`. Expected: V12 holds top_n names through ticks 0-2, liquidates at tick 3, holds cash through ticks 3-5, rebuilds at tick 6 from empty state.
2. **Debouncing hysteresis-rebuild**: 12-day panel with regime sequence `[BEAR, BEAR, WEAK_BULL, BEAR, BEAR, BEAR, BEAR, WEAK_BULL, WEAK_BULL, WEAK_BULL, WEAK_BULL, WEAK_BULL]`, `min_regime_days=3`. Expected: V12 stays in V11 mode through ticks 0-4 (no regime validated for 3 consecutive ticks until tick 5, when BEAR streak hits 3 -- the WEAK_BULL at tick 2 reset the BEAR streak, so the relevant BEAR streak starts at tick 3 and validates at tick 5). Cash active on ticks 5-8 inclusive. On tick 9, WEAK_BULL streak hits 3 (validates), variant returns normal, V11 rebuilds -- tick 9 is the rebuild tick, not a cash tick.

## Readiness orchestrator changes (rev3 -- gate vs sensitivity split, unchanged)

New file `scripts/backtest_scripts/ramp_phase4_v12_readiness.py` mirroring V11's structure:
- `CROSS_VARIANTS = ('V01', 'V04', 'V05', 'V06', 'V11', 'V12')` -- six variants for PBO.
- Replace `'V11'` -> `'V12'` as the gate target throughout.

**Gate-influencing runs** (13 total -- feed the 5 pass gates):
  - **Cost grid**: V12 (v12.0.0 defaults) across 4 cost tiers (1, 5, 7.5, 10 bps) x 2 lag modes (near_close, one_day_lag) = 8 runs.
  - **Cross-variants for PBO**: V01, V04, V05, V06, V11 at 5 bps near_close = 5 runs.

**Sensitivity appendix runs** (4 total -- informational, do NOT feed gates):
  - **UNPREDICTABLE A/B**: `V12-up-cash` at 5 bps near_close = 1 run.
  - **Debouncing sensitivity**: `V12-deb-2`, `V12-deb-3`, `V12-deb-5` (min_regime_days=2/3/5) at 5 bps near_close = 3 runs.

**Total: 17 runs**. Estimated wall-clock: ~16-18 min on t4g.medium.

The sensitivity runs are appended to the experiment registry (so n_trials_project does reflect them, conservatively tightening DSR), but are NOT selected from to define v12.0.0's published metrics. If any sensitivity variant shows materially better behavior, that motivates a new V12b or V12c spec with its own readiness gate -- not an in-place default swap.

Output: `docs/reports/ramp/YYYYMMDD_phase4_v12_readiness.md`, with:

- **Headline (5-gate verdict)**: PSR / DSR / PBO / lag-delta / cost. Computed on v12.0.0 alone.

- **Detector-onset alignment panel** (rev2 first-class deliverable). For each detected BEAR period in the test window (2017-2025):
  - SPY price trajectory from day -20 through day +30 relative to detector flip-to-BEAR.
  - V12 cash window overlay (start/end days within the trajectory).
  - V12's avoided return = sum of regime-day returns during cash window.
  - Compare: "detector-perfect" BEAR-avoidance (BEAR-onset = SPY drawdown trough +/- 1 day, hypothetical) vs. realized V12 BEAR-avoidance. Gap = lag tax.

  This panel is the input to the V12-vs-WS-3 decision. If the lag tax is large but V12 still pays off, BEAR-to-cash is a real edge that gets bigger with better detection -> deploy V12, queue WS-3 as additive improvement. If V12's apparent benefit shrinks to zero when corrected for lag, V12 is a coincidence -> WS-3 is the right priority.

- **Sensitivity appendix** (explicitly NOT gate-influencing):
  - UNPREDICTABLE A/B: V12 default vs V12-up-cash at 5 bps near_close. Sharpe + Max DD.
  - Debouncing sensitivity table: Sharpe at 1/5/7.5/10 bps for `min_regime_days` in {0, 2, 3, 5}.
  - Reading guide: if any sensitivity variant beats v12.0.0 by >= 0.1 Sharpe AND retains the cost gate at 7.5 bps one_day_lag, that's input for a follow-on V12b/V12c spec. If sensitivity is within noise (SE ~0.17), v12.0.0 defaults stand.

## Pass criteria (rev4 -- Gate 4 floored, rest unchanged from rev3)

V12 passes if ALL five hold:

1. **PSR (vs SR=0) > 0.95** -- absolute significance. V12 v12.0.0 at 5 bps near_close.
2. **DSR > 0.95** -- multi-trial-corrected significance.
   - `n_trials` = project-wide cumulative count from `output/experiments.duckdb` (methodology Section 9.4) at orchestrator-run time. Sensitivity-appendix runs DO increment this count.
   - Implementer reports DSR at both current and V11-era `n_trials` for context. If `n_trials >= 30`, report which historical variants would have passed at V11-era count for calibration.
3. **PBO across 6 variants < 0.5** -- CSCV on (V01, V04, V05, V06, V11, V12) at 5 bps near_close.
4. **Lag-degradation gate (rev4 floored): `Sharpe(near_close) - Sharpe(one_day_lag) <= max(0.2 * Sharpe(near_close), 0.1)`** at 5 bps cost. Catches the structural lookahead failure mode (lag underperforms near_close). The 0.1 floor avoids vacuous tightness when near_close Sharpe is small. The opposite direction (lag > near_close) is safe and not penalized.
5. **Cost gate (rev4-followup, Issue 4 fix): both clauses must hold**:
   - `Sharpe(V12 @ 7.5 bps, one_day_lag) > 0.3` -- absolute floor, catches V8-like collapse under realistic costs.
   - `Sharpe(V12 @ 7.5 bps, one_day_lag) >= 0.9 * Sharpe(V11 @ 7.5 bps, one_day_lag)` -- no-regress vs V11, prevents an awkward "passed gate, failed success criteria" state where V12 clears 0.3 but underperforms V11.
   - V11 reference (from `docs/reports/ramp/20260523_phase4_v11_readiness.md`): Sharpe(V11 @ 7.5 bps, one_day_lag) = 0.531. So V12 must be >= ~0.478 to clear the no-regress clause. The 0.3 absolute floor is binding only if V11 itself collapsed below ~0.33 at this measurement (it didn't).

If V12 clears all 5: Phase D candidate; deploy mirrors V11 path after IBKR-migration paper-comparator framework supports V11+V12 in parallel.

If V12 clears structural (3, 4, 5) but misses PSR or DSR: compare absolute Sharpe to V11; pick better; consider WS-3 conditional on detector-lag-analysis output.

If V12 fails structural: investigate via the detector-onset alignment panel.

## Expected magnitude of lift (rev2 sanity check, unchanged)

V8 lifted V0 by +0.501 Sharpe in EXT-OOS (2025-2026), but EXT-OOS had BEAR at 19.3% of days -- unusually BEAR-heavy. The 9-year readiness window (2017-2025) has BEAR closer to ~10-12% of days on average. Naive scaling:

```
expected_full_window_lift ~ BEAR_fraction * Sharpe_drag_avoided
                          ~ 0.10 * (0.5 to 1.0)
                          ~ 0.05 to 0.10
```

The "Sharpe lift >= 0.15 over V11" criterion is *tight* against the realistic ceiling. **Modal outcome is probably Tier 2 (Max DD reduction with marginal Sharpe lift), not Tier 1.** This reflects EXT-OOS being 2x more BEAR-heavy than the population average.

If the readiness report shows V12 at +0.05 to +0.12 Sharpe over V11 with a 10-15 percentage point Max DD reduction, that's the *expected* outcome, not a failure.

## Risk table (rev4 -- row 8 cleaned up, others unchanged)

| Risk | Probability | Impact | Mitigation |
|------|---|---|---|
| Engine treats empty target_weights as no-op instead of liquidate | Low | High | First test (`test_v12_bear_day_returns_empty_targets`) verifies the contract; if false, fix engine before V12 logic. |
| V12 ties or loses to V11 because UNPREDICTABLE is too rare to matter | Low | Low | UNPREDICTABLE default is `normal` so the BEAR signal is isolated. `V12-up-cash` sensitivity run measures the alternative. |
| BEAR-to-cash whipsaws around detector threshold | Medium-High | Medium | BEAR median run length ~3-4 days + detector lag ~14 days makes whipsaw the expected mode. Debouncing logic ships; sensitivity at {2, 3, 5}. |
| Sharpe lift doesn't materialize because V11's filters already neutralize some BEAR exposure | Medium | High | V8 finding was on V0 (no filters), not V11. V11's rank_buffer already retains held names through soft BEAR signals. Marginal improvement is Tier 2 territory -- defensible deployment, not failure. |
| Re-entry after BEAR uses fresh open_dates so min_hold can't protect | Low | Low | Intentional. Fresh-entry positions are not yet aged. Normal V11 behavior. |
| BEAR-to-cash benefit is mostly accidental (avoiding lagged-bottoming days, not the actual selloff) | Medium | High | Detector-onset alignment panel is the diagnostic. If lag tax is large, WS-3 takes priority over V12 deployment. |
| DSR gate gets harder as project-wide trial count grows | Medium | Medium | `n_trials` documented in readiness report; DSR reported at both current and V11-era counts. Sensitivity runs increment `n_trials`. |
| No good `min_regime_days` value exists | Medium-High | High | With BEAR median run length ~3-4 days: `min_regime_days=3` triggers cash on day 3 just as BEAR ends (costly whipsaw, near-zero exposure-avoidance); `min_regime_days=5` essentially never goes to cash. Sensitivity range {2, 3, 5} likely shows {0, 2} as the only viable options with marginal differences. If true, V12's verdict probably is Tier 3 (WS-3 priority) regardless of debouncing choice -- because faster/better BEAR detection is the only way to make cash-avoidance pay off given current run lengths. Detector-onset alignment panel quantifies this. |
| Pre-variant vs post-variant tick-ordering implementation bug | Low | Medium | rev4 commits to **pre-variant update** uniformly. The canonical pinning test enforces the expected `(streak, last_validated_regime, active_mode)` triples per tick. An implementation with post-variant ordering would fail the canonical test on at least tick 5 (cash starts there, not tick 6). The test is the implementation guard. |

## Success criteria (unchanged from rev2/rev3)

V12 succeeds if ANY of:

1. **Tier 1 (preferred)**: V12 clears all 5 readiness gates AND lifts net Sharpe at 5 bps by >= 0.15 over V11. -> Phase D candidate, deploy mirroring V11 path.
2. **Tier 2 (modal expected outcome)**: V12 clears all 5 readiness gates but lifts Sharpe by < 0.15. Max DD must be reduced by >= 10 percentage points absolute relative to V11. **V11 baseline (from `docs/reports/ramp/20260523_phase4_v11_readiness.md`): Max DD at 5 bps near_close = -66.20%.** So Tier 2 requires V12 Max DD better than (less negative than) -56.20% at the same measurement. -> Preferred candidate for risk-conscious deployment; same deployment path.
3. **Tier 3 (diagnostic value)**: V12 clears structural gates (PBO, lag, cost) but misses PSR/DSR. Detector-onset alignment panel must show gap > +0.10 between detector-perfect and realized V12 lift. -> Activate WS-3 as the higher-leverage path; V12 deployment deferred until WS-3 + V12 readiness re-run.
4. **Tier 4 (failure)**: V12 fails structural gates or shows no diagnostic signal. -> Strategy reset; reconsider regime overlay's role from scratch.

In Tiers 1-3 the spec is "successful" because it produces a defensible decision. Tier 4 means the spec failed to advance the question -- itself useful information.

Practical note: Tiers 1/2 and Tier 3 are not exclusive in the deployment sense. A V12 verdict in Tier 2 with a Tier-3-quality lag tax can defensibly become "deploy V12 AND queue WS-3 immediately" -- the readiness verdict informs the decision; it doesn't constrain it to a single track.

## Decision gates (rev4 -- test count corrected)

1. **After variant + engine implementation + unit tests**: 14 variant tests + 5 engine tests pass + 2 integration tests on synthetic panels (liquidate-rebuild + debouncing-rebuild). Canonical pinning test passes. If any fail, fix before readiness re-run.
2. **After readiness orchestrator**: review the five-gate verdict + Max DD change vs V11 + detector-onset alignment panel + sensitivity appendix. Apply the success-criteria branching above.
3. **Post-decision**: either proceed to V12 production paper deploy, activate WS-3, spawn V12b/V12c if sensitivity motivates one, or some combination.

## Appendix A -- File touchpoints

New files:
- `docs/superpowers/specs/2026-05-23-v12-bear-to-cash-design.md` (this doc; rev4 in place of rev3)
- `docs/superpowers/plans/2026-05-23-v12-bear-to-cash.md` (implementation plan; from writing-plans)
- `docs/strategies/RAMP_VARIANTS.md` (canonical glossary; one-time setup populated for V01-V12)
- `scripts/backtest_scripts/ramp_phase4_v12_readiness.py` (V12 readiness orchestrator with gate/sensitivity split)
- `docs/reports/ramp/YYYYMMDD_phase4_v12_readiness.md` (readiness output; includes detector-onset alignment panel + sensitivity appendix)
- `docs/progress/YYYYMMDD_RAMP_V12_SESSION_LOG.md` (session log at end)

Modified files:
- `src/research/ramp_phase4/variants.py` (+ `_variant_v12`, +`'V12'` in REGISTRY; ~65 LOC)
- `src/research/ramp_phase4/config.py` (+ `regime_positions` + `min_regime_days` + validation; ~25 LOC)
- `src/research/ramp_phase4/engine.py` (+ regime_streak + last_validated_regime in pre-variant update; ~10 LOC)
- `tests/research/ramp_phase4/test_variants.py` (+ 14 unit tests including the canonical pinning test; ~280 LOC)
- `tests/research/ramp_phase4/test_engine.py` (+ 5 tests; ~150 LOC)

Branch: `v12-bear-to-cash` (new, based on `ramp-phase4-turnover-regime-research`).

## Appendix B -- Why this is conservative for the first iteration

The roadmap (`docs/superpowers/plans/2026-05-23-ramp-research-roadmap.md`) WS-2.1 brainstorm topics included "defensive asset" and "strategy routing" as questions. Both are deferred to V13+. Reasons:

1. **YAGNI on instruments, not on parameters**: V12 with cash-only tests the dominant hypothesis (BEAR exposure is the dominant lever). Sophisticated alternatives (SH/TLT/GLD) are incremental tuning. Parameters whose absence forces a re-run of expensive backtest infrastructure (e.g. `min_regime_days`) are NOT YAGNI candidates -- they ship exposed with sensible defaults so the readiness orchestrator can probe them in one wall-clock window.
2. **Honesty discipline**: parameters that ship exposed but are NOT chosen ex ante go in the sensitivity appendix, NOT the gate computation. Selection-from-N inside the gate inflates effective trial count without DSR seeing it. v12.0.0 ships with the principled defaults (BEAR-cash, no debouncing, UNPREDICTABLE=normal); the appendix data informs whether a follow-on V12b/V12c is justified, not whether v12.0.0 itself ships.
3. **Universe constraints**: SH/TLT/GLD aren't in `sp500-2025.csv`. Adding them touches the SIP daily fetcher, the variant data loader, and the production paper deploy's universe config. Real work; deserves a separate spec.
4. **Strategy routing complexity**: requires running TWO live engines simultaneously and switching between them based on regime. The current Phase 4 harness has no such abstraction. V14 territory.

V12 ships the minimum needed to test the regime-conditional-exposure idea, plus the debouncing logic (off by default) whose absence would cost a re-run if the data demands it.

## Appendix C -- Kalman parallel-filter constraint for V13+ defensive ticker work (rev2, unchanged)

Worth recording here because the V13 spec will need to confront it.

RAMP's `MarketRegimeDetector` requires three parallel Kalman filters (fast/medium/slow) to preserve the three-SMA structure (`above_20`, `above_50`, `above_200`) used in `_score_regime()`. Collapsing to a single trend estimate breaks regime classification.

What this means for V13:

- **V13a (simple swap)**: BEAR-day position becomes SH/TLT/GLD (defensive tickers) instead of cash, but classification is still SPY-based via the three-Kalman MarketRegimeDetector. The defensive ticker is an alternative target weight when V12 would have signaled cash. Universe expansion is real (SH/TLT/GLD added to the data pipeline + ticker config + execution path) but the detection architecture is unchanged.
- **V13b / V14 (strategy routing)**: different strategy classes per regime, with state and execution context routed via a per-regime adapter layer. The detection architecture is still SPY-based via the three-Kalman MarketRegimeDetector; the *response* to the regime classification is per-regime strategy invocation.

The constraint forces V13a to be the simpler of the two, which is why V13 is the right name for the defensive-ticker variant and V14 for routed strategies. The constraint does NOT block V13a; it just defines what V13a is.

## Appendix D -- Debouncing design decision log (rev4 -- terminology corrected)

### Terminology note

`min_regime_days` is technically *debouncing* (require N consecutive samples before accepting a state change), not *hysteresis*. Control-systems hysteresis is asymmetric: a thermostat with set-point 70 F that turns heat ON at 68 F and OFF at 72 F has different thresholds for the two directions, and that's literally what makes it hysteresis (the asymmetric threshold creates a "dead zone" where state doesn't change). Debouncing is what oscilloscopes do to noisy buttons: require N stable readings before accepting a transition.

Phase 4's classifier emits discrete regime labels, not a continuous score, so we can't apply asymmetric thresholds at the score level. Debouncing-on-labels is the closest equivalent of "ignore short-lived classifications". rev1-rev3 called this "hysteresis" colloquially because that's how the team referred to it; rev4 keeps "hysteresis" as the informal name in conversation but corrects the spec to use "debouncing" where precision matters (function and parameter names stay as-is to avoid a code rename).

### Why symmetric debouncing over asymmetric (entry-only)

1. **Protects the cost thesis under whipsaw**. Asymmetric (entry-only) debouncing lets the strategy re-enter on the first non-BEAR day, paying ~50 bps for one day of exposure if the regime flips back. Symmetric stalls re-entry until the new regime is itself validated, eliminating these short-cycle round-trips.

2. **Matches the cost-protection intuition**. The point of `min_regime_days` is "ignore regime classifications that don't persist." This is a symmetric statement about classifications, not an asymmetric one about entries. Asymmetric would be "ignore exit signals but trust entry signals" -- which is debouncing-only-in-one-direction and has a different name.

3. **Cheaper to implement correctly**. Symmetric needs one piece of state (`last_validated_regime`) computed by the engine. Asymmetric needs to track entry vs exit transitions separately, which is more bookkeeping and more places for off-by-one bugs.

4. **Easier to reason about given the BEAR-run-length tension**. With BEAR median run length 3-4 days, symmetric debouncing at `min_regime_days=3` means: BEAR runs of length < 3 never trigger cash (good -- they were noise), BEAR runs of length >= 3 trigger cash on the third day with at most ~1 day of utility (bad -- triggered too late). Tradeoff is legible. Asymmetric would also pay re-entry costs on every short non-BEAR run during a long BEAR period, making the calculus worse.

If the readiness appendix shows debouncing hurts at all tested values (likely per the risk-table tension), the lesson is "BEAR detection is too lagged for any reactive filter to help" -- the WS-3 case. Symmetric debouncing is the right baseline against which to make that claim.

### Why pre-variant ordering over post-variant

Pre-variant: engine updates `(regime_streak, last_validated_regime)` BEFORE the variant runs on each tick. The variant sees the updated state.

Post-variant: engine updates AFTER the variant runs. The variant sees state as of the previous tick's update.

Pre-variant is one tick faster to react: a regime that just hit the threshold takes effect on the same tick it hit it. Post-variant adds one tick of delay. Given that the underlying detector already lags by ~14 days and BEAR runs are ~3-4 days long, every tick of avoidable delay matters. Pre-variant is the right choice.

The canonical pinning test enforces this: under post-variant ordering, tick 5 in the canonical test would show `active_mode='normal'` (variant sees pre-update state where streak=2, last_validated_regime=WEAK_BULL), and cash would start on tick 6. The test expects cash on tick 5; this is the pre-variant commit.
