# V12 -- Per-Regime Position Override on V11 Base (rev3)

**Date**: 2026-05-23
**Status**: Approved (brainstorming -> spec -> rev2 -> rev3 from Claude re-review)
**Owner**: Shuyang
**Type**: Research strategy variant (no production deploy in this spec; readiness orchestrator decides Phase D candidacy)
**Base**: V11 (`ramp-phase4-turnover-regime-research` at `fc7de60`)
**Related**:
- `docs/reports/ramp/20260523_phase4_v11_readiness.md` (V11 PARTIAL verdict)
- `docs/reports/ramp/20260523_regime_detector_diagnostic.md` (Phase 5 synthesis recommends BEAR-day cash logic)
- `docs/superpowers/plans/2026-05-23-ramp-research-roadmap.md` (WS-2)
- `docs/strategies/production/RAMP_STRATEGY.md` (production reference)

## Revision history

- **rev1 (initial)**: BEAR + UNPREDICTABLE both default to cash; pass criteria PSR/DSR/PBO/lag (4 gates); no hysteresis parameter (YAGNI); detector-lag interaction flagged as Open Question.
- **rev2**: UNPREDICTABLE default flipped to `normal` with cash as A/B; added cost-sensitivity gate (5th gate); exposed `min_regime_days` parameter; promoted detector-onset alignment to first-class deliverable; reframed PSR/DSR as jointly binding; added Kalman parallel-filter constraint for V13.
- **rev3 (this doc)**: incorporates three fixes from re-review:
  1. **Hysteresis semantics committed to symmetric**. rev2 had three contradictory descriptions of how `min_regime_days` should behave (description text said symmetric, pseudo-code was entry-only asymmetric, re-entry section reverted to symmetric). rev3 commits to symmetric (the principled cost-thesis-protecting choice) and rewrites the pseudo-code using engine-managed `state.last_validated_regime` so the variant stays read-only on state. A pinning test makes the semantics non-ambiguous in code.
  2. **Hysteresis and UNPREDICTABLE A/Bs downgraded to sensitivity-only**. The 5 pass gates apply to v12.0.0 (BEAR-cash, `min_regime_days=0`, UNPREDICTABLE=normal) alone. The `min_regime_days` in {2, 3, 5} runs and `V12-up-cash` variant are still run in the orchestrator but go in an appendix; their results don't enter the gate computation. Selection-from-N inside a single readiness gate inflates effective trial count without DSR seeing it; this is the honest split. If sensitivity shows clear lift, tuning becomes V12b/V12c with its own readiness gate.
  3. **Pass gates 4 and 5 tightened with direction and lag mode**. Gate 4 is now directional: `Sharpe(one_day_lag) >= 0.8 * Sharpe(near_close)` (catches structural lookahead, allows the safe direction where one_day_lag is higher). Gate 5 explicitly evaluates at `one_day_lag` mode (realistic execution assumption), not `near_close` (research convenience).
- rev3 also: fixes off-by-one in test count (was "11+", actually 13), adds first-tick engine initialization notes for `last_regime` / `regime_streak` / `last_validated_regime`, and acknowledges in the risk table that BEAR median run length ~3-4 days plus detector lag ~14 days creates a "no good hysteresis value" tension that may push the V12 verdict toward Tier 3 (WS-3 priority).

This spec is a working document. The defaults below have been reviewed twice; they remain explicitly subject to revision based on readiness orchestrator findings.

---

## Context

V11 cleared the structural gates (PBO 0.126, one-day-lag delta +9.79%) but missed strict significance: PSR 0.944 (just below 0.95), DSR 0.811 (further below). Absolute Sharpe of 0.528 over 9 years is one binding constraint; DSR-under-multi-trial-correction is the other. A small Sharpe lift that doesn't also raise consistency-vs-trial-variance can clear PSR while failing DSR. V12 must lift *both* the point estimate and the effective edge relative to the project's accumulated trial count.

Two independent prior analyses point at the same lever:
- **May 2026 root-cause investigation** found V8 (V0 + BEAR-to-cash) beat V1 (no regime) by ~0.26 Sharpe in EXT-OOS, but V8 failed cost sensitivity at the time (Sharpe -0.714 at 7.5 bps, because non-BEAR daily-rotation costs (~0.10%/day at 5 bps, turnover ~1.0) ate the ~0.045%/day gross edge).
- **2026-05-23 regime detector diagnostic** Phase 5 recommended (c) both tracks in parallel, prioritizing **RAMP BEAR-day cash logic** over detector revision.

V12 is the obvious synthesis: V11's filter base already reduces turnover (rank_buffer + min_hold + delta_rebalance), so non-BEAR cost drag should be lower than V8's. Layering BEAR-to-cash on top tests whether the gross edge minus the (now lower) cost drag clears the cost gate V8 couldn't.

## Goals

1. Variant `V12` registered in `src/research/ramp_phase4/variants.py` such that `cfg.regime_positions` controls per-regime position behavior, with an optional `cfg.min_regime_days` hysteresis parameter (symmetric semantics; see Design).
2. Default v12.0.0 config holds cash on BEAR only, defers to V11 logic on STRONG_BULL + WEAK_BULL + SIDEWAYS + UNPREDICTABLE, preserves prior positions on SAFE_MODE, and ships with `min_regime_days=0` (hysteresis dormant).
3. Readiness orchestrator re-run with V12 added to the cross-variant PBO set; emit a PSR/DSR/PBO/lag/cost verdict report (5 gates) computed on v12.0.0 alone.
4. Detector-onset alignment analysis emitted as part of the same readiness report -- input to the V12-vs-WS-3 decision.
5. Sensitivity appendix in the readiness report: V12-up-cash and `min_regime_days` in {2, 3, 5}. Informational only; if anything shows clear lift, it becomes input for a V12b/V12c spec, not a v12.0.0 default swap.
6. New canonical glossary doc `docs/strategies/RAMP_VARIANTS.md` documenting V01 through V12.

## Non-goals

- Defensive ticker exposure (SH/TLT/GLD as BEAR-day position). Requires universe extension; deferred to V13. See Appendix C for the Kalman parallel-filter constraint that affects how V13 must be designed.
- Per-regime strategy routing (different strategy class per regime). Requires adapter layer; deferred to V13+.
- Production paper deploy of V12. Gated on readiness verdict. If V12 clears, deploy mirrors V11's path (toggle.yaml `variant: v12`, A7 comparator extended) but only after the IBKR migration paper-comparator framework can run V11 and V12 in parallel.
- Modifying the detector itself. WS-3 (v1 detector with hysteresis) is conditional and out of scope here.
- **rev3**: changing v12.0.0 defaults post-readiness based on sensitivity-appendix results. If sensitivity says hysteresis helps, that's a V12b spec, not an in-place default swap. Otherwise the readiness gate's DSR count is structurally dishonest.

## Design

### Variant implementation (rev3 -- symmetric hysteresis)

`_variant_v12(t, state, panel, cfg)` in `src/research/ramp_phase4/variants.py`:

```python
def _variant_v12(t, state, panel, cfg):
    # 1. Get V11's plan (computes regime as side effect).
    plan = _variant_v11(t, state, panel, cfg)
    regime = plan['__regime__']

    # 2. Determine the active position mode under symmetric hysteresis.
    if cfg.min_regime_days > 0:
        # The engine sets state.last_validated_regime when a regime's streak
        # has reached min_regime_days. Until then, default to 'normal' (V11
        # behavior) -- symmetric semantics: we don't switch modes until the
        # NEW regime has been observed for min_regime_days days, and we
        # stay in the current mode through transient flips back.
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

The variant reads `state.last_validated_regime` but does not mutate state. All state updates happen in the engine's per-tick post-processing (see "Engine regime-streak tracking" below).

### Symmetric semantics walk-through (rev3 -- pinning the contradiction)

Setup: `cfg.min_regime_days = 3`, `cfg.regime_positions['BEAR'] = 'cash'`, all others = 'normal'.

| Day | Regime | regime_streak | last_validated_regime | active_mode | Behavior |
|---|---|---|---|---|---|
| 0 | WEAK_BULL | {WB: 1} | None (cold) | normal | V11 plan executes |
| 1 | WEAK_BULL | {WB: 2} | None | normal | V11 plan |
| 2 | WEAK_BULL | {WB: 3} | WEAK_BULL | normal | V11 plan (WB just validated, but mode unchanged) |
| 3 | BEAR | {B: 1} | WEAK_BULL | normal | V11 plan (BEAR not yet validated) |
| 4 | BEAR | {B: 2} | WEAK_BULL | normal | V11 plan |
| 5 | BEAR | {B: 3} | BEAR | cash | LIQUIDATE -- BEAR just validated |
| 6 | BEAR | {B: 4} | BEAR | cash | Hold cash |
| 7 | WEAK_BULL | {WB: 1} | BEAR | cash | **Still cash -- WEAK_BULL not yet validated. This is the symmetric stall.** |
| 8 | BEAR | {B: 1} | BEAR | cash | Still cash -- regime streak reset but last_validated didn't change |
| 9 | BEAR | {B: 2} | BEAR | cash | Cash |
| 10 | WEAK_BULL | {WB: 1} | BEAR | cash | Still cash |
| 11 | WEAK_BULL | {WB: 2} | BEAR | cash | Still cash |
| 12 | WEAK_BULL | {WB: 3} | WEAK_BULL | normal | RE-ENTER -- WEAK_BULL just re-validated |

Day 7 is the test that pins the choice. Under asymmetric (entry-only) hysteresis, Day 7 would re-enter via V11 because the BEAR->WB flip lifts the hysteresis. Under symmetric, Day 7 stays in cash because WB hasn't been validated. **Symmetric is the V12 design.** This protects the cost thesis: short non-BEAR runs that wouldn't justify a round-trip don't trigger one.

### Config schema

Add to `src/research/ramp_phase4/config.py::HarnessConfig`:

```python
regime_positions: Dict[str, str] = field(default_factory=lambda: {
    'STRONG_BULL':   'normal',
    'WEAK_BULL':     'normal',
    'SIDEWAYS':      'normal',
    'UNPREDICTABLE': 'normal',   # rev2: was 'cash'; cash version is sensitivity-only in readiness orchestrator
    'BEAR':          'cash',
    'SAFE_MODE':     'hold',
})
min_regime_days: int = 0  # rev2: hysteresis. 0 = no hysteresis (v12.0.0 default).
                          # rev3: semantics are symmetric -- see Design.
```

Validation in `HarnessConfig.__post_init__`:
- raise `ValueError` if any value in `regime_positions` is not one of `{'normal', 'cash', 'hold'}`
- raise `ValueError` if `min_regime_days < 0`
- Allow unknown KEYS in `regime_positions` (regime names) to fall through to `'normal'` -- future-proofing

### Engine regime-streak tracking (rev3 -- adds last_validated_regime)

Engine state additions:
```python
state.last_regime: Optional[str] = None              # most recent regime classification
state.regime_streak: Dict[str, int] = {}             # consecutive day count for current regime
state.last_validated_regime: Optional[str] = None    # rev3: most recent regime whose streak >= min_regime_days
```

Per-tick **PRE-variant** processing in `engine.py` (rev3: explicit ordering choice -- engine updates state BEFORE the variant reads it):

```python
# 1. Update regime streak.
if state.last_regime == regime:
    state.regime_streak[regime] = state.regime_streak.get(regime, 0) + 1
else:
    # Regime flip: reset streak. First-tick behavior: last_regime is None,
    # which never equals any real regime name, so this branch fires correctly.
    state.regime_streak = {regime: 1}
state.last_regime = regime

# 2. Update last_validated_regime if current regime has cleared threshold.
# rev3: with min_regime_days=0 (default), the >= check passes on every tick
# (streak >= 1 >= 0 is always true), so last_validated_regime tracks the
# instantaneous regime -- bit-equivalent to no-hysteresis behavior.
if state.regime_streak[regime] >= cfg.min_regime_days:
    state.last_validated_regime = regime
```

**First-tick correctness** (rev3 explicit note): on `t=0`, `state.last_regime is None`. The equality check `None == "BEAR"` is False, so we hit the else branch: `regime_streak = {"BEAR": 1}`. Then `1 >= cfg.min_regime_days` is True iff `min_regime_days <= 1`. With default `min_regime_days=0`, this is True, and `last_validated_regime = "BEAR"` on tick 0 -- matching no-hysteresis behavior. With `min_regime_days=3`, `last_validated_regime` stays None until tick 2, and the variant falls through to `active_mode = 'normal'` for ticks 0-1.

Default `min_regime_days=0` makes all of this a no-op for V01-V11; they remain bit-equivalent.

### Engine cash-handling (unchanged from rev2)

Confirm before implementation: when `target_weights == {}` and regime != `'SAFE_MODE'`, does the engine liquidate all positions?

Tracing `src/research/ramp_phase4/engine.py:74-130`: target_weights empty -> `compute_trades` sees all current positions and zero targets -> generates sell trades for each held position. Yes, the engine already does the right thing for empty target_weights.

If the engine actually treats empty as "no-op" instead of "liquidate", that's a contract bug we'd need to fix as a sub-task. The implementer verifies this in the first test (`test_v12_bear_day_liquidates_all_positions`).

### Re-entry semantics (rev3 -- symmetric clarification)

When BEAR is validated and V12 holds cash, then regime flips to non-BEAR:

- **With `min_regime_days = 0` (v12.0.0 default)**: `last_validated_regime` updates instantly to the new regime; on the *next* tick V12 calls V11, which sees `state.positions = {}` and `state.position_open_dates = {}`. V11's `rank_buffer` and `min_hold` both no-op on empty state. V11 returns standard top_n picks. Engine buys them at the next day's prices.
- **With `min_regime_days > 0` (sensitivity-only in v12.0.0)**: per the walk-through above, `last_validated_regime` stays at BEAR until the new regime accumulates `min_regime_days` consecutive days. During the stall, V12 remains in cash. After the stall, V11 fires with empty state and the same no-op-then-rebuild path applies.

In both cases, V11's filters degrade gracefully because empty state defaults trigger no protections. No special re-entry code needed.

### Cost realism

Engine already models `cost_bps_per_side` per trade. A BEAR regime onset costs ~5 bps x N positions ~= ~50 bps round-trip for full liquidation (V11's typical N=10, top_n varies by regime). Re-entry costs another ~50 bps. A single BEAR-then-recover cycle is ~100 bps of friction.

The cost-sensitivity gate at 7.5 bps tests whether this cost is acceptable. V12 must pass it as a **hard requirement** (Gate 5, explicit in rev2/rev3).

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
## V12 -- V11 + BEAR-to-cash (symmetric hysteresis available; default off)

## V12b / V12c -- reserved (rev3)
- V12b candidate: V12 with `min_regime_days > 0` if sensitivity appendix motivates
- V12c candidate: V12 with UNPREDICTABLE='cash' if sensitivity appendix motivates

## V13+ -- reserved
- V13 candidate: defensive ticker support (SH/TLT/GLD as BEAR-day position) -- see Kalman constraint, Appendix C
- V14 candidate: per-regime strategy routing (RAMP for bull, OMR for sideways, etc.)
```

For V01-V11 the descriptions are pulled from existing reports + the inline docstrings in `variants.py`. For V12 onwards the entry is written at spec/plan/implementation time.

## Open questions / room to revise

These are deferred until readiness orchestrator output is in hand:

1. **Default for SIDEWAYS**: shipped as `'normal'` (V11 logic). If readiness shows V11 SIDEWAYS days are net-negative, that's a finding -- but a v12.0.0 default change would be a new spec (V12d), not an in-place edit. The same DSR-hygiene argument from rev3 applies.
2. **UNPREDICTABLE cash version**: runs as sensitivity-only in v12.0.0 readiness (informational appendix). If `V12-up-cash` shows clear lift on the readiness window, becomes V12c spec with own readiness gate.
3. **`min_regime_days` value**: hysteresis logic ships, but v12.0.0 default is 0. The {2, 3, 5} runs are sensitivity-only. If sensitivity shows clear lift, becomes V12b spec with own readiness gate. See risk table for the BEAR-run-length tension that makes "no good value" a plausible outcome.
4. **Defensive ticker support**: deferred to V13. Universe expansion (`SH`, `TLT`, `GLD`) is non-trivial and deserves its own spec. See Appendix C for the Kalman parallel-filter constraint.
5. **Strategy routing**: deferred to V13+ once we have a per-regime adapter layer.

## Test plan (rev3 -- 13 unit tests, was 11 in rev2)

Add to `tests/research/ramp_phase4/test_variants.py`:

```python
def test_v12_normal_regime_matches_v11():
    """V12 with default config on STRONG_BULL day == V11 output exactly."""

def test_v12_bear_day_returns_empty_targets():
    """V12 on BEAR day returns {'__regime__': 'BEAR'} with no weights (liquidate)."""

def test_v12_unpredictable_day_defaults_to_v11():
    """V12 on UNPREDICTABLE day with default config == V11 output (rev2: was cash, now normal)."""

def test_v12_unpredictable_day_returns_cash_when_configured():
    """V12 with cfg.regime_positions['UNPREDICTABLE'] = 'cash' returns empty targets."""

def test_v12_sideways_default_matches_v11():
    """V12 SIDEWAYS day with default config == V11 output."""

def test_v12_safe_mode_preserves_positions():
    """V12 on SAFE_MODE returns {'__regime__': 'SAFE_MODE'} -- engine preserves positions."""

def test_v12_bear_then_safe_mode_stays_in_cash():
    """rev2: regime BEAR -> SAFE_MODE while V12 holds nothing: positions stay empty, no re-entry."""

def test_v12_config_override_sideways_to_cash():
    """V12 with cfg.regime_positions['SIDEWAYS'] = 'cash' returns empty targets."""

def test_v12_hysteresis_day_0_starts_normal():
    """rev3: with min_regime_days=3, day 0 BEAR -> V11 plan (not cash).
    last_validated_regime is None on cold start, so active_mode defaults to 'normal'."""

def test_v12_hysteresis_validates_after_threshold():
    """rev3: BEAR for 3 consecutive days with min_regime_days=3 -> day 3 returns cash.
    Engine sets last_validated_regime='BEAR' on tick 2 (streak=3 >= 3); tick 3 sees it."""

def test_v12_hysteresis_symmetric_holds_cash_through_short_non_bear():
    """rev3 PINNING TEST: BEAR for 5 days -> cash validated. Then WEAK_BULL day 1
    with min_regime_days=3 -> STILL cash (WB streak=1 < 3, last_validated_regime
    still BEAR). This pins the symmetric semantics -- under asymmetric hysteresis
    this test would re-enter on the first WEAK_BULL day."""

def test_v12_hysteresis_revalidates_on_sustained_flip():
    """rev3: BEAR for 5 days -> cash. Then WEAK_BULL for 3 days -> day 3 returns
    to V11 (last_validated_regime flips to WEAK_BULL on tick 7)."""

def test_harness_config_rejects_unknown_position_value():
    """HarnessConfig validation raises ValueError on regime_positions value not in {normal, cash, hold}."""

def test_harness_config_rejects_negative_min_regime_days():
    """rev2: HarnessConfig validation raises ValueError on min_regime_days < 0."""
```

Plus engine-level tests in `tests/research/ramp_phase4/test_engine.py`:

```python
def test_engine_regime_streak_increments():
    """Two consecutive ticks of the same regime: streak goes 1 -> 2."""

def test_engine_regime_streak_resets_on_flip():
    """Regime BEAR -> WEAK_BULL: streak dict becomes {WEAK_BULL: 1}."""

def test_engine_last_validated_regime_with_min_zero():
    """rev3: with min_regime_days=0, last_validated_regime tracks instantaneous regime."""

def test_engine_last_validated_regime_with_min_three():
    """rev3: with min_regime_days=3, last_validated_regime stays None for ticks 0-1,
    becomes the regime on tick 2."""

def test_engine_first_tick_initialization():
    """rev3: t=0 with last_regime=None correctly enters the 'flip' branch and
    initializes regime_streak={regime: 1}."""
```

Plus an integration test verifying that running V12 through `run_variant()` on a synthetic 10-day panel where regime transitions BEAR -> WEAK_BULL (with `min_regime_days=0`) produces the expected liquidate-then-rebuild trade sequence. A second integration test verifies hysteresis: BEAR-BEAR-WEAK_BULL-BEAR-BEAR-BEAR with `min_regime_days=3` produces cash only after the third consecutive BEAR (tick 5 = third BEAR in {0,1,3,4,5}? No -- streak resets on WEAK_BULL on tick 2, so BEAR streak restarts at tick 3. Day 5 sees streak=3, validates BEAR, returns cash from tick 6 onward; this is the symmetric design).

## Readiness orchestrator changes (rev3 -- gate vs sensitivity split)

New file `scripts/backtest_scripts/ramp_phase4_v12_readiness.py` mirroring V11's structure:
- `CROSS_VARIANTS = ('V01', 'V04', 'V05', 'V06', 'V11', 'V12')` -- six variants for PBO.
- Replace `'V11'` -> `'V12'` as the gate target throughout.

**Gate-influencing runs** (13 total -- these feed the 5 pass gates):
  - **Cost grid**: V12 (v12.0.0 defaults) across 4 cost tiers (1, 5, 7.5, 10 bps) x 2 lag modes (near_close, one_day_lag) = 8 runs.
  - **Cross-variants for PBO**: V01, V04, V05, V06, V11 at 5 bps near_close = 5 runs. V12 at 5 bps near_close is already in the cost grid (no double-counting; PBO uses all six).

**Sensitivity appendix runs** (4 total -- informational, do NOT feed gates):
  - **UNPREDICTABLE A/B**: `V12-up-cash` (UNPREDICTABLE='cash', all other defaults) at 5 bps near_close = 1 run.
  - **Hysteresis sensitivity**: `V12-hyst-2`, `V12-hyst-3`, `V12-hyst-5` (min_regime_days=2/3/5) at 5 bps near_close = 3 runs.

**Total: 17 runs**. Estimated wall-clock: ~16-18 min on t4g.medium.

rev3 explicit: the sensitivity runs are appended to the experiment registry (so n_trials_project does reflect them, conservatively making DSR tighter), but they are NOT selected from to define v12.0.0's published metrics. If any sensitivity variant shows materially better behavior, that motivates a new V12b or V12c spec with its own readiness gate -- not an in-place default swap.

Output: `docs/reports/ramp/YYYYMMDD_phase4_v12_readiness.md`, with these sections:

- **Headline (5-gate verdict)**: PSR / DSR / PBO / lag-delta / cost. Computed on v12.0.0 alone.

- **Detector-onset alignment panel** (rev2 first-class deliverable). For each detected BEAR period in the test window (2017-2025):
  - SPY price trajectory from day -20 through day +30 relative to detector flip-to-BEAR.
  - V12 cash window overlay (start/end days within the trajectory).
  - V12's avoided return = sum of regime-day returns during cash window.
  - Compare: "detector-perfect" BEAR-avoidance (BEAR-onset = SPY drawdown trough +/- 1 day, hypothetical) vs. realized V12 BEAR-avoidance. Gap = lag tax.

  This panel is the input to the V12-vs-WS-3 decision. If the lag tax is large but V12 still pays off, BEAR-to-cash is a real edge that gets bigger with better detection -> deploy V12, queue WS-3 as additive improvement. If V12's apparent benefit shrinks to zero when corrected for lag, V12 is a coincidence -> WS-3 is the right priority.

- **Sensitivity appendix** (rev3 -- explicitly NOT gate-influencing):
  - UNPREDICTABLE A/B: V12 default vs V12-up-cash at 5 bps near_close. Sharpe + Max DD.
  - Hysteresis sensitivity table: Sharpe at 1/5/7.5/10 bps for `min_regime_days` in {0, 2, 3, 5}.
  - Reading guide: if any sensitivity variant beats v12.0.0 by >= 0.1 Sharpe AND retains the cost gate at 7.5 bps one_day_lag, that's input for a follow-on V12b/V12c spec. If sensitivity is within noise (SE ~0.17 over the test window), v12.0.0 defaults stand.

## Pass criteria (rev3 -- 5 gates, directional, lag-mode-explicit)

V12 passes the readiness gate if ALL five hold:

1. **PSR (vs SR=0) > 0.95** -- absolute significance. Computed on V12 v12.0.0 at 5 bps near_close.
2. **DSR > 0.95** -- multi-trial-corrected significance.
   - `n_trials` = project-wide cumulative count from `output/experiments.duckdb` (per methodology Section 9.4) at the time the orchestrator queries the registry. The sensitivity-appendix runs DO increment this count (conservatism), making this gate tighter than a strict "v12.0.0 only" count would.
   - The implementer reports DSR at both the current `n_trials` and the V11-era `n_trials` for context. If `n_trials` >= 30, also report which historical variants (V11, V8, etc.) would have passed at the V11-era count for calibration.
3. **PBO across 6 variants < 0.5** -- low overfitting evidence. Computed via CSCV on (V01, V04, V05, V06, V11, V12) at 5 bps near_close.
4. **rev3 directional: `Sharpe(V12 @ 5 bps, one_day_lag) >= 0.8 * Sharpe(V12 @ 5 bps, near_close)`** -- catches structural lookahead. The failure mode is `Sharpe_lag << Sharpe_near_close` (strategy relied on same-bar information); the opposite direction (`Sharpe_lag > Sharpe_near_close`) is safe and not penalized. Was `within 20%` in rev1/rev2 which was symmetric and ambiguous.
5. **rev3 lag-mode-explicit: `Sharpe(V12 @ 7.5 bps, one_day_lag) > 0.3`** -- cost robustness under realistic execution. rev1/rev2 didn't specify which lag mode; rev3 gates on `one_day_lag` because that's the deployment-realistic measurement. The `near_close` 7.5-bps Sharpe is reported for context but doesn't gate.

If V12 clears all 5: it becomes the Phase D candidate (skip WS-3, redeploy mirroring V11's deploy path *after* the IBKR-migration paper-comparator framework can run V11 and V12 in parallel paper).

If V12 clears structural (Gates 3, 4, 5) but misses PSR or DSR: compare absolute Sharpe to V11; pick better; consider WS-3 conditional on detector-lag-analysis output.

If V12 fails structural: investigate. Possibly BEAR-to-cash relied on detector-lagged drawdown selection -- the detector-onset alignment panel disambiguates.

## Expected magnitude of lift (rev2 sanity check, unchanged)

V8 lifted V0 by +0.501 Sharpe in EXT-OOS (2025-2026), but EXT-OOS had BEAR at 19.3% of days -- unusually BEAR-heavy. The 9-year readiness window (2017-2025) has BEAR closer to ~10-12% of days on average. Naive scaling:

```
expected_full_window_lift ~ BEAR_fraction * Sharpe_drag_avoided
                          ~ 0.10 * (0.5 to 1.0)
                          ~ 0.05 to 0.10
```

The "Sharpe lift >= 0.15 over V11" success criterion is *tight* against the realistic ceiling. **The modal outcome is probably Tier 2 (Max DD reduction with marginal Sharpe lift), not Tier 1.** This isn't a bug in V12 -- it reflects that the strongest BEAR-day evidence was on a window where BEAR was 2x more frequent than the population average.

If the readiness report shows V12 at +0.05 to +0.12 Sharpe over V11 with a 10-15 percentage point Max DD reduction, that's the *expected* outcome, not a failure.

## Risk table

| Risk | Probability | Impact | Mitigation |
|------|---|---|---|
| Engine treats empty target_weights as no-op instead of liquidate | Low | High | First test (`test_v12_bear_day_liquidates_all_positions`) verifies the contract; if false, fix engine before V12 logic. |
| V12 ties or loses to V11 because UNPREDICTABLE is too rare to matter | Low | Low | rev2: UNPREDICTABLE default is `normal` so the BEAR signal is isolated. `V12-up-cash` sensitivity run measures the alternative. |
| BEAR-to-cash whipsaws around detector threshold | Medium-High | Medium | BEAR median run length ~3-4 days + detector lag ~14 days makes whipsaw the expected mode. Hysteresis logic ships (rev2); sensitivity at {2, 3, 5} runs in readiness (rev3). |
| Sharpe lift doesn't materialize because V11's filters already neutralize some BEAR exposure | Medium | High | V8 finding was on V0 (no filters), not V11. V11's rank_buffer already retains held names through soft BEAR signals. If V12 only marginally improves V11, that's Tier 2 territory -- defensible deployment, not failure. |
| Re-entry after BEAR uses fresh open_dates so min_hold can't protect | Low | Low | Intentional. Fresh-entry positions are not yet aged into protection. Normal V11 behavior. |
| BEAR-to-cash benefit is mostly accidental (avoiding lagged-bottoming days, not the actual selloff) | Medium | High | Detector-onset alignment panel is the diagnostic. If gap between detector-perfect and realized V12 benefit is large, WS-3 takes priority over V12 deployment. |
| DSR gate gets harder as project-wide trial count grows | Medium | Medium | `n_trials` documented in readiness report. Implementer reports DSR at both current and V11-era trial counts. Sensitivity runs increment `n_trials` (conservatism). |
| **rev3: No good `min_regime_days` value exists** | Medium-High | High | With BEAR median run length ~3-4 days: `min_regime_days=3` triggers cash on day 3 just as BEAR ends (costly whipsaw, near-zero exposure-avoidance); `min_regime_days=5` essentially never goes to cash. The sensitivity range {2, 3, 5} likely shows {0, 2} as the only viable options, with marginal differences. If true, V12's verdict probably is Tier 3 (WS-3 priority) regardless of hysteresis choice -- because faster/better BEAR detection is the only way to make cash-avoidance pay off given current run lengths. The detector-onset alignment panel quantifies this. |
| **rev3: Symmetric hysteresis implementation bug (variant doesn't see `last_validated_regime` update on the same tick as the engine sets it)** | Low | Medium | Engine sets `last_validated_regime` in per-tick *post*-processing, after the variant runs. So a regime achieving threshold on tick T is visible to the variant on tick T+1, not tick T. This is consistent with the walk-through table (BEAR streak hits 3 on tick 5, variant returns cash on tick 5 *because* the engine updated `last_validated_regime` at the end of tick 4 -- wait, this needs care). Implementer must confirm the engine updates `last_validated_regime` BEFORE the variant runs on each tick. If after, increment the walk-through table indices by 1 and re-run mental check. Pinning tests catch the off-by-one. |

The rev3 final row is worth treating as a near-miss waiting to happen. The implementer should explicitly choose the engine tick ordering (pre-variant update vs post-variant update) and document it. Recommended: **pre-variant update** -- engine updates streak and validated_regime first, then variant reads. This matches the walk-through table.

## Success criteria (rev2 reframed, rev3 unchanged)

V12 succeeds if ANY of:

1. **Tier 1 (preferred)**: V12 clears all 5 readiness gates AND lifts net Sharpe at 5 bps by >= 0.15 over V11. -> Phase D candidate, deploy mirroring V11 path (after IBKR-migration paper-comparator gate).
2. **Tier 2 (modal expected outcome)**: V12 clears all 5 readiness gates but lifts Sharpe by < 0.15. Max DD must be reduced by >= 10 percentage points absolute. -> Preferred candidate for risk-conscious deployment; same deployment path as Tier 1.
3. **Tier 3 (diagnostic value)**: V12 clears structural gates (PBO, lag, cost) but misses PSR/DSR. Detector-onset alignment panel must show gap > +0.10 between detector-perfect and realized V12 lift. -> Activate WS-3 (detector improvement) as the higher-leverage path; V12 deployment deferred until WS-3 + V12 readiness re-run.
4. **Tier 4 (failure)**: V12 fails structural gates or shows no diagnostic signal in the alignment panel. -> Strategy reset; reconsider regime overlay's role from scratch.

In Tiers 1-3, the spec is "successful" because it produces a defensible decision. Tier 4 means the spec failed to advance the question -- itself useful information.

## Decision gates

1. **After variant + engine implementation + unit tests**: 13 variant tests + 5 engine tests pass + 2 integration tests on synthetic panels (basic liquidate-rebuild + hysteresis walk-through). If any fail, fix before readiness re-run.
2. **After readiness orchestrator**: review the five-gate verdict + Max DD change vs V11 + detector-onset alignment panel + sensitivity appendix. Apply the success-criteria branching above.
3. **Post-decision**: either proceed to V12 production paper deploy (mirror V11 path, after IBKR-migration comparator gate), activate WS-3, or spawn V12b/V12c if the sensitivity appendix motivates one.

## Appendix A -- File touchpoints

New files:
- `docs/superpowers/specs/2026-05-23-v12-bear-to-cash-design.md` (this doc; rev3 in place of rev2)
- `docs/superpowers/plans/2026-05-23-v12-bear-to-cash.md` (implementation plan; from writing-plans)
- `docs/strategies/RAMP_VARIANTS.md` (canonical glossary; one-time setup populated for V01-V12)
- `scripts/backtest_scripts/ramp_phase4_v12_readiness.py` (V12 readiness orchestrator with gate/sensitivity split)
- `docs/reports/ramp/YYYYMMDD_phase4_v12_readiness.md` (readiness output; includes detector-onset alignment panel + sensitivity appendix)
- `docs/progress/YYYYMMDD_RAMP_V12_SESSION_LOG.md` (session log at end)

Modified files:
- `src/research/ramp_phase4/variants.py` (+ `_variant_v12`, +`'V12'` in REGISTRY; ~50 LOC core + ~15 LOC hysteresis read = ~65 LOC)
- `src/research/ramp_phase4/config.py` (+ `regime_positions` field + `min_regime_days` field + validation; ~25 LOC)
- `src/research/ramp_phase4/engine.py` (+ regime_streak tracking + last_validated_regime in per-tick PRE-variant update; ~10 LOC; behind `min_regime_days > 0` no-op when default)
- `tests/research/ramp_phase4/test_variants.py` (+ 13 unit tests; ~250 LOC)
- `tests/research/ramp_phase4/test_engine.py` (+ 5 tests for state tracking; ~150 LOC)

Branch: `v12-bear-to-cash` (new, based on `ramp-phase4-turnover-regime-research`).

## Appendix B -- Why this is conservative for the first iteration

The roadmap (`docs/superpowers/plans/2026-05-23-ramp-research-roadmap.md`) WS-2.1 brainstorm topics included "defensive asset" and "strategy routing" as questions. Both are deferred to V13+. Reasons:

1. **YAGNI on instruments, not on parameters**: V12 with cash-only tests the dominant hypothesis (BEAR exposure is the dominant lever). Sophisticated alternatives (SH/TLT/GLD) are incremental tuning. Parameters whose absence forces a re-run of expensive backtest infrastructure (e.g. `min_regime_days`) are NOT YAGNI candidates -- they ship exposed with sensible defaults so the readiness orchestrator can probe them in one wall-clock window.
2. **rev3 honesty discipline**: parameters that ship exposed but are NOT chosen ex ante go in the sensitivity appendix, NOT the gate computation. Selection-from-N inside the gate inflates effective trial count without DSR seeing it. v12.0.0 ships with the principled defaults (BEAR-cash, no hysteresis, UNPREDICTABLE=normal); the appendix data informs whether a follow-on V12b/V12c is justified, not whether v12.0.0 itself ships.
3. **Universe constraints**: SH/TLT/GLD aren't in `sp500-2025.csv`. Adding them touches the SIP daily fetcher, the variant data loader, and the production paper deploy's universe config. Real work; deserves a separate spec.
4. **Strategy routing complexity**: requires running TWO live engines simultaneously and switching between them based on regime. The current Phase 4 harness has no such abstraction. V14 territory.

V12 ships the minimum needed to test the regime-conditional-exposure idea, plus the hysteresis logic (off by default) whose absence would cost a re-run if the data demands it.

## Appendix C -- Kalman parallel-filter constraint for V13+ defensive ticker work (rev2, unchanged)

Worth recording here because the V13 spec will need to confront it.

RAMP's `MarketRegimeDetector` requires three parallel Kalman filters (fast/medium/slow) to preserve the three-SMA structure (`above_20`, `above_50`, `above_200`) used in `_score_regime()`. Collapsing to a single trend estimate breaks regime classification.

What this means for V13:

- **V13a (simple swap)**: BEAR-day position becomes SH/TLT/GLD (defensive tickers) instead of cash, but classification is still SPY-based via the three-Kalman MarketRegimeDetector. The defensive ticker is an alternative target weight when V12 would have signaled cash. Universe expansion is real (SH/TLT/GLD added to the data pipeline + ticker config + execution path) but the detection architecture is unchanged.
- **V13b / V14 (strategy routing)**: different strategy classes per regime, with state and execution context routed via a per-regime adapter layer. The detection architecture is still SPY-based via the three-Kalman MarketRegimeDetector; the *response* to the regime classification is per-regime strategy invocation.

The constraint forces V13a to be the simpler of the two, which is why V13 is the right name for the defensive-ticker variant and V14 for routed strategies. The constraint does NOT block V13a; it just defines what V13a is.

## Appendix D -- rev3 hysteresis design decision log (for future-spec reference)

Why symmetric over asymmetric:

1. **Protects the cost thesis under whipsaw**. Asymmetric (entry-only) hysteresis lets the strategy re-enter on the first non-BEAR day, paying ~50 bps for one day of exposure if the regime flips back. Symmetric stalls re-entry until the new regime is itself validated, eliminating these short-cycle round-trips.

2. **Matches the natural reading of "hysteresis"**. In control systems, hysteresis is symmetric by definition -- the threshold for switching ON is different from the threshold for switching OFF, and both sides have a "dead zone" where state doesn't change. Asymmetric hysteresis is a different mechanism (debouncing).

3. **Cheaper to implement correctly**. Symmetric needs one piece of state (`last_validated_regime`) computed by the engine. Asymmetric needs to track entry vs exit transitions separately, which is more bookkeeping and more places for off-by-one bugs.

4. **Easier to reason about with the BEAR-run-length tension**. Symmetric hysteresis with `min_regime_days=3` and BEAR median run length 3-4 days means: BEAR runs of length < 3 never trigger cash (good -- they were noise), BEAR runs of length >= 3 trigger cash on the third day with at most ~1 day of utility (bad -- triggered too late). The tradeoff is legible. With asymmetric hysteresis, you'd also pay re-entry costs on every short non-BEAR run during a long BEAR period, making the calculus worse.

If the readiness appendix shows hysteresis hurts at all tested values (likely, per the risk-table tension), the lesson is "BEAR detection is too lagged for any reactive hysteresis to help" -- which is the WS-3 case. Symmetric hysteresis is the right baseline against which to make that claim.
