# V12 -- Per-Regime Position Override on V11 Base

**Date**: 2026-05-23
**Status**: Approved (brainstorming -> spec)
**Owner**: Shuyang
**Type**: Research strategy variant (no production deploy in this spec; readiness orchestrator decides Phase D candidacy)
**Base**: V11 (`ramp-phase4-turnover-regime-research` at `fc7de60`)
**Related**:
- `docs/reports/ramp/20260523_phase4_v11_readiness.md` (V11 PARTIAL verdict)
- `docs/reports/ramp/20260523_regime_detector_diagnostic.md` (Phase 5 synthesis recommends BEAR-day cash logic)
- `docs/superpowers/plans/2026-05-23-ramp-research-roadmap.md` (WS-2)
- `docs/strategies/production/RAMP_STRATEGY.md` (production reference)

This spec is a working document. The defaults below were chosen during brainstorming on 2026-05-23 and are explicitly subject to revision based on readiness orchestrator findings.

---

## Context

V11 cleared the structural gates (PBO 0.126, one-day-lag delta +9.79%) but failed strict significance (PSR 0.944, DSR 0.811). Absolute Sharpe of 0.528 over 9 years is the binding constraint -- not multi-trial selection bias.

Two independent prior analyses point at the same lever:
- **May 2026 root-cause investigation** found V8 (V0 + BEAR-to-cash) beats V1 (no regime) by ~0.26 Sharpe in EXT-OOS, but failed cost sensitivity at the time.
- **2026-05-23 regime detector diagnostic** Phase 5 recommended (c) both tracks in parallel, prioritizing **RAMP BEAR-day cash logic** over detector revision.

V12 implements that lever on V11's filter base, with extended scope to UNPREDICTABLE (also defensive) and SIDEWAYS (configurable but default `normal`).

## Goals

1. Variant `V12` registered in `src/research/ramp_phase4/variants.py` such that `cfg.regime_positions` controls per-regime position behavior.
2. Default config holds cash on BEAR + UNPREDICTABLE, defers to V11 logic on STRONG_BULL + WEAK_BULL + SIDEWAYS, preserves prior positions on SAFE_MODE.
3. Readiness orchestrator re-run with V12 added to the cross-variant PBO set; emit a PSR/DSR/PBO/lag verdict report.
4. New canonical glossary doc `docs/strategies/RAMP_VARIANTS.md` documenting V01 through V12 with concrete one-liners + parameters.

## Non-goals

- Defensive ticker exposure (SH/TLT/GLD as BEAR-day position). Requires universe extension; deferred to V13.
- Per-regime strategy routing (different strategy class per regime). Requires adapter layer; deferred to V13+.
- Production paper deploy of V12. Gated on readiness verdict. If V12 clears, deploy mirrors V11's path (toggle.yaml `variant: v12`, A7 comparator extended).
- Modifying the detector itself. WS-3 (v1 detector with hysteresis) is conditional and out of scope here.

## Design

### Variant implementation

`_variant_v12(t, state, panel, cfg)` in `src/research/ramp_phase4/variants.py`:

1. Call `_variant_v11(t, state, panel, cfg)` to get the V11 plan.
2. Read `regime = plan_output['__regime__']`.
3. Look up `position_mode = cfg.regime_positions.get(regime, 'normal')`.
4. Branch:
   - `'normal'`: return V11's output unchanged.
   - `'cash'`: return `{'__regime__': regime}` (no target weights -> engine liquidates).
   - `'hold'`: return `{'__regime__': 'SAFE_MODE'}` (engine preserves positions, no trades).
   - Other (TICKER or strategy name): raise `NotImplementedError` -- reserved for V13+.

The variant must NOT mutate state; it returns a fresh dict each call.

### Config schema

Add to `src/research/ramp_phase4/config.py::HarnessConfig`:

```python
regime_positions: Dict[str, str] = field(default_factory=lambda: {
    'STRONG_BULL':   'normal',
    'WEAK_BULL':     'normal',
    'SIDEWAYS':      'normal',
    'UNPREDICTABLE': 'cash',
    'BEAR':          'cash',
    'SAFE_MODE':     'hold',
})
```

Validation: raise `ValueError` in `HarnessConfig.__post_init__` if any value is not one of `{'normal', 'cash', 'hold'}`. Allow unknown KEYS (regime names) to fall through to `'normal'` -- future-proofing in case the detector emits a new regime name.

### Engine cash-handling

Confirm before implementation: when `target_weights == {}` and regime != `'SAFE_MODE'`, does the engine liquidate all positions?

Tracing `src/research/ramp_phase4/engine.py:74-130`: target_weights empty -> `compute_trades` sees all current positions and zero targets -> generates sell trades for each held position. Yes, the engine already does the right thing for empty target_weights.

If the engine actually treats empty as "no-op" instead of "liquidate", that's a contract bug we'd need to fix as a sub-task. The implementer verifies this in the first test (`test_v12_bear_day_liquidates_all_positions`).

### Re-entry semantics

When regime flips BEAR -> non-BEAR:
- Day T (BEAR): V12 returned cash. After execution, `state.positions = {}` and `state.position_open_dates = {}`.
- Day T+1 (non-BEAR, say WEAK_BULL): V12 calls V11. V11's `rank_buffer` checks `state.positions` (empty -> no held names retained, no-op). `min_hold` checks `state.position_open_dates` (empty -> no protection, no-op). V11 returns its standard top_n picks. Engine buys them at day T+1's prices.

No special re-entry logic needed. V11's filter machinery handles it correctly because empty state defaults trigger no protections.

### Cost realism

Engine already models `cost_bps_per_side` per trade. A BEAR regime onset costs ~5 bps x N positions = ~50 bps round-trip for full liquidation (V11's typical N=10, top_n varies by regime). Re-entry costs another ~50 bps. So a single BEAR-then-recover cycle is ~100 bps of friction.

The cost-sensitivity gate at 7.5 bps tests whether this cost is acceptable. V12 must pass it as a hard requirement.

## Variants glossary deliverable

New file: `docs/strategies/RAMP_VARIANTS.md`. One-time setup: documents every named variant in the research harness with the structure below. Subsequent variants add one section.

```markdown
# RAMP Variants Reference

Canonical glossary of every named RAMP variant. Each entry links to:
- code definition in `src/research/ramp_phase4/variants.py`
- spec doc (if any) under `docs/superpowers/specs/`
- readiness report (if any) under `docs/reports/ramp/`
- production status (paper-deployed / archived / research-only)

## V01 -- baseline (fresh portfolio every rebalance)

[one-paragraph description, key params, hypotheses tested]

## V03 -- V01 + planner-correct crash exposure

[...]

## V04 -- V01 + rank_buffer

## V05 -- V01 + min_hold

## V06 -- V01 + delta_rebalance_pct threshold

## V11 -- combined turnover-lite

## V12 -- V11 + per-regime position override

## V13+ -- reserved
- V13 candidate: defensive ticker support (SH/TLT/GLD as BEAR-day position)
- V14 candidate: per-regime strategy routing (RAMP for bull, OMR for sideways, etc.)
```

For V01-V11 the descriptions are pulled from existing reports + the inline docstrings in `variants.py`. For V12 onwards the entry is written at spec/plan/implementation time.

## Open questions / room to revise

These are explicitly deferred until readiness orchestrator output is in hand:

1. **Default for SIDEWAYS**: shipped as `'normal'` (V11 logic). If readiness shows V11 SIDEWAYS days are net-negative, switch default to `'cash'`. Decided post-readiness, not now.
2. **Hysteresis around regime flips**: V12 ships WITHOUT a min-N-days-in-BEAR gate. If readiness shows V12 whipsaws on regime flips (high turnover, comparable Sharpe to V11), revisit by adding `min_regime_days: int = 0` to config and gate the flip-to-cash on it. Borderline overlap with WS-3 (detector hysteresis). YAGNI: skip for v12.0.0 and add only if data demands.
3. **Defensive ticker support**: deferred to V13. Universe expansion (`SH`, `TLT`, `GLD`) is non-trivial and deserves its own spec.
4. **Strategy routing**: deferred to V13+ once we have a per-regime adapter layer.
5. **One-day-lag interaction with cash**: V12 returns cash on BEAR detection -> next-day execution under `one_day_lag`. There's a one-day delay between BEAR signal and cash entry. Phase 0 of the diagnostic already noted SMA inputs lag onset by ~14 days median; one_day_lag adds another day. Probably immaterial but worth checking in the orchestrator output.

## Test plan

Add to `tests/research/ramp_phase4/test_variants.py`:

```python
def test_v12_normal_regime_matches_v11():
    """V12 with default config on STRONG_BULL day == V11 output exactly."""

def test_v12_bear_day_returns_empty_targets():
    """V12 on BEAR day returns {'__regime__': 'BEAR'} with no weights."""

def test_v12_unpredictable_day_returns_empty_targets():
    """V12 on UNPREDICTABLE day returns {'__regime__': 'UNPREDICTABLE'} with no weights."""

def test_v12_sideways_default_matches_v11():
    """V12 SIDEWAYS day with default config == V11 output (because default is 'normal')."""

def test_v12_safe_mode_preserves_positions():
    """V12 on SAFE_MODE returns {'__regime__': 'SAFE_MODE'} -- engine preserves positions."""

def test_v12_config_override_sideways_to_cash():
    """V12 with cfg.regime_positions['SIDEWAYS'] = 'cash' returns empty targets on SIDEWAYS."""

def test_harness_config_rejects_unknown_position_value():
    """HarnessConfig validation raises ValueError on regime_positions value other than {normal, cash, hold}."""
```

Plus an integration test verifying that running V12 through `run_variant()` on a synthetic 10-day panel where regime transitions BEAR -> WEAK_BULL produces the expected liquidate-then-rebuild trade sequence.

## Readiness orchestrator changes

New file `scripts/backtest_scripts/ramp_phase4_v12_readiness.py` mirroring V11's structure:
- `CROSS_VARIANTS = ('V01', 'V04', 'V05', 'V06', 'V11', 'V12')` -- six variants for PBO.
- Replace `'V11'` -> `'V12'` as the gate target throughout.
- Same 12-backtest structure: 6 variants at 5 bps near_close + V12 across 4 cost tiers near_close + V12 across 4 cost tiers one_day_lag.
- Total: 14 backtests (6 cross + 4 + 4 = ~14). Estimated wall-clock: ~13-14 min.

Output: `docs/reports/ramp/YYYYMMDD_phase4_v12_readiness.md`.

## Pass criteria

V12 passes the readiness gate if ALL four hold:

1. **PSR (vs SR=0) > 0.95** -- absolute significance.
2. **DSR (n_trials=20) > 0.95** -- multi-trial-corrected significance.
3. **PBO across 6 variants < 0.5** -- low overfitting evidence.
4. **One-day-lag Sharpe at 5 bps within 20% of near_close** -- no structural lookahead.

If V12 clears all 4: it becomes the Phase D candidate (skip WS-3, redeploy mirroring V11's deploy path).
If V12 clears structural only (PBO + lag): compare absolute Sharpe to V11; pick better; consider WS-3 conditional on residual.
If V12 fails structural: investigate. Possibly BEAR-to-cash relied on detector-lagged drawdown selection -- a non-trivial result.

## Risk table

| Risk | Probability | Impact | Mitigation |
|------|---|---|---|
| Engine treats empty target_weights as no-op instead of liquidate | Low | High | First test in the implementation explicitly verifies this; if false, fix engine before V12 logic. |
| V12 ties or loses to V11 because UNPREDICTABLE is too rare to matter | Medium | Low | Readiness output shows the contribution; can flip UNPREDICTABLE default back to 'normal' as a single-line config change. |
| BEAR-to-cash whipsaws (regime flips frequently around detector threshold) | Medium | Medium | Phase 4 diagnostic Analysis B showed BEAR median run length ~3-4 days; whipsaw is likely. Open question 2 above flags the hysteresis lever. If V12 readiness shows high turnover with no Sharpe lift, add `min_regime_days` config. |
| Sharpe lift doesn't materialize because V11's filters already neutralize some BEAR-day exposure | Medium | High | The May 2026 V8 finding was on V0 (no filters), not V11. V11's rank_buffer already retains held names through soft BEAR signals. If V12 only marginally improves V11, that's a finding -- not necessarily a failure. |
| Re-entry after BEAR re-enters with fresh open_dates so min_hold can't protect | Low | Low | Intentional. Fresh-entry positions are not yet aged into protection. Normal V11 behavior. |

## Success criteria

V12 succeeds if EITHER:

1. V12 clears all 4 readiness gates AND lifts net Sharpe at 5 bps by >=0.15 over V11 (>0.68 absolute), making V12 the preferred Phase D candidate, OR
2. V12 fails strict significance the same way V11 did but shows quantitatively meaningful Max DD reduction (>= 10 percentage points absolute), making V12 the preferred candidate for risk-conscious deployment, OR
3. V12 fails to clear OR reduce Max DD, in which case the diagnostic Phase 5's WS-3 recommendation (v1 detector with hysteresis) becomes active.

In all three cases the spec is "successful" because it produces a defensible Phase D / WS-3 decision. The spec only "fails" if the readiness orchestrator crashes or produces output we can't trust.

## Decision gates

1. **After variant implementation + unit tests**: 7+ unit tests pass + integration test on synthetic panel. If any fail, fix before readiness re-run.
2. **After readiness orchestrator**: review the four-gate verdict + Max DD change vs V11. Apply the success-criteria branching above.
3. **Post-decision**: either proceed to V12 production paper deploy (mirror V11 path) or activate WS-3.

## Appendix A -- File touchpoints

New files:
- `docs/superpowers/specs/2026-05-23-v12-bear-to-cash-design.md` (this doc)
- `docs/superpowers/plans/2026-05-23-v12-bear-to-cash.md` (implementation plan; from writing-plans)
- `docs/strategies/RAMP_VARIANTS.md` (canonical glossary; one-time setup populated for V01-V12)
- `scripts/backtest_scripts/ramp_phase4_v12_readiness.py` (V12 readiness orchestrator)
- `docs/reports/ramp/YYYYMMDD_phase4_v12_readiness.md` (readiness output; emitted by orchestrator)
- `docs/progress/YYYYMMDD_RAMP_V12_SESSION_LOG.md` (session log at end)

Modified files:
- `src/research/ramp_phase4/variants.py` (+ `_variant_v12`, +`'V12'` in REGISTRY; ~50 LOC)
- `src/research/ramp_phase4/config.py` (+ `regime_positions` field + validation; ~15 LOC)
- `tests/research/ramp_phase4/test_variants.py` (+ 7 unit tests; ~150 LOC)

Branch: `v12-bear-to-cash` (new, based on `ramp-phase4-turnover-regime-research`).

## Appendix B -- Why this is conservative for the first iteration

The roadmap (`docs/superpowers/plans/2026-05-23-ramp-research-roadmap.md`) WS-2.1 brainstorm topics included "defensive asset" and "strategy routing" as questions. Both are deferred to V13+. Reasons:

1. **YAGNI**: V12 with cash-only already tests the dominant hypothesis (BEAR exposure is the dominant lever). If cash works, more sophisticated alternatives are incremental tuning. If cash doesn't, more sophistication is unlikely to.
2. **Universe constraints**: SH/TLT/GLD aren't in `sp500-2025.csv`. Adding them touches the SIP daily fetcher, the variant data loader, and the production paper deploy's universe config. Real work; deserves a separate spec.
3. **Strategy routing complexity**: requires running TWO live engines simultaneously and switching between them based on regime. The current Phase 4 harness has no such abstraction. V14 territory.

V12 ships the minimum needed to test the regime-conditional-exposure idea. If the data says it works, V13 expands.
