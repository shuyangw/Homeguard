# V12 -- Per-Regime Position Override on V11 Base (rev2)

**Date**: 2026-05-23
**Status**: Approved (brainstorming -> spec -> rev2 from Claude review)
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
- **rev2 (this doc)**: incorporates four edits from review:
  1. UNPREDICTABLE default flipped to `normal`; cash version becomes explicit A/B in the readiness matrix.
  2. Pass criteria gains a 5th gate: net Sharpe at 7.5 bps (1.5x cost) > 0.3. This is V12's whole reason for existing relative to V8 and was previously implicit.
  3. `min_regime_days` exposed as a config field with default 0 (no behavior change), tested A/B at {0, 2, 3, 5} in the readiness orchestrator. Costs ~10 LOC vs. a future readiness re-run.
  4. Detector-onset alignment analysis promoted from Open Question 5 to a first-class readiness deliverable. This is the input to the V12-vs-WS-3 decision.
- rev2 also fixes the "binding constraint is absolute Sharpe, not multi-trial bias" framing (both PSR and DSR are binding), adds a SAFE_MODE-after-BEAR test, documents the `n_trials` source for DSR, and notes the Kalman parallel-filter constraint in V13 deferral.

This spec is a working document. The defaults below were chosen during brainstorming on 2026-05-23 and reviewed once; they are still explicitly subject to revision based on readiness orchestrator findings.

---

## Context

V11 cleared the structural gates (PBO 0.126, one-day-lag delta +9.79%) but missed strict significance: PSR 0.944 (just below 0.95), DSR 0.811 (further below). Absolute Sharpe of 0.528 over 9 years is one binding constraint; DSR-under-multi-trial-correction is the other. A small Sharpe lift that doesn't also raise consistency-vs-trial-variance can clear PSR while failing DSR. V12 must lift *both* the point estimate and the effective edge relative to the project's accumulated trial count.

Two independent prior analyses point at the same lever:
- **May 2026 root-cause investigation** found V8 (V0 + BEAR-to-cash) beat V1 (no regime) by ~0.26 Sharpe in EXT-OOS, but V8 failed cost sensitivity at the time (Sharpe -0.714 at 7.5 bps, because non-BEAR daily-rotation costs (~0.10%/day at 5 bps, turnover ~1.0) ate the ~0.045%/day gross edge).
- **2026-05-23 regime detector diagnostic** Phase 5 recommended (c) both tracks in parallel, prioritizing **RAMP BEAR-day cash logic** over detector revision.

V12 is the obvious synthesis: V11's filter base already reduces turnover (rank_buffer + min_hold + delta_rebalance), so non-BEAR cost drag should be lower than V8's. Layering BEAR-to-cash on top tests whether the gross edge minus the (now lower) cost drag clears the cost gate V8 couldn't.

## Goals

1. Variant `V12` registered in `src/research/ramp_phase4/variants.py` such that `cfg.regime_positions` controls per-regime position behavior, with an optional `cfg.min_regime_days` hysteresis parameter.
2. Default config holds cash on BEAR only (UNPREDICTABLE deferred to V11 -- see rev2 note), defers to V11 logic on STRONG_BULL + WEAK_BULL + SIDEWAYS + UNPREDICTABLE, preserves prior positions on SAFE_MODE.
3. Readiness orchestrator re-run with V12 added to the cross-variant PBO set; emit a PSR/DSR/PBO/lag/cost verdict report (5 gates).
4. Detector-onset alignment analysis (rev2 addition) emitted as part of the same readiness report -- not a follow-on artifact.
5. New canonical glossary doc `docs/strategies/RAMP_VARIANTS.md` documenting V01 through V12 with concrete one-liners + parameters.

## Non-goals

- Defensive ticker exposure (SH/TLT/GLD as BEAR-day position). Requires universe extension; deferred to V13. See Appendix C (rev2) for the Kalman parallel-filter constraint that affects how V13 must be designed.
- Per-regime strategy routing (different strategy class per regime). Requires adapter layer; deferred to V13+.
- Production paper deploy of V12. Gated on readiness verdict. If V12 clears, deploy mirrors V11's path (toggle.yaml `variant: v12`, A7 comparator extended) but only after the IBKR migration paper-comparator framework can run V11 and V12 in parallel.
- Modifying the detector itself. WS-3 (v1 detector with hysteresis) is conditional and out of scope here.

## Design

### Variant implementation

`_variant_v12(t, state, panel, cfg)` in `src/research/ramp_phase4/variants.py`:

1. Call `_variant_v11(t, state, panel, cfg)` to get the V11 plan.
2. Read `regime = plan_output['__regime__']`.
3. **rev2: hysteresis gate** -- if `cfg.min_regime_days > 0`, check `state.regime_streak.get(regime, 0)`. If streak < `min_regime_days`, return the V11 plan unchanged (treat as if regime hadn't switched yet). Streak is maintained by the engine as a per-regime running count of consecutive days; resets on any regime change. If the engine doesn't already track this, add it as a sub-task (see test plan).
4. Look up `position_mode = cfg.regime_positions.get(regime, 'normal')`.
5. Branch:
   - `'normal'`: return V11's output unchanged.
   - `'cash'`: return `{'__regime__': regime}` (no target weights -> engine liquidates).
   - `'hold'`: return `{'__regime__': 'SAFE_MODE'}` (engine preserves positions, no trades).
   - Other (TICKER or strategy name): raise `NotImplementedError` -- reserved for V13+.

The variant must NOT mutate state directly; it returns a fresh dict each call. `state.regime_streak` is mutated by the engine in the per-tick post-processing.

### Config schema

Add to `src/research/ramp_phase4/config.py::HarnessConfig`:

```python
regime_positions: Dict[str, str] = field(default_factory=lambda: {
    'STRONG_BULL':   'normal',
    'WEAK_BULL':     'normal',
    'SIDEWAYS':      'normal',
    'UNPREDICTABLE': 'normal',   # rev2: was 'cash'; UNPREDICTABLE has positive return contribution in EXT-OOS (+44.6% over 4 days). Cash version available as A/B.
    'BEAR':          'cash',
    'SAFE_MODE':     'hold',
})
min_regime_days: int = 0  # rev2: hysteresis on regime flip -> position mode change. 0 = no hysteresis (original behavior).
```

Validation: raise `ValueError` in `HarnessConfig.__post_init__` if any value in `regime_positions` is not one of `{'normal', 'cash', 'hold'}`, or if `min_regime_days < 0`. Allow unknown KEYS (regime names) to fall through to `'normal'` -- future-proofing in case the detector emits a new regime name.

### Rationale: UNPREDICTABLE default flipped to `'normal'` (rev2)

V8's EXT-OOS regime breakdown:

| Regime | % of days | Sharpe | Return contrib |
|---|---|---|---|
| UNPREDICTABLE | 1.2% | 7.668 | +44.6% |

n=4 days, so the point estimate is a statistical artifact. But the *direction* of the artifact says UNPREDICTABLE outperforms, not underperforms. Defaulting to cash is going in the wrong direction relative to the data we have. The principled default for a regime with insufficient data is "defer to existing logic" (`normal`), not "act defensively." A cash version is run as an A/B variant in the readiness orchestrator (see `Readiness orchestrator changes`), so we don't lose the ability to test it -- we just don't bake the untested choice into the v12.0.0 default.

### Engine cash-handling

Confirm before implementation: when `target_weights == {}` and regime != `'SAFE_MODE'`, does the engine liquidate all positions?

Tracing `src/research/ramp_phase4/engine.py:74-130`: target_weights empty -> `compute_trades` sees all current positions and zero targets -> generates sell trades for each held position. Yes, the engine already does the right thing for empty target_weights.

If the engine actually treats empty as "no-op" instead of "liquidate", that's a contract bug we'd need to fix as a sub-task. The implementer verifies this in the first test (`test_v12_bear_day_liquidates_all_positions`).

### Engine regime-streak tracking (rev2)

If `min_regime_days > 0`, the engine must track per-regime consecutive day count and pass it into `state` for the variant to read. Approximately 5 LOC in `engine.py`'s per-tick loop:

```python
if state.last_regime == regime:
    state.regime_streak[regime] = state.regime_streak.get(regime, 0) + 1
else:
    state.regime_streak = {regime: 1}  # reset on flip
state.last_regime = regime
```

Default `min_regime_days = 0` means this is computed but never gated on, so existing variants (V01-V11) are bit-equivalent.

### Re-entry semantics

When regime flips BEAR -> non-BEAR:
- Day T (BEAR): V12 returned cash. After execution, `state.positions = {}` and `state.position_open_dates = {}`.
- Day T+1 (non-BEAR, say WEAK_BULL): V12 calls V11. V11's `rank_buffer` checks `state.positions` (empty -> no held names retained, no-op). `min_hold` checks `state.position_open_dates` (empty -> no protection, no-op). V11 returns its standard top_n picks. Engine buys them at day T+1's prices.

No special re-entry logic needed. V11's filter machinery handles it correctly because empty state defaults trigger no protections. With `min_regime_days > 0`, the same flip stalls until non-BEAR streak >= threshold, but the semantics are still correct.

### Cost realism

Engine already models `cost_bps_per_side` per trade. A BEAR regime onset costs ~5 bps x N positions ≈ ~50 bps round-trip for full liquidation (V11's typical N=10, top_n varies by regime). Re-entry costs another ~50 bps. So a single BEAR-then-recover cycle is ~100 bps of friction.

The cost-sensitivity gate at 7.5 bps tests whether this cost is acceptable. V12 must pass it as a **hard requirement** (rev2: now explicit in pass criteria, not implicit).

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
- V13 candidate: defensive ticker support (SH/TLT/GLD as BEAR-day position) -- see Kalman constraint, Appendix C
- V14 candidate: per-regime strategy routing (RAMP for bull, OMR for sideways, etc.)
```

For V01-V11 the descriptions are pulled from existing reports + the inline docstrings in `variants.py`. For V12 onwards the entry is written at spec/plan/implementation time.

## Open questions / room to revise

These remain explicitly deferred until readiness orchestrator output is in hand:

1. **Default for SIDEWAYS**: shipped as `'normal'` (V11 logic). If readiness shows V11 SIDEWAYS days are net-negative, switch default to `'cash'`. Decided post-readiness.
2. **UNPREDICTABLE cash version**: now ships as an explicit A/B variant (`V12-up-cash`) in the orchestrator (rev2). Post-readiness, if `V12-up-cash` beats default V12 by >= 0.05 Sharpe and PBO doesn't degrade, swap defaults.
3. **`min_regime_days` value**: shipped at 0; A/B tested at {0, 2, 3, 5}. Post-readiness, pick the value that maximizes cost-tier Sharpe at 7.5 bps. If all four are within noise (SE ~0.17 over the test window), keep 0 (simpler is better).
4. **Defensive ticker support**: deferred to V13. Universe expansion (`SH`, `TLT`, `GLD`) is non-trivial and deserves its own spec. See Appendix C for the Kalman parallel-filter constraint that affects how V13 must be designed.
5. **Strategy routing**: deferred to V13+ once we have a per-regime adapter layer.

## Test plan

Add to `tests/research/ramp_phase4/test_variants.py`:

```python
def test_v12_normal_regime_matches_v11():
    """V12 with default config on STRONG_BULL day == V11 output exactly."""

def test_v12_bear_day_returns_empty_targets():
    """V12 on BEAR day returns {'__regime__': 'BEAR'} with no weights."""

def test_v12_unpredictable_day_defaults_to_v11():
    """V12 on UNPREDICTABLE day with default config == V11 output (rev2: was cash, now normal)."""

def test_v12_unpredictable_day_returns_cash_when_configured():
    """V12 with cfg.regime_positions['UNPREDICTABLE'] = 'cash' returns empty targets on UNPREDICTABLE."""

def test_v12_sideways_default_matches_v11():
    """V12 SIDEWAYS day with default config == V11 output (because default is 'normal')."""

def test_v12_safe_mode_preserves_positions():
    """V12 on SAFE_MODE returns {'__regime__': 'SAFE_MODE'} -- engine preserves positions."""

def test_v12_bear_then_safe_mode_stays_in_cash():
    """rev2: regime sequence BEAR -> SAFE_MODE while V12 holds nothing: positions stay empty, no re-entry."""

def test_v12_config_override_sideways_to_cash():
    """V12 with cfg.regime_positions['SIDEWAYS'] = 'cash' returns empty targets on SIDEWAYS."""

def test_v12_min_regime_days_blocks_premature_flip():
    """rev2: with min_regime_days=3, BEAR streak=2 -> V11 logic (not cash); BEAR streak=3 -> cash."""

def test_v12_min_regime_days_resets_on_flip():
    """rev2: BEAR streak=5, then WEAK_BULL day 1 -> streak resets; day 1 of WEAK_BULL gates as streak=1."""

def test_harness_config_rejects_unknown_position_value():
    """HarnessConfig validation raises ValueError on regime_positions value other than {normal, cash, hold}."""

def test_harness_config_rejects_negative_min_regime_days():
    """rev2: HarnessConfig validation raises ValueError on min_regime_days < 0."""
```

Plus an integration test verifying that running V12 through `run_variant()` on a synthetic 10-day panel where regime transitions BEAR -> WEAK_BULL produces the expected liquidate-then-rebuild trade sequence. A second integration test verifies hysteresis: BEAR-BEAR-WEAK_BULL-BEAR-BEAR-BEAR (with `min_regime_days=3`) produces cash only after the third consecutive BEAR.

## Readiness orchestrator changes

New file `scripts/backtest_scripts/ramp_phase4_v12_readiness.py` mirroring V11's structure:
- `CROSS_VARIANTS = ('V01', 'V04', 'V05', 'V06', 'V11', 'V12')` -- six variants for PBO.
- Replace `'V11'` -> `'V12'` as the gate target throughout.
- rev2: matrix expanded to test A/B variants in the same orchestrator run:
  - **Cost grid**: V12 across 4 cost tiers (1, 5, 7.5, 10 bps) x 2 lag modes (near_close, one_day_lag) = 8 runs.
  - **UNPREDICTABLE A/B**: `V12-up-cash` (UNPREDICTABLE='cash') at 5 bps near_close = 1 run.
  - **Hysteresis A/B**: `V12-hyst-2`, `V12-hyst-3`, `V12-hyst-5` (min_regime_days=2/3/5) at 5 bps near_close = 3 runs.
  - **Cross-variants for PBO**: V01, V04, V05, V06, V11 at 5 bps near_close = 5 runs.
  - V12 default at 5 bps near_close is shared with the cost-grid table (no double-counting).
  - **Total: 8 + 1 + 3 + 5 = 17 runs**. Estimated wall-clock: ~16-18 min on t4g.medium.

Output: `docs/reports/ramp/YYYYMMDD_phase4_v12_readiness.md`, with the new sections:

- **Detector-onset alignment panel** (rev2 first-class deliverable). For each detected BEAR period in the test window (2017-2025):
  - SPY price trajectory from day -20 through day +30 relative to detector flip-to-BEAR.
  - V12 cash window overlay (start/end days within the trajectory).
  - V12's avoided return = sum of regime-day returns during cash window.
  - Compare: "detector-perfect" BEAR-avoidance (BEAR-onset = SPY drawdown trough +/- 1 day, hypothetical) vs. realized V12 BEAR-avoidance. Gap = lag tax.

  This panel is the input to the V12-vs-WS-3 decision. If the lag tax is large but V12 still pays off, BEAR-to-cash is a real edge that gets bigger with better detection -> deploy V12, queue WS-3 as additive improvement. If V12's apparent benefit shrinks to zero when corrected for lag, V12 is a coincidence -> WS-3 is the right priority.

- **Hysteresis A/B comparison table**: Sharpe at 1 / 5 / 7.5 / 10 bps for `min_regime_days` in {0, 2, 3, 5}. The picked value goes into the default for any subsequent paper deploy.

- **UNPREDICTABLE A/B comparison**: V12-default vs. V12-up-cash at 5 bps, both Sharpe and Max DD.

## Pass criteria (rev2: 5 gates, was 4)

V12 passes the readiness gate if ALL five hold:

1. **PSR (vs SR=0) > 0.95** -- absolute significance.
2. **DSR (n_trials = project-wide cumulative count from `output/experiments.duckdb` Section 9.4) > 0.95** -- multi-trial-corrected significance.
   - rev2: document the actual `n_trials` value pulled at orchestrator-run time. If unknown at spec time, the orchestrator queries the registry. If `n_trials` >= 30, this gate is structurally harder than at V11 time and the implementer should report which variants would have passed at V11's trial count for context.
3. **PBO across 6 variants < 0.5** -- low overfitting evidence.
4. **One-day-lag Sharpe at 5 bps within 20% of near_close Sharpe** -- no structural lookahead.
5. **rev2 (new): Net Sharpe at 7.5 bps (1.5x of 5 bps base) > 0.3** -- cost robustness. This is V12's reason for existing relative to V8 and was previously implicit in the methodology Section 4 gate. Made explicit because the methodology gate is the floor and V12 must clear it by construction.

If V12 clears all 5: it becomes the Phase D candidate (skip WS-3, redeploy mirroring V11's deploy path *after* the IBKR-migration paper-comparator framework can run V11 and V12 in parallel paper).

If V12 clears structural (PBO, lag, cost) but misses PSR or DSR: compare absolute Sharpe to V11; pick better; consider WS-3 conditional on detector-lag-analysis output.

If V12 fails structural: investigate. Possibly BEAR-to-cash relied on detector-lagged drawdown selection -- a non-trivial result that the detector-onset alignment panel will help interpret.

## Expected magnitude of lift (rev2 sanity check)

V8 lifted V0 by +0.501 Sharpe in EXT-OOS (2025-2026), but EXT-OOS had BEAR at 19.3% of days -- unusually BEAR-heavy. The 9-year readiness window (2017-2025) has BEAR closer to ~10-12% of days on average. Naive scaling:

```
expected_full_window_lift ~ BEAR_fraction * Sharpe_drag_avoided
                          ~ 0.10 * (0.5 to 1.0)
                          ~ 0.05 to 0.10
```

So the "Sharpe lift >= 0.15 over V11" success criterion (below) is *tight* against the realistic ceiling, not comfortably above it. **The modal outcome is probably criterion 2 (Max DD reduction with marginal Sharpe lift), not criterion 1 (clear Sharpe lift).** This isn't a bug in V12 -- it reflects that the strongest BEAR-day evidence was on a window where BEAR was 2x more frequent than the population average.

If the readiness report shows V12 at +0.05 to +0.12 Sharpe over V11 with a 10-15 percentage point Max DD reduction, that's the *expected* outcome, not a failure.

## Risk table

| Risk | Probability | Impact | Mitigation |
|------|---|---|---|
| Engine treats empty target_weights as no-op instead of liquidate | Low | High | First test in the implementation explicitly verifies this; if false, fix engine before V12 logic. |
| V12 ties or loses to V11 because UNPREDICTABLE is too rare to matter | Low | Low | rev2: UNPREDICTABLE default is now `normal` so the BEAR signal is isolated. `V12-up-cash` A/B variant in orchestrator independently measures UNPREDICTABLE contribution. |
| BEAR-to-cash whipsaws (regime flips frequently around detector threshold) | Medium-High | Medium | rev2: BEAR median run length ~3-4 days + detector lag ~14 days makes whipsaw the expected mode. `min_regime_days` parameter ships exposed and A/B tested at {0, 2, 3, 5}. If V12 readiness shows high turnover with no Sharpe lift on default, pick the best hysteresis value as the default before deployment. |
| Sharpe lift doesn't materialize because V11's filters already neutralize some BEAR-day exposure | Medium | High | The May 2026 V8 finding was on V0 (no filters), not V11. V11's rank_buffer already retains held names through soft BEAR signals. If V12 only marginally improves V11, that's a finding -- not necessarily a failure (criterion 2 still produces a defensible deployment). |
| Re-entry after BEAR re-enters with fresh open_dates so min_hold can't protect | Low | Low | Intentional. Fresh-entry positions are not yet aged into protection. Normal V11 behavior. |
| rev2: BEAR-to-cash benefit is mostly accidental (avoiding lagged-bottoming days, not the actual selloff) | Medium | High | Detector-onset alignment panel is the diagnostic. If gap between detector-perfect and realized V12 benefit is large, WS-3 (detector improvement) takes priority over V12 deployment. |
| rev2: DSR gate gets harder as project-wide trial count grows | Medium | Medium | `n_trials` documented in readiness report. Implementer reports DSR at both current and V11-era trial counts for context. |
| rev2: V12 ships with `min_regime_days=0` and the readiness picks a higher value as optimal; default in code is then stale | Low | Low | Post-readiness, update HarnessConfig default to the picked value. One-line change. |

## Success criteria (rev2 reframed)

V12 succeeds if ANY of:

1. **Tier 1 (preferred)**: V12 clears all 5 readiness gates AND lifts net Sharpe at 5 bps by >= 0.15 over V11. -> Phase D candidate, deploy mirroring V11 path (after IBKR-migration paper-comparator gate).
2. **Tier 2 (modal expected outcome, rev2 reframing)**: V12 clears all 5 readiness gates but lifts Sharpe by < 0.15. Max DD must be reduced by >= 10 percentage points absolute. -> Preferred candidate for risk-conscious deployment; same deployment path as Tier 1.
3. **Tier 3 (diagnostic value)**: V12 clears structural gates (PBO, lag, cost) but misses PSR/DSR. Detector-onset alignment panel must show gap > +0.10 between detector-perfect and realized V12 lift. -> Activate WS-3 (detector improvement) as the higher-leverage path; V12 deployment deferred until WS-3 + V12 readiness re-run.
4. **Tier 4 (failure)**: V12 fails structural gates or shows no diagnostic signal in the alignment panel. -> Strategy reset; reconsider regime overlay's role from scratch.

In Tiers 1-3, the spec is "successful" because it produces a defensible decision. Tier 4 means the spec failed to advance the question, which is itself useful information.

## Decision gates

1. **After variant implementation + unit tests**: 11+ unit tests pass + integration tests on synthetic panels (incl. hysteresis test). If any fail, fix before readiness re-run.
2. **After readiness orchestrator**: review the five-gate verdict + Max DD change vs V11 + detector-onset alignment panel + hysteresis A/B output + UNPREDICTABLE A/B output. Apply the success-criteria branching above. If a hysteresis value or UNPREDICTABLE mode beats default by > noise, update the default before any paper deploy.
3. **Post-decision**: either proceed to V12 production paper deploy (mirror V11 path, after IBKR-migration comparator gate) or activate WS-3.

## Appendix A -- File touchpoints

New files:
- `docs/superpowers/specs/2026-05-23-v12-bear-to-cash-design.md` (this doc; rev2 in place of rev1)
- `docs/superpowers/plans/2026-05-23-v12-bear-to-cash.md` (implementation plan; from writing-plans)
- `docs/strategies/RAMP_VARIANTS.md` (canonical glossary; one-time setup populated for V01-V12)
- `scripts/backtest_scripts/ramp_phase4_v12_readiness.py` (V12 readiness orchestrator with rev2 A/B matrix)
- `docs/reports/ramp/YYYYMMDD_phase4_v12_readiness.md` (readiness output; emitted by orchestrator; rev2 includes detector-onset alignment panel and hysteresis A/B section)
- `docs/progress/YYYYMMDD_RAMP_V12_SESSION_LOG.md` (session log at end)

Modified files:
- `src/research/ramp_phase4/variants.py` (+ `_variant_v12`, +`'V12'` in REGISTRY; ~50 LOC + ~15 LOC for hysteresis check ≈ ~65 LOC)
- `src/research/ramp_phase4/config.py` (+ `regime_positions` field + `min_regime_days` field + validation; ~25 LOC)
- `src/research/ramp_phase4/engine.py` (+ regime_streak tracking in per-tick loop; ~5 LOC behind `min_regime_days > 0` no-op when default)
- `tests/research/ramp_phase4/test_variants.py` (+ 11 unit tests; ~200 LOC)
- `tests/research/ramp_phase4/test_engine.py` (+ 1 test for regime_streak tracking; ~30 LOC)

Branch: `v12-bear-to-cash` (new, based on `ramp-phase4-turnover-regime-research`).

## Appendix B -- Why this is conservative for the first iteration

The roadmap (`docs/superpowers/plans/2026-05-23-ramp-research-roadmap.md`) WS-2.1 brainstorm topics included "defensive asset" and "strategy routing" as questions. Both are deferred to V13+. Reasons:

1. **YAGNI on instruments, not on parameters (rev2 clarification)**: V12 with cash-only already tests the dominant hypothesis (BEAR exposure is the dominant lever). If cash works, more sophisticated alternatives (SH/TLT/GLD) are incremental tuning. If cash doesn't, more sophistication is unlikely to. *However*, parameters whose absence forces a re-run of expensive backtest infrastructure (e.g. `min_regime_days`) are NOT YAGNI candidates -- they ship exposed with sensible defaults so the readiness orchestrator can A/B them in one wall-clock window instead of two. rev2 ships `min_regime_days` for exactly this reason.
2. **Universe constraints**: SH/TLT/GLD aren't in `sp500-2025.csv`. Adding them touches the SIP daily fetcher, the variant data loader, and the production paper deploy's universe config. Real work; deserves a separate spec.
3. **Strategy routing complexity**: requires running TWO live engines simultaneously and switching between them based on regime. The current Phase 4 harness has no such abstraction. V14 territory.

V12 ships the minimum needed to test the regime-conditional-exposure idea, plus the one piece of optionality (`min_regime_days`) whose absence would cost a re-run if the data demands it.

## Appendix C -- Kalman parallel-filter constraint for V13+ defensive ticker work (rev2 addition)

Worth recording here because the V13 spec will need to confront it.

RAMP's `MarketRegimeDetector` requires three parallel Kalman filters (fast/medium/slow) to preserve the three-SMA structure (`above_20`, `above_50`, `above_200`) used in `_score_regime()`. Collapsing to a single trend estimate breaks regime classification.

What this means for V13:

- **V13a (simple swap)**: BEAR-day position becomes SH/TLT/GLD (defensive tickers) instead of cash, but classification is still SPY-based via the three-Kalman MarketRegimeDetector. The defensive ticker is just an alternative target weight when V12 would have signaled cash. Universe expansion is real (SH/TLT/GLD added to the data pipeline + ticker config + execution path) but the detection architecture is unchanged.
- **V13b / V14 (strategy routing)**: different strategy classes per regime, with state and execution context routed via a per-regime adapter layer. The detection architecture is still SPY-based via the three-Kalman MarketRegimeDetector; the *response* to the regime classification is per-regime strategy invocation.

The constraint forces V13a to be the simpler of the two, which is why V13 is the right name for the defensive-ticker variant and V14 for routed strategies. The constraint does NOT block V13a; it just defines what V13a is.
