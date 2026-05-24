# V14 Factorial Spec rev2 -- Open-Questions Resolution

**Date**: 2026-05-24
**Status**: Resolves the 5 open questions in `docs/superpowers/specs/2026-05-24-v14-soft-bear-factorial-design.md` "Open questions to resolve before implementation" section.
**Branch**: v12-bear-to-cash

This addendum unblocks implementation. Each open question is answered with the supporting code/data reference. Where the answer changes the implementation plan, the change is itemized.

---

## Q1: Is `classify_regime` currently idempotent?

**Answer**: Yes in the **output** sense (deterministic; same inputs -> same outputs). No in the **caching** sense (no memoization; the function recomputes on every call).

**Evidence**: `src/strategies/advanced/market_regime_detector.py:111-188`
- `classify_regime(spy_data, vix_data, timestamp, ...)` has no internal state that varies across same-input calls.
- Computes coverage check, calls `_calculate_indicators` (rolling stats over the input frames), scores 5 regimes against `REGIME_CRITERIA`, returns argmax + confidence.
- No randomness, no global state, no time-dependent branches beyond the `timestamp` argument.
- Sets `self.last_indicators` and `self.last_regime_scores` as side effects.

**Implementation impact**:
- The rev2 spec's "freshness assertion" via `last_classification_timestamp == t` requires that field to exist. It does **not** currently exist on the detector. Add as a 2-line additive change:

```python
# src/strategies/advanced/market_regime_detector.py
# In __init__:
self.last_classification_timestamp: Optional[datetime] = None
# Just before the return in classify_regime (line ~188):
self.last_classification_timestamp = timestamp
```

This is a read-only addition (no logic change; no behavior change for V01-V13). Spec validation gate #4 (idempotency confirmed or refactored) is satisfied by this addition alone -- no cache needed because:
- The recompute cost per call is one indicators pass + 5 regime scorings = a few ms on a 252-day window. Trivial relative to the 30-min orchestrator wall-clock.
- Output is already deterministic, so a double-call produces consistent state.

**No cache refactor is required.** The rev2 spec's optional cache fork is not taken.

---

## Q2: What is the actual V14-warm-start date?

**Answer**: **2017-01-03**, identical to the V11 warm-start.

**Evidence**: `diagnostics/data/spy_vix_2016_2026.parquet` and `diagnostics/regime/v0/labels.parquet`
- Panel rows: 2,612 covering 2016-01-04 to 2026-05-22 (warm-up year 2016 + 2017-2026 evaluation).
- Labels rows: 2,360 covering 2017-01-03 to 2026-05-22 (replay starts at 2017-01-03 with first valid regime classification).
- First non-SAFE_MODE label: 2017-01-03 (252-day VIX percentile + 200-day SMA both satisfied by the 2016 warm-up).

V11 calls `_DETECTOR.classify_regime` via `_compute_plan_from_panel` (`src/research/ramp_phase4/variants.py:62`), which requires `len(spy_slice) >= 252 and len(vix_slice) >= 252` (line 52 of variants.py). The same 252-day requirement determines V14's first valid scoring day. Both variants therefore have the **same warm-up**.

**Implementation impact**: The rev2 spec's warm-up-parity concern is automatically satisfied -- no special V14-warm-start window slicing is needed. V11's reference Sharpe from the V11 readiness report (computed over 2017-01-03 to 2026-05-22, ~2,360 trading days, ~9.4 years) **is** the V14-window V11 Sharpe.

**Action**: Spec's "Reference Sharpe convention" section is correct in principle but the conversion is a no-op for this implementation; the V11 reference Sharpe values cited in the V12c readiness report (V11 = 0.5306 at 7.5 bps lag) carry over unchanged.

---

## Q3: Does V11's `position_open_dates` mechanism interact with V14 mid-stream regime entry?

**Answer**: No -- V14's plan-replacement is clean.

**Evidence**: V14a's sketch in the rev2 spec replaces V11's plan output entirely after the call:

```python
plan = _variant_v11(t, state, panel, cfg)  # V11 runs through rank_buffer + min_hold
if state.in_bear_soft_mode:
    return PLAN_CASH_BEAR_SOFT              # V11's plan is discarded
return plan
```

When V11's plan is discarded, the filters (`rank_buffer`, `min_hold` in `src/research/ramp_phase4/filters.py`) have already run -- they affect the contents of `plan` but not the engine's state. The engine then receives `PLAN_CASH_BEAR_SOFT`, which is a `_SentinelPlan(reason='BEAR_SOFT_CASH')`. The engine's dispatch (rev2 spec sketch) routes this to `zero_target_orders()` -- generating sells for ALL current positions regardless of `state.position_open_dates`.

**Exit cost accounting**: A sell at close T pays one half-spread cost. The engine's existing dict-based diff (target = {} vs current = {...}) generates one sell per position. No double-counting.

For V14b (SPY) and V14c (dampen), the diff against V11's filtered plan is also clean -- the engine treats `{SPY: 1.0}` (V14b) or `V11_plan_weights * 0.5` (V14c) as new targets, computes the diff, and applies one cost per generated order.

**Implementation impact**: None. The spec's concern is unfounded for the current engine architecture. Document in the orchestrator that "V14 plans override V11's filter outputs; min_hold protection does NOT apply on V14a/b/c transition days by design."

---

## Q4: Is V11's gross constant or time-varying (for V14b allocation)?

**Answer**: Constant at **1.0**.

**Evidence**: `src/research/ramp_phase4/variants.py:94-108` (V01), :190+ (V11)
- V01: `per_weight = 1.0 / len(targets)` (line 105) -- ignores `plan.exposure_pct`. Sum of weights = 1.0 always.
- V11 builds on V01: V01 plan -> rank_buffer (output renormalized to sum 1.0 per `filters.py:57`) -> min_hold (output renormalized to sum 1.0 per `filters.py:109`).
- Delta-threshold filter (V06/V11 via cfg.delta_rebalance_pct) is a trade-skip filter, not a weight-rescaling filter -- it preserves target sums.

V11's gross is **always 1.0**, every day, every regime. (RAMP_VARIANTS.md V01 line: "ignores crash exposure.")

**Implementation impact**: V14b's "100% of V11's gross to SPY" simplifies to **`{SPY: 1.0}`** as a fixed allocation. No same-day-gross-matching code path needed. The variant returns `{SPY: 1.0, '__regime__': 'BEAR'}` (or equivalent sentinel) when `in_bear_soft_mode`.

The "leverage cap" concern in the rev2 spec risk table is also moot here -- V11's gross of 1.0 is below any reasonable leverage cap. V14b's `{SPY: 1.0}` is a single-name 100%-gross position; this is concentration risk (already flagged in the spec) but not a leverage-cap issue.

---

## Q5: Is the G1_BEAR labeler locked since the diagnostic?

**Answer**: Yes. **One commit** in history; never modified since.

**Evidence**: `git log --oneline -- scripts/diagnostics/ground_truth_labelers.py`:
```
9c48245 diagnostic(regime): Phase 3 ground-truth labelers
```
- Single commit, dated before the V12/V12c/V13 work.
- Function `label_g1_drawdown_bear` (line 19 of the file) is the canonical G1_BEAR labeler.
- The function definition has not been touched.

**Implementation impact**: The pre-spec script `scripts/diagnostics/compute_tau_in_from_g1.py` can import `label_g1_drawdown_bear` directly. The labeler version is the commit hash `9c48245`; record it in `v14_tau_constants.json` alongside the computed tau values for reproducibility:

```json
{
  "tau_in": <computed_value>,
  "tau_out": <tau_in - 0.1>,
  "g1_labeler_commit": "9c48245",
  "computed_at": "<ISO timestamp>",
  "computation_script": "scripts/diagnostics/compute_tau_in_from_g1.py",
  "computation_method": "median BEAR_score on G1_BEAR days (drawdown > 10% from 252-day trailing high)",
  "source_data": {
    "labels": "diagnostics/regime/v0/labels.parquet",
    "scores": "diagnostics/regime/v0_scores/labels.parquet"
  }
}
```

If `label_g1_drawdown_bear` is ever modified after this point, the JSON's `g1_labeler_commit` field will diverge from `git rev-parse HEAD -- scripts/diagnostics/ground_truth_labelers.py`, and any V14 re-run must re-derive tau.

---

## Summary of implementation adjustments vs rev2 spec

| Rev2 section | Adjustment | Net effect |
|---|---|---|
| Detector idempotency cache (Q1) | NOT needed. Add only `last_classification_timestamp` field (2 lines). | Saves implementation effort; preserves cheap-recompute semantics. |
| V14-warm-start window slicing (Q2) | NOT needed. V14-warm-start == V11-warm-start == 2017-01-03. | V11 reference Sharpes from V12c readiness report carry over unchanged. Orchestrator code simpler. |
| V11 position_open_dates interaction (Q3) | NOT a concern with current engine. | Document explicitly in orchestrator + pinning tests: V14 plans override V11 filter outputs by design. |
| V14b gross-preserving SPY (Q4) | V14b in_bear_soft_mode returns fixed `{SPY: 1.0}`. | No same-day-gross-matching code; pinning test simpler. |
| G1 labeler version pinning (Q5) | Record `g1_labeler_commit: '9c48245'` in `v14_tau_constants.json`. | Reproducibility cite locked. |

**No spec changes triggered.** All five questions are resolved without modifying rev2's design choices. The rev2 spec is **ready for implementation**.

---

## Implementation plan kickoff checklist

Before launching subagents:

- [ ] User reviews this addendum and the rev2 spec.
- [ ] User approves writing the implementation plan via `writing-plans` skill.
- [ ] Plan target file: `docs/superpowers/plans/2026-05-24-v14-soft-bear-factorial.md`.

Plan-time considerations from rev2 + this addendum:

1. **Task 0** (pre-spec): write `scripts/diagnostics/compute_tau_in_from_g1.py`, run it, commit `config/research/v14_tau_constants.json` with the G1 labeler commit pinned.
2. **Task 1**: add `MarketRegimeDetector.last_classification_timestamp` (2-line additive change).
3. **Task 2**: create `src/research/ramp_phase4/plans.py` with `_SentinelPlan` + `PLAN_CASH_BEAR_SOFT`.
4. **Task 3**: extend engine -- `state.in_bear_soft_mode` field + `_engine_pre_variant_update_soft_bear` + `_SentinelPlan` dispatch.
5. **Task 4**: extend config -- `soft_bear_tau_in`/`soft_bear_tau_out`/`soft_bear_dampen_factor` + JSON loader + validation predicate.
6. **Task 5**: add `_variant_v14a_soft_bear_cash`, `_variant_v14b_soft_bear_spy`, `_variant_v14c_soft_bear_dampen` + REGISTRY entries.
7. **Task 6**: write 50+ tests covering plan sentinel, state machine boundaries, three canonical pinning tests, detector freshness, V14b gross fixed-allocation.
8. **Task 7**: clone V12c readiness orchestrator -> `scripts/backtest_scripts/ramp_phase4_v14_factorial_readiness.py` with 35-backtest grid + DSR n_trials=36 + 8-variant gate PBO + 4-variant diagnostic PBO.
9. **Task 8**: run orchestrator (~30 min wall-clock), produce `docs/reports/ramp/20260526_phase4_v14_factorial_readiness.md`.
10. **Task 9**: write `docs/progress/20260524_RAMP_V14_FACTORIAL_READINESS.md` session log + update `docs/strategies/RAMP_VARIANTS.md`.

Cycle time estimate (per rev2): 4-5 days analyst + ~30 min compute. Subagent-driven execution should compress this materially since the implementation is mostly mechanical given the canonical pinning tests as source of truth.
