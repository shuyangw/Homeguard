# V12 Phase D Readiness -- Session Log (2026-05-24)

## Summary

V12 (BEAR-to-cash on V11 base) readiness orchestrator ran 18 backtests in 15.86 min on local fintech env. Verdict (after Gate 4 implementation fix): **PARTIAL / Tier 3 (diagnostic value)** per spec rev4 success criteria. Structural gates pass, absolute-significance gates fail. Detector lag tax is the binding constraint. **V12 deployment deferred**; WS-3 (regime detector improvement) is the next higher-leverage research priority.

## What ran

- 18 backtests in `scripts/backtest_scripts/ramp_phase4_v12_readiness.py`:
  - 14 gate-influencing: V12 cost grid {1, 5, 7.5, 10} bps x {near_close, one_day_lag} = 8 + cross-variants V01/V04/V05/V06/V11 at 5 bps near_close = 5 + V11 reference at 7.5 bps one_day_lag = 1.
  - 4 sensitivity-appendix: V12-up-cash (UNPREDICTABLE='cash') + V12-deb-{2, 3, 5} (min_regime_days=2/3/5) at 5 bps near_close.
- Output: `docs/reports/ramp/20260524_phase4_v12_readiness.md`.
- Wall-clock: 15.86 min.
- n_trials_project: 22 (4 from experiments.duckdb + 18 from this run).

## Headline verdict (after Gate 4 fix)

| Gate | Result | Value | Threshold |
|---|:---:|---:|---:|
| 1. PSR (vs SR=0) | FAIL | 0.7881 | > 0.95 |
| 2. DSR (n_trials=22) | FAIL | 0.5418 | > 0.95 |
| 3. PBO across 6 variants | PASS | 0.3934 | < 0.5 |
| 4. Lag-degradation (5 bps) | PASS (after fix) | nc-lag = -0.397 (lag > nc is safe) | <= 0.100 |
| 5a. Cost floor | PASS | 0.6081 | > 0.30 |
| 5b. No-regress vs V11 | PASS | 0.6081 vs 0.9*V11=0.4776 | >= 0.9*V11 |

**Overall**: PARTIAL / Tier 3.

## Key findings

1. **V12 hurts on near_close, helps on lag.** V12 cost grid at 5 bps: near_close Sharpe = 0.268, one_day_lag Sharpe = 0.665. V11 reference: near_close 0.528, lag 0.580. So V12 lag > V11 lag (+0.085), but V12 near_close << V11 near_close (-0.260). The cost of cycling in/out at the same close is the binding drag in near_close mode; under lag mode the next-day-execution slightly mitigates this.

2. **Detector lag tax confirmed.** 59 BEAR onsets in the 2017-2026 test window. Mean gap_days = -3.42: the detector fires ~3.4 trading days AFTER the SPY drawdown trough on average. V12 cash periods bracket the recovery, not the crash. Mean avoided_return = +0.18% per onset (small). This is the structural reason V12 fails on absolute significance.

3. **V12-up-cash (sensitivity) is surprisingly strong.** UNPREDICTABLE='cash' yields Sharpe 0.586 vs V12 default 0.268 (+0.32), AND beats V11 (0.528) by +0.06. Per spec rev4 honesty discipline this is informational only -- but it motivates a **V12c spec** for the next research cycle.

4. **Debouncing doesn't help materially.** Sensitivity at 5 bps near_close: deb-2=0.130, deb-3=0.315, deb-5=0.437. deb-5 modestly beats V12 default (0.268) but still under-performs V11 (0.528). Combined with BEAR median run length ~3-4 days from the regime diagnostic, this confirms the "no good debouncing value" risk anticipated in spec rev4 -- the detector lag is the binding constraint.

## Gate 4 bug (fixed)

The orchestrator at `scripts/backtest_scripts/ramp_phase4_v12_readiness.py:360` originally implemented `lag_pass = abs(delta_abs) <= cap` (symmetric absolute-value check). Spec rev4 specifies directional `(nc - lag) <= cap`, with the rationale "lag > near_close is the safe direction and is not penalized." Under the bug, V12 with nc=0.268 and lag=0.665 produced |delta|=0.397 > 0.100 -> FAIL. Under the fix, nc-lag=-0.397 <= 0.100 -> PASS.

Fixed in commit `5a88903`. The 2026-05-24 report was emitted under the bug; it was patched directly (commit `409c575`) rather than re-running the orchestrator -- the underlying Sharpes are unchanged; only the Gate 4 verdict label flips. The orchestrator code change ensures future variants (V13+) inherit the directional check.

## Decisions

- **V12 NOT deployed.** Tier 3 verdict per spec success criteria; production paper deploy deferred.
- **WS-3 activated** (regime detector improvement). Per Tier 3 rule, this is the higher-leverage path. WS-3 brainstorm is the top next-session candidate.
- **V12c queued** as a follow-on candidate (UNPREDICTABLE='cash' as v12.1.0 default). Requires its own readiness gate per spec honesty discipline; not an in-place V12 default swap.
- **V11 paper validation continues** independently on `ramp-phase4-turnover-regime-research`. V12 outcome does not change V11's path.

## Commits this session

- `5a88903` fix(orchestrator): V12 Gate 4 directional check (spec rev4)
- `409c575` report(ramp): V12 readiness verdict corrected to Tier 3 (spec rev4 Gate 4 fix)
- `6efa6a5` docs(strategies): RAMP_VARIANTS V12 entry -- Tier 3 verdict + V12c motivation
- `fd28289` docs(strategies): clean V12b paragraph in RAMP_VARIANTS
- `<TBD>` docs(progress): V12 Phase D readiness session log (this file)

## Branch state

Local `v12-bear-to-cash` at `<TBD>` (head after this commit). Local only; not pushed.

## Next session candidates (priority order)

1. **WS-3 brainstorm** -- regime detector improvement spec. Top candidate per Phase 5 of the 2026-05-23 diagnostic + Tier 3 escalation here. Focus: Option B (hysteresis layer at the detector level, not at V12's regime-mode level) per the diagnostic's verdict.
2. **V12c brainstorm** -- UNPREDICTABLE='cash' as v12.1.0 default. Smaller scope than WS-3 (single config change + readiness re-run). Could run in parallel with WS-3 if appetite warrants.
3. **Monitor V11 paper validation** -- A7 counter on EC2. Continues regardless of V12 outcome.

WS-3 and V12c are independent; both could land before the next V12 readiness re-run.
