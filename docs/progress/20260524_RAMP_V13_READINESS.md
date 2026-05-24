# V13-bear-invert Readiness -- Session Log (2026-05-24)

## Summary

V13-bear-invert tests the BEAR-as-buy hypothesis discovered from V12's onset-alignment panel (mean gap_days = -3.42 across 59 events 2017-2026; detector fires AFTER the SPY drawdown trough). V13 inverts V12's sign: on BEAR days, allocate 100% SPY instead of cash. Readiness orchestrator ran 15 backtests in 13.98 min. Verdict: **TIER 4 -- BEAR-as-buy is spurious on this sample.** PBO across 7 variants = 0.629 (structural fail), Sharpe @ 5bps near_close = 0.400 vs V11's 0.528 (-0.128), no-regress vs V11 @ 7.5bps lag FAILS (0.308 < 0.478 = 0.9 * V11). V13 NOT deployed. Continue WS-3c roadmap.

## What ran

- 15 backtests in `scripts/backtest_scripts/ramp_phase4_v13_readiness.py`:
  - V13-bear-invert cost grid {1, 5, 7.5, 10} bps x {near_close, one_day_lag} = 8 runs.
  - Cross-variants at 5 bps near_close: V01, V04, V05, V06, V11, V12 = 6 runs.
  - V11 reference at 7.5 bps one_day_lag (Gate 5) = 1 run.
  - No sensitivity appendix (V13 is structurally V11 + single BEAR-branch differ; no UNPREDICTABLE-cash or debouncing analog).
- Output: `docs/reports/ramp/20260525_phase4_v13_readiness.md`.
- Wall-clock: 13.98 min on local fintech env.
- DSR n_trials_project: 23 (hard-coded; 22 from V12 readiness cumulative + 1 for V13 introduction).
- PBO matrix: 7 variants (V01, V04, V05, V06, V11, V12, V13-bear-invert). Expanded from V12 readiness's 6-variant matrix so V12 vs V13 is a direct PBO neighbor (documented).

## Headline verdict

| Gate | Result | Value | Threshold |
|---|:---:|---:|---:|
| 1. PSR (vs SR=0) | FAIL | 0.8830 | > 0.95 |
| 2. DSR (n_trials=23) | FAIL | 0.7074 | > 0.95 |
| 3. PBO across 7 variants | FAIL | 0.6290 | < 0.50 |
| 4. Lag-degradation (directional) | PASS | nc-lag = +0.019 | <= max(0.2*|nc|, 0.1) = 0.100 |
| 5a. Cost floor | PASS | 0.3080 | > 0.30 |
| 5b. No-regress vs V11 | FAIL | 0.3080 vs 0.9*V11=0.4776 | >= 0.9*V11 |

**Overall**: TIER 4 (structural gate fail + no-regress fail).

## Key findings

1. **V13 strictly worse than V11 on net Sharpe.** V13 @ 5bps near_close = 0.400, V11 inline = 0.528 (delta -0.128). V13 @ 5bps one_day_lag = 0.381, V11 doc reference = 0.580 (delta -0.199). Whatever V13 captures on BEAR onset is overpowered by the V11 momentum positions it abandons.

2. **Gross BEAR-day SPY return is positive but tiny.** Per the alignment panel: 59 BEAR onsets, mean gap_days = -3.42 (confirms V12's finding -- the detector fires ~3.4 trading days AFTER the SPY drawdown trough), mean SPY return during BEAR window = +0.18%. The sign is right (consistent with the hypothesis) but the magnitude is too small to compensate for V11's momentum-name returns that recover into the trough.

3. **Lag-asymmetry is the inverse of V12.** V12 (BEAR -> cash) had lag > near_close (helped on lag); V13 (BEAR -> SPY) has lag = near_close (no asymmetry). V13 at 5bps: nc=0.400, lag=0.381 (-0.019 close to zero). The lag-tax that V12 had is not the V13 story; V13's failure is the absolute level.

4. **PBO degrades sharply when V13 enters the matrix.** V12 readiness PBO across 6 variants {V01,V04,V05,V06,V11,V12} = 0.393 (PASS). V13 readiness PBO across 7 variants {V01,V04,V05,V06,V11,V12,V13} = 0.629 (FAIL). Adding V13 (and V12) to the cross-section makes the rank-ordering less stable across CSCV submatrices -- consistent with the variant set growing more redundant rather than more informative.

5. **gap_days mean reproduced exactly across V12 and V13 alignment panels** (-3.42). The structural lag tax is a property of the detector, not the variant -- as expected; V13's BEAR-onset trigger uses the same MarketRegimeDetector.

## Tier verdict + honesty discipline framing

- **TIER 4** per V13 spec success criteria: structural gate (PBO) fails AND no-regress vs V11 fails. The TIER 1 lift threshold (V13 > V11 + 0.10 at 5bps near_close) is not just missed but inverted: V13 - V11 = -0.128.
- **NOT OOS in strict sense.** V13 was discovered from inspection of V12's 2017-2026 BEAR onset panel. The same window is now the test window. The PSR/DSR n_trials_project increment to 23 partially corrects for the multiple-comparison cost of V13's introduction, but the only definitive check is forward OOS data.
- **The negative result is informative.** V13's failure rules out the simplest BEAR-as-buy lever (single-asset SPY on BEAR onset). Any future BEAR-onset strategy needs either (a) a different asset (the V13a/SH/TLT/GLD direction in the old reservation note) or (b) detector improvement so the BEAR onset fires earlier (WS-3).

## Decisions

- **V13 NOT deployed.** Tier 4 verdict; PBO structural fail + no-regress fail are both blocking.
- **Continue WS-3c roadmap.** E3 (soft scores) produced the WS-3c verdict; V13 spurious doesn't change that. The detector lag tax is real but the BEAR-as-buy single-SPY lever is not the answer.
- **V13-defensive (SH/TLT/GLD) NOT motivated** by this run. The mean SPY return of +0.18% during the BEAR window is suggestive but tiny; replacing SPY with a defensive sleeve would have to overcome both the V11 momentum opportunity cost AND the cross-asset basis risk. Not a higher-leverage path than WS-3.

## Commits this session

- `728919a` feat(variants): V13-bear-invert -- BEAR onset goes to SPY 100% (experiment 1)
- `acabc0f` feat(orchestrator): V13 readiness gate -- 5 gates with DSR n_trials=23, PBO includes V13 (experiment 1)
- `0d21b32` report(ramp): V13-bear-invert readiness -- BEAR-as-buy verdict (RAMP_VARIANTS update)
- `7da5970` (parallel E6 agent commit) bundled the report file `docs/reports/ramp/20260525_phase4_v13_readiness.md` due to concurrent staging; the file content was produced by this V13 orchestrator run.
- `<TBD>` docs(progress): V13 readiness session log (this file)

## Branch state

Local `v12-bear-to-cash` branch. Not pushed.

## Next session candidates (priority order)

1. **WS-3 (detector improvement)** remains the binding constraint. Both V12 and V13 land at the same wall -- the detector's structural lag (mean gap_days -3.4 across 59 events) caps what any BEAR-onset variant can achieve. WS-3c roadmap from E3 is the path.
2. **Forward OOS validation of V11** continues independently on the production paper deploy. Decisions on V13/V12/V12c are orthogonal.
3. **No further V13-family iteration motivated.** V13 (BEAR -> SPY) failed; V13-defensive (BEAR -> SH/TLT/GLD) requires universe expansion AND the same detector to fire on the right side of the trough; not higher-leverage than WS-3.
