# V12c Phase D Readiness -- Session Log (2026-05-24, Experiment 6)

## Summary

V12c (V12 BEAR-to-cash + UNPREDICTABLE-to-cash) was formalized through its own 5-gate readiness orchestrator after surfacing as the strongest V12-readiness sensitivity (V12-up-cash, Sharpe 0.586 vs V12 default 0.268). Verdict: **TIER 4 -- BLOCKED**. The PBO across the 7-variant set {V01,V04,V05,V06,V11,V12,V12c} is 0.7085 (>= 0.50 = elevated overfitting risk), the binding structural failure. PSR clears the 0.95 floor (0.9645) but DSR (0.8337, n_trials=23) fails. Gates 4 and 5 PASS. Importantly, the COVID-excluded panel (E2-required robustness check) shows the V12c edge is NOT concentrated in COVID -- Sharpe drops only 2.5% (0.586 -> 0.571) when 2020-02-24 .. 2020-04-30 is removed. So the gate failure is structural-stability driven, not COVID-fragility driven. V12c not advanced; WS-3 (detector improvement) remains the higher-leverage path per V12's prior Tier 3 escalation.

## What ran

- 15 unique backtests in `scripts/backtest_scripts/ramp_phase4_v12c_readiness.py`:
  - 8 gate-influencing V12c cost grid: {1, 5, 7.5, 10} bps x {near_close, one_day_lag}
  - 6 NEW cross-variants at 5 bps near_close: V01, V04, V05, V06, V11, V12 (default BEAR-only cash). V12c at 5 bps near_close was reused from the cost grid -- total 7 variants in the PBO set.
  - 1 V11 reference at 7.5 bps one_day_lag (Gate 5 no-regress baseline)
- 1 post-hoc panel: COVID-excluded Sharpe (filter on the 5 bps near_close record stream; not a fresh backtest).
- Output: `docs/reports/ramp/20260526_phase4_v12c_readiness.md`
- Wall-clock: 13.74 min
- n_trials_project: **23 hard-coded** (V12 used 22; V12c is trial #23; V12-up-cash sensitivity now formalized as a distinct trial)

## Headline verdict

| Gate | Result | Value | Threshold |
|---|:---:|---:|---:|
| 1. PSR(V12c @ 5bps near_close, vs SR=0) | PASS | 0.9645 | > 0.95 |
| 2. DSR(V12c, n_trials=23) | FAIL | 0.8337 | > 0.95 |
| 3. PBO across 7 variants | **FAIL** | **0.7085** | < 0.5 |
| 4. Lag-degradation (5 bps) | PASS | nc=0.586, lag=0.850, nc-lag=-0.263 | <= max(0.2*\|nc\|, 0.1) = 0.117 |
| 5a. Cost floor | PASS | 0.7762 | > 0.30 |
| 5b. No-regress vs V11 @ 7.5bps lag | PASS | V12c=0.7762, 0.9*V11=0.4776 | >= 0.9 * V11 |

**Overall**: TIER 4 (PBO structural fail).

## Key findings

1. **PBO is the binding gate.** Adding V12c to the 6-variant PBO set used in V12's readiness (PBO 0.39) pushed PBO to 0.7085 -- a ~80% jump. V12c is correlated with V11/V12 in regime-driven ways that CSCV picks up as overfitting evidence: 7 variants all derive from the same V01 momentum base, all use the same regime detector, and V12c amplifies V12's structure rather than adding a genuinely independent signal. The PBO methodology is doing exactly what it should -- penalizing the family-resemblance.

2. **V12c is dramatically better than V12 default on both timing modes.** At 5 bps:
   - near_close: V12c 0.586 vs V12 0.268 (+0.32)
   - one_day_lag: V12c 0.850 vs V12 0.665 (+0.19, approx -- V12 lag from 2026-05-24 V12 report)
   At 7.5 bps one_day_lag: V12c 0.776 vs V12 ~0.608 (from V12 report). The UNPREDICTABLE-also-to-cash overlay materially improves both modes, AND beats V11 (0.528 near_close, 0.580 lag) at the same cost tier. The Sharpe is real; the structural-stability gate is the blocker.

3. **V12c beats V11 in absolute terms.** V12c near_close 0.586 > V11 0.528 (+0.058). V12c lag 0.850 > V11 lag 0.580 (+0.27). If PBO were the only gate failure, V12c could potentially deploy as a V11-superior alternative; but PBO 0.70 is too high to ignore on a single-pass backtest.

4. **COVID is NOT the alpha source.** The E2-required robustness check shows ex-COVID Sharpe = 0.5714 vs full 0.5863 -- delta -0.0149, only 2.5% of the full-window magnitude. The E2 verdict (AMBIGUOUS, 53.6% attribution in top-3 events) was about drawdown-avoidance attribution specifically; V12c's overall edge is broadly distributed across 2017-2026. This is a genuinely useful finding -- it would have been worse to have found ex-COVID Sharpe near zero.

5. **Detector lag tax is unchanged.** 59 BEAR/UNPREDICTABLE onsets in the window, mean gap_days -4.05 (UNPREDICTABLE-included broadens the V12 -3.42 baseline by one onset count). The detector still fires AFTER the SPY trough on average; V12c is cashing the recovery, not the crash. This is the structural reason no V12-family variant clears DSR even when Sharpe is decent: the detector signal is too noisy in time, and V12c is just adding another defensive trigger from the same noisy detector.

6. **Cost-grid monotonicity is preserved.** V12c degrades smoothly with cost (5 bps near_close 0.586 -> 10 bps 0.346; 5 bps lag 0.850 -> 10 bps 0.702). No cost-grid pathology.

## Tier verdict + honesty discipline

**TIER 4** per spec rev4 5-gate readiness classification: any structural gate failure (PBO, lag-degradation, or cost regression) blocks advancement. PBO = 0.7085 >= 0.50 is the failure.

Honesty discipline:
- V12c was discovered from V12's sensitivity grid (not a fresh hypothesis), incrementing n_trials_project from 22 to 23 -- properly recorded in the report and the DSR computation.
- E2 hand-inspection of UNPREDICTABLE attribution returned AMBIGUOUS (COVID-event dominant) BEFORE this gate ran. The conditional-proceed protocol required a COVID-excluded subgroup panel. That panel was added (informational only, gate stands on full-window numbers per spec rev4). The honesty discipline finding is **better than feared**: V12c is NOT COVID-fragile, but it IS structurally unstable per PBO.
- E4 lag-asymmetry returned DIFFUSE (38.1% transition-day share, below 50%); the prescribed 10 bps stress add did NOT apply. Standard cost grid was used.

The COVID-fragility-flag in the report's sensitivity panel reads: "_Sharpe shift under COVID exclusion is small (-0.0149, 2.5% of full-window magnitude). V12c edge is not concentrated in the COVID event._" This is the auto-generated language; the actual blocker is elsewhere (PBO).

## Decisions

- **V12c NOT deployed.** TIER 4 per spec; production paper deploy blocked.
- **WS-3 (regime detector improvement) remains the priority.** V12c's failure mode is structural-stability (PBO), and the underlying cause is the detector-lag tax that bedevils every V12-family variant. WS-3 attacks the root cause; V12c attacks a symptom.
- **V12b (debouncing) still NOT recommended.** V12 readiness sensitivity showed deb-{2,3,5} below V11 baseline; V12c's PBO failure suggests adding another regime-overlay variable would only deepen the family-overfitting signal.
- **V11 paper validation continues** on `ramp-phase4-turnover-regime-research`. V12c outcome does not change V11's path.
- **NEXT IF V12 family is revisited**: combine V12c's UNPREDICTABLE-cash overlay with a detector improvement (WS-3 output), not in isolation. The Sharpe-significance is there (PSR 0.96) -- it's the multiple-trials penalty and the variant-family clustering that's killing it.

## Commits this session

- `<TBD>` feat(orchestrator): V12c readiness gate -- UNPREDICTABLE+BEAR to cash, DSR n_trials=23 (experiment 6)
- `<TBD>` report(ramp): V12c readiness verdict -- formal gate after V12 sensitivity (experiment 6)
- `<TBD>` docs(progress): V12c readiness session log -- experiment 6 outcome

## Branch state

Local `v12-bear-to-cash` at head after this session's commits. Local only; not pushed.

## Next session candidates (priority order)

1. **WS-3 brainstorm** -- regime detector improvement spec. Highest leverage per the consistent V12-family detector-lag finding (mean gap_days -3.4 to -4.0 across V12 and V12c). The Sharpe is real (V12c near_close 0.586, lag 0.850); the detector noise is killing the multiple-trials gates.
2. **V11 paper validation monitoring** -- A7 counter on EC2; continues regardless of V12c outcome.
3. **V12c re-run AFTER WS-3 lands** -- if the detector improvement reduces the family-overlap signature, the V12c PBO failure may resolve. Not before.
4. **V13 brainstorm** -- defensive ticker support (SH/TLT/GLD on BEAR days) instead of cash. Universe expansion required. Could be a parallel research thread to WS-3.
