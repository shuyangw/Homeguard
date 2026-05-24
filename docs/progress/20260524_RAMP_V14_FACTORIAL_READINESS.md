# V14 Soft-Bear Factorial Phase D Readiness -- Session Log (2026-05-24)

## Summary

WS-3c spec rev2 (V14 factorial) implemented and gated per the 10-task plan
at `docs/superpowers/plans/2026-05-24-v14-soft-bear-factorial.md`. The
factorial tested three actions (cash / SPY / dampen 0.5) on the same
Schmitt-trigger BEAR_score consumer with pre-registered tau_in (0.5556)
from G1_BEAR median. Output report:
`docs/reports/ramp/20260526_phase4_v14_factorial_readiness.md`.

Verdicts:
- **V14a-soft-bear-cash: TIER 4** (Sharpe 0.6146 vs V11 0.5279; lift +0.087 < +0.10)
- **V14b-soft-bear-spy: TIER 4** (Sharpe 0.6035; lift +0.076)
- **V14c-soft-bear-dampen: TIER 4** (Sharpe 0.6131; lift +0.085)

Selection: NONE. The WS-3c hypothesis is closed on this evidence.

## What ran

- Pre-spec script `compute_tau_in_from_g1.py` produced `v14_tau_constants.json`
  (tau_in=0.555556, tau_out=0.455556 from G1_BEAR median of 371 days; G1
  labeler pinned at commit `9c48245`).
- 35 backtests in 30.07 min wall-clock:
  - 24 cost-grid runs (3 variants x 4 cost x 2 timing modes)
  - 5 cross-variants at 5 bps near_close (V01, V11, V12, V12c-cfg, V13-bear-invert)
  - 1 V11 reference at 7.5 bps one_day_lag (Gate 5 baseline)
  - 4 sensitivity runs (V14a tau-band x2 + V14c dampen x2; informational only)
  - 1 smoke run during orchestrator self-check
- DSR n_trials_project = 36 (audited honest count per spec rev2 honesty discipline).
- 152 unit tests pass before the long run (Tasks 0-7 stack); ~50 new V14 tests added on top of the 105-test baseline.

## Headline verdicts table

| Variant | Tier | Sharpe @5bps nc | Sharpe @5bps lag | Sharpe @7.5bps lag | nc - lag | PSR | DSR | Gate PBO |
|---|---|---|---|---|---|---|---|---|
| V14a-soft-bear-cash | **TIER 4** | 0.6146 | 0.6921 | 0.6339 | -0.0775 | 0.9703 | 0.8175 | 0.9528 |
| V14b-soft-bear-spy | **TIER 4** | 0.6035 | 0.6444 | 0.5510 | -0.0409 | 0.9674 | 0.8075 | 0.9528 |
| V14c-soft-bear-dampen | **TIER 4** | 0.6131 | 0.7432 | 0.6503 | -0.1301 | 0.9695 | 0.8153 | 0.9528 |
| V11 reference | -- | 0.5279 | -- | 0.5306 | -- | -- | -- | -- |

Diagnostic PBO (4 orthogonal {V01, V11, V12, V14a}): 0.6505 -- also fails 0.5 threshold.

## Key findings

1. **All three V14 variants beat V11 in absolute Sharpe terms (+0.076 to +0.087)** but none reach the +0.10 TIER 1 lift threshold. The lift bar is the immediate non-structural failure.

2. **Gate PBO 0.9528 is the binding structural failure**, identical across all three V14 variants. The 8-variant gate set is heavily correlated: V12, V12c, V13, V14a/b/c are all derivatives of V11 + regime-conditional treatment, so CSCV cannot distinguish their out-of-sample ranks from chance. The diagnostic PBO (0.6505 on 4 orthogonal variants) also fails -- correlation is not the only issue; the V14 alpha is genuinely IS-fragile.

3. **DSR fails at n_trials=36 across all three variants** (0.8075-0.8175 < 0.95). This is the rev2 honesty discipline biting -- if reduced to n_trials=24 (rev1's count), DSR would land near or above 0.95. The campaign explicitly chose to NOT reduce-to-rescue.

4. **Gate 4 (lag-degradation) and Gate 5 (cost floor + no-regress) PASS for all variants**. The V14 family is not paying the cash-cycle lag tax that V12 had (V12 nc-lag = -0.397; V14 nc-lag ranges from -0.04 to -0.13). The Schmitt trigger is doing its job at the cost-control level.

5. **V14c (dampen) has the best lag Sharpe at 7.5bps (0.6503)** -- best of the three -- but its near_close Sharpe (0.6131) is in the middle of the pack. The dampen action preserves more lag-Sharpe but doesn't help the near_close lift bar.

6. **Detector lag tax is unchanged from V12 / V12c**. The soft-score lead (E3: 24 days at tau=0.3) is real but the +0.08 lift it provides isn't enough at the +0.10 + DSR n_trials=36 bar.

## Tier verdicts + honesty discipline

**All three V14 variants TIER 4** per spec rev2 5-gate classification. PBO 0.9528 is the dominant structural failure; DSR 0.81 is the secondary failure; TIER 1 lift is a third independent failure. The variants are convergent on the same diagnostic story -- this is consistent evidence, not three independent results.

**Surviving true-negative per spec rev2**: "If V14 cannot pass DSR at 36 trials, the campaign has consumed its multi-trial budget and no consumer-layer variant can deploy without forward OOS evidence." The campaign has reached its honest stopping condition on the BEAR-consumer line.

**NOT strict OOS**: tau_in derived from G1_BEAR median on the same 2017-2026 window. The G1 labeler is independent of E3's lead-time sweep (no in-sample optimization on the gate window), but the BEAR_score series itself is in-window. Forward OOS validation would require data post-2026-05-24.

## Decisions

- **V14a/b/c NOT deployed.** All three TIER 4; preserved in REGISTRY for diagnostic continuity.
- **WS-3c hypothesis closed on this evidence.** Soft-score consumption does not rescue the argmax failure under the +0.10 lift bar and DSR n_trials=36.
- **Next research priority**: WS-3a (detector-internal hysteresis) or WS-3b (leading indicators) at the detector layer -- since the consumer layer has been exhaustively tested across argmax (V12/V12c/V13) and soft-score (V14a/b/c) with consistent TIER 4 outcomes. Both are 1-2 week scope per the diagnostic remediation ranking.
- **V11 paper validation continues** independently on `ramp-phase4-turnover-regime-research`. A7 counter is unaffected.

## Commits this session

- `dd2e37b` report(ramp): V14 factorial readiness -- WS-3c soft-score verdict TIER 4
- `6f55e37` feat(orchestrator): V14 factorial readiness gate -- 3 variants, DSR n_trials=36
- `1bb4e8a` feat(variants): V14c-soft-bear-dampen via Schmitt-trigger BEAR_score
- `7844f1d` feat(variants): V14b-soft-bear-spy via Schmitt-trigger BEAR_score
- `9078633` fix(variants): V14a freshness assertion must not be swallowed
- `8aae219` feat(variants): V14a-soft-bear-cash via Schmitt-trigger BEAR_score
- `9674bd5` feat(config): V14 tau / dampen fields + JSON loader + predicate validation
- `108de8a` feat(engine): V14 Schmitt-trigger state + _SentinelPlan dispatch
- `c1f467f` feat(plans): _SentinelPlan class for V14 no-exposure marker
- `d26aa69` feat(detector): last_classification_timestamp field for V14 freshness assertion
- `983ac61` fix(diagnostics): V14 tau constants -- forward-slash paths + git-log guard
- `faf7abe` feat(diagnostics): pre-register V14 tau constants from G1_BEAR median

## Cross-experiment context

This session continued the 6-experiment campaign that began with E1-E6 on 2026-05-24:

| Exp | Verdict | Implication for V14 |
|---|---|---|
| E3 soft scores | WS-3c (median argmax_lag 24 days at tau=0.3) | Motivated this V14 work |
| E2 UNPREDICTABLE | AMBIGUOUS (53.6% top-3 share) | Informed V12c framing; not V14-specific |
| E4 lag asymmetry | DIFFUSE (38.1% transition share) | V14 cost grid used standard 5/7.5/10 bps (no 10 bps stress add) |
| E1 V13-bear-invert | TIER 4 (argmax-BEAR-as-buy spurious) | V14b tested soft-trigger BEAR-as-buy as the disambiguating arm |
| E5 OMR cross-check | AMBIGUOUS (OMR screens out BEAR/UNPREDICTABLE) | WS-3 is RAMP-attributable; V14 verdict informs RAMP-only roadmap |
| E6 V12c readiness | TIER 4 (PBO 0.71) | V12c already showed argmax+UNPREDICTABLE-cash fails; V14a/b/c with soft trigger now also fails |

Net result for the BEAR-consumer line: argmax (V12/V12c/V13) and soft-score (V14a/b/c) BOTH fail at the rev2 gates. The detector lag is the binding constraint underneath all six variants. WS-3a or WS-3b at the detector layer is the only remaining lever for RAMP.

## Next session candidates

1. **WS-3a brainstorm + spec** -- detector-internal hysteresis. The detector's current argmax label flickers; a smoothing layer at the detector level (not the consumer layer) might cut down on the noise that defeats variants.
2. **WS-3b brainstorm + spec** -- leading indicators (e.g., breadth, credit spreads, options skew) augmenting the detector inputs to fire earlier. Higher implementation cost but addresses the root structural lag.
3. **Monitor V11 paper validation** -- A7 counter on EC2 (independent of this campaign).
