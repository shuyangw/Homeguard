# RAMP Variants Reference

Canonical glossary of every named RAMP variant. Each entry links to code, spec, readiness report, and production status.

## V01 -- baseline (fresh portfolio every rebalance)
- **Code**: `src/research/ramp_phase4/variants.py::_variant_v01`
- **Description**: Production REGIME_PARAMS; fresh portfolio every rebalance; ignores crash exposure.
- **Status**: research baseline.

## V03 -- V01 + planner-correct crash exposure
- **Code**: `src/research/ramp_phase4/variants.py::_variant_v03`
- **Description**: Same selection as V01 but honors planner's `exposure_pct` (1.0 normally, 0.5 in crash regimes).
- **Spec**: `docs/superpowers/specs/2026-05-19-ramp-phase4-phaseB-harness.md`
- **Report**: `docs/reports/ramp/20260522_phase4_v01_vs_v03_parity.md` (V03 worse than V01 net; turnover-control needed before V03 viable)
- **Status**: archived; V03's crash-halving cuts gross more than it cuts turnover-cost.

## V04 -- V01 + rank_buffer
- **Code**: `src/research/ramp_phase4/variants.py::_variant_v04`
- **Description**: Keeps currently-held names within `top_n + (top_n // 2)` rank buffer.
- **Status**: research; subsumed by V11.

## V05 -- V01 + min_hold
- **Code**: `src/research/ramp_phase4/variants.py::_variant_v05`
- **Description**: Protects positions younger than 5 trading days from forced exit.
- **Status**: research; subsumed by V11.

## V06 -- V01 + delta_rebalance_pct threshold
- **Code**: `src/research/ramp_phase4/variants.py::_variant_v06` (uses `_variant_v01` plan_fn + `cfg.delta_rebalance_pct=0.02`)
- **Description**: Skips trades smaller than 2% of NAV; full exits bypass the floor.
- **Status**: research; subsumed by V11.

## V11 -- combined turnover-lite
- **Code**: `src/research/ramp_phase4/variants.py::_variant_v11`
- **Description**: V01 base + rank_buffer (5-name buffer for top_n=10) + min_hold (5 trading days) + delta_threshold (2% via cfg).
- **Spec**: `docs/superpowers/specs/2026-05-22-ramp-phase4-phaseC-wave1-design.md`
- **Plan**: `docs/superpowers/plans/2026-05-22-ramp-phase4-phaseC-wave1.md`
- **Readiness report**: `docs/reports/ramp/20260523_phase4_v11_readiness.md` (PARTIAL: passes PBO + lag-robustness, fails PSR + DSR; deployed to production paper)
- **Status**: production paper (since 2026-05-23); Phase D paper validation in progress.

## V12 -- V11 + per-regime position override
- **Code**: `src/research/ramp_phase4/variants.py::_variant_v12`
- **Description**: V11 base. On detector-BEAR days, returns cash (no equity exposure). On UNPREDICTABLE/SIDEWAYS days, defers to V11. Optional symmetric debouncing via `cfg.min_regime_days` (default 0, off).
- **Spec**: `docs/superpowers/specs/2026-05-23-v12-bear-to-cash-design.md` (rev4 + rev4-followup)
- **Plan**: `docs/superpowers/plans/2026-05-23-v12-bear-to-cash.md`
- **Readiness report**: `docs/reports/ramp/20260524_phase4_v12_readiness.md`
- **Readiness verdict (2026-05-24)**: **Tier 3 (diagnostic value)**. Structural gates PASS (PBO 0.39, lag-degradation, cost robustness). PSR (0.79) and DSR (0.54) FAIL on absolute significance. V12 near_close at 5 bps Sharpe = 0.268, materially below V11's 0.528 -- BEAR-to-cash hurts in near_close, helps in lag (V12 lag at 5 bps = 0.665, beats V11's 0.580). Detector-onset alignment shows mean -3.42 gap days between BEAR onset and SPY trough.
- **Decision**: V12 NOT deployed to production paper. Per spec Tier 3 rule, escalate to WS-3 (regime detector improvement) as the higher-leverage path.
- **Sensitivity findings (NOT gate-influencing)**: V12-up-cash (UNPREDICTABLE='cash') at Sharpe 0.586 beats V12 default and slightly beats V11. Motivated the **V12c readiness gate (Experiment 6, 2026-05-24)**; see V12c section below for the formal verdict.
- **Status**: research; deployment deferred pending WS-3 + V12 re-run.

## V12c -- V12 + UNPREDICTABLE also to cash

- **Code**: `src/research/ramp_phase4/variants.py::_variant_v12` (same code as V12; differs only in `cfg.regime_positions[UNPREDICTABLE]='cash'`)
- **Description**: V11 base. On detector-BEAR AND detector-UNPREDICTABLE days, returns cash. SIDEWAYS/STRONG_BULL/WEAK_BULL: V11 logic.
- **Discovery context**: V12 readiness sensitivity (2026-05-24) showed V12-up-cash Sharpe 0.586 vs V12 default 0.268 (+0.32). Pre-readiness analysis: AMBIGUOUS (53.6% of attribution in top-3 events; COVID-dominant per E2 hand-inspection).
- **Readiness report**: `docs/reports/ramp/20260526_phase4_v12c_readiness.md`
- **Readiness verdict (2026-05-24)**: **TIER 4 -- BLOCKED**. PBO across the 7-variant set {V01,V04,V05,V06,V11,V12,V12c} = 0.7085 (>= 0.50 threshold, elevated overfitting risk) is the binding structural failure. PSR (0.9645) just clears the 0.95 floor, but DSR (0.8337, n_trials=23) fails. Gate 4 (lag-degradation) and Gate 5 (cost floor + V11 no-regress) both PASS. V12c Sharpe @ 5 bps near_close = 0.586; @ 5 bps one_day_lag = 0.850; @ 7.5 bps one_day_lag = 0.776 (>= 0.9 * V11 = 0.478).
- **COVID-excluded subgroup**: Sharpe(V12c @ 5 bps near_close) = 0.5863 (full) vs 0.5714 (ex-COVID 2020-02-24 .. 2020-04-30) -- delta -0.0149 (2.5% magnitude). V12c's measured edge is NOT concentrated in COVID; the COVID-event story from E2 was about UNPREDICTABLE attribution specifically, but V12c's gross alpha is broadly distributed across the 2017-2026 window.
- **Honesty discipline**: V12c was discovered from V12's sensitivity grid; DSR n_trials_project=23 (V12 used 22, V12c is trial #23). E2 hand-inspection returned AMBIGUOUS, COVID-event-dominant for UNPREDICTABLE attribution; this report's COVID-excluded panel shows the OVERALL V12c edge is not COVID-fragile, but the PBO failure remains the binding gate. Detector-onset alignment: 59 BEAR/UNPREDICTABLE onsets, mean gap_days -4.05 -- the detector fires ~4 trading days AFTER the SPY drawdown trough on average; V12c is still cashing around the recovery, not the crash.
- **Status**: research; **deployment BLOCKED** (PBO structural fail). Recommend WS-3 (regime detector improvement) before any further V12-family iteration -- the detector-lag tax is the binding constraint underneath both V12 and V12c.

## V12b -- reserved

- **V12b** candidate: V12 with `min_regime_days > 0`. V12 readiness sensitivity (2026-05-24) at 5 bps near_close: deb-2=0.130 (worse than V12 default's 0.268), deb-3=0.315 (+0.05), deb-5=0.437 (+0.17). V12-deb-5 modestly beats V12 default but still under-performs V11 (0.528). Combined with the detector-lag finding (mean -3.42 gap_days), the lesson is "no debouncing value can recover what the detector misses." NOT motivated as a separate spec; deferred until WS-3 (detector improvement) lands.

## V13-bear-invert -- V11 + BEAR onset goes to SPY 100%

- **Code**: `src/research/ramp_phase4/variants.py::_variant_v13_bear_invert`
- **Description**: V11 base. On detector-BEAR days, returns 100% SPY (single-name). Tests the BEAR-as-buy hypothesis discovered from V12's onset-alignment panel (mean gap_days = -3.42).
- **Discovery context**: V12 readiness panel showed detector fires ~3.4 trading days AFTER the SPY drawdown trough across 59 events 2017-2026. V13 inverts the sign of V12's BEAR consumption.
- **Readiness report**: `docs/reports/ramp/20260525_phase4_v13_readiness.md`
- **Readiness verdict (2026-05-24)**: **TIER 4 -- BEAR-as-buy is spurious on this sample.** V13 @ 5bps near_close Sharpe = 0.400 vs V11's 0.528 (-0.128); V13 @ 5bps one_day_lag = 0.381 vs V11's 0.580 (-0.199). Gate 3 (PBO=0.629 across 7 variants) FAIL, Gate 5b (no-regress vs V11 @ 7.5bps lag: V13=0.308 vs 0.9*V11=0.478) FAIL, Gate 1 PSR (0.883) FAIL, Gate 2 DSR (0.707, n=23) FAIL. Gate 4 (lag-degradation, nc-lag=+0.019) and Gate 5a (cost-floor, 0.308>0.30) PASS. Detector-onset alignment: 59 BEAR onsets, mean gap_days -3.42 (matches V12 panel), mean SPY return during BEAR window = +0.18% -- positive but tiny, too small to overcome turnover cost relative to V11 holding momentum names that recover into the trough.
- **Honesty discipline**: V13 was discovered from EXT-OOS inspection of V12's panel. DSR n_trials_project incremented (22 -> 23) to reflect V13's introduction. **NOT OOS in the strict sense**; forward OOS validation required before any deploy regardless of verdict.
- **Status**: research; **NOT deployed**. Continue WS-3c roadmap (E3 produced WS-3c verdict; V13 spurious confirms BEAR-as-buy at single-asset SPY is not the lever).

## V14+ -- reserved
- **V14** candidate: per-regime strategy routing (RAMP for bull, OMR for sideways, etc.). Requires per-regime adapter layer; Phase 4 harness has no such abstraction.
