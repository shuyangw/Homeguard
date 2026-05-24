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
- **Sensitivity findings (NOT gate-influencing)**: V12-up-cash (UNPREDICTABLE='cash') at Sharpe 0.586 beats V12 default and slightly beats V11. Motivates a future **V12c spec**.
- **Status**: research; deployment deferred pending WS-3 + V12 re-run.

## V12b / V12c -- reserved
- **V12b** candidate: V12 with `min_regime_days > 0`. V12 readiness sensitivity (2026-05-24) at 5 bps near_close: deb-2=0.130 (worse than V12 default's 0.268), deb-3=0.315 (+0.05), deb-5=0.437 (+0.17). V12-deb-5 modestly beats V12 default but still under-performs V11 (0.528). Combined with the detector-lag finding (mean -3.42 gap_days), the lesson is "no debouncing value can recover what the detector misses." NOT motivated as a separate spec; deferred until WS-3 (detector improvement) lands.
- **V12c** candidate (sensitivity-motivated): UNPREDICTABLE='cash' as the new default. Readiness sensitivity (2026-05-24) showed Sharpe 0.586 vs V12 default 0.268 (+0.32 lift) AND beats V11 (0.528). Spec honesty discipline says this requires its own readiness gate; not an in-place V12 default swap. **Strong candidate for the next research cycle after WS-3 or in parallel.**

## V13+ -- reserved
- **V13** candidate: defensive ticker support (`SH` / `TLT` / `GLD` as BEAR-day position) instead of cash. Universe expansion required. See spec Appendix C re: three-SMA structure constraint that defines V13a vs V14.
- **V14** candidate: per-regime strategy routing (RAMP for bull, OMR for sideways, etc.). Requires per-regime adapter layer; Phase 4 harness has no such abstraction.
