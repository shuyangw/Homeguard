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
- **Readiness report**: `docs/reports/ramp/20260523_phase4_v12_readiness.md`
- **Status**: research; readiness verdict pending.

## V12b / V12c -- reserved
- **V12b** candidate: V12 with `min_regime_days > 0` if the V12 readiness sensitivity appendix motivates.
- **V12c** candidate: V12 with `UNPREDICTABLE: 'cash'` if the V12 readiness sensitivity appendix motivates.
- Both spawned only if sensitivity shows >= 0.1 Sharpe lift + structural-gate retention; otherwise NOT separate variants.

## V13+ -- reserved
- **V13** candidate: defensive ticker support (`SH` / `TLT` / `GLD` as BEAR-day position) instead of cash. Universe expansion required. See spec Appendix C re: three-SMA structure constraint that defines V13a vs V14.
- **V14** candidate: per-regime strategy routing (RAMP for bull, OMR for sideways, etc.). Requires per-regime adapter layer; Phase 4 harness has no such abstraction.
