# RAMP Phase 4 Phase C Wave 1 — 2026-05-22

## Summary

Built four turnover-control variants (V04 rank buffer, V05 min hold, V06 delta threshold, V11 combined) on the Phase B harness and ran them against Alpaca SIP daily data 2017-01-01 to 2026-05-16 at four cost tiers. **V11 passes the methodology Section 4 cost-sensitivity gate that V01 fails**, with EXT-OOS 2025-26 Sharpe rescued from -0.216 to +0.527 and turnover cut from 91% to 39%. V11 is the Phase D paper-trade candidate.

## Changes Made

- **`src/research/ramp_phase4/config.py`**: added `delta_rebalance_pct: float = 0.0` to `HarnessConfig`.
- **`src/research/ramp_phase4/engine.py`**: added `position_open_dates` and `last_target_symbols` to `HarnessState`; threaded `current_date` through `apply_trades`; widened `compute_trades` to use `delta_rebalance_pct` with full-exit bypass; `run_variant` maintains the new state fields.
- **`src/research/ramp_phase4/filters.py`** (new): pure functions `rank_buffer` and `min_hold` with equal-weight renormalization.
- **`src/research/ramp_phase4/variants.py`**: added V04, V05, V06, V11 to REGISTRY. V04 and V11 use the full-universe momentum ranking (fix landed in `391ebea` after initial run showed V04 was a no-op).
- **`scripts/backtest_scripts/ramp_phase4_backtest.py`**: CLI auto-sets `cfg.delta_rebalance_pct = 0.02` for V06 and V11.
- **`docs/strategies/production/RAMP_STRATEGY.md`**: linked the Wave 1 findings report.
- **`docs/reports/ramp/`**: V04, V05, V06, V11 per-variant reports + Wave 1 findings cross-variant report.

## Commits

- `45ddf5a` HarnessState fields
- `913a692` apply_trades + current_date
- `ebd7b33` delta_rebalance_pct + widened compute_trades
- `7ab517b` last_target_symbols tracking
- `b66776b` rank_buffer filter
- `7980645` min_hold filter
- `f35943d` V04 rank-buffer variant
- `61812d2` V05 min-hold variant
- `f3dc600` V06 variant entry
- `c8eadf2` V11 combined turnover-lite
- `578dfa8` CLI delta_rebalance_pct map
- `391ebea` fix: V04/V11 full-universe momentum ranking
- `569df37` V04/V05/V06/V11 backtest reports
- `479b6f6` Wave 1 cross-variant findings

## Headline at 5 bps per side (full window)

| Variant | CAGR | Sharpe | Max DD | Turnover | EXT-OOS Sharpe |
|---|---:|---:|---:|---:|---:|
| V01 baseline | 3.74% | 0.282 | -79.88% | 91% | -0.216 |
| V04 | 4.89% | 0.313 | -78.87% | 82% | -0.099 |
| V05 | 11.08% | 0.503 | -67.22% | 45% | +0.556 |
| V06 | 3.62% | 0.278 | -79.57% | 90% | -0.215 |
| **V11** | **11.93%** | **0.528** | **-66.20%** | **39%** | **+0.527** |

## Known Issues / Remaining Work

- **Strict acceptance: V11 fails the 2022 OOS degradation gate** (Δ Sharpe -0.343 vs V01). Composite verdict still favors V11 because the EXT-OOS rescue is much larger than the 2022 cost.
- **V06 effectively a no-op at 2%.** A follow-up with `delta_rebalance_pct=0.05` could provide additional turnover reduction.
- **V12 (BEAR-to-cash on V11 base) is the natural Wave 2 next step** — V11's biggest weakness is 2022 OOS (a BEAR year), and BEAR-to-cash directly addresses that.
- **A7 paper-validation comparator was built for V01.** V11 has rank-buffer and min-hold filters that the comparator's `_recompute_plan` doesn't model. Phase D paper-trade of V11 requires extending the comparator.

## Validation

- All 28 `tests/research/ramp_phase4/` unit tests + 7 `test_variants.py` tests pass under the `fintech` env.
- V11 cost-sensitivity sweep: Sharpe 0.693 / 0.605 / 0.528 / 0.452 across 0 / 2.5 / 5 / 7.5 bps. **Passes 1.5x base cost gate.**
- V01 baseline numbers from Task 57 unchanged (sanity check: `delta_rebalance_pct=0.0` default preserves V01 behavior).
