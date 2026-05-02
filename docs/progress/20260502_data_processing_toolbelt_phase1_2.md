# Data Processing Toolbelt (Phase 1 + 2) -- 2026-05-02

## Summary
Established `src/features/` as the canonical home for stateless data
processing primitives. Shipped the full Tier 1 set (11 primitives:
log_transform, log_returns, zscore_rolling, robust_zscore_rolling,
robust_zscore_cross_sectional, winsorize, rank_transform, close_to_close_rv,
parkinson_rv, garman_klass_rv, yang_zhang_rv) with full test coverage and
refactored four existing call sites onto the new module.

## Changes Made
- **src/features/**: new package, 11 primitives across `normalizers.py` and
  `volatility.py`, flat public API at the package level. 100% line coverage.
- **tests/features/**: full property + edge-case + index-preservation test
  suite per primitive. Includes a contract test that guards `close_to_close_rv`
  against future refactor regressions for the three FRS/DSTS/EVR callers
  (which have no per-strategy regression tests).
- **src/strategies/advanced/frs_indicators.py:651-653**: swapped inline
  rolling-std-times-sqrt block for `close_to_close_rv`. Bit-equivalent.
- **src/strategies/advanced/dsts_indicators.py:484-485**: same swap;
  pandas defaults `min_periods=window` for integer windows so this is
  bit-equivalent despite DSTS's omission of the explicit kwarg.
- **src/strategies/advanced/evr_indicators.py:510-512**: same swap.
  Bit-equivalent.
- **src/strategies/advanced/ml_crypto_mr_indicators.py:110-127**: swapped
  inline `(x - mean) / (std + 1e-10)` for `zscore_rolling`. Behavior delta:
  zero-variance windows now produce NaN. Pre-flight audit of the two
  consumers (ml_crypto_mr_indicators.py:395 and hurst_mr_strategy.py:184)
  confirmed NaN safety; both assign to DataFrame columns. The hurst_mr
  strategy has an explicit `_check_indicator_nan_for_entry` guard that
  skips entry on NaN.

## Commits
- `bd9aa85` feat(features): create src/features package skeleton
- `1e93b04` feat(features): add log_transform primitive
- `a68514a` feat(features): add log_returns primitive
- `ea1f557` feat(features): add zscore_rolling primitive (sigma-based, legacy migration)
- `73ec56d` feat(features): add robust_zscore_rolling (MAD-based)
- `4f85311` feat(features): add robust_zscore_cross_sectional
- `69e6ebf` feat(features): add winsorize primitive
- `e815179` feat(features): add rank_transform primitive
- `1ec3a69` feat(features): add close_to_close_rv with Phase 2 contract test
- `f0ed8ba` feat(features): add parkinson_rv estimator
- `e0ca826` feat(features): add garman_klass_rv estimator
- `2fcf628` feat(features): add yang_zhang_rv estimator
- `15e1776` test(features): cover residual lines for 100% coverage
- `4cf07eb` refactor(frs): use src.features.close_to_close_rv for realized vol
- `f282d9b` refactor(dsts): use src.features.close_to_close_rv for realized vol
- `977451f` refactor(evr): use src.features.close_to_close_rv for realized vol
- `190cd18` refactor(ml_crypto_mr): use src.features.zscore_rolling for z-score

## Known Issues / Remaining Work
None for Phase 1+2. Phase 3 (RAMP cross-sectional z-score), Phase 4
(`TransformedDataProvider`), and Phase 5 (three-parallel Kalman) remain
pending and are tracked in the broader Data Processing Toolbelt plan.

The `zscore_rolling` primitive (sigma-based) is documented as
migration-only; replacing the ML Crypto MR consumer with
`robust_zscore_rolling` is a separate follow-up requiring a strategy
backtest.

## Validation
- `pytest tests/features/ -v` -- 76 tests pass.
- `pytest tests/features/ --cov=src/features` -- 100% line coverage on
  `__init__.py`, `normalizers.py`, `volatility.py`.
- All 11 primitives importable from top-level `src.features` package.
- Manual import smoke checks for each refactored module pass.
- `git diff --stat` shows 20 insertions(+), 10 deletions(-) LOC across the four
  refactored source files (net +10 LOC, primarily from imports and
  multi-line function calls vs compact inline logic; trade-off for DRY and
  maintainability is acceptable).

## Implementation Notes

The net-positive LOC is primarily from adding local imports and spreading
function calls across multiple lines for readability:
- FRS: +5,-2 (import + multi-line call replaces 2-line inline logic)
- DSTS: +4,-1 (import + multi-line call replaces 1-line compact expression)
- EVR: +5,-2 (import + multi-line call replaces 2-line inline logic)
- ml_crypto_mr: +6,-5 (import + 2-line call replaces 3-line inline logic, plus
  doc expansion about NaN behavior)

Trade-off rationale: canonical location in `src/features`, single point of
change for optimization, reduced duplication across four strategy indicator
files, improved maintainability and discoverability. Future optimizations
(Numba/Cython) to these primitives will benefit all four call sites
automatically.
