# Chunk 5: Roll Manager + Position Sizer + Imbalance Proxy -- 2026-05-11

## Summary

Three independent Phase 1 components bundled into one chunk per master spec section 4: `FuturesRollManager` (thin wrapper over Chunk 3's roll detection), `FuturesPositionSizer` (contract-based vol-targeted sizing with hard caps), and `estimate_signed_volume_from_bars` (tick-rule signed volume from 1-minute bars, Adaptation A's imbalance proxy). All three use synthetic-fixture tests only -- they are mathematically simple enough that integration testing against real data adds little additional confidence beyond the unit tests.

## Files Changed

- `src/data/roll_detector.py` (new, 49 lines) -- `FuturesRollManager` + `RollEvent` dataclass
- `src/backtesting/utils/position_sizer_futures.py` (new, 36 lines) -- `FuturesPositionSizer` + `ContractSpec` dataclass
- `src/data/signed_volume_estimator.py` (new, 39 lines) -- `estimate_signed_volume_from_bars(symbol, date)`
- `tests/data/test_roll_detector.py` (new, 2 tests)
- `tests/data/test_signed_volume_estimator.py` (new, 3 tests)
- `tests/backtesting/utils/__init__.py` (new, empty package marker)
- `tests/backtesting/utils/test_position_sizer_futures.py` (new, 4 tests)

## Commits

- `00a847f` feat(data): FuturesRollManager.get_active_contract + RollEvent dataclass
- `647776f` feat(backtesting): FuturesPositionSizer with hard contract caps
- `725b1c0` feat(data): tick-rule signed volume estimator (Adaptation A imbalance proxy)

## Design Notes

- **FuturesRollManager**: `get_active_contract(root, d)` delegates to `ContinuousContractDataLoader._active_contract_per_day` (same volume-ranked outright logic as Chunk 3 and Chunk 4). `get_upcoming_rolls()` is a v1 stub that returns `[]` -- accurate upcoming-roll prediction requires expiration date lookup (from `futures_definitions/`) plus rule-based timing (volume crossover heuristic or fixed-day-before-expiration). Documented as v1 limitation.
- **FuturesPositionSizer**: Plain formula `n = vol_target_dollars / (multiplier * underlying * current_vol)`, clamped to `[0, max_contracts]`. `ContractSpec` is a frozen dataclass holding static contract metadata (multiplier, tick_size, tick_value, max_contracts). Sign of position is the caller's responsibility -- sizer returns magnitude only.
- **estimate_signed_volume_from_bars**: Polars expression chain: `tick_sign = sign(close - close.shift(1)).fill_null(0).cast(Int8)`, then `signed_volume = volume.cast(Int64) * tick_sign`. First bar of the day has no prior so its `tick_sign` is 0 and `signed_volume` is 0. Returned DataFrame retains all original columns plus the two new ones, so callers can do their own aggregation (5-min buckets, daily totals, etc.).

## Validation

- 9 unit tests pass across the 3 modules:
  - `test_roll_detector.py`: 2 tests (active contract correct, upcoming rolls returns empty)
  - `test_position_sizer_futures.py`: 4 tests (basic math, max cap, zero-vol guard, never-negative)
  - `test_signed_volume_estimator.py`: 3 tests (tick rule signs, missing data returns empty, volume conservation)
- All tests use synthetic fixtures (no real-data dependency).

## Known Issues / Remaining Work

- **`get_upcoming_rolls` is a v1 stub returning empty list**. Live trading code that depends on upcoming-roll alerts needs a real implementation: either (a) read expiration from `futures_definitions/` and apply a fixed-day-before-expiration rule, or (b) extrapolate volume crossover trend from the last 30 days. Acceptable to defer because no current live strategy uses this method.
- **No integration test on real data**. Synthetic fixtures cover the algorithms fully; real-data validation here would mostly re-test infrastructure already covered by Chunks 3-4 integration tests.
- **`signed_volume` only handles single-day reads**. For multi-day analyses, the caller iterates dates. A multi-day batch variant could be added if a strategy's backtest shows the per-day Python overhead matters.

## Decision Gate

PROCEED to Chunk 6 (`feature/ibkr-futures-broker`) once the merge to main lands. Chunks 5 and 6 are parallel-safe but we'll execute sequentially.

## Reproduction Commands

```bash
cd C:/Users/qwqw1/Dropbox/cs/github/Homeguard
conda run -n fintech pytest tests/data/test_roll_detector.py tests/data/test_signed_volume_estimator.py tests/backtesting/utils/test_position_sizer_futures.py -v
# Expected: 9 passed
```
