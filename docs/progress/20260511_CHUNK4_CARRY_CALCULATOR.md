# Chunk 4: Carry Calculator -- 2026-05-11

## Summary

Built `CarryCalculator` with per-asset-class formulas for commodity, equity_index, fx, and bond carry signals. Front and second-month contracts identified by volume ranking from `futures_per_contract_1min/` (reuses spread-filtering helper from Chunk 3). Bond carry for Micro Yield futures (2YY/5YY/10Y/30Y) uses duration table + SOFR derivation; standard bond roots (ZT/ZF/ZN/ZB/TN/UB) return 0 as v1 fallback pending CTD conversion logic. Validated against real ES + GC 2024 data: 252+ trading days each with median carry magnitudes < 10%.

## Files Changed

- `src/data/carry_calculator.py` (new, 159 lines) -- `CarryCalculator` class with `_find_front_second_close`, `_months_between`, `compute`, `compute_history`; `_is_outright` helper; `_MONTH_CODES`, `DURATION_BY_ROOT`, `MICRO_YIELD_ROOTS` constants
- `tests/data/test_carry_calculator.py` (new, 6 synthetic tests)
- `tests/data/test_carry_calculator_integration.py` (new, 2 integration tests against real ES/GC)

## Commits

- `f497ea8` feat(data): CarryCalculator scaffold + _find_front_second_close helper
- `6ec4a30` feat(data): carry for commodity + equity_index + fx asset classes
- `83ed005` feat(data): bond carry with duration table + SOFR funding
- `38039e3` feat(data): compute_history + integration tests for CarryCalculator

## Design Notes

- **Front/second contract identification**: Top-2 outright contracts by daily volume on the target date. Spreads (symbols containing "-") filtered via `_is_outright` regex that matches `<root><month_code_letter><year_digits>`. This is a generalization of Chunk 3's `_active_contract_per_day` but returns both front and second in a single pass.
- **days_to_second**: Computed from contract month-code parsing. Each month is ~30 days, so ES front (M) to second (U) = 3 months = ~90 days. Approximation acceptable per master spec ("magnitudes match expectations" not "exact").
- **Asset class formulas**:
  - commodity: `(second - front) / front * (365 / days_to_second)`
  - equity_index: `(front - second) / second * (365 / days_to_second)` (sign flipped — equity carry is conventionally cost-of-carry from the holder's perspective)
  - fx: same shape as commodity (front close is the spot proxy)
  - bond: `duration * (front_yield - funding_rate) / 100` for Micro Yield only; 0.0 fallback for price-traded bond futures
- **Bond carry funding rate** comes from `derive_sofr(d)` in the validation framework's derivations layer (Chunk 1 merge). Pre-2018 GE/Eurodollar funding is not yet wired; would need to be added before bond carry backtests extend to that era.
- **Implementer-discovered bug fix** during Task 4.2: My initial `_months_between` spec mishandled same-year same-decade contracts with second month earlier than front (e.g., front ESM4, second ESH4). The implementer simplified to `s_year < f_year` triggering decade wrap, which is correct given the calling convention (caller uses `abs(months)` for days_to_second). Fix committed in 6ec4a30 as part of normal implementation.

## Validation

- 6 synthetic tests pass (`test_carry_calculator.py`): front/second identification with spreads filtered; commodity carry magnitude; equity_index carry magnitude; bond Micro Yield carry magnitude; bond standard fallback to 0.0; `compute_history` walks dates correctly.
- 2 integration tests pass (`test_carry_calculator_integration.py`) against real `H:/Stock_Data/futures_per_contract_1min/`: ES equity_index carry for 2024 has 252 trading days with median magnitude < 10%; GC commodity carry similarly. Integration test wallclock ~12 minutes (heavy I/O across the full year).
- Total Chunk 4 test count: **8 passed**.

## Known Issues / Remaining Work

- **Bond carry for ZT/ZF/ZN/ZB/TN/UB returns 0.0** — these are price-traded futures whose yields require cheapest-to-deliver conversion factor lookup. Master spec accepts this as a v1 limitation. Adaptation B's bond sleeve can use TSMOM only on these instruments for now, or use Micro Yield futures (2YY/5YY/10Y/30Y) which have direct yield quotes.
- **Pre-2018 funding rate** uses SOFR via `derive_sofr` which raises ValueError for dates before 2018-05-07. GE/Eurodollar fallback is not yet wired. Affects bond carry backtests for the 2010-2018 era. Per `docs/progress/20260509_VALIDATION_FRAMEWORK_DEFERMENTS.md`, this is acceptable for v1 — extend later if a backtest specifically needs pre-2018 bond carry.
- **`days_to_second` approximation** uses 30 days per month. For most quarterly/bi-monthly cycles this is within ~5% of actual days-to-expiration. Could refine using `futures_definitions/` expiration column if a strategy's backtest shows the approximation matters.
- **No FX integration test** — only commodity and equity_index are integration-tested. FX uses the same formula shape as commodity so the algorithm is exercised; per-currency calibration left to Adaptation B build.

## Decision Gate

PROCEED to Chunk 5 (`feature/roll-sizing-imbalance`) once the merge to main lands. Chunks 5 and 6 are parallel-safe with Chunk 4 (no shared files); we'll execute sequentially per the master spec.

## Reproduction Commands

```bash
cd C:/Users/qwqw1/Dropbox/cs/github/Homeguard
conda run -n fintech pytest tests/data/test_carry_calculator.py -v
# Expected: 6 passed

# Integration (~12 min wallclock):
conda run -n fintech pytest tests/data/test_carry_calculator_integration.py -v
# Expected: 2 passed
```
