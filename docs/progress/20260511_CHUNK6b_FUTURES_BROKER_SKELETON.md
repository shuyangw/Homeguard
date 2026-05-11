# Chunk 6b: FuturesTradingInterface + IBKRFuturesBroker Skeleton -- 2026-05-11

## Summary

Second sub-chunk of futures broker safeguards (per `docs/superpowers/specs/2026-05-11-futures-broker-safeguards-design.md`). Adds the `FuturesTradingInterface` ABC and the `IBKRFuturesBroker` skeleton class. All 8 abstract methods (plus 4 inherited from `OrderManagementInterface`) exist with correct signatures; bodies raise `NotImplementedError` pending sub-chunks 6c-6j which fill in the safeguards and IBKR API integration. The broker contract test catches missing-method regressions at commit time -- 10 new parametrized tests over `REQUIRED_FUTURES_METHODS`.

## Files Changed

- `src/trading/brokers/interfaces/futures_trading.py` (new, ~120 lines) -- `FuturesTradingInterface(OrderManagementInterface)` with 8 abstract methods: place_futures_order, place_futures_combo_order, get_futures_positions, get_futures_position, close_futures_position, close_all_futures_positions, what_if_order, get_margin_status
- `src/trading/brokers/ibkr/ibkr_futures_broker.py` (new, ~100 lines) -- `IBKRFuturesBroker(FuturesTradingInterface)` skeleton; all methods raise NotImplementedError with a comment naming the sub-chunk that will fill them in
- `src/trading/brokers/ibkr/__init__.py` (modified) -- exports IBKRFuturesBroker
- `tests/trading/brokers/interfaces/__init__.py` (new, empty)
- `tests/trading/brokers/interfaces/test_futures_trading.py` (new, 6 tests on ABC behavior)
- `tests/trading/brokers/ibkr/test_ibkr_futures_broker_skeleton.py` (new, 4 tests on skeleton implementation)
- `tests/trading/brokers/test_broker_contract.py` (modified) -- adds `REQUIRED_FUTURES_METHODS` + parametrized test for `IBKRFuturesBroker`

## Commits

- `51a3ff2` feat(trading): FuturesTradingInterface ABC for IBKRFuturesBroker
- `f39bb1d` feat(trading): IBKRFuturesBroker skeleton with NotImplementedError stubs
- `49bcf13` test(trading): broker contract test parametrized over IBKRFuturesBroker

## Design Notes

- **Interface is parallel, not extending**: `FuturesTradingInterface` extends `OrderManagementInterface` (gets cancel_order, get_order, get_orders, get_open_orders for free) but is NOT a subclass of `StockTradingInterface` or `OptionsTradingInterface`. Per safeguards spec section 1, futures semantics (contract months, SPAN margin, physical vs cash settlement) differ enough that mixing forces every implementor to handle all asset classes. Keep concerns separate.
- **Skeleton raises NotImplementedError on every method**: each error message names the downstream sub-chunk that fills it in (e.g. `"wired in sub-chunk 6f (margin guard)"`). This makes incremental development traceable and prevents accidental use of an unwired method.
- **Constructor stashes `IBKRConfig`** but does NOT start a connection. Subsequent sub-chunks (likely 6j integration) wire up the connection lifecycle. This keeps unit tests possible without a real IBKR gateway.
- **Broker contract test scope**: `REQUIRED_FUTURES_METHODS` includes the 8 futures-specific methods + `cancel_order` + `get_order` (inherited from OrderManagementInterface). Stock methods are NOT in this list -- a futures-only broker shouldn't need to implement `place_stock_order`. Existing `REQUIRED_STOCK_METHODS` parametrization for `AlpacaBroker`/`IBKRBroker` is unchanged.
- **Lazy import of IBKRFuturesBroker inside the test function**: avoids loading `ib_async` at module-import time, keeping the test suite startable in stock-only CI environments.

## Validation

- 4 unit tests pass on IBKRFuturesBroker skeleton (ABC compliance, abstract method coverage, signature matching, NotImplementedError raising)
- 6 unit tests pass on FuturesTradingInterface ABC (inheritance, abstract enforcement, signatures, instantiation block)
- 10 new parametrized tests pass on `test_broker_contract.py` (one per required futures method)
- 24 existing stock broker contract tests still pass -- no regression
- Total Chunk 6b test count: **20 new tests passing; 24 existing tests still passing**.

## Known Issues / Remaining Work

- **All broker methods raise NotImplementedError**. Functional implementation deferred to sub-chunks 6c (audit log), 6d (symbol resolver), 6e (expiration guard), 6f (margin guard + what_if_order + get_margin_status), 6g (combo atomicity), 6h (per-cycle reconciler), 6i (real upcoming rolls), 6j (final integration wiring).
- **No IBKR connection management**. The skeleton's `__init__` stashes `IBKRConfig` but does not start an `ib_async.IB` connection. Sub-chunk 6j wires this.
- **No new types added**. `place_futures_combo_order` takes `legs: list[dict]` for now; a typed `ComboLeg` / `ComboOrder` dataclass arrives in sub-chunk 6g.
- **`get_orders` and `get_open_orders` inherited from OrderManagementInterface** are stubbed but not in REQUIRED_FUTURES_METHODS -- they're not on the strategy call surface yet. Adding them to the test if a strategy starts using them is a one-line change.

## Decision Gate

PROCEED to sub-chunk 6c (`AuditLog` -- ~1.5 days per safeguards spec section 2.7). Audit log is orthogonal to other safeguards and can be developed in parallel with 6d/6e if desired; we'll execute sequentially.

## Reproduction Commands

```bash
cd C:/Users/qwqw1/Dropbox/cs/github/Homeguard
conda run -n fintech pytest tests/trading/brokers/interfaces/test_futures_trading.py tests/trading/brokers/ibkr/test_ibkr_futures_broker_skeleton.py tests/trading/brokers/test_broker_contract.py -v
# Expected: 44 passed (20 new + 24 existing stock contract tests)
```
