# Futures Broker Paper Validation Blockers (A + B) -- 2026-05-11

## Summary

Two technical blockers preventing the 14-day futures paper validation window (per `docs/trading/futures_paper_validation_runbook.md`) were resolved. Both landed on main; only the operational blocker remains (a strategy must call `submit_resolved_order` in paper).

## Changes Made

### Blocker B: Real expiration dates via FuturesDefinitionsLoader (merge 3c757e1)
- `src/data/futures_definitions_loader.py` (new): reads `H:/Stock_Data/futures_definitions/year=Y/month=M/data.parquet`, filters `instrument_class=='F'`, returns `ContractDefinition` with expiration / activation / tick_size / tick_value. Per-partition cache. 10 unit tests + real-data check (ESM4 -> 2024-06-21).
- `src/trading/futures/symbol_resolver.py`: `FuturesSymbolResolver.__init__` accepts optional `definitions_loader`. `ResolvedOrder` gains optional `expiration_date` field.
- `src/trading/brokers/ibkr/ibkr_futures_broker.py`: `submit_resolved_order` reads `expiration_date` from `resolved_order` when not explicitly passed; raises `ValueError` if neither is available.
- `scripts/trading/futures_paper_smoke_test.py`: drops the placeholder expiration; uses real definitions loader; logs the actual expiration in the resolve step.

### Blocker A: Real ib_async integration (merge 9c9cc95)
- `src/trading/brokers/ibkr/ibkr_futures_broker.py`: full rewrite from skeleton to real implementation, mirroring `IBKRBroker` (stocks) patterns. Uses `IBKRConnectionManager` singleton for the asyncio event loop.
  - Connection lifecycle: `start()`, `stop()`, `_ensure_connection()`
  - `_EXCHANGE_BY_ROOT` routing table: 40+ symbol roots -> CME / NYMEX / COMEX / CBOT
  - Helpers: `_build_future_contract`, `_build_order`, `_translate_trade`, `_translate_position`, `_side_to_ibkr`, `_tif_to_ibkr`, `_exchange_for`
  - All `FuturesTradingInterface` methods implemented (was `NotImplementedError`): `place_futures_order`, `place_futures_combo_order`, `get_futures_positions`, `get_futures_position`, `close_futures_position`, `close_all_futures_positions`, `what_if_order`, `get_margin_status`, plus `cancel_order` / `get_order` / `get_orders` / `get_open_orders` from `OrderManagementInterface`
- `tests/trading/brokers/ibkr/test_ibkr_futures_broker_ib_layer.py` (new, 22 tests): exercises the ib_async layer via mocked `IBKRConnectionManager`. Covers exchange routing, enum translation, order construction (all 4 types + error paths), trade translation, cancel/get/list filtering to `secType == 'FUT'`, position translation, and margin status parsing.
- `tests/trading/brokers/ibkr/test_ibkr_futures_broker_integration.py`: stubs `_ibkr_submit` in the passing-guards fixture so safeguard chain tests don't need a live IBKR connection.
- `tests/trading/brokers/ibkr/test_ibkr_futures_broker_skeleton.py`: drops the obsolete `test_skeleton_methods_raise_not_implemented` test; renamed file purpose to "shape tests".

## Commits

- `d4f2c46` feat(data,trading): FuturesDefinitionsLoader + wire real expirations through resolver
- `3c757e1` Merge feature/futures-definitions-loader: blocker B for paper validation
- `fe2d332` feat(trading): real ib_async integration in IBKRFuturesBroker
- `9c9cc95` Merge feature/ibkr-futures-real-integration: blocker A for paper validation

## Validation

- 119/119 futures-broker-related tests pass: 25 (definitions_loader + symbol_resolver + integration) + 22 (ib_layer) + others
- Real-data cross-check: `loader.get_definition("ESM4", "ES", date(2024, 6, 15))` returns `expiration=date(2024, 6, 21)` (third Friday of June, correct for CME E-mini quarterly)
- Smoke test syntax-clean and routed through real loader path; awaits live IBKR Gateway for end-to-end validation

## Known Issues / Remaining Work

- **Operational blocker remains**: no strategy currently invokes `submit_resolved_order`. The 14-day paper-validation window assumes a strategy is generating orders in paper for two weeks. That's Phase 2-5 work (Adaptation E / D / B / A / tactical overlay paper deployment).
- **`min_price_increment_amount` field appears mis-scaled**: ES shows `0.125` but real tick value is `$12.50`. `ContractDefinition.tick_value` is in this raw upstream unit; downstream consumers must verify the encoding before using as a dollar value. Documented in commit message; not blocking expiration-gate work.
- **`contract_multiplier` field is unreliable** (i32 sentinel `2147483647` for ES). Not exposed by `ContractDefinition`. Callers needing multiplier should source from a hardcoded per-root table.
- **No live test against IBKR paper yet**. Once IB Gateway is running on port 4002 with futures permissions, the smoke test should be run end-to-end to confirm the wiring works against the real API.

## Validation procedure (when ready to start the 14-day window)

1. Start IB Gateway on port 4002 (paper) with futures permissions enabled
2. Run `python scripts/trading/futures_paper_smoke_test.py` -- should exit 0 with `[OK]` markers (no `[!]` warnings about stubs)
3. Wire a strategy to call `submit_resolved_order` (Phase 2+ work)
4. Follow `docs/trading/futures_paper_validation_runbook.md` for the 14-day procedure and Day-14 acceptance checklist
