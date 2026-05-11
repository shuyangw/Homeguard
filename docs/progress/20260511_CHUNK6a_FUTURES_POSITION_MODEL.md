# Chunk 6a: FuturesPosition + v3 Schema Migration -- 2026-05-11

## Summary

First sub-chunk of futures broker safeguards (per `docs/superpowers/specs/2026-05-11-futures-broker-safeguards-design.md` Section 2.1). Adds the `FuturesPosition` dataclass with contract-month identity -- the foundation every other futures safeguard depends on. Includes a v2 -> v3 migration script that adds nullable futures-aware fields to existing stock/options position entries, and a schema doc describing the dual-asset-class loader branching rule.

## Files Changed

- `src/trading/futures/__init__.py` (new, empty package marker)
- `src/trading/futures/position.py` (new, 45 lines) -- `FuturesPosition` dataclass with `position_key` (reconciliation tuple) and `days_to_expiration` properties
- `tests/trading/futures/__init__.py` (new, empty)
- `tests/trading/futures/test_position.py` (new, 5 tests covering construction, key tuple, days-to-expiration past/future, short quantity)
- `scripts/data/migrate_positions_v2_to_v3.py` (new, ~70 lines) -- pure-function `migrate_state(dict) -> dict` + `migrate_file(src, dest)` + CLI; idempotent on already-v3 files
- `tests/data/test_migrate_positions_v2_to_v3.py` (new, 6 tests on synthetic v2 fixtures)
- `docs/storage/position_state_schema.md` (new) -- documents v3 schema, loader branching rule, migration procedure

## Commits

- `4bca0a7` feat(trading): FuturesPosition dataclass with contract-month identity
- `bc49c8b` feat(trading): v2 to v3 strategy_positions migration with futures fields

## Design Notes

- **Reconciliation key**: `position_key = (symbol_root, contract_month)`. Two positions match iff their keys are equal. This is the fix for the futures-specific failure mode where "long 2 MES" is ambiguous without contract month — production state and broker reports can silently disagree without a key like this.
- **Schema versioning is implicit on v2**: existing `data/trading/strategy_positions.json` has no `version` field. v3 adds `"version": 3` at the top level. Migration detects v3 by reading that field; absence implies v2.
- **Loader branching on `contract_month`**: stock/options entries get `contract_month: null` after migration; futures entries get `"YYYYMM"`. Loader code branches on null to dispatch between the existing `PositionInfo` path and the new `FuturesPosition` path. One file serves all asset classes.
- **Migration is idempotent**: running on a v3 file returns it unchanged. Safe to invoke from automation that doesn't know the current state.
- **Migration does NOT touch production**: the script is a tool operators invoke explicitly with input/output paths; this chunk leaves `data/trading/strategy_positions.json` untouched. The migration will be executed during paper-trading deployment of the first futures strategy.
- **`_today()` indirection in position.py**: monkeypatch target for tests so `days_to_expiration` is testable without freezing time globally.

## Validation

- 5 unit tests pass on `FuturesPosition` (construction, key, days-to-expiration past/future, short quantity).
- 6 unit tests pass on migration (version field, structure preservation, null-field addition, original-field preservation, idempotency, atomic file write).
- CLI smoke test: ran `migrate_positions_v2_to_v3.py` on a synthetic v2 fixture, verified output has `version: 3` and `contract_month: null` on stock entries.
- Total Chunk 6a test count: **11 passed**.

## Known Issues / Remaining Work

- **No loader code yet**. The schema doc describes the loader branching rule but no production code consumes v3 yet. The state manager (`src/trading/state/strategy_state_manager.py`) still uses the v2 schema. Updating that loader to branch on `contract_month` is part of sub-chunk 6b (`IBKRFuturesBroker` core skeleton) or later -- not blocking for this sub-chunk because no futures strategy is live yet.
- **Migration not yet executed against production state**. The script ships; the operator runs it when first futures position is about to open. Per the schema doc's "Migration procedure" section, this requires stopping running strategies first.

## Decision Gate

PROCEED to sub-chunk 6b (`FuturesTradingInterface` ABC + `IBKRFuturesBroker` core skeleton -- ~2 days per safeguards spec Section 4) once the merge to main lands.

## Reproduction Commands

```bash
cd C:/Users/qwqw1/Dropbox/cs/github/Homeguard
conda run -n fintech pytest tests/trading/futures/test_position.py tests/data/test_migrate_positions_v2_to_v3.py -v
# Expected: 11 passed
```
