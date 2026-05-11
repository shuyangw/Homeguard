# strategy_positions.json Schema

Reference for the on-disk schema of `data/trading/strategy_positions.json`, the source-of-truth for live strategy positions across all brokers. Last updated 2026-05-11.

## Versions

| Version | Introduced | Adds |
|---|---|---|
| (implicit v2) | Original | `strategies/<name>/positions/<symbol>/{qty, entry_price, entry_time, order_id}` |
| **v3** | 2026-05-11 | Futures-position fields with `null` markers on stocks/options for uniform loader branching |

The v2 schema has no explicit `version` field. v3 adds `"version": 3` at the top level. Migration tool: `scripts/data/migrate_positions_v2_to_v3.py`.

## v3 schema

```json
{
  "version": 3,
  "strategies": {
    "<strategy_name>": {
      "positions": {
        "<symbol>": {
          "qty": <int>,
          "entry_price": <float>,
          "entry_time": "<iso8601 utc>",
          "order_id": "<broker-specific id or null>",

          "contract_month": null | "<YYYYMM>",
          "raw_symbol": null | "<broker contract id, e.g. MESH6>",
          "multiplier": null | <float>,
          "tick_size": null | <float>,
          "tick_value": null | <float>,
          "expiration_date": null | "<iso8601 date>"
        }
      },
      "last_execution": "<iso8601 utc>"
    }
  }
}
```

### Field semantics

| Field | Stocks/Options | Futures |
|---|---|---|
| `qty` | shares (positive = long, negative = short) | contracts (positive = long, negative = short) |
| `entry_price` | per-share price | per-index-point price; multiply by `multiplier` for notional |
| `entry_time` | order-fill timestamp | order-fill timestamp |
| `order_id` | broker-specific order ID | broker-specific order ID |
| `contract_month` | **must be `null`** | `"YYYYMM"`, e.g. `"202603"` |
| `raw_symbol` | **must be `null`** | broker symbol, e.g. `"MESH6"` |
| `multiplier` | **must be `null`** | USD per point (e.g. 5.0 for MES) |
| `tick_size` | **must be `null`** | minimum price increment |
| `tick_value` | **must be `null`** | `multiplier * tick_size` |
| `expiration_date` | **must be `null`** | ISO date string (e.g. `"2026-03-20"`) |

### Loader branching rule

Loader code reading v3 entries branches on `contract_month`:

```python
if entry["contract_month"] is None:
    # stock or equity option -- handle via existing PositionInfo path
    ...
else:
    # futures -- handle via FuturesPosition path (src/trading/futures/position.py)
    ...
```

This pattern lets one mixed file serve all asset classes without forcing two parallel JSON files.

## Reconciliation key

| Asset class | Reconciliation key |
|---|---|
| Stocks / equity options | `symbol` (the outer key) |
| Futures | `(symbol_root, contract_month)` tuple |

Stocks: `"AAPL"` is `"AAPL"` regardless of when bought. Futures: long 2 `MESH6` and long 2 `MESM6` are TWO distinct positions even though both have `symbol_root = "MES"`. The reconciler at `src/trading/futures/position.py::FuturesPosition.position_key` returns the tuple key.

## Migration procedure

1. **Stop all running strategies** before migration. The `strategy_positions.json` file uses a cross-platform exclusive lock (`msvcrt.locking` on Windows, `fcntl.flock` on Unix); migrating while live trading runs will fail to acquire the lock.
2. **Back up the current file**: `cp data/trading/strategy_positions.json data/trading/strategy_positions.v2.bak.json`.
3. **Run the migration**:
   ```bash
   python scripts/data/migrate_positions_v2_to_v3.py \
     data/trading/strategy_positions.json \
     data/trading/strategy_positions.v3.json
   ```
4. **Verify**: open the output file and confirm `"version": 3` at top level; every existing position has `contract_month: null`.
5. **Replace the live file**: `mv data/trading/strategy_positions.v3.json data/trading/strategy_positions.json`.
6. **Restart strategies**. They should resume against the v3 file with no behavioral change (stocks/options paths see `contract_month: null` and use existing logic).

The migration is **idempotent** — running it on an already-v3 file is a no-op.

## When v3 is required

- **Now (paper trading)**: optional. Existing stock/options strategies (OMR, RAMP, RAMP-CSP) work with the v2 schema; they don't reference the new fields.
- **Before any futures position opens**: required. Without `contract_month`, two futures positions on the same root cannot be distinguished and reconciliation breaks.

The migration is queued as part of futures broker chunk 6a (this work). It will be executed as part of paper-trading deployment for the first futures strategy (Adaptation D — currently scheduled for Phase 3 per `02_IMMEDIATE_NEXT_STEPS.md`).

## Related

- `src/trading/futures/position.py` — `FuturesPosition` dataclass
- `src/trading/state/strategy_state_manager.py` — production loader and lock infrastructure
- `scripts/data/migrate_positions_v2_to_v3.py` — migration tool
- `tests/data/test_migrate_positions_v2_to_v3.py` — migration tests
- `docs/superpowers/specs/2026-05-11-futures-broker-safeguards-design.md` Section 2.1 — design rationale for contract-month identity
