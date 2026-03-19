# Data Acquisition Consolidation + Databento Futures Integration

Date: 2026-02-25
Status: Approved

## Problem

Data downloading is spread across 5+ independent modules with ~70% duplicated
infrastructure (threading, retry, hive-partitioned storage, skip-existing, progress
reporting). Adding a new data source (Databento futures) requires re-implementing all
of this from scratch.

Current modules:
- `src/data/downloader.py` - Alpaca equities
- `src/data/crypto_downloader.py` - Alpaca crypto
- `src/data/news/news_downloader.py` - Alpaca news
- `scripts/download_options_combined.py` + `tools/options-downloader/` - ThetaData options
- (NEW) Databento futures

## Approach

**Plugin-Based DataAcquisitionManager** - consolidate all data acquisition into a
single `src/data/acquisition/` module with shared infrastructure and per-source plugins.

## Module Structure

```
src/data/acquisition/
    __init__.py              # Public API
    manager.py               # DataAcquisitionManager - orchestrator + CLI
    base.py                  # BaseDownloader - shared infrastructure
    schemas.py               # Canonical schemas per asset class + validation
    manifest.py              # Unified JSON manifest tracker
    aggregators.py           # Trade -> OHLCV reconstruction
    plugins/
        __init__.py          # Plugin registry (lazy-loaded)
        alpaca_equities.py   # Wraps AlpacaDownloader fetch logic
        alpaca_crypto.py     # Wraps CryptoDownloader fetch logic
        databento_futures.py # NEW - Databento GLBX.MDP3 trades
        thetadata_options.py # Thin wrapper for options
        alpaca_news.py       # Wraps NewsDownloader fetch logic
```

## BaseDownloader

Shared infrastructure extracted from existing downloaders:

- ThreadPoolExecutor with configurable workers
- Retry with exponential backoff (configurable per-plugin)
- End-of-run retry rounds
- Hive-partitioned parquet writes (symbol={S}/year={Y}/month={M}/data.parquet)
- Skip-existing detection
- Manifest tracking (JSON, states: pending/in_progress/complete/failed/partial)
- Progress logging with ETA
- Failed symbol tracking

Each plugin implements only:
- `_create_client() -> Any` - thread-local API client
- `_fetch_symbol_data(client, symbol, start, end) -> pd.DataFrame` - API call
- `_get_schema() -> list[str]` - canonical columns for validation
- `_get_storage_subdir() -> str` - e.g., "futures_1min"
- `_normalize_symbol(symbol) -> str` - filesystem-safe name

## Databento Futures Plugin

**Contracts** (continuous front-month): ES.c.0, NQ.c.0, CL.c.0, GC.c.0, ZN.c.0,
6E.c.0, ZC.c.0, YM.c.0, RTY.c.0

**Date range**: 2020-01-01 to present

**Schema**: `trades` (raw tick data from GLBX.MDP3)

**Two-stage storage**:

| Stage | Directory | Schema |
|-------|-----------|--------|
| Raw trades | futures_trades/symbol=ES/year=Y/month=M/data.parquet | timestamp, price, size, side, trade_id |
| Reconstructed 1m | futures_1min/symbol=ES/year=Y/month=M/data.parquet | 8-col canonical |

**API flow**:
```
databento.Historical(key=DATABENTO_API_KEY)
    .timeseries.get_range(
        dataset="GLBX.MDP3",
        schema="trades",
        stype_in="continuous",
        symbols=["ES.c.0"],
        start="2020-01-01",
    )
    -> stream to DataFrame -> store raw trades -> reconstruct OHLCV-1m
```

**API key**: Stored as DATABENTO_API_KEY in .env, loaded via os.getenv().

## OHLCV-1m Reconstruction from Trades

`aggregators.py` - `trades_to_ohlcv_1m(trades_df) -> pd.DataFrame`

Rules:
- Group by 1-minute floor of timestamp
- open: first trade price in window
- high: max trade price
- low: min trade price
- close: last trade price
- volume: sum of trade sizes
- trade_count: count of trades in window
- vwap: sum(price * size) / sum(size)
- Minutes with no trades: excluded (no forward-fill)

Edge cases:
- Single trade in minute -> O=H=L=C
- Zero-volume trades -> defensive handling in VWAP
- Duplicate trade IDs -> deduplicate before aggregation
- CME session boundaries (Sun 5pm - Fri 4pm CT)
- Databento fixed-point price scaling

## DataAcquisitionManager

Single entry point for all downloads:

```python
manager = DataAcquisitionManager()
result = manager.download(
    source="futures",
    symbols=["ES", "NQ", "CL"],
    start_date="2020-01-01",
    skip_existing=True,
)
```

Unified CLI:
```
python -m src.data.acquisition --source futures --symbols ES,NQ,CL --start 2020-01-01
python -m src.data.acquisition --source equities --csv config/universes/sp500-2025.csv --skip-existing
python -m src.data.acquisition --status
python -m src.data.acquisition --source futures --retry-failed
```

## Plugin Registry

```python
PLUGIN_REGISTRY = {
    "equities": "src.data.acquisition.plugins.alpaca_equities.AlpacaEquitiesPlugin",
    "crypto": "src.data.acquisition.plugins.alpaca_crypto.AlpacaCryptoPlugin",
    "futures": "src.data.acquisition.plugins.databento_futures.DatabentoFuturesPlugin",
    "options": "src.data.acquisition.plugins.thetadata_options.ThetaDataOptionsPlugin",
    "news": "src.data.acquisition.plugins.alpaca_news.AlpacaNewsPlugin",
}
```

Lazy-loaded - plugins only import their SDK when instantiated.

## Migration Strategy

Existing modules remain as thin re-exports for backward compatibility:
- `from src.data import AlpacaDownloader` continues to work
- `scripts/data/download_symbols.py` delegates to new CLI under the hood
- No breaking changes to existing imports or tests

## Manifest System

Per-source manifests at `{storage_dir}/_manifests/{source}.json`:
```json
{
    "source": "futures",
    "entries": {
        "ES|2024-01": {"status": "complete", "rows": 28450, "updated_at": "..."},
        "NQ|2024-02": {"status": "failed", "error": "timeout", "updated_at": "..."}
    }
}
```

## Error Handling

| Error type | Behavior |
|-----------|----------|
| Rate limit (429) | Exponential backoff, configurable per-plugin |
| Timeout | Retry up to max_retries, then mark failed |
| Auth failure | Fail fast, no retry |
| Partial data | Log warning, save what we got, mark partial |
| Network error | Retry with backoff |
| Schema mismatch | Validate columns, raise SchemaValidationError |

## Configuration

API key in .env:
```
DATABENTO_API_KEY="<YOUR_DATABENTO_API_KEY>"
```

Each plugin declares required env vars. Base class validates at init and fails fast.

## Testing

### test_base.py - BaseDownloader infrastructure
- test_hive_partition_structure
- test_skip_existing
- test_retry_on_failure
- test_end_of_run_retry_rounds
- test_manifest_tracking
- test_manifest_resume
- test_schema_validation
- test_progress_reporting
- test_thread_safety

### test_aggregators.py - OHLCV reconstruction (comprehensive)
- test_basic_ohlcv_reconstruction
- test_single_trade_per_minute
- test_empty_minutes_excluded
- test_vwap_calculation
- test_vwap_zero_volume
- test_trade_count_accuracy
- test_output_schema_matches_canonical
- test_timestamp_alignment
- test_duplicate_trades_deduplicated
- test_large_volume_no_overflow
- test_overnight_session
- test_reconstruction_deterministic

### test_plugins.py - Plugin-specific
- test_databento_symbol_normalization
- test_databento_schema_output
- test_databento_missing_api_key
- test_equities_plugin_matches_original
- test_crypto_plugin_symbol_normalization
- test_plugin_registry_lazy_loading
- test_all_plugins_discoverable
