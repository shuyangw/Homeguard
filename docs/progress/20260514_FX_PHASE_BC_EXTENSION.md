# FX Data Phase B/C Extension - 2026-05-14 / 2026-05-15

## Summary

Continued the FX comprehensive expansion past the Phase 0/A/D/E baseline merged on 2026-05-14: ran Phase C (CME FX futures) and Phase B Tier 1 (FX BBO quote events). Phase C OHLCV-1m succeeded for 17 contracts; Phase C MBP-1 blocked at Databento billing. Phase B Tier 1 pulled 985 partitions of raw quote events (4.02B rows across 5 G7 pairs spanning 2010-01 -> 2026-05) and produced 30.0M derived minute bars; mid-price reconciles to existing `fx_1min/` within ~0.4 bps mean / ~1.8 bps p99.

## Changes Made

- **`src/data/acquisition/plugins/databento_futures.py`** (commit `193e2f6`):
  Extended for Databento `mbp-1` schema. Added `MBP1_CANONICAL_COLUMNS = ["ts_event", "bid_px", "ask_px", "bid_sz", "ask_sz"]`, `_normalize_mbp1` (maps `bid_px_00`/`ask_px_00`/`bid_sz_00`/`ask_sz_00`/`ts_event` columns), `_is_supported_schema("mbp-1")`, `_save_partitioned` override that partitions on `ts_event` instead of `timestamp`, and `_get_schema` returning the right canonical columns per schema.

- **`scripts/data/download_cme_fx_futures.py`**:
  Per-schema iteration. Tier 1 + Tier 2 for `ohlcv-1m`, Tier 1 only for `mbp-1`. Uses `DatabentoFuturesPlugin(schema=...)`. Replaces the `NotImplementedError` stub from the prior commit.

- **`src/data/acquisition/plugins/massive_fx_quotes_flat.py`** (NEW, commits `923b0e2` -> `cdee9c8` -> `a068ccc` -> `2730bd9`):
  Phase B Tier 1 plugin. Iterates days (not symbol-days) because Massive quotes_v1 files are per-day-all-pairs. Streams the gzipped CSV via `io.TextIOWrapper(gzip.GzipFile(...))` to bound RAM (dense days >5M rows / >1GB if naively `.splitlines()`'d). Header-mapped column indices (CSV header is alphabetical: `ticker, ask_exchange, ask_price, bid_exchange, bid_price, participant_timestamp` -- NOT logical order). Month-level skip-existing short-circuit: if all target pairs already have a parquet for (year, month), skip the per-day downloads entirely.

- **`scripts/data/download_fx_quotes.py`** (NEW): thin CLI wrapper mirroring `download_fx.py` (loads universe CSV, calls `download_pairs`, prints summary).

- **`config/universes/fx_quotes_tier1-2026.csv`** (NEW): 5 G7 pairs (EURUSD, USDJPY, GBPUSD, AUDUSD, USDCAD) with `effective_start_date=2010-01-01`.

- **`scripts/data/aggregate_fx_quotes_to_minute.py`** (NEW): walks `fx_quotes_raw/`, group_by_dynamic("1m"), separate OHLC per side, percentile-based spread distribution. Output: `fx_quotes_minute_aggregated/symbol={SYM}/year={Y}/month={M}/data.parquet`.

- **`tests/data/test_acquisition/test_massive_fx_quotes_flat.py`** (NEW): unit tests covering S3 key path, header-mapped parsing, target-filter behavior, row dtype assertions, parquet schema/dtype, and `iter_days_by_month`. (Tests use the hypothetical canonical header order in the sample CSV; current parser tolerates either order via index lookup.)

- **`docs/reference/DATA_INVENTORY.md`** (this session, uncommitted before merge):
  - Updated summary table: `futures_1min/` grew from 7,839 -> 9,180 files via 9 new CME FX contracts; added `fx_quotes_raw/` (4.02B / 14 GB) and `fx_quotes_minute_aggregated/` (30.0M / 0.9 GB) rows.
  - Updated "Pulled but not on disk" to include the Phase C CME FX MBP-1 deferment.
  - Updated `futures_1min/` symbol list to include `6L, 6Z, 6R, M6E, M6A, M6B, M6C, M6J, M6S`.
  - Replaced the CME FX section to reflect OHLCV-1m completion + MBP-1 billing block.
  - Added full sections for `fx_quotes_raw/` and `fx_quotes_minute_aggregated/`.
  - Bumped total to 31.0B rows / 386 GB.

## Commits

- `193e2f6` feat(data): wire MBP-1 through full Databento download pipeline
- `923b0e2` feat(data): Phase B Tier 1 quote data plugin + aggregation + CLI + tests
- `cdee9c8` fix(data): header-mapped CSV parser for quotes_v1 (actual order is alphabetical)
- `a068ccc` fix(data): stream-parse quote CSV to bound memory on dense days
- `2730bd9` fix(data): short-circuit month if all target outputs exist (skip S3 fetch)
- (pending in this session) docs(data,progress): Phase B/C inventory + session log

## Validation

- **Phase C OHLCV-1m**: 9 new CME FX contracts present under `futures_1min/symbol={6L,6Z,6R,M6E,M6A,M6B,M6C,M6J,M6S}/` with partition counts 66-192 each (varies by contract listing date). Original G10 (6E/6J/6B/6A/6C/6S/6N/6M) already had 189 partitions and were left as-is.

- **Phase C MBP-1**: all 11 Tier-1 submission attempts returned Databento `402 account_insufficient_funds`. Plugin code is exercised end-to-end (the 402 is at the billing gate, after schema validation) -- when budget is authorized, just re-run the same CLI.

- **Phase B Tier 1**:
  - 985 partitions / 4,019,671,795 rows / 14.0 GB on disk
  - All 5 pairs have full 197-month coverage (2010-01 -> 2026-05)
  - Per-pair: EURUSD 878.4M / USDJPY 836.1M / GBPUSD 929.1M / AUDUSD 692.8M / USDCAD 683.4M
  - Pull elapsed: ~6h overall (multiple restarts after the 3 plugin fixes); ~102.5 min for the final dense 2026-03/04/05 stretch
  - Aggregation: 30,025,933 minute bars / 891 MB / 985 partitions in `fx_quotes_minute_aggregated/`
  - **Spot-check** mid-close vs `fx_1min/` close on 2025-12 aligned minutes:
    - EURUSD: 30,979 rows, mean 0.38 bps, median 0.30 bps, p95 1.29 bps, p99 1.79 bps, max 7.44 bps
    - USDJPY: 30,932 rows, mean 0.37 bps, median 0.32 bps, p99 1.38 bps, max 6.68 bps
  - Spread microstructure looks reasonable (EURUSD 2025-12 median quoted spread `spread_p50` mean ~ 8e-5 = 0.8 pips).

## Known Issues / Remaining Work

- **Phase C MBP-1**: blocked on Databento billing. To resume: top up account, then `python scripts/data/download_cme_fx_futures.py --schema mbp-1` (will re-issue the same 11 Tier-1 requests).
- **`futures_mbp1/`** (ES/MES/NQ/MNQ legacy job): the older 480 GB pull is still in limbo. Different question from Phase C FX, same `account_insufficient_funds` issue.
- **`futures_trades_window/`** ES+MES trade-window pull: still pending budget decision.
- **Phase B Tier 2 / Tier 3**: deferred (mid-tier ~25-30 pairs, EM/Scandi ~30-40 pairs). YAGNI per the gate check -- no strategy currently consumes the quote data; Phase B Tier 1 is speculative infrastructure.
- **Phase F (history extension to pre-2010)**: deferred. Polygon/Massive archive does not extend significantly before 2010 for quotes; a separate source would be needed.
- **`fx_quotes_minute_aggregated/` `quote_count` dtype**: Polars wrote it as `UInt32`; existing FX OHLCV in `fx_1min/` uses Int64 for `volume`/`trade_count`. Compatible numerically but cross-dataset joins may need a cast. Not blocking.
