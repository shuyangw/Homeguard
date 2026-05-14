# Data Inventory

Reference for all market data on `H:/Stock_Data/` (Windows) / `/home/ec2-user/stock_data` (Linux/EC2). Resolve at runtime via `from src.settings import get_local_storage_dir`. Last updated 2026-05-09.

## Summary

| Dataset | Source | Rows | Files | Size | Date range |
|---|---|---:|---:|---:|---|
| `equities_1min/` | Alpaca | 1.56B | 292,574 | 31.9 GB | 2016-01 → 2026-02 |
| `equities_1min_by_date/` | Derived (rebuilt 2026-05-06) | 1.56B | 2,587 | 35.7 GB | 2016-01 → 2026-01 |
| `crypto_1min/` | Alpaca / CoinAPI | 92.2M | 2,617 | 3.3 GB | 2020-11 → 2026-04 |
| `crypto_1hour/` | Alpaca | 657K | 919 | 0.04 GB | 2021-07 → 2025-12 |
| `crypto_1day/` | Alpaca | 33K | 1,092 | 0.01 GB | 2021-10 → 2025-09 |
| `crypto_1min_alpaca_archive/` | Alpaca (archive) | 29.7M | 919 | 1.2 GB | 2021-10 → 2025-09 |
| `futures_1min/` | Databento (.v.0) | 164.5M | 7,839 | 1.5 GB | 2010-06 → 2026-02 |
| `futures_1min_oi_roll/` | Databento (GC.n.0) | 5.5M | 189 | 0.06 GB | 2010-06 → 2026-02 |
| `futures_per_contract_1min/` | Databento (.FUT) | 578.0M | 189 | 8.3 GB | 2010-06 → 2026-02 |
| `futures_per_contract_daily/root=ED/` | Databento (GE.FUT, ohlcv-1d) | 1.2M | 14 | 0.02 GB | 2010-06 → 2023-12 |
| `futures_options_1min/` | Databento (.OPT) | 26.1M | 189 | 0.4 GB | 2010-06 → 2026-02 |
| `futures_definitions/` | Databento | 103.6M | 189 | 2.2 GB | 2010-06 → 2026-02 |
| `futures_statistics/` | Databento | 464.1M | 189 | 8.3 GB | 2010-06 → 2026-02 |
| `futures_status/` | Databento (status, .v.0 + .FUT) | 281.5M | 17 | 1.4 GB | 2010-06 → 2026-02 |
| `fx_1min/` | Polygon/Massive | 284.6M | 9,903 | 6.8 GB | 2010-01 → 2026-05 |
| `options/` | ThetaData / IBKR | 24.1B | 4,510 | 250.0 GB | 2012-06 → 2026-02 |
| `news/` | Alpaca / Benzinga | 587K | 2,985 | 0.15 GB | 2020-01 → 2025-12 |
| `sentiment/` | derived (FinBERT) | 424K | 1,719 | 0.05 GB | 2020-01 → 2025-12 |
| `futures_trades/` | Databento (stub) | 374K | 1 | <0.01 GB | 2024-01 (single day) |

**Total: ~27.0B rows across ~371 GB.** All timestamps `[us, UTC]` except crypto (`[ns, UTC]`, off-spec, see "Known dtype drift" below).

**Pulled but not on disk:**
- `futures_mbp1/` — MBP-1 tick data for ES/MES/NQ/MNQ Aug 2025-Feb 2026. Job F at Databento (5.9 GB partial dbn.zst staging only; full pull = 486 GB tick stream). Decision pending: resume the 480 GB download or drop entirely.
- `futures_trades_window/` — Trades schema for ES+MES, last hour daily, 5y. Job submission rejected with `402 account_insufficient_funds` (full 5y pull cost = $1040.68; 1y = $89.30; ES-only 5y = $601.52). Pending budget decision.

---

## Equities

### `equities_1min/` — Alpaca minute bars (canonical)

- **Schema**: `timestamp, open, high, low, close, volume, trade_count, vwap` (8 cols, `[us, UTC]`)
- **Partitioning**: `symbol={SYM}/year={YYYY}/month={M}/data.parquet`
  - Note: `month=` is unpadded (e.g. `month=1` not `month=01`)
- **Symbols**: 3,492 (full Russell 3000 + ETFs + leveraged ETFs)
- **Source**: Alpaca REST API via `src/data/acquisition/plugins/alpaca_equities.py`
- **Cleanup history**: `data_0.parquet` legacy filenames consolidated to canonical `data.parquet` (commit f63bc35 + normalize_equities_partitions.py, 2026-05-06). All ns→us dtype normalized.
- **Backup**: `H:/Stock_Data_backup_20260506/equities_1min/` has the pre-cleanup state (mixed dtypes, dual filenames). Will be deleted after a week of green production use.

### `equities_1min_by_date/` — Derived by-date pivot for backtest I/O

- **Schema**: `timestamp, open, high, low, close, volume, trade_count, vwap, symbol` (canonical OHLCV + symbol identifier column; `[us, UTC]`)
- **Partitioning**: flat — one parquet per trading day, `{YYYY-MM-DD}.parquet`
- **2,587 dates** spanning 2016-01-01 to 2026-01-14
- **Generator**: `scripts/data/migrate_to_time_partitioned.py`
- **Used by**: `src/backtesting/engine/streaming_data_loader.py` (when `prefer_time_partitioned=True`); each date file contains all symbols active that day, allowing single-file walk-forward I/O
- **Rebuilt**: 2026-05-06 (added 286 dates the previous biased migration silently dropped, including all of 2016)

---

## Crypto

### `crypto_1min/` — primary minute bars

- **Schema**: `timestamp, open, high, low, close, volume, trade_count, vwap` (`[ns, UTC]` — see dtype drift note)
- **Partitioning**: `symbol={SYM_USD}/year={YYYY}/month={M}/data.parquet`
- **Symbols**: ~10 majors + alts in pair format (`BTC_USD`, `ETH_USD`, `YFI_USD`, …)
- **Source**: Alpaca crypto endpoints + CoinAPI fallback (per CompositeDataProvider)
- **Used by**: CSCM strategy (cross-sectional crypto momentum)

### `crypto_1hour/` and `crypto_1day/` — aggregated bars

- Same schema, same partition layout (note `crypto_1hour` uses zero-padded `month=01` style; minor inconsistency)
- Generated via `scripts/data/aggregate_crypto_to_hourly.py` from minute bars

### `crypto_1min_alpaca_archive/` — frozen Alpaca-native snapshot

- Older subset, kept as a deterministic reference for backtest reproducibility before any data-source mixing happened
- Not actively maintained

---

## Futures (Databento GLBX.MDP3)

All futures were pulled in the bulk plan execution on 2026-05-07. Plan source: `docs/strategies/research/DATABENTO_BULK_PULL_PLAN.md` (filed under Downloads).

### `futures_1min/` — continuous OHLCV-1m, volume-roll (`.v.0`)

- **Schema**: `timestamp, open, high, low, close, volume` (6 cols; Databento ohlcv-1m has no trade_count/vwap)
- **Partitioning**: `symbol={ROOT}/year={YYYY}/month={M}/data.parquet` (unpadded month)
- **Symbols (53)**: equity index full + micros, energy, metals, rates, FX, ag, crypto
  - `ES, NQ, YM, RTY, MES, MNQ, M2K, MYM` (equity index)
  - `CL, NG, HO, RB, BZ, MCL, MNG` (energy)
  - `GC, SI, HG, PL, MGC, SIL` (metals)
  - `ZT, ZF, ZN, TN, ZB, UB, SR3, SR1, 10Y, 30Y, 5YY, 2YY` (rates)
  - `6E, 6J, 6B, 6A, 6C, 6S, 6N, 6M` (FX)
  - `ZC, ZS, ZW, KE, ZL, ZM, LE, HE` (ag)
  - `BTC, MBT, ETH, MET` (crypto)
- **Roll rule**: volume (`.v.0`) — replaced the broken calendar-roll (`.c.0`) which had ~7 bars/day on metals/grains. New density: ~1117 bars/day on GC (160× improvement).
- **Date floor**: 2010-06-06 (dataset floor; symbols listed later use their listing date)
- **Replaces**: `futures_1min_calendar_legacy_20260506/` (the old broken `.c.0` data, kept temporarily for verification then will be deleted)

### `futures_1min_oi_roll/` — GC continuous open-interest-roll (`.n.0`) diagnostic only

- Same schema and layout as `futures_1min/` but only contains `symbol=GC/`
- Used to validate `.v.0` roll behavior on metals (cross-check with OI-based roll)

### `futures_per_contract_1min/` — per-contract minute bars (parent symbology)

- **Schema**: `timestamp, rtype, publisher_id, instrument_id, open, high, low, close, volume, symbol` (10 cols)
- **Partitioning**: `year={YYYY}/month={M}/data.parquet` — all 53 .FUT families and all their expirations mixed in one monthly file
- Filter by `symbol` column at read time (raw CME symbols like `ESH4`, `GCM5`)
- Used to compute carry signals (front-month vs second-month basis on the same date) — continuous data can't give you this

### `futures_options_1min/` — options on futures, minute bars

- Same schema as `futures_per_contract_1min/`
- 13 .OPT families: `ES.OPT, NQ.OPT, RTY.OPT, CL.OPT, NG.OPT, GC.OPT, SI.OPT, ZN.OPT, ZB.OPT, 6E.OPT, 6J.OPT, ZC.OPT, ZS.OPT`
- All strike-expirations mixed; filter by `symbol` column at read time

### `futures_definitions/` — contract metadata events

- **Schema**: 65 columns including `timestamp, ts_recv, raw_symbol, security_update_action, expiration, activation, min_price_increment, display_factor, contract_multiplier, asset, cfi, security_type, strike_price, …` — full Databento `definition` schema
- **Partitioning**: `year={YYYY}/month={M}/data.parquet` (filename-based, not ts_event-based — see "Section D/E partitioning quirk" below)
- One row per contract per (Add/Modify/Delete) event
- Used for: roll detection, tick size lookup, contract multipliers, expiration dates, building a security master

### `futures_statistics/` — settle/OI/volume events

- **Schema**: `timestamp, ts_recv, ts_ref, price, quantity, stat_type, update_action, stat_flags, symbol, …` (15 cols)
- **Partitioning**: `year={YYYY}/month={M}/data.parquet` (filename-based)
- `stat_type` indicates the event kind: settlement, open interest, cleared volume, indicative open/close, etc. (see Databento docs for stat_type code table)

### `futures_trades/` — stub

- Only one parquet file present (`symbol=ES/year=2024/month=1/data.parquet`) from an early test run
- Schema: `timestamp, price, size` (raw trade events)
- Not actively maintained

### `futures_status/` — exchange status events (added 2026-05-09)

- **Schema**: `ts_recv, timestamp, rtype, publisher_id, instrument_id, action, reason, trading_event, is_trading, is_quoting, is_short_sell_restricted, symbol` (12 cols)
- **Partitioning**: `year={YYYY}/data.parquet` (year-flat, all symbols mixed)
- **Source**: Databento `status` schema, two jobs merged: `Status_continuous` (all 53 .v.0 symbols, 924K events) + `Status_parent` (53 .FUT parents expanded to all instruments, 280.5M events)
- **Total**: 281.5M rows across 17 yearly files (2010-2026)
- **Use**: catches halts, limits, pre-open/post-close transitions for backtest realism — strategies that simulate fills should consult this to skip bars where `is_trading=False` or `action` indicates halt/pause
- **Note**: `ts_recv` is `[ns, UTC]`, `timestamp` is `[us, UTC]` — partitioning is by `timestamp`

### `futures_per_contract_daily/` — per-contract daily OHLCV (added 2026-05-09)

- Currently populated only for `root=ED` (Eurodollar futures, phased out by CME 2023)
- **Schema**: `timestamp, rtype, publisher_id, instrument_id, open, high, low, close, volume, symbol` (10 cols, same as `futures_per_contract_1min/`)
- **Partitioning**: `root={ROOT}/year={YYYY}/data.parquet`
- **Source**: Databento `ohlcv-1d` for `GE.FUT` parent (CME's legacy Eurodollar symbol; Databento doesn't accept `ED.FUT`)
- **Coverage**: 1.2M rows across 14 yearly files, 2010-06-06 to 2023-12-31
- **Use**: pre-2018 funding rate proxy (before SR1 listing on 2018-05-07). Eurodollar 3-month rate ≈ short-term USD interbank rate

---

## FX

### `fx_1min/`

- **Schema**: `timestamp, open, high, low, close, volume, trade_count, vwap` (canonical OHLCV; `[ns, UTC]` -- off-spec but internally consistent)
- **Partitioning**: `symbol={SYM}/year={YYYY}/month={M}/data.parquet` (unpadded month)
- **55 symbols**, 9,903 partitions, 284.6M rows across 2010-2026
- **Source**: Polygon/Massive flat-files (S3 bucket `flatfiles`, path `global_forex/minute_aggs_v1/{YYYY}/{MM}/{YYYY-MM-DD}.csv.gz`). Authenticated via `MASSIVE_S3_*` env vars (separate from REST `MASSIVE_API_KEY`).
- **Ingestion**: `src/data/acquisition/plugins/massive_fx_flat.py` + `scripts/data/download_fx.py`. Universe at `config/universes/fx-2026.csv`. Per-day all-pairs CSV.gz files (1,200+ tickers), parsed and filtered per-symbol-per-month into Parquet matching canonical schema.
- **Coverage notes**:
  - 16-year depth (2010-onward) for 54 pairs; SGDJPY starts 2020 (Polygon's archive limit for this cross)
  - Sparse early-2010s for EM pairs: USDBRL, USDCLP, USDINR, USDKRW, USDRUB minute coverage thin pre-2017; daily coverage existed but minute didn't
  - USDCNH starts 2014-04; XAGUSD starts 2013-07 (Polygon archive limits)
  - 2019-09: cross-asset thin month for multiple G10 pairs (4-5% density) — Polygon-side gap; deferred Dukascopy patch
  - 2020-10/11 EURUSD outage: Polygon-specific, deferred Dukascopy patch
- **`volume` field**: FX is OTC market with no centralized volume; `volume == trade_count` in flat-file source (tick count, not value)
- **`vwap` field**: Polygon's flat-file schema omits vwap; new pairs from `massive_fx_flat.py` set `vwap = close` as documented approximation. Existing 50 pairs (pre-2026-05-13) have a separate vwap value (provider-computed).

---

## Options (equities)

### `options/options_combined/`

- **Schema**: `symbol, expiration, strike, right, timestamp, open, high, low, close, volume, trade_count, vwap, bid_close, ask_close, implied_vol, delta, theta, vega, underlying_px, gamma_eod, open_interest_eod` (21 cols)
- **Partitioning**: `options_combined/root={ROOT}/year={YYYY}/month={MM}/data.parquet` (zero-padded month here)
- 24.1B rows across 4,510 files (~250 GB) — by far the largest dataset
- Source: ThetaData via `src/data/options/thetadata_adapter.py` + IBKR options chains
- Used by RAMP options pipeline research, CSCM, future options strategies

---

## News and sentiment

### `news/`

- **Schema**: `id, timestamp, symbol, symbols, headline, summary, source, url, author, content`
- **Partitioning**: `symbol={SYM}/year={YYYY}/{news|news_with_sentiment}.parquet`
- Source: Alpaca News API / Benzinga
- Note: 8 partitions have BOTH `news.parquet` and `news_with_sentiment.parquet` — these are different files (sentiment-enriched vs raw), not duplicates

### `sentiment/`

- **Schema**: `id, timestamp, symbol, headline, sentiment_score, sentiment_positive, sentiment_negative, sentiment_neutral, sentiment_label, confidence`
- **Partitioning**: `symbol={SYM}/year={YYYY}/sentiment.parquet`
- Generated via `scripts/data/compute_sentiment.py` using FinBERT

---

## Cross-cutting notes

### Canonical timestamp dtype

**Documented standard** (see `.claude/data_handling.md`): `Datetime[us, UTC]`.

**Actual state**:
- equities and futures (post-2026-05 cleanup): `[us, UTC]` ✓
- crypto, fx (existing): `[ns, UTC]` (off-spec, internally consistent within each dataset)
- news, sentiment, options: mixed; check before assuming

The dtype drift on crypto/fx is documented but not actively fixed. Polars normalizes transparently in the streaming loader; pyarrow is stricter. If a tool fails on dtype mismatch, cast on read.

### Section D / E partitioning quirk

For `futures_definitions/` and `futures_statistics/`, partitioning is by **filename date** (Databento's batch-job split boundary), not by `ts_event`. This is because definition events reference original contract creation dates which can predate the file's nominal date range — `ts_event` is when a contract was last modified, not when the snapshot was published. Partitioning by `ts_event` would route boundary-spillover rows to the wrong month, causing partition collisions and silent data loss (53 D + 22 E partitions were lost this way before the fix on 2026-05-08; full 189-partition coverage now restored).

For sections A_v/A_n_diag/B/C (OHLCV bars), `ts_event` matches the bar's minute and partitioning by it is correct.

### Storage path resolution

Always use:
```python
from src.settings import get_local_storage_dir
data_root = get_local_storage_dir()
df = pl.read_parquet(data_root / "futures_1min" / "symbol=ES" / "year=2024" / "month=1" / "data.parquet")
```

Hard-coded paths break on EC2 (Linux) where the root is `/home/ec2-user/stock_data` rather than `H:/Stock_Data/`.

### Read patterns

- Single (symbol, month): `pl.read_parquet(data_root / "futures_1min" / f"symbol={s}" / f"year={y}" / f"month={m}" / "data.parquet")`
- Full symbol: `pl.scan_parquet(data_root / "futures_1min" / f"symbol={s}").collect()`
- Cross-symbol on a date (equities): `pl.scan_parquet(data_root / "equities_1min_by_date" / f"{date}.parquet")`
- Per-contract carry (futures): scan `futures_per_contract_1min/year=Y/month=M/data.parquet` then filter by `symbol` for the front-month and second-month contracts on a given date
