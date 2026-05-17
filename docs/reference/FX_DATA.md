# FX Data

Consolidated reference for all FX-related market data on Homeguard. Last updated 2026-05-15.

For datasets outside FX (equities, futures, options, etc.), see [`DATA_INVENTORY.md`](DATA_INVENTORY.md). This file is the single source-of-truth for "what FX data do we have?"

## TL;DR

| Dataset | Frequency | Symbols | Rows | Disk | Coverage |
|---|---|---:|---:|---:|---|
| [`fx_1min/`](#fx_1min--minute-ohlcv-from-trades) | 1-min OHLCV (trades) | 80 pairs | 383.4M | 8.6 GB | 2010-01 → 2026-05 |
| [`fx_quotes_raw/`](#fx_quotes_raw--bbo-quote-events-tick-level) | Per-event BBO | 5 G7 pairs | 4.02B | 14.0 GB | 2010-01 → 2026-05 |
| [`fx_quotes_minute_aggregated/`](#fx_quotes_minute_aggregated--minute-bars-from-quotes) | 1-min bid/ask + spread | 5 G7 pairs | 30.0M | 0.9 GB | 2010-01 → 2026-05 |
| [`futures_1min/` (CME FX subset)](#cme-fx-futures-in-futures_1min) | 1-min OHLCV | 17 contracts | ~10M | ~80 MB | 2010-06 → 2026-04 |
| [`equities_1min/` (FX-adjacent ETFs)](#fx-adjacent-etfs-in-equities_1min) | 1-min OHLCV | 27 ETFs | 18.1M | n/a | varies |
| [`alt_data/fred/`](#alt_datafred--macro-rates) | Daily | 28 series | 173K | <1 MB | series-dependent |
| [`alt_data/cot/`](#alt_datacot--cftc-positioning) | Weekly | 11 CME FX | 6.2K | <1 MB | 2010 → 2026 |

**Spot FX provider**: Polygon (rebranded as Massive) S3 flat-files.
**CME FX futures**: Databento.
**FX-adjacent ETFs**: Alpaca.
**Macro + positioning**: FRED + CFTC.

There is no real-time FX integration. All FX is batch-pulled for research; live trading uses Alpaca/IBKR for equities/futures only.

---

## Provider & auth

### Massive (Polygon) — spot FX

Two separate credential systems on the Massive account; the REST key does NOT grant flat-file access.

| Credential | What it grants | env vars |
|---|---|---|
| REST API key | `api.polygon.io` REST endpoints (returns same data as flat-files for FX, but byte-byte slower) | `MASSIVE_API_KEY` |
| S3 flat-files | `flatfiles/global_forex/{minute_aggs_v1,quotes_v1,day_aggs_v1}/...` | `MASSIVE_S3_ACCESS_KEY`, `MASSIVE_S3_SECRET_KEY`, `MASSIVE_S3_ENDPOINT`, `MASSIVE_S3_BUCKET` |

S3 layout (canonical):
```
flatfiles/
  global_forex/
    minute_aggs_v1/{YYYY}/{MM}/{YYYY-MM-DD}.csv.gz   -- one daily file = ALL pairs
    quotes_v1/{YYYY}/{MM}/{YYYY-MM-DD}.csv.gz        -- one daily file = ALL pairs
    day_aggs_v1/...                                  -- not currently ingested
```

Polygon's archive itself has known gaps (documented per-dataset below). REST backfill returns byte-identical data to flat-files; gaps are not on our end.

### Databento — CME FX futures

API key in `.env` as `DATABENTO_API_KEY`. Schemas used: `ohlcv-1m` (working), `mbp-1` (plugin extended but blocked on billing). Dataset: `GLBX.MDP3`.

### Alpaca — FX-adjacent equity ETFs

Standard `ALPACA_*` env vars. Uses the same `alpaca_equities.py` plugin as the rest of the equity universe.

### FRED / CFTC — macro & positioning

Public data; no auth required. FRED uses `pandas-datareader` (already in `requirements.txt`). CFTC pulls from `https://www.cftc.gov/files/dea/history/fut_fin_txt_{YEAR}.zip` (Traders in Financial Futures historical archive).

---

## Datasets

### `fx_1min/` — minute OHLCV from trades

Trade aggregations (no bid/ask). The workhorse FX dataset.

- **Schema (8 cols)**: `timestamp, open, high, low, close, volume, trade_count, vwap`
- **Dtypes**: `Datetime[ns, UTC]` + Float64×5 + Int64×2 + Float64 (`[ns, UTC]` is off-spec — canonical is `[us, UTC]` — but internally consistent across the dataset)
- **Partitioning**: `symbol={SYM}/year={YYYY}/month={M}/data.parquet` (unpadded month)
- **80 pairs**, 13,321 partitions, 383.4M rows
- **Source**: Polygon flat-files `global_forex/minute_aggs_v1/{YYYY}/{MM}/{YYYY-MM-DD}.csv.gz` (per-day all-pairs CSV with ~1,200 tickers; we filter post-download)
- **Plugin**: [`src/data/acquisition/plugins/massive_fx_flat.py`](../../src/data/acquisition/plugins/massive_fx_flat.py)
- **CLI**: [`scripts/data/download_fx.py`](../../scripts/data/download_fx.py) `--csv config/universes/fx-2026.csv --skip-existing`

**`volume` semantics**: FX is OTC with no centralized volume; `volume == trade_count` in the flat-file source (tick count, not value).

**`vwap` semantics**: Polygon's flat-file schema omits vwap. Pairs ingested by `massive_fx_flat.py` set `vwap = close` as a documented approximation. The 50 pairs already on disk pre-2026-05-13 have provider-computed vwap values from an older REST-based pull.

**Pair universe** ([`config/universes/fx-2026.csv`](../../config/universes/fx-2026.csv), 34 pairs cataloged + 50 pre-existing = 80 on disk):

Majors / G10 (10):
- EURUSD, USDJPY, GBPUSD, AUDUSD, USDCAD, USDCHF, NZDUSD, USDNOK, USDSEK + EUR/GBP/AUD/JPY/CHF crosses

G10 crosses already present pre-2026-05 (~30):
- EURGBP, EURJPY, EURAUD, EURCAD, EURCHF, EURNZD, EURNOK, EURSEK, GBPJPY, GBPAUD, GBPCAD, GBPCHF, GBPNZD, AUDJPY, AUDCAD, AUDCHF, AUDNZD, CADJPY, CADCHF, CHFJPY, NZDJPY, NZDCAD, NZDCHF, NOKSEK, NOKJPY*, SEKJPY*, AUDSGD, USDSGD, EURSGD, SGDJPY*

EM (USD-pegged):
- USDBRL, USDCNH (from 2014-04), USDCLP, USDCZK, USDHKD, USDHUF, USDILS, USDINR, USDKRW, USDMXN, USDPLN, USDRUB, USDTRY, USDZAR

Cross EM (Phase A L3, 2026-05-14):
- EURMXN, EURZAR, EURCNH, EURPLN, GBPMXN, GBPZAR, GBPCNH*, AUDCNH, AUDMXN

Metals (spot, settled in USD or counter currency):
- XAUUSD (gold, from 2011), XAGUSD (silver, from 2013-07), XAUEUR*, XAUGBP*, XAUJPY*, XAUAUD*, XAGEUR*, XAGGBP*, XAGJPY*

\* = recent-only (Polygon archive starts 2020 or later for that pair).

**Coverage caveats:**
- Pre-2017 EM minute coverage is thin for USDBRL, USDCLP, USDINR, USDKRW, USDRUB (daily coverage existed but minute aggregations didn't)
- `2019-09` is a cross-asset thin month for multiple G10 pairs (4-5% density) — Polygon-side gap. Deferred Dukascopy patch.
- `2020-10`/`2020-11`: EURUSD outage in Polygon archive (specific to Polygon). Deferred Dukascopy patch.
- USDCNH starts 2014-04, XAGUSD starts 2013-07, SGDJPY/NOKJPY/SEKJPY/all XAU+XAG crosses start 2020 — all Polygon archive floors.

### `fx_quotes_raw/` — BBO quote events (tick level)

Every bid/ask update an exchange published. 4.02B events across 5 pairs, the largest FX dataset.

- **Schema (5 cols)**: `timestamp, bid_price, ask_price, bid_exchange, ask_exchange`
- **Dtypes**: `Datetime[ns, UTC]` + Float64×2 + Int32×2
- **Partitioning**: `symbol={SYM}/year={YYYY}/month={M}/data.parquet` (unpadded month)
- **5 pairs × 197 months = 985 partitions**, 14.0 GB

| Symbol | Files | Events |
|---|---:|---:|
| EURUSD | 197 | 878.4M |
| USDJPY | 197 | 836.1M |
| GBPUSD | 197 | 929.1M |
| AUDUSD | 197 | 692.8M |
| USDCAD | 197 | 683.4M |
| **Total** | **985** | **4,019.7M** |

- **Source**: Polygon flat-files `global_forex/quotes_v1/{YYYY}/{MM}/{YYYY-MM-DD}.csv.gz`
- **Plugin**: [`src/data/acquisition/plugins/massive_fx_quotes_flat.py`](../../src/data/acquisition/plugins/massive_fx_quotes_flat.py)
- **CLI**: [`scripts/data/download_fx_quotes.py`](../../scripts/data/download_fx_quotes.py) `--csv config/universes/fx_quotes_tier1-2026.csv`
- **Universe**: [`config/universes/fx_quotes_tier1-2026.csv`](../../config/universes/fx_quotes_tier1-2026.csv) (5 G7 pairs)

**Source CSV schema** (alphabetical header — NOT logical order):
```
ticker, ask_exchange, ask_price, bid_exchange, bid_price, participant_timestamp
```
The plugin maps columns by header name, not positional index.

**Streaming parser**: gzipped daily files exceed 80 MB decompressed (>5M lines / >1 GB if naively `.splitlines()`'d). The parser uses `io.TextIOWrapper(gzip.GzipFile(...))` to iterate line-by-line and bound RAM at ~100 MB per worker. With `concurrency=4` (default), peak memory is ~400 MB.

**Month-level skip-existing**: if all target pairs already have `data.parquet` for `(year, month)`, the per-day S3 downloads for that month are short-circuited. Resume after partial failure does not re-download already-completed months.

**Use cases**:
- Spread microstructure (effective vs quoted spread, sub-second flicker)
- Execution-cost research
- Source for [`fx_quotes_minute_aggregated/`](#fx_quotes_minute_aggregated--minute-bars-from-quotes)

### `fx_quotes_minute_aggregated/` — minute bars from quotes

Derived from `fx_quotes_raw/`. Per minute: separate bid OHLC, ask OHLC, and the within-minute spread distribution.

- **Schema (14 cols)**: `timestamp, bid_open, bid_high, bid_low, bid_close, ask_open, ask_high, ask_low, ask_close, spread_mean, spread_p50, spread_p95, spread_p99, quote_count`
- **Dtypes**: `Datetime[ns, UTC]` + Float64×12 + UInt32
- **Partitioning**: same as `fx_quotes_raw/` (5 pairs × 197 months = 985 partitions)
- **30.0M minute bars**, 0.9 GB

| Symbol | Bars |
|---|---:|
| EURUSD | 6.01M |
| USDJPY | 5.99M |
| GBPUSD | 6.01M |
| AUDUSD | 6.04M |
| USDCAD | 5.98M |

- **Generator**: [`scripts/data/aggregate_fx_quotes_to_minute.py`](../../scripts/data/aggregate_fx_quotes_to_minute.py) (polars `group_by_dynamic("1m")` over the raw event stream)

**Sanity-check** vs `fx_1min/` close on 2025-12 aligned minutes:

| Pair | rows | mean | median | p95 | p99 | max |
|---|---:|---:|---:|---:|---:|---:|
| EURUSD | 30,979 | 0.38 bps | 0.30 bps | 1.29 bps | 1.79 bps | 7.44 bps |
| USDJPY | 30,932 | 0.37 bps | 0.32 bps | — | 1.38 bps | 6.68 bps |

Mid-price `(bid_close + ask_close) / 2` matches `fx_1min/` trade-close within ~0.4 bps mean — internally consistent.

**Use cases**: realistic spread/slippage for backtest fills, microstructure features (quoted-spread distribution, quote intensity).

### CME FX futures (in `futures_1min/`)

Continuous OHLCV-1m for CME FX contracts, volume-roll (`.v.0`). Live in the main `futures_1min/` tree alongside other CME products — there is no separate `fx_futures_1min/` dataset.

| Symbol | Description | Partitions | Tier |
|---|---|---:|---|
| 6E | Euro FX (full) | 189 | 1 (G10) |
| 6J | Japanese Yen | 189 | 1 (G10) |
| 6B | British Pound | 189 | 1 (G10) |
| 6A | Australian Dollar | 189 | 1 (G10) |
| 6C | Canadian Dollar | 189 | 1 (G10) |
| 6S | Swiss Franc | 189 | 1 (G10) |
| 6N | New Zealand Dollar | 189 | 1 (G10) |
| 6M | Mexican Peso | 189 | 1 (EM) |
| 6L | Brazilian Real | 192 | 1 (EM, Phase C) |
| 6Z | South African Rand | 192 | 1 (EM, Phase C) |
| 6R | Russian Ruble | 148 | 1 (EM, Phase C; delisted 2022) |
| M6E | E-micro Euro | 192 | 2 (Phase C) |
| M6A | E-micro AUD | 192 | 2 (Phase C) |
| M6B | E-micro GBP | 192 | 2 (Phase C) |
| M6C | E-micro CAD | 73 | 2 (Phase C) |
| M6J | E-micro JPY | 94 | 2 (Phase C) |
| M6S | E-micro CHF | 66 | 2 (Phase C) |

- **Schema**: `timestamp, open, high, low, close, volume` (6 cols; Databento `ohlcv-1m` omits trade_count/vwap)
- **Plugin**: [`src/data/acquisition/plugins/databento_futures.py`](../../src/data/acquisition/plugins/databento_futures.py)
- **CLI**: [`scripts/data/download_cme_fx_futures.py`](../../scripts/data/download_cme_fx_futures.py)
- **Universe**: [`config/universes/cme_fx_futures-2026.csv`](../../config/universes/cme_fx_futures-2026.csv) (17 contracts)

**MBP-1 status**: plugin extended for `mbp-1` schema (top-of-book per-event, would write to a separate `futures_mbp1/` tree). All 11 Tier-1 MBP-1 submissions on 2026-05-14 rejected with Databento `402 account_insufficient_funds`. Code is ready; pending budget decision.

### FX-adjacent ETFs (in `equities_1min/`)

27 ETFs live in the main `equities_1min/` tree (no separate dataset). Universe at [`config/universes/fx_adjacent_equity-2026.csv`](../../config/universes/fx_adjacent_equity-2026.csv).

**Currency ETFs (9)**: FXE (EUR), FXY (JPY), FXB (GBP), FXA (AUD), FXC (CAD), FXF (CHF), FXS (SEK), UUP (USD bull), UDN (USD bear)

**Country equity ETFs (14)**: EWJ, EWZ, EWW, EWA, EWC, FXI, MCHI, INDA, EZA, EWY, EWS, EWG, EWU, ILF

**EM bond ETFs (4)**: EMB, EMLC, LEMB, PCY

Acquired via [`scripts/data/download_fx_adjacent_equity.py`](../../scripts/data/download_fx_adjacent_equity.py) (Phase D, 2026-05-14). 18.1M rows total. Schema matches the standard `equities_1min/` canonical 8 cols.

### `alt_data/fred/` — macro rates

28 FRED series at daily frequency. FX-relevant subset:
- US Treasury curve: DGS1MO, DGS3MO, DGS6MO, DGS1, DGS2, DGS5, DGS10, DGS30
- SOFR family + Fed Funds (FEDFUNDS, EFFR, SOFR)
- Foreign policy rates (where published): ECB MRO, BoE Bank Rate, BoJ overnight, BoC, RBA, RBNZ, SNB, Riksbank, Norges, BCB, Banxico
- TIPS inflation expectations: T5YIE, T10YIE
- Daily fixings: select USD/foreign FX where FRED publishes (DEXUSEU, DEXJPUS, DEXUSUK, etc.)

- **Schema**: `date (pl.Date), value (pl.Float64)`
- **Partitioning**: `{series_id}/daily.parquet`
- 173K rows total, <1 MB
- **Plugin**: [`src/data/acquisition/plugins/fred_rates.py`](../../src/data/acquisition/plugins/fred_rates.py)
- **Universe**: `config/universes/fred_series-2026.csv`

### `alt_data/cot/` — CFTC positioning

Weekly Traders in Financial Futures (TFF) reports for the 11 CME FX contracts where positioning is published. Source: `https://www.cftc.gov/files/dea/history/fut_fin_txt_{YEAR}.zip`.

- **Schema**: `report_date, dealer_long, dealer_short` (canonical subset of CFTC TFF's ~87 columns)
- **Partitioning**: `{instrument}/weekly.parquet`
- **Instruments (11)**: 6E, 6J, 6B, 6S, 6C, 6A, 6N, 6M, 6L, 6Z, 6R
- 6,189 weekly rows total
- Coverage 2010-2026 (RUB delisted 2022, lower row count)
- **Plugin**: [`src/data/acquisition/plugins/cftc_cot.py`](../../src/data/acquisition/plugins/cftc_cot.py)
- **Universe**: `config/universes/cot_instruments-2026.csv`

---

## Cross-cutting

### Timestamp dtype drift

All `fx_*` datasets use `Datetime[ns, UTC]`. The canonical Homeguard standard is `[us, UTC]` (see `.claude/data_handling.md`). FX is off-spec but internally consistent — polars normalizes transparently in the streaming loader; pyarrow is stricter and may need an explicit cast on read.

### Storage path resolution

```python
from src.settings import get_local_storage_dir
data_root = get_local_storage_dir()

# fx_1min
df = pl.read_parquet(data_root / "fx_1min" / "symbol=EURUSD" / "year=2025" / "month=12" / "data.parquet")

# fx_quotes_minute_aggregated (same shape)
df = pl.read_parquet(data_root / "fx_quotes_minute_aggregated" / "symbol=EURUSD" / "year=2025" / "month=12" / "data.parquet")

# Scan a full symbol
df = pl.scan_parquet(data_root / "fx_1min" / "symbol=EURUSD").collect()
```

Hard-coded paths break on EC2 (Linux root is `/home/ec2-user/stock_data`).

### Known caveats / gaps

| Issue | Affects | Status |
|---|---|---|
| Polygon archive `2019-09` thin (4-5% density) | Multiple G10 in `fx_1min/` | Deferred Dukascopy patch |
| Polygon archive `2020-10`/`2020-11` EURUSD outage | `fx_1min/EURUSD/` | Deferred Dukascopy patch |
| USDCNH starts 2014-04 | `fx_1min/USDCNH/` | Polygon archive floor; structural |
| XAGUSD starts 2013-07 | `fx_1min/XAGUSD/` | Polygon archive floor; structural |
| SGDJPY / NOKJPY / SEKJPY / XAU+XAG crosses start 2020 | listed pairs | Polygon archive floor; structural |
| Pre-2017 thin minute coverage for several EM USD pairs | USDBRL, USDCLP, USDINR, USDKRW, USDRUB | Polygon archive; daily exists, minute doesn't |
| `fx_quotes_*` Tier 2/3 not pulled | mid-tier + EM pairs | YAGNI deferred — no consumer strategy yet |
| CME FX MBP-1 not pulled | 11 Tier-1 contracts | Databento billing block |
| `fx_quotes_minute_aggregated/quote_count` is UInt32, `fx_1min/trade_count` is Int64 | cross-dataset joins | Cast on read if needed; not blocking |
| `fx_1min_polygon_backfill/` legacy staging dir (780 KB, 3 symbols) | n/a | Pre-merger staging from 2026-04-21; safe to delete |

### Pipeline summary

```
Polygon S3 (minute_aggs_v1)  ->  massive_fx_flat.py        ->  fx_1min/
Polygon S3 (quotes_v1)       ->  massive_fx_quotes_flat.py ->  fx_quotes_raw/
fx_quotes_raw/               ->  aggregate_fx_quotes_to_minute.py
                                                            ->  fx_quotes_minute_aggregated/

Databento (CME FX)           ->  databento_futures.py      ->  futures_1min/ (selected symbols)
Alpaca (ETF universe)        ->  alpaca_equities.py        ->  equities_1min/ (FX-adjacent ETFs)
FRED                         ->  fred_rates.py             ->  alt_data/fred/
CFTC TFF archive             ->  cftc_cot.py               ->  alt_data/cot/
```

### Roadmap / deferred

- **Phase B Tier 2**: mid-tier quote universe (~25-30 pairs). YAGNI — no consumer.
- **Phase B Tier 3**: EM/Scandi quote universe (~30-40 pairs). YAGNI.
- **Phase C MBP-1**: 11 Tier-1 CME FX contracts. Blocked on Databento budget.
- **Phase F**: history extension to pre-2010. Would need a non-Polygon source (Dukascopy or HistData).
- **`fx_validation/`**: dedicated FX validation domain mirroring `src/data/validation/futures/`. Out of scope until a consumer strategy needs it.
- **dtype canonicalization**: migrate `fx_*` from `[ns, UTC]` to `[us, UTC]`. Separate migration plan.
