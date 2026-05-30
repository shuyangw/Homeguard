# Data Handling Guidelines

## Overview

This document covers the standardized data download framework for the Homeguard trading system. Market data downloads route through `src/data/acquisition/` via the `DataAcquisitionManager` plugin registry; the most common entry point is the `scripts/data/download_symbols.py` CLI (equities) or one of the per-asset-class downloaders alongside it.

## Quick Reference

### Download Data (Recommended Method)

```bash
# Download from CSV file
python scripts/data/download_symbols.py --csv config/universes/sp500-2025.csv --skip-existing

# Download specific symbols
python scripts/data/download_symbols.py --symbols AAPL,MSFT,GOOGL
```

The current CLI is 1-minute only. For hourly/daily, aggregate from the 1-min parquet store (see `scripts/data/aggregate_*` if applicable). Other asset classes have dedicated downloaders under `scripts/data/`.

### Programmatic Usage

```python
from src.data.acquisition import DataAcquisitionManager

manager = DataAcquisitionManager()
result = manager.download(
    source="equities",
    symbols=["AAPL", "MSFT"],
    start_date="2020-01-01",
    skip_existing=True,
)
print(f"Downloaded {result.total_rows} rows, {result.failed} failures")
```

## Storage Structure

### Directory Layout

Data is stored in Hive-partitioned format under `<asset_class>/<source>/<frequency>/`:

```
{local_storage_dir}/
├── equities/
│   ├── iex/1min/           # Alpaca IEX, raw
│   ├── sip_raw/1min/       # Alpaca SIP, raw (Algo Trader Plus)
│   └── sip_split/1min/     # Alpaca SIP, split-adjusted
├── crypto/
│   └── alpaca/{1min,1hour,1day}/
├── futures/
│   └── databento/{1min,mbp1,trades,...}/
├── fx/
│   ├── massive/{1min,quotes_raw,...}/
│   └── polygon/1min_backfill/
├── news/alpaca/
├── options/{chains,gex_daily,options_combined}/
└── alt_data/{fred,cot}/    # macro/positioning, not strict asset class

Per-symbol layout (same across all asset classes):
  <subdir>/symbol={SYM}/year={YYYY}/month={MM}/data.parquet
```

Canonical subdir constants live in `src/settings/data_paths.py` -- e.g. `EQUITIES_SIP_RAW_1MIN`, `CRYPTO_ALPACA_1MIN`, `FUTURES_DATABENTO_1MIN`. Always reference these instead of hardcoding strings.

### Platform-Specific Paths

| Platform | Path |
|----------|------|
| Windows | `H:\Stock_Data` |
| Linux/EC2 | `/home/ec2-user/stock_data` |

Always use `from src.settings import get_local_storage_dir` to get the correct path.

## Canonical Schema

All downloaded OHLCV data MUST match this schema exactly:

| Column | Type | Description |
|--------|------|-------------|
| `timestamp` | `datetime64[us, UTC]` | Bar timestamp (microsecond precision, UTC) |
| `open` | `float64` | Opening price |
| `high` | `float64` | High price |
| `low` | `float64` | Low price |
| `close` | `float64` | Closing price |
| `volume` | `float64` | Volume traded |
| `trade_count` | `float64` | Number of trades |
| `vwap` | `float64` | Volume-weighted average price |

### Schema Rules

1. Column names MUST be **lowercase** (`open`, not `Open`)
2. Include ALL 8 columns from Alpaca API
3. Do NOT rename or drop columns
4. Do NOT change dtypes (keep `volume` as `float64`)

## CLI Options

| Option | Description |
|--------|-------------|
| `--symbols, -s` | Comma-separated symbols: `AAPL,MSFT,GOOGL` |
| `--csv, -c` | CSV file with `Symbol` or `Ticker` column |
| `--file, -f` | Text file with one symbol per line |
| `--timeframe, -t` | `minute` (default), `hour`, or `day` |
| `--skip-existing` | Skip symbols already downloaded |
| `--start` | Start date: `YYYY-MM-DD` (default: 2017-01-01) |
| `--end` | End date: `YYYY-MM-DD` (default: today) |
| `--threads` | Parallel threads (default: 6) |

## Features

The download framework provides:

- **6 parallel download threads** for fast bulk downloads
- **3 retries per symbol** with exponential backoff
- **3 end-of-run retry rounds** for transient failures
- **Skip-existing mode** to avoid re-downloading
- **Canonical schema enforcement** for data consistency
- **Hive partitioned output** for efficient querying

## Symbol Lists

Available symbol lists in `config/universes/`:

| File | Description |
|------|-------------|
| `sp500-2025.csv` | S&P 500 symbols |
| `russell1000-2025.csv` | Russell 1000 symbols |
| `russell2000-2025.csv` | Russell 2000 symbols |
| `russell1000_non_sp500-2025.csv` | R1000 minus S&P 500 |
| `russell2000_non_r1000_sp500-2025.csv` | R2000 minus R1000 minus S&P 500 |

## Other Data Scripts

See `scripts/data/` for the current set of asset-class downloaders (FX, futures, options, crypto, news, FRED rates, COT, etc.). Each routes through the corresponding plugin in `src/data/acquisition/plugins/`.

## Common Tasks

### Download all R1000 + R2000 + S&P500

```bash
# Download all indices (skip existing to resume interrupted downloads)
python scripts/data/download_symbols.py --csv config/universes/russell1000-2025.csv --skip-existing
python scripts/data/download_symbols.py --csv config/universes/russell2000-2025.csv --skip-existing
python scripts/data/download_symbols.py --csv config/universes/sp500-2025.csv --skip-existing
```

### Update Russell Lists

```bash
# Re-download constituent lists from web sources
python scripts/download_russell_lists.py
```

### Check Download Status

```python
from src.data import AlpacaDownloader, Timeframe

downloader = AlpacaDownloader()
existing = downloader.get_existing_symbols(Timeframe.MINUTE)
print(f"Have {len(existing)} symbols downloaded")
```

## Error Handling

Failed symbols are automatically logged to:
```
{output_dir}/failed_symbols_{timeframe}.txt
```

Format: `SYMBOL,error_message`

Common failure reasons:
- `No data` - Symbol is delisted or has no Alpaca data
- API rate limits (handled by retry logic)
- Network timeouts (handled by retry logic)

## Unit Tests

Tests are in `tests/data/test_downloader.py`:

```bash
python -m pytest tests/data/test_downloader.py -v
```
