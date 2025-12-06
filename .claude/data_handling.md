# Data Handling Guidelines

## Overview

This document covers the standardized data download framework for the Homeguard trading system. All market data downloads should use the unified `AlpacaDownloader` class or the `download_symbols.py` CLI.

## Quick Reference

### Download Data (Recommended Method)

```bash
# Download from CSV file
python scripts/download_symbols.py --csv backtest_lists/sp500-2025.csv --skip-existing

# Download specific symbols
python scripts/download_symbols.py --symbols AAPL,MSFT,GOOGL

# Download hourly or daily data
python scripts/download_symbols.py --csv symbols.csv --timeframe hour
python scripts/download_symbols.py --csv symbols.csv --timeframe day
```

### Programmatic Usage

```python
from src.data import AlpacaDownloader, Timeframe

downloader = AlpacaDownloader(start_date='2020-01-01')
result = downloader.download_symbols(
    symbols=['AAPL', 'MSFT'],
    timeframe=Timeframe.MINUTE,
    skip_existing=True
)
print(f"Downloaded {result.total_bars} bars, {result.failed} failures")
```

## Storage Structure

### Directory Layout

Data is stored in Hive-partitioned format by timeframe:

```
{local_storage_dir}/
├── equities_1min/           # Minute data
│   └── symbol={SYMBOL}/
│       └── year={YYYY}/
│           └── month={MM}/
│               └── data.parquet
├── equities_1hour/          # Hourly data
│   └── ...
└── equities_1day/           # Daily data
    └── ...
```

### Platform-Specific Paths

| Platform | Path |
|----------|------|
| Windows | `F:\Stock_Data` |
| macOS | `/Users/shuyangw/Library/CloudStorage/Dropbox/cs/stonk/data` |
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

Available symbol lists in `backtest_lists/`:

| File | Description |
|------|-------------|
| `sp500-2025.csv` | S&P 500 symbols |
| `russell1000-2025.csv` | Russell 1000 symbols |
| `russell2000-2025.csv` | Russell 2000 symbols |
| `russell1000_non_sp500-2025.csv` | R1000 minus S&P 500 |
| `russell2000_non_r1000_sp500-2025.csv` | R2000 minus R1000 minus S&P 500 |

## Other Data Scripts

| Script | Purpose |
|--------|---------|
| `scripts/download_russell_lists.py` | Download Russell constituent lists from web |
| `backtest_scripts/download_leveraged_etfs.py` | Download daily leveraged ETF data via yfinance |

## Common Tasks

### Download all R1000 + R2000 + S&P500

```bash
# Download all indices (skip existing to resume interrupted downloads)
python scripts/download_symbols.py --csv backtest_lists/russell1000-2025.csv --skip-existing
python scripts/download_symbols.py --csv backtest_lists/russell2000-2025.csv --skip-existing
python scripts/download_symbols.py --csv backtest_lists/sp500-2025.csv --skip-existing
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
