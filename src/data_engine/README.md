# Data Engine

**A high-performance, modular data ingestion and storage system for quantitative trading and financial analysis.**

## Overview

The Data Engine is a production-ready framework designed to efficiently collect, process, and store large-scale financial market data. Built with scalability and reliability in mind, it provides a clean, extensible architecture for ingesting millions of bars of historical stock data from multiple sources and storing them in a query-optimized format.

### What It Does

- **Ingests** historical stock market data from Alpaca Markets API (with support for multiple data feeds: SIP, IEX, OTC)
- **Processes** data in parallel using multithreaded workers (configurable based on your system)
- **Stores** data in partitioned Parquet format using DuckDB for optimal query performance
- **Tracks** index membership and metadata for easy portfolio reconstruction
- **Monitors** progress in real-time with time estimates and success/failure metrics

### Key Features

- ✅ **Multithreaded Execution**: Parallel processing with 6-16 workers (OS-dependent) for up to 16x speedup
- ✅ **Progress Tracking**: Real-time updates with completion percentage, elapsed time, and ETA
- ✅ **Thread Visibility**: Every operation tagged with thread ID for debugging and performance analysis
- ✅ **Error Resilient**: Per-symbol error isolation ensures one failure doesn't stop the entire pipeline
- ✅ **Partitioned Storage**: Symbol/year/month partitioning for efficient querying and data management
- ✅ **Idempotent Operations**: Safe to re-run without creating duplicates (OVERWRITE_OR_IGNORE)
- ✅ **OS-Aware**: Automatic platform detection with optimized settings for Windows, macOS, and Linux
- ✅ **Feed Flexible**: Support for multiple Alpaca data feeds (SIP, IEX, OTC, DELAYED_SIP)
- ✅ **Metadata Tracking**: Automatic index membership tracking with timestamps

### Use Cases

- **Backtesting**: Historical data for algorithmic trading strategy development
- **Research**: Large-scale quantitative analysis of market behavior
- **Portfolio Analysis**: Track and analyze custom portfolios or market indices
- **Data Science**: Feature engineering for machine learning models
- **Real-time Systems**: Foundation for live trading systems (historical data layer)

### Performance

Ingest 500 symbols with 9 years of minute-level data:
- **Sequential**: ~16 minutes
- **Parallel (6 threads)**: ~3 minutes
- **Parallel (16 threads)**: ~1 minute
- **Speedup**: Up to **16x** faster with multithreading

### Architecture Philosophy

The Data Engine follows clean architecture principles with clear separation of concerns:

- **API Layer**: Handles communication with external data providers
- **Storage Layer**: Manages data persistence and metadata
- **Loader Layer**: Reads symbol lists from various sources
- **Orchestration Layer**: Coordinates multithreaded execution and progress tracking

This modular design makes it easy to:
- Add new data sources (Yahoo Finance, IEX, etc.)
- Swap storage backends (S3, databases, etc.)
- Extend functionality without breaking existing code

## Architecture

The data engine is organized into separate, well-defined modules following the separation of concerns principle:

```
src/data_engine/
├── __init__.py                          # Package initialization, public API
├── api/
│   ├── __init__.py
│   └── alpaca_client.py                 # Alpaca API client wrapper
├── storage/
│   ├── __init__.py
│   ├── parquet_storage.py               # DuckDB/Parquet storage operations
│   └── metadata_store.py                # Metadata management for indices
├── loaders/
│   ├── __init__.py
│   └── symbol_loader.py                 # CSV/file loading for symbols
├── orchestration/
│   ├── __init__.py
│   └── ingestion_pipeline.py            # Multithreaded ingestion orchestration
└── alpaca_api_data.py                   # [DEPRECATED] Backward compatibility wrapper
```

## Quick Start

### Running the Ingestion Pipeline

The easiest way to run the data ingestion pipeline is using the runner script:

```bash
cd src
python run_ingestion.py
```

This will ingest all S&P 500 symbols from the CSV file into the configured storage directory.

### Using the Pipeline Programmatically

```python
from data_engine import IngestionPipeline
from alpaca.data import TimeFrame

# Initialize the pipeline
pipeline = IngestionPipeline()

# Ingest from a CSV file
pipeline.ingest_from_csv(
    csv_path="backtest_lists/sp500-2025.csv",
    start_date="2016-01-01",
    end_date="2025-01-01",
    timeframe=TimeFrame.Minute,
    index_name="s_and_p_500"
)

# Or ingest a specific list of symbols
symbols = ["AAPL", "MSFT", "GOOGL"]
pipeline.ingest_symbols(
    symbols=symbols,
    start_date="2020-01-01",
    end_date="2025-01-01",
    timeframe=TimeFrame.Day
)
```

## Module Usage

### API Client

```python
from data_engine.api import AlpacaClient

# Initialize client
client = AlpacaClient()

# Fetch data for a single symbol
data = client.fetch_bars("AAPL", "2024-01-01", "2024-12-31")
```

### Storage

```python
from data_engine.storage import ParquetStorage

# Initialize storage
storage = ParquetStorage()

# Store data
storage.store(dataframe, timeframe=TimeFrame.Minute)
```

### Metadata

```python
from data_engine.storage import MetadataStore

# Initialize metadata store
metadata = MetadataStore()

# Store index metadata
metadata.store("s_and_p_500", ["AAPL", "MSFT", "GOOGL"])

# Load index metadata
symbols_df = metadata.load("s_and_p_500")
```

### Symbol Loading

```python
from data_engine.loaders import SymbolLoader

# Load from CSV
symbols = SymbolLoader.from_csv("sp500-2025.csv")

# Load from text file (one symbol per line)
symbols = SymbolLoader.from_text_file("my_symbols.txt")

# Use a predefined list
symbols = SymbolLoader.from_list(["AAPL", "MSFT", "GOOGL"])
```

### Pipeline Orchestration

```python
from data_engine.orchestration import IngestionPipeline

# Initialize with custom thread count
pipeline = IngestionPipeline(max_workers=10)

# Ingest data
result = pipeline.ingest_symbols(
    symbols=["AAPL", "MSFT"],
    start_date="2024-01-01",
    end_date="2024-12-31"
)

# Check results
print(f"Successful: {result['successful']}")
print(f"Failed: {result['failed']}")
```

## Configuration

The data engine uses OS-aware configuration from `settings.ini`:

```ini
[macos]
local_storage_dir = /path/to/data
api_threads = 6

[windows]
local_storage_dir = F:\Stock_Data
api_threads = 8

[linux]
local_storage_dir = /home/user/stock_data
api_threads = 8
```

The OS is automatically detected using the `config.py` module.

## Features

- **Multithreaded ingestion**: Configurable number of threads based on OS
- **Partitioned storage**: Data stored in Parquet format with symbol/year/month partitioning
- **Metadata tracking**: Automatic metadata storage for indices
- **OS-aware**: Automatically detects OS and uses appropriate settings
- **Modular design**: Easy to extend with new data sources or storage backends
- **Backward compatible**: Original API still works through wrapper functions

## Data Storage Format

Data is stored in a partitioned Parquet format:

```
{base_path}/
├── equities_1min/
│   ├── symbol=AAPL/
│   │   ├── year=2024/
│   │   │   ├── month=01/
│   │   │   │   └── data.parquet
│   │   │   ├── month=02/
│   │   │   │   └── data.parquet
│   └── symbol=MSFT/
│       └── ...
└── metadata/
    └── indices/
        ├── s_and_p_500.csv
        └── nasdaq_100.csv
```

## Backward Compatibility

The original `alpaca_api_data.py` file has been converted to a wrapper that re-exports all functions. Existing code will continue to work:

```python
# Old code still works
from data_engine.alpaca_api_data import fetch_data, store_data

# But this is preferred for new code
from data_engine.api import AlpacaClient
from data_engine.storage import ParquetStorage
```

## Adding New Data Sources

To add a new data source (e.g., Yahoo Finance, IEX):

1. Create a new client in `api/` (e.g., `yahoo_client.py`)
2. Implement a similar interface to `AlpacaClient`
3. Update the orchestration pipeline to support the new source

Example:

```python
# data_engine/api/yahoo_client.py
class YahooClient:
    def fetch_bars(self, symbol, start_date, end_date, timeframe):
        # Implementation here
        pass
```
