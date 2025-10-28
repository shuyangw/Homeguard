# Homeguard

## Components

### Data Engine

The Data Engine is a production-ready data ingestion and storage system for financial market data.

**Key Features:**
- Multithreaded data ingestion (up to 16x speedup)
- Partitioned Parquet storage with DuckDB backend
- Real-time progress tracking with thread visibility
- Support for multiple data feeds (SIP, IEX, OTC)
- OS-aware configuration (Windows, macOS, Linux)

[Read the Data Engine Documentation](src/data_engine/README.md)

### Backtesting Engine

*(Coming soon)*

## Quick Start

### 1. Installation

**Requirements:**
- Python 3.13.5 or higher
- Alpaca Markets API account

**Install dependencies:**
```bash
pip install -r requirements.txt
```

### 2. Configuration

**Set up environment variables:**
```bash
cp .env.example .env
# Edit .env and add your Alpaca API credentials
```

**Configure storage paths:**
```bash
cp settings.ini.example settings.ini
# Edit settings.ini and set your preferred data storage directory
```

See [SETUP.md](SETUP.md) for detailed configuration instructions.

### 3. Run Data Ingestion

```bash
cd src
python run_ingestion.py
```

This will download historical market data for configured symbols and store them in partitioned Parquet format.

## Project Structure

```
Homeguard/
├── src/
│   ├── data_engine/          # Data ingestion and storage
│   │   ├── api/              # API clients (Alpaca, etc.)
│   │   ├── storage/          # Parquet storage and metadata
│   │   ├── loaders/          # Symbol loaders
│   │   └── orchestration/    # Pipeline orchestration
│   ├── run_ingestion.py      # Data ingestion runner
│   └── config.py             # OS-aware configuration
├── docs/                      # Architecture documentation
├── backtest_lists/           # Symbol lists for ingestion
├── settings.ini.example      # Configuration template
├── .env.example              # API credentials template
└── requirements.txt          # Python dependencies
```

## Documentation

- [Setup Instructions](SETUP.md) - Initial configuration and setup
- [Data Engine Documentation](src/data_engine/README.md) - Detailed API and usage guide
- [Data Ingestion Pipeline Architecture](docs/DATA_INGESTION_PIPELINE.md) - System design and flow

## Features

- **High-Performance Data Ingestion**: Parallel processing with configurable workers
- **Efficient Storage**: Partitioned Parquet format optimized for time-series queries
- **Flexible Symbol Loading**: CSV, text files, or programmatic lists
- **Progress Monitoring**: Real-time tracking with time estimates
- **Error Resilience**: Per-symbol error isolation
- **Metadata Tracking**: Index membership and historical tracking
- **Cross-Platform**: Works on Windows, macOS, and Linux

## Development

### Running Tests

*(Coming soon)*

### Contributing

*(Coming soon)*

## License

*(Add your license here)*
    