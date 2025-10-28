# Data Ingestion Pipeline - Complete Reference

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DATA INGESTION PIPELINE                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────┐
│  Runner Script  │  run_ingestion.py (Entry Point)
│                 │  • ingest_single_symbol()
│                 │  • ingest_symbol_list()
│                 │  • ingest_from_csv()
└────────┬────────┘
         │
         ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                        ORCHESTRATION LAYER                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  IngestionPipeline                                                  │   │
│  │  • Manages multithreaded execution (ThreadPoolExecutor)             │   │
│  │  • Real-time progress tracking with time estimates                  │   │
│  │  • Coordinates all components                                       │   │
│  │  • Tracks success/failure per symbol                                │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└──────────┬────────────────────────┬────────────────────┬───────────────────┘
           │                        │                    │
           ▼                        ▼                    ▼
┌──────────────────────┐  ┌──────────────────┐  ┌─────────────────────┐
│   LOADER LAYER       │  │   API LAYER      │  │  STORAGE LAYER      │
│  SymbolLoader        │  │  AlpacaClient    │  │  ParquetStorage     │
│  • from_csv()        │  │  • fetch_bars()  │  │  • store()          │
│  • from_list()       │  │  • feed support  │  │  • DuckDB backend   │
│  • from_text_file()  │  │                  │  │                     │
│                      │  │                  │  │  MetadataStore      │
│                      │  │                  │  │  • store()          │
│                      │  │                  │  │  • load()           │
└──────────────────────┘  └──────────────────┘  └─────────────────────┘
```

## Component Details

### Configuration Layer
- **OS Detection**: Automatically detects Windows/macOS/Linux
- **Thread Configuration**:
  - Windows: 16 threads
  - macOS: 6 threads
  - Linux: 8 threads
- **Storage Paths**: OS-specific base directories from settings.ini

### API Layer (AlpacaClient)
- Fetches historical stock data from Alpaca API
- Supports multiple data feeds: SIP, IEX, OTC, DELAYED_SIP
- Defaults to account-level feed if not specified
- Returns pandas DataFrame with MultiIndex (symbol, timestamp)

### Loader Layer (SymbolLoader)
- Loads symbols from CSV files, lists, or text files
- Validates column existence and file paths
- Returns list of symbols for ingestion

### Storage Layer
**ParquetStorage**:
- Partitioned storage: `base_path/equities_1min/symbol=X/year=Y/month=Z/data.parquet`
- Uses DuckDB for efficient partitioned writes
- Timezone standardization (UTC)
- Idempotent operations (OVERWRITE_OR_IGNORE)

**MetadataStore**:
- Tracks index membership: `base_path/metadata/indices/{index_name}.csv`
- Stores symbol lists with timestamps
- Enables index reconstruction

### Orchestration Layer (IngestionPipeline)
- Initializes all components (API, storage, metadata)
- Manages ThreadPoolExecutor for parallel processing
- Real-time progress tracking with:
  - Completion percentage
  - Success/failure counts
  - Elapsed time
  - Estimated remaining time
- Error isolation (one symbol failure doesn't stop others)

## Logic Flow

### Entry Points

```python
# Option 1: Single Symbol
ingest_single_symbol('AAPL', '2016-01-01', '2025-01-01', TimeFrame.Minute)

# Option 2: Symbol List
ingest_symbol_list(['AAPL', 'GOOGL', 'MSFT'], '2016-01-01', '2025-01-01',
                   TimeFrame.Minute, index_name='tech_stocks')

# Option 3: CSV File
ingest_from_csv('sp500-2025.csv', '2016-01-01', '2025-01-01',
                TimeFrame.Minute, index_name='s_and_p_500')
```

### Execution Flow

#### 1. CSV Ingestion Path
```
ingest_from_csv(csv_path, start_date, end_date, timeframe, index_name)
  │
  ├─> Create IngestionPipeline instance
  │   ├─> AlpacaClient (with API keys)
  │   ├─> ParquetStorage (with OS-specific base path)
  │   └─> MetadataStore
  │
  └─> pipeline.ingest_from_csv()
      ├─> Derive index_name from filename if not provided
      ├─> SymbolLoader.from_csv() → Load symbols from CSV
      ├─> Print ingestion header
      ├─> Call ingest_symbols()
      └─> Print ingestion footer
```

#### 2. Symbol Ingestion Path
```
pipeline.ingest_symbols(symbols, start_date, end_date, timeframe, index_name)
  │
  ├─> If index_name: MetadataStore.store(index_name, symbols)
  │
  ├─> Initialize progress tracking (successful, failed, start_time)
  │
  ├─> Create ThreadPoolExecutor(max_workers)
  │   │
  │   ├─> Submit tasks: _fetch_and_store_symbol() for each symbol
  │   │
  │   └─> Process completed tasks:
  │       ├─> Track success/failure
  │       ├─> Calculate progress metrics
  │       ├─> Print progress update:
  │       │   • "Progress: 125/500 (25.0%)"
  │       │   • "Successful: 120 | Failed: 5 | Remaining: 375"
  │       │   • "Elapsed: 5m 30s | Est. remaining: 16m 15s"
  │       └─> Continue until all complete
  │
  ├─> Print total ingestion time
  ├─> Print summary (successful/failed breakdown)
  └─> Return {'successful': [...], 'failed': [...]}
```

#### 3. Worker Thread Execution
```
_fetch_and_store_symbol(symbol, start_date, end_date, timeframe)
  │
  ├─> AlpacaClient.fetch_bars(symbol, start_date, end_date, timeframe)
  │   ├─> Parse dates to datetime objects
  │   ├─> Determine data_feed (method → client → None/Alpaca default)
  │   ├─> Create StockBarsRequest
  │   ├─> Call Alpaca API: client.get_stock_bars()
  │   └─> Return DataFrame with OHLCV data
  │
  ├─> ParquetStorage.store(data, timeframe)
  │   ├─> Check if data is empty
  │   ├─> Preprocess: reset_index(), tz_convert('UTC'), add year/month
  │   ├─> Create storage_path: {base_path}/equities_1min
  │   ├─> DuckDB operations:
  │   │   ├─> Connect to in-memory database
  │   │   ├─> Register DataFrame as virtual table
  │   │   ├─> Execute COPY with PARTITION_BY (symbol, year, month)
  │   │   └─> Close connection
  │   └─> Result: symbol=X/year=Y/month=Z/data.parquet
  │
  └─> Return (symbol, True, None) or (symbol, False, error_msg)
```

## End-to-End Example

### Running: `python run_ingestion.py`

```
Step 1: Entry
  └─> main() → ingest_from_csv('sp500-2025.csv', '2016-01-01', '2025-01-01')

Step 2: Pipeline Initialization
  ├─> IngestionPipeline()
  ├─> AlpacaClient (feed=None, uses account default)
  ├─> ParquetStorage (base_path=F:\Stock_Data on Windows)
  └─> MetadataStore

Step 3: CSV Processing
  ├─> Derive index_name = 'sp500_2025'
  ├─> SymbolLoader.from_csv() → 500 symbols
  └─> pipeline.ingest_symbols()

Step 4: Metadata Storage
  └─> F:\Stock_Data\metadata\indices\sp500_2025.csv

Step 5: Parallel Ingestion (16 workers on Windows)
  ├─> Submit 500 tasks (one per symbol)
  ├─> Workers fetch from Alpaca + store to Parquet
  ├─> Progress updates every completion:
  │   • "Progress: 50/500 (10.0%)"
  │   • "Successful: 48 | Failed: 2 | Remaining: 450"
  │   • "Elapsed: 2m 15s | Est. remaining: 18m 30s"
  └─> Continue until all 500 complete

Step 6: Final Summary
  ├─> Total ingestion time: 22m 45s
  ├─> Successful: 495/500
  ├─> Failed: 5/500
  └─> List of failed symbols with error messages
```

## Storage Structure

```
F:\Stock_Data\                                  (Windows)
├── equities_1min\
│   ├── symbol=AAPL\
│   │   ├── year=2016\
│   │   │   ├── month=01\data.parquet
│   │   │   ├── month=02\data.parquet
│   │   │   └── month=03\data.parquet
│   │   ├── year=2017\
│   │   │   └── month=01\data.parquet
│   │   └── ... (up to year=2025)
│   ├── symbol=MSFT\
│   │   └── ... (same structure)
│   └── ... (500 symbols)
│
└── metadata\
    └── indices\
        └── sp500_2025.csv                      (symbols + added_date)
```

### Parquet File Schema
```
Column          Type                Description
──────────────────────────────────────────────────────────────────
timestamp       datetime64[ns,UTC]  Bar timestamp
open            float64             Opening price
high            float64             Highest price
low             float64             Lowest price
close           float64             Closing price
volume          float64             Trading volume
trade_count     float64             Number of trades
vwap            float64             Volume weighted average price
symbol          object              Stock symbol
year            int64               Year (partition key)
month           int64               Month (partition key)
```

### Metadata CSV Schema
```
Column          Type                Description
──────────────────────────────────────────────────────────────────
symbol          object              Stock symbol
added_date      datetime64[ns,UTC]  When added to index
```

## Configuration

### settings.ini
```ini
[windows]
local_storage_dir = F:\Stock_Data
api_threads = 16

[macos]
local_storage_dir = /Users/shuyangw/Library/CloudStorage/Dropbox/cs/stonk/data
api_threads = 6

[linux]
local_storage_dir = /home/user/stock_data
api_threads = 8
```

### OS Detection (config.py)
```python
platform.system().lower()
  ├─> 'darwin' → 'macos'
  ├─> 'windows' → 'windows'
  └─> 'linux' → 'linux'
```

## Performance

### Multithreading Impact
| Scenario | Sequential | Parallel (6 threads) | Parallel (16 threads) |
|----------|-----------|---------------------|----------------------|
| 500 symbols @ 2s each | 1000s (~16 min) | 167s (~3 min) | 63s (~1 min) |
| Speedup | 1x | ~6x | ~16x |

### Time Complexity
| Component | Operation | Complexity |
|-----------|-----------|-----------|
| SymbolLoader | Load CSV | O(n) |
| AlpacaClient | Fetch data | O(api_latency) |
| ParquetStorage | Store data | O(m) |
| IngestionPipeline | Process all | O(n/w × latency) |

*where n = symbols, m = bars, w = workers*

## Error Handling

### Per-Symbol Isolation
- Symbol failures tracked individually
- Failed symbols don't stop pipeline
- Error messages captured for debugging

### Graceful Degradation
- Empty DataFrames handled without crashing
- Missing files raise informative exceptions
- API errors caught and logged

### Final Reporting
```
Ingestion Summary:
  Successful: 495/500
  Failed: 5/500

Failed symbols:
  - GOOGL: API rate limit exceeded
  - BRK.B: No data available
  - XYZ: Invalid symbol
  - ABC: Connection timeout
  - DEF: Authentication error
```

## Usage Examples

### Basic Usage
```python
from data_engine import IngestionPipeline

pipeline = IngestionPipeline()
pipeline.ingest_from_csv("sp500-2025.csv", "2024-01-01", "2024-12-31")
```

### Custom Configuration
```python
from data_engine import IngestionPipeline
from alpaca.data import TimeFrame

pipeline = IngestionPipeline(max_workers=10)
result = pipeline.ingest_symbols(
    symbols=['AAPL', 'TSLA'],
    start_date="2020-01-01",
    end_date="2024-12-31",
    timeframe=TimeFrame.Day,
    index_name="my_portfolio"
)

print(f"Success rate: {len(result['successful'])/2*100:.1f}%")
```

### Modular Usage
```python
from data_engine import AlpacaClient, ParquetStorage

client = AlpacaClient()
storage = ParquetStorage()

data = client.fetch_bars("AAPL", "2024-01-01", "2024-12-31")
storage.store(data)
```

## Key Features

✅ **Multithreaded**: Parallel processing with configurable workers
✅ **Progress Tracking**: Real-time updates with time estimates
✅ **Error Resilient**: Per-symbol error isolation
✅ **Partitioned Storage**: Efficient query-optimized structure
✅ **Metadata Tracking**: Index membership and timestamps
✅ **OS-Aware**: Platform-specific configuration
✅ **Idempotent**: Safe to re-run without duplicates
✅ **Feed Flexible**: Support for multiple Alpaca data feeds

## Design Patterns

1. **Thread Pool Executor**: Parallel symbol processing
2. **Factory Pattern**: Component initialization in pipeline
3. **Template Method**: CSV/list methods use core ingestion
4. **Strategy Pattern**: Multiple ingestion entry points
5. **Partition Strategy**: Symbol/year/month for efficiency
