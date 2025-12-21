# Homeguard Data Flow Documentation

**Version**: 1.1
**Last Updated**: 2025-12-15
**Purpose**: Comprehensive data flow diagrams for all system operations

---

## Table of Contents

1. [Data Ingestion Flow](#data-ingestion-flow)
2. [Streaming Data Flow](#streaming-data-flow) (NEW)
3. [News & Sentiment Pipeline Flow](#news--sentiment-pipeline-flow) (NEW)
4. [Single Symbol Backtest Flow](#single-symbol-backtest-flow)
5. [Multi-Symbol Sweep Flow](#multi-symbol-sweep-flow)
6. [Multi-Asset Portfolio Flow](#multi-asset-portfolio-flow)
7. [GUI Backtest Flow](#gui-backtest-flow)
8. [Live Trading Execution Flow](#live-trading-execution-flow) (NEW)
9. [Visualization Pipeline Flow](#visualization-pipeline-flow)
10. [Risk Management Flow](#risk-management-flow)
11. [Signal Generation Flow](#signal-generation-flow)

---

## Data Ingestion Flow

### High-Level Overview

```
User Input (Symbol List, Date Range)
    v
IngestionPipeline
    ├─-> SymbolLoader (load symbol list)
    ├─-> ThreadPoolExecutor (spawn workers)
    │   ├─ Worker 1: AlpacaClient -> ParquetStorage (AAPL)
    │   ├─ Worker 2: AlpacaClient -> ParquetStorage (MSFT)
    │   ├─ Worker 3: AlpacaClient -> ParquetStorage (GOOGL)
    │   └─ Worker 4: AlpacaClient -> ParquetStorage (TSLA)
    └─-> MetadataStore (update metadata)
        v
Data stored in Parquet files
```

### Detailed Flow

```
┌──────────────────────────────────────────────────────────────┐
│ 1. USER INPUT                                                │
│    - Symbol list: ['AAPL', 'MSFT', 'GOOGL', 'TSLA']        │
│    - Date range: 2023-01-01 to 2024-01-01                  │
│    - Timeframe: 1Min                                        │
└────────────────┬─────────────────────────────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────────────────────────────┐
│ 2. INGESTION PIPELINE (src/run_ingestion.py)                │
│    pipeline = IngestionPipeline(data_dir='data/')           │
│    pipeline.run(symbols, start, end, timeframe, workers=4)  │
└────────────────┬─────────────────────────────────────────────┘
                 │
                 ├─-> Load symbol list
                 │   SymbolLoader.load_from_csv('symbols.csv')
                 │
                 ├─-> Initialize workers
                 │   ThreadPoolExecutor(max_workers=4)
                 │
                 └─-> Spawn parallel workers
                      │
      ┌───────────────┼───────────────┬───────────────┐
      │               │               │               │
      ▼               ▼               ▼               ▼
┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│  WORKER 1   │ │  WORKER 2   │ │  WORKER 3   │ │  WORKER 4   │
│   (AAPL)    │ │   (MSFT)    │ │  (GOOGL)    │ │   (TSLA)    │
└──────┬──────┘ └──────┬──────┘ └──────┬──────┘ └──────┬──────┘
       │               │               │               │
       ▼               ▼               ▼               ▼
┌──────────────────────────────────────────────────────────────┐
│ 3. ALPACA CLIENT (data_engine/api/alpaca_client.py)         │
│    client.fetch_bars(symbol, start, end, '1Min')            │
│                                                              │
│    API Request:                                              │
│    GET /v2/stocks/{symbol}/bars                             │
│    ?start=2023-01-01&end=2024-01-01&timeframe=1Min          │
└────────────────┬─────────────────────────────────────────────┘
                 │
                 ▼ (Rate limited: 200 req/min, auto-retry)
┌──────────────────────────────────────────────────────────────┐
│ 4. ALPACA API RESPONSE                                       │
│    {                                                         │
│      "bars": [                                               │
│        {                                                     │
│          "t": "2023-01-03T09:30:00Z",                       │
│          "o": 125.50, "h": 126.00,                          │
│          "l": 125.40, "c": 125.80,                          │
│          "v": 1000000                                        │
│        },                                                    │
│        ...                                                   │
│      ]                                                       │
│    }                                                         │
└────────────────┬─────────────────────────────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────────────────────────────┐
│ 5. CONVERT TO DATAFRAME                                      │
│    bars_df = pd.DataFrame(bars)                              │
│    bars_df.set_index('timestamp', inplace=True)             │
│                                                              │
│    Result: DataFrame with columns [open, high, low, close, volume]
│            Index: DatetimeIndex                              │
└────────────────┬─────────────────────────────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────────────────────────────┐
│ 6. PARQUET STORAGE (data_engine/storage/parquet_storage.py) │
│    storage.save_bars(symbol, bars_df, '1Min')               │
│                                                              │
│    Writes to:                                                │
│    data/equities_1min/AAPL/2023-01-03.parquet               │
│    data/equities_1min/AAPL/2023-01-04.parquet               │
│    ...                                                       │
│                                                              │
│    Compression: Snappy (default)                             │
│    Partition: By symbol, then by date                        │
└────────────────┬─────────────────────────────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────────────────────────────┐
│ 7. METADATA UPDATE (data_engine/storage/metadata_store.py)  │
│    store.update_symbol_metadata(                             │
│        symbol='AAPL',                                        │
│        start_date='2023-01-03',                              │
│        end_date='2024-01-01',                                │
│        timeframe='1Min'                                      │
│    )                                                         │
│                                                              │
│    Writes to: data/metadata.json                             │
└────────────────┬─────────────────────────────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────────────────────────────┐
│ 8. COMPLETION CALLBACK                                       │
│    on_symbol_complete(symbol='AAPL', status='success')      │
│    Update progress: 1/4 complete (25%)                       │
└──────────────────────────────────────────────────────────────┘

All workers complete -> Pipeline finished -> Data ready for backtesting
```

### Error Handling Flow

```
AlpacaClient.fetch_bars(symbol)
    │
    ├─-> Success -> Return data
    │
    ├─-> ConnectionError
    │   └─-> Retry (3 attempts, exponential backoff)
    │       ├─-> Success -> Return data
    │       └─-> Fail -> Log error, skip symbol, continue
    │
    ├─-> AuthenticationError
    │   └─-> Fail fast, abort pipeline
    │
    └─-> RateLimitError
        └─-> Sleep 60 seconds, retry
```

---

## Streaming Data Flow

### Real-Time Market Data via WebSocket

The streaming platform provides 32x faster data access compared to REST polling.

### High-Level Overview

```
WebSocket Connection (Alpaca)
    |
StreamManager
    |-> Subscribe to symbols (AAPL, MSFT, ...)
    |-> Receive real-time bars/quotes/trades
    |
BarBuffer (in-memory cache)
    |-> Store last 500 bars per symbol
    |-> LRU eviction for memory management
    |
LiveDataProvider (public API)
    |-> get_price(symbol) -> float
    |-> get_bars(symbol, n) -> DataFrame
    |-> get_quote(symbol) -> Quote
    |-> get_vwap(symbol) -> float
    |
Strategy Adapters (OMR, RAMP)
    |-> Real-time signal generation
```

### Detailed Flow

```
+--------------------------------------------------------------+
| 1. INITIALIZATION (live_data_provider.py)                    |
|    provider = LiveDataProvider(                              |
|        symbols=['TQQQ', 'SOXL', ...],                        |
|        feed='iex'  # or 'sip' for paid feed                  |
|    )                                                         |
|    provider.start()                                          |
+--------------------------------------------------------------+
                 |
                 v
+--------------------------------------------------------------+
| 2. WEBSOCKET CONNECTION (streaming/_stream.py)               |
|    stream = StreamManager(                                   |
|        api_key=ALPACA_API_KEY,                               |
|        feed='iex'                                            |
|    )                                                         |
|    stream.subscribe_bars(symbols)                            |
|    stream.subscribe_quotes(symbols)                          |
|                                                              |
|    WebSocket URL:                                            |
|    wss://stream.data.alpaca.markets/v2/{feed}               |
+--------------------------------------------------------------+
                 |
                 v (real-time messages)
+--------------------------------------------------------------+
| 3. MESSAGE HANDLING                                          |
|    on_bar(bar: Bar):                                         |
|        symbol = bar.symbol                                   |
|        buffer.add_bar(symbol, bar)                           |
|                                                              |
|    on_quote(quote: Quote):                                   |
|        symbol = quote.symbol                                 |
|        buffer.update_quote(symbol, quote)                    |
|                                                              |
|    Bar format:                                               |
|    {symbol, timestamp, open, high, low, close, volume, vwap} |
+--------------------------------------------------------------+
                 |
                 v
+--------------------------------------------------------------+
| 4. BAR BUFFER (streaming/_buffer.py)                         |
|    buffer = BarBuffer(max_bars=500)                          |
|                                                              |
|    Internal storage:                                         |
|    {                                                         |
|        'TQQQ': deque([bar1, bar2, ..., bar500]),            |
|        'SOXL': deque([bar1, bar2, ..., bar500]),            |
|        ...                                                   |
|    }                                                         |
|                                                              |
|    Methods:                                                  |
|    - add_bar(symbol, bar)                                    |
|    - get_bars(symbol, n) -> list[Bar]                        |
|    - get_latest(symbol) -> Bar                               |
+--------------------------------------------------------------+
                 |
                 v
+--------------------------------------------------------------+
| 5. FALLBACK MECHANISM (streaming/_fallback.py)               |
|    if buffer.coverage(symbol) < 0.9:  # Less than 90% filled |
|        # Fall back to REST API                               |
|        bars = alpaca_broker.get_historical_bars(             |
|            symbol, limit=n                                   |
|        )                                                     |
|        return bars                                           |
|                                                              |
|    Triggers:                                                 |
|    - Mid-day restart (buffer not filled)                     |
|    - WebSocket disconnect                                    |
|    - Before market open (no streaming data yet)              |
+--------------------------------------------------------------+
                 |
                 v
+--------------------------------------------------------------+
| 6. STRATEGY CONSUMPTION                                      |
|    # OMR Adapter                                             |
|    price = provider.get_price('TQQQ')                        |
|    bars = provider.get_bars('TQQQ', 10)                      |
|                                                              |
|    # RAMP Adapter (500 symbols)                              |
|    for symbol in sp500:                                      |
|        close = provider.get_price(symbol)                    |
|        closes[symbol] = close                                |
|                                                              |
|    Performance: 150s -> 0.5s for 500 symbols                 |
+--------------------------------------------------------------+
```

### Feature Flag Control

```
Environment Variables:
    USE_STREAMING=true     # Enable streaming (default: false)
    STREAMING_FEED=iex     # Feed type: 'iex' or 'sip'

Behavior:
    if USE_STREAMING == 'true':
        provider = LiveDataProvider(...)  # WebSocket
    else:
        provider = CompositeDataProvider(...)  # REST API
```

---

## News & Sentiment Pipeline Flow

### Alpaca News API to Sentiment Scores

```
Alpaca News API
    |
NewsDownloader
    |-> Parallel download by symbol
    |-> Thread-safe with ThreadPoolExecutor
    |
Parquet Storage
    |-> news/symbol={SYMBOL}/year={YYYY}/news.parquet
    |
SentimentAnalyzer (FinBERT)
    |-> Process headlines
    |-> Score: -1 (negative) to +1 (positive)
    |
SentimentCache
    |-> sentiment/symbol={SYMBOL}/year={YYYY}/sentiment.parquet
    |
Strategy Filter
    |-> Filter trades based on sentiment
```

### Detailed Flow

```
+--------------------------------------------------------------+
| 1. NEWS DOWNLOAD (data/news/news_downloader.py)              |
|    downloader = NewsDownloader()                             |
|    result = downloader.download_symbol(                      |
|        symbol='AAPL',                                        |
|        start_date='2024-01-01',                              |
|        end_date='2024-12-31'                                 |
|    )                                                         |
|                                                              |
|    API Request:                                              |
|    GET /v1beta1/news                                         |
|    ?symbols=AAPL&start=2024-01-01&end=2024-12-31            |
+--------------------------------------------------------------+
                 |
                 v
+--------------------------------------------------------------+
| 2. NEWS SCHEMA (data/news/news_schema.py)                    |
|    Canonical schema:                                         |
|    - id: string (unique article ID)                          |
|    - timestamp: datetime64[us, UTC]                          |
|    - symbol: string                                          |
|    - headline: string                                        |
|    - summary: string (optional)                              |
|    - source: string (e.g., 'Benzinga')                       |
|    - url: string                                             |
|    - author: string (optional)                               |
+--------------------------------------------------------------+
                 |
                 v
+--------------------------------------------------------------+
| 3. PARQUET STORAGE                                           |
|    Path: {storage}/news/symbol=AAPL/year=2024/news.parquet  |
|                                                              |
|    Hive partitioning for efficient queries:                  |
|    news/                                                     |
|    +-- symbol=AAPL/                                          |
|    |   +-- year=2024/                                        |
|    |       +-- news.parquet                                  |
|    +-- symbol=MSFT/                                          |
|        +-- year=2024/                                        |
|            +-- news.parquet                                  |
+--------------------------------------------------------------+
                 |
                 v
+--------------------------------------------------------------+
| 4. SENTIMENT ANALYSIS (data/news/sentiment_analyzer.py)      |
|    analyzer = SentimentAnalyzer()                            |
|                                                              |
|    For each headline:                                        |
|    1. Tokenize with FinBERT tokenizer                        |
|    2. Run through FinBERT model                              |
|    3. Extract probabilities: [negative, neutral, positive]   |
|    4. Calculate score: positive_prob - negative_prob         |
|                                                              |
|    Result:                                                   |
|    {                                                         |
|        'timestamp': '2024-01-11 08:30:00',                   |
|        'headline': 'Bitcoin ETF Approved...',                |
|        'sentiment_score': 0.65,  # Positive                  |
|        'confidence': 0.82                                    |
|    }                                                         |
+--------------------------------------------------------------+
                 |
                 v
+--------------------------------------------------------------+
| 5. SENTIMENT CACHE (data/news/sentiment_cache.py)            |
|    Path: {storage}/sentiment/symbol=AAPL/year=2024/          |
|          sentiment.parquet                                   |
|                                                              |
|    Schema:                                                   |
|    - timestamp: datetime64[us, UTC]                          |
|    - headline: string                                        |
|    - sentiment_score: float64 (-1 to +1)                     |
|    - confidence: float64 (0 to 1)                            |
+--------------------------------------------------------------+
                 |
                 v
+--------------------------------------------------------------+
| 6. STRATEGY USAGE                                            |
|    # In HV ORB Strategy                                      |
|    if use_sentiment:                                         |
|        # Get premarket sentiment (prev day 4PM to 9:30AM)    |
|        sentiment = get_premarket_sentiment(symbol, date)     |
|                                                              |
|        # Filter: Skip if negative sentiment on gap-up        |
|        if gap_pct > 0 and sentiment.avg_score < -0.1:        |
|            skip_trade = True  # Divergence detected          |
|                                                              |
|        # Or: Require positive sentiment                      |
|        if sentiment.avg_score < min_sentiment_score:         |
|            skip_trade = True                                 |
+--------------------------------------------------------------+
```

### News Timing Distribution

```
News Publication Times (Eastern Time):
    04:00-09:30  Premarket    30.2%  <- Available before ORB entry
    09:30-16:00  Market Hours 54.5%
    16:00-20:00  After Hours  10.8%
    20:00-04:00  Overnight     4.5%

Key Insight: ~35% of news available before 9:35 AM ORB entry
```

---

## Single Symbol Backtest Flow

### High-Level Overview

```
CLI Command
    v
BacktestEngine
    ├─-> DataLoader (load OHLCV)
    ├─-> Strategy.generate_signals()
    ├─-> PortfolioSimulator (execute trades)
    └─-> Return Portfolio
        v
BacktestVisualizer (charts, tearsheet, reports)
    v
Output files (HTML, PNG, CSV)
```

### Detailed Flow

```
┌──────────────────────────────────────────────────────────────┐
│ 1. CLI COMMAND                                               │
│    python -m src.backtest_runner \                          │
│      --strategy MovingAverageCrossover \                     │
│      --symbols AAPL \                                        │
│      --start 2023-01-01 --end 2024-01-01                    │
└────────────────┬─────────────────────────────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────────────────────────────┐
│ 2. BACKTEST ENGINE INITIALIZATION                            │
│    engine = BacktestEngine(                                  │
│        initial_capital=100000,                               │
│        fees=0.001,                                           │
│        slippage=0.001,                                       │
│        risk_config=RiskConfig.moderate()                     │
│    )                                                         │
└────────────────┬─────────────────────────────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────────────────────────────┐
│ 3. DATA LOADING (backtesting/engine/data_loader.py)         │
│    loader = DataLoader(data_dir='data/')                     │
│    data = loader.load_bars('AAPL', '2023-01-01', '2024-01-01', '1Min')
│                                                              │
│    Steps:                                                    │
│    ├─-> Query Parquet files via DuckDB SQL                   │
│    │   SELECT * FROM 'data/equities_1min/AAPL/*.parquet'    │
│    │   WHERE timestamp >= '2023-01-01'                       │
│    │   AND timestamp < '2024-01-01'                          │
│    │                                                          │
│    ├─-> Load into DataFrame (pandas)                          │
│    │                                                          │
│    ├─-> Market calendar filtering                             │
│    │   calendar.filter_market_days(data)                     │
│    │   (removes weekends, NYSE holidays)                     │
│    │                                                          │
│    └─-> Market hours filtering (9:35 AM - 3:55 PM)           │
│        (optional, if market_hours_only=True)                 │
│                                                              │
│    Result: DataFrame[timestamp, open, high, low, close, volume]
│            ~98,000 bars (1 year of 1-min data)               │
└────────────────┬─────────────────────────────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────────────────────────────┐
│ 4. DATA VALIDATION (backtest_engine.py)                     │
│    ├─-> Check for duplicate timestamps                       │
│    │   if data.index.duplicated().any():                    │
│    │       data = data[~data.index.duplicated(keep='first')]│
│    │                                                          │
│    ├─-> Check for NaN values                                 │
│    │   data.dropna(inplace=True)  # or forward fill         │
│    │                                                          │
│    └─-> Validate OHLC relationships                           │
│        assert (data['high'] >= data['low']).all()            │
│                                                              │
│    Clean data ready for strategy                             │
└────────────────┬─────────────────────────────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────────────────────────────┐
│ 5. STRATEGY INITIALIZATION                                   │
│    strategy = MovingAverageCrossover(                        │
│        fast_window=20,                                       │
│        slow_window=50                                        │
│    )                                                         │
└────────────────┬─────────────────────────────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────────────────────────────┐
│ 6. SIGNAL GENERATION (strategy.generate_signals)            │
│    entries, exits = strategy.generate_signals(data)         │
│                                                              │
│    Process:                                                  │
│    ├─-> Calculate indicators                                  │
│    │   fast_ma = sma(data['close'], window=20)              │
│    │   slow_ma = sma(data['close'], window=50)              │
│    │                                                          │
│    ├─-> Generate entry signals                                │
│    │   entries = (fast_ma > slow_ma) & (fast_ma.shift(1) <= slow_ma.shift(1))
│    │                                                          │
│    └─-> Generate exit signals                                 │
│        exits = (fast_ma < slow_ma) & (fast_ma.shift(1) >= slow_ma.shift(1))
│                                                              │
│    Result: Two boolean Series                                │
│            entries[i] = True -> Enter long on bar i           │
│            exits[i] = True -> Exit long on bar i              │
└────────────────┬─────────────────────────────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────────────────────────────┐
│ 7. PORTFOLIO SIMULATION (portfolio_simulator.py)            │
│    portfolio = from_signals(                                 │
│        close=data['close'],                                  │
│        entries=entries,                                      │
│        exits=exits,                                          │
│        init_cash=100000,                                     │
│        fees=0.001,                                           │
│        slippage=0.001,                                       │
│        risk_config=RiskConfig.moderate(),                    │
│        price_data=data  # For OHLC-aware fills               │
│    )                                                         │
│                                                              │
│    See: Portfolio Simulation Flow (detailed below)           │
└────────────────┬─────────────────────────────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────────────────────────────┐
│ 8. PORTFOLIO RESULT                                          │
│    Portfolio object with:                                    │
│    - equity_curve: pd.Series (portfolio value over time)     │
│    - trades: List[Dict] (all executed trades)                │
│    - positions: List (current open positions)                │
│    - cash: float (current available cash)                    │
│                                                              │
│    stats = portfolio.stats()                                 │
│    {                                                         │
│        'Total Return [%]': 25.34,                            │
│        'Sharpe Ratio': 1.87,                                 │
│        'Max Drawdown [%]': -12.45,                           │
│        'Win Rate [%]': 58.3,                                 │
│        'Total Trades': 24                                    │
│    }                                                         │
└────────────────┬─────────────────────────────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────────────────────────────┐
│ 9. VISUALIZATION (visualization/integration.py)              │
│    visualizer = BacktestVisualizer()                         │
│    visualizer.generate_all(portfolio, strategy, data)       │
│                                                              │
│    Generates:                                                │
│    ├─-> QuantStats Tearsheet (HTML/PDF)                      │
│    ├─-> Candlestick Chart with trade markers (PNG)           │
│    ├─-> Trade log (CSV)                                      │
│    └─-> Summary report (TXT)                                 │
└────────────────┬─────────────────────────────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────────────────────────────┐
│ 10. OUTPUT FILES                                             │
│     logs/AAPL_20250105_153045/                              │
│     ├── tearsheet.html                                       │
│     ├── candlestick_chart.png                                │
│     ├── trade_log.csv                                        │
│     └── summary.txt                                          │
└──────────────────────────────────────────────────────────────┘
```

---

## Multi-Symbol Sweep Flow

### High-Level Overview

```
CLI Command (multiple symbols)
    v
SweepRunner
    ├─-> ThreadPoolExecutor (4 workers)
    │   ├─ Worker 1: BacktestEngine('AAPL')
    │   ├─ Worker 2: BacktestEngine('MSFT')
    │   ├─ Worker 3: BacktestEngine('GOOGL')
    │   └─ Worker 4: BacktestEngine('TSLA')
    │       (each follows Single Symbol Backtest Flow)
    │
    └─-> ResultsAggregator
        v
    Combined reports (comparison tables, aggregate metrics)
```

### Detailed Parallel Execution Flow

```
┌──────────────────────────────────────────────────────────────┐
│ 1. CLI COMMAND (Multiple Symbols)                           │
│    python -m src.backtest_runner \                          │
│      --strategy MovingAverageCrossover \                     │
│      --symbols AAPL,MSFT,GOOGL,TSLA \                       │
│      --start 2023-01-01 --end 2024-01-01 \                  │
│      --workers 4                                             │
└────────────────┬─────────────────────────────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────────────────────────────┐
│ 2. SWEEP RUNNER INITIALIZATION                               │
│    runner = SweepRunner(                                     │
│        engine=BacktestEngine(...),                           │
│        workers=4,                                            │
│        fail_fast=False                                       │
│    )                                                         │
│                                                              │
│    runner.set_callbacks(                                     │
│        on_start=callback_start,                              │
│        on_complete=callback_complete,                        │
│        on_error=callback_error                               │
│    )                                                         │
└────────────────┬─────────────────────────────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────────────────────────────┐
│ 3. THREAD POOL EXECUTOR                                      │
│    executor = ThreadPoolExecutor(max_workers=4)              │
│                                                              │
│    Submit 4 tasks:                                           │
│    ├─ future_1 = executor.submit(run_backtest, 'AAPL')      │
│    ├─ future_2 = executor.submit(run_backtest, 'MSFT')      │
│    ├─ future_3 = executor.submit(run_backtest, 'GOOGL')     │
│    └─ future_4 = executor.submit(run_backtest, 'TSLA')      │
└────────────────┬─────────────────────────────────────────────┘
                 │
     ┌───────────┼───────────┬───────────┬───────────┐
     │           │           │           │           │
     ▼           ▼           ▼           ▼           ▼
┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐
│ WORKER 1│ │ WORKER 2│ │ WORKER 3│ │ WORKER 4│ │  MAIN   │
│  AAPL   │ │  MSFT   │ │ GOOGL   │ │  TSLA   │ │ THREAD  │
└────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘
     │           │           │           │           │
     │ T+0s      │           │           │           │
     ├─-> on_start('AAPL')───────────────────────────->│
     │           │ T+0.1s    │           │           │
     │           ├─-> on_start('MSFT')────────────────->│
     │           │           │ T+0.1s    │           │
     │           │           ├─-> on_start('GOOGL')───->│
     │           │           │           │ T+0.1s    │
     │           │           │           ├─-> on_start('TSLA')
     │           │           │           │           │
     ├── Load data (DuckDB)                          │
     ├── Generate signals                            │
     ├── Simulate portfolio                          │
     │           │           │           │           │
     │ T+3.2s    │           │           │           │
     ├─-> on_complete('AAPL', portfolio)──────────────->│
     │           │           │           │           │
     │           │ T+3.5s    │           │           │
     │           ├─-> on_complete('MSFT', portfolio)──->│
     │           │           │           │           │
     │           │           │ T+4.1s    │           │
     │           │           ├─-> on_complete('GOOGL', portfolio)->│
     │           │           │           │           │
     │           │           │           │ T+3.8s    │
     │           │           │           ├─-> on_complete('TSLA', portfolio)
     │           │           │           │           │
     └───────────┴───────────┴───────────┴───────────┘
                                                     │
                            All futures complete ────┘
                                                     │
                                                     ▼
┌──────────────────────────────────────────────────────────────┐
│ 4. RESULTS AGGREGATION (results_aggregator.py)              │
│    aggregator = ResultsAggregator()                          │
│    aggregator.add_result('AAPL', portfolio_aapl)            │
│    aggregator.add_result('MSFT', portfolio_msft)            │
│    aggregator.add_result('GOOGL', portfolio_googl)          │
│    aggregator.add_result('TSLA', portfolio_tsla)            │
│                                                              │
│    summary = aggregator.generate_summary()                   │
│    {                                                         │
│        'AAPL': {'return': 25.3%, 'sharpe': 1.87, ...},      │
│        'MSFT': {'return': 18.2%, 'sharpe': 1.42, ...},      │
│        'GOOGL': {'return': 32.1%, 'sharpe': 2.15, ...},     │
│        'TSLA': {'return': -5.4%, 'sharpe': -0.32, ...}      │
│    }                                                         │
└────────────────┬─────────────────────────────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────────────────────────────┐
│ 5. COMPARISON REPORTS                                        │
│    ├─-> Comparison table (CSV)                                │
│    │   Symbol | Return | Sharpe | Max DD | Trades            │
│    │   -------|--------|--------|--------|-------            │
│    │   AAPL   | 25.3%  | 1.87   | -12.4% | 24                │
│    │   MSFT   | 18.2%  | 1.42   | -8.7%  | 18                │
│    │   GOOGL  | 32.1%  | 2.15   | -15.2% | 31                │
│    │   TSLA   | -5.4%  | -0.32  | -28.3% | 42                │
│    │                                                          │
│    ├─-> Combined equity curve chart                           │
│    │   (All 4 symbols on same chart)                         │
│    │                                                          │
│    └─-> Individual tearsheets for each symbol                 │
│        (4 separate HTML files)                               │
└──────────────────────────────────────────────────────────────┘
```

---

## Multi-Asset Portfolio Flow

### Simultaneous Positions Across Symbols

```
┌──────────────────────────────────────────────────────────────┐
│ 1. MULTI-ASSET CONFIGURATION                                 │
│    engine = BacktestEngine(                                  │
│        portfolio_mode='multi',                               │
│        weighting='equal',  # or 'risk_parity', 'ranked'      │
│        rebalance_frequency='monthly'                         │
│    )                                                         │
└────────────────┬─────────────────────────────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────────────────────────────┐
│ 2. LOAD DATA FOR ALL SYMBOLS                                │
│    data_dict = {                                             │
│        'AAPL': loader.load_bars('AAPL', ...),               │
│        'MSFT': loader.load_bars('MSFT', ...),               │
│        'GOOGL': loader.load_bars('GOOGL', ...)              │
│    }                                                         │
└────────────────┬─────────────────────────────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────────────────────────────┐
│ 3. GENERATE SIGNALS FOR ALL SYMBOLS                         │
│    signals_dict = {}                                         │
│    for symbol, data in data_dict.items():                   │
│        entries, exits = strategy.generate_signals(data)      │
│        signals_dict[symbol] = (entries, exits)               │
└────────────────┬─────────────────────────────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────────────────────────────┐
│ 4. MULTI-ASSET PORTFOLIO SIMULATION                          │
│    portfolio = MultiAssetPortfolio(                          │
│        initial_capital=100000,                               │
│        weighting='equal'                                     │
│    )                                                         │
│                                                              │
│    FOR each bar (synchronized across all symbols):          │
│    │                                                          │
│    ├─-> Check exit signals for all open positions            │
│    │   for symbol in portfolio.open_positions:              │
│    │       if signals_dict[symbol].exits[current_bar]:      │
│    │           portfolio.close_position(symbol)              │
│    │                                                          │
│    ├─-> Calculate available cash                              │
│    │   available_cash = portfolio.cash                       │
│    │                                                          │
│    ├─-> Check entry signals                                   │
│    │   entry_signals = []                                    │
│    │   for symbol in symbols:                                │
│    │       if signals_dict[symbol].entries[current_bar]:    │
│    │           entry_signals.append(symbol)                  │
│    │                                                          │
│    ├─-> Calculate position sizes (using weighting scheme)     │
│    │   if weighting == 'equal':                              │
│    │       allocation_per_symbol = available_cash / len(entry_signals)
│    │   elif weighting == 'risk_parity':                      │
│    │       # Allocate inversely proportional to volatility   │
│    │       allocations = calculate_risk_parity_weights()     │
│    │                                                          │
│    ├─-> Enter new positions                                   │
│    │   for symbol in entry_signals:                          │
│    │       shares = allocation / current_price[symbol]       │
│    │       portfolio.add_position(symbol, shares, price)     │
│    │                                                          │
│    ├─-> Check rebalancing trigger                             │
│    │   if should_rebalance(current_date):                    │
│    │       portfolio.rebalance(current_prices)               │
│    │                                                          │
│    └─-> Update equity curve                                   │
│        portfolio.equity = cash + sum(position_values)        │
└────────────────┬─────────────────────────────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────────────────────────────┐
│ 5. PORTFOLIO-LEVEL METRICS                                   │
│    - Total portfolio return                                  │
│    - Portfolio Sharpe ratio                                  │
│    - Portfolio max drawdown                                  │
│    - Per-symbol contribution to return                       │
│    - Diversification benefit                                 │
└──────────────────────────────────────────────────────────────┘
```

---

## GUI Backtest Flow

### Thread-Safe GUI Communication

```
┌──────────────────────────────────────────────────────────────┐
│ 1. USER INTERACTION (Main Thread - GUI)                     │
│    User clicks "Run Backtest" button in SetupView           │
│    ├─ Strategy: MovingAverageCrossover                       │
│    ├─ Symbols: ['AAPL', 'MSFT', 'GOOGL']                   │
│    ├─ Date range: 2023-01-01 to 2024-01-01                  │
│    └─ Workers: 4                                             │
└────────────────┬─────────────────────────────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────────────────────────────┐
│ 2. GUI CONTROLLER (gui/workers/gui_controller.py)           │
│    controller = GUIBacktestController(sweep_runner)          │
│    controller.start(strategy, symbols, start, end)           │
│                                                              │
│    Creates:                                                  │
│    ├─ worker_thread = Thread(target=run_backtest)           │
│    ├─ progress_queue = Queue()  # Thread-safe queue         │
│    └─ result_queue = Queue()                                 │
│                                                              │
│    worker_thread.start()  # Starts in background            │
│    return immediately -> GUI stays responsive                 │
└────────────────┬─────────────────────────────────────────────┘
                 │
        ┌────────┴────────┐
        │                 │
        ▼                 ▼
┌──────────────┐   ┌──────────────────┐
│ MAIN THREAD  │   │  WORKER THREAD   │
│   (GUI)      │   │  (Backtest)      │
└──────┬───────┘   └────────┬─────────┘
       │                    │
       │                    ├─-> SweepRunner.run_sweep(...)
       │                    │
       │                    ├─-> Callback: on_symbol_start('AAPL')
       │                    │   progress_queue.put({
       │                    │       'type': 'start',
       │                    │       'symbol': 'AAPL'
       │                    │   })
       │                    │
       ├── Poll queue every 100ms
       │   update = progress_queue.get_nowait()
       │   if update['type'] == 'start':
       │       RunView.update_status('Starting AAPL...')
       │                    │
       │                    ├─-> Run backtest for AAPL
       │                    │   (Load data, signals, simulate)
       │                    │
       │                    ├─-> Callback: on_symbol_complete('AAPL', portfolio)
       │                    │   progress_queue.put({
       │                    │       'type': 'complete',
       │                    │       'symbol': 'AAPL',
       │                    │       'stats': portfolio.stats()
       │                    │   })
       │                    │
       ├── Poll queue
       │   update = progress_queue.get_nowait()
       │   if update['type'] == 'complete':
       │       RunView.update_progress('AAPL: [+] 25.3%')
       │       RunView.progress_bar.value = 0.33  # 1/3 complete
       │                    │
       │                    ├─-> Repeat for MSFT, GOOGL
       │                    │
       │                    ├─-> All symbols complete
       │                    │   result_queue.put({
       │                    │       'type': 'complete',
       │                    │       'portfolios': [...]
       │                    │   })
       │                    │
       ├── Poll queue
       │   result = result_queue.get_nowait()
       │   if result['type'] == 'complete':
       │       Navigate to ResultsView
       │       Display metrics table
       │                    │
       │                    └─-> Thread exits
       │
       └─-> Continue polling (cleanup)
```

---

## Portfolio Simulation Flow (Bar-by-Bar)

### Detailed Execution Logic

```
┌──────────────────────────────────────────────────────────────┐
│ PORTFOLIO SIMULATOR - Bar-by-Bar Execution                   │
└────────────────┬─────────────────────────────────────────────┘
                 │
    Initialize: cash = 100000, positions = [], equity = [100000]
                 │
                 ▼
        ┌────────────────┐
        │  FOR each bar  │
        └────────┬───────┘
                 │
                 ▼
    ┌────────────────────────────────────────────────┐
    │ 1. CHECK EXIT SIGNALS                          │
    │    if exits[i] and currently_in_position:      │
    └────────┬───────────────────────────────────────┘
             │ Yes
             ▼
    ┌────────────────────────────────────────────────┐
    │ 2. CALCULATE EXIT PRICE                        │
    │    base_price = close[i]                       │
    │    slippage_cost = base_price * slippage       │
    │    exit_price = base_price - slippage_cost     │
    │    # Ensure within OHLC range                  │
    │    exit_price = clip(exit_price, low[i], high[i])
    └────────┬───────────────────────────────────────┘
             │
             ▼
    ┌────────────────────────────────────────────────┐
    │ 3. EXECUTE EXIT                                │
    │    proceeds = exit_price * shares              │
    │    exit_fee = proceeds * fees                  │
    │    net_proceeds = proceeds - exit_fee          │
    │    cash += net_proceeds                        │
    │                                                │
    │    Calculate P&L:                              │
    │    cost_basis = entry_price * shares + entry_fee
    │    pnl = net_proceeds - cost_basis             │
    │    pnl_pct = pnl / cost_basis * 100            │
    │                                                │
    │    Log trade:                                  │
    │    trades.append({                             │
    │        'type': 'exit',                         │
    │        'timestamp': timestamp[i],              │
    │        'price': exit_price,                    │
    │        'shares': shares,                       │
    │        'value': net_proceeds,                  │
    │        'fees': exit_fee,                       │
    │        'pnl': pnl,                             │
    │        'pnl_pct': pnl_pct                      │
    │    })                                          │
    │                                                │
    │    positions = []  # Close position            │
    └────────┬───────────────────────────────────────┘
             │
             ▼
    ┌────────────────────────────────────────────────┐
    │ 4. CHECK STOP LOSS (if position open)         │
    │    if in_position:                             │
    │        current_price = close[i]                │
    │        if stop_loss_type == 'percentage':      │
    │            stop_price = entry_price * (1 - stop_loss_pct)
    │            if current_price <= stop_price:     │
    │                Force exit (same as step 2-3)   │
    └────────┬───────────────────────────────────────┘
             │
             ▼
    ┌────────────────────────────────────────────────┐
    │ 5. CHECK ENTRY SIGNALS                         │
    │    if entries[i] and not in_position:          │
    └────────┬───────────────────────────────────────┘
             │ Yes
             ▼
    ┌────────────────────────────────────────────────┐
    │ 6. POSITION SIZING                             │
    │    sizer = PositionSizer(risk_config)          │
    │    shares = sizer.calculate_shares(            │
    │        symbol=symbol,                          │
    │        current_price=close[i],                 │
    │        available_cash=cash,                    │
    │        atr=atr[i]  # For volatility-based      │
    │    )                                           │
    │                                                │
    │    if position_sizing_method == 'fixed_percentage':
    │        target_value = cash * position_size_pct │
    │        shares = int(target_value / close[i])   │
    │                                                │
    │    elif position_sizing_method == 'volatility':
    │        risk_per_share = atr[i] * atr_multiplier
    │        target_risk = cash * risk_pct           │
    │        shares = int(target_risk / risk_per_share)
    └────────┬───────────────────────────────────────┘
             │
             ▼
    ┌────────────────────────────────────────────────┐
    │ 7. RISK MANAGEMENT CHECKS                      │
    │    manager = RiskManager(risk_config)          │
    │                                                │
    │    # Check capital constraint                  │
    │    required_cash = shares * close[i]           │
    │    if required_cash > cash:                    │
    │        shares = int(cash / close[i])  # Clip   │
    │                                                │
    │    # Check max position size                   │
    │    max_shares = int((cash * max_position_pct) / close[i])
    │    shares = min(shares, max_shares)            │
    │                                                │
    │    # Check max positions                       │
    │    if len(open_positions) >= max_positions:    │
    │        Skip entry  # Too many positions        │
    └────────┬───────────────────────────────────────┘
             │
             ▼
    ┌────────────────────────────────────────────────┐
    │ 8. CALCULATE ENTRY PRICE                       │
    │    base_price = close[i]                       │
    │    slippage_cost = base_price * slippage       │
    │    entry_price = base_price + slippage_cost    │
    │    # Ensure within OHLC range                  │
    │    entry_price = clip(entry_price, low[i], high[i])
    └────────┬───────────────────────────────────────┘
             │
             ▼
    ┌────────────────────────────────────────────────┐
    │ 9. EXECUTE ENTRY                               │
    │    cost = entry_price * shares                 │
    │    entry_fee = cost * fees                     │
    │    total_cost = cost + entry_fee               │
    │    cash -= total_cost                          │
    │                                                │
    │    Log trade:                                  │
    │    trades.append({                             │
    │        'type': 'entry',                        │
    │        'timestamp': timestamp[i],              │
    │        'price': entry_price,                   │
    │        'shares': shares,                       │
    │        'value': cost,                          │
    │        'fees': entry_fee                       │
    │    })                                          │
    │                                                │
    │    positions.append({                          │
    │        'symbol': symbol,                       │
    │        'shares': shares,                       │
    │        'entry_price': entry_price,             │
    │        'entry_date': timestamp[i]              │
    │    })                                          │
    └────────┬───────────────────────────────────────┘
             │
             ▼
    ┌────────────────────────────────────────────────┐
    │ 10. UPDATE EQUITY CURVE                        │
    │     position_value = 0                         │
    │     for pos in positions:                      │
    │         position_value += pos['shares'] * close[i]
    │                                                │
    │     equity[i] = cash + position_value          │
    └────────┬───────────────────────────────────────┘
             │
             ├──-> Next bar (i+1)
             │
             ▼
    ┌────────────────────────────────────────────────┐
    │ ALL BARS COMPLETE                              │
    │                                                │
    │ Return Portfolio object:                       │
    │ - equity_curve: pd.Series(equity, index=timestamps)
    │ - trades: List of all trades                   │
    │ - final_value: equity[-1]                      │
    │ - cash: current cash                           │
    └────────────────────────────────────────────────┘
```

---

## Risk Management Flow

### Stop Loss and Position Constraints

```
Entry Signal Triggered
    │
    ▼
┌─────────────────────────────────────┐
│ RiskManager.can_enter()             │
│                                     │
│ Checks:                             │
│ ├─ Portfolio has < max_positions?   │
│ ├─ Sufficient cash available?       │
│ ├─ Position size < max_position_%?  │
│ └─ Not already in position?         │
│                                     │
│ If all pass -> Allow entry           │
│ If any fail -> Reject entry          │
└────────┬────────────────────────────┘
         │ Allowed
         ▼
┌─────────────────────────────────────┐
│ PositionSizer.calculate_shares()    │
│                                     │
│ Based on method:                    │
│ ├─ Fixed %: shares = (cash * %) / price
│ ├─ Volatility: shares = risk / (ATR * mult)
│ └─ Kelly: shares = (kelly_pct * cash) / price
└────────┬────────────────────────────┘
         │
         ▼
    Execute Entry
         │
         ▼
┌─────────────────────────────────────┐
│ RiskManager.update_position()       │
│ - Add to open_positions list        │
│ - Track entry_price for stop loss   │
└────────┬────────────────────────────┘
         │
         │ On each subsequent bar...
         ▼
┌─────────────────────────────────────┐
│ RiskManager.check_stop_loss()       │
│                                     │
│ if stop_loss_type == 'percentage':  │
│     stop_price = entry * (1 - %)    │
│     if current_price <= stop_price: │
│         -> Force exit                │
│                                     │
│ elif stop_loss_type == 'atr':       │
│     stop_price = entry - (N * ATR)  │
│     if current_price <= stop_price: │
│         -> Force exit                │
└─────────────────────────────────────┘
```

---

## Signal Generation Flow

### Indicator Calculation -> Signal Logic

```
Raw OHLCV Data
    │
    ▼
┌──────────────────────────────────────┐
│ 1. CALCULATE INDICATORS              │
│    (backtesting/utils/indicators.py) │
│                                      │
│    fast_ma = sma(close, window=20)   │
│    slow_ma = sma(close, window=50)   │
│    rsi = rsi(close, window=14)       │
│    atr = atr(high, low, close, 14)   │
└────────┬─────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────┐
│ 2. APPLY STRATEGY LOGIC              │
│                                      │
│    # Moving Average Crossover        │
│    bullish_cross = (                 │
│        (fast_ma > slow_ma) &         │
│        (fast_ma.shift(1) <= slow_ma.shift(1))
│    )                                 │
│                                      │
│    bearish_cross = (                 │
│        (fast_ma < slow_ma) &         │
│        (fast_ma.shift(1) >= slow_ma.shift(1))
│    )                                 │
└────────┬─────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────┐
│ 3. GENERATE BOOLEAN SIGNALS          │
│                                      │
│    entries = bullish_cross           │
│    exits = bearish_cross             │
│                                      │
│    # Result: pd.Series of booleans   │
│    entries[i] = True -> Enter at bar i│
│    exits[i] = True -> Exit at bar i   │
└────────┬─────────────────────────────┘
         │
         ▼
    Return (entries, exits)
```

---

## Live Trading Execution Flow

### Strategy Signal to Order Execution (EC2 Deployment)

```
Scheduled Trigger (systemd timer)
    |
Strategy Adapter (OMR/RAMP)
    |-> Load market data (streaming or REST)
    |-> Generate signals
    |-> Apply filters (regime, position limits)
    |
Execution Engine
    |-> Validate signals
    |-> Calculate position sizes
    |-> Submit orders to Alpaca
    |
State Manager
    |-> Persist positions to file
    |-> Track pending orders
```

### Detailed Flow

```
+--------------------------------------------------------------+
| 1. SCHEDULED EXECUTION (systemd timer)                       |
|    Schedule:                                                 |
|    - OMR Entry: 3:50 PM ET (Mon-Fri)                         |
|    - OMR Exit: 9:31 AM ET (Mon-Fri)                          |
|    - RAMP Rebalance: 3:55 PM ET (Mon-Fri)                    |
|                                                              |
|    systemctl start homeguard-omr.service                     |
|    systemctl start homeguard-ramp.service                    |
+--------------------------------------------------------------+
                 |
                 v
+--------------------------------------------------------------+
| 2. STRATEGY ADAPTER INITIALIZATION                           |
|    # OMR Adapter (trading/adapters/omr_live_adapter.py)      |
|    adapter = OMRLiveAdapter(                                 |
|        broker=AlpacaBroker(),                                |
|        provider=LiveDataProvider(),  # or CompositeProvider  |
|        config=load_omr_config()                              |
|    )                                                         |
+--------------------------------------------------------------+
                 |
                 v
+--------------------------------------------------------------+
| 3. MARKET DATA RETRIEVAL                                     |
|    # With streaming (USE_STREAMING=true)                     |
|    for symbol in universe:                                   |
|        price = provider.get_price(symbol)                    |
|        bars = provider.get_bars(symbol, 10)                  |
|                                                              |
|    # Without streaming (REST fallback)                       |
|    for symbol in universe:                                   |
|        bars = broker.get_historical_bars(symbol, limit=10)   |
|        price = bars[-1].close                                |
+--------------------------------------------------------------+
                 |
                 v
+--------------------------------------------------------------+
| 4. SIGNAL GENERATION                                         |
|    # OMR: Bayesian overnight mean reversion                  |
|    signals = strategy.generate_entry_signals(                |
|        prices=current_prices,                                |
|        historical_data=historical_bars,                      |
|        regime=market_regime                                  |
|    )                                                         |
|                                                              |
|    # RAMP: Cross-sectional momentum ranking                  |
|    signals = strategy.generate_rebalance_signals(            |
|        prices=sp500_closes,                                  |
|        momentum_scores=calculate_momentum(prices),           |
|        top_n=10                                              |
|    )                                                         |
|                                                              |
|    Signal format:                                            |
|    [                                                         |
|        Signal(symbol='TQQQ', direction='long', weight=0.2),  |
|        Signal(symbol='SOXL', direction='long', weight=0.2),  |
|        ...                                                   |
|    ]                                                         |
+--------------------------------------------------------------+
                 |
                 v
+--------------------------------------------------------------+
| 5. FILTER APPLICATION                                        |
|    # Market regime filter                                    |
|    if regime == 'BEAR' and not allow_bear_trades:            |
|        signals = []  # No trades in bear regime              |
|                                                              |
|    # Position limit filter                                   |
|    if len(open_positions) >= max_positions:                  |
|        signals = signals[:max_positions - len(open_positions)]
|                                                              |
|    # Daily loss limit filter                                 |
|    if daily_pnl < -daily_max_loss:                           |
|        signals = []  # Stop trading for the day              |
+--------------------------------------------------------------+
                 |
                 v
+--------------------------------------------------------------+
| 6. ORDER SUBMISSION (broker/alpaca_broker.py)                |
|    for signal in signals:                                    |
|        # Calculate position size                             |
|        shares = calculate_shares(                            |
|            capital=available_capital,                        |
|            weight=signal.weight,                             |
|            price=signal.price                                |
|        )                                                     |
|                                                              |
|        # Submit market order                                 |
|        order = broker.submit_order(                          |
|            symbol=signal.symbol,                             |
|            qty=shares,                                       |
|            side='buy' if signal.direction == 'long' else 'sell',
|            type='market',                                    |
|            time_in_force='day'                               |
|        )                                                     |
|                                                              |
|        logger.info(f"Submitted: {signal.symbol} {shares} shares")
+--------------------------------------------------------------+
                 |
                 v
+--------------------------------------------------------------+
| 7. STATE PERSISTENCE (trading/state/strategy_state_manager.py)|
|    state_manager.save_position(                              |
|        symbol=signal.symbol,                                 |
|        shares=shares,                                        |
|        entry_price=fill_price,                               |
|        entry_time=now(),                                     |
|        strategy='OMR'                                        |
|    )                                                         |
|                                                              |
|    State file: strategy_positions.json                       |
|    {                                                         |
|        "OMR": {                                               |
|            "TQQQ": {"shares": 100, "entry_price": 45.50},    |
|            "SOXL": {"shares": 50, "entry_price": 32.10}      |
|        },                                                    |
|        "RAMP": {                                              |
|            "AAPL": {"shares": 25, "entry_price": 195.00},    |
|            ...                                                |
|        }                                                     |
|    }                                                         |
+--------------------------------------------------------------+
                 |
                 v
+--------------------------------------------------------------+
| 8. ORDER FILL VERIFICATION                                   |
|    # Wait for fills                                          |
|    for order_id in pending_orders:                           |
|        status = broker.get_order(order_id)                   |
|        if status == 'filled':                                |
|            logger.info(f"Filled: {order_id}")                |
|        elif status == 'rejected':                            |
|            logger.error(f"Rejected: {order_id}")             |
|                                                              |
|    # Log final positions                                     |
|    positions = broker.get_positions()                        |
|    for pos in positions:                                     |
|        logger.info(f"{pos.symbol}: {pos.qty} @ {pos.avg_entry_price}")
+--------------------------------------------------------------+
```

### Multi-Strategy Coordination

```
homeguard-trading.target
    |
    +-- homeguard-omr.service (3:50 PM entry, 9:31 AM exit)
    |   |-- Leveraged ETF universe (TQQQ, SOXL, UPRO, ...)
    |   +-- Max 5 concurrent positions at 20% each
    |
    +-- homeguard-ramp.service (3:55 PM rebalance)
        |-- S&P 500 universe
        +-- Top 10 by momentum, dynamic 1/N sizing

Coordination:
    - Non-overlapping universes (no conflicts)
    - Execution lock for 3:50-4:00 PM window
    - Shared state file with atomic writes
```

---

## Visualization Pipeline Flow

```
Portfolio object
    │
    ├─-> equity_curve (pd.Series)
    ├─-> trades (List[Dict])
    └─-> stats (Dict)
        │
        ▼
┌──────────────────────────────────────────────┐
│ BacktestVisualizer.generate_all()            │
└────────┬─────────────────────────────────────┘
         │
         ├─-> QuantStatsReporter
         │   ├─ Convert equity to returns
         │   ├─ Load benchmark (SPY)
         │   ├─ Calculate 50+ metrics
         │   └─ Generate HTML tearsheet
         │       ├─ Returns heatmap
         │       ├─ Drawdown chart
         │       ├─ Rolling Sharpe
         │       └─ Monte Carlo
         │
         ├─-> CandlestickChart
         │   ├─ Plot OHLC bars
         │   ├─ Add trade markers
         │   │   ├─ Green ▲ for entries
         │   │   └─ Red ▼ for exits
         │   └─ Save as PNG
         │
         ├─-> ReportGenerator
         │   ├─ Create summary table
         │   ├─ Format metrics
         │   └─ Save as TXT/CSV
         │
         └─-> OutputManager
             └─ Organize files in output directory
                 logs/AAPL_20250105_153045/
                 ├── tearsheet.html
                 ├── candlestick.png
                 ├── trade_log.csv
                 └── summary.txt
```

---

## Summary: Data Flow Paths

| Operation | Entry Point | Data Path | Output |
|-----------|-------------|-----------|--------|
| **Data Ingestion** | `run_ingestion.py` | AlpacaAPI -> ParquetStorage -> Metadata | Parquet files |
| **Streaming Data** | `LiveDataProvider` | WebSocket -> StreamManager -> BarBuffer -> Strategy | Real-time prices |
| **News/Sentiment** | `NewsDownloader` | NewsAPI -> Parquet -> SentimentAnalyzer -> Cache | Sentiment scores |
| **Single Backtest (CLI)** | `backtest_runner.py` | DataLoader -> Strategy -> PortfolioSim -> Visualizer | Reports/Charts |
| **Multi-Symbol Sweep** | `backtest_runner.py` | SweepRunner -> [Parallel Engines] -> Aggregator | Comparison tables |
| **Multi-Asset Portfolio** | `backtest_runner.py` | DataLoader -> MultiAssetPortfolio -> Visualizer | Portfolio reports |
| **GUI Backtest** | `gui/__main__.py` | GUIController -> [Worker Thread] SweepRunner -> Queue -> UI | Interactive results |
| **Live Trading** | `homeguard-*.service` | Adapter -> Provider -> Signals -> Broker -> StateManager | Orders/Positions |

---

**Last Updated**: 2025-12-15
**Maintainers**: Update when adding new data flows or changing execution paths
**Related Docs**: [ARCHITECTURE_OVERVIEW.md](ARCHITECTURE_OVERVIEW.md), [MODULE_REFERENCE.md](MODULE_REFERENCE.md)
