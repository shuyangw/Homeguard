# Homeguard Module Reference

**Version**: 1.1
**Last Updated**: 2025-11-11
**Purpose**: Comprehensive module-by-module reference for the Homeguard codebase

---

## Table of Contents

1. [Root Level Modules](#root-level-modules)
2. [Data Engine Layer](#data-engine-layer)
3. [Backtesting Engine Layer](#backtesting-engine-layer)
4. [Strategy Layer](#strategy-layer)
5. [Visualization Layer](#visualization-layer)
6. [GUI Layer](#gui-layer)
7. [Utility Layer](#utility-layer)

---

## Root Level Modules

### `src/config/` (Package)
**Purpose**: Consolidated configuration management for application settings and config-driven backtesting

**Submodules**:
- `settings.py`: Application settings from `settings.ini` (OS detection, directory paths)
- `schema.py`: Pydantic models for backtest configuration validation
- `loader.py`: YAML loading with inheritance support (`extends:` directive)
- `defaults.py`: Default configuration values, date presets, symbol universes

**Key Classes**:
- `BacktestConfig`: Root configuration model for backtests
- `BacktestMode`: Enum (single, sweep, optimize, walk_forward)
- `StrategyConfig`, `SymbolsConfig`, `DatesConfig`: Sub-configuration models

**Key Functions**:
- `load_config(path)`: Load and validate YAML config file
- `get_backtest_results_dir()`: Returns backtest output directory
- `get_symbol_universe(name)`: Get predefined symbol list
- `get_date_preset(name)`: Get predefined date range

**Configuration Types**:
1. **Application Settings** (from `settings.ini`):
   - OS-specific paths (Windows/macOS/Linux)
   - Data directory, log output directory
   - Tearsheet frequency settings

2. **Backtest Config** (from YAML files):
   - Strategy selection and parameters
   - Symbol lists/universes
   - Date ranges/presets
   - Risk management settings
   - Mode-specific settings (sweep, optimization, walk-forward)

**Dependencies**: `pydantic`, `yaml`, `configparser`, `pathlib`

**Usage Example**:
```python
# Application settings
from src.config import settings, get_backtest_results_dir
results_dir = get_backtest_results_dir()

# Config-driven backtest
from src.config import load_config
config = load_config("config/backtesting/omr_backtest.yaml")
print(config.strategy.name)
print(config.symbols.list)
```

---

### `src/api_key.py`
**Purpose**: API credential management (Alpaca API keys)

**Security**:
- ⚠️ **Never commit this file** (in `.gitignore`)
- Store API keys securely

**Key Variables**:
- `ALPACA_API_KEY`: Alpaca API key
- `ALPACA_SECRET_KEY`: Alpaca secret key
- `ALPACA_BASE_URL`: API endpoint URL

**Usage Example**:
```python
from api_key import ALPACA_API_KEY, ALPACA_SECRET_KEY
client = AlpacaClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
```

---

### `src/backtest_runner.py`
**Purpose**: CLI entry point for running backtests (supports both traditional CLI and config-driven modes)

**Key Functions**:
- `main()`: CLI argument parser and execution
- `run_backtest()`: Single symbol backtest (traditional mode)
- `sweep_backtest()`: Multi-symbol sweep (traditional mode)
- `run_from_config()`: Config-driven backtest entry point
- `run_single_from_config()`, `run_sweep_from_config()`, etc.: Mode-specific handlers

**CLI Arguments (Config Mode - RECOMMENDED)**:
- `--config`: Path to YAML config file
- `--mode`: Override mode from config (single/sweep/optimize/walk_forward)

**CLI Arguments (Traditional Mode)**:
- `--strategy`: Strategy class name
- `--symbols`: Ticker symbols (comma-separated)
- `--start`: Start date (YYYY-MM-DD)
- `--end`: End date (YYYY-MM-DD)
- `--capital`: Starting capital
- `--fees`: Trading fees (decimal)
- `--sweep`: Run multi-symbol sweep
- `--optimize`: Run parameter optimization
- `--quantstats`: Generate QuantStats tearsheet

**Dependencies**: `argparse`, `BacktestEngine`, `src.config`, `src.strategies.registry`

**Usage Examples**:
```bash
# Config-driven (RECOMMENDED)
python -m src.backtest_runner --config config/backtesting/omr_backtest.yaml
python -m src.backtest_runner --config config/backtesting/ma_sweep.yaml --mode sweep

# Traditional CLI
python -m src.backtest_runner \
  --strategy MovingAverageCrossover \
  --symbols AAPL,MSFT \
  --start 2023-01-01 \
  --end 2024-01-01
```

---

### `src/run_ingestion.py`
**Purpose**: Data ingestion entry point

**Key Functions**:
- `main()`: Orchestrates data ingestion pipeline
- `fetch_symbols()`: Loads symbol list
- `run_pipeline()`: Executes multi-threaded ingestion

**Dependencies**: `IngestionPipeline`, `AlpacaClient`, `ParquetStorage`

**Usage Example**:
```bash
python -m src.run_ingestion
```

---

## Data Engine Layer

### `src/data_engine/api/alpaca_client.py`
**Purpose**: Alpaca API client for fetching market data

**Key Classes**:
- `AlpacaClient`: REST API client

**Key Methods**:
- `fetch_bars(symbol, start, end, timeframe)`: Fetch OHLCV bars
- `get_latest_trade(symbol)`: Get latest trade price
- `get_account()`: Get account information

**Rate Limiting**:
- Implements automatic retry with exponential backoff
- Respects Alpaca API rate limits (200 requests/minute)

**Error Handling**:
- Connection errors → retry
- Authentication errors → fail fast
- Data errors → log and skip

**Dependencies**: `requests`, `pandas`, `api_key`

**Usage Example**:
```python
from data_engine.api.alpaca_client import AlpacaClient
client = AlpacaClient(api_key, secret_key)
bars = client.fetch_bars('AAPL', '2023-01-01', '2024-01-01', '1Min')
```

---

### `src/data_engine/loaders/symbol_loader.py`
**Purpose**: Load symbol lists from files

**Key Classes**:
- `SymbolLoader`: Symbol list loader

**Key Methods**:
- `load_from_csv(filepath)`: Load from CSV file
- `load_from_txt(filepath)`: Load from text file (one per line)
- `load_from_list(symbols)`: Load from Python list

**Supported Formats**:
- CSV: `symbol,name,sector`
- TXT: `AAPL\nMSFT\nGOOGL`
- List: `['AAPL', 'MSFT', 'GOOGL']`

**Dependencies**: `pandas`, `pathlib`

**Usage Example**:
```python
from data_engine.loaders.symbol_loader import SymbolLoader
loader = SymbolLoader()
symbols = loader.load_from_csv('sp500.csv')
```

---

### `src/data_engine/storage/parquet_storage.py`
**Purpose**: Store and retrieve market data in Parquet format

**Key Classes**:
- `ParquetStorage`: Parquet file manager

**Key Methods**:
- `save_bars(symbol, data, timeframe)`: Save OHLCV data
- `load_bars(symbol, start, end, timeframe)`: Load OHLCV data
- `get_available_dates(symbol, timeframe)`: List available dates
- `delete_symbol(symbol, timeframe)`: Delete all data for symbol

**Storage Structure**:
```
data/
└── equities_1min/
    ├── AAPL/
    │   ├── 2023-01-01.parquet
    │   ├── 2023-01-02.parquet
    │   └── ...
    └── MSFT/
        └── ...
```

**Partitioning**:
- By timeframe (1min, 5min, 1hour, 1day)
- By symbol (AAPL, MSFT, etc.)
- By date (one file per trading day)

**Compression**: Parquet default compression (snappy)

**Dependencies**: `pandas`, `pyarrow`, `pathlib`

**Usage Example**:
```python
from data_engine.storage.parquet_storage import ParquetStorage
storage = ParquetStorage(data_dir='data/')
storage.save_bars('AAPL', bars_df, '1Min')
data = storage.load_bars('AAPL', '2023-01-01', '2024-01-01', '1Min')
```

---

### `src/data_engine/storage/metadata_store.py`
**Purpose**: Store metadata about symbol universes

**Key Classes**:
- `MetadataStore`: JSON-based metadata storage

**Metadata Tracked**:
- Symbol universes (DOW30, NASDAQ100, S&P500)
- Last update timestamp
- Data availability ranges
- Symbol sector/industry information

**Storage Format**: JSON file (`metadata.json`)

**Dependencies**: `json`, `pathlib`, `datetime`

**Usage Example**:
```python
from data_engine.storage.metadata_store import MetadataStore
store = MetadataStore()
store.update_universe('DOW30', ['AAPL', 'MSFT', ...])
symbols = store.get_universe('DOW30')
```

---

### `src/data_engine/orchestration/ingestion_pipeline.py`
**Purpose**: Multi-threaded data ingestion orchestration

**Key Classes**:
- `IngestionPipeline`: Pipeline orchestrator

**Key Methods**:
- `run(symbols, start, end, timeframe, workers)`: Execute ingestion
- `fetch_and_store(symbol)`: Worker function for single symbol
- `on_progress(symbol, status)`: Progress callback
- `on_complete()`: Completion callback

**Features**:
- **Multi-threading**: ThreadPoolExecutor (1-8 workers)
- **Progress tracking**: Real-time progress updates
- **Error handling**: Continue on errors, log failures
- **Retry logic**: Retry failed symbols up to 3 times

**Dependencies**: `ThreadPoolExecutor`, `AlpacaClient`, `ParquetStorage`, `logger`

**Usage Example**:
```python
from data_engine.orchestration.ingestion_pipeline import IngestionPipeline
pipeline = IngestionPipeline(data_dir='data/')
pipeline.run(
    symbols=['AAPL', 'MSFT', 'GOOGL'],
    start='2023-01-01',
    end='2024-01-01',
    timeframe='1Min',
    workers=4
)
```

---

## Backtesting Engine Layer

### Core Engine Modules

#### `src/backtesting/engine/backtest_engine.py`
**Purpose**: Primary backtest orchestrator

**Key Classes**:
- `BacktestEngine`: Main engine class

**Initialization Parameters**:
- `initial_capital`: Starting capital (default: $100,000)
- `fees`: Trading fees as decimal (default: 0.001 = 0.1%)
- `slippage`: Slippage as decimal (default: 0.0)
- `market_hours_only`: Filter to market hours (default: True)
- `risk_config`: RiskConfig instance (default: moderate)

**Key Methods**:
- `run(strategy, symbols, start, end)`: Run backtest (auto-routes to single/multi)
- `run_with_data(strategy, data)`: Run with pre-loaded data
- `_run_single_symbol(strategy, symbol, start, end)`: Single symbol mode
- `_run_multi_asset(strategy, symbols, start, end)`: Multi-asset mode

**Data Validation**:
- Removes duplicate timestamps (logs warning)
- Handles NaN values gracefully
- Validates OHLC relationships

**Dependencies**: `DataLoader`, `PortfolioSimulator`, `RiskManager`, `logger`

**Usage Example**:
```python
from backtesting.engine.backtest_engine import BacktestEngine
from backtesting.utils.risk_config import RiskConfig
from strategies.moving_average import MovingAverageCrossover

engine = BacktestEngine(
    initial_capital=100000,
    fees=0.001,
    slippage=0.001,
    risk_config=RiskConfig.moderate()
)

portfolio = engine.run(
    strategy=MovingAverageCrossover(fast=20, slow=50),
    symbols='AAPL',
    start='2023-01-01',
    end='2024-01-01'
)
```

---

#### `src/backtesting/engine/portfolio_simulator.py`
**Purpose**: Custom portfolio simulator (replaces VectorBT)

**Key Classes**:
- `Portfolio`: Portfolio state and results
- `PortfolioSimulator`: Bar-by-bar simulator (internal use)

**Portfolio Attributes**:
- `init_cash`: Initial capital
- `equity_curve`: pd.Series of portfolio value over time
- `trades`: List of trade dictionaries
- `positions`: Current open positions
- `cash`: Available cash

**Portfolio Methods**:
- `stats()`: Calculate performance statistics
- `returns(freq)`: Get returns series (daily, hourly, etc.)
- `plot()`: Plot equity curve

**Simulation Logic**:
1. Iterate through each bar
2. Check for exit signals → close positions
3. Check for entry signals → open positions
4. Update equity curve
5. Log trades
6. Check stop losses
7. Update metrics

**Trade Dictionary Structure**:
```python
{
    'type': 'entry' | 'exit',
    'timestamp': datetime,
    'symbol': str,
    'price': float,
    'shares': int,
    'value': float,
    'fees': float,
    'pnl': float (exit only),
    'pnl_pct': float (exit only)
}
```

**Statistics Calculated**:
- Total Return [%]
- Annual Return [%]
- Sharpe Ratio
- Max Drawdown [%]
- Win Rate [%]
- Total Trades
- Start/End Value

**Dependencies**: `pandas`, `numpy`, `RiskManager`, `PositionSizer`, `logger`

**Usage Example**:
```python
# Usually called internally by BacktestEngine
from backtesting.engine.portfolio_simulator import from_signals

portfolio = from_signals(
    close=price_series,
    entries=entry_signals,
    exits=exit_signals,
    init_cash=100000,
    fees=0.001,
    slippage=0.0,
    risk_config=RiskConfig.moderate()
)

stats = portfolio.stats()
print(f"Total Return: {stats['Total Return [%]']:.2f}%")
```

---

#### `src/backtesting/engine/data_loader.py`
**Purpose**: Load OHLCV data from Parquet via DuckDB

**Key Classes**:
- `DataLoader`: Data loading manager

**Key Methods**:
- `load_bars(symbol, start, end, timeframe)`: Load data
- `load_multiple(symbols, start, end, timeframe)`: Load multiple symbols
- `get_available_symbols(timeframe)`: List available symbols
- `get_date_range(symbol, timeframe)`: Get available date range

**Performance Optimization**:
- **DuckDB SQL queries**: 10-100x faster than Pandas
- **Lazy loading**: Only loads requested date range
- **Market day filtering**: Built-in NYSE calendar filtering

**Market Calendar Integration**:
- Filters weekends automatically
- Filters NYSE holidays
- Trading hours: 9:30 AM - 4:00 PM EST
- Can optionally filter to 9:35 AM - 3:55 PM (avoid auction)

**Dependencies**: `duckdb`, `pandas`, `MarketCalendar`, `ParquetStorage`

**Usage Example**:
```python
from backtesting.engine.data_loader import DataLoader

loader = DataLoader(data_dir='data/')
data = loader.load_bars('AAPL', '2023-01-01', '2024-01-01', '1Min')
# Returns: pd.DataFrame with OHLCV columns, DatetimeIndex
```

---

#### `src/backtesting/engine/sweep_runner.py`
**Purpose**: Run strategy across multiple symbols in parallel

**Key Classes**:
- `SweepRunner`: Multi-symbol sweep orchestrator

**Initialization Parameters**:
- `engine`: BacktestEngine instance
- `workers`: Number of parallel workers (1-8, default: 4)
- `fail_fast`: Stop on first error (default: False)

**Key Methods**:
- `run_sweep(strategy, symbols, start, end)`: Run sweep
- `set_callbacks(on_start, on_complete, on_error)`: Progress callbacks

**Execution Model**:
- **ThreadPoolExecutor** for parallel execution
- Each symbol runs in separate thread
- GIL-safe (I/O bound, not CPU bound)

**Callbacks**:
```python
def on_symbol_start(symbol: str):
    print(f"Starting {symbol}...")

def on_symbol_complete(symbol: str, portfolio: Portfolio):
    print(f"Completed {symbol}: {portfolio.stats()['Total Return [%]']:.2f}%")

def on_symbol_error(symbol: str, error: Exception):
    print(f"Error on {symbol}: {error}")
```

**Dependencies**: `ThreadPoolExecutor`, `BacktestEngine`, `logger`

**Usage Example**:
```python
from backtesting.engine.sweep_runner import SweepRunner
from backtesting.engine.backtest_engine import BacktestEngine

engine = BacktestEngine()
runner = SweepRunner(engine, workers=4)

runner.set_callbacks(
    on_start=lambda s: print(f"Starting {s}"),
    on_complete=lambda s, p: print(f"Done {s}"),
    on_error=lambda s, e: print(f"Error {s}: {e}")
)

portfolios = runner.run_sweep(
    strategy=MovingAverageCrossover(),
    symbols=['AAPL', 'MSFT', 'GOOGL'],
    start='2023-01-01',
    end='2024-01-01'
)
```

---

#### `src/backtesting/engine/multi_asset_portfolio.py`
**Purpose**: Manage portfolios with simultaneous positions across symbols

**Key Classes**:
- `MultiAssetPortfolio`: Portfolio manager

**Key Features**:
- **Simultaneous positions**: Hold multiple symbols at once
- **Portfolio weighting**: Equal weight, risk parity, ranked, etc.
- **Rebalancing**: Periodic, threshold-based, or signal-based
- **Cash management**: Shared cash pool across all symbols

**Weighting Schemes**:
1. **Equal Weight**: 1/N allocation per symbol
2. **Risk Parity**: Equal risk contribution (volatility-scaled)
3. **Fixed Count**: Top N symbols by rank
4. **Ranked**: Proportional to ranking score
5. **Adaptive**: Dynamic based on conditions

**Rebalancing Triggers**:
- **Periodic**: Daily, weekly, monthly
- **Threshold**: When drift > X%
- **Signal-based**: On strategy signals

**Dependencies**: `PortfolioSimulator`, `RiskManager`, `portfolio_construction`, `rebalancing`

**Usage Example**:
```python
from backtesting.engine.multi_asset_portfolio import MultiAssetPortfolio

portfolio = MultiAssetPortfolio(
    initial_capital=100000,
    weighting='equal',  # or 'risk_parity', 'ranked'
    rebalance_frequency='monthly'
)

# Add positions
portfolio.add_position('AAPL', shares=100, price=150.0)
portfolio.add_position('MSFT', shares=50, price=300.0)

# Rebalance
portfolio.rebalance(current_prices={'AAPL': 155.0, 'MSFT': 310.0})
```

---

#### `src/backtesting/engine/pairs_portfolio.py`
**Purpose**: Synchronized execution of pairs trading strategies

**Key Classes**:
- `PairsPortfolio`: Portfolio manager for pairs strategies

**Key Features**:
- **Synchronized execution**: Both legs trade simultaneously
- **Position sizing**: Uses `PairsPositionSizer` for capital allocation
- **Direction tracking**: Tracks long/short/flat per symbol
- **Dollar-neutral**: Option to maintain market-neutral positions
- **Risk management**: Integrated position size limits and stop losses

**Initialization Parameters**:
- `symbol1`: First symbol in pair (e.g., 'AAPL')
- `symbol2`: Second symbol in pair (e.g., 'MSFT')
- `init_cash`: Initial capital (default: $100,000)
- `fees`: Trading fees per trade (default: 0.001 = 0.1%)
- `slippage`: Slippage per trade (default: 0.0)
- `position_sizer`: PairsPositionSizer instance (default: DollarNeutralSizer)
- `risk_config`: RiskConfig instance (optional)

**Key Methods**:
```python
def from_signals(
    data1: pd.DataFrame,
    data2: pd.DataFrame,
    signals: Dict,
    init_cash: float,
    fees: float,
    slippage: float,
    position_sizer: PairsPositionSizer
) -> PairsPortfolio:
    """Create portfolio from pair signals."""
    pass
```

**Execution Flow**:
1. Iterate through each timestamp
2. Check for exit signals → close both legs simultaneously
3. Check for entry signals → open both legs simultaneously
4. Calculate position sizes via PairsPositionSizer
5. Update equity curve
6. Log synchronized trades
7. Check stop losses

**Trade Dictionary Structure** (Pairs-Specific):
```python
{
    'type': 'entry' | 'exit',
    'timestamp': datetime,
    'entry_symbol1': str,  # e.g., 'AAPL'
    'entry_symbol2': str,  # e.g., 'MSFT'
    'entry_price1': float,
    'entry_price2': float,
    'entry_shares1': int,
    'entry_shares2': int,
    'entry_direction1': int,  # 1=long, -1=short
    'entry_direction2': int,
    'total_value': float,
    'fees': float,
    'pnl': float (exit only),
    'pnl_pct': float (exit only)
}
```

**Position Sizing**:
- Delegates to `PairsPositionSizer` classes
- Three strategies: DollarNeutral, VolatilityAdjusted, RiskParity
- Returns `(shares1, shares2)` tuple

**Risk Management**:
- Max position size per leg
- Total capital constraint
- Stop loss per pair
- No new entries if capital depleted

**Statistics Calculated**:
- Same as PortfolioSimulator (total return, Sharpe, drawdown, etc.)
- Additional pairs-specific metrics (spread P&L, correlation)

**Dependencies**: `pandas`, `numpy`, `PairsPositionSizer`, `RiskManager`, `logger`

**Usage Example**:
```python
from backtesting.engine.pairs_portfolio import PairsPortfolio
from backtesting.utils.pairs_position_sizer import DollarNeutralSizer

# Usually called internally by BacktestEngine
sizer = DollarNeutralSizer(position_pct=0.5)

portfolio = PairsPortfolio.from_signals(
    data1=aapl_data,
    data2=msft_data,
    signals={
        'AAPL': {
            'entries': entries1,
            'exits': exits1,
            'direction': directions1
        },
        'MSFT': {
            'entries': entries2,
            'exits': exits2,
            'direction': directions2
        }
    },
    init_cash=100000,
    fees=0.001,
    slippage=0.0,
    position_sizer=sizer
)

stats = portfolio.stats()
print(f"Total Return: {stats['Total Return [%]']:.2f}%")
```

**Integration with BacktestEngine**:
```python
# BacktestEngine automatically detects PairsStrategy
if isinstance(strategy, PairsStrategy):
    # Route to PairsPortfolio
    portfolio = PairsPortfolio.from_signals(...)
else:
    # Route to standard PortfolioSimulator
    portfolio = PortfolioSimulator.from_signals(...)
```

---

#### `src/backtesting/engine/metrics.py`
**Purpose**: Calculate performance metrics

**Key Functions**:
- `calculate_sharpe_ratio(returns, risk_free_rate)`: Sharpe ratio
- `calculate_sortino_ratio(returns, risk_free_rate)`: Sortino ratio
- `calculate_calmar_ratio(returns, max_drawdown)`: Calmar ratio
- `calculate_max_drawdown(equity_curve)`: Maximum drawdown
- `calculate_win_rate(trades)`: Win rate percentage

**Metrics Calculated**:
- **Returns**: Total, annual, monthly, daily
- **Risk**: Volatility, max drawdown, downside deviation
- **Risk-Adjusted**: Sharpe, Sortino, Calmar
- **Trade Stats**: Win rate, avg win/loss, profit factor
- **Advanced**: Beta, alpha, correlation (if benchmark)

**Dependencies**: `pandas`, `numpy`

**Usage Example**:
```python
from backtesting.engine.metrics import calculate_sharpe_ratio

sharpe = calculate_sharpe_ratio(returns_series, risk_free_rate=0.02)
```

---

### Utility Modules

#### `src/backtesting/utils/indicators.py`
**Purpose**: Technical indicators library

**Available Indicators** (15+):
- `sma(prices, window)`: Simple Moving Average
- `ema(prices, window)`: Exponential Moving Average
- `rsi(prices, window)`: Relative Strength Index
- `atr(high, low, close, window)`: Average True Range
- `macd(prices, fast, slow, signal)`: MACD
- `bollinger_bands(prices, window, std_dev)`: Bollinger Bands
- `stochastic(high, low, close, window)`: Stochastic Oscillator
- `adx(high, low, close, window)`: ADX
- `obv(close, volume)`: On-Balance Volume
- `vwap(high, low, close, volume)`: VWAP
- `pivots(high, low, close)`: Pivot points
- `ichimoku_cloud(high, low, close)`: Ichimoku Cloud
- `supertrend(high, low, close, period, multiplier)`: Supertrend
- `donchian_channel(high, low, window)`: Donchian Channel
- `keltner_channel(high, low, close, window, atr_mult)`: Keltner Channel

**Dependencies**: `pandas`, `numpy`

**Usage Example**:
```python
from backtesting.utils.indicators import sma, rsi, atr

# Simple Moving Average
sma_20 = sma(close_prices, window=20)
sma_50 = sma(close_prices, window=50)

# RSI
rsi_14 = rsi(close_prices, window=14)

# ATR (for volatility-based position sizing)
atr_14 = atr(high, low, close, window=14)
```

---

#### `src/backtesting/utils/position_sizer.py`
**Purpose**: Calculate position sizes using various methods

**Key Classes**:
- `PositionSizer`: Position sizing calculator

**5 Position Sizing Methods**:

1. **Fixed Percentage**:
   ```python
   shares = (capital * position_size_pct) / current_price
   # Example: ($100k * 10%) / $150 = 66 shares
   ```

2. **Fixed Dollar**:
   ```python
   shares = fixed_dollar_amount / current_price
   # Example: $10,000 / $150 = 66 shares
   ```

3. **Volatility-Based (ATR)**:
   ```python
   risk_per_share = atr * atr_multiplier
   shares = (capital * risk_pct) / risk_per_share
   # Example: ($100k * 2%) / ($5 * 2) = 200 shares
   ```

4. **Kelly Criterion**:
   ```python
   kelly_pct = (win_rate * avg_win - loss_rate * avg_loss) / avg_win
   shares = (capital * kelly_pct * kelly_fraction) / current_price
   ```

5. **Risk Parity**:
   ```python
   # Equal risk contribution across assets
   volatility_scaled_allocation = base_allocation / volatility
   shares = (capital * volatility_scaled_allocation) / current_price
   ```

**Dependencies**: `pandas`, `numpy`, `indicators` (for ATR)

**Usage Example**:
```python
from backtesting.utils.position_sizer import PositionSizer
from backtesting.utils.risk_config import RiskConfig

sizer = PositionSizer(risk_config=RiskConfig.moderate())

shares = sizer.calculate_shares(
    symbol='AAPL',
    current_price=150.0,
    available_cash=100000,
    atr=5.0  # Optional, for volatility-based
)
```

---

#### `src/backtesting/utils/pairs_position_sizer.py`
**Purpose**: Position sizing for pairs trading strategies

**Key Classes**:

1. **PairsPositionSizer** (Abstract Base):
   ```python
   class PairsPositionSizer(ABC):
       @abstractmethod
       def calculate_position_size(
           self,
           cash: float,
           price1: float,
           price2: float,
           hedge_ratio: float = 1.0,
           **kwargs
       ) -> Tuple[float, float]:
           """Calculate shares for both legs of the pair."""
           pass
   ```

2. **DollarNeutralSizer**:
   - **Purpose**: Equal dollar allocation to both legs
   - **Logic**: Split capital 50/50 between both symbols
   - **Parameters**: `position_pct` (default: 0.5 = 50% of capital)
   - **Use case**: Baseline pairs strategy, simplest approach

   ```python
   total_capital = cash * position_pct  # e.g., $100k * 0.5 = $50k
   capital_per_leg = total_capital / 2  # $25k per leg
   shares1 = capital_per_leg / price1
   shares2 = capital_per_leg / price2
   ```

3. **VolatilityAdjustedSizer**:
   - **Purpose**: Allocate inversely to volatility
   - **Logic**: Less capital to more volatile symbol, more to less volatile
   - **Parameters**: `position_pct` (0.5), `volatility1`, `volatility2`
   - **Use case**: Risk-balanced pairs trading

   ```python
   # Inverse volatility weights
   weight1 = (1 / volatility1) / (1/volatility1 + 1/volatility2)
   weight2 = 1 - weight1

   total_capital = cash * position_pct
   capital1 = total_capital * weight1
   capital2 = total_capital * weight2
   shares1 = capital1 / price1
   shares2 = capital2 / price2
   ```

4. **RiskParitySizer**:
   - **Purpose**: Equal risk contribution from both legs
   - **Logic**: Accounts for volatility AND correlation
   - **Parameters**: `position_pct` (0.5), `target_risk` (0.02), `correlation`
   - **Use case**: Advanced risk management with correlation awareness

   ```python
   # Calculate target portfolio volatility
   target_vol = target_risk

   # Solve for weights that equalize risk contribution
   # Account for correlation between symbols
   # More complex optimization (see implementation)
   ```

**Factory Function**:
```python
def create_pairs_sizer(
    method: str = 'dollar_neutral',
    position_pct: float = 0.5,
    **kwargs
) -> PairsPositionSizer:
    """
    Create pairs position sizer.

    Args:
        method: 'dollar_neutral' | 'volatility_adjusted' | 'risk_parity'
        position_pct: % of capital to deploy (0.0-1.0)
        **kwargs: Additional parameters for specific sizers
            - volatility1, volatility2 (for volatility_adjusted)
            - target_risk, correlation (for risk_parity)

    Returns:
        PairsPositionSizer instance
    """
```

**Common Parameters**:
- `position_pct`: Fraction of capital to deploy (default: 0.5 = 50%)
- `min_shares`: Minimum shares per leg (default: 1)
- `max_position_pct`: Max capital per leg (default: 0.5 = 50%)

**Dependencies**: `pandas`, `numpy`, `ABC`

**Usage Example**:
```python
from backtesting.utils.pairs_position_sizer import (
    create_pairs_sizer,
    DollarNeutralSizer,
    VolatilityAdjustedSizer,
    RiskParitySizer
)

# Method 1: Factory function
sizer = create_pairs_sizer(method='dollar_neutral', position_pct=0.5)

# Method 2: Direct instantiation
sizer = DollarNeutralSizer(position_pct=0.5)

# Calculate position sizes
shares1, shares2 = sizer.calculate_position_size(
    cash=100000,
    price1=150.0,  # AAPL
    price2=300.0,  # MSFT
    hedge_ratio=1.0
)
# Result: 166 shares AAPL, 83 shares MSFT (approx $25k each)

# Volatility-adjusted sizing
vol_sizer = VolatilityAdjustedSizer(position_pct=0.5)
shares1, shares2 = vol_sizer.calculate_position_size(
    cash=100000,
    price1=150.0,
    price2=300.0,
    hedge_ratio=1.0,
    volatility1=0.03,  # 3% daily vol
    volatility2=0.02   # 2% daily vol
)
# Result: More capital to MSFT (lower vol)

# Risk parity sizing
rp_sizer = RiskParitySizer(position_pct=0.5, target_risk=0.02)
shares1, shares2 = rp_sizer.calculate_position_size(
    cash=100000,
    price1=150.0,
    price2=300.0,
    hedge_ratio=1.0,
    volatility1=0.03,
    volatility2=0.02,
    correlation=0.7  # 70% correlation
)
# Result: Optimized for equal risk contribution
```

**Comparison Table**:

| Sizer | Pros | Cons | Best For |
|-------|------|------|----------|
| **DollarNeutral** | Simple, predictable | Ignores volatility | Quick testing, baseline |
| **VolatilityAdjusted** | Risk-aware, balanced | Ignores correlation | Most pairs strategies |
| **RiskParity** | Optimal risk balance | Complex, needs correlation | Advanced strategies |

**Integration with PairsPortfolio**:
```python
from backtesting.engine.pairs_portfolio import PairsPortfolio
from backtesting.utils.pairs_position_sizer import VolatilityAdjustedSizer

sizer = VolatilityAdjustedSizer(position_pct=0.5)

portfolio = PairsPortfolio.from_signals(
    data1=data1,
    data2=data2,
    signals=signals,
    init_cash=100000,
    position_sizer=sizer  # Pass sizer to portfolio
)
```

---

#### `src/backtesting/utils/risk_manager.py`
**Purpose**: Manage risk and enforce portfolio constraints

**Key Classes**:
- `RiskManager`: Risk management orchestrator

**Key Methods**:
- `can_enter(symbol, shares, price)`: Check if entry allowed
- `check_stop_loss(symbol, current_price)`: Check stop loss conditions
- `update_position(symbol, shares, entry_price)`: Track position
- `close_position(symbol)`: Close and remove position
- `get_portfolio_exposure()`: Calculate total exposure

**Risk Constraints Enforced**:
- **Max positions**: Limit concurrent positions (default: 10)
- **Max position size**: Single position cannot exceed X% (default: 30%)
- **Capital limits**: Total deployed ≤ available capital
- **Stop losses**: Automatic stop loss execution
- **Reserve cash**: Maintain minimum cash reserve (conservative mode)

**Stop Loss Types**:
1. **Percentage**: Fixed % below entry (e.g., 2%)
2. **ATR-based**: N * ATR below entry (e.g., 2 * ATR)
3. **Time-based**: Exit after N bars regardless of P&L
4. **Profit target**: Exit when profit reaches target

**Dependencies**: `pandas`, `logger`

**Usage Example**:
```python
from backtesting.utils.risk_manager import RiskManager
from backtesting.utils.risk_config import RiskConfig

manager = RiskManager(
    initial_capital=100000,
    risk_config=RiskConfig.moderate()
)

# Check if entry allowed
if manager.can_enter('AAPL', shares=100, price=150.0):
    manager.update_position('AAPL', shares=100, entry_price=150.0)

# Check stop loss
if manager.check_stop_loss('AAPL', current_price=147.0):
    manager.close_position('AAPL')
```

---

#### `src/backtesting/utils/risk_config.py`
**Purpose**: Risk management configuration dataclass

**Key Classes**:
- `RiskConfig`: Risk configuration dataclass

**Attributes**:
- `position_size_pct`: % of capital per trade (0.05 = 5%)
- `position_sizing_method`: 'fixed_percentage' | 'fixed_dollar' | 'volatility' | 'kelly' | 'risk_parity'
- `use_stop_loss`: Enable stop losses (bool)
- `stop_loss_pct`: Stop loss % (0.02 = 2%)
- `stop_loss_type`: 'percentage' | 'atr' | 'time' | 'profit_target'
- `max_positions`: Max concurrent positions (int)
- `max_position_pct`: Max single position size (0.30 = 30%)

**Preset Profiles**:
```python
RiskConfig.conservative():
    position_size_pct=0.05  # 5% per trade
    use_stop_loss=True
    stop_loss_pct=0.02      # 2% stop
    max_positions=5

RiskConfig.moderate():
    position_size_pct=0.10  # 10% per trade
    use_stop_loss=True
    stop_loss_pct=0.02
    max_positions=10

RiskConfig.aggressive():
    position_size_pct=0.20  # 20% per trade
    use_stop_loss=True
    stop_loss_pct=0.03      # 3% stop
    max_positions=15

RiskConfig.disabled():
    position_size_pct=0.99  # 99% per trade (testing only!)
    use_stop_loss=False
```

**Dependencies**: `dataclasses`

**Usage Example**:
```python
from backtesting.utils.risk_config import RiskConfig

# Use preset
config = RiskConfig.moderate()

# Custom config
config = RiskConfig(
    position_size_pct=0.15,
    position_sizing_method='volatility',
    use_stop_loss=True,
    stop_loss_type='atr',
    max_positions=8
)
```

---

#### `src/backtesting/utils/market_calendar.py`
**Purpose**: NYSE trading calendar for market day filtering

**Key Classes**:
- `MarketCalendar`: Trading calendar manager

**Key Methods**:
- `is_market_day(date)`: Check if date is trading day
- `filter_market_days(dates)`: Filter to trading days only
- `get_trading_hours()`: Get market hours (9:30 AM - 4:00 PM)
- `is_market_hours(timestamp)`: Check if timestamp in market hours

**Filters**:
- **Weekends**: Saturday, Sunday
- **NYSE Holidays**: New Year's Day, MLK Day, Presidents Day, Good Friday, Memorial Day, Independence Day, Labor Day, Thanksgiving, Christmas
- **Early closes**: Day before Thanksgiving, Christmas Eve (if weekday)

**Market Hours**:
- Regular: 9:30 AM - 4:00 PM EST
- Pre-market: 4:00 AM - 9:30 AM EST (not used by default)
- After-hours: 4:00 PM - 8:00 PM EST (not used by default)

**Dependencies**: `pandas`, `datetime`

**Usage Example**:
```python
from backtesting.utils.market_calendar import MarketCalendar

calendar = MarketCalendar()

# Check if trading day
if calendar.is_market_day('2023-01-02'):  # Monday after New Year's
    print("Market open")

# Filter DataFrame to trading days
trading_data = data[data.index.map(calendar.is_market_day)]
```

---

## Strategy Layer

### Strategy Registry

#### `src/strategies/registry.py`
**Purpose**: Unified strategy registry for dynamic strategy lookup by name

**Key Functions**:
- `get_strategy_class(name)`: Get strategy class by name or display name
- `list_strategies()`: List all available strategy class names
- `list_strategy_display_names()`: Get mapping of display names to class names
- `get_strategy_info(name)`: Get strategy info including parameters
- `register_strategy(name, cls, display_name)`: Register custom strategy at runtime

**Registered Strategies**:
- `MovingAverageCrossover`, `TripleMovingAverage`
- `MeanReversion`, `RSIMeanReversion`, `MeanReversionLongShort`
- `MomentumStrategy`, `BreakoutStrategy`
- `VolatilityTargetedMomentum`, `OvernightMeanReversion`
- `CrossSectionalMomentum`, `PairsTrading`

**Features**:
- Lazy loading to avoid import chain issues
- Supports both class names and display names
- Case-insensitive name lookup
- Strategy class caching

**Usage Example**:
```python
from src.strategies.registry import get_strategy_class, list_strategies

# Get strategy by name
strategy_cls = get_strategy_class("MovingAverageCrossover")
strategy_cls = get_strategy_class("Moving Average Crossover")  # Display name also works

# List available strategies
print(list_strategies())
# ['BreakoutStrategy', 'CrossSectionalMomentum', 'MeanReversion', ...]

# Get strategy info
info = get_strategy_info("MovingAverageCrossover")
print(info['parameters'])  # {'fast_window': 10, 'slow_window': 50}
```

---

### Base Strategy Classes

#### `src/backtesting/base/strategy.py`
**Purpose**: Base classes for all trading strategies

**Key Classes**:

1. **Strategy** (Abstract):
   ```python
   class Strategy(ABC):
       @abstractmethod
       def generate_signals(self, data: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
           """Generate entry and exit signals for single symbol."""
           pass
   ```

2. **MultiSymbolStrategy** (Abstract):
   - Inherits from `Strategy`
   - For strategies that trade multiple symbols simultaneously
   - Returns dict of signals per symbol
   ```python
   class MultiSymbolStrategy(Strategy):
       @abstractmethod
       def generate_signals_multi(self, data: Dict[str, pd.DataFrame]) -> Dict:
           """Generate signals for multiple symbols."""
           pass
   ```

**Signal Format**:
```python
# Single symbol
entries = pd.Series([False, False, True, False, ...], index=data.index)
exits = pd.Series([False, False, False, True, ...], index=data.index)

# Multi-symbol
signals = {
    'AAPL': {
        'entries': pd.Series([False, True, ...]),
        'exits': pd.Series([False, False, ...])
    },
    'MSFT': {
        'entries': pd.Series([False, True, ...]),
        'exits': pd.Series([False, False, ...])
    }
}
```

**Dependencies**: `pandas`, `ABC`

**Usage Example**:
```python
from backtesting.base.strategy import Strategy

class MyStrategy(Strategy):
    def __init__(self, param1, param2):
        self.param1 = param1
        self.param2 = param2

    def generate_signals(self, data):
        # Implement your logic
        entries = ...
        exits = ...
        return entries, exits
```

---

#### `src/backtesting/base/pairs_strategy.py`
**Purpose**: Base class for pairs trading strategies

**Key Classes**:
- `PairsStrategy`: Abstract base for pairs strategies

**Inheritance**:
```
Strategy (base)
  └─→ MultiSymbolStrategy
       └─→ PairsStrategy (adds pairs-specific requirements)
```

**Key Methods**:
```python
@abstractmethod
def generate_signals_multi(self, data: Dict[str, pd.DataFrame]) -> Dict:
    """
    Generate synchronized signals for both legs of the pair.

    Args:
        data: Dict with keys 'symbol1' and 'symbol2', each containing OHLCV DataFrame

    Returns:
        Dict with signals for both symbols including direction:
        {
            'symbol1': {
                'entries': pd.Series (bool),
                'exits': pd.Series (bool),
                'direction': pd.Series (int: 1=long, -1=short, 0=flat)
            },
            'symbol2': {
                'entries': pd.Series (bool),
                'exits': pd.Series (bool),
                'direction': pd.Series (int: 1=long, -1=short, 0=flat)
            }
        }
    """
    pass
```

**Pairs Requirements**:
- **Synchronized signals**: Both legs must trade simultaneously
- **Opposite directions**: When symbol1 is long, symbol2 is short (or vice versa)
- **Market neutral**: Dollar-neutral or risk-neutral positioning
- **Direction field**: Required for pairs to indicate long/short/flat

**Execution Routing**:
- BacktestEngine detects `PairsStrategy` via `isinstance()` check
- Routes to `PairsPortfolio` automatically
- Uses `PairsPositionSizer` for capital allocation

**Dependencies**: `MultiSymbolStrategy`, `pandas`, `ABC`

**Usage Example**:
```python
from backtesting.base.pairs_strategy import PairsStrategy
import pandas as pd

class MyPairsStrategy(PairsStrategy):
    def __init__(self, threshold=2.0):
        self.threshold = threshold

    def generate_signals_multi(self, data):
        symbol1_data = data['symbol1']
        symbol2_data = data['symbol2']

        # Calculate spread, z-score, etc.
        spread = symbol1_data['close'] - symbol2_data['close']
        zscore = (spread - spread.mean()) / spread.std()

        # Generate synchronized signals
        entries_long = zscore < -self.threshold  # Long spread
        entries_short = zscore > self.threshold  # Short spread
        exits = abs(zscore) < 0.5

        return {
            'symbol1': {
                'entries': entries_long | entries_short,
                'exits': exits,
                'direction': pd.Series(
                    np.where(entries_long, 1, np.where(entries_short, -1, 0)),
                    index=symbol1_data.index
                )
            },
            'symbol2': {
                'entries': entries_long | entries_short,
                'exits': exits,
                'direction': pd.Series(
                    np.where(entries_long, -1, np.where(entries_short, 1, 0)),  # Opposite
                    index=symbol2_data.index
                )
            }
        }
```

---

### Base Strategies

#### `src/strategies/base_strategies/moving_average.py`
**Purpose**: Moving average-based strategies

**Strategies**:

1. **MovingAverageCrossover**:
   - **Logic**: Buy when fast MA > slow MA, sell when fast MA < slow MA
   - **Parameters**: `fast_window` (default: 20), `slow_window` (default: 50)
   - **Use case**: Trend following

2. **TripleMovingAverage**:
   - **Logic**: Three-level MA system with filters
   - **Parameters**: `fast` (10), `medium` (20), `slow` (50)
   - **Use case**: Trend confirmation

**Dependencies**: `indicators.sma`, `pandas`

**Usage Example**:
```python
from strategies.moving_average import MovingAverageCrossover

strategy = MovingAverageCrossover(fast_window=10, slow_window=30)
entries, exits = strategy.generate_signals(data)
```

---

#### `src/strategies/base_strategies/momentum.py`
**Purpose**: Momentum-based strategies

**Strategies**:

1. **MomentumStrategy**:
   - **Logic**: Buy when price momentum > threshold
   - **Parameters**: `lookback_period` (20), `momentum_threshold` (0.02)
   - **Calculation**: `momentum = (close / close.shift(lookback)) - 1`

2. **BreakoutStrategy**:
   - **Logic**: Buy on breakout above N-day high
   - **Parameters**: `breakout_period` (20), `confirmation_bars` (2)
   - **Use case**: Trend initiation

**Dependencies**: `pandas`, `numpy`

**Usage Example**:
```python
from strategies.momentum import BreakoutStrategy

strategy = BreakoutStrategy(breakout_period=50)
entries, exits = strategy.generate_signals(data)
```

---

#### `src/strategies/base_strategies/mean_reversion.py`
**Purpose**: Mean reversion strategies

**Strategies**:

1. **MeanReversion**:
   - **Logic**: Buy when price below lower Bollinger Band
   - **Parameters**: `window` (20), `std_dev` (2.0)

2. **RSIMeanReversion**:
   - **Logic**: Buy when RSI < oversold, sell when RSI > overbought
   - **Parameters**: `rsi_period` (14), `oversold` (30), `overbought` (70)

**Dependencies**: `indicators.bollinger_bands`, `indicators.rsi`, `pandas`

**Usage Example**:
```python
from strategies.mean_reversion import RSIMeanReversion

strategy = RSIMeanReversion(oversold=25, overbought=75)
entries, exits = strategy.generate_signals(data)
```

---

### Advanced Strategies

#### `src/strategies/advanced/volatility_targeted_momentum.py`
**Purpose**: Volatility-scaled momentum strategy

**Logic**:
- Calculate momentum score (returns over lookback period)
- Scale position size inversely to volatility
- Target constant volatility contribution

**Parameters**:
- `lookback_period`: Momentum lookback (60)
- `volatility_window`: Volatility calculation window (20)
- `target_volatility`: Target annual volatility (0.15 = 15%)

**Use Case**: Consistent risk-adjusted returns across volatility regimes

**Dependencies**: `indicators.sma`, `pandas`, `numpy`

---

#### `src/strategies/advanced/overnight_mean_reversion.py`
**Purpose**: Exploit overnight gap reversions

**Logic**:
- Calculate overnight gap: `gap = open / prev_close - 1`
- Buy if large gap down (expecting reversion)
- Sell intraday if gap fills

**Parameters**:
- `gap_threshold`: Minimum gap to trade (0.02 = 2%)
- `exit_time`: Intraday exit time ('15:00')

**Use Case**: Intraday mean reversion on overnight gaps

**Dependencies**: `pandas`

---

#### `src/strategies/advanced/cross_sectional_momentum.py`
**Purpose**: Momentum-based ranking across universe

**Logic**:
- Rank all symbols by momentum score
- Buy top N, sell bottom N
- Rebalance periodically

**Parameters**:
- `lookback_period`: Momentum lookback (126 days)
- `top_n`: Number of longs (10)
- `rebalance_frequency`: Rebalance period ('monthly')

**Use Case**: Long-short equity strategies

**Dependencies**: `ranking`, `pandas`

---

#### `src/strategies/advanced/pairs_trading.py`
**Purpose**: Statistical arbitrage between correlated pairs (market-neutral)

**Key Classes**:
- `PairsTrading`: Concrete pairs trading strategy

**Logic**:
1. **Pair Selection**: Test cointegration using Engle-Granger test
2. **Spread Calculation**: `spread = price1 - hedge_ratio * price2`
3. **Z-score**: `zscore = (spread - mean) / std`
4. **Entry**: Open when |zscore| > entry_threshold
5. **Exit**: Close when |zscore| < exit_threshold or stop loss hit

**Parameters**:
- `pair_selection_window`: Cointegration lookback (252 days)
- `cointegration_pvalue`: Max p-value for cointegration (0.05)
- `entry_zscore`: Z-score entry threshold (2.0)
- `exit_zscore`: Z-score exit threshold (0.5)
- `stop_loss_zscore`: Stop loss threshold (3.5)
- `zscore_window`: Rolling window for spread stats (20)

**Trading Logic**:
```python
# Long pair: Buy symbol1, sell symbol2
if zscore < -entry_zscore:
    enter_long_pair()

# Short pair: Sell symbol1, buy symbol2
if zscore > entry_zscore:
    enter_short_pair()

# Exit on mean reversion
if abs(zscore) < exit_zscore:
    exit_pair()

# Stop loss
if abs(zscore) > stop_loss_zscore:
    exit_pair()
```

**Signal Format** (Multi-Symbol):
```python
signals = {
    'AAPL': {
        'entries': pd.Series([False, True, ...]),
        'exits': pd.Series([False, False, ...]),
        'direction': pd.Series([0, 1, ...])  # 1=long, -1=short
    },
    'MSFT': {
        'entries': pd.Series([False, True, ...]),
        'exits': pd.Series([False, False, ...]),
        'direction': pd.Series([0, -1, ...])  # Opposite direction
    }
}
```

**Execution Requirements**:
- Requires `PairsPortfolio` for synchronized execution
- BacktestEngine automatically routes to PairsPortfolio
- Position sizing via `PairsPositionSizer` classes

**Use Case**: Market-neutral arbitrage, low correlation to market

**Dependencies**: `PairsStrategy`, `statsmodels.tsa.stattools`, `pandas`, `numpy`

**Usage Example**:
```python
from strategies.advanced.pairs_trading import PairsTrading
from backtesting.engine.backtest_engine import BacktestEngine

strategy = PairsTrading(
    pair_selection_window=252,
    entry_zscore=2.0,
    exit_zscore=0.5,
    zscore_window=20
)

engine = BacktestEngine(initial_capital=100000)
portfolio = engine.run(
    strategy=strategy,
    symbols=['AAPL', 'MSFT'],  # Must provide exactly 2 symbols
    start_date='2020-01-01',
    end_date='2022-12-31'
)
```

---

### Regime Analysis & Validation Modules

#### `src/backtesting/regimes/detector.py`
**Purpose**: Detect different market regimes (trend, volatility, drawdown)

**Key Classes**:
- `RegimeDetector`: Market regime detector

**Key Methods**:
- `detect_trend_regimes(prices, lookback)`: Detect bull/bear/sideways markets
- `detect_volatility_regimes(prices, lookback)`: Detect high/low volatility periods
- `detect_drawdown_regimes(prices, threshold)`: Detect drawdown/recovery/calm periods

**Regime Types**:

1. **Trend Regimes** (based on moving average slope):
   - Bull: Price trending up (MA slope > threshold)
   - Bear: Price trending down (MA slope < -threshold)
   - Sideways: Price range-bound

2. **Volatility Regimes** (based on rolling volatility):
   - High Volatility: Vol > median
   - Low Volatility: Vol ≤ median

3. **Drawdown Regimes** (based on peak-to-trough):
   - Drawdown: Currently in drawdown > threshold
   - Recovery: Recovering from drawdown
   - Calm: Not in drawdown

**Dependencies**: `pandas`, `numpy`

**Usage Example**:
```python
from backtesting.regimes.detector import RegimeDetector

detector = RegimeDetector(
    trend_lookback=60,  # 60-day MA for trend
    vol_lookback=20,    # 20-day vol calculation
    drawdown_threshold=10.0  # 10% drawdown threshold
)

# Detect regimes
trend_regimes = detector.detect_trend_regimes(prices)
vol_regimes = detector.detect_volatility_regimes(prices)
dd_regimes = detector.detect_drawdown_regimes(prices)
```

---

#### `src/backtesting/regimes/analyzer.py`
**Purpose**: Analyze strategy performance across market regimes

**Key Classes**:
- `RegimeAnalyzer`: Regime-based performance analyzer

**Key Methods**:
- `analyze(portfolio_returns, market_prices, trades)`: Full regime analysis
- `calculate_robustness_score()`: Calculate consistency score (0-100)
- `format_results()`: Format results for display

**Analysis Output**:
- **Per-Regime Performance**: Sharpe, returns, drawdown, trade count
- **Robustness Score**: 0-100 metric measuring consistency across regimes
- **Best/Worst Regimes**: Identifies strengths and weaknesses

**Robustness Score Calculation**:
```python
# Variance of Sharpe ratios across regimes (lower is better)
variance_penalty = np.std(regime_sharpes) * 20
# Negative regime penalty (more negative regimes = lower score)
negative_penalty = (num_negative_regimes / total_regimes) * 30
robustness = max(0, 100 - variance_penalty - negative_penalty)
```

**Interpretation**:
- **80-100**: Excellent - Highly consistent across all conditions
- **60-80**: Good - Reasonable consistency
- **40-60**: Fair - Some regime-specific weaknesses
- **< 40**: Needs improvement - Inconsistent performance

**Dependencies**: `pandas`, `numpy`, `RegimeDetector`

**Usage Example**:
```python
from backtesting.regimes.analyzer import RegimeAnalyzer

analyzer = RegimeAnalyzer(
    trend_lookback=60,
    vol_lookback=20,
    drawdown_threshold=10.0
)

results = analyzer.analyze(
    portfolio_returns=portfolio.returns(),
    market_prices=spy_prices,
    trades=portfolio.trades
)

print(f"Robustness Score: {results['robustness_score']}")
```

---

#### `src/backtesting/chunking/walk_forward.py`
**Purpose**: Walk-forward validation for strategy testing

**Key Classes**:
- `WalkForwardValidator`: Rolling train/test window validator

**Key Methods**:
- `run(strategy_class, param_grid, data, train_days, test_days)`: Execute validation
- `optimize_window(train_data, param_grid)`: Optimize params on training window
- `test_window(test_data, best_params)`: Test on out-of-sample window

**Walk-Forward Process**:
```
Timeline:
[-------Train 1-------][--Test 1--]
                [-------Train 2-------][--Test 2--]
                                [-------Train 3-------][--Test 3--]
```

**Key Features**:
- **Rolling windows**: Train on past N days, test on next M days
- **Parameter optimization**: Grid search on training window
- **Out-of-sample testing**: No lookahead bias
- **Degradation tracking**: Compare in-sample vs out-of-sample performance

**Validation Output**:
```python
{
    'windows': [
        {
            'train_start': '2023-01-01',
            'train_end': '2023-06-30',
            'test_start': '2023-07-01',
            'test_end': '2023-09-30',
            'best_params': {'fast': 10, 'slow': 50},
            'train_return': 15.2,
            'test_return': 8.7,
            'degradation_pct': 42.8
        },
        # ... more windows
    ],
    'avg_train_return': 14.5,
    'avg_test_return': 9.2,
    'avg_degradation': 36.5,
    'stability_score': 68.0
}
```

**Dependencies**: `BacktestEngine`, `pandas`, `itertools`

**Usage Example**:
```python
from backtesting.chunking.walk_forward import WalkForwardValidator
from strategies.moving_average import MovingAverageCrossover

validator = WalkForwardValidator(
    engine=engine,
    symbol='AAPL',
    train_days=180,  # 6 months training
    test_days=90     # 3 months testing
)

param_grid = {
    'fast_window': [10, 20, 30],
    'slow_window': [50, 100, 200]
}

results = validator.run(
    strategy_class=MovingAverageCrossover,
    param_grid=param_grid,
    start_date='2023-01-01',
    end_date='2024-01-01'
)

print(f"Avg Test Return: {results['avg_test_return']:.2f}%")
print(f"Degradation: {results['avg_degradation']:.2f}%")
```

---

#### `backtest_engine.py` - Regime Analysis Integration
**Level 1 Enhancement**: Transparent regime analysis

**New Parameters**:
- `enable_regime_analysis`: Enable automatic regime analysis (default: False)

**New Methods**:
- `_print_regime_analysis(portfolio)`: Print regime-based performance breakdown

**Features Added**:
- Automatic data caching for regime analysis
- Daily resampling for regime detection
- Console output with regime performance tables
- Robustness scoring

**Usage Example**:
```python
# Enable regime analysis automatically
engine = BacktestEngine(
    initial_capital=100000,
    enable_regime_analysis=True  # NEW: Auto-analyze by regime
)

portfolio = engine.run(
    strategy=strategy,
    symbols='AAPL',
    start_date='2023-01-01',
    end_date='2024-01-01'
)
# Regime analysis printed automatically after backtest results
```

---

#### GUI Integration - Level 2
**Files Modified**: `setup_view.py`, `app.py`, `gui_controller.py`

**New GUI Elements**:
- Regime analysis checkbox in SetupView (Output Settings section)
- Tooltip: "Automatically analyze strategy performance across different market regimes"
- State persistence with saved configurations

**Data Flow**:
```
SetupView (checkbox)
  └─→ App (_on_run_backtests)
       └─→ GUIBacktestController (start_backtests)
            └─→ BacktestEngine (enable_regime_analysis parameter)
```

**User Experience**:
1. User checks "Enable regime analysis" checkbox
2. Runs backtest normally
3. Regime analysis appears in terminal output after results
4. No GUI changes to results view (terminal output only)

---

## Visualization Layer

### `src/visualization/integration.py`
**Purpose**: Unified interface to visualization pipeline

**Key Classes**:
- `BacktestVisualizer`: Main visualization orchestrator

**Key Methods**:
- `generate_all(portfolio, strategy, data)`: Generate all visualizations
- `generate_tearsheet(portfolio)`: QuantStats tearsheet
- `generate_charts(data, trades)`: Price charts with trade markers
- `generate_report(portfolio, strategy)`: Summary report

**Dependencies**: `QuantStatsReporter`, `Charts`, `ReportGenerator`

---

### `src/visualization/reports/quantstats_reporter.py`
**Purpose**: QuantStats integration for professional tearsheets

**Key Classes**:
- `QuantStatsReporter`: QuantStats wrapper

**Key Methods**:
- `create_tearsheet(returns, benchmark, output_file)`: Full tearsheet
- `create_html(returns, benchmark)`: HTML tearsheet
- `create_pdf(returns, benchmark)`: PDF tearsheet

**50+ Metrics Included**:
- Returns: Total, CAGR, YTD, MTD
- Risk: Volatility, max drawdown, downside deviation
- Risk-Adjusted: Sharpe, Sortino, Calmar
- Drawdown: Max, avg, recovery time
- Monthly/Yearly returns heatmaps
- Rolling metrics charts

**Dependencies**: `quantstats`, `pandas`

---

### `src/visualization/charts/`
**Purpose**: Chart generation modules

**Modules**:
- `candlestick.py`: Interactive candlestick charts (Plotly)
- `mplfinance_chart.py`: Technical charts with indicators (mplfinance)

**Dependencies**: `plotly`, `mplfinance`, `matplotlib`

---

## GUI Layer

### `src/gui/app.py`
**Purpose**: Main Flet GUI application

**Key Classes**:
- `BacktestApp`: Main app class

**Key Methods**:
- `build()`: Build UI layout
- `navigate_to(view)`: Navigate between views
- `run()`: Run Flet app

**Views Managed**:
- SetupView: Configuration
- RunView: Execution monitoring
- ResultsView: Results display

**Dependencies**: `flet`, `views`, `workers`

---

### `src/gui/workers/gui_controller.py`
**Purpose**: Thread-safe GUI controller

**Key Classes**:
- `GUIBacktestController`: Worker controller

**Thread Safety**:
- Worker thread runs backtest
- Main thread renders UI
- Communication via `queue.Queue`

**Key Methods**:
- `start(strategy, symbols, start, end)`: Start backtest in background
- `poll_updates()`: Check queue for updates (call from main thread)
- `is_running()`: Check if backtest in progress

**Dependencies**: `threading`, `queue`, `SweepRunner`

---

## Utility Layer

### `src/utils/logger.py`
**Purpose**: Centralized Rich-based logging

**Key Functions**:
- `get_logger(name)`: Get logger instance
- Color-coded output:
  - 🟢 Green: Success, completions
  - 🔵 Blue: Info, progress
  - 🟡 Yellow: Warnings
  - 🔴 Red: Errors, failures

**Log Levels**:
- DEBUG, INFO, WARNING, ERROR, CRITICAL

**Usage Example**:
```python
from utils.logger import get_logger
logger = get_logger(__name__)

logger.info("Starting backtest...")
logger.success("Backtest complete!")
logger.error("Failed to load data")
```

**Dependencies**: `rich`, `logging`

---

### `src/utils/cache_manager.py`
**Purpose**: Caching for expensive operations

**Key Classes**:
- `CacheManager`: Cache manager

**Cached Operations**:
- Symbol lists
- Metadata
- Frequently accessed data

**Dependencies**: `functools.lru_cache`, `pathlib`

---

## Module Dependency Graph

```
GUI Layer
  └─→ Workers (GUIBacktestController)
       └─→ Backtesting Engine (SweepRunner, BacktestEngine)

Visualization Layer
  └─→ Backtesting Engine (Portfolio results)

Backtesting Engine Layer
  ├─→ Strategy Layer (signal generation)
  ├─→ Data Layer (data loading)
  └─→ Utilities (indicators, risk management)

Strategy Layer
  └─→ Utilities (indicators)

Data Layer
  └─→ External APIs (Alpaca)
```

---

## Module Update Checklist

When adding/modifying modules, update:
- [ ] This document (MODULE_REFERENCE.md)
- [ ] Architecture overview
- [ ] Data flow diagrams
- [ ] API reference (if public API)
- [ ] User guides (if user-facing)

---

---

## Trading System Layer

### Broker Interface Architecture (ISP-Compliant)

The broker abstraction layer follows the **Interface Segregation Principle (ISP)**, providing focused, composable interfaces instead of one monolithic interface. This enables:
- Support for multiple broker backends (Alpaca, TastyTrade, IBKR, etc.)
- Options trading support without affecting stock-only brokers
- Clean separation of concerns

**Interface Hierarchy**:
```
AccountInterface              ← Account info, connection testing
MarketHoursInterface          ← Market schedule queries
MarketDataInterface           ← Quotes, trades, historical bars
OrderManagementInterface      ← Order retrieval, cancellation
  ├─ StockTradingInterface    ← Stock positions, orders (inherits OrderManagement)
  └─ OptionsTradingInterface  ← Options chains, orders (inherits OrderManagement)

BrokerInterface               ← Composite: Account + MarketHours + MarketData + StockTrading
```

---

### `src/trading/brokers/interfaces/` (Directory)
**Purpose**: Focused, composable interfaces following ISP

**Files**:
- `base.py`: Shared enums (`OrderSide`, `OrderType`, `OrderStatus`, `TimeInForce`, `OptionType`, `OptionRight`) and exceptions (`BrokerError`, `InvalidOrderError`, etc.)
- `account.py`: `AccountInterface` - `get_account()`, `test_connection()`
- `market_hours.py`: `MarketHoursInterface` - `is_market_open()`, `get_market_hours(date)`
- `market_data.py`: `MarketDataInterface` - `get_latest_quote()`, `get_latest_trade()`, `get_bars()`
- `order_management.py`: `OrderManagementInterface` - `cancel_order()`, `get_order()`, `get_orders()`
- `stock_trading.py`: `StockTradingInterface` - `get_stock_positions()`, `place_stock_order()`, `close_stock_position()`
- `options_trading.py`: `OptionsTradingInterface` - `get_option_chain()`, `place_option_order()`, `OptionLeg` dataclass

**Dependencies**: `ABC`, `Enum`, `dataclasses`, `datetime`, `pandas`

---

### `src/trading/brokers/broker_interface.py`
**Purpose**: Composite interface for backward compatibility

**Key Classes**: `BrokerInterface` (inherits `AccountInterface`, `MarketHoursInterface`, `MarketDataInterface`, `StockTradingInterface`)

**Backward Compatibility Aliases**:
- `get_positions()` → `get_stock_positions()`
- `get_position()` → `get_stock_position()`
- `place_order()` → `place_stock_order()`
- `close_position()` → `close_stock_position()`
- `close_all_positions()` → `close_all_stock_positions()`

**Note**: Does NOT inherit `OptionsTradingInterface` - options support is opt-in per broker

**Dependencies**: `interfaces/*`, `ABC`

---

### `src/trading/brokers/alpaca_broker.py`
**Purpose**: Alpaca Markets API implementation

**Features**: Paper/live trading, real-time data, WebSocket streaming, historical bars

**Implements**: `BrokerInterface` (full stock trading support)

**Method Names**: Uses new interface methods (`get_stock_positions`, `place_stock_order`, etc.)

**Configuration**: Via `.env` (ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_PAPER)

**Dependencies**: `alpaca-py`, `broker_interface`, `logger`

---

### `src/trading/brokers/broker_factory.py`
**Purpose**: Factory for broker instantiation

**Methods**: `create_from_env()`, `create_from_yaml()`, `create_broker()`

**Dependencies**: `broker_interface`, `alpaca_broker`, `dotenv`, `yaml`

---

### `src/trading/core/execution_engine.py`
**Purpose**: Order execution with retry logic

**Features**: Automatic retry, fill timeout, error handling

**Config**: max_retries=3, retry_delay=1.0s, fill_timeout=30s

**Dependencies**: `broker_interface`, `logger`, `time`

---

### `src/trading/core/position_manager.py`
**Purpose**: Risk management and position tracking

**Features**: Position limits, size limits, stop losses, exposure tracking

**Config**: max_positions, max_position_size, stop_loss_pct

**Dependencies**: `broker_interface`, `logger`, `datetime`

---

### `src/trading/core/paper_trading_bot.py`
**Purpose**: Scheduled trading bot

**Features**: Schedule-based execution, market hours awareness, auto logging

**Dependencies**: `schedule`, `execution_engine`, `position_manager`, `logger`

---

### `src/trading/adapters/strategy_adapter.py`
**Purpose**: Base adapter interface

**Methods**: `generate_signals()`, `get_required_symbols()`, `update_data()`

**Dependencies**: `ABC`, `pandas`

---

### `src/trading/adapters/omr_live_adapter.py`
**Purpose**: Overnight Mean Reversion adapter

**Features**: Intraday data fetch, Bayesian probability, regime filtering

**Dependencies**: `strategy_adapter`, `bayesian_reversion_model`, `market_regime_detector`, `yfinance`

---

### `src/trading/adapters/ma_live_adapter.py`
**Purpose**: Moving Average crossover adapter

**Features**: Real-time MA calculation, trend confirmation

**Dependencies**: `strategy_adapter`, `pandas`

---

### `src/trading/strategies/omr_live_strategy.py`
**Purpose**: Live OMR strategy implementation

**Features**: 3:50 PM entry, next morning exit, Bayesian signals

**Dependencies**: `omr_live_adapter`, `broker_interface`, `logger`

---

### `src/trading/utils/portfolio_health_check.py`
**Purpose**: Pre-trade validation

**Classes**: `HealthCheckResult`, `PortfolioHealthChecker`

**Checks**: Buying power, portfolio value, position limits, capital requirements

**Dependencies**: `broker_interface`, `logger`, `dataclasses`

---

### `src/strategies/advanced/bayesian_reversion_model.py`
**Purpose**: Probabilistic reversion forecasting

**Training**: 10 years historical data, regime × move bucket combinations

**Output**: Win rate, expected return, Sharpe ratio, confidence

**Dependencies**: `pandas`, `numpy`, `pickle`, `market_regime_detector`

---

### `src/strategies/advanced/market_regime_detector.py`
**Purpose**: Market condition classification

**Regimes**: STRONG_BULL, WEAK_BULL, SIDEWAYS, UNPREDICTABLE, BEAR

**Features**: SPY 200MA, VIX thresholds, drawdown detection

**Dependencies**: `pandas`, `yfinance`, `Enum`

---

### `src/strategies/advanced/overnight_mean_reversion.py`
**Purpose**: Backtest OMR strategy

**Features**: Intraday-to-overnight reversion, regime filtering

**Dependencies**: `Strategy`, `pandas`, `bayesian_reversion_model`

---

### `src/strategies/advanced/overnight_signal_generator.py`
**Purpose**: Signal generation for OMR

**Features**: Probability calculation, confidence scoring

**Dependencies**: `pandas`, `bayesian_reversion_model`, `market_regime_detector`

---

### `src/utils/logger.py`
**Purpose**: Centralized logging system

**Classes**: `Logger`, `CSVLogger`, `TradingLogger`

**Features**: Color-coded console, CSV logging, file logging, trading-specific methods

**Dependencies**: `logging`, `pathlib`, `csv`, `datetime`

---

## Trading Scripts Layer

### `scripts/trading/test_alpaca_connection.py`
**Purpose**: Test Alpaca API connection

**Output**: Account info, positions, market status, quote fetching

---

### `scripts/trading/test_live_portfolio_health.py`
**Purpose**: Test portfolio health checks with live broker

**Output**: Pre-entry/pre-exit validation, broker data type verification

---

### `scripts/trading/execute_test_trade.py`
**Purpose**: Execute small test trade to validate pipeline

**Output**: Full trade cycle (buy 1 SPY, verify, close)

---

### `scripts/trading/close_test_position.py`
**Purpose**: Close test positions

**Output**: Position closure verification

---

### `scripts/trading/demo_omr_paper_trading.py`
**Purpose**: Demo overnight mean reversion paper trading

**Output**: Live OMR strategy execution

---

### `scripts/trading/demo_ma_paper_trading.py`
**Purpose**: Demo moving average paper trading

**Output**: Live MA crossover strategy execution

---

### `scripts/trading/run_live_paper_trading.py`
**Purpose**: Main live paper trading runner

**Features**: `TradingSessionTracker` with CSV logging

---

## Infrastructure & Deployment Layer

**Purpose**: AWS cloud deployment infrastructure with automated scheduling and monitoring

**Location**: `terraform/`, `scripts/ec2/`, systemd service files

**Total Components**: 20+ infrastructure resources + 10 management scripts

---

### Terraform Infrastructure as Code

#### `terraform/main.tf`
**Purpose**: Core infrastructure definition
- EC2 instance (t4g.small ARM64)
- Security group (SSH access only)
- Elastic IP (static IP address)
- EBS volume (8 GB encrypted)

#### `terraform/scheduled_start_stop.tf`
**Purpose**: Automated scheduling infrastructure
- Lambda functions for start/stop
- EventBridge cron rules (9 AM start, 4:30 PM stop ET Mon-Fri)
- IAM roles and policies
- CloudWatch Log Groups (90-day retention)

#### `terraform/user-data.sh`
**Purpose**: Bootstrap script for EC2 instance
- Installs Python 3.11 and dependencies
- Clones repository from GitHub
- Creates systemd service
- Configures environment variables

#### `terraform/lambda/start_instance.py`
**Purpose**: Lambda function to start EC2 instance
- Triggered by EventBridge at 9 AM ET weekdays

#### `terraform/lambda/stop_instance.py`
**Purpose**: Lambda function to stop EC2 instance
- Triggered by EventBridge at 4:30 PM ET weekdays

---

### SSH Management Scripts

**Location**: `scripts/ec2/`

**Purpose**: Quick-access scripts for monitoring and managing cloud-deployed bot

#### `scripts/ec2/connect.bat` / `connect.sh`
**Purpose**: SSH into EC2 instance
- Windows batch and Linux/Mac shell versions

#### `scripts/ec2/check_bot.bat` / `check_bot.sh`
**Purpose**: Check bot status and recent activity
- Shows systemd service status
- Displays last 10 log lines

#### `scripts/ec2/view_logs.bat` / `view_logs.sh`
**Purpose**: Stream live bot logs
- Real-time log monitoring
- Press Ctrl+C to stop

#### `scripts/ec2/restart_bot.bat` / `restart_bot.sh`
**Purpose**: Restart trading bot service
- Restarts systemd service
- Shows status after restart

#### `scripts/ec2/daily_health_check.bat` / `daily_health_check.sh`
**Purpose**: Automated 6-point health validation
- Instance state check
- Bot service status
- Recent errors count
- Resource usage (memory/CPU)
- Last activity logs
- Market status

#### `scripts/ec2/view_logs_plain.bat`
**Purpose**: View logs without ANSI colors
- For Windows CMD compatibility

---

### Systemd Service

#### `/etc/systemd/system/homeguard-trading.service` (on EC2)
**Purpose**: Systemd service configuration for bot
- Auto-restart on failure (10-second delay)
- Resource limits (1GB RAM, 150% CPU)
- Logging to systemd journal
- Runs as ec2-user (non-root)

**Command**: `python scripts/trading/run_live_paper_trading.py --strategy omr`

---

### Infrastructure Documentation

#### `docs/INFRASTRUCTURE_OVERVIEW.md`
**Purpose**: Complete AWS architecture documentation
- Infrastructure diagrams
- Resource breakdown (16 resources)
- Daily operation flow
- Cost breakdown (~$7/month)
- Management commands

#### `terraform/README.md`
**Purpose**: Terraform deployment guide
- Prerequisites and setup
- Configuration options
- Common operations
- Troubleshooting
- Post-deployment management

#### `scripts/ec2/SSH_SCRIPTS_README.md`
**Purpose**: SSH scripts documentation
- Script descriptions and usage
- Troubleshooting guide
- Manual commands reference

#### `HEALTH_CHECK_CHEATSHEET.md`
**Purpose**: Comprehensive monitoring guide
- Daily health check routines
- Common issues and fixes
- Advanced monitoring commands
- Lambda scheduler verification

---

**Infrastructure Summary**:
- **16 AWS resources**: EC2, Lambda (2), EventBridge (4), IAM (4), CloudWatch Logs (2), EBS, Security Group, Elastic IP
- **10 management scripts**: Windows & Linux/Mac versions (5 each)
- **1 systemd service**: Auto-restart trading bot
- **4 documentation files**: Complete setup and monitoring guides
- **Cost**: ~$7/month with automated scheduling
- **Uptime**: Monday-Friday 9 AM - 4:30 PM ET (automated)

---

**Last Updated**: 2025-11-25
**Total Modules**: 100+ modules across 9 major components (including infrastructure)
**Lines of Code**: ~47,000 LOC

**Recent Additions** (2025-11-25):
- Broker Interface Refactoring - ISP-Compliant Design (15 files)
  - 7 new focused interface files in `src/trading/brokers/interfaces/`
  - `BrokerInterface` now composite of focused interfaces (backward compatible)
  - Backward-compatible method aliases for existing code
  - Standardized return types (`cancel_order` → bool, `get_bars` → DataFrame)
  - New `OptionsTradingInterface` for future options support
  - 39 new interface compliance tests (`tests/trading/test_interface_compliance.py`)
  - All 131 trading tests pass, OMR live strategy unaffected

**Recent Additions** (2025-11-15):
- Infrastructure & Deployment Layer (30+ new components)
  - Complete Terraform infrastructure as code (5 .tf files)
  - Lambda-powered automated scheduling (2 Lambda functions)
  - SSH management scripts (10 scripts for Windows & Linux/Mac)
  - Systemd service with auto-restart
  - Comprehensive infrastructure documentation (4 guides)
  - ~$7/month AWS deployment with 46% cost savings vs 24/7

**Recent Additions** (2025-11-14):
- Live trading system (16 new modules)
  - Complete broker abstraction layer with Alpaca implementation
  - Execution engine with retry logic and risk management
  - Strategy adapters for live trading (OMR, MA)
  - Portfolio health checks and pre-trade validation
  - Bayesian probability model for overnight mean reversion
  - Market regime detection system (5 regimes)
  - Centralized logging with CSV audit trails
  - 10+ trading scripts for testing and execution

**Previous Additions** (2025-11-11):
- Pairs trading framework (4 new modules)
  - `src/backtesting/base/pairs_strategy.py`: PairsStrategy base class
  - `src/strategies/advanced/pairs_trading.py`: PairsTrading implementation
  - `src/backtesting/engine/pairs_portfolio.py`: PairsPortfolio executor
  - `src/backtesting/utils/pairs_position_sizer.py`: Position sizing strategies
- Comprehensive test suite: 148+ pairs-specific tests
- Integration with GridSearchOptimizer for multi-symbol support
