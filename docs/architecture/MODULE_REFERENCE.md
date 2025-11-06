# Homeguard Module Reference

**Version**: 1.0
**Last Updated**: 2025-11-05
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

### `src/config.py`
**Purpose**: Global configuration and settings management

**Key Classes**:
- `Config`: Singleton configuration manager

**Key Functions**:
- `load_settings()`: Loads from `settings.ini`
- `get_data_directory()`: Returns data storage path
- `get_log_directory()`: Returns log output path

**Configuration Sections**:
- OS-specific paths (Windows/macOS/Linux)
- Data directory location
- Log output directory
- Tearsheet frequency settings

**Dependencies**: `configparser`, `pathlib`

**Usage Example**:
```python
from config import Config
config = Config()
data_dir = config.get_data_directory()
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
**Purpose**: CLI entry point for running backtests

**Key Functions**:
- `main()`: CLI argument parser and execution
- `run_single_backtest()`: Single symbol backtest
- `run_sweep()`: Multi-symbol backtest
- `run_optimization()`: Parameter optimization (future)

**CLI Arguments**:
- `--strategy`: Strategy class name
- `--symbols`: Ticker symbols (comma-separated)
- `--start`: Start date (YYYY-MM-DD)
- `--end`: End date (YYYY-MM-DD)
- `--initial-capital`: Starting capital
- `--fees`: Trading fees (decimal)
- `--slippage`: Slippage (decimal)
- `--risk-profile`: Risk profile (conservative/moderate/aggressive)

**Dependencies**: `argparse`, `BacktestEngine`, `strategies`

**Usage Example**:
```bash
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

### Base Strategy Classes

#### `src/backtesting/base/strategy.py`
**Purpose**: Base classes for all trading strategies

**Key Classes**:

1. **BaseStrategy** (Abstract):
   ```python
   class BaseStrategy(ABC):
       @abstractmethod
       def generate_signals(self, data: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
           """Generate entry and exit signals."""
           pass
   ```

2. **LongOnlyStrategy** (Abstract):
   - Inherits from `BaseStrategy`
   - Enforces long-only constraint
   - All concrete strategies inherit from this

**Signal Format**:
```python
entries = pd.Series([False, False, True, False, ...], index=data.index)
exits = pd.Series([False, False, False, True, ...], index=data.index)
```

**Dependencies**: `pandas`, `ABC`

**Usage Example**:
```python
from backtesting.base.strategy import LongOnlyStrategy

class MyStrategy(LongOnlyStrategy):
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
**Purpose**: Statistical arbitrage between correlated pairs

**Logic**:
- Identify correlated/cointegrated pairs
- Calculate spread: `spread = price_A - beta * price_B`
- Buy when spread < -threshold, sell when spread > +threshold

**Parameters**:
- `lookback_period`: Correlation/cointegration window (60)
- `entry_threshold`: Z-score entry (2.0)
- `exit_threshold`: Z-score exit (0.5)

**Use Case**: Market-neutral arbitrage

**Dependencies**: `pairs`, `pandas`, `statsmodels`

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

**Last Updated**: 2025-11-05
**Total Modules**: 60+ modules across 7 major components
**Lines of Code**: ~38,000 LOC
