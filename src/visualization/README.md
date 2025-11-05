# Visualization Engine

**Comprehensive data visualization and logging for backtesting results**

## Overview

The Visualization Engine provides rich, interactive visualizations and detailed logging for backtesting results. It creates organized output directories with timestamped folders, generates candlestick charts with trade overlays, and produces comprehensive reports with configurable verbosity levels.

## Features

- **Interactive Candlestick Charts**: Plotly-based interactive HTML charts with:
  - OHLCV candlestick visualization
  - Volume subplot with color-coded bars
  - Trade count subplot
  - Buy/sell trade markers overlaid on price action
  - Hover tooltips with detailed information

- **Configurable Logging**: Trade event logging with 4 verbosity levels:
  - `MINIMAL (0)`: Only summary statistics
  - `NORMAL (1)`: Trade entries and exits
  - `DETAILED (2)`: Trade details + portfolio changes
  - `VERBOSE (3)`: Everything including intermediate calculations

- **Organized Output Structure**: Human-readable timestamped folders:
  ```
  {log_dir}/
  ├── 20250129_143022_MovingAverageCrossover_AAPL/
  │   ├── charts/
  │   │   ├── AAPL_price_chart.html
  │   │   └── AAPL_trades.html
  │   ├── logs/
  │   │   ├── trade_log.txt
  │   │   └── trades.csv
  │   └── reports/
  │       ├── summary_report.txt
  │       ├── trade_summary.txt
  │       └── backtest_results.json
  ```

- **Multiple Output Formats**:
  - Interactive HTML charts (Plotly)
  - Static PNG images (mplfinance)
  - Text reports
  - CSV trade logs
  - JSON data for programmatic access

## Architecture

```
src/visualization/
├── __init__.py                      # Package exports
├── config.py                        # Configuration and settings
├── logger.py                        # Trade logging with verbosity control
├── charts/
│   ├── __init__.py
│   ├── candlestick.py              # Candlestick chart generation
│   └── overlays.py                 # Trade signal overlays
├── reports/
│   ├── __init__.py
│   └── report_generator.py         # Report generation
└── utils/
    ├── __init__.py
    └── output_manager.py            # Directory management
```

## Quick Start

### Basic Usage

```python
from visualization import VisualizationConfig, TradeLogger, CandlestickChart, OutputManager
from visualization.config import LogLevel

# Create configuration
config = VisualizationConfig(
    log_level=LogLevel.DETAILED,
    save_charts=True,
    save_logs=True,
    chart_format='html'
)

# Initialize components
output_mgr = OutputManager(config.log_dir)
run_dir = output_mgr.create_run_directory(
    strategy_name="MovingAverageCrossover",
    symbols=["AAPL"]
)

logger = TradeLogger(log_level=config.log_level)

# Log trades during backtest
from visualization.logger import TradeEvent

event = TradeEvent(
    timestamp=pd.Timestamp('2024-01-15 10:30:00'),
    symbol='AAPL',
    action='BUY',
    price=150.25,
    size=100,
    portfolio_value=50000,
    cash=35000,
    position_value=15000
)
logger.log_trade(event)

# Create chart
chart = CandlestickChart.create_plotly_chart(
    data=price_data,
    symbol='AAPL',
    show_volume=True,
    show_trade_count=True
)

# Add trade markers
trades_df = logger.get_trades_dataframe()
chart = CandlestickChart.add_trade_markers(chart, trades_df)

# Save outputs
CandlestickChart.save_html(chart, output_mgr.get_chart_path('AAPL_chart.html'))
logger.save_log(output_mgr.get_log_path('trade_log.txt'))
logger.save_trades_csv(output_mgr.get_log_path('trades.csv'))
```

### Command-Line Usage

```bash
# With visualization enabled
python src/backtest_runner.py \
  --strategy MovingAverageCrossover \
  --symbols AAPL \
  --start 2023-01-01 \
  --end 2024-01-01 \
  --visualize \
  --verbosity 2

# Disable visualization
python src/backtest_runner.py \
  --strategy MovingAverageCrossover \
  --symbols AAPL \
  --start 2023-01-01 \
  --end 2024-01-01
```

### Batch Script Usage

```batch
REM Enable visualization with default verbosity (1)
RUN_QUICK_TEST.bat --visualize

REM Enable visualization with detailed logging
RUN_QUICK_TEST.bat --visualize --verbosity 2

REM Run all basic tests with visualization
RUN_ALL_BASIC.bat --visualize --verbosity 1
```

## Configuration

### Settings (settings.ini)

```ini
[windows]
log_output_dir = C:\Users\username\Dropbox\cs\stonk\logs

[macos]
log_output_dir = /Users/username/Dropbox/cs/stonk/logs

[linux]
log_output_dir = /home/username/stonk/logs
```

### Verbosity Levels

| Level | Value | Description | Output |
|-------|-------|-------------|--------|
| MINIMAL | 0 | Summary only | Final statistics |
| NORMAL | 1 | Trade events | Entry/exit notifications |
| DETAILED | 2 | + Portfolio changes | Cash, positions, portfolio value |
| VERBOSE | 3 | + Calculations | Intermediate computations |

### Chart Formats

- **HTML** (default): Interactive Plotly charts with zoom, pan, hover
- **PNG**: Static images using mplfinance

## Components

### VisualizationConfig

Configuration object for visualization settings:
- `log_dir`: Base directory for output
- `log_level`: Verbosity level (LogLevel enum)
- `save_charts`: Enable/disable chart generation
- `save_logs`: Enable/disable log file generation
- `chart_format`: 'html' or 'png'
- `chart_width`, `chart_height`: Chart dimensions

### OutputManager

Manages directory structure:
- `create_run_directory()`: Creates timestamped folder
- `get_chart_path()`, `get_log_path()`, `get_report_path()`: Path helpers

### TradeLogger

Logs trading activity:
- `log_trade()`: Log a trade event
- `log_message()`: Log general messages
- `get_trades_dataframe()`: Get trades as DataFrame
- `save_log()`: Save text log
- `save_trades_csv()`: Export trades to CSV
- `get_trade_summary()`: Get summary statistics

### CandlestickChart

Chart generation:
- `create_plotly_chart()`: Interactive candlestick chart
- `add_trade_markers()`: Overlay buy/sell markers
- `save_html()`, `save_png()`: Export charts
- `create_mplfinance_chart()`: Static PNG charts

### ReportGenerator

Report creation:
- `generate_summary_report()`: Text summary report
- `generate_json_report()`: JSON export
- `generate_trade_log_summary()`: Trade statistics

## Dependencies

```
plotly==5.24.1          # Interactive charts
kaleido==0.2.1          # PNG export for plotly
mplfinance==0.12.10b0   # Static candlestick charts
pandas>=2.3.1           # Data handling
```

## Integration with Backtesting Engine

The visualization engine integrates seamlessly with the backtesting engine. When enabled, it:

1. Creates an output directory for the run
2. Logs all trade events as they occur
3. Generates charts with trade overlays
4. Produces comprehensive reports
5. Saves all outputs to organized folders

## Output Examples

### Trade Log (NORMAL verbosity)
```
[2024-01-15 09:30:00] BUY AAPL: 100.00 shares @ $150.25
[2024-01-20 14:45:00] SELL AAPL: 100.00 shares @ $155.80
```

### Trade Log (DETAILED verbosity)
```
[2024-01-15 09:30:00] BUY AAPL: 100.00 shares @ $150.25
  Portfolio Value: $50,000.00 | Cash: $34,975.00 | Position Value: $15,025.00
[2024-01-20 14:45:00] SELL AAPL: 100.00 shares @ $155.80
  Portfolio Value: $50,555.00 | Cash: $50,555.00 | Position Value: $0.00
```

### Summary Report
```
================================================================================
BACKTEST SUMMARY REPORT
================================================================================

STRATEGY CONFIGURATION
--------------------------------------------------------------------------------
Strategy:        MovingAverageCrossover
Symbols:         AAPL
Period:          2023-01-01 to 2024-01-01
Initial Capital: $100,000.00
Transaction Fees: 0.0010 (0.10%)

PERFORMANCE METRICS
--------------------------------------------------------------------------------
Total Return [%]                   :              15.50
Annual Return [%]                  :              15.50
Sharpe Ratio                       :               1.25
Max Drawdown [%]                   :              -8.30

TRADE SUMMARY
--------------------------------------------------------------------------------
Total Trades:              25
Buy Orders:                13
Sell Orders:               12
Total Volume:          10,500.00 shares
Avg Trade Size:           420.00 shares
```

## Best Practices

1. **Use appropriate verbosity**: DETAILED for development, NORMAL for production
2. **Check disk space**: Charts and logs can accumulate quickly
3. **Archive old runs**: Set up periodic cleanup of old output directories
4. **Use HTML for analysis**: Interactive charts are better for exploring results
5. **Use PNG for reports**: Static images are better for documentation

## Future Enhancements

- [ ] Equity curve visualization
- [ ] Drawdown charts
- [ ] Multiple strategy comparison
- [ ] Performance heatmaps
- [ ] Live streaming visualization
- [ ] Custom indicator overlays
- [ ] Automated report emailing
