# Backtest Runner GUI - User Guide

## Overview

The Backtest Runner GUI is a modern, cross-platform desktop application for running and managing backtesting operations with an intuitive visual interface.

**Features:**
- Visual strategy selector with parameter configuration
- Real-time progress monitoring for multiple symbols
- Interactive results table with color-coded metrics
- Support for up to 16 parallel workers
- Export results to CSV and HTML formats
- Cross-platform (Windows, macOS, Linux)

## Installation

### Prerequisites

- Python 3.8+ with conda
- Flet 0.28.3+ (already installed in `fintech` environment)

### Quick Start

1. Activate the fintech conda environment:
```bash
conda activate fintech
```

2. Launch the GUI:
```bash
python run_gui.py
```

The application window will open automatically.

## User Interface

The GUI has three main views:

### 1. Setup View (Configuration)

**Strategy Selection:**
- Choose from 10 built-in strategies
- View strategy description
- Configure strategy parameters dynamically

**Available Strategies:**
- Moving Average Crossover
- Triple Moving Average
- Mean Reversion
- RSI Mean Reversion
- Momentum Strategy
- Breakout Strategy
- Volatility Targeted Momentum
- Overnight Mean Reversion
- Cross-Sectional Momentum
- Pairs Trading

**Symbol Input:**
- Enter comma-separated symbols (e.g., `AAPL, MSFT, GOOGL`)
- Supports any valid stock ticker
- Minimum: 1 symbol
- Recommended maximum: 100 symbols for optimal performance

**Date Range:**
- Start Date: Beginning of backtest period (YYYY-MM-DD)
- End Date: End of backtest period (YYYY-MM-DD)
- Default: Last 60 days

**Execution Settings:**
- **Parallel Mode**: Enable/disable parallel execution
- **Worker Count**: Slider to select 1-16 parallel workers
  - Recommended: 4-8 workers (displayed based on CPU count)
  - More workers = faster execution for multiple symbols
  - Diminishing returns above 8 workers

**Run Button:**
- Validates all inputs
- Starts backtest execution
- Switches to Execution View

### 2. Execution View (Real-Time Monitoring)

**Summary Bar:**
- Overall progress percentage
- Completed / Total symbols
- Currently running symbols
- Failed symbols count

**Symbol Cards:**
Each symbol displays:
- Symbol name
- Status indicator (Pending, Running, Completed, Failed)
- Progress bar (0-100%)
- Current operation message (e.g., "Loading data...", "Computing metrics...")

**Status Icons:**
- ⏳ Pending (grey) - Waiting to start
- ⏱ Running (blue) - Currently executing
- ✓ Completed (green) - Successfully finished
- ✗ Failed (red) - Error occurred

**Actions:**
- **Cancel**: Stop execution (cooperative cancellation)
  - Current symbols will complete
  - New symbols won't start
- **View Results**: Available when all symbols complete

### 3. Results View (Analysis)

**Summary Cards:**
Four key metrics displayed prominently:
1. **Total Symbols**: Number of symbols tested
2. **Average Return**: Mean return across all symbols
3. **Average Sharpe**: Mean Sharpe ratio
4. **Win Rate**: Percentage of profitable symbols

**Results Table:**
Displays detailed metrics for each symbol:
- Symbol
- Total Return (%)
- Annual Return (%)
- Sharpe Ratio
- Max Drawdown (%)
- Win Rate (%)
- Total Trades

**Color Coding:**
- 🟢 Green: Positive returns, good Sharpe (≥1.0)
- 🟠 Orange: Neutral metrics
- 🔴 Red: Negative returns, poor Sharpe

**Export Options:**
- **Export CSV**: Save results as CSV file
- **Export HTML**: Generate HTML report with formatted table
- Files saved to configured log directory (see `settings.ini`)

**Navigation:**
- **Back to Setup**: Return to configuration view

## Workflow Example

1. **Configure Backtest:**
   - Select "Moving Average Crossover"
   - Set fast_window=10, slow_window=50
   - Enter symbols: `AAPL, MSFT, GOOGL, AMZN`
   - Set dates: 2024-01-01 to 2024-12-31
   - Choose 4 workers, parallel mode enabled
   - Click "Run Backtests"

2. **Monitor Execution:**
   - Watch real-time progress for each symbol
   - See status change: Pending → Running → Completed
   - Overall progress updates automatically
   - Wait for all symbols to complete (~30-60 seconds for 4 symbols)

3. **Analyze Results:**
   - Review summary cards (avg return, Sharpe, win rate)
   - Examine detailed table (sort by any column)
   - Export results to CSV for further analysis
   - Click "Back to Setup" to run another backtest

## Performance Tips

**Optimal Settings:**
- **1-10 symbols**: 2-4 workers
- **10-50 symbols**: 4-8 workers
- **50+ symbols**: 8-16 workers

**Speed Factors:**
- More symbols = longer execution time
- Longer date ranges = more data to process
- Parallel mode is always faster for 2+ symbols
- SSD storage speeds up data loading

**Memory Usage:**
- Each symbol requires ~100-500 MB RAM (depends on date range)
- Monitor system memory with 50+ symbols
- Close other applications if running 100+ symbols

## Troubleshooting

### GUI Won't Start

**Error:** `ModuleNotFoundError: No module named 'flet'`

**Solution:**
```bash
conda activate fintech
pip install flet
```

### Invalid Date Format Error

**Error:** "Invalid date format. Use YYYY-MM-DD"

**Solution:**
- Ensure dates are in format: `2024-01-01`
- Start date must be before end date
- Don't use slashes: ~~`01/01/2024`~~ ❌

### No Results to Display

**Possible Causes:**
1. All backtests failed (check error messages)
2. No data available for selected symbols/dates
3. Strategy parameters invalid

**Solution:**
- Try different symbols (e.g., large-cap stocks like AAPL, MSFT)
- Check date range (ensure market was open during this period)
- Verify strategy parameters are valid

### Backtest Cancelled

**Behavior:**
- Shows message: "Backtests will stop after current symbols complete"
- Already-running symbols finish normally
- Remaining symbols show status "Pending"

**To Resume:**
- Go back to Setup view
- Click "Run Backtests" again

## Keyboard Shortcuts

*(Future enhancement - not yet implemented)*

## Advanced Features

### Custom Strategies

To add your own strategy to the GUI:

1. Create strategy class in `src/strategies/custom/`
2. Add import to `src/strategies/__init__.py`
3. Add to `get_strategy_registry()` in `src/gui/utils/strategy_utils.py`
4. Restart GUI

The GUI will automatically detect parameters from `__init__()` method and create appropriate input controls.

### Exporting Portfolio Objects

For advanced users who want to generate custom charts:

```python
from gui.workers.gui_controller import GUIBacktestController

# After backtests complete
portfolios = controller.get_portfolios()

# Access individual portfolio objects
aapl_portfolio = portfolios['AAPL']

# Generate equity curve
equity = aapl_portfolio.value()

# Access trades
trades = aapl_portfolio.trades.records_readable
```

## Comparison: GUI vs CLI

| Feature | GUI | CLI (backtest_runner.py) |
|---------|-----|--------------------------|
| **Ease of Use** | ⭐⭐⭐⭐⭐ Visual | ⭐⭐⭐ Command-line |
| **Real-time Progress** | ✅ Live updates | ❌ Console logs only |
| **Multi-Symbol Monitoring** | ✅ Individual cards | ❌ Sequential logs |
| **Parameter Configuration** | ✅ Dynamic UI | ⚠️ Manual code editing |
| **Results Visualization** | ✅ Color-coded table | ⚠️ Text output |
| **Export** | ✅ CSV, HTML | ✅ CSV, HTML, QuantStats |
| **Automation** | ❌ Manual only | ✅ Scriptable |
| **Advanced Options** | ⚠️ Basic | ✅ Full control |

**Recommendation:**
- **GUI**: Interactive testing, exploring strategies, quick analysis
- **CLI**: Automation, advanced customization, production pipelines

## FAQ

**Q: Can I run multiple backtests simultaneously?**
A: No, only one backtest can run at a time in the GUI. Wait for completion or cancel the current run.

**Q: Where are the exported files saved?**
A: Files are saved to the directory specified in `settings.ini` under `log_output_dir` for your OS (Windows, macOS, or Linux).

**Q: Can I customize the theme (dark mode)?**
A: Not yet. Dark mode support is planned for Phase 5. Current version uses light theme only.

**Q: How do I view QuantStats tearsheets from GUI results?**
A: The GUI doesn't generate QuantStats reports yet. Use the CLI `backtest_runner.py` with `--quantstats` flag for full tearsheets.

**Q: What happens if my internet disconnects during backtesting?**
A: Backtesting uses local data from the database. Internet connection is not required after data is downloaded.

**Q: Can I pause and resume a backtest?**
A: No, pause/resume is not supported. You can cancel and restart from the beginning.

## Next Steps

After using the GUI, explore:

1. **CLI Backtesting** (`docs/BACKTESTING_GUIDE.md`) - More advanced features
2. **Sweep Runner** (`docs/API_REFERENCE_SWEEP.md`) - Batch testing across symbols
3. **Strategy Development** (`docs/ADVANCED_STRATEGIES_GUIDE.md`) - Build custom strategies
4. **QuantStats Reports** (`docs/quantstats/`) - Generate professional tearsheets

## Support

For issues or questions:
- Check troubleshooting section above
- Review `docs/BACKTESTING_GUIDE.md` for general concepts
- Open an issue on GitHub (if applicable)

---

**Version:** Phase 2 Complete
**Last Updated:** 2025-01-01
**Status:** Production Ready ✅
