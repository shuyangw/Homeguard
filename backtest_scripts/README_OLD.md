# Backtest Scripts Collection

Ready-to-run batch scripts for backtesting trading strategies with varying complexity levels.

## ⚡ NEW: All Scripts Now Use Sweep Mode

**All backtest scripts have been upgraded to sweep mode** for robust multi-symbol testing with parallel execution.

**Key Changes**:
- ✅ Tests across 5-30 stocks instead of single symbols
- ✅ Parallel execution (3-4x faster)
- ✅ Automated CSV + HTML reports
- ✅ Summary statistics (median Sharpe, win rates)
- ✅ No more `--visualize` or `--quantstats` flags (incompatible with sweeps)

**Migration Guide**: See [docs/SWEEP_MIGRATION_GUIDE.md](../docs/SWEEP_MIGRATION_GUIDE.md)

## Overview

This directory contains **23+ pre-configured batch scripts** organized by complexity:

```
backtest_scripts/
├── basic/           # 5 sweep backtests (FAANG universe)
├── intermediate/    # 6 sweep backtests (DOW30/TECH_GIANTS)
├── optimization/    # 5 parameter optimization scripts
├── advanced/        # 8 complex sweep scenarios (sector-specific)
├── sweeps/          # Utility scripts for sweep mode
├── RUN_ALL_STRATEGIES.bat   # Run ALL tests (22 strategies)
├── RUN_ALL_BASIC.bat        # Run all basic tests (5 strategies)
├── RUN_QUICK_TEST.bat       # Quick verification test
└── LIST_ALL_STRATEGIES.bat  # View available strategies
```

## Quick Start

**Important**: All scripts run from the repository root directory.

### 1. Verify System Works

From the repository root, double-click `backtest_scripts\RUN_QUICK_TEST.bat` to run a single backtest and verify everything is configured correctly.

**Expected Time**: 1-2 minutes

### 2. Run Your First Backtest

Navigate to `backtest_scripts\basic\` and double-click any `.bat` file:

- `01_simple_ma_crossover.bat` - Classic MA crossover strategy
- `02_rsi_mean_reversion.bat` - RSI oversold/overbought
- `03_bollinger_bands.bat` - Bollinger Bands bounce
- `04_macd_momentum.bat` - MACD momentum trading
- `05_breakout_strategy.bat` - Price breakout system

### 3. Run All Basic Tests

Double-click `RUN_ALL_BASIC.bat` to execute all 5 basic backtests sequentially and compare results.

**Expected Time**: 5-10 minutes

## Script Categories

### Basic Scripts (basic/)

Simple sweep backtests across **FAANG** (5 stocks) with default parameters. Perfect for:
- Learning how sweep backtesting works
- Comparing different strategy types across multiple stocks
- Quick performance checks with parallel execution

**Scripts** (all test FAANG: META, AAPL, AMZN, NFLX, GOOGL):
1. `01_simple_ma_crossover.bat` - MA 20/50 sweep
2. `02_rsi_mean_reversion.bat` - RSI 30/70 sweep
3. `03_bollinger_bands.bat` - BB 20/2.0 sweep
4. `04_macd_momentum.bat` - MACD 12/26/9 sweep
5. `05_breakout_strategy.bat` - 20-period breakout sweep

**Estimated Runtime**: 2-3 seconds per stock (parallel), ~10-15 seconds total

**Output**: CSV + HTML reports in logs/ directory

### Intermediate Scripts (intermediate/)

Custom parameters, larger universes, and varied settings. Learn:
- How to customize strategy parameters in sweeps
- Multi-symbol sweep testing (DOW30, TECH_GIANTS)
- Impact of fees and capital size across many stocks
- Longer time periods with parallel execution

**Scripts**:
1. `01_custom_ma_parameters.bat` - Fast 10/30 EMA sweep (DOW30)
2. `02_multi_symbol_portfolio.bat` - TECH_GIANTS sweep (10 stocks)
3. `03_tight_rsi_levels.bat` - Aggressive RSI 25/75 sweep (DOW30)
4. `04_triple_ma_trend.bat` - 10/20/50 triple MA sweep (DOW30)
5. `05_higher_capital_lower_fees.bat` - $500k institutional sweep (DOW30)
6. `06_long_period_test.bat` - 2-year sweep with top 10 results (DOW30)

**Estimated Runtime**: 8-12 seconds per script (30 stocks parallel)

### Optimization Scripts (optimization/)

Parameter grid searches to find optimal strategy settings. Learn:
- How to optimize strategy parameters
- Different optimization metrics
- Balancing returns vs risk

**Scripts**:
1. `01_optimize_ma_crossover.bat` - Find best MA windows (15 combinations)
2. `02_optimize_rsi_levels.bat` - Find best RSI thresholds (16 combinations)
3. `03_optimize_for_returns.bat` - Maximize total return
4. `04_optimize_minimize_drawdown.bat` - Minimize risk
5. `05_optimize_breakout_windows.bat` - Find best breakout periods

**Estimated Runtime**: 5-15 minutes each (tests multiple parameter combinations)

### Advanced Scripts (advanced/)

Complex sweep scenarios for experienced users. Includes:
- Sector-specific sweeps (SEMICONDUCTORS, TECH_GIANTS)
- Advanced strategies with multiple filters
- Multi-sector comparison sweeps
- Volatility-targeted strategies

**Scripts**:
1. `01_enhanced_breakout.bat` - Breakout + volatility filter sweep (SEMICONDUCTORS)
2. `01_large_portfolio_10_symbols.bat` - TECH_GIANTS sweep (10 stocks)
3. `02_volatility_targeted_momentum.bat` - Vol-scaled momentum sweep (TECH_GIANTS)
4. `04_sector_rotation_portfolio.bat` - Multi-sector sweep (4 sectors × 10 stocks each)
5. Additional scripts use sector-specific universes

**Estimated Runtime**: 8-30 seconds per sweep (depends on universe size)

## Master Scripts

### RUN_ALL_STRATEGIES.bat

**Purpose**: Run ALL backtests (Basic + Intermediate + Advanced + Optimization) sequentially with unified parameters.

**Total Tests**: 22 strategies
**Estimated Runtime**: 15-30 minutes

**Usage**:
```batch
RUN_ALL_STRATEGIES.bat [OPTIONS]
```

**Options**:
- `--fees VALUE` - Transaction fees (e.g., 0.002 = 0.2%)
- `--capital VALUE` - Initial capital (e.g., 50000)
- `--start DATE` - Start date (YYYY-MM-DD)
- `--end DATE` - End date (YYYY-MM-DD)
- `--quantstats` - Enable QuantStats tearsheet reports
- `--visualize` - Enable TradingView charts (deprecated)
- `--verbosity LEVEL` - Logging level (0-3)
- `--basic-only` - Run only basic strategies (5 tests)
- `--inter-only` - Run only intermediate strategies (6 tests)
- `--adv-only` - Run only advanced strategies (6 tests)
- `--opt-only` - Run only optimization strategies (5 tests)

**Examples**:
```batch
REM Run ALL strategies with QuantStats reports
RUN_ALL_STRATEGIES.bat --quantstats

REM Run ALL with custom fees and capital
RUN_ALL_STRATEGIES.bat --fees 0.002 --capital 50000 --quantstats

REM Run ONLY basic strategies with detailed logging
RUN_ALL_STRATEGIES.bat --basic-only --verbosity 2 --quantstats

REM Run ALL for a specific date range
RUN_ALL_STRATEGIES.bat --start 2023-01-01 --end 2024-12-31 --quantstats
```

**Parameter Cascading**: All parameters passed to the master script cascade down to individual strategy scripts, allowing you to run comprehensive backtests with consistent settings across all strategies.

**QuantStats Reports**: When using `--quantstats`, each backtest generates:
- **tearsheet.html** - Interactive HTML performance report
- **quantstats_metrics.txt** - Text file with 34+ metrics
- **equity_curve.csv** - Portfolio value over time
- **daily_returns.csv** - Daily returns data

Reports are saved to the directory configured in `settings.ini` under `log_output_dir`.

### RUN_ALL_BASIC.bat

**Purpose**: Run all 5 basic strategies sequentially with unified parameters.

**Total Tests**: 5 strategies
**Estimated Runtime**: 5-10 minutes

**Usage**: Same as RUN_ALL_STRATEGIES.bat but limited to basic strategies only.

### RUN_QUICK_TEST.bat

**Purpose**: Quick verification test to ensure the system is working correctly.

**Tests**: 1 strategy (MA Crossover on AAPL, 6 months)
**Estimated Runtime**: 1-2 minutes

Use this to verify your setup before running longer backtests.

## Script Output

Each sweep script displays:

```
========================================
RUNNING SWEEP: MovingAverageCrossover
========================================
Symbols: 5 (META, AAPL, ... GOOGL)
Period: 2023-01-01 to 2024-01-01
Mode: Parallel
========================================

[1/5] Completed AAPL
[2/5] Completed META
[3/5] Completed AMZN
[4/5] Completed NFLX
[5/5] Completed GOOGL

Sweep complete: 5 successful, 0 failed
========================================

All results (sorted by Sharpe Ratio):

Symbol    Total Return [%]  Sharpe Ratio  Max Drawdown [%]  Win Rate [%]  Total Trades
AAPL              15.2          1.34            -6.7          55.0          10
META              22.7          1.89            -8.4          60.0          12
GOOGL             18.3          1.56           -12.1          52.5          15
AMZN               8.5          0.92            -9.2          48.0           8
NFLX              -2.3         -0.15           -15.3          42.0           7

========================================
SUMMARY STATISTICS
========================================
Total Symbols: 5
Profitable: 4
Unprofitable: 1
Win Rate (Symbols): 80.0%

Median Sharpe Ratio: 1.34
Median Total Return: 15.2%
Mean Max Drawdown: -10.3%
========================================

Sweep complete! Check logs for CSV and HTML reports.
```

**Reports Generated**:
- `logs/<timestamp>_<Strategy>_<Universe>/sweep_results.csv`
- `logs/<timestamp>_<Strategy>_<Universe>/sweep_results.html`

## Customizing Scripts

Each script is a text file. Open with Notepad to modify:

### Change Universe
```batch
--universe FAANG
```
Change to: `--universe DOW30` or `--universe TECH_GIANTS`

**Available universes**: Run `LIST_ALL_STRATEGIES.bat` or see `sweeps/list_universes.bat`

### Use Custom Symbol List
```batch
--universe FAANG
```
Change to:
```batch
--symbols-file my_watchlist.txt
```
(Create `my_watchlist.txt` with one symbol per line)

### Change Date Range
```batch
--start 2023-01-01 ^
--end 2024-01-01
```

### Change Capital (Per Symbol)
```batch
--capital 100000
```
Change to: `--capital 50000` or `--capital 500000`

### Change Fees
```batch
--fees 0.001
```
Change to: `--fees 0.0005` (lower) or `--fees 0.002` (higher)

### Change Strategy Parameters
```batch
--params "fast_window=20,slow_window=50"
```

### Control Parallel Execution

**All sweep scripts run in parallel by default** (4 workers). To customize:

**Disable parallel execution** (run sequentially):
```batch
RUN_ALL_BASIC.bat --no-parallel
```

**Change number of workers**:
```batch
RUN_ALL_BASIC.bat --max-workers 8
```

**Individual scripts** respect environment variables:
```batch
set BACKTEST_PARALLEL=
set BACKTEST_MAX_WORKERS=8
01_simple_ma_crossover.bat
```

**Linux/Mac**:
```bash
export BACKTEST_PARALLEL=""  # Disable parallel
./run_all_basic.sh --max-workers 8
```

**When to use sequential mode**:
- Debugging strategy logic
- Memory-constrained systems
- Profiling performance

**When to increase workers**:
- High-core-count CPUs (8+ cores)
- Testing 20+ symbols
- I/O-bound systems (fast SSDs)

### Sort Results
```batch
--sort-by "Sharpe Ratio"
```
Change to: `--sort-by "Total Return [%]"` or `--sort-by "Win Rate [%]"`

### Show Top N Results
```batch
--top-n 10
```
Show only top 10 performers (useful for large universes)

## Creating Your Own Script

1. Copy an existing script
2. Rename it
3. Modify the parameters
4. Save and double-click to run

**Sweep Mode Template**:
```batch
@echo off
REM Your Description Here

echo ========================================
echo Your Title (SWEEP)
echo Universe: FAANG (5 stocks)
echo Mode: Parallel
echo ========================================
echo.

python src\backtest_runner.py ^
  --strategy MovingAverageCrossover ^
  --universe FAANG ^
  --sweep ^
  --parallel ^
  --start 2023-01-01 ^
  --end 2024-01-01 ^
  --capital 100000 ^
  --fees 0.001 ^
  --sort-by "Sharpe Ratio"

echo.
echo Sweep complete! Check logs for CSV and HTML reports.
pause
```

**Note**: All backtest scripts run from the repository root directory using `python src\backtest_runner.py`.

## Troubleshooting

### "python: command not found"
- Python is not installed or not in PATH
- Install Python 3.8+ and ensure it's added to PATH

### "No data found for symbols"
- Data hasn't been ingested for those symbols/dates
- Run data ingestion first: `python src\run_ingestion.py` (from repo root)

### "ModuleNotFoundError: No module named 'vectorbt'"
- Dependencies not installed
- Run: `pip install -r requirements.txt`

### Script closes immediately
- There's an error - the script is closing too fast to see it
- Right-click script → Edit → Check configuration
- Or run from Command Prompt to see errors

### Slow performance
- Large date ranges take longer
- Multi-symbol portfolios require more processing
- Optimization tests many parameter combinations
- This is normal - be patient!

## Best Practices

### For Learning
1. Start with `RUN_QUICK_TEST.bat`
2. Run individual basic scripts
3. Try `RUN_ALL_BASIC.bat` to compare strategies
4. Move to intermediate scripts

### For Research
1. Use optimization scripts to find best parameters
2. Test on training period (e.g., 2022-2023)
3. Validate on test period (e.g., 2023-2024)
4. Use advanced scripts for robustness testing

### For Development
1. Create custom scripts for your strategies
2. Test with different symbols and periods
3. Compare against built-in strategies
4. Document your findings

## Data Requirements

All scripts require that you have:
1. Run data ingestion for the symbols used
2. Data available for the specified date ranges
3. Parquet files in the configured storage directory

**Check Available Data** (from repo root):
```bash
python -c "import sys; sys.path.insert(0, 'src'); from backtesting.engine.data_loader import DataLoader; print(DataLoader().get_available_symbols())"
```

## Performance Expectations (Sweep Mode)

**FAANG Sweep (5 stocks), 1 Year**: 10-15 seconds (parallel)
**DOW30 Sweep (30 stocks), 1 Year**: 8-12 seconds (parallel)
**TECH_GIANTS (10 stocks), 1 Year**: 6-10 seconds (parallel)
**Sector Sweep (10 stocks), 1 Year**: 6-10 seconds (parallel)
**Optimization (10-20 combinations)**: 5-15 minutes (unchanged)

**Speedup**: ~3.75x faster than sequential with 4 parallel workers

*Times vary based on CPU speed, data size, and number of parallel workers*

## Sweep Utilities (sweeps/)

Specialized sweep scripts for advanced workflows:

1. **list_universes.bat/.sh** - Show all available universes
2. **sweep_universe.bat/.sh** - Generic universe sweep
3. **quick_sweep.bat/.sh** - Fast FAANG testing
4. **compare_strategies.bat/.sh** - Compare multiple strategies
5. **sweep_all_sectors.bat/.sh** - Test all sectors
6. **01_sweep_faang.bat/.sh** - Example FAANG sweep
7. **02_sweep_dow30_parallel.bat/.sh** - Example DOW30 parallel
8. **03_sweep_custom_symbols.bat/.sh** - Example custom symbols

See [sweeps/README.md](sweeps/README.md) for details.

## Available Strategies

Run `LIST_ALL_STRATEGIES.bat` to see all 10 available strategies and their parameters:

1. **MovingAverageCrossover** - Fast/slow MA crossover
2. **TripleMovingAverage** - Triple MA trend alignment
3. **MeanReversion** - Bollinger Bands bounce
4. **RSIMeanReversion** - RSI oversold/overbought
5. **OvernightMeanReversion** - VWAP-based overnight trades
6. **MomentumStrategy** - MACD momentum
7. **BreakoutStrategy** - Price breakout (enhanced with filters)
8. **VolatilityTargetedMomentum** - Vol-scaled momentum
9. **CrossSectionalMomentum** - Top performers ranking
10. **PairsTrading** - Cointegration-based pairs

## Next Steps

After running backtests:

1. **Compare Results**: Run multiple scripts and compare metrics
2. **Optimize Parameters**: Use optimization scripts to improve performance
3. **Validate**: Test optimal parameters on different time periods
4. **Create Custom**: Build your own strategies and scripts
5. **Document Findings**: Keep track of what works and what doesn't

## Additional Resources

- **Sweep Migration Guide**: `../docs/SWEEP_MIGRATION_GUIDE.md` ⭐ NEW
- **Sweep API Reference**: `../docs/API_REFERENCE_SWEEP.md`
- **Full Documentation**: `../docs/BACKTESTING_GUIDE.md`
- **API Reference**: `../docs/API_REFERENCE.md`
- **Example Scripts**: `../examples/`
- **Custom Strategy Template**: `../src/strategies/custom/template.py`

---

**Total Scripts**: 15+ sweep batch files + 8 sweep utilities + 5 optimization scripts
**Total Strategies**: 10 built-in strategies
**Universes**: 11 predefined universes (FAANG, DOW30, sectors, etc.)
**Complexity Levels**: 4 (Basic, Intermediate, Optimization, Advanced)
**Execution Mode**: Sweep with parallel execution (3.75x faster)
**Ready to Run**: Yes - just double-click!

**New in v2.0**: All scripts migrated to sweep mode for multi-symbol testing with parallel execution and automated reporting.
