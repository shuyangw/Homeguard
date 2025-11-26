# Sweep Scripts - Multi-Symbol Backtesting

This directory contains utility scripts for running backtests across multiple symbols (sweeps).

## Quick Start

### Windows
```batch
REM Quick test with FAANG stocks
quick_sweep.bat MovingAverageCrossover

REM Sweep DOW30
sweep_universe.bat DOW30 BreakoutStrategy

REM Compare strategies on FAANG
compare_strategies.bat FAANG
```

### Linux/Mac
```bash
# Quick test with FAANG stocks
./quick_sweep.sh MovingAverageCrossover

# Sweep DOW30
./sweep_universe.sh DOW30 BreakoutStrategy

# Compare strategies on FAANG
./compare_strategies.sh FAANG
```

---

## Available Scripts

### 1. `list_universes` - Show Available Universes

Lists all predefined universes and usage examples.

**Usage:**
```batch
# Windows
list_universes.bat

# Linux/Mac
./list_universes.sh
```

**Output:**
- DOW30, NASDAQ100, SP100
- FAANG, MAGNIFICENT7, TECH_GIANTS
- Sector universes: SEMICONDUCTORS, ENERGY, FINANCE, HEALTHCARE, CONSUMER

---

### 2. `sweep_universe` - Sweep Any Universe

General-purpose sweep script that accepts universe and strategy as parameters.

**Usage:**
```batch
# Windows
sweep_universe.bat UNIVERSE STRATEGY [START_DATE] [END_DATE]

# Linux/Mac
./sweep_universe.sh UNIVERSE STRATEGY [START_DATE] [END_DATE]
```

**Examples:**
```batch
# Sweep FAANG with default dates (2023-2024)
sweep_universe.bat FAANG MovingAverageCrossover

# Sweep DOW30 with custom dates
sweep_universe.bat DOW30 BreakoutStrategy 2022-01-01 2024-01-01

# Sweep semiconductors
sweep_universe.bat SEMICONDUCTORS MeanReversion
```

**Features:**
- Parallel execution by default
- Auto-generated output directory
- CSV and HTML reports
- Sorted by Sharpe Ratio

---

### 3. `quick_sweep` - Fast Testing

Quick sweep using FAANG (5 stocks) for fast validation.

**Usage:**
```batch
# Windows
quick_sweep.bat STRATEGY [PARAMS]

# Linux/Mac
./quick_sweep.sh STRATEGY [PARAMS]
```

**Examples:**
```batch
# Test with default parameters
quick_sweep.bat MovingAverageCrossover

# Test with custom parameters
quick_sweep.bat MeanReversion "window=30,num_std=2.5"

# Test breakout strategy
quick_sweep.bat BreakoutStrategy "breakout_window=15,exit_window=8"
```

**Use Cases:**
- Quick validation of strategy logic
- Parameter testing before full sweep
- Debugging strategy implementation

---

### 4. `compare_strategies` - Strategy Comparison

Runs multiple strategies on the same universe for comparison.

**Usage:**
```batch
# Windows
compare_strategies.bat UNIVERSE [START_DATE] [END_DATE]

# Linux/Mac
./compare_strategies.sh UNIVERSE [START_DATE] [END_DATE]
```

**Examples:**
```batch
# Compare strategies on FAANG
compare_strategies.bat FAANG

# Compare strategies on DOW30 with custom dates
compare_strategies.bat DOW30 2023-01-01 2024-01-01
```

**Strategies Tested:**
1. MovingAverageCrossover (MA Cross)
2. BreakoutStrategy (Donchian Breakout)
3. MeanReversion (Bollinger Bands)
4. MomentumStrategy (MACD)
5. RSIMeanReversion (RSI)

**Output:**
- 5 separate CSV/HTML reports (one per strategy)
- All saved with prefix: `compare_{UNIVERSE}_`
- Easy to compare which strategy works best

---

### 5. `sweep_all_sectors` - Sector Analysis

Tests one strategy across all sector universes.

**Usage:**
```batch
# Windows
sweep_all_sectors.bat STRATEGY [START_DATE] [END_DATE]

# Linux/Mac
./sweep_all_sectors.sh STRATEGY [START_DATE] [END_DATE]
```

**Examples:**
```batch
# Test MA crossover on all sectors
sweep_all_sectors.bat MovingAverageCrossover

# Test breakout on all sectors with custom dates
sweep_all_sectors.bat BreakoutStrategy 2023-01-01 2024-01-01
```

**Sectors Tested:**
1. SEMICONDUCTORS (10 stocks)
2. ENERGY (10 stocks)
3. FINANCE (10 stocks)
4. HEALTHCARE (10 stocks)
5. CONSUMER (10 stocks)
6. TECH_GIANTS (10 stocks)

**Output:**
- 6 separate reports (one per sector)
- All saved with prefix: `sector_{SECTOR}_{STRATEGY}`
- Identify which sectors work best with your strategy

---

### 6. Example Sweeps (01-03)

Pre-configured example sweeps for learning and reference.

**01_sweep_faang.bat**
- Sweeps FAANG stocks
- Uses MovingAverageCrossover
- Default parameters

**02_sweep_dow30_parallel.bat**
- Sweeps DOW30 stocks
- Uses BreakoutStrategy
- Parallel execution with 4 workers
- Shows top 10 results

**03_sweep_custom_symbols.bat**
- Creates example watchlist
- Loads symbols from file
- Uses MeanReversion

---

## Environment Variables

Customize behavior with environment variables:

```batch
# Windows
set BACKTEST_START=2022-01-01
set BACKTEST_END=2024-01-01
set BACKTEST_CAPITAL=50000
set BACKTEST_FEES=0.002
set BACKTEST_VERBOSITY=2

# Linux/Mac
export BACKTEST_START=2022-01-01
export BACKTEST_END=2024-01-01
export BACKTEST_CAPITAL=50000
export BACKTEST_FEES=0.002
export BACKTEST_VERBOSITY=2
```

**Variables:**
- `BACKTEST_START`: Start date (default: 2023-01-01)
- `BACKTEST_END`: End date (default: 2024-01-01)
- `BACKTEST_CAPITAL`: Initial capital (default: 100000)
- `BACKTEST_FEES`: Trading fees as decimal (default: 0.001)
- `BACKTEST_VERBOSITY`: Logging level 0-3 (default: 1)

---

## Available Universes

### Major Indices
- **DOW30**: 30 Dow Jones Industrial Average stocks
- **NASDAQ100**: Top 50 NASDAQ-100 stocks
- **SP100**: Top 50 S&P 100 stocks

### Popular Groups
- **FAANG**: META, AAPL, AMZN, NFLX, GOOGL (5 stocks)
- **MAGNIFICENT7**: AAPL, MSFT, GOOGL, AMZN, NVDA, TSLA, META (7 stocks)
- **TECH_GIANTS**: 10 major technology companies

### Sector Universes
- **SEMICONDUCTORS**: 10 semiconductor stocks
- **ENERGY**: 10 energy stocks
- **FINANCE**: 10 financial stocks
- **HEALTHCARE**: 10 healthcare stocks
- **CONSUMER**: 10 consumer stocks

---

## Available Strategies

### Base Strategies
- **MovingAverageCrossover**: Fast MA crosses slow MA
- **TripleMovingAverage**: Three-MA trend filter
- **MeanReversion**: Bollinger Bands mean reversion
- **RSIMeanReversion**: RSI-based mean reversion
- **MomentumStrategy**: MACD momentum
- **BreakoutStrategy**: Donchian channel breakouts

### Advanced Strategies (if available)
- **VolatilityTargetedMomentum**: Volatility-scaled positions
- **OvernightMeanReversion**: Close-to-open mean reversion
- **CrossSectionalMomentum**: Rank-based momentum
- **PairsTrading**: Statistical arbitrage

---

## Output Files

All sweeps generate results in the `logs/` directory:

**Directory Structure:**
```
logs/
└── {TIMESTAMP}_{RUN_NAME}_{SYMBOLS}/
    ├── {TIMESTAMP}_{STRATEGY}_sweep_results.csv
    ├── {TIMESTAMP}_{STRATEGY}_sweep_results.html
    └── (optional tearsheet.html if --quantstats enabled)
```

**CSV Report Contains:**
- Individual results for each symbol
- Summary statistics section at bottom
- Sortable by any column

**HTML Report Contains:**
- Summary statistics grid
- Full results table
- Color-coded returns (green=profit, red=loss)
- Responsive design

---

## Custom Symbols

Create a custom watchlist file:

**Example: `my_watchlist.txt`**
```
# My custom watchlist
AAPL
MSFT
GOOGL
AMZN
TSLA
NVDA
META
```

**Use with sweep:**
```batch
# Windows
python src\backtest_runner.py --strategy MovingAverageCrossover --symbols-file my_watchlist.txt --sweep --start 2023-01-01 --end 2024-01-01

# Linux/Mac
python src/backtest_runner.py --strategy MovingAverageCrossover --symbols-file my_watchlist.txt --sweep --start 2023-01-01 --end 2024-01-01
```

---

## Advanced Usage

### Custom Parameters

Pass custom parameters to strategies:

```batch
# Windows
sweep_universe.bat FAANG MovingAverageCrossover

# With custom parameters (use backtest_runner.py directly)
python src\backtest_runner.py ^
  --strategy MovingAverageCrossover ^
  --universe FAANG ^
  --sweep ^
  --params "fast_window=10,slow_window=30,ma_type=ema" ^
  --start 2023-01-01 ^
  --end 2024-01-01
```

### Filter Results

Show only top performers:

```batch
python src\backtest_runner.py ^
  --strategy BreakoutStrategy ^
  --universe DOW30 ^
  --sweep ^
  --sort-by "Total Return [%]" ^
  --top-n 10 ^
  --start 2023-01-01 ^
  --end 2024-01-01
```

### Sort by Different Metrics

```batch
# Sort by Sharpe Ratio (default)
--sort-by "Sharpe Ratio"

# Sort by Total Return
--sort-by "Total Return [%]"

# Sort by Win Rate
--sort-by "Win Rate [%]"

# Sort by Max Drawdown (best = smallest drawdown)
--sort-by "Max Drawdown [%]"
```

### Parallel Execution

Control parallel workers:

```batch
# Use 8 parallel workers (faster for many symbols)
python src\backtest_runner.py ^
  --strategy MovingAverageCrossover ^
  --universe DOW30 ^
  --sweep ^
  --parallel ^
  --max-workers 8 ^
  --start 2023-01-01 ^
  --end 2024-01-01
```

---

## Troubleshooting

### Slow Execution

**Problem:** Sweep is taking too long

**Solutions:**
- Use `--parallel` flag
- Increase `--max-workers` (e.g., 8 or 16)
- Reduce date range
- Use smaller universe (FAANG instead of DOW30)

### Out of Memory

**Problem:** System runs out of memory

**Solutions:**
- Reduce `--max-workers` (try 2 or 1)
- Run sequentially (remove `--parallel`)
- Close other applications

### Some Symbols Fail

**Problem:** Some symbols show errors

**Expected Behavior:** SweepRunner continues even if some symbols fail. Check error messages in output.

**Common Causes:**
- Symbol not in database
- Insufficient data for date range
- Strategy parameters incompatible with data

**Solution:**
- Check data availability for failed symbols
- Adjust date range
- Review error messages in output

---

## Tips & Best Practices

### 1. Start Small
- Test with FAANG (5 stocks) first using `quick_sweep.bat`
- Once working, scale up to larger universes

### 2. Use Parallel Execution
- Speeds up sweeps by 3-4x
- Safe for most systems with 4+ workers
- Increase workers for many symbols (50+)

### 3. Compare Results
- Use `compare_strategies.bat` to find best strategy for a universe
- Use `sweep_all_sectors.bat` to find best sectors for a strategy
- Review HTML reports for visual comparison

### 4. Custom Parameters
- Test parameters with `quick_sweep` first
- Use full sweep once parameters validated
- Consider universe-wide optimization (see API docs)

### 5. Output Organization
- Use meaningful `--run-name` for easy identification
- Results auto-saved with timestamp
- HTML reports easier to read than CSV for quick review

---

## Examples Gallery

### Example 1: Find Best Stocks for MA Crossover

```batch
sweep_universe.bat DOW30 MovingAverageCrossover
```

**Result:** CSV/HTML report showing which DOW30 stocks work best with MA crossover

### Example 2: Test Multiple Strategies on FAANG

```batch
compare_strategies.bat FAANG
```

**Result:** 5 reports comparing different strategies on same stocks

### Example 3: Find Best Sector for Breakout Strategy

```batch
sweep_all_sectors.bat BreakoutStrategy
```

**Result:** 6 reports showing which sector performs best with breakout strategy

### Example 4: Quick Parameter Testing

```batch
quick_sweep.bat MeanReversion "window=30,num_std=2.5"
quick_sweep.bat MeanReversion "window=20,num_std=2.0"
quick_sweep.bat MeanReversion "window=15,num_std=1.5"
```

**Result:** Quick comparison of different parameters

---

## See Also

- **[API Reference - Sweep](../../docs/API_REFERENCE_SWEEP.md)** - Complete API documentation
- **[Strategy Implementation Report](../../docs/STRATEGY_IMPLEMENTATION_REPORT.md)** - Implementation details
- **[Advanced Strategies Guide](../../docs/ADVANCED_STRATEGIES_GUIDE.md)** - Strategy documentation
- **[Backtesting Guide](../../docs/BACKTESTING_GUIDE.md)** - General backtesting guide

---

## Command Reference

### Quick Reference Table

| Script | Purpose | Usage |
|--------|---------|-------|
| `list_universes` | Show available universes | `list_universes.bat` |
| `sweep_universe` | Sweep any universe | `sweep_universe.bat DOW30 MovingAverageCrossover` |
| `quick_sweep` | Fast testing (FAANG) | `quick_sweep.bat BreakoutStrategy` |
| `compare_strategies` | Compare strategies | `compare_strategies.bat FAANG` |
| `sweep_all_sectors` | Test all sectors | `sweep_all_sectors.bat MomentumStrategy` |

---

**For detailed API documentation, see [API_REFERENCE_SWEEP.md](../../docs/API_REFERENCE_SWEEP.md)**
