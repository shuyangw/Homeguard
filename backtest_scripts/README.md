# Backtest Scripts - Standardized Infrastructure

> **✨ NEW**: Standardized backtesting infrastructure with YAML configuration, reusable frameworks, and common CLI interface.

Standardized system for strategy validation, optimization, and pairs discovery that eliminates code duplication and enables configuration-driven backtesting.

## 🚀 Quick Start

```bash
# Run validation with default config
python example_validation_with_framework.py

# Use custom config with CLI overrides
python example_validation_with_framework.py \
  --config config/pairs_trading.yaml \
  --start-date 2023-01-01 \
  --symbols SPY IWM

# Use predefined universes and date ranges
python example_validation_with_framework.py \
  --universe production.conservative \
  --date-range bull_2019_2021
```

## 📊 What's New

### Configuration System
- **YAML configs** with inheritance (`extends`)
- **150+ predefined symbols** organized by category
- **Predefined date ranges** (bull/bear markets, crises)
- **CLI overrides** for any config value

### Reusable Frameworks
- **ValidationFramework** - Eliminates ~2,000 lines of duplicated code
- **OptimizationFramework** - Eliminates ~1,800 lines of duplicated code
- **PairsDiscoveryFramework** - Eliminates ~1,500 lines of duplicated code

### Results
- **87% code reduction** (250 lines → 40 lines per script)
- **Zero hardcoding** - all parameters configurable
- **Parallel execution** - built-in multi-core support
- **Standardized output** - CSV/JSON/Markdown exports

## 📁 Directory Structure

```
backtest_scripts/
├── config/                      # YAML configuration
│   ├── default_backtest.yaml    # Base configuration
│   ├── pairs_trading.yaml       # Pairs strategy
│   ├── overnight_strategy.yaml  # OMR strategy
│   ├── universes.yaml           # Symbol lists
│   └── date_ranges.yaml         # Time periods
│
├── frameworks/                  # Reusable frameworks
│   ├── validation_framework.py  # Validation pipeline
│   ├── optimization_framework.py # Optimization pipeline
│   └── pairs_discovery_framework.py # Pairs discovery
│
├── utils/                       # Common utilities
│   ├── path_setup.py
│   ├── config_loader.py
│   └── cli_args.py
│
└── *.py                        # Backtest scripts
```

## 🔧 Configuration Examples

### Basic Validation

```bash
python example_validation_with_framework.py \
  --config config/pairs_trading.yaml
```

### Custom Parameters

```bash
python example_validation_with_framework.py \
  --config config/pairs_trading.yaml \
  --start-date 2023-01-01 \
  --end-date 2024-12-31 \
  --symbols SPY QQQ IWM \
  --position-size 0.15
```

### Predefined Universes

```bash
# Conservative universe (SPY, QQQ, IWM)
python example_validation_with_framework.py \
  --universe production.conservative

# FAANG stocks
python example_validation_with_framework.py \
  --universe technology.faang

# 3x leveraged ETFs
python example_validation_with_framework.py \
  --universe leveraged.triple_long
```

### Predefined Date Ranges

```bash
# Bull market period (2019-2021)
python example_validation_with_framework.py \
  --date-range bull_2019_2021

# Bear market (2022)
python example_validation_with_framework.py \
  --date-range bear_2022

# COVID crash
python example_validation_with_framework.py \
  --date-range covid_crash
```

## 📝 Using Frameworks

### ValidationFramework

```python
from frameworks.validation_framework import ValidationFramework
from strategies.advanced.pairs_trading import PairsTrading
from utils.cli_args import create_base_parser, parse_with_config

# Parse arguments and load config
parser = create_base_parser()
args, config = parse_with_config(parser)

# Initialize framework
framework = ValidationFramework(config)

# Run validation
framework.run_validation(
    strategy_class=PairsTrading,
    symbols=["SPY", "IWM"]
)

# Export results
framework.export_results(format='both')
framework.generate_report()
```

### OptimizationFramework

```python
from frameworks.optimization_framework import OptimizationFramework

framework = OptimizationFramework(config)

# Define parameter grid
param_grid = {
    'entry_zscore': [1.5, 2.0, 2.5],
    'exit_zscore': [0.0, 0.5, 1.0]
}

# Run grid search (with parallel execution)
results = framework.grid_search(
    strategy_class=PairsTrading,
    symbols=["SPY", "IWM"],
    param_grid=param_grid
)

framework.export_best_params()
```

### PairsDiscoveryFramework

```python
from frameworks.pairs_discovery_framework import PairsDiscoveryFramework

framework = PairsDiscoveryFramework(config)

# Discover cointegrated pairs
pairs = framework.discover_pairs(
    universe=['SPY', 'QQQ', 'IWM', 'DIA'],
    method='cointegration'
)

framework.export_pairs()
framework.generate_report()
```

## 🎯 Common CLI Arguments

All scripts support these arguments:

- `--config PATH` - YAML configuration file
- `--start-date DATE` - Backtest start (YYYY-MM-DD)
- `--end-date DATE` - Backtest end (YYYY-MM-DD)
- `--date-range NAME` - Predefined range (bull_2019_2021, etc.)
- `--symbols SYM1 SYM2` - Symbol list
- `--universe PATH` - Predefined universe (production.conservative, etc.)
- `--position-size PCT` - Position size (0.0-1.0)
- `--commission PCT` - Commission rate
- `--optimize` - Run optimization
- `--n-jobs N` - Parallel processes
- `--verbose` - DEBUG logging
- `--quiet` - Minimal logging

## 📈 Before & After

### Before (300 lines of boilerplate)

```python
# Hardcoded parameters
START_DATE = "2020-01-01"
END_DATE = "2024-12-31"
SYMBOLS = ["SPY", "IWM"]
INITIAL_CASH = 100000

# Manual backtest execution
# Custom result aggregation
# Manual report generation
# ... 250+ more lines ...
```

### After (40 lines with framework)

```python
from frameworks.validation_framework import ValidationFramework
from utils.cli_args import create_base_parser, parse_with_config

parser = create_base_parser()
args, config = parse_with_config(parser)

framework = ValidationFramework(config)
framework.run_validation(...)
framework.export_results()
framework.generate_report()
```

**Result: 87% code reduction**

## 📚 Documentation

- **Configuration Details**: See YAML files in `config/`
- **Framework API**: See docstrings in `frameworks/*.py`
- **Standardization Plan**: See `docs/todos/20251117_BACKTEST_SCRIPTS_STANDARDIZATION.md`
- **Project Guidelines**: See `CLAUDE.md` in project root

## 🔍 Troubleshooting

**"No module named 'utils.path_setup'"**
- Run from project root: `python backtest_scripts/script.py`

**"Config file not found"**
- Use path relative to `config/`: `--config pairs_trading.yaml`

**"No symbols specified"**
- Provide via CLI or config: `--symbols SPY QQQ` or `--universe production.conservative`

---

**Last Updated**: 2025-11-18
**Standardization Status**: Phase 1-3 Complete ✅
