# Configuration System

**Centralized configuration with application settings, YAML-driven backtesting configs, and Pydantic validation.**

**Last Updated**: 2025-12-08

---

## Overview

### What It Does
- Provides cross-platform path resolution (Windows, macOS, Linux)
- Loads YAML configuration files with inheritance support
- Validates backtest configurations with Pydantic models
- Manages directory paths for data, logs, outputs, and models

### Key Features
- **OS Detection**: Auto-detect Windows, macOS, or Linux (EC2)
- **YAML Inheritance**: `extends` directive for config composition
- **Pydantic Validation**: Type-safe configs with automatic validation
- **Presets**: Date ranges and symbol universes for common scenarios
- **CLI Overrides**: Dot-notation overrides from command line

### Use Cases
- Configure backtest parameters via YAML files
- Access platform-specific storage paths
- Validate configuration before running backtests
- Create reusable config templates with inheritance

---

## Architecture

```
src/settings/
├── __init__.py              # Public API exports
├── settings.py              # Application settings from settings.ini
├── loader.py                # YAML loading, merging, validation
├── schema.py                # Pydantic models for validation
└── defaults.py              # Default configs, presets, universes
```

### Design Philosophy

1. **Separation of Concerns**: Application settings vs backtest configs
2. **OS Abstraction**: Platform differences hidden from application code
3. **Type Safety**: Pydantic models catch config errors early
4. **Inheritance**: Base configs reduce duplication
5. **Sensible Defaults**: Every setting has a reasonable default

---

## Key Components

### Application Settings (`settings.py`)

**Purpose**: Load platform-specific paths from `settings.ini`.

**Key Functions**:
- `get_local_storage_dir()`: Data storage path
- `get_output_dir()`: Backtest output directory
- `get_log_output_dir()`: Log file directory
- `get_backtest_results_dir()`: Results directory
- `get_live_trading_dir()`: Live trading data
- `get_models_dir()`: ML model storage
- `get_discord_bot_log_dir()`: Discord bot logs

**Platform Paths** (from `settings.ini`):
| Platform | Storage |
|----------|---------|
| Windows | `F:\Stock_Data` |
| macOS | `/Users/shuyangw/Library/CloudStorage/Dropbox/cs/stonk/data` |
| Linux/EC2 | `/home/ec2-user/stock_data` |

**Usage**:
```python
from src.settings import get_local_storage_dir, get_output_dir

# Get platform-appropriate paths
data_dir = get_local_storage_dir()
# Windows: F:\Stock_Data
# macOS: /Users/.../stonk/data
# EC2: /home/ec2-user/stock_data

output_dir = get_output_dir()
# Auto-creates if doesn't exist
```

### Config Loader (`loader.py`)

**Purpose**: Load, merge, and validate YAML configurations.

**Key Functions**:
- `load_config(path, overrides)`: Load and validate to BacktestConfig
- `load_config_dict(path, overrides)`: Load as raw dict
- `load_yaml(path)`: Load raw YAML file
- `merge_dicts(base, override)`: Deep merge dictionaries
- `apply_overrides(config, overrides)`: Apply dot-notation overrides
- `get_nested(config, key, default)`: Get nested value

**Features**:
- **Inheritance**: Use `extends: parent.yaml` to inherit from another config
- **Default Merging**: Automatically merges with DEFAULT_CONFIG
- **CLI Overrides**: `{"backtest.start_date": "2023-01-01"}` format

**Usage**:
```python
from src.settings import load_config

# Load from YAML
config = load_config("config/backtesting/omr.yaml")

# Load with overrides
config = load_config("config.yaml", overrides={
    "backtest.initial_capital": 50000,
    "dates.start": "2023-01-01"
})

# Access validated settings
print(config.strategy.name)
print(config.backtest.initial_capital)
```

### Pydantic Schema (`schema.py`)

**Purpose**: Type-safe configuration models with validation.

**Key Models**:

| Model | Purpose |
|-------|---------|
| `BacktestConfig` | Root config for all backtests |
| `StrategyConfig` | Strategy name and parameters |
| `SymbolsConfig` | Symbol list, universe, or file |
| `DatesConfig` | Date range or preset |
| `BacktestSettings` | Capital, fees, slippage |
| `RiskSettings` | Position sizing, stop loss |
| `SweepSettings` | Multi-symbol sweep options |
| `OptimizationSettings` | Parameter optimization |
| `WalkForwardSettings` | Walk-forward validation |
| `OutputSettings` | Output and reporting |

**Enums**:
- `BacktestMode`: `single`, `sweep`, `optimize`, `walk_forward`
- `PositionSizingMethod`: `fixed_percent`, `kelly`, `volatility_target`, etc.

**Usage**:
```python
from src.settings import BacktestConfig, validate_config

# Validate a dictionary
config_dict = {
    "strategy": {"name": "MovingAverageCrossover", "parameters": {"fast": 10}},
    "symbols": {"list": ["AAPL", "MSFT"]},
    "dates": {"start": "2023-01-01", "end": "2024-01-01"}
}
config = validate_config(config_dict)
```

### Defaults (`defaults.py`)

**Purpose**: Default values, date presets, and symbol universes.

**Date Presets**:
```python
DATE_PRESETS = {
    "full_periods.one_year": {"start": "2024-01-01", "end": "2024-12-31"},
    "full_periods.three_years": {"start": "2022-01-01", "end": "2024-12-31"},
    "full_periods.five_years": {"start": "2020-01-01", "end": "2024-12-31"},
    "full_periods.ten_years": {"start": "2015-01-01", "end": "2024-12-31"},
}
```

**Symbol Universes**:
```python
SYMBOL_UNIVERSES = {
    "production.conservative": ["TQQQ", "SOXL", "UPRO"],
    "production.aggressive": [...],
    "testing.small": ["AAPL", "MSFT", "GOOGL"],
}
```

**Usage**:
```python
from src.settings import get_date_preset, get_symbol_universe

# Use preset in YAML: dates.preset: "full_periods.five_years"
dates = get_date_preset("full_periods.five_years")
# {"start": "2020-01-01", "end": "2024-12-31"}

symbols = get_symbol_universe("production.conservative")
# ["TQQQ", "SOXL", "UPRO", ...]
```

---

## Data Flow

```
YAML Config File
        v
  load_yaml() -> Raw Dict
        v
  resolve_extends() -> Merge Parent Configs
        v
  merge_dicts(DEFAULT_CONFIG, ...) -> Full Dict
        v
  apply_overrides(CLI args) -> Final Dict
        v
  BacktestConfig.model_validate() -> Validated Config
        v
  Backtest Engine
```

---

## Public API

### Application Settings

```python
from src.settings import (
    settings,                    # ConfigParser instance
    OS_ENVIRONMENT,              # "windows", "macos", or "linux"
    PROJECT_ROOT,                # Path to project root
    get_local_storage_dir,       # Data storage path
    get_output_dir,              # Backtest output
    get_log_output_dir,          # Log directory
    get_backtest_results_dir,    # Results directory
    get_live_trading_dir,        # Live trading data
    get_models_dir,              # ML models
    get_discord_bot_log_dir,     # Bot logs
)
```

### Config-Driven Backtesting

```python
from src.settings import (
    # Config loading
    load_config,          # Load and validate
    load_config_dict,     # Load as dict
    validate_config,      # Validate dict
    # Schema classes
    BacktestConfig,
    BacktestMode,
    PositionSizingMethod,
    StrategyConfig,
    SymbolsConfig,
    DatesConfig,
    # Presets
    get_date_preset,
    get_symbol_universe,
)
```

---

## Configuration

### settings.ini Structure

```ini
[os]
environment = windows  # or macos, linux

[directories]
local_storage_dir = F:\Stock_Data
output_dir = output/
log_output_dir = logs/
backtest_results_dir = output/backtests/
live_trading_dir = data/trading/
models_dir = data/models/
discord_bot_log_dir = logs/discord_bot/

[tearsheet]
frequency = weekly  # or daily, monthly
```

### Example YAML Config

```yaml
# config/backtesting/ma_single.yaml
extends: default  # Inherit from DEFAULT_CONFIG

mode: single

strategy:
  name: MovingAverageCrossover
  parameters:
    fast_period: 10
    slow_period: 50

symbols:
  list: ["AAPL", "MSFT", "GOOGL"]
  # OR
  # universe: "production.conservative"
  # OR
  # file: "config/universes/sp500-2025.csv"

dates:
  start: "2023-01-01"
  end: "2024-01-01"
  # OR
  # preset: "full_periods.one_year"

backtest:
  initial_capital: 100000
  fees: 0.001
  slippage: 0.0005

risk:
  enabled: true
  position_sizing_method: fixed_percent
  position_size_pct: 0.10
  max_positions: 5
```

---

## Dependencies

### Internal (src/ modules)
- None (settings is a foundational module)

### External (pip packages)
- `pydantic` - Configuration validation
- `pyyaml` - YAML loading

---

## Error Handling

| Error | Cause | Solution |
|-------|-------|----------|
| `FileNotFoundError` | Config file missing | Check file path |
| `ValidationError` | Invalid config value | Check schema requirements |
| `KeyError` | Missing required field | Add required field |
| `yaml.YAMLError` | Invalid YAML syntax | Fix YAML formatting |

---

## Testing

### Test Location
- `tests/settings/` - Unit tests

### Running Tests
```bash
pytest tests/settings/ -v
```

---

## Related Documentation

- [Architecture Overview](../../docs/architecture/ARCHITECTURE_OVERVIEW.md)
- [Backtesting Engine](../backtesting/BACKTESTING_ENGINE.md)
- [CLAUDE.md - Configuration](../../CLAUDE.md)

---

## Changelog

- **2025-12-08**: Initial documentation created
- **2025-11-XX**: Pydantic schema validation added
- **2025-10-XX**: YAML config inheritance
- **2025-09-XX**: Initial settings module
