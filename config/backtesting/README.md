# Backtesting Configuration Files

This directory contains YAML configuration files for running backtests.

## Production Configs (Root Level)

These configs are actively used or referenced in production code:

| Config | Strategy | Description |
|--------|----------|-------------|
| `omr_backtest.yaml` | OMR (Overnight Mean Reversion) | Production config for leveraged ETF overnight strategy |
| `ma_single.yaml` | Moving Average | Example single-symbol MA crossover config |
| `ma_sweep.yaml` | Moving Average | Parameter sweep for MA optimization |
| `orb_baseline.yaml` | ORB (Opening Range Breakout) | Baseline config for ORB strategy |
| `orb_single.yaml` | ORB | Single-symbol ORB config |
| `ict_production.yaml` | ICT/SMC | Production config for ICT strategy |

## Usage

Run backtests using the config-driven runner:

```bash
# Single backtest
python -m src.backtest_runner --config config/backtesting/omr_backtest.yaml

# Parameter sweep
python -m src.backtest_runner --config config/backtesting/ma_sweep.yaml --mode sweep
```

## Experimental Configs

The `experimental/` subdirectory contains:
- Parameter sweep variants
- Optimization experiments
- Test configurations
- Strategy variants being researched

These are NOT referenced in production code and may be incomplete or outdated.

## Config Schema

All configs should follow this structure:

```yaml
strategy:
  name: <strategy_display_name>
  class: <strategy_class_name>
  params:
    # Strategy-specific parameters

backtest:
  start_date: YYYY-MM-DD
  end_date: YYYY-MM-DD
  symbols: [SYM1, SYM2] or symbols_file: path/to/file.csv

data:
  timeframe: minute | hour | day
  source: streaming  # Use StreamingDataLoader

engine:
  initial_capital: 100000
  fees: 0.001
  slippage: 0.0
  market_hours_only: true

risk:
  position_size_pct: 0.10
  use_stop_loss: false

optimization:  # Optional - for sweep mode
  method: grid | bayesian | random
  param_grid:
    param_name: [val1, val2, val3]
```

## Adding New Configs

1. Production configs: Add to root level with clear naming
2. Experimental configs: Add to `experimental/` subdirectory
3. Update this README when adding production configs
