# Backtesting Configuration Files

One canonical YAML config per strategy. Parameter sweeps and research variants go in `scratch/` (gitignored).

## Canonical Configs

| Config | Strategy | Description |
|--------|----------|-------------|
| `omr_backtest.yaml` | OMR | Overnight Mean Reversion on leveraged ETFs |
| `orb_baseline.yaml` | ORB | Opening Range Breakout |
| `hv_orb_baseline.yaml` | HV ORB | High Volatility ORB (Stocks in Play) |
| `ict_production.yaml` | ICT/SMC | Smart Money Concepts / ICT |
| `bmsb_crypto.yaml` | BMSB | Bull Market Support Band (crypto) |
| `ml_crypto_mr_baseline.yaml` | ML Crypto MR | ML-based Crypto Mean Reversion |
| `hurst_mr_baseline.yaml` | Hurst MR | Hurst Exponent Mean Reversion |
| `cscm_baseline.yaml` | CSCM | Cross-Sectional Crypto Momentum |
| `dsts_btc.yaml` | DSTS | Dual Signal Trend Sentinel |
| `frs_crypto.yaml` | FRS | Fractal Regime Switching |
| `evr_crypto.yaml` | EVR | Effort vs Result (Volume Spread Analysis) |
| `opex_pinning.yaml` | OpEx | Options Expiration Pinning |
| `ma_single.yaml` | MA | Moving Average Crossover (reference/example) |

## Usage

```bash
python -m src.backtest_runner --config config/backtesting/omr_backtest.yaml
```

## Scratch Configs

The `scratch/` subdirectory is **gitignored** and holds:
- Parameter sweep variants
- Optimization experiments
- Research configurations

If a scratch config proves valuable, promote it here as the new canonical config.

## Config Schema

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
  source: streaming

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
