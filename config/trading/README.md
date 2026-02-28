# Live Trading Configuration

YAML configs for production trading on EC2.

## Files

| Config | Purpose |
|--------|---------|
| `strategy_toggle.yaml` | Enable/disable live strategies (modified by API) |
| `broker_alpaca.yaml` | Alpaca API connection and order defaults |
| `omr_trading_config.yaml` | OMR strategy params: timing, sizing, risk limits |
| `omr_expanded_config.yaml` | OMR expanded symbol universe variant |
| `omr_position_configs.yaml` | OMR position manager settings |
| `cscm_live.yaml` | CSCM live trading params |
| `momentum_trading_config.yaml` | Momentum Protection / RAMP params |

## Active Strategies

See `strategy_toggle.yaml` for current enable/disable state. Currently:
- **OMR** - Enabled (entry 3:50 PM, exit 9:31 AM ET)
- **RAMP** - Enabled (rebalance 3:55 PM ET)
- **MP** - Enabled (Momentum Protection)
- **CSCM** - Disabled

## Notes

- API keys are loaded from environment variables (`.env`), never hardcoded
- `strategy_toggle.yaml` is modified programmatically by the trading API
- All configs use paper trading by default (`enable_paper_trading_only: true`)
