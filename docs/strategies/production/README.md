# Production Strategies

Strategies currently deployed on EC2 for live/paper trading.

## Active Strategies

| Strategy | Description | Deployed |
|----------|-------------|----------|
| [RAMP](RAMP_STRATEGY.md) | Regime-Aware Momentum Protection for S&P 500 | 2025-12-08 |
| [OMR](OMR_STRATEGY.md) | Overnight Mean Reversion on leveraged ETFs | 2025-11-XX |

## Deployment Details

- **EC2 Instance**: See `docs/INFRASTRUCTURE_OVERVIEW.md`
- **Service**: `homeguard-multi.service` runs both strategies
- **Schedule**:
  - OMR Entry: 3:50 PM ET
  - RAMP Rebalance: 3:55 PM ET
  - OMR Exit: 9:31 AM ET (next day)

## Adding New Strategies

Before adding a strategy to production:
1. Complete walk-forward validation (see `docs/strategies/research/`)
2. Document in this directory with `{STRATEGY}_STRATEGY.md` naming
3. Update deployment scripts in `infra/ec2/`
4. Add to systemd service configuration
