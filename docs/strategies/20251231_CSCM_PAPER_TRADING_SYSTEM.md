# CSCM Paper Trading System

Paper trading implementation for the Cross-Sectional Crypto Momentum (CSCM) strategy.

## Overview

This system enables paper trading of the CSCM strategy with:
- **Alpaca** for live price data (primary)
- **Binance** as fallback data source
- **Alpaca paper trading** for order execution
- **Monday weekly rebalancing at 21:30 UTC** (instead of Sunday for live)
- Portfolio tracking during equity hours + on-demand refresh
- EC2 shell commands for quick status queries
- Discord slash commands for remote monitoring

## Architecture

```
[Alpaca API] -----> [AlpacaCryptoBroker] ------+
     |                                          |
     | (fallback after 3 failures)              v
     |                              [CryptoDataProviderWithFallback]
     v                                          |
[Binance API] ----> [BinanceDataProvider] -----+
                                                |
                                                v
                                    [CSCMPaperAdapter]
                                          |
                     +--------------------+--------------------+
                     |                    |                    |
                     v                    v                    v
            [CSCM Strategy]     [Alpaca Paper Trading]  [PortfolioTracker]
                     |                    |                    |
                     v                    v                    v
              Momentum Signals     Order Execution      Value Updates
                                                              |
                     +----------------------------------------+
                     |                    |
                     v                    v
              [EC2 Commands]      [Discord Bot]
```

## Components

### 1. Binance Data Provider

**Location**: `src/data/providers/binance.py`

Provides live crypto price data from Binance REST API.

**Features**:
- Rate limiting (100ms between requests, 1200 req/min max)
- Exponential backoff retries (max 3 attempts)
- Symbol normalization (BTC/USD -> BTCUSDT)
- Historical OHLCV bars and current prices

**Symbol Mapping**:

| Internal | Binance | Alpaca |
|----------|---------|--------|
| BTC/USD  | BTCUSDT | BTCUSD |
| ETH/USD  | ETHUSDT | ETHUSD |
| SOL/USD  | SOLUSDT | SOLUSD |

### 2. Failover Wrapper

**Location**: `src/data/providers/binance.py` (CryptoDataProviderWithFallback)

Provides automatic failover from Alpaca to Binance.

**Behavior**:
- Uses Alpaca by default (live prices via broker API, historical via local parquet)
- After 3 consecutive failures, switches to Binance for 5 minutes
- Automatically retries Alpaca after fallback window expires
- Logs all failover events

### 3. CSCM Paper Adapter

**Location**: `src/trading/adapters/cscm_paper_adapter.py`

Paper trading variant of the CSCM strategy.

**Differences from Live Adapter**:
- Uses Alpaca for price data (with Binance fallback)
- Uses Alpaca paper trading for execution
- Rebalances on Monday (instead of Sunday)
- Separate state file (`cscm_paper_state.pkl`)

**Configuration**:
```python
adapter = CSCMPaperAdapter(
    universe=['BTC/USD', 'ETH/USD', 'SOL/USD', ...],  # Default: top 20 cryptos
    top_n=5,                                           # Positions to hold
    rebalance_day='monday',                            # Weekly rebalance day
    rebalance_hour_utc=21,                             # Rebalance at 21:30 UTC
    rebalance_minute_utc=30,
    trailing_stop_pct=0.25,                            # 25% trailing stop
)
```

### 4. Portfolio Tracker

**Location**: `src/trading/portfolio/tracker.py`

Tracks portfolio value with persistence.

**Features**:
- Equity hours detection (9:30 AM - 4:00 PM EST, Mon-Fri)
- Scheduled updates every 15 minutes during equity hours
- On-demand refresh via `force=True`
- Value history persistence to parquet
- Performance summary (returns, drawdown)

**Storage**:
- State: `~/.homeguard/cache/cscm_portfolio/tracker_state.pkl`
- History: `~/.homeguard/cache/cscm_portfolio/portfolio_history.parquet`

### 5. EC2 Commands

Shell scripts for quick status queries on EC2.

| Command | Script | Purpose |
|---------|--------|---------|
| `cscm-status` | `infra/ec2/cscm_status.sh` | Portfolio value, positions, regime |
| `cscm-refresh` | `infra/ec2/cscm_refresh.sh` | Force refresh portfolio value |
| `cscm-positions` | `infra/ec2/cscm_positions.sh` | Detailed position breakdown |

### 6. Discord Bot

**Location**: `src/discord_cscm/bot.py`

Simple slash commands for remote monitoring (no LLM required).

**Commands**:
- `/cscm-status` - Portfolio value, cash, positions, regime
- `/cscm-positions` - Detailed position table with P&L
- `/cscm-refresh` - Force update portfolio value
- `/cscm-regime` - BTC regime and top momentum coins

## Deployment

### EC2 Service

**Service File**: `infra/ec2/services/homeguard-cscm-paper.service`

```bash
# Install service
sudo cp infra/ec2/services/homeguard-cscm-paper.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable homeguard-cscm-paper
sudo systemctl start homeguard-cscm-paper

# Check status
sudo systemctl status homeguard-cscm-paper

# View logs
sudo journalctl -u homeguard-cscm-paper -f
```

### Environment Variables

Add to `.env`:

```bash
# Alpaca Paper Trading
ALPACA_PAPER_KEY_ID="your_paper_key"
ALPACA_PAPER_SECRET_KEY="your_paper_secret"

# Binance (optional - for higher rate limits)
BINANCE_API_KEY=""
BINANCE_API_SECRET=""

# Discord Bot
DISCORD_CSCM_TOKEN="your_discord_bot_token"
DISCORD_CSCM_GUILD_ID=""  # Optional: for instant command sync

# CSCM Configuration
CSCM_REBALANCE_DAY="monday"
```

### Discord Bot Setup

1. Create application at https://discord.com/developers/applications
2. Create bot and copy token
3. Enable "applications.commands" scope
4. Invite bot to server with slash command permissions
5. Add token to `.env` as `DISCORD_CSCM_TOKEN`

## Usage

### Local Testing

```python
from src.trading.adapters.cscm_paper_adapter import CSCMPaperAdapter
from src.trading.portfolio.tracker import PortfolioTracker

# Create adapter
adapter = CSCMPaperAdapter()

# Get current status
status = adapter.get_status()
print(f"Regime: {status['regime']}")
print(f"Positions: {status['positions']}")

# Create tracker
tracker = PortfolioTracker(adapter)

# Get portfolio value
value = tracker.update_value(force=True)
print(f"Portfolio Value: ${value:,.2f}")

# Get performance summary
perf = tracker.get_performance_summary()
print(f"Total Return: {perf['total_return_pct']:.2f}%")
```

### Running the Service

```bash
# Main entry point
python scripts/trading/run_cscm_paper.py

# Or via systemd
sudo systemctl start homeguard-cscm-paper
```

## Testing

Unit tests for all components:

```bash
# Run all CSCM paper trading tests
pytest tests/data/providers/test_binance.py \
       tests/trading/test_cscm_paper_adapter.py \
       tests/trading/test_portfolio_tracker.py -v
```

**Test Coverage**:
- Binance provider: 22 tests (symbol normalization, rate limiting, retries, failover)
- CSCM paper adapter: 12 tests (initialization, data fetching, status, factory)
- Portfolio tracker: 14 tests (equity hours, updates, history, performance)

## File Structure

```
src/
    data/providers/
        binance.py                    # Binance REST API + failover wrapper
        __init__.py                   # Exports BinanceDataProvider

    trading/
        adapters/
            cscm_paper_adapter.py     # Paper trading adapter
            __init__.py               # Exports CSCMPaperAdapter

        portfolio/
            __init__.py               # Portfolio module
            tracker.py                # Value tracking with persistence

    discord_cscm/
        __init__.py                   # Discord module
        bot.py                        # Slash command bot

scripts/
    ec2/
        cscm_status.sh               # Quick status command
        cscm_refresh.sh              # Force refresh command
        cscm_positions.sh            # Position details command
        services/
            homeguard-cscm-paper.service  # Systemd service

    trading/
        run_cscm_paper.py            # Main entry point

tests/
    data/providers/
        test_binance.py              # Binance provider tests
    trading/
        test_cscm_paper_adapter.py   # Adapter tests
        test_portfolio_tracker.py    # Tracker tests
```

## Risk Considerations

1. **Alpaca Rate Limits**: Standard API rate limits apply
2. **Price Divergence**: Alpaca (USD) vs Binance (USDT) may differ 0.1-1%
3. **Failover Latency**: 5-minute fallback window on Alpaca failures
4. **Paper vs Live**: Paper trading results may differ from live due to:
   - No slippage simulation
   - Perfect fills at quoted prices
   - Different fee structures

## Monitoring

### Logs

```bash
# EC2 service logs
sudo journalctl -u homeguard-cscm-paper -f --output=cat

# Discord bot logs
sudo journalctl -u homeguard-cscm-discord -f --output=cat
```

### Status Checks

```bash
# Service status
sudo systemctl status homeguard-cscm-paper

# Quick portfolio check
cscm-status

# Force refresh
cscm-refresh
```

### Discord

Use `/cscm-status` in your Discord channel for remote monitoring.
