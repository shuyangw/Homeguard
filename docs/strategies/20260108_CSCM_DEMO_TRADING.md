# CSCM Demo Trading System

**Date:** 2026-01-08
**Status:** Production Ready (Paper Trading)

## Overview

The CSCM (Cross-Sectional Crypto Momentum) Demo Trading System is a self-contained paper trading platform for testing the CSCM strategy with real-time Binance data and simulated execution. It runs independently of any external paper trading platform.

### Key Features

- Real-time Binance WebSocket streaming for live crypto prices
- Simulated order execution with configurable slippage and fees
- BTC regime filter (reduces exposure in bear markets)
- Weekly rebalancing with momentum-based position selection
- Trailing stop and profit target protection
- Persistent portfolio state across restarts
- EC2 deployment with Lambda-based scheduling

---

## Architecture

```
Historical Data (Primary):
+------------------+     +---------------------+     +------------------+
|  Binance.US REST | --> | BinanceDataProvider | --> | CSCMDemoAdapter  |
| (60-day klines)  |     | (api.binance.us)    |     | (Strategy Logic) |
+------------------+     +---------------------+     +------------------+
                                                             |
                                                             v
                                                     +------------------+
                                                     |   CSCMSignals    |
                                                     | (Momentum Calc)  |
                                                     +------------------+

Real-time Quotes (Optional - with REST fallback):
+------------------+     +-------------------+     +------------------+
|  Binance.US WS   | --> | BinanceStreamMgr  | --> |   DemoBroker     |
|  (1-min bars)    |     | (stream.binance.us)|    | (Simulated Exec) |
+------------------+     +-------------------+     +------------------+
```

### Data Flow

| Data Type | Primary Source | Fallback | Used For |
|-----------|----------------|----------|----------|
| Historical (40-day SMA, 28-day momentum) | REST API | None | Regime detection, momentum ranking |
| Real-time quotes | Streaming buffer | REST API | Order execution prices |
| Portfolio values | Streaming buffer | REST API | P&L calculation |

### Components

| Component | Location | Purpose |
|-----------|----------|---------|
| `BinanceDataProvider` | `src/data/providers/binance.py` | REST API client for historical klines (api.binance.us) |
| `DemoBroker` | `src/trading/demo/demo_broker.py` | Simulated crypto execution with REST/streaming quotes |
| `BinanceStreamManager` | `src/streaming/binance_stream.py` | WebSocket connection for real-time 1-min bars (optional) |
| `CSCMDemoAdapter` | `src/trading/adapters/cscm_demo_adapter.py` | Strategy logic: momentum ranking, regime filter, rebalancing |
| `CSCMSignals` | `src/strategies/advanced/cscm_signals.py` | Momentum calculation and signal generation |
| `run_cscm_demo.py` | `scripts/trading/run_cscm_demo.py` | Service entry point for EC2 deployment |

### Binance.US Endpoints

| Endpoint | URL | Purpose |
|----------|-----|---------|
| REST API | `https://api.binance.us` | Historical klines, current prices |
| WebSocket | `wss://stream.binance.us:9443` | Real-time 1-minute bars (optional) |

Note: Binance.com is geo-blocked from US-based EC2 instances. All endpoints use Binance.US.

---

## Strategy Configuration

### Optimal Parameters (Backtested)

| Parameter | Value | Description |
|-----------|-------|-------------|
| Initial Cash | $100,000 | Starting capital |
| Top N | 5 | Number of positions to hold |
| Allocation | 18% | Capital per position (90% total max) |
| Trailing Stop | 8% | Exit if price drops 8% from peak |
| Profit Target | 20% | Take profit at 20% gain |
| Momentum Period | 28 days | Lookback for momentum calculation |
| BTC SMA Period | 40 days | BTC regime filter period |
| Rebalance Day | Sunday | Weekly rebalance at 00:00 UTC |

### Backtested Performance (2020-2024)

- **CAGR:** 19.5%
- **Sharpe Ratio:** 1.72
- **Max Drawdown:** 15.6%

### Universe (14 Coins)

```
BTC/USD, ETH/USD, SOL/USD, AVAX/USD, LINK/USD, DOGE/USD, DOT/USD,
LTC/USD, BCH/USD, UNI/USD, AAVE/USD, SUSHI/USD, XRP/USD, CRV/USD
```

Note: BTC is used for regime detection only (not traded).

---

## Execution Model

### Simulated Execution

The DemoBroker simulates realistic execution:

| Parameter | Default | Description |
|-----------|---------|-------------|
| Slippage | 5 bps | Price impact on execution |
| Fees | 10 bps | Trading fees per transaction |

### Order Types

- **Market Orders:** Immediate execution at current price +/- slippage
- **Close Position:** Sells entire position for a symbol

### State Persistence

Portfolio state is saved to `~/.homeguard/demo/`:

```
~/.homeguard/demo/
  portfolio_state.json      # Cash, positions, realized P&L
  cscm_adapter_state.json   # Entry prices, regime, trailing stops
  trades/                   # Trade log history (CSV)
```

---

## EC2 Deployment

### Service File

Location: `infra/ec2/services/homeguard-cscm-demo.service`

```bash
sudo cp infra/ec2/services/homeguard-cscm-demo.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable homeguard-cscm-demo
sudo systemctl start homeguard-cscm-demo
```

### Lambda Scheduling

The EC2 instance is automatically started/stopped for CSCM rebalance:

| Event | Schedule (UTC) | Description |
|-------|----------------|-------------|
| Start | Saturday 23:00 | 1 hour before rebalance |
| Stop | Sunday 00:10 | 10 minutes after rebalance |

Terraform resources in `infra/terraform/scheduled_start_stop.tf`:
- `aws_cloudwatch_event_rule.start_instance_sunday`
- `aws_cloudwatch_event_rule.stop_instance_sunday`

### EC2 Commands

After running `./infra/ec2/instance_setup_bashrc.sh --force`:

| Command | Description |
|---------|-------------|
| `cscm-start` | Start CSCM demo service |
| `cscm-stop` | Stop CSCM demo service |
| `cscm-restart` | Restart CSCM demo service |
| `cscm-status` | Show portfolio status and service state |
| `cscm-logs` | Stream live logs |
| `cscm-positions` | Show detailed position breakdown |
| `cscm-reset` | Reset portfolio to $100k (with confirmation) |
| `cscm-refresh` | Force immediate rebalance |

### EC2 Shell Scripts

| Script | Purpose |
|--------|---------|
| `cscm_demo_status.sh` | Portfolio summary, service status, recent logs |
| `cscm_demo_positions.sh` | Detailed position table with P&L |
| `cscm_demo_logs.sh` | Stream journalctl logs |
| `cscm_demo_reset.sh` | Reset portfolio with confirmation |
| `cscm_demo_refresh.sh` | Force rebalance |

---

## Strategy Logic

### Weekly Rebalance (Sunday 00:00 UTC)

1. **Fetch Data:** Get 28-day historical bars from Binance buffer
2. **Check Regime:** Calculate BTC 40-day SMA
   - BTC > SMA: Bull regime (trade normally)
   - BTC < SMA: Bear regime (go to cash)
3. **Calculate Momentum:** Rank coins by 28-day return
4. **Select Top N:** Pick top 5 momentum coins
5. **Rebalance:**
   - Close positions not in top 5
   - Open/adjust positions for top 5
   - Each position = 18% of portfolio

### Intraday Checks (Every Minute)

1. **Trailing Stop:** Exit if any position drops 8% from peak
2. **Profit Target:** Exit if any position gains 20% from entry
3. **Log Status:** Every 5 minutes log portfolio summary

### Risk Management

- **Position Sizing:** Fixed 18% per position (max 90% invested)
- **Trailing Stop:** 8% from highest price since entry
- **Profit Target:** 20% gain triggers exit
- **Regime Filter:** BTC below SMA -> exit all positions

---

## Usage

### Local Development

```bash
# Start with default config ($100k, market hours)
python scripts/trading/run_cscm_demo.py

# Custom starting capital
python scripts/trading/run_cscm_demo.py --cash 50000

# Run 24/7 instead of market hours only
python scripts/trading/run_cscm_demo.py --24-7

# Reset portfolio before starting
python scripts/trading/run_cscm_demo.py --reset
```

### CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--cash` | 100000 | Initial cash balance |
| `--slippage` | 5 | Slippage in basis points |
| `--fees` | 10 | Fees in basis points |
| `--top-n` | 5 | Number of positions |
| `--rebalance-day` | sunday | Day to rebalance |
| `--24-7` | False | Run 24/7 (default: market hours) |
| `--reset` | False | Reset portfolio before start |

### Programmatic Usage

```python
from src.trading.adapters.cscm_demo_adapter import CSCMDemoAdapter

# Create adapter with optimal config
adapter = CSCMDemoAdapter(
    universe=['BTC/USD', 'ETH/USD', 'SOL/USD', ...],
    top_n=5,
    allocation_pct=0.18,
    trailing_stop_pct=0.08,
    profit_target_pct=0.20,
    initial_cash=100000.0,
)

# Start streaming
adapter.start_streaming()

# Force rebalance
adapter.rebalance()

# Check status
status = adapter.get_status()
print(f"Regime: {status['regime']}")
print(f"Positions: {status['positions']}")

# Stop streaming
adapter.stop_streaming()
```

---

## Monitoring

### Portfolio Status

```bash
cscm-status
# Output:
# ==========================================
# CSCM Demo Trading Status
# ==========================================
# Service:        RUNNING
#
# --- Portfolio ---
# Total Value:    $102,450.00
# Cash:           $10,000.00
# Realized P&L:   $1,200.00
# Unrealized P&L: $1,250.00
#
# --- Status ---
# Positions:      5
# Streaming:      Yes
# Bars Buffered:  14400
```

### Position Details

```bash
cscm-positions
# Output:
# Symbol       Quantity       Entry      Current          P&L     P&L%
# ------------------------------------------------------------------------
# ETH/USD      5.234500    $2,150.00   $2,280.00      $680.00     6.0%
# SOL/USD     85.000000      $110.00     $118.50      $722.50     7.7%
# ...
```

### Live Logs

```bash
cscm-logs
# [CSCMDemo] Portfolio Status:
#   Total Value: $102,450.00
#   Cash: $10,000.00
#   Positions: 5
# [CSCMDemo] Current Positions:
#   ETH/USD: 5.234500 @ $2,150.00 (P&L: $680.00 / 6.0%)
```

---

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| Service won't start | Check `cscm-logs` for errors; verify websocket-client installed |
| No data received | Binance WebSocket may be blocked; check firewall |
| Positions not opening | Verify sufficient cash; check regime filter status |
| State not persisting | Ensure `~/.homeguard/demo/` directory exists |

### Reset Portfolio

If portfolio gets into bad state:

```bash
cscm-reset
# Confirms before deleting all state
# Reinitializes with $100k cash
```

Or with force flag (no confirmation):

```bash
cscm-reset --force
```

### View Raw State Files

```bash
cat ~/.homeguard/demo/portfolio_state.json | python -m json.tool
cat ~/.homeguard/demo/cscm_adapter_state.json | python -m json.tool
```

---

## Related Documentation

- [CSCM Strategy Overview](./CSCM_STRATEGY.md) (if exists)
- [DemoBroker Implementation](../../src/trading/demo/README.md)
- [Infrastructure Overview](../INFRASTRUCTURE_OVERVIEW.md)
- [EC2 Health Check Cheatsheet](../HEALTH_CHECK_CHEATSHEET.md)

---

## Changelog

- **2026-01-12:** REST API refactor for reliable regime detection
  - Switch from streaming-only to REST-primary for historical data
  - BinanceDataProvider now fetches 60-day daily klines via REST API
  - DemoBroker adds REST fallback for quotes when streaming unavailable
  - Changed endpoints from binance.com to binance.us (geo-blocking fix)
  - Streaming remains optional for real-time quotes
  - Fixes issue where 40-day SMA calculation failed due to insufficient streaming buffer data

- **2026-01-08:** Initial deployment with optimal configuration
  - DemoBroker with Binance streaming
  - CSCMDemoAdapter with momentum + regime filter
  - EC2 service and Lambda scheduling
  - Portfolio reset with confirmation
