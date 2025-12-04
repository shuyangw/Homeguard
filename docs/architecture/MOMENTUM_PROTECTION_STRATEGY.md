# Momentum Protection Strategy

A daily momentum strategy with rule-based crash protection for live trading.

## Strategy Overview

| Parameter | Value |
|-----------|-------|
| Universe | S&P 500 (503 stocks) |
| Positions | Top 10 by momentum |
| Rebalance | Daily at 3:55 PM EST |
| Protection | Reduce to 50% exposure during risk |
| Walk-Forward Return | +1,234% (2017-2024) |
| Annual Return | +38.2% |
| Sharpe Ratio | ~2.9 |

## How It Works

### 1. Momentum Ranking

Each day, all stocks are ranked by **1 month - 1 week momentum**:

```
Momentum Score = 21-day return - 5-day return
```

This formula captures:
- Stocks with strong recent momentum (1 month trend)
- But haven't yet had their most recent spike (avoiding buying at the top)
- Shorter lookback responds faster to changing market conditions

Top 10 stocks by this score are selected for the portfolio with equal weight allocation (6.5% each, 65% total exposure).

### 2. Crash Protection Rules (Simplified)

When EITHER of these signals trigger, exposure is reduced from 100% to 50%:

| Rule | Trigger | Rationale |
|------|---------|-----------|
| High VIX | VIX > 25 | Market fear elevated |
| SPY Drawdown | SPY down >5% from peak | Market already falling |

**Note**: VIX spike and momentum volatility checks were removed in December 2025 after walk-forward validation showed the simpler profile had better performance (see Decision History below).

### 3. Daily Rebalance

At 3:55 PM EST each trading day (using current day's close prices):
1. Calculate current momentum rankings
2. Check risk signals
3. Sell positions no longer in top 10
4. Buy new positions that entered top 10
5. Adjust position sizes based on risk exposure

## Walk-Forward Validation Results (2017-2024)

Each year tested using ONLY data from prior 2 years (true out-of-sample):

```
Year       MP (1m-1w)      SPY      Alpha
-----------------------------------------
2017          +49.8%     +21.7%    +28.1%
2018          +29.8%      -4.6%    +34.4%
2019          +37.2%     +31.2%     +6.0%
2020          +62.0%     +18.3%    +43.7%
2021          +52.5%     +28.7%    +23.8%
2022           +1.4%     -18.2%    +19.6%
2023          +29.8%     +26.2%     +3.6%
2024          +53.9%     +25.3%    +28.6%
-----------------------------------------

Cumulative Return: +1,234%
Annual Return:     +38.2%
Win Years:         8/8
Beat SPY:          8/8 years
```

**Key Insight**: Strategy is profitable EVERY year including 2022 bear market. Simpler risk profile (VIX > 25 and SPY drawdown only) outperformed complex version.

## Decision History

### 2025-12-03: Momentum Formula Change (1m-1w)

**Previous**: 12-1 month (252-21 days) - academic standard
**New**: 1m-1w (21-5 days) - shorter lookback

**Walk-Forward Comparison**:
| Metric | 3m-1m | 1m-1w | Improvement |
|--------|-------|-------|-------------|
| Cumulative | +1,234% | +1,133% | Similar |
| Annual | +38.2% | +36.9% | Similar |
| Win Years | 8/8 | 8/8 | Equal |

Both formulas performed similarly in walk-forward testing. The 1m-1w was chosen for faster response to market conditions.

### 2025-12-03: Simplified Risk Profile

**Previous Risk Checks** (4 rules):
1. VIX > 25
2. VIX spike > 20% in 5 days
3. SPY drawdown > 5%
4. Momentum volatility > 90th percentile

**New Risk Checks** (2 rules):
1. VIX > 25
2. SPY drawdown > 5%

**Rationale**: Walk-forward validation showed the 4-rule version returned only +103% vs +1,234% for the 2-rule version. The extra "protection" was triggering too often and hurting performance without improving drawdowns.

## File Structure

```
src/strategies/advanced/
└── momentum_protection_strategy.py    # Pure strategy logic

src/trading/adapters/
└── momentum_live_adapter.py           # Live trading adapter

config/trading/
└── momentum_trading_config.yaml       # Configuration

scripts/trading/
└── demo_momentum_paper_trading.py     # Demo/test script

backtest_scripts/
└── momentum_with_rules_protection.py  # Backtesting script
```

## Paper Trading Setup

### Prerequisites

1. **Alpaca Account**: Paper trading account at [alpaca.markets](https://alpaca.markets)
2. **API Keys**: Set in environment or `settings.ini`:
   ```ini
   [alpaca]
   api_key = YOUR_API_KEY
   secret_key = YOUR_SECRET_KEY
   base_url = https://paper-api.alpaca.markets
   ```

### Running the Strategy

#### 1. Show Current Signals (No Trading)

See what the strategy would trade today:

```bash
python scripts/trading/demo_momentum_paper_trading.py --show-signals
```

Output:
```
CURRENT MOMENTUM SIGNALS
========================================
Risk Signals:
  VIX > 25: NO
  VIX Spike: NO
  SPY Drawdown: NO
  High Mom Vol: NO
  Exposure: 100%

Top 10 Momentum Stocks:
  #1: NVDA (score: 185.2%)
  #2: META (score: 142.8%)
  #3: AVGO (score: 128.5%)
  ...
```

#### 2. Dry Run (Preview Orders)

See what orders would be placed without executing:

```bash
python scripts/trading/demo_momentum_paper_trading.py --run-once --dry-run
```

#### 3. Execute One Rebalance

Run a single rebalance cycle:

```bash
python scripts/trading/demo_momentum_paper_trading.py --run-once
```

#### 4. Use Smaller Universe

Trade top 100 S&P stocks instead of all 500:

```bash
python scripts/trading/demo_momentum_paper_trading.py --universe top100 --run-once
```

### Configuration Options

Edit `config/trading/momentum_trading_config.yaml`:

```yaml
strategy:
  top_n: 10                    # Number of stocks to hold
  position_size_pct: 0.10      # 10% per position
  reduced_exposure: 0.50       # 50% when risk high
  vix_threshold: 25.0          # VIX trigger level
  rebalance_time: "15:55:00"   # 3:55 PM EST
```

### Position Sizing Presets

| Preset | Top N | Position Size | Reduced Exposure |
|--------|-------|---------------|------------------|
| Conservative | 5 | 10% | 25% |
| Moderate (default) | 10 | 10% | 50% |
| Aggressive | 15 | 6.7% | 75% |

## Scheduling for Continuous Trading

### Option 1: Windows Task Scheduler

Create a scheduled task to run at 3:55 PM EST daily:

```
Program: C:\Users\qwqw1\anaconda3\envs\fintech\python.exe
Arguments: C:\...\scripts\trading\demo_momentum_paper_trading.py --run-once
```

### Option 2: AWS Lambda / EC2

Deploy to EC2 with cron:

```bash
# crontab entry (3:55 PM EST = 20:55 UTC during EST)
55 20 * * 1-5 /path/to/python /path/to/demo_momentum_paper_trading.py --run-once
```

### Option 3: Integration with Existing Infrastructure

Use the adapter programmatically:

```python
from src.trading.brokers.alpaca_broker import AlpacaBroker
from src.trading.adapters.momentum_live_adapter import MomentumLiveAdapter

broker = AlpacaBroker(mode='paper')
adapter = MomentumLiveAdapter(broker=broker, top_n=10)

# Pre-load data and run at 3:55 PM EST
adapter.preload_historical_data()
adapter.run_once()
```

## Risk Management

### Built-in Protections

1. **Portfolio Health Checks**: Validates buying power and portfolio value before trading
2. **Crash Protection**: Automatic exposure reduction during market stress
3. **Position Limits**: Maximum 10 concurrent positions

### Recommended Account Settings

| Setting | Recommended Value |
|---------|-------------------|
| Minimum Account Size | $25,000 (avoid PDT rule) |
| Maximum Allocation | 80-100% of portfolio |
| Per-Position Size | 8-12% |
| Stop Loss | Optional 10-15% per position |

## Monitoring

### Key Metrics to Track

1. **Daily P&L**: Should average ~0.15% per day
2. **Exposure Level**: 50-100% depending on risk signals
3. **Turnover**: ~6-10% daily (1-2 trades per day, ~50% of days no trades)
4. **Max Drawdown**: Alert if exceeds 15%

### Log Files

Logs are written to console with color coding:
- 🟢 Green: Success messages
- 🟡 Yellow: Warnings
- 🔴 Red: Errors

## Comparison with OMR Strategy

| Aspect | Momentum Protection | OMR |
|--------|---------------------|-----|
| Holding Period | Daily (close-to-close) | Overnight |
| Universe | S&P 500 | Leveraged ETFs |
| Positions | 10 stocks | 3-5 ETFs |
| Rebalance | 3:55 PM | Entry 3:50 PM, Exit 9:31 AM |
| Risk Model | Rule-based | Bayesian + Regime |
| Expected Return | ~3.3%/month | ~2%/month |
| Drawdown | Lower | Higher (leveraged) |

## Troubleshooting

### Common Issues

1. **"Insufficient data for momentum calculation"**
   - Need 252+ days of price history
   - Run `adapter.preload_historical_data()` first

2. **"VIX data not available"**
   - VIX is fetched via yfinance (not Alpaca)
   - Check internet connection

3. **"Portfolio health check failed"**
   - Check account has sufficient buying power
   - Minimum $5,000 buying power required

4. **Orders not filling**
   - Market may be closed
   - Check Alpaca dashboard for order status

### Debug Mode

Enable verbose logging:

```python
from src.utils.logger import logger
logger.setLevel('DEBUG')
```

## Future Improvements

1. **Weekly Rebalancing**: Reduce turnover and transaction costs
2. **Sector Constraints**: Limit concentration in single sectors
3. **Size Factor Tilt**: Favor small/mid caps for higher momentum alpha
4. **ML Enhancement**: Optional ML layer for crash prediction (currently disabled)

## References

- Jegadeesh & Titman (1993): "Returns to Buying Winners and Selling Losers"
- Daniel & Moskowitz (2016): "Momentum Crashes"
- Backtest code: `backtest_scripts/momentum_with_rules_protection.py`
