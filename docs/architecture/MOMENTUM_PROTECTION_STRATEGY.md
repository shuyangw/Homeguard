# Momentum Protection Strategy

A daily momentum strategy with rule-based crash protection for live trading.

## Strategy Overview

| Parameter | Value |
|-----------|-------|
| Universe | S&P 500 (503 stocks) |
| Positions | Top 10 by momentum |
| Rebalance | Daily at 9:31 AM EST |
| Protection | Reduce to 50% exposure during risk |
| Backtest Return | +2,309% (2017-2024) |
| Monthly Return | ~3.3% |
| Sharpe Ratio | 2.28 |

## How It Works

### 1. Momentum Ranking

Each day, all stocks are ranked by **12-1 month momentum**:

```
Momentum Score = 12-month return - 1-month return
```

- Skipping the most recent month avoids short-term reversal effects
- Top 10 stocks by this score are selected for the portfolio
- Equal weight allocation (10% each)

### 2. Crash Protection Rules

When ANY of these signals trigger, exposure is reduced from 100% to 50%:

| Rule | Trigger | Rationale |
|------|---------|-----------|
| High VIX | VIX > 25 | Market fear elevated |
| VIX Spike | VIX up >20% in 5 days | Sudden fear increase |
| SPY Drawdown | SPY down >5% from peak | Market already falling |
| Momentum Volatility | Mom vol > 90th percentile | Factor getting choppy |

### 3. Daily Rebalance

At 9:31 AM EST each trading day (based on prior day's close):
1. Calculate current momentum rankings
2. Check risk signals
3. Sell positions no longer in top 10
4. Buy new positions that entered top 10
5. Adjust position sizes based on risk exposure

## Backtest Results (2017-2024)

```
Year       Base Ret     Prot Ret    Base DD    Prot DD  Avg Exposure
----------------------------------------------------------------------
2017          38.9%        44.9%     -11.4%      -8.5%        93.0%
2018          15.1%        43.4%     -31.0%     -17.7%        74.7%
2019          75.3%        67.9%     -17.1%      -9.9%        81.3%
2020         114.1%        93.7%     -41.1%     -22.8%        64.6%
2021          18.3%        40.6%     -21.1%     -17.1%        93.3%
2022           9.8%         6.5%     -25.0%     -13.2%        52.6%
2023          38.2%        29.3%     -15.8%      -9.5%        64.2%
2024          86.0%        84.2%     -26.7%     -18.0%        81.6%
----------------------------------------------------------------------

Cumulative Baseline:  +1,903%
Cumulative Protected: +2,309%

Avg Baseline Sharpe:  1.61
Avg Protected Sharpe: 2.28
```

**Key Insight**: Protection improves returns AND reduces drawdowns. The strategy avoids the worst momentum crashes (2018, 2020, 2022).

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
  rebalance_time: "09:31:00"   # 9:31 AM EST
```

### Position Sizing Presets

| Preset | Top N | Position Size | Reduced Exposure |
|--------|-------|---------------|------------------|
| Conservative | 5 | 10% | 25% |
| Moderate (default) | 10 | 10% | 50% |
| Aggressive | 15 | 6.7% | 75% |

## Scheduling for Continuous Trading

### Option 1: Windows Task Scheduler

Create a scheduled task to run at 9:31 AM EST daily:

```
Program: C:\Users\qwqw1\anaconda3\envs\fintech\python.exe
Arguments: C:\...\scripts\trading\demo_momentum_paper_trading.py --run-once
```

### Option 2: AWS Lambda / EC2

Deploy to EC2 with cron:

```bash
# crontab entry (9:31 AM EST = 14:31 UTC during EST)
31 14 * * 1-5 /path/to/python /path/to/demo_momentum_paper_trading.py --run-once
```

### Option 3: Integration with Existing Infrastructure

Use the adapter programmatically:

```python
from src.trading.brokers.alpaca_broker import AlpacaBroker
from src.trading.adapters.momentum_live_adapter import MomentumLiveAdapter

broker = AlpacaBroker(mode='paper')
adapter = MomentumLiveAdapter(broker=broker, top_n=10)

# Pre-load data and run at 9:31 AM EST
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
| Holding Period | Daily | Overnight |
| Universe | S&P 500 | Leveraged ETFs |
| Positions | 10 stocks | 3-5 ETFs |
| Rebalance | 9:31 AM | Entry 3:50 PM, Exit 9:31 AM |
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
