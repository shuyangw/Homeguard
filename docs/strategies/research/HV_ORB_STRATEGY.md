# HV ORB (High Volatility Opening Range Breakout) Strategy

**Status**: Research/Backtesting
**Last Updated**: 2025-12-14

---

## Overview

The HV ORB (High Volatility Opening Range Breakout) Strategy is an advanced intraday breakout strategy that focuses on "Stocks in Play" - stocks with abnormally high opening volume driven by news catalysts. It combines multiple confirmation filters with tiered exit management for improved risk-adjusted returns.

### Key Features

- **5-minute opening range** (9:30-9:35 AM ET) for faster signals
- **Stocks in Play (SIP) scoring**: Opening volume vs 14-day average
- **Pre-market gap filtering**: 2-15% gaps only
- **ATR-based volatility bounds**: Filters extremes
- **Tiered exits**: Target 1 (50%), Target 2 (remaining), Trailing stop, EOD
- **Daily risk limits**: Max loss %, max trades, max concurrent positions
- **Optional FinBERT sentiment filtering**: News-driven entry confirmation

### Research Foundation

Based on research by Zarattini, Barbon & Aziz (2024):
- General ORB on all stocks **underperforms**
- The edge concentrates in **"Stocks in Play"** with abnormally high volume
- Volume anomaly must be driven by **news catalysts** (not random)

Expected Performance:
- **Filtered ORB on stocks**: 2.81 Sharpe, 36% annualized alpha
- **Realistic post-cost**: 1.0-1.5 Sharpe for systematic implementation

---

## Strategy Logic

### Opening Range Calculation

```
Time Window: 9:30 AM - 9:35 AM ET (first 5 minutes)
OR High:     Max of all highs during window
OR Low:      Min of all lows during window
OR Height:   OR High - OR Low
```

### Pre-Entry Filters

Before scanning for breakouts, these conditions must pass:

| Filter | Condition | Default |
|--------|-----------|---------|
| SIP Score | `sip_score >= sip_min_score` | >= 2.0 |
| Gap Size | `min_gap_pct <= abs(gap) <= max_gap_pct` | 2-15% |
| Volatility | `min_atr_pct <= atr/close <= max_atr_pct` | 2-10% |
| Time | After OR end, before entry cutoff | 9:35 AM - 3:30 PM |

### Entry Conditions

#### Long Entry (All must be true)

1. Time > 9:35 AM ET (OR complete for 5-min setting)
2. Close > OR High (price breakout)
3. SIP Score >= 2.0 (volume confirmation)
4. Gap is positive (2-15%)
5. ATR% in range (2-10%)
6. RVOL > 1.5x (optional volume filter)
7. Sentiment > -0.2 (optional, when enabled)
8. Regime != BEAR (optional regime filter)

#### Short Entry (All must be true)

1. Time > 9:35 AM ET
2. Close < OR Low (price breakdown)
3. SIP Score >= 2.0
4. Gap is negative (2-15%)
5. ATR% in range
6. RVOL > 1.5x (optional)
7. Sentiment < 0.2 (optional, when enabled)
8. Regime != STRONG_BULL (optional)

### Tiered Exit Flow

```
Position Entry
    |
    v
[Check Stop Loss] --> Exit 100% at OR opposite
    |
    v (price moving favorably)
[Target 1 Hit] --> Exit 50%, move stop to breakeven + buffer
    |
    v (price continues)
[Target 2 Hit] --> Exit remaining 50%
    |
    v (alternative after T1)
[Trailing Stop] --> Follow price with offset
    |
    v (fallback)
[3:55 PM] --> Force exit all (no overnight)
```

---

## Parameters

### Opening Range

| Parameter | Default | Description |
|-----------|---------|-------------|
| `opening_range_minutes` | 5 | Minutes for OR (3, 5, 10, 15, 30) |

### SIP Score

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sip_lookback_days` | 14 | Days for volume average |
| `sip_min_score` | 2.0 | Minimum SIP score for entry |

### Gap Filtering

| Parameter | Default | Description |
|-----------|---------|-------------|
| `min_gap_pct` | 0.02 | Minimum gap (2%) |
| `max_gap_pct` | 0.15 | Maximum gap (15%) |
| `gap_direction_filter` | true | Require gap direction match |

### Volatility

| Parameter | Default | Description |
|-----------|---------|-------------|
| `atr_period` | 14 | ATR calculation period |
| `min_atr_pct` | 2.0 | Minimum ATR% |
| `max_atr_pct` | 10.0 | Maximum ATR% |

### Tiered Exits

| Parameter | Default | Description |
|-----------|---------|-------------|
| `target1_multiplier` | 1.0 | T1 = Entry +/- 1x OR Height |
| `target1_exit_pct` | 0.5 | Exit 50% at T1 |
| `target2_multiplier` | 2.0 | T2 = Entry +/- 2x OR Height |
| `use_trailing_stop` | true | Enable trailing after T1 |
| `trailing_offset_pct` | 0.02 | 2% trailing offset |
| `eod_exit_hour` | 15 | EOD exit hour |
| `eod_exit_minute` | 55 | EOD exit minute |

### Risk Management

| Parameter | Default | Description |
|-----------|---------|-------------|
| `daily_max_loss_pct` | 0.03 | Max 3% daily loss |
| `max_daily_trades` | 10 | Max trades per day |
| `max_concurrent_positions` | 5 | Max simultaneous positions |

### Entry Filters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `entry_cutoff_hour` | 15 | Entry cutoff hour |
| `entry_cutoff_minute` | 30 | Entry cutoff minute |
| `rvol_threshold` | 1.5 | RVOL threshold (0 to disable) |

### Sentiment (Optional)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `use_sentiment` | false | Enable FinBERT sentiment |
| `min_sentiment_score` | 0.2 | Sentiment threshold |

### Regime

| Parameter | Default | Description |
|-----------|---------|-------------|
| `use_regime` | true | Enable regime filtering |
| `long_only` | false | Skip short trades |

---

## Sentiment Analysis

### Overview

The HV ORB strategy supports optional sentiment filtering using FinBERT, a financial domain-specific BERT model. Sentiment analysis helps filter entries based on news-driven market expectations.

### How It Works

1. **News Download**: Download news from Alpaca API via `scripts/data/download_news.py`
2. **Sentiment Computation**: Run FinBERT via `scripts/compute_sentiment.py`
3. **Cache Storage**: Results stored in Parquet files for fast loading
4. **Entry Filtering**: Long entries need sentiment >= -0.2, shorts need <= 0.2

### Data Storage

```
{local_storage_dir}/news/
  symbol={SYMBOL}/
    year={YYYY}/
      news.parquet      # Raw news articles
      sentiment.parquet # Pre-computed sentiment scores
```

### Setup

```bash
# 1. Install sentiment dependencies
pip install transformers torch

# 2. Download news for symbols
python scripts/data/download_news.py --symbols TQQQ,SQQQ --start 2024-01-01

# 3. Compute sentiment scores
python scripts/compute_sentiment.py --all

# 4. Run backtest with sentiment
python -m src.backtest_runner --config config/backtesting/hv_orb_sentiment.yaml
```

### Sentiment Filtering Logic

```python
# For longs: allow neutral to positive sentiment
if direction == 'long':
    passes = sentiment_score >= -min_sentiment_score  # >= -0.2

# For shorts: allow neutral to negative sentiment
else:
    passes = sentiment_score <= min_sentiment_score   # <= 0.2
```

---

## Best Instruments

### Primary Universe (3x Leveraged ETFs)

| Symbol | Description | Notes |
|--------|-------------|-------|
| TQQQ | 3x Nasdaq | Highest liquidity |
| SQQQ | 3x Nasdaq Bear | Short-side alternative |
| SOXL | 3x Semiconductor | High volatility |
| SOXS | 3x Semiconductor Bear | |
| UPRO | 3x S&P 500 | |
| SPXU | 3x S&P 500 Bear | |
| TNA | 3x Small Cap | |
| TECL | 3x Technology | |

### Why Leveraged ETFs?

1. **High liquidity** - Clean breakouts, tight spreads
2. **High beta** - Stronger breakout moves
3. **Predictable patterns** - Well-defined opening ranges
4. **No earnings surprises** - Unlike individual stocks
5. **Guaranteed SIP** - Always have volume anomalies on volatility days

---

## Regime Behavior

Uses `MarketRegimeDetector` for adaptive filtering:

| Regime | Long Trades | Short Trades | Notes |
|--------|-------------|--------------|-------|
| STRONG_BULL | Yes | **No** | Skip counter-trend shorts |
| WEAK_BULL | Yes | Yes | Full capability |
| SIDEWAYS | Yes | Yes | Full capability |
| UNPREDICTABLE | Yes | Yes | Consider reducing size |
| BEAR | **No** | Yes | Skip counter-trend longs |

---

## Schedule

| Time (ET) | Action |
|-----------|--------|
| 9:30 AM | Market opens, OR formation begins |
| 9:35 AM | OR complete (5-min setting), entry scanning begins |
| 3:30 PM | Entry cutoff (no new trades) |
| 3:55 PM | Force exit all positions |
| 4:00 PM | Market close |

---

## Usage

### Backtest (Single, No Sentiment)

```bash
python -m src.backtest_runner --config config/backtesting/hv_orb_baseline.yaml
```

### Backtest (With Sentiment)

```bash
# First, download news and compute sentiment
python scripts/data/download_news.py --symbols TQQQ,SQQQ --start 2024-01-01
python scripts/compute_sentiment.py --all

# Then run with sentiment config
python -m src.backtest_runner --config config/backtesting/hv_orb_sentiment.yaml
```

### Walk-Forward Validation

```bash
python -m src.backtest_runner --config config/backtesting/hv_orb_walk_forward.yaml
```

### Programmatic Usage

```python
from src.strategies.advanced.hv_orb_strategy import HVORBStrategy

strategy = HVORBStrategy(
    opening_range_minutes=5,
    sip_min_score=2.0,
    target1_multiplier=1.0,
    target2_multiplier=2.0,
    use_trailing_stop=True,
    use_sentiment=False,  # Set True if sentiment data available
    use_regime=True,
    long_only=False
)

# Get parameters
params = strategy.get_parameters()

# Load sentiment data (if using)
strategy.load_sentiment_for_symbols(
    symbols=['TQQQ', 'SQQQ'],
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 6, 30)
)

# Generate signals (called by backtest engine)
long_e, long_x, short_e, short_x = strategy.generate_long_short_signals(minute_data)
```

---

## Data Requirements

### Market Data

- **Timeframe**: 1-minute bars
- **Required columns**: open, high, low, close, volume
- **Optional columns**: vwap
- **Schema**: Canonical 8-column parquet format
- **Historical**: 14+ days for SIP score calculation

### News Data (Optional)

- **Source**: Alpaca News API
- **Required columns**: timestamp, symbol, headline, summary, source
- **Storage**: Hive-partitioned parquet

### Sentiment Data (Optional)

- **Source**: Pre-computed via FinBERT
- **Required columns**: timestamp, symbol, sentiment_score
- **Storage**: Hive-partitioned parquet alongside news

---

## Risk Management

### Per-Trade Risk

| Metric | Long | Short |
|--------|------|-------|
| Stop-Loss | OR Low | OR High |
| Target 1 | Entry + 1x OR Height | Entry - 1x OR Height |
| Target 2 | Entry + 2x OR Height | Entry - 2x OR Height |
| Partial Exit | 50% at T1 | 50% at T1 |

### Daily Limits

- **Max Loss**: 3% of capital per day
- **Max Trades**: 10 per day
- **Max Positions**: 5 concurrent

### Why Intraday Only?

- Avoids overnight gap risk (leveraged ETFs gap significantly)
- No conflict with OMR strategy (overnight)
- Cleaner risk management with defined stop losses

---

## Configuration Files

| File | Purpose |
|------|---------|
| `config/backtesting/hv_orb_baseline.yaml` | Basic backtest without sentiment |
| `config/backtesting/hv_orb_sentiment.yaml` | Backtest with sentiment enabled |
| `config/backtesting/hv_orb_relaxed.yaml` | Relaxed filters for more signals |
| `config/backtesting/hv_orb_walk_forward.yaml` | Out-of-sample validation |

---

## Troubleshooting

### No Signals Generated

1. **Low SIP Score**: Volume not exceeding 2x threshold
2. **Gap Out of Range**: Gap < 2% or > 15%
3. **ATR Filter**: Volatility too low or too high
4. **Direction Mismatch**: Gap direction doesn't match breakout
5. **Entry Cutoff**: After 3:30 PM - no new entries
6. **Sentiment Filter**: When enabled, negative sentiment blocking longs

### Too Many False Breakouts

1. Increase `sip_min_score` (try 2.5 or 3.0)
2. Increase `rvol_threshold` (try 2.0)
3. Enable `use_sentiment` for news confirmation
4. Narrow gap range (`min_gap_pct=0.03`, `max_gap_pct=0.10`)

### Stops Too Tight

1. Increase `target1_multiplier` (improves R:R)
2. Use 15-minute OR (wider range, more room)
3. Reduce `trailing_offset_pct` (2% -> 1.5%)

### Missing Sentiment Data

1. Verify transformers and torch are installed
2. Check news was downloaded: `ls {storage_dir}/news/symbol=*/year=*/news.parquet`
3. Run sentiment computation: `python scripts/compute_sentiment.py --all`
4. Verify sentiment files exist: `ls {storage_dir}/news/symbol=*/year=*/sentiment.parquet`

---

## Module Structure

```
src/strategies/advanced/
  hv_orb_strategy.py       # Main strategy class
  hv_orb_indicators.py     # SIP score, gap calc, ATR filter

src/backtesting/utils/
  tiered_exit_manager.py   # Tiered exit logic

src/data/news/
  __init__.py              # Module exports
  news_schema.py           # Parquet schema definitions
  news_downloader.py       # Alpaca News API collector
  sentiment_analyzer.py    # FinBERT model wrapper
  sentiment_cache.py       # Pre-computed sentiment storage

scripts/
  download_news.py         # CLI for news download
  compute_sentiment.py     # CLI for sentiment computation
```

---

## Related Documentation

- [ORB Strategy](./ORB_STRATEGY.md) - Basic ORB without HV features
- [Strategy Framework](../../src/strategies/STRATEGY_FRAMEWORK.md)
- [Backtesting Engine](../../src/backtesting/BACKTESTING_ENGINE.md)
- [Data Handling](../../CLAUDE.md#data-handling) - Data requirements

---

## Changelog

- **2025-12-14**: Phase 4 Complete - Full Integration
  - Sentiment integration in entry conditions
  - Sentiment-enhanced confidence scoring
  - SentimentCache integration for backtesting
  - 54 unit tests passing
  - Created hv_orb_sentiment.yaml config

- **2025-12-14**: Phase 3 Complete - Sentiment Analysis
  - FinBERT sentiment analyzer with lazy loading
  - MockSentimentAnalyzer for testing
  - SentimentCache for pre-computed results
  - compute_sentiment.py CLI script
  - 25 sentiment tests passing

- **2025-12-14**: Phase 2 Complete - News Infrastructure
  - Alpaca News API collector
  - Hive-partitioned parquet storage
  - download_news.py CLI script
  - 21 news downloader tests passing

- **2025-12-13**: Phase 1 Complete - Core Strategy
  - HVORBStrategy with SIP score
  - Gap detection and filtering
  - ATR volatility bounds
  - Tiered exit management
  - Daily risk limits
  - Unit tests passing
