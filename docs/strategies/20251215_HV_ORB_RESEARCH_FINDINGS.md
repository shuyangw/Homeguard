# HV ORB Strategy Research Findings

**Date**: 2025-12-15
**Status**: Shelved - Needs further research
**Backtest Period**: 2024-01-01 to 2024-06-30
**Universe**: 100 S&P 500 symbols with good data coverage

## Strategy Overview

High Volatility Opening Range Breakout (HV ORB) strategy that:
- Identifies stocks gapping up with high volatility (SIP score)
- Waits for opening range formation (first 5 minutes)
- Enters on breakout above OR high
- Uses tiered profit targets and trailing stops

## Key Parameters Tested

```yaml
opening_range_minutes: 5
sip_lookback_days: 14
sip_min_score: 1.5
min_gap_pct: 0.02
max_gap_pct: 0.15
target1_multiplier: 1.0
target2_multiplier: 2.0
use_trailing_stop: true
trailing_offset_pct: 0.02
eod_exit_hour: 15
eod_exit_minute: 55
long_only: true
```

---

## Experiment 1: Pullback Entry vs Immediate Entry

### Hypothesis
Waiting for a pullback to the OR high before entering should improve win rate by confirming support.

### Pullback Entry Conditions
1. `touched_pullback`: Low touches within 2% of OR high
2. `is_bouncing`: Close > Open (green candle)
3. `still_above_or`: Close > OR high

### Results

| Entry Type | Return | Win Rate | Trades | Avg Win | Avg Loss |
|------------|--------|----------|--------|---------|----------|
| Immediate | -6.48% | 52.94% | 85 | - | - |
| Pullback | -7.76% | 60.53% | 76 | - | - |

### Findings
- Pullback entry **improves win rate** by ~8 percentage points
- But **worse total return** (-7.76% vs -6.48%)
- Fewer trades taken (76 vs 85)
- Problem: Catches "dead cat bounces" - failed breakouts that briefly retrace

---

## Experiment 2: Time-Based Stop

### Hypothesis
Exit if trade is not profitable after N minutes to cut losses early.

### Results

| Time Stop | Return | Win Rate | Trades |
|-----------|--------|----------|--------|
| None | -7.76% | 60.53% | 76 |
| 30 min | -11.38% | 29.27% | 82 |
| 60 min | -11.04% | 35.00% | 80 |

### Findings
- Time-based stops **significantly worsen** results
- Cuts winners short more than it saves on losers
- Winners need time to develop; premature exits hurt profitability
- **Conclusion**: Do not use time-based stops

---

## Experiment 3: Sentiment Analysis

### Hypothesis
Stocks gapping up with negative sentiment (divergence) should be avoided.

### News Timing Analysis
- 30.2% of news published premarket (4:00-9:30 AM)
- 54.5% during market hours
- ~35% available before ORB entry (9:35 AM)
- Premarket news can be analyzed without speed concerns

### Sentiment-Gap Divergence Test

Analyzed sentiment for 7 biggest losing trades:

| Trade | Loss | Sentiment | Divergence? |
|-------|------|-----------|-------------|
| COIN 2024-01-11 | -$2,410 | +0.152 (POSITIVE) | No |
| AMD 2024-05-23 | -$1,322 | -0.106 (NEGATIVE) | **YES** |
| NVDA 2024-06-20 | -$1,185 | No data | - |
| COIN 2024-02-16 | -$1,172 | +0.159 (POSITIVE) | No |
| DAL 2024-04-10 | -$1,069 | +0.430 (POSITIVE) | No |
| DELL 2024-05-23 | -$992 | +0.862 (POSITIVE) | No |
| TSLA 2024-06-13 | -$720 | No data | - |

### Findings
- Only **1 of 5** trades with sentiment data showed divergence
- Divergence filter would catch ~19% of losses (~$1,322 of $6,965)
- Most losing trades had **positive sentiment aligned with gap**
- **Conclusion**: Sentiment divergence is NOT the primary cause of losses

---

## Root Cause Analysis: Why Trades Fail

### Pattern Identified: "Dead Cat Bounce"

Losing trades follow this pattern:
1. Stock gaps up on positive news
2. OR breakout occurs
3. Price pulls back to OR high (triggers pullback entry)
4. Price briefly bounces (appears to confirm support)
5. Price then drops significantly, held until EOD

### Example: COIN 2024-01-11 (-$2,410)
- Bitcoin ETF approval news (positive sentiment)
- Large gap up, OR breakout
- Pullback entry triggered at $160.56
- Price dropped 3% within 8 minutes after entry
- Held until EOD, exited at $141.13

### Key Insight
The pullback entry catches **failing breakouts** that look like valid retracements. The bounce is real but temporary - momentum has already shifted bearish.

---

## Summary of Findings

| Enhancement | Impact | Recommendation |
|-------------|--------|----------------|
| Pullback Entry | +8% win rate, worse return | Needs refinement |
| Time Stop (30 min) | Much worse (-11.38%) | Do not use |
| Time Stop (60 min) | Worse (-11.04%) | Do not use |
| Sentiment Divergence | Marginal (~19% of losses) | Low priority |

---

## Future Research Directions

1. **Volume Confirmation**: Require increasing volume on pullback bounce
2. **Momentum Filters**: Check if momentum is still positive at pullback
3. **Stricter Pullback**: Require deeper pullback or longer consolidation
4. **Early Exit on Failed Bounce**: If price breaks below OR high after entry, exit immediately
5. **Regime-Based Entry**: Only take pullback entries in strong bull regimes
6. **Sector Correlation**: Avoid entries when sector is weak despite individual stock strength

---

## Files Modified

- `src/strategies/advanced/hv_orb_strategy.py` - Added time-based stop (disabled)
- `config/backtesting/hv_orb_100_symbols.yaml` - Test configuration

## Debug Scripts Created

- `scripts/debug/analyze_losing_trade.py` - Minute-by-minute price analysis
- `scripts/debug/analyze_losing_trade_sentiment.py` - Sentiment divergence analysis
- `scripts/debug/analyze_news_timing.py` - News publication timing analysis

---

## Conclusion

The HV ORB strategy with pullback entry shows improved win rate but overall negative returns in H1 2024. The main issue is catching "dead cat bounces" - failed breakouts that briefly retrace before continuing lower.

Neither time-based stops nor sentiment divergence filtering address the core problem. Future work should focus on better distinguishing genuine pullbacks from failing breakouts, potentially using volume or momentum confirmation.

**Strategy Status**: Shelved pending further research on pullback quality indicators.
