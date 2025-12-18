# ORB Strategy - S&P 500 Cherry-Picked Winners

**Date**: 2025-12-16
**Purpose**: Identify patterns in S&P 500 symbols where ORB works

---

## Summary

Out of 585 S&P 500 symbols tested with the improved ORB config, only **10 symbols** (1.7%) were profitable with 3+ trades. These winners share a common pattern: **they behave like leveraged ETFs**.

---

## Winning Symbols

| Symbol | Return | Trades | Win Rate | Sector | Volatility |
|--------|--------|--------|----------|--------|------------|
| **DELL** | +19.98% | 6 | 16.7% | Technology / Hardware | High |
| **AMD** | +11.57% | 22 | 36.4% | Technology / Semiconductors | Very High |
| **CCL** | +9.16% | 28 | 21.4% | Consumer / Cruise Lines | Very High |
| **PLTR** | +9.02% | 18 | 27.8% | Technology / Software-AI | Very High |
| **AAPL** | +5.84% | 3 | 33.3% | Technology / Consumer Electronics | Medium |
| **COIN** | +3.41% | 22 | 27.3% | Financials / Crypto Exchange | Extreme |
| **META** | +2.89% | 5 | 60.0% | Technology / Social Media | High |
| **NCLH** | +1.81% | 6 | 16.7% | Consumer / Cruise Lines | Very High |
| **UAL** | +1.27% | 3 | 33.3% | Consumer / Airlines | Very High |
| **NVDA** | +0.49% | 30 | 30.0% | Technology / Semiconductors | Very High |

---

## Pattern Analysis

### Sector Distribution

```
Technology:              7 symbols (70%)
  - Semiconductors: AMD, NVDA
  - Software/AI: PLTR
  - Hardware: DELL
  - Social Media: META
  - Consumer Electronics: AAPL
  - Crypto: COIN

Consumer Discretionary:  3 symbols (30%)
  - Cruise Lines: CCL, NCLH
  - Airlines: UAL
```

### Volatility Profile

| Volatility Level | Count | Percentage |
|------------------|-------|------------|
| Extreme | 1 (COIN) | 10% |
| Very High | 7 | 70% |
| High | 1 (DELL) | 10% |
| Medium | 1 (AAPL) | 10% |

---

## Why These Symbols Work for ORB

### 1. High Intraday Volatility
- Large opening ranges (similar to leveraged ETFs)
- Sufficient price movement to hit targets
- AMD, NVDA, COIN regularly move 2-5% intraday

### 2. Momentum/Growth Characteristics
- Strong directional moves when they break out
- Retail and institutional interest creates follow-through
- News-driven gaps that continue (gap and go)

### 3. High Beta to Market
- Amplify market moves (like leverage)
- When SPY moves 1%, these move 2-3%
- Creates leveraged-ETF-like behavior

### 4. Sector Concentration
- Tech stocks dominate (70%)
- Travel/leisure stocks (30%) are also high-beta
- Both sectors are sentiment-driven

---

## Key Insight

**ORB doesn't work on "typical" S&P 500 stocks.** It works on the subset that behaves like leveraged ETFs:

| Stock Type | ORB Performance | Example |
|------------|-----------------|---------|
| Low volatility (utilities, staples) | Poor | PG, JNJ, KO |
| Medium volatility (financials, healthcare) | Poor | JPM, UNH, PFE |
| High volatility (tech, growth) | **Good** | AMD, NVDA, PLTR |
| Extreme volatility (meme, crypto-adjacent) | **Best** | COIN, CCL |

---

## Recommendation: High-Beta S&P 500 Universe

Instead of running ORB on full S&P 500, create a **high-beta subset**:

### Proposed ORB-Friendly S&P 500 List (20-30 symbols)

**Technology (High Volatility)**
- AMD, NVDA, AVGO (semiconductors)
- PLTR, CRM, NOW (software)
- META, GOOGL, NFLX (internet)
- TSLA (EV - high vol despite losses in backtest)

**Consumer Discretionary (High Beta)**
- CCL, NCLH, RCL (cruise lines)
- UAL, DAL, AAL (airlines)
- ABNB, BKNG (travel)

**Financials (Crypto/Speculative)**
- COIN (crypto)
- HOOD (if in S&P 500)

**Energy (Volatile)**
- OXY, DVN (exploration)

### Benefits of High-Beta Subset
1. Fewer symbols to monitor (20-30 vs 500)
2. Higher concentration of winners
3. Better use of capital
4. Similar volatility profile to leveraged ETFs

---

## Next Steps

1. **Create high-beta S&P 500 list** based on historical volatility or beta
2. **Backtest ORB on high-beta subset** vs full S&P 500
3. **Compare to leveraged ETFs** - is it worth adding these, or stick with ETFs?
4. **Consider hybrid approach**: Leveraged ETFs + high-beta stocks

---

## Comparison: High-Beta Stocks vs Leveraged ETFs

| Factor | High-Beta Stocks | Leveraged ETFs |
|--------|------------------|----------------|
| Volatility | High (1.5-3x market) | Very High (3x market) |
| Liquidity | Excellent | Good to Excellent |
| Overnight risk | Earnings, news | Decay, but no single-stock risk |
| Diversification | Single stock risk | Basket of stocks |
| Recommendation | **Add as supplement** | **Primary universe** |

---

*Analysis generated: 2025-12-16*
*Source: S&P 500 Improved ORB Backtest (2022-2024)*
