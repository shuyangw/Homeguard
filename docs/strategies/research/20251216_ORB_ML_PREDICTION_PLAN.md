# ORB ML Winner Prediction - Design Document

**Date**: 2025-12-16
**Status**: Planning
**Author**: Claude + Human collaboration

---

## Executive Summary

This document outlines a plan to use machine learning to predict which S&P 500 symbols will be profitable with the Opening Range Breakout (ORB) strategy. The goal is to transform ORB from a losing strategy (-0.36% avg return on full S&P 500) to a profitable one by intelligently selecting which symbols to trade.

**Key insight**: ORB works on high-volatility instruments (leveraged ETFs: +108%) but fails on most S&P 500 stocks. Only 49 of 503 symbols (9.7%) were profitable. If we can predict these winners with >40% precision, the strategy becomes viable.

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Data Analysis & Findings](#2-data-analysis--findings)
3. [Noise Analysis](#3-noise-analysis)
4. [Kalman Filter for Noise Reduction](#4-kalman-filter-for-noise-reduction)
5. [Feature Engineering](#5-feature-engineering)
6. [Model Architecture](#6-model-architecture)
7. [Training Strategy](#7-training-strategy)
8. [Evaluation Framework](#8-evaluation-framework)
9. [Implementation Plan](#9-implementation-plan)
10. [Risks & Mitigations](#10-risks--mitigations)
11. [Success Criteria](#11-success-criteria)

---

## 1. Problem Statement

### 1.1 Current State

| Universe | Config | Return | Trades | Win Rate | Verdict |
|----------|--------|--------|--------|----------|---------|
| S&P 500 (503) | Baseline | -0.36% | 4,899 | 47.4% | Losing |
| S&P 500 (503) | Improved | -13.16% | 1,160 | 11.0% | Worse |
| Leveraged ETFs (62) | Baseline | +108.59% | 1,098 | 54.5% | Profitable |
| Leveraged ETFs (62) | Improved | +92.40% | 111 | 46.9% | Profitable |

**Problem**: ORB doesn't work on most stocks but works well on high-volatility instruments.

### 1.2 Hypothesis

Some S&P 500 stocks behave like "natural leverage" - they have sufficient volatility and momentum characteristics for ORB to work. We observed 49 such winners in backtesting. If we can predict which stocks will be winners BEFORE trading, we can:

1. Focus capital on high-probability symbols
2. Avoid the 454 losing/neutral symbols
3. Transform ORB into a profitable stock strategy

### 1.3 Goal

Build an ML model that:
- **Input**: Symbol characteristics (features) at time T
- **Output**: Probability that ORB will be profitable on this symbol in period T+1
- **Target precision**: >40% at top-20 selection (vs 10% baseline)

---

## 2. Data Analysis & Findings

### 2.1 Winner Characteristics

From baseline backtest (2022-2024), 49 profitable symbols with 3+ trades:

**Sector Distribution**:
| Sector | Winners | % of Winners | Avg Return |
|--------|---------|--------------|------------|
| Technology | 20 | 41% | +2.49% |
| Consumer Discretionary | 13 | 27% | +1.78% |
| Financials | 7 | 14% | +0.52% |
| Energy | 4 | 8% | +1.78% |
| Healthcare | 3 | 6% | +3.53% |
| Communication | 2 | 4% | +2.72% |

**Volatility Profile**:
| Volatility Level | Winners | % |
|------------------|---------|---|
| Extreme (COIN) | 1 | 2% |
| Very High | 35 | 71% |
| High | 8 | 16% |
| Medium | 5 | 10% |

**Key Finding**: 90% of winners have "High" or "Very High" volatility.

### 2.2 Top Winners Deep Dive

| Symbol | Return | Trades | Win Rate | Sector | Why It Works |
|--------|--------|--------|----------|--------|--------------|
| CRWD | +7.74% | 21 | 66.7% | Cybersecurity | High vol, momentum, news-driven |
| WBA | +7.16% | 48 | 43.8% | Retail Pharmacy | Turnaround volatility |
| META | +4.85% | 82 | 57.3% | Social Media | Mega-cap with high beta |
| TSLA | +4.21% | 451 | 49.9% | EV | Extreme volatility, trends |
| DELL | +4.03% | 76 | 47.4% | Hardware | AI trade momentum |

### 2.3 Clusters Identified

**Cluster 1: High-Vol Tech** (CRWD, PLTR, SMCI, NVDA, AMD)
- Characteristics: Beta >1.5, intraday range >2%, momentum stocks
- Why ORB works: Large opening ranges, strong breakout follow-through

**Cluster 2: Airlines** (DAL, UAL, LUV)
- Characteristics: High beta to economy, gap-prone
- Why ORB works: News-driven gaps, trend after breakout

**Cluster 3: Cybersecurity** (CRWD, FTNT, PANW)
- Characteristics: Growth sector, news-sensitive
- Why ORB works: Sector momentum, institutional buying on breakouts

**Cluster 4: Mega-Cap Volatility** (META, TSLA, NVDA)
- Characteristics: Huge volume, retail interest, high beta despite size
- Why ORB works: Liquidity for clean execution, consistent OR patterns

---

## 3. Noise Analysis

### 3.1 Types of Noise

| Noise Type | Description | Severity | Mitigation |
|------------|-------------|----------|------------|
| **Label noise** | "Winners" may be lucky, not skilled | Critical | Trade count weighting |
| **Variance noise** | Small sample sizes inflate uncertainty | High | Minimum trade thresholds |
| **Regime noise** | 2022 patterns != 2024 patterns | High | Regime features, retraining |
| **Feature noise** | Most correlations are spurious | Medium | Regularization, feature selection |
| **Survivorship** | Only testing current S&P 500 | Medium | Point-in-time universe |

### 3.2 Signal-to-Noise Ratio

```
Signal: ~2% difference between winners and losers (avg return)
Noise:  ~8% standard deviation per symbol (due to trade variance)

SNR = 2% / 8% = 0.25 (very low)
```

**Implication**: We're trying to detect a weak signal. Need robust methods.

### 3.3 True Winners vs Lucky Winners

Using Bayesian analysis:

| Observed Winners | Estimated True Winners | Lucky Winners |
|------------------|------------------------|---------------|
| 49 | ~4-8 | ~41-45 |

**Key insight**: Most of our 49 "winners" are probably just lucky. Only symbols with high trade counts (TSLA, NVDA, PLTR, META) have statistically reliable signals.

### 3.4 Trade Count Reliability

| Trades | 95% CI on Win Rate | Reliability |
|--------|-------------------|-------------|
| 5 | ±44% | Noise |
| 20 | ±22% | Very noisy |
| 50 | ±14% | Noisy |
| 100 | ±10% | Moderate |
| 500 | ±4% | Reliable |

**Recommendation**: Weight training samples by sqrt(trade_count) or require minimum 30 trades.

---

## 4. Kalman Filter for Noise Reduction

### 4.1 Why Kalman Filter?

The Kalman filter is a recursive algorithm that estimates hidden "true" states from noisy observations. For our problem:

| Concept | Application |
|---------|-------------|
| **True state** | Symbol's actual edge in ORB (unobservable) |
| **Observations** | Individual trade returns (noisy) |
| **Process noise** | Edge changes over time (regime shifts) |
| **Measurement noise** | Per-trade variance (luck) |

**Key insight**: A symbol with 5 trades showing +2% return might have true edge of +0.5% (rest is luck). Kalman filter can estimate this.

### 4.2 Mathematical Framework

**State Model** (true edge evolution):
```
edge_t = edge_{t-1} + process_noise_t
process_noise ~ N(0, Q)
```

**Observation Model** (observed returns):
```
observed_return_t = edge_t + measurement_noise_t
measurement_noise ~ N(0, R)
```

**Key parameters**:
- `Q` (process variance): How much does true edge change between periods? (~0.001 for stable symbols, ~0.01 for volatile)
- `R` (measurement variance): How noisy are individual trades? (~0.01-0.04, estimated from data)

### 4.3 Implementation for ORB Edge Estimation

```python
import numpy as np
from filterpy.kalman import KalmanFilter

def estimate_true_edge(trade_returns: np.ndarray,
                       process_variance: float = 0.001,
                       measurement_variance: float = None) -> dict:
    """
    Estimate true ORB edge from noisy trade returns using Kalman filter.

    Args:
        trade_returns: Array of individual trade returns
        process_variance: Q - how much edge changes between trades
        measurement_variance: R - per-trade noise (estimated if None)

    Returns:
        dict with estimated edge, uncertainty, and soft label
    """
    if len(trade_returns) < 2:
        return {
            'estimated_edge': 0.0,
            'uncertainty': 1.0,
            'soft_label': 0.5,  # Maximum uncertainty
            'reliability': 0.0
        }

    # Estimate measurement variance from data if not provided
    if measurement_variance is None:
        measurement_variance = np.var(trade_returns)

    # Initialize Kalman filter
    kf = KalmanFilter(dim_x=1, dim_z=1)

    # State transition (edge persists)
    kf.F = np.array([[1.0]])

    # Measurement (observe edge directly, plus noise)
    kf.H = np.array([[1.0]])

    # Process noise (edge drift)
    kf.Q = np.array([[process_variance]])

    # Measurement noise (per-trade variance)
    kf.R = np.array([[measurement_variance]])

    # Initial state: prior = 0 edge (conservative)
    kf.x = np.array([[0.0]])
    kf.P = np.array([[0.01]])  # Moderate initial uncertainty

    # Run filter over all trades
    for trade_return in trade_returns:
        kf.predict()
        kf.update(np.array([[trade_return]]))

    estimated_edge = kf.x[0, 0]
    uncertainty = np.sqrt(kf.P[0, 0])

    # Convert to soft label using probability edge > 0
    from scipy.stats import norm
    prob_positive = 1 - norm.cdf(0, loc=estimated_edge, scale=uncertainty)

    # Reliability based on uncertainty reduction
    reliability = max(0, 1 - uncertainty / 0.1)  # 0.1 = initial uncertainty

    return {
        'estimated_edge': estimated_edge,
        'uncertainty': uncertainty,
        'soft_label': prob_positive,
        'reliability': reliability,
        'n_trades': len(trade_returns)
    }
```

### 4.4 Generating Soft Labels for ML Training

Instead of hard labels (winner=1, loser=0), use Kalman-derived soft labels:

```python
def compute_kalman_soft_labels(symbols: list,
                                period: tuple,
                                min_trades: int = 5) -> pd.DataFrame:
    """
    Compute soft labels for all symbols using Kalman filtering.

    Returns DataFrame with:
    - symbol
    - hard_label: 1 if total return > 0, else 0
    - soft_label: Kalman-estimated P(edge > 0)
    - uncertainty: Estimation uncertainty
    - n_trades: Number of trades
    """
    results = []

    for symbol in symbols:
        trades = run_orb_backtest(symbol, period[0], period[1])

        if len(trades) < min_trades:
            continue

        trade_returns = trades['return'].values
        kalman_result = estimate_true_edge(trade_returns)

        results.append({
            'symbol': symbol,
            'total_return': trade_returns.sum(),
            'hard_label': 1 if trade_returns.sum() > 0 else 0,
            'soft_label': kalman_result['soft_label'],
            'estimated_edge': kalman_result['estimated_edge'],
            'uncertainty': kalman_result['uncertainty'],
            'reliability': kalman_result['reliability'],
            'n_trades': len(trades)
        })

    return pd.DataFrame(results)
```

### 4.5 Example: True vs Lucky Winners

| Symbol | Trades | Return | Hard Label | Soft Label | True Winner? |
|--------|--------|--------|------------|------------|--------------|
| TSLA | 451 | +4.21% | 1 | **0.89** | Yes (high confidence) |
| NVDA | 447 | +1.65% | 1 | **0.78** | Yes (high confidence) |
| META | 82 | +4.85% | 1 | **0.82** | Yes (moderate confidence) |
| FTNT | 5 | +4.45% | 1 | **0.58** | Uncertain (too few trades) |
| SCHW | 3 | +0.70% | 1 | **0.52** | Uncertain (too few trades) |
| DAL | 7 | +3.03% | 1 | **0.61** | Uncertain (borderline) |

**Key insight**: Kalman soft labels naturally penalize low-trade-count "winners" - they get labels closer to 0.5.

### 4.6 Using Soft Labels in Training

```python
# Option 1: Use soft labels directly (regression target)
model = LGBMRegressor()
model.fit(X_train, soft_labels, sample_weight=reliability)

# Option 2: Use soft labels for sample weighting
# Hard labels, but weight by confidence
sample_weight = np.abs(soft_labels - 0.5) * 2  # 0 for uncertain, 1 for confident
model = LGBMClassifier()
model.fit(X_train, hard_labels, sample_weight=sample_weight)

# Option 3: Filter out uncertain samples
confident_mask = (soft_labels > 0.7) | (soft_labels < 0.3)
model.fit(X_train[confident_mask], hard_labels[confident_mask])
```

### 4.7 Feature Smoothing with Kalman Filter

Beyond labels, Kalman can smooth noisy features:

```python
def smooth_feature_series(values: np.ndarray,
                          process_variance: float = 0.01,
                          measurement_variance: float = None) -> np.ndarray:
    """
    Smooth a noisy feature time series using Kalman filter.

    Useful for features like:
    - Rolling volatility estimates
    - Win rate estimates
    - Momentum indicators
    """
    if measurement_variance is None:
        measurement_variance = np.var(np.diff(values)) / 2

    kf = KalmanFilter(dim_x=1, dim_z=1)
    kf.F = np.array([[1.0]])
    kf.H = np.array([[1.0]])
    kf.Q = np.array([[process_variance]])
    kf.R = np.array([[measurement_variance]])
    kf.x = np.array([[values[0]]])
    kf.P = np.array([[measurement_variance]])

    smoothed = []
    for value in values:
        kf.predict()
        kf.update(np.array([[value]]))
        smoothed.append(kf.x[0, 0])

    return np.array(smoothed)

# Example: Smooth rolling volatility
raw_vol = df['return'].rolling(20).std()
smoothed_vol = smooth_feature_series(raw_vol.dropna().values)
```

### 4.8 Integration with ML Pipeline

```
Data Pipeline with Kalman Filtering:
==================================

1. Raw Trade Data
   └── Per-symbol ORB trades with returns

2. Kalman Edge Estimation
   └── estimate_true_edge() for each symbol-period
   └── Outputs: estimated_edge, uncertainty, soft_label

3. Feature Generation
   └── Standard features (volatility, momentum, etc.)
   └── Kalman-smoothed features (optional)

4. Label Construction
   └── Use soft_labels instead of hard_labels
   └── Or: Use reliability as sample_weight

5. Model Training
   └── LightGBM with soft targets or weighted samples

6. Prediction
   └── Output: P(winner) for each symbol
   └── Select top-N by predicted probability
```

### 4.9 Expected Impact

| Approach | Estimated Precision@20 | Notes |
|----------|------------------------|-------|
| Hard labels | 30-35% | Overfits to lucky winners |
| Soft labels (reliability weighting) | 35-40% | Downweights noisy samples |
| Kalman soft labels | **40-45%** | Estimates true edge |
| Kalman + feature smoothing | **42-48%** | Reduces feature noise too |

**Key benefit**: Kalman filtering helps the model focus on symbols with **statistically reliable** edges, not just lucky ones.

---

## 5. Feature Engineering

### 5.1 Feature Categories

#### Category 1: Volatility Features (Highest Priority)

These directly measure "ORB-friendliness."

```python
# Opening Range Characteristics
or_width_mean_20d          # Mean OR width (%) over 20 days
or_width_std_20d           # Std dev of OR width
or_width_percentile_75     # 75th percentile OR width
or_breakout_rate_20d       # % of days with OR breakout

# Intraday Volatility
atr_pct_14d                # ATR / price (14-day)
intraday_range_mean_20d    # Mean (high-low)/close
true_range_expansion       # Today's TR vs 20-day avg TR

# Historical Volatility
realized_vol_20d           # 20-day realized volatility (annualized)
realized_vol_60d           # 60-day realized volatility
vol_regime_ratio           # 20d/60d volatility ratio
parkinson_vol_20d          # Parkinson (high-low) volatility estimator

# Gap Behavior
gap_frequency_20d          # % of days with >0.5% gap
gap_magnitude_mean         # Mean absolute gap size
gap_fill_rate              # % of gaps that fill same day
```

#### Category 2: Historical ORB Performance

Past ORB success is the best predictor of future success.

```python
# Rolling ORB Backtest Features (60-day lookback)
orb_return_60d             # Total ORB return last 60 days
orb_win_rate_60d           # Win rate on ORB trades
orb_profit_factor_60d      # Gross profit / gross loss
orb_trade_count_60d        # Number of ORB signals
orb_avg_winner_60d         # Average winning trade size
orb_avg_loser_60d          # Average losing trade size
orb_max_win_60d            # Best single trade
orb_max_loss_60d           # Worst single trade

# Breakout Quality
false_breakout_rate_60d    # % of breakouts that reverse
breakout_follow_thru_60d   # Avg move size after breakout
breakout_speed_60d         # Time to reach target (if hit)
```

#### Category 3: Momentum Features

Winners tend to be momentum stocks.

```python
# Price Momentum
return_1m                  # 1-month return
return_3m                  # 3-month return
return_6m                  # 6-month return
return_12m                 # 12-month return

# Relative Strength
rs_vs_spy_1m               # Return vs SPY (1 month)
rs_vs_sector_1m            # Return vs sector (1 month)
rs_percentile_sp500        # Percentile rank in S&P 500

# Trend Quality
adx_14                     # Trend strength (0-100)
trend_consistency_20d      # % days closing > 10-day MA
ma_slope_20d               # Normalized slope of 20-day MA
higher_highs_rate_20d      # % of days making higher highs
```

#### Category 4: Volume/Liquidity Features

```python
# Volume
avg_dollar_volume_20d      # Average daily dollar volume
volume_trend               # 20d volume / 60d volume
rvol_mean_20d              # Mean relative volume
rvol_above_1_5_rate        # % days with RVOL > 1.5

# Opening Session
open_15min_volume_pct      # First 15 min as % of daily volume
open_volume_trend          # Opening volume trend

# Liquidity
market_cap_log             # Log market cap
bid_ask_spread_mean        # Mean spread (if available)
```

#### Category 5: Fundamental/Categorical

```python
# Sector (categorical - use embedding or one-hot)
gics_sector                # 11 GICS sectors
gics_industry_group        # 24 industry groups

# Style
market_cap_bucket          # Mega/Large/Mid (categorical)
beta_spy_252d              # Beta to SPY

# Events
days_to_earnings           # Days until next earnings
earnings_in_5d             # Binary: earnings within 5 days
```

#### Category 6: Market Regime

```python
# VIX
vix_level                  # Current VIX
vix_percentile_252d        # VIX percentile (1 year)
vix_term_structure         # VIX - VIX3M (contango/backwardation)

# Market Breadth
spy_above_50ma             # Binary: SPY > 50-day MA
pct_sp500_above_20ma       # % of S&P 500 above 20-day MA
advance_decline_ratio      # Market breadth

# Sector Context
sector_rs_1m               # Sector relative strength
sector_momentum_rank       # Sector rank (1-11)
```

### 5.2 Feature Selection Strategy

1. **Start with ~50 features**
2. **Remove highly correlated** (>0.8 correlation) - keep most predictive
3. **Use recursive feature elimination** with cross-validation
4. **Target 15-25 features** for final model
5. **Monitor feature importance stability** across time periods

### 5.3 Point-in-Time Requirements

**CRITICAL**: All features must be computable BEFORE the prediction period.

```python
# WRONG - Lookahead bias
orb_return = compute_orb_return(symbol, period)  # Uses future data

# CORRECT - Point-in-time
features = compute_features(symbol, period_start - 1_day)  # Only past data
target = compute_orb_return(symbol, period)  # Future data for label only
```

---

## 6. Model Architecture

### 6.1 Primary Model: LightGBM

**Why LightGBM**:
- Handles mixed feature types (numeric + categorical)
- Built-in regularization
- Feature importance for interpretability
- Fast training
- Already used successfully in RAMP strategy

```python
model_params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'min_child_samples': 20,
    'reg_alpha': 0.1,           # L1 regularization
    'reg_lambda': 0.1,          # L2 regularization
    'max_depth': 6,             # Limit depth to prevent overfitting
    'n_estimators': 500,
    'early_stopping_rounds': 50,
    'is_unbalance': True,       # Handle 10% positive rate
    'random_state': 42
}
```

### 6.2 Alternative: Two-Stage Model

```
Stage 1: Filter (High Recall)
├── Goal: Don't miss potential winners
├── Model: LightGBM with low threshold
├── Output: ~100-150 candidates (from 500)

Stage 2: Rank (High Precision)
├── Goal: Rank candidates by expected return
├── Model: LightGBM regressor or LambdaRank
├── Output: Top 20-30 symbols to trade
```

### 6.3 Ensemble Option

```python
# Combine multiple weak models
models = {
    'lgbm_binary': LGBMClassifier(**params),
    'lgbm_regression': LGBMRegressor(**params),  # Predict return
    'logistic': LogisticRegression(C=0.1),       # Simple baseline
}

# Weighted average of predictions
final_pred = (
    0.5 * lgbm_binary_pred +
    0.3 * normalize(lgbm_regression_pred) +
    0.2 * logistic_pred
)
```

### 6.4 Alternative Framing: Loser Filtering

Instead of predicting winners (hard), predict losers (easier):

```python
# Easier problem: Identify symbols that will DEFINITELY lose
# These have clearer signal:
# - Low volatility (OR too narrow)
# - Mean-reverting (breakouts fail)
# - Choppy price action (no follow-through)

# Strategy:
# 1. Predict P(loser) for all symbols
# 2. Filter out symbols with P(loser) > 0.7
# 3. Trade ORB on remaining ~100-150 symbols
```

---

## 7. Training Strategy

### 7.1 Data Structure

Each training sample = (symbol, period) pair

```
| symbol | period   | features (X)           | target (y) |
|--------|----------|------------------------|------------|
| AAPL   | 2022-Q1  | [vol, mom, orb_hist...] | 0         |
| AAPL   | 2022-Q2  | [vol, mom, orb_hist...] | 0         |
| CRWD   | 2022-Q1  | [vol, mom, orb_hist...] | 1         |
| CRWD   | 2022-Q2  | [vol, mom, orb_hist...] | 1         |
| ...    | ...      | ...                    | ...        |
```

**Period granularity options**:
| Granularity | Samples | Signal Quality | Recommendation |
|-------------|---------|----------------|----------------|
| Monthly | 18,108 | Noisy | Too noisy |
| Quarterly | 6,036 | Moderate | **Recommended** |
| Semi-annual | 3,018 | Clean | Too few samples |

### 7.2 Target Variable Construction

```python
def compute_target(symbol: str, period: tuple,
                   threshold: float = 0.0,
                   min_trades: int = 1) -> float:
    """
    Compute target label for a symbol-period.

    Returns:
        Soft label between 0 and 1 based on:
        - ORB return during period
        - Number of trades (reliability weighting)
    """
    # Run ORB backtest for this symbol-period
    trades = run_orb_backtest(symbol, period[0], period[1])

    if len(trades) < min_trades:
        return np.nan  # Exclude from training

    total_return = trades['return'].sum()
    n_trades = len(trades)

    # Hard label
    if threshold is not None:
        return 1.0 if total_return > threshold else 0.0

    # Soft label (accounts for uncertainty)
    # Shrink toward 0.5 based on trade count reliability
    reliability = min(1.0, n_trades / 50)  # Full reliability at 50 trades
    raw_label = 1.0 if total_return > 0 else 0.0
    soft_label = reliability * raw_label + (1 - reliability) * 0.5

    return soft_label
```

### 7.3 Walk-Forward Validation

**CRITICAL**: Prevent lookahead bias with proper time-series CV.

```
Timeline:
|--2022 Q1--|--2022 Q2--|--2022 Q3--|--2022 Q4--|--2023 Q1--|--2023 Q2--|--2023 Q3--|--2023 Q4--|--2024 Q1--|--2024 Q2--|

Fold 1: Train[2022 Q1-Q2]  Purge[Q3]  Test[Q4]
Fold 2: Train[2022 Q1-Q4]  Purge[2023 Q1]  Test[Q2]
Fold 3: Train[2022-2023 Q2]  Purge[Q3]  Test[Q4]
Fold 4: Train[2022-2023]  Purge[2024 Q1]  Test[Q2]
```

```python
class WalkForwardCV:
    def __init__(self, n_splits=4, min_train_periods=2,
                 purge_periods=1, test_periods=1):
        self.n_splits = n_splits
        self.min_train_periods = min_train_periods
        self.purge_periods = purge_periods
        self.test_periods = test_periods

    def split(self, X, y, periods):
        unique_periods = sorted(periods.unique())

        for i in range(self.n_splits):
            train_end_idx = self.min_train_periods + i
            test_start_idx = train_end_idx + self.purge_periods
            test_end_idx = test_start_idx + self.test_periods

            if test_end_idx > len(unique_periods):
                break

            train_periods = unique_periods[:train_end_idx]
            test_periods_list = unique_periods[test_start_idx:test_end_idx]

            train_mask = periods.isin(train_periods)
            test_mask = periods.isin(test_periods_list)

            yield np.where(train_mask)[0], np.where(test_mask)[0]
```

### 7.4 Sample Weighting

Weight samples by reliability (trade count):

```python
def compute_sample_weights(df: pd.DataFrame) -> np.ndarray:
    """
    Weight samples by statistical reliability.

    More trades = more reliable label = higher weight
    """
    trade_counts = df['n_trades'].values

    # Square root scaling (diminishing returns)
    weights = np.sqrt(trade_counts)

    # Normalize to mean=1
    weights = weights / weights.mean()

    # Cap extreme weights
    weights = np.clip(weights, 0.1, 5.0)

    return weights
```

### 7.5 Class Imbalance Handling

With only ~10% positive rate:

```python
# Option 1: Built-in (recommended)
model = LGBMClassifier(is_unbalance=True)

# Option 2: Class weights
from sklearn.utils.class_weight import compute_class_weight
weights = compute_class_weight('balanced', classes=[0, 1], y=y_train)
model = LGBMClassifier(class_weight={0: weights[0], 1: weights[1]})

# Option 3: SMOTE (if needed)
from imblearn.over_sampling import SMOTE
X_resampled, y_resampled = SMOTE().fit_resample(X_train, y_train)

# Option 4: Threshold optimization (always do this)
from sklearn.metrics import precision_recall_curve
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
f1 = 2 * precision * recall / (precision + recall + 1e-8)
optimal_threshold = thresholds[np.argmax(f1)]
```

---

## 8. Evaluation Framework

### 8.1 Primary Metrics

| Metric | Target | Why |
|--------|--------|-----|
| **AUC-ROC** | > 0.65 | Overall discrimination |
| **Precision@20** | > 40% | Of top 20, how many win? |
| **Recall@50%P** | > 50% | At 50% precision, catch 50%+ winners |
| **Profit Factor** | > 1.3 | Simulated trading P&L |

### 8.2 Business Metric (Most Important)

```python
def evaluate_trading_simulation(y_true: np.ndarray,
                                 y_pred_proba: np.ndarray,
                                 returns: np.ndarray,
                                 top_n: int = 20) -> dict:
    """
    Simulate trading top-N predicted winners.

    Args:
        y_true: Actual binary labels
        y_pred_proba: Predicted probabilities
        returns: Actual ORB returns per symbol
        top_n: Number of symbols to select

    Returns:
        Trading simulation metrics
    """
    # Select top N by predicted probability
    top_n_idx = np.argsort(y_pred_proba)[-top_n:]

    # Calculate metrics
    selected_returns = returns[top_n_idx]
    baseline_return = returns.mean()  # Trading all symbols

    return {
        'n_selected': top_n,
        'n_actual_winners': y_true[top_n_idx].sum(),
        'precision': y_true[top_n_idx].mean(),
        'selected_avg_return': selected_returns.mean(),
        'baseline_avg_return': baseline_return,
        'improvement': selected_returns.mean() - baseline_return,
        'total_return': selected_returns.sum(),
    }
```

### 8.3 Comparison Benchmarks

| Benchmark | Description | Expected Precision@20 |
|-----------|-------------|----------------------|
| Random | Randomly select 20 symbols | 10% |
| Volatility | Top 20 by volatility | ~15-20% |
| Momentum | Top 20 by 3m momentum | ~15-20% |
| Sector | All tech stocks | ~20% |
| **ML Model** | Our prediction | **>40% (target)** |

### 8.4 Stability Metrics

```python
def evaluate_stability(cv_results: list) -> dict:
    """
    Evaluate model stability across CV folds.
    """
    aucs = [r['auc'] for r in cv_results]
    precisions = [r['precision_at_20'] for r in cv_results]

    return {
        'auc_mean': np.mean(aucs),
        'auc_std': np.std(aucs),
        'auc_min': np.min(aucs),
        'precision_mean': np.mean(precisions),
        'precision_std': np.std(precisions),
        'precision_min': np.min(precisions),
        'is_stable': np.std(aucs) < 0.05,  # <5% std is stable
    }
```

---

## 9. Implementation Plan

### Phase 1: Data Pipeline (Week 1-2)

**Deliverables**:
- Feature generation module
- Target label computation
- Training dataset builder

```
src/ml/orb/
├── __init__.py
├── features.py          # Feature engineering
├── labels.py            # Target label computation
├── dataset.py           # Dataset builder
└── utils.py             # Helpers
```

**Key tasks**:
- [ ] Implement volatility feature generators
- [ ] Implement momentum feature generators
- [ ] Implement historical ORB performance features
- [ ] Build rolling backtest for ORB return computation
- [ ] Create dataset with (symbol, period, features, target)
- [ ] Validate point-in-time correctness (no lookahead)

### Phase 2: Model Development (Week 3-4)

**Deliverables**:
- Training pipeline
- Walk-forward CV implementation
- Model evaluation framework

```
src/ml/orb/
├── model.py             # ORBWinnerPredictor class
├── training.py          # Training pipeline
├── evaluation.py        # Metrics and evaluation
└── cv.py                # Walk-forward cross-validation
```

**Key tasks**:
- [ ] Implement LightGBM classifier wrapper
- [ ] Implement walk-forward CV
- [ ] Implement sample weighting
- [ ] Implement threshold optimization
- [ ] Build evaluation dashboard
- [ ] Run baseline experiments

### Phase 3: Feature Selection & Tuning (Week 5-6)

**Deliverables**:
- Optimized feature set
- Tuned hyperparameters
- Stability analysis

**Key tasks**:
- [ ] Run feature importance analysis
- [ ] Remove correlated/redundant features
- [ ] Hyperparameter tuning with Optuna
- [ ] Stability analysis across time periods
- [ ] Document final feature set

### Phase 4: Integration & Testing (Week 7-8)

**Deliverables**:
- Integrated ORB-ML strategy
- Backtesting results
- Production pipeline

```
src/strategies/advanced/
├── orb_ml_strategy.py   # ML-enhanced ORB strategy

src/ml/orb/
├── predictor.py         # Production predictor
├── monitor.py           # Performance monitoring
└── retrain.py           # Retraining pipeline
```

**Key tasks**:
- [ ] Integrate predictor with ORB strategy
- [ ] Run full backtest on 2024 data (out-of-sample)
- [ ] Compare ML-selected vs baseline performance
- [ ] Build retraining pipeline
- [ ] Document deployment process

---

## 10. Risks & Mitigations

### 10.1 Technical Risks

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| **Overfitting** | Model fails live | High | Walk-forward CV, regularization, simple features |
| **Label noise** | Learning noise not signal | High | Sample weighting, soft labels, trade count thresholds |
| **Regime change** | 2022 model fails in 2025 | Medium | Quarterly retraining, regime features |
| **Feature decay** | Predictive features lose power | Medium | Monitor feature importance, adaptive selection |
| **Data leakage** | Artificially good results | Medium | Strict point-in-time validation |

### 10.2 Business Risks

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| **No alpha** | Model doesn't beat random | Medium | Start with loser filtering (easier problem) |
| **Low signal** | SNR too low for ML | Medium | Focus on noise reduction, ensemble weak signals |
| **Complexity cost** | Maintenance burden | Low | Keep model simple, document thoroughly |

### 10.3 Mitigation Strategies

**For overfitting**:
```python
# 1. Strong regularization
reg_alpha = 0.1
reg_lambda = 0.1
max_depth = 6

# 2. Feature count limit
max_features = 20

# 3. Early stopping
early_stopping_rounds = 50

# 4. Out-of-sample testing
# Always hold out most recent period for final validation
```

**For label noise**:
```python
# 1. Sample weighting by trade count
sample_weight = np.sqrt(n_trades)

# 2. Soft labels
soft_label = reliability * hard_label + (1 - reliability) * 0.5

# 3. Minimum trade threshold
df = df[df['n_trades'] >= 10]  # Require 10+ trades

# 4. Label smoothing
y_smoothed = y * 0.9 + 0.05  # Pull 0->0.05, 1->0.95
```

**For regime change**:
```python
# 1. Regime features
features['vix_regime'] = get_vix_regime(date)
features['market_trend'] = get_market_trend(date)

# 2. Shorter training window (more recent data)
train_window = '2_years'  # Not all history

# 3. Quarterly retraining
retrain_schedule = 'quarterly'
```

---

## 11. Success Criteria

### 11.1 Minimum Viable Model

| Metric | Threshold | Status |
|--------|-----------|--------|
| AUC-ROC | > 0.60 | |
| Precision@20 | > 30% | |
| Stability (AUC std) | < 0.08 | |
| Improvement vs random | > 50% relative | |

### 11.2 Target Model

| Metric | Threshold | Status |
|--------|-----------|--------|
| AUC-ROC | > 0.70 | |
| Precision@20 | > 45% | |
| Stability (AUC std) | < 0.05 | |
| Simulated return | > +1.0% avg | |
| Improvement vs random | > 3x relative | |

### 11.3 Decision Criteria

**Ship if**:
- AUC > 0.65 consistently across CV folds
- Precision@20 > 40%
- Out-of-sample (2024) performance within 80% of CV performance
- Model is interpretable (can explain why symbols are selected)

**Don't ship if**:
- AUC < 0.60 on any CV fold
- Performance degrades >30% on out-of-sample
- Feature importance is unstable across folds
- Model relies on suspicious/spurious features

---

## Appendix A: Feature Computation Code

```python
# Example feature computation (pseudocode)

class ORBFeatureGenerator:
    def __init__(self, lookback_days: int = 60):
        self.lookback_days = lookback_days

    def compute_volatility_features(self,
                                     df: pd.DataFrame,
                                     end_date: str) -> dict:
        """Compute volatility features up to end_date."""
        data = df[df['date'] <= end_date].tail(self.lookback_days)

        return {
            'realized_vol_20d': data['return'].tail(20).std() * np.sqrt(252),
            'atr_pct': (data['atr'] / data['close']).mean(),
            'intraday_range': ((data['high'] - data['low']) / data['close']).mean(),
            'gap_frequency': (data['gap'].abs() > 0.005).mean(),
        }

    def compute_orb_features(self,
                              symbol: str,
                              end_date: str) -> dict:
        """Compute historical ORB performance features."""
        # Run rolling ORB backtest up to end_date
        trades = self.run_orb_backtest(symbol, end_date, lookback=60)

        if len(trades) == 0:
            return {'orb_return_60d': 0, 'orb_win_rate_60d': 0, ...}

        return {
            'orb_return_60d': trades['return'].sum(),
            'orb_win_rate_60d': (trades['return'] > 0).mean(),
            'orb_trade_count_60d': len(trades),
            'orb_profit_factor_60d': self.compute_profit_factor(trades),
        }
```

---

## Appendix B: Alternative Approaches Considered

### B.1 Neural Network

**Rejected because**:
- Not enough data (503 symbols x 12 periods = 6,036 samples)
- Tabular data works better with tree models
- Less interpretable

### B.2 Time Series Model (LSTM)

**Rejected because**:
- We're predicting cross-sectionally (which symbols), not temporally
- Features already capture time-series info (rolling stats)
- Overkill for this problem

### B.3 Reinforcement Learning

**Rejected because**:
- Sample efficiency is terrible
- Much harder to debug
- Not appropriate for this problem structure

---

## Appendix C: References

1. [Advances in Financial Machine Learning](https://www.wiley.com/en-us/Advances+in+Financial+Machine+Learning-p-9781119482086) - Marcos Lopez de Prado
2. [Machine Learning for Asset Managers](https://www.cambridge.org/core/elements/machine-learning-for-asset-managers/6D9211305EA2E425D33A9F38D0AE3545) - Marcos Lopez de Prado
3. [LightGBM Documentation](https://lightgbm.readthedocs.io/)
4. [RAMP Strategy Implementation](../strategies/RAMP_STRATEGY.md) - Internal

---

*Document created: 2025-12-16*
*Last updated: 2025-12-16*
*Status: Planning - Kalman filter section added*
