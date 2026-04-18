# LightGBM Momentum Strategy - Implementation Status

**Date:** 2025-11-29
**Status:** Phase 3 Complete - Model Trained on ETF Universe

## Overview

The LightGBM Momentum Strategy uses gradient boosting to predict cross-sectional momentum winners. Unlike traditional momentum strategies that use fixed lookback periods, this ML-based approach learns optimal feature combinations from historical data.

## Architecture

### Key Design Decisions

1. **Cross-Sectional Approach**: One model for all symbols (not per-symbol models)
   - Learns relative performance patterns across the universe
   - Target: Binary classification - "will this symbol outperform the median?"
   - More robust with limited data per symbol

2. **Data Infrastructure**: Uses existing Alpaca parquet storage
   - Minute data resampled to daily OHLCV
   - Data source: `F:\Stock_Data\equities_1min`
   - No external API calls during training (uses cached parquet files)

3. **Feature Engineering**: 65 features across multiple categories
   - Momentum: 5d, 21d, 63d, 126d returns and ROC
   - Volatility: Parkinson, Garman-Klass, realized volatility
   - Technical: RSI, Bollinger bands, SMA distances
   - Cross-sectional: Ranks and z-scores across symbols

## Implementation Files

### Core Components

| File | Purpose |
|------|---------|
| `src/strategies/advanced/lightgbm_momentum_model.py` | LightGBM model wrapper with train/predict/save/load |
| `src/strategies/advanced/lightgbm_momentum_strategy.py` | Strategy class for backtesting integration |
| `src/strategies/advanced/momentum_features.py` | Feature engineering engine (65 features) |
| `scripts/trading/train_lgbm_momentum.py` | Training script using parquet data |
| `config/backtesting/lgbm_walk_forward.yaml` | Walk-forward validation config |

### Universe Definitions

| Universe | Symbols | Description |
|----------|---------|-------------|
| Core ETFs | 28 | Major sector ETFs, bonds, commodities |
| Leveraged | 12 | 3x leveraged ETFs |
| All | 40 | Combined universe |

## Training Results (2025-11-29)

### Configuration
- **Universe**: Core ETFs (25/28 symbols loaded)
- **Training Period**: 5 years (2020-11-30 to 2025-11-29)
- **Holdout Period**: 63 days (last ~3 months)
- **Target**: 5-day forward return, outperform median

### Data Summary
| Metric | Value |
|--------|-------|
| Symbols Successfully Loaded | 25 |
| Failed Symbols | XLRE, EEM, XLC (no parquet data) |
| Training Samples | 29,372 |
| Validation Samples | 1,491 |
| Features | 65 |
| Target Distribution | 48.1% positive |

### Model Performance
| Metric | Value | Interpretation |
|--------|-------|----------------|
| Validation AUC | 0.5680 | Modest predictive power |
| Validation Accuracy | 53.05% | Above random (50%) |
| Precision | 56.53% | When predicting winners |
| Recall | 25.03% | Conservative predictions |

### Top 15 Feature Importance

| Rank | Feature | Importance | Description |
|------|---------|------------|-------------|
| 1 | `dollar_vol_20d` | 33.2% | 20-day dollar volume |
| 2 | `vol_21d_rank` | 8.4% | Volatility cross-sectional rank |
| 3 | `ret_5d` | 7.6% | 5-day return |
| 4 | `roc_5d` | 4.2% | 5-day rate of change |
| 5 | `rel_vol_5d` | 4.0% | 5-day relative volume |
| 6 | `ret_126d_zscore` | 3.7% | 6-month return z-score |
| 7 | `mom_63d_zscore` | 2.9% | 3-month momentum z-score |
| 8 | `vol_21d_zscore` | 2.6% | Volatility z-score |
| 9 | `ret_21d_zscore` | 2.6% | 1-month return z-score |
| 10 | `gk_vol` | 2.5% | Garman-Klass volatility |
| 11 | `parkinson_vol` | 2.5% | Parkinson volatility |
| 12 | `vol_63d` | 2.0% | 3-month volatility |
| 13 | `mom_63d_rank` | 1.8% | Momentum rank |
| 14 | `dist_sma10` | 1.6% | Distance from 10-day SMA |
| 15 | `bb_width` | 1.6% | Bollinger band width |

### Model Artifacts
- **Timestamped**: `models/lightgbm_momentum_20251129_004124.joblib`
- **Latest symlink**: `models/lightgbm_momentum_latest.joblib`

## Usage

### Training
```bash
# Train on core ETFs (recommended)
python scripts/trading/train_lgbm_momentum.py --universe core --years 5

# Train on all symbols
python scripts/trading/train_lgbm_momentum.py --universe all --years 10

# Dry run (show what would be trained)
python scripts/trading/train_lgbm_momentum.py --dry-run
```

### Backtesting
```bash
# Run walk-forward validation
python -m src.backtest_runner --config config/backtesting/lgbm_walk_forward.yaml
```

### Strategy Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `long_threshold` | 0.55 | Probability threshold for long signals |
| `max_positions` | 10 | Maximum concurrent positions |
| `holding_period` | 5 | Days to hold each position |
| `position_method` | "equal" | Position sizing method |
| `rebalance_frequency` | "weekly" | How often to rebalance |

## Interpretation

### Why AUC ~0.57?

The AUC of 0.57 indicates modest but real predictive power:

1. **Cross-sectional prediction is hard**: Predicting relative winners among correlated ETFs is challenging
2. **Regime changes**: Model trained on 5 years includes multiple market regimes
3. **Feature overlap**: Many features capture similar information (momentum at different horizons)
4. **Conservative target**: Binary "beat median" is harder than directional prediction

### Key Insights from Feature Importance

1. **Liquidity matters most**: `dollar_vol_20d` at 33% suggests liquidity is the strongest predictor
2. **Volatility is informative**: Multiple volatility features in top 15
3. **Cross-sectional features work**: Ranks and z-scores add value over raw features
4. **Short-term momentum**: 5-day features outperform longer horizons

## Next Steps

### Phase 4: Walk-Forward Validation
- [ ] Run full walk-forward backtest with 24-month train / 6-month test windows
- [ ] Analyze performance across market regimes
- [ ] Calculate strategy metrics (Sharpe, max drawdown, Calmar)

### Phase 5: Live Integration
- [ ] Add to live trading signal generation
- [ ] Implement position sizing based on prediction confidence
- [ ] Add model retraining schedule

### Potential Improvements
1. **Feature engineering**: Add VIX-based features, sector rotation signals
2. **Hyperparameter tuning**: Grid search over LightGBM parameters
3. **Ensemble methods**: Combine with other models
4. **Universe expansion**: Include individual stocks after ETF validation

## Registry Integration

The strategy is registered in `src/strategies/registry.py`:

```python
"LightGBMMomentumStrategy": ("src.strategies.advanced.lightgbm_momentum_strategy", "LightGBMMomentumStrategy"),
"LightGBMMomentum": ("src.strategies.advanced.lightgbm_momentum_strategy", "LightGBMMomentumStrategy"),
```

Display name aliases:
- "LightGBM Momentum"
- "LightGBM Momentum Strategy"
- "LGBM Momentum"

## Related Documentation

- Strategy Registry: `src/strategies/registry.py`
- Universe Definitions: `src/strategies/universe/momentum_universe.py`
- Backtesting Guide: `.claude/backtesting.md`
