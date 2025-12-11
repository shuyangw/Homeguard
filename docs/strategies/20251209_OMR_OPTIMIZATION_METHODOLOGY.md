# OMR Parameter Optimization Methodology

**Date**: 2025-12-09
**Strategy**: Overnight Mean Reversion (OMR)
**Universe**: 62 Leveraged ETFs
**Test Period**: 2020-01-01 to 2024-12-31 (5-year in-sample)
**Validation Period**: 2017-01-01 to 2019-12-31 (3-year out-of-sample)

---

## Baseline Configuration

**Current Production Parameters** (before optimization):

| Parameter | Value | Notes |
|-----------|-------|-------|
| Stop loss | -1.5% | Minute-level checking |
| Position size | 20% | 0.20 per trade |
| Max positions | 5 | Concurrent positions |
| Min win rate | 0.52 | 52% minimum |
| Min expected return | 0.3% | 0.003 minimum |
| VIX threshold | 45 | Entry filter |
| Min sample size | 5 | Bayesian model requirement |
| Skip bear | False | Trade in all regimes |
| Sorting | Expected return | Order trades by this metric |

**Baseline Performance Metrics**:

| Metric | Value |
|--------|-------|
| CAGR | 37.7% |
| Sharpe Ratio | 1.53 |
| Max Drawdown | -12.0% |
| Win Rate | 47.5% |
| Total Trades | 3,218 |
| Stop Loss Triggered | 39.0% |

---

## Optimization Goals

**Primary Objective**: Maximize Sharpe ratio (risk-adjusted returns)

**Secondary Objectives**:
1. Minimize maximum drawdown
2. Maintain CAGR above 35%
3. Ensure parameter robustness (stable across time periods)

**Success Criteria**:
- Sharpe ratio > 1.53 (improvement over baseline)
- CAGR > 35% (maintain profitability)
- Max DD < -15% (acceptable risk)
- Parameter sensitivity < 0.2 (stability)

---

## Parameters Under Optimization

### 1. Stop Loss Threshold
**Range**: -1.0% to -2.0%
**Test Values**: [-1.0%, -1.25%, -1.5%, -1.75%, -2.0%]
**Rationale**:
- Tighter stops (larger magnitude, e.g., -1.0%) reduce catastrophic losses but exit winners early
- Looser stops (smaller magnitude, e.g., -2.0%) allow more recovery but increase drawdown
- Current 39% stop loss triggering suggests room for optimization

**Expected Impact**: HIGH
Hypothesis: Optimal around -1.5% to -1.75% (current is -1.5%)

### 2. Position Size
**Range**: 15% to 25%
**Test Values**: [15%, 17.5%, 20%, 22.5%, 25%]
**Rationale**:
- Larger sizes (25%) amplify returns but increase portfolio volatility and drawdown
- Smaller sizes (15%) reduce risk but lower absolute returns
- Current 20% is moderate; tests explore trade-off

**Expected Impact**: MEDIUM-HIGH
Hypothesis: Sweet spot around 17.5%-20% (smaller positions = lower drawdown risk)

### 3. Max Positions
**Range**: 3 to 7
**Test Values**: [3, 4, 5, 6, 7]
**Rationale**:
- More positions (7) = better diversification, lower vol, but execution complexity
- Fewer positions (3) = concentrated bets, higher returns but higher drawdown
- Current 5 is moderate; tests whether more diversification improves Sharpe

**Expected Impact**: MEDIUM
Hypothesis: Sharpe ratio improves with more positions (up to 6-7) due to diversification

### 4. Min Expected Return
**Range**: 0.2% to 0.5%
**Test Values**: [0.2%, 0.25%, 0.3%, 0.35%, 0.4%, 0.5%]
**Rationale**:
- Higher thresholds (0.5%) select only highest-conviction trades, fewer trades, higher quality
- Lower thresholds (0.2%) include marginal trades, more trades, potentially higher noise
- Current 0.3% is moderate; tests signal quality trade-off

**Expected Impact**: LOW-MEDIUM
Hypothesis: Higher threshold (0.35%-0.5%) may improve Sharpe by filtering low-quality signals

### 5. VIX Threshold
**Range**: 35 to 50
**Test Values**: [35, 40, 45, 50]
**Rationale**:
- Lower thresholds (35) = more selective, avoid high-vol periods
- Higher thresholds (50) = trade more, include high-vol periods
- Current 45 is moderate; tests whether extreme vol is problematic

**Expected Impact**: LOW-MEDIUM
Hypothesis: 40-45 optimal (avoid extremes without over-filtering)

### 6. Min Win Rate
**Range**: 50% to 54%
**Test Values**: [0.50, 0.51, 0.52, 0.53, 0.54]
**Rationale**:
- Higher thresholds (54%) select only very high-probability trades
- Lower thresholds (50%) include marginal trades
- Current 52% filters on Bayesian model quality; tests how stringent to be

**Expected Impact**: LOW
Hypothesis: Current 52% is near-optimal; higher thresholds reduce trades without much gain

---

## Optimization Strategy

### Phase 1: Parameter Sensitivity Analysis (CURRENT)

**Approach**: Test each parameter independently while holding others at baseline

**Execution**:
1. `omr_parameter_focus_optimizer.py` - Sequential parameter testing
2. Test approximately 30-35 configurations total
3. Identify which parameters have largest impact on Sharpe
4. Estimate runtime: 1-2 hours

**Output**:
- Sensitivity curves for each parameter
- Prioritization of which parameters to optimize further
- Initial insights on parameter interactions

### Phase 2: Focused Grid Search

**Approach**: Combination search on most impactful parameters

**Execution**:
1. Focus on 2-3 most impactful parameters from Phase 1
2. `omr_grid_search_optimizer.py --focus-*` commands
3. Test promising combinations (5-15 per focus)
4. Estimate runtime: 2-4 hours

**Output**:
- Top 5 parameter combinations by Sharpe
- Trade-off analysis (Sharpe vs CAGR vs Drawdown)
- Robustness assessment (parameter stability)

### Phase 3: Full Grid Search (Optional)

**Approach**: Comprehensive testing across all parameters

**Execution**:
1. `omr_grid_search_optimizer.py` (full grid)
2. Test 7,500 combinations if justified by Phase 1/2 results
3. Estimate runtime: 6-12 hours

**Output**:
- Comprehensive results across all parameter space
- Global optima identification
- Complete parameter sensitivity matrix

### Phase 4: Out-of-Sample Validation

**Approach**: Validate top configurations on 2017-2019 data

**Execution**:
1. `omr_validate_best_configs.py` --configs <top_5.json>
2. Test top 3-5 configurations from Phase 2
3. Compare in-sample vs out-of-sample performance
4. Estimate runtime: 30-60 minutes

**Output**:
- Out-of-sample Sharpe ratios for top configs
- Evidence of overfitting (if any)
- Validation metrics by period

---

## Expected Runtime Summary

| Phase | Duration | Status |
|-------|----------|--------|
| Phase 1: Sensitivity | 1-2 hours | Running (started 3:27 AM) |
| Phase 2: Focused Search | 2-4 hours | Pending |
| Phase 3: Full Grid (optional) | 6-12 hours | Pending |
| Phase 4: Validation | 0.5-1 hour | Pending |
| **Total (Phases 1-2-4)** | **3.5-7 hours** | - |

**Current Status**: All Phase 1 scripts created and executing:
- `e086ef`: Quick grid search (36 configs) - Running
- `ac2907`: Focused stop loss test (5 configs) - Running

---

## Backtesting Best Practices Applied

### No Lookahead Bias
- Signals generated from daily data (not future bars)
- Entry at 3:50 PM (actual execution time)
- Exit at 9:30 AM next day (MOO)
- All prices from actual minute bars

### Proper Risk Management
- Position sizing: 15-25% per trade (realistic)
- Can hold multiple concurrent positions (realistic portfolio)
- Proper stop loss execution (minute-level checking)
- Transaction costs modeled (3 bps entry, 5 bps exit)

### Out-of-Sample Validation
- In-sample: 2020-2024 (for parameter optimization)
- Out-of-sample: 2017-2019 (for validation)
- No overlap between training and test periods

### Market Regime Awareness
- Tests include all regimes (bull, bear, sideways)
- Regime detection applied to all signals
- VIX threshold as secondary regime filter

---

## Success Metrics

### In-Sample Optimization
- Find configuration with Sharpe > 1.53 (baseline)
- Maintain CAGR > 35%
- Keep Max DD < -15%

### Out-of-Sample Validation
- OOS Sharpe > 80% of in-sample Sharpe (acceptable degradation)
- OOS CAGR > 25% (lower bar for OOS)
- Parameter stability score > 60%

---

## Documentation Files

- **This file**: Methodology and approach
- **Progress Chronicle**: `docs/agent-learnings/20251209_OMR_PARAMETER_OPTIMIZATION.md`
- **Results Summary**: Generated after each phase
- **Final Report**: Comprehensive analysis with recommendations

---

## Next Steps

1. Monitor Phase 1 optimization progress
2. Upon Phase 1 completion: Identify top 2-3 parameters
3. Run Phase 2 focused search on impactful parameters
4. Validate top 5 configurations on OOS data
5. Generate final report with recommendations

---

## Questions & Notes

**Key Uncertainties**:
1. Will position size have more impact than stop loss?
2. Is current 5-position maximum optimal or limiting diversification?
3. Are current min win rate (0.52) and expected return (0.3%) filters appropriately stringent?

**Known Constraints**:
- OMR only trades leveraged ETFs (limited universe: 62 symbols)
- Entry is hard-fixed at 3:50 PM (no flexibility)
- Exit is hard-fixed at 9:30 AM MOO (no flexibility)
- Overnight gap risk is not hedged

---

**Created by**: Claude Quantitative Research Agent
**Last Updated**: 2025-12-09 03:30 AM
