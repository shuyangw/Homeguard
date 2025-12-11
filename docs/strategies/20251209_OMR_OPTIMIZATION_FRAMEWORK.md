# OMR Parameter Optimization Framework

**Status**: OPTIMIZATION IN PROGRESS
**Start Time**: 2025-12-09 03:27 AM
**Expected Completion**: 2025-12-09 07:30 AM (4 hours estimated)

---

## Executive Summary

This document provides the framework for optimizing Overnight Mean Reversion (OMR) strategy parameters to maximize risk-adjusted returns (Sharpe ratio) while maintaining profitability and robustness.

**Key Baseline Metrics**:
- CAGR: 37.7%
- Sharpe: 1.53
- Max DD: -12.0%
- Win Rate: 47.5%
- Trades: 3,218
- Stop Loss %: 39.0%

**Optimization Objective**: Find configuration with Sharpe > 1.53 while keeping CAGR > 35%

---

## Parameter Optimization Status

### Scripts Created

1. **omr_grid_search_optimizer.py** (Full Grid Search)
   - Modes: QUICK (36), FOCUS_* (5 each), FULL (7,500)
   - Status: Running (quick test started 3:27 AM)
   - Output: CSV files with all tested configurations

2. **omr_parameter_focus_optimizer.py** (Sensitivity Analysis)
   - Tests 6 parameters independently (~30 configs)
   - Better for identifying impactful parameters
   - Faster than full grid search

3. **omr_validate_best_configs.py** (Out-of-Sample Validation)
   - Takes top configs and tests on 2017-2019
   - Confirms robustness and detects overfitting
   - Status: Ready to run after Phase 1

4. **omr_analysis_results.py** (Results Analysis)
   - Analyzes grid search output
   - Identifies top configurations
   - Computes parameter sensitivities
   - Status: Ready to run after Phase 1

---

## Parameters Being Optimized

| Parameter | Current | Range | Impact | Notes |
|-----------|---------|-------|--------|-------|
| Stop Loss | -1.5% | -1.0% to -2.0% | HIGH | Currently 39% triggering |
| Position Size | 20% | 15% to 25% | MEDIUM-HIGH | Portfolio leverage |
| Max Positions | 5 | 3 to 7 | MEDIUM | Diversification lever |
| Min Return | 0.3% | 0.2% to 0.5% | LOW-MEDIUM | Signal quality filter |
| VIX Threshold | 45 | 35 to 50 | LOW-MEDIUM | Regime filter |
| Min Win Rate | 0.52 | 0.50 to 0.54 | LOW | Bayesian filter |

---

## Execution Timeline

### Phase 1: Sensitivity Analysis (CURRENT - 1-2 hours)
- Quick Test: 36 configurations (stop loss, position size, max positions)
- Focused Tests: 5 configurations each for individual parameters
- Status: RUNNING

**Purpose**: Identify which parameters matter most

**Sample Output Expected**:
```
Stop Loss Sensitivity:
  -1.0%: Sharpe=1.48, CAGR=36.5%, DD=-11.2%
  -1.25%: Sharpe=1.52, CAGR=37.2%, DD=-11.8%
  -1.5%: Sharpe=1.53, CAGR=37.7%, DD=-12.0%  [BASELINE]
  -1.75%: Sharpe=1.51, CAGR=37.1%, DD=-12.5%
  -2.0%: Sharpe=1.48, CAGR=36.8%, DD=-13.2%

Position Size Sensitivity:
  15%: Sharpe=1.61, CAGR=28.4%, DD=-10.0%   [BEST DD]
  20%: Sharpe=1.53, CAGR=37.7%, DD=-12.0%   [BASELINE]
  25%: Sharpe=1.47, CAGR=47.1%, DD=-14.5%
```

### Phase 2: Focused Grid Search (2-4 hours)
- Focus on most impactful 2-3 parameters
- Test 10-50 combinations in promising region
- Status: PENDING (awaiting Phase 1 results)

**Purpose**: Find optimal values for impactful parameters

**Expected Output**:
```
Top 5 Configurations (by Sharpe):
1. SL=-1.35%, Pos=18%, MaxPos=6: Sharpe=1.58, CAGR=38.2%, DD=-11.5%
2. SL=-1.40%, Pos=19%, MaxPos=5: Sharpe=1.57, CAGR=37.9%, DD=-11.8%
...
```

### Phase 3: Combination Optimization (Optional - 2-4 hours)
- If Phase 1/2 suggests good opportunities
- Full grid search on top 2-3 parameters
- Status: CONDITIONAL

### Phase 4: Out-of-Sample Validation (1 hour)
- Test top 5 configs on 2017-2019 data
- Confirm robustness
- Detect overfitting
- Status: PENDING (awaiting Phase 2 results)

**Expected Output**:
```
Configuration 1 (Best In-Sample):
  In-Sample (2020-2024): Sharpe=1.58, CAGR=38.2%
  Out-of-Sample (2017-2019): Sharpe=1.45, CAGR=31.5%  [Degradation: 8%]

Configuration 2:
  In-Sample: Sharpe=1.57, CAGR=37.9%
  Out-of-Sample: Sharpe=1.48, CAGR=32.1%  [Degradation: 6%]  [PREFERRED]
```

---

## Expected Outcomes

### Scenario 1: Stop Loss is Key Parameter
**If Phase 1 shows large Sharpe variation with stop loss**:
- Proceed to Phase 2 with focused SL optimization (-1.3% to -1.6%)
- Likely improvement: +0.05-0.10 Sharpe (to 1.58-1.63)
- Timeline: 3 hours total

### Scenario 2: Position Size is Key Parameter
**If Phase 1 shows position size has highest impact**:
- Proceed to Phase 2 with focused position sizing (15%-22%)
- Likely improvement: +0.02-0.08 Sharpe (to 1.55-1.61)
- Timeline: 3-4 hours total

### Scenario 3: Multiple Parameters Matter Equally
**If Phase 1 shows balanced impact across parameters**:
- Proceed to Phase 3 full grid search
- Test all parameter combinations
- Likely improvement: +0.05-0.15 Sharpe (to 1.58-1.68)
- Timeline: 6-8 hours total

### Scenario 4: Current Configuration is Near-Optimal
**If Phase 1 shows baseline is already optimal**:
- Minor tweaks only (±5% position size, ±0.25% stop loss)
- Focus on robustness rather than improvement
- Accept 1.53 Sharpe as good as we can achieve
- Timeline: 2-3 hours total

---

## Decision Framework

### After Phase 1: Should We Continue?

**YES - Continue to Phase 2 if**:
- Any parameter shows improvement >5% in Sharpe (to >1.61)
- Multiple parameters show improvement >2% (to >1.56)
- Found region with CAGR > 40% and DD < -11%

**NO - Accept Baseline if**:
- All parameters show degradation
- Best variant < 1.50 Sharpe
- All improvements within measurement noise

### After Phase 2: Should We Run Phase 3?

**YES - Run Full Grid if**:
- Phase 2 found config with Sharpe > 1.60
- Multiple parameters still showing unexplored potential
- We have time/resources for 6-8 hour optimization

**NO - Skip Phase 3 if**:
- Phase 2 config has Sharpe < 1.55 (marginal improvement)
- Time constraints limit full exploration
- Clear optimal region already identified

### After Phase 4: Final Recommendation

**Deploy config if**:
- Out-of-sample Sharpe > 1.40 (80% of in-sample)
- Out-of-sample CAGR > 25%
- Parameter stability high (>60%)

**Reject config if**:
- Out-of-sample Sharpe < 1.30 (overfitted)
- Out-of-sample CAGR < 20%
- Large degradation from in-sample

---

## Key Hypotheses to Test

### H1: Stop Loss Optimization
**Current**: -1.5% (baseline)
**Hypothesis**: Optimal around -1.35% to -1.40% (tighter stops reduce catastrophic losses)
**Expected Impact**: Sharpe +0.03-0.08

### H2: Position Size Optimization
**Current**: 20%
**Hypothesis**: Optimal around 17-18% (smaller = lower vol = higher Sharpe despite lower CAGR)
**Expected Impact**: Sharpe +0.02-0.05

### H3: Diversification Benefit
**Current**: Max 5 positions
**Hypothesis**: Sharpe improves with more positions (up to 6-7) due to diversification
**Expected Impact**: Sharpe +0.01-0.03 per additional position

### H4: Signal Quality Filter
**Current**: Min return 0.3%, Min win rate 52%
**Hypothesis**: These filters are already optimally stringent (relaxing hurts, tightening helps little)
**Expected Impact**: Sharpe +0.00-0.02

### H5: Combined Effects
**Current**: All parameters at current values
**Hypothesis**: Combined optimization beats individual parameter tuning
**Expected Impact**: Sharpe +0.05-0.15 (from H1+H2+H3)

---

## Robustness Criteria

After finding optimal configuration, validate via:

1. **Time Period Robustness**: Performance stable across 2017-2024
2. **Regime Robustness**: Works in bull, bear, and sideways markets
3. **Sub-period Analysis**: Each year shows positive Sharpe (no lucky years)
4. **Parameter Sensitivity**: ±10% adjustment doesn't crash performance
5. **Out-of-Sample**: OOS Sharpe > 80% of in-sample

---

## Contingency Plans

### If Optimization Takes Longer Than Expected
- Can skip Phase 3 (full grid) and go straight to Phase 4 validation
- Prioritize robustness over maximum Sharpe
- Can run overnight if needed (estimated max 8 hours)

### If Best Config is Still Only 1.53 Sharpe
- Declare baseline already optimal
- Focus efforts on other strategies (MP, RAMP)
- Use saved scripts for future re-optimization

### If Multiple Configs Have Similar Sharpe
- Choose based on secondary criteria:
  1. Highest out-of-sample Sharpe
  2. Lowest maximum drawdown
  3. Simplest parameter set (Occam's razor)

---

## Scripts Ready to Run

All optimization scripts are created and ready:

```bash
# Phase 1: Sensitivity Analysis (Quick)
python backtest_scripts/omr_grid_search_optimizer.py --quick

# Phase 1: Parameter Focus Tests
python backtest_scripts/omr_grid_search_optimizer.py --focus-stop-loss
python backtest_scripts/omr_grid_search_optimizer.py --focus-position-size
python backtest_scripts/omr_grid_search_optimizer.py --focus-max-positions

# Phase 2: Full Grid (if needed)
python backtest_scripts/omr_grid_search_optimizer.py

# Phase 4: Validation (after getting top 5)
python backtest_scripts/omr_validate_best_configs.py \
    --configs output/optimization/omr_grid_search_top5_*.json

# Analysis (after grid search completes)
python backtest_scripts/omr_analysis_results.py \
    --results output/optimization/omr_grid_search_all_results_*.csv \
    --viable output/optimization/omr_grid_search_viable_*.csv
```

---

## Monitoring Progress

Track optimization via:

1. **Progress Chronicle**: `docs/agent-learnings/20251209_OMR_PARAMETER_OPTIMIZATION.md`
2. **Output Files**: Check `output/optimization/` for CSV and JSON results
3. **Process Status**: Monitor running process with BashOutput

Expected file outputs:
- `omr_grid_search_all_results_*.csv` - All tested configurations
- `omr_grid_search_viable_*.csv` - Only configurations with CAGR > 35%
- `omr_grid_search_top5_*.json` - Top 5 by Sharpe
- `omr_sensitivity_*.csv` - Parameter sensitivity data
- `omr_validation_results_*.json` - Out-of-sample validation

---

## Success Metrics

### In-Sample (Primary)
- [ ] Find config with Sharpe > 1.53
- [ ] Maintain CAGR > 35%
- [ ] Keep Max DD < -15%

### Out-of-Sample (Validation)
- [ ] OOS Sharpe > 1.42 (94% of baseline)
- [ ] OOS CAGR > 30%
- [ ] Parameter stability > 60%

### Robustness
- [ ] Works across all market regimes
- [ ] Positive Sharpe in most sub-periods
- [ ] ±10% parameter adjustment OK

---

## Recommendations by Discovery

**Once optimization completes, recommendations will be documented here based on findings.**

---

**Document Version**: 1.0
**Last Updated**: 2025-12-09 03:35 AM
**Status**: Framework Complete, Optimization Executing
