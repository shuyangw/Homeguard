# RAMP Strategy Documentation

**Regime-Aware Momentum Protection (RAMP)** - A production trading strategy that adapts momentum parameters based on detected market regimes.

**Status**: Production paper (Phase 4 in progress — see Phase 4 note below)
**Last Updated**: 2026-05-19

---

> **2026-05-19 Phase 4 update — the 0.846 OOS Sharpe is GROSS, not tradeable.**
>
> The "Walk-Forward Validation" Sharpe numbers below were computed at **0% transaction costs** with no stateful turnover accounting (each rebalance day instantiated a fresh equal-weight portfolio in the validation script). Phase B of the Phase 4 review (`docs/superpowers/specs/2026-05-19-ramp-phase4-phaseB-design.md`) re-ran the same strategy on split-adjusted Alpaca SIP data with proper turnover state and realistic costs and produced very different numbers. Use the "Net-of-cost performance" section below as the load-bearing reference for any production-readiness claim; the "Walk-Forward Validation" table immediately below is retained for traceability but is gross-of-cost only.

---

## Overview

RAMP extends the basic momentum protection strategy with market regime detection, dynamically adapting parameters based on whether the market is in a bull, bear, sideways, or unpredictable state.

### Key Features

- **Regime Detection**: Uses VIX, SPY momentum, and moving averages to classify 5 market regimes
- **Adaptive Parameters**: Different momentum periods, weights, and position counts per regime
- **Walk-Forward Validated** (gross-of-cost): Trained on 2017-2021, tested out-of-sample on 2022-2024
- **Crash Protection**: Reduces exposure when VIX > 25 or SPY drawdown > 5%
- **Dynamic Position Sizing**: 1/N equal-weight allocation across top_n positions

### Performance (Walk-Forward Validation, gross-of-cost, yfinance)

| Metric | In-Sample (2017-2021) | Out-of-Sample (2022-2024) |
|--------|----------------------|---------------------------|
| **Sharpe Ratio** | 0.784 | **0.846** |
| **CAGR** | 16.2% | 16.3% |
| **Max Drawdown** | -46.9% | -15.0% |
| **Win Rate** | 53.3% | 52.9% |

**Validation**: 2025-12-12 using Yahoo Finance split-adjusted data, 0% costs, no turnover state.
See full validation report: [`20251212_RAMP_WALK_FORWARD_VALIDATION.md`](20251212_RAMP_WALK_FORWARD_VALIDATION.md)

### Net-of-cost performance (Phase 4 Phase B, 2026-05-19, Alpaca SIP split-adjusted)

Stateful target-weight backtest over 2017-01-01 → 2026-05-16 with realistic cost tiers:

| Cost tier (per side) | CAGR | Sharpe | Max DD | Avg turnover | Cost drag |
|---|---:|---:|---:|---:|---:|
| **0 bps (gross)** | 16.36% | **0.614** | -75.46% | 91% | 0% |
| 2.5 bps | 9.85% | 0.448 | -77.82% | 91% | 38% |
| **5.0 bps (typical IBKR-like)** | 3.74% | **0.282** | -79.88% | 91% | **75%** |
| 7.5 bps (stress) | -2.02% | 0.116 | -81.76% | 90% | 114% |

**Interpretation:**

- The strategy has a real but modest gross edge (Sharpe 0.614 on split-adjusted SIP, vs the gross 0.846 reported above on yfinance — yfinance's adjustments evidently flatter the numbers).
- The dominant problem is **~91% daily turnover**: at realistic 5 bps per side, transaction costs consume 75% of gross return.
- The cost-sensitivity gate from `docs/methodology/backtesting.md` §4 (variants must survive 1.5x base cost) is **not met** by the current production strategy.
- Turnover control (Phase 4 Wave 1 — V04 rank buffer, V05 minimum hold, V06 delta threshold, V11 combined) is the gating research item before any net-positive RAMP variant can be produced.

**Per-regime attribution (5 bps tier, 2017-2026):**

| Regime | Days | Net return |
|---|---:|---:|
| STRONG_BULL | 593 | +145.85% |
| WEAK_BULL | 698 | +469.53% |
| SIDEWAYS | 398 | -26.56% |
| BEAR | 375 | -29.66% |
| UNPREDICTABLE | 40 | -80.52% |
| SAFE_MODE | 251 | 0% |

Edge is concentrated in BULL regimes; SIDEWAYS/BEAR/UNPREDICTABLE are drags. Consistent with Phase 3A's earlier "BEAR-to-cash improves gross" finding.

**Source reports** (under `docs/reports/ramp/`, gitignored locally, pushed to `origin/ramp-phase4-turnover-regime-research`):

- `20260519_phase4_v01.md` — production REGIME_PARAMS, no crash exposure, four cost tiers.
- `20260519_phase4_v03.md` — production REGIME_PARAMS WITH crash exposure (V03 of the Phase 4 plan).
- `20260519_phase4_v01_vs_v03_parity.md` — V01 vs V03 parity finding: V03's crash-exposure halving cuts gross more than it cuts turnover-cost, so V03 is **worse** net than V01. Wave 1 turnover-control must come before any V03-style crash-exposure refinement.
- `20260522_phase4_re_baseline_vs_yfinance.md` — Phase B SIP harness vs the 2025-12-12 walk-forward and 2026-05-04 re-evaluation: documents the gross-vs-net divergence and confirms V01 RAMP fails the methodology section 4 cost-sensitivity gate at 7.5 bps per side.
- `20260522_phase4_wave1_findings.md` — Phase C Wave 1 (V04 rank buffer + V05 min hold + V06 delta threshold + V11 combined) results vs V01 base. **V11 PASSES the cost-sensitivity gate** at 7.5 bps with Sharpe 0.452 and CAGR +9.38% (V01 collapses to Sharpe 0.116 / CAGR -2.02%); turnover drops 91% -> 39%; EXT-OOS 2025-26 Sharpe goes -0.216 -> +0.527. V11 is the Phase D paper-trade candidate.
- `20260523_phase4_v11_readiness.md` — **V11 PARTIAL** readiness: passes the structural gates (PBO across 5 variants = 0.126; one-day-lag Sharpe robustness at 5 bps = +9.79%, lag is actually better than near_close, no structural lookahead) but FAILS the absolute-significance gates (PSR vs SR=0 = 0.944 just below 0.95; DSR at n_trials=20 = 0.811). The prior version of this doc reported PSR/DSR = 1.000 PASS; that was due to a units bug in how the Bailey-Lopez de Prado formula was being applied (annualized SR with daily `n`, which inflates the z-statistic by ~sqrt(252)). The corrected per-period application produces 0.944 / 0.811. V11 is structurally sound but its Sharpe magnitude (0.528 annualized over 9 years) isn't large enough to clear strict statistical-significance hurdles after multi-trial correction. Decision to advance to Phase D paper is a judgment call, not a clean PASS.

---

---

## Strategy Logic

### Universe
- S&P 500 stocks (approximately 500 symbols)
- Excludes leveraged ETFs (to avoid conflict with OMR strategy)

### Momentum Formula
```
momentum_score = (long_weight * return_long_period) - (penalty_weight * return_short_period)
```

The penalty term penalizes recent gains, avoiding buying stocks that have already run up significantly (avoids "buying at the top").

### Regime Detection

RAMP uses `MarketRegimeDetector` to classify the current market into 5 states:

| Regime | Description | Characteristics |
|--------|-------------|-----------------|
| **STRONG_BULL** | Strong uptrend | SPY above 50/200 MA, low VIX, positive momentum |
| **WEAK_BULL** | Weakening uptrend | SPY above MAs but momentum fading |
| **SIDEWAYS** | Range-bound | SPY between MAs, low directional momentum |
| **UNPREDICTABLE** | High uncertainty | Mixed signals, elevated VIX |
| **BEAR** | Downtrend | SPY below MAs, negative momentum |

### Regime-Specific Parameters

| Regime | Long Period | Short Period | Long Weight | Penalty Weight | Top N |
|--------|-------------|--------------|-------------|----------------|-------|
| **STRONG_BULL** | 21 | 5 | 0.3 | 5.0 | 20 |
| **WEAK_BULL** | 21 | 5 | 0.3 | 5.0 | 10 |
| **SIDEWAYS** | 21 | 5 | 0.5 | 2.0 | 5 |
| **UNPREDICTABLE** | 42 | 21 | 0.5 | 4.0 | 10 |
| **BEAR** | 21 | 5 | 0.3 | 3.0 | 10 |

**Rationale**:
- **STRONG_BULL**: More positions (20) for diversification in rising markets
- **SIDEWAYS**: Fewer positions (5) with larger size, lower penalty (2.0) for more stable signals
- **UNPREDICTABLE**: Longer lookback (42/21) to smooth noisy signals
- **BEAR**: Lower penalty (3.0) to catch oversold bounces

---

## Position Sizing

### Dynamic 1/N Allocation

RAMP uses equal-weight position sizing based on current regime's `top_n`:

```
position_pct = max_capital_allocation / current_top_n
target_value = portfolio_value * position_pct
```

### Capital Allocation Examples

With `max_capital_allocation = 1.0` (100%) and **$100,000** portfolio:

| Regime | top_n | Position Size | Per Position | Total Allocated |
|--------|-------|---------------|--------------|-----------------|
| STRONG_BULL | 20 | 5% | $5,000 | $100,000 |
| WEAK_BULL | 10 | 10% | $10,000 | $100,000 |
| SIDEWAYS | 5 | 20% | $20,000 | $100,000 |
| UNPREDICTABLE | 10 | 10% | $10,000 | $100,000 |
| BEAR | 10 | 10% | $10,000 | $100,000 |

**Key Points**:
- RAMP is always fully invested when enough buy signals exist
- Number of positions varies by regime (5-20)
- Each position is equally weighted within the regime
- In uncertain markets (SIDEWAYS), fewer but larger positions reduce noise

---

## Risk Management

### Crash Protection Triggers

| Signal | Threshold | Action |
|--------|-----------|--------|
| High VIX | > 25 | Reduce exposure to 50% |
| SPY Drawdown | > 5% from recent high | Reduce exposure to 50% |

When protection triggers, `exposure_pct = 0.5` (50%), so:
- All position sizes are halved
- Maintains portfolio diversification but with less capital at risk

### Portfolio Health Checks

Before each execution, RAMP verifies:
- Minimum buying power: $5,000
- Minimum portfolio value: $10,000
- Maximum positions: 25 (buffer above max 20)
- Position age: < 48 hours (drift detection)

---

## Schedule

| Time (EST) | Event |
|------------|-------|
| **9:30 AM** | Market opens, preload historical data |
| **3:55 PM** | Execute rebalancing (near close for accurate prices) |
| **4:00 PM** | Market closes |

**Why 3:55 PM?**
- Prices are more stable near close
- Reduces overnight gap risk from end-of-day volatility
- Allows time for order execution before close

---

## Configuration

### Live Adapter Parameters

```python
RAMPLiveAdapter(
    broker=broker,
    symbols=None,                    # Default: S&P 500
    max_capital_allocation=1.0,      # 100% of portfolio
    reduced_exposure=0.5,            # 50% when protection triggers
    vix_threshold=25.0,              # VIX level for protection
    spy_dd_threshold=-0.05,          # 5% SPY drawdown threshold
    slippage_per_share=0.01,         # Expected slippage
    data_provider=None               # Uses broker if not provided
)
```

### Strategy Toggle Configuration

Location: `config/trading/strategy_toggle.yaml`

```yaml
strategies:
  ramp:
    enabled: true
    shutdown_requested: false
  omr:
    enabled: true
    shutdown_requested: false
  mp:
    enabled: false  # Replaced by RAMP
    shutdown_requested: false
```

---

## Deployment

### EC2 Service

Service file: `infra/ec2/services/homeguard-ramp.service`

```bash
# Start RAMP service
sudo systemctl start homeguard-ramp

# Check status
sudo systemctl status homeguard-ramp

# View logs
journalctl -u homeguard-ramp -f

# Restart after code changes
sudo systemctl restart homeguard-ramp
```

### Multi-Strategy Coordination

RAMP runs alongside OMR with proper isolation:
- Separate systemd services (`homeguard-ramp`, `homeguard-omr`)
- Separate position tracking in `strategy_positions.json`
- Execution locks prevent simultaneous order execution
- Different universes (S&P 500 vs leveraged ETFs)

---

## Code Architecture

### File Locations

| File | Purpose |
|------|---------|
| `src/strategies/advanced/ramp_strategy.py` | Pure signal generation logic |
| `src/trading/adapters/ramp_live_adapter.py` | Live trading adapter |
| `src/strategies/advanced/market_regime_detector.py` | Regime classification |
| `infra/ec2/services/homeguard-ramp.service` | Systemd service definition |
| `config/trading/strategy_toggle.yaml` | Enable/disable toggle |

### Class Hierarchy

```
StrategyAdapter (base)
    └── RAMPLiveAdapter
            │
            ├── RAMPSignalWrapper (StrategySignals)
            │       └── RAMPSignals (core logic)
            │
            ├── StrategyStateManager (position tracking)
            └── PortfolioHealthChecker (risk checks)
```

### Signal Flow

```
Market Data (Alpaca API)
        v
RAMPSignals.generate_signals()
        v
MarketRegimeDetector.detect_regime()
        v
Calculate momentum with regime-specific params
        v
Rank stocks, select top_n
        v
Compare with current positions
        v
Generate BUY/SELL signals
        v
RAMPLiveAdapter.execute_signals()
        v
AlpacaBroker.place_stock_order()
        v
StrategyStateManager.add_position()
```

---

## Decision History

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-05-19 | Phase 4 Phase B re-baseline | Stateful SIP-adjusted backtest: gross Sharpe 0.614 (vs yfinance 0.846), net 0.282 at 5 bps. Turnover-control (Wave 1) becomes gating before any further regime/exposure refinement. RAMP paused on production paper pending A7 validation. |
| 2026-05-15 | Phase 4 Phase A code complete | F1 planner + F2 target-aware execution + F3 parity tests + F4 safe mode + F5 decision-log enrichment landed on branch `ramp-phase4-turnover-regime-research`. |
| 2025-12-12 | Re-validated with clean data (gross) | OOS Sharpe 0.846 (YF split-adjusted data, 0% costs, no turnover state). **Gross-only**; see 2026-05-19 entry for net-of-cost re-baseline. |
| 2025-12-08 | Deployed RAMP, disabled MP | Regime detection adds value over static momentum |
| 2025-12-08 | Changed to 1/N position sizing | Simpler, matches backtest methodology |
| 2025-12-07 | Walk-forward validation | Confirmed regime detection adds value OOS |
| 2025-12-06 | Optimized regime parameters | Grid search on 2017-2021 data |

---

## Monitoring

### Log Messages

Key log prefixes to watch:
- `[RAMP]` - General RAMP messages
- `[RAMP] Regime:` - Current detected regime
- `[RAMP] Position sizing:` - Per-position allocation
- `[RAMP] Risk signals:` - Protection triggers

### Health Indicators

| Indicator | Good | Warning |
|-----------|------|---------|
| Regime detection | Stable regime for 2+ days | Rapid regime changes |
| Position count | Matches regime's top_n | Significantly different |
| Fill rate | > 95% | < 90% |
| Buying power | > $5,000 | < $5,000 |

---

## Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| No signals generated | Data fetch failed | Check Alpaca API connection |
| Wrong position count | Regime changed | Normal - will rebalance next day |
| Protection triggered | VIX > 25 or SPY drawdown | Expected behavior, positions halved |
| Execution lock timeout | Other strategy running | Wait for release or check logs |

### Debugging Commands

```bash
# Check current regime and positions
journalctl -u homeguard-ramp --since "1 hour ago" | grep -E "Regime:|Position"

# Check for errors
journalctl -u homeguard-ramp --since "1 hour ago" | grep -i error

# View full execution log
journalctl -u homeguard-ramp --since "3:50 PM" --until "4:05 PM"
```

---

## Related Documentation

- [Strategy Framework](../../src/strategies/STRATEGY_FRAMEWORK.md)
- [Live Trading System](../../src/trading/LIVE_TRADING_SYSTEM.md)
- [Multi-Strategy Position Management](../architecture/MULTI_STRATEGY_POSITION_MANAGEMENT.md)
- [Market Regime Detector](../../src/strategies/advanced/market_regime_detector.py)

---

## Changelog

- **2025-12-08**: Initial production deployment, replaced MP strategy
- **2025-12-08**: Documentation created
- **2026-05-23**: V11 (Phase 4 Wave 1 combined turnover-lite) deployed to production paper. Variant flipped via `config/trading/strategy_toggle.yaml`. PARTIAL readiness verdict: passes PBO (0.126) and one-day-lag robustness (+9.79%); fails strict PSR (0.944) and DSR (0.811) under per-period BLdP application. Paper validation gate: A7 counter must reach 5 clean sessions. Production live remains gated. Deploy details: `docs/progress/20260523_RAMP_PHASE4_V11_PRODUCTION_PAPER.md`.
