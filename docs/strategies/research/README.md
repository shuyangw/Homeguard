# Research Strategies

Strategies under research, backtesting, or shelved after evaluation.

## Naming Convention

- **Dated files** (`YYYYMMDD_*.md`): Research findings, backtest results, analysis
- **Undated files** (`*_STRATEGY.md`): Strategy specifications and documentation

## Strategy Status

| Strategy | Status | Best Result | Notes |
|----------|--------|-------------|-------|
| HV_ORB | Active Research | TBD | High volatility opening range breakout |
| ICT/SMC | Shelved | 0.81 Sharpe | See `20251215_ICT_SMC_BACKTEST_RESULTS.md` |
| ML Crypto MR | Active Research | TBD | ML-enhanced mean reversion for crypto |
| BMSB | New | TBD | Bull Market Support Band for crypto |
| ORB | Analysis | TBD | Opening range breakout variants |

## File Index

### Strategy Specifications
- [HV_ORB_STRATEGY.md](HV_ORB_STRATEGY.md) - High Volatility ORB
- [ICT_SMC_STRATEGY.md](ICT_SMC_STRATEGY.md) - ICT Smart Money Concepts
- [ORB_STRATEGY.md](ORB_STRATEGY.md) - Opening Range Breakout

### Research Findings (December 2025)
- [20251215_HV_ORB_RESEARCH_FINDINGS.md](20251215_HV_ORB_RESEARCH_FINDINGS.md)
- [20251215_ICT_SMC_BACKTEST_RESULTS.md](20251215_ICT_SMC_BACKTEST_RESULTS.md)
- [20251216_ML_CRYPTO_MEAN_REVERSION.md](20251216_ML_CRYPTO_MEAN_REVERSION.md)
- [20251216_ML_CRYPTO_MR_FINDINGS.md](20251216_ML_CRYPTO_MR_FINDINGS.md)
- [20251216_ORB_LEVERAGED_ETF_BACKTEST.md](20251216_ORB_LEVERAGED_ETF_BACKTEST.md)
- [20251216_ORB_ML_PREDICTION_PLAN.md](20251216_ORB_ML_PREDICTION_PLAN.md)
- [20251216_ORB_SP500_CHERRY_PICKS.md](20251216_ORB_SP500_CHERRY_PICKS.md)
- [20251216_ORB_SP500_WINNERS.md](20251216_ORB_SP500_WINNERS.md)
- [20251217_BMSB_CRYPTO_STRATEGY.md](20251217_BMSB_CRYPTO_STRATEGY.md)

## Promoting to Production

To move a strategy from research to production:
1. Complete walk-forward validation with positive OOS results
2. Create production strategy doc in `../production/`
3. Update deployment configurations
4. Test on paper account for at least 1 week
