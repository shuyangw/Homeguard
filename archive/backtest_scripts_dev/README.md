# Archived Backtest Scripts

This directory contains development/historical backtest scripts that are no longer actively used.

## Why Archived?

These 76+ scripts were created during strategy development and debugging. Most fall into categories:

1. **One-off debugging** - Scripts like `diagnose_nov14_*.py` that investigated specific issues
2. **Development iterations** - v1/v2/v3 versions superseded by final implementations
3. **Period-specific tests** - `*_bull_2019_2021.py` variants that tested hypotheses
4. **Redundant variations** - Multiple scripts doing essentially the same thing

## Active Scripts (Still in backtest_scripts/)

Only 8 operational scripts remain active:

| Script | Purpose | Usage |
|--------|---------|-------|
| `simulate_today_signals.py` | Generate today's trading signals | Daily |
| `walk_forward_validation.py` | Periodic strategy robustness check | Quarterly |
| `overnight_walk_forward_validation.py` | OMR-specific walk-forward | Quarterly |
| `validate_overnight_strategy_v3_full_universe.py` | Full validation of production strategy | After changes |
| `optimize_overnight_strategy.py` | Re-optimize OMR parameters | Annually |
| `run_pairs_trading.py` | Run pairs trading backtest | If using pairs |
| `download_leveraged_etfs.py` | Download new symbol data | When adding symbols |
| `check_available_symbols.py` | Verify data availability | Before backtests |

## If You Need Something From Here

1. Check if the config-driven system can do it: `python -m src.backtest_runner --config configs/examples/ma_single.yaml`
2. If not, copy the specific script back to `backtest_scripts/`
3. Scripts may have stale imports - update paths if needed

## Archived Date

2025-11-26

## Categories of Archived Scripts

### Debugging (Safe to Delete Eventually)
- `diagnose_nov14_*.py` (5 scripts)
- `debug_*.py` (2 scripts)
- `fix_*.py` (2 scripts)
- `test_vix_*.py` (3 scripts)

### Strategy Development (Historical Reference)
- `optimize_*.py` - Various optimization experiments
- `validate_*.py` - Validation iterations
- `comprehensive_pairs_validation*.py` - Pairs trading development

### Analysis (One-time Research)
- `analyze_*.py` - Various analysis scripts
- `monthly_crisis_*.py` - Crisis period analysis

### Subdirectories
- `frameworks/` - Validation framework code
- `utils/` - Shared utilities
- `config/` - Configuration files
- `basic/`, `intermediate/`, `advanced/` - Categorized scripts
- `optimization/`, `sweeps/` - Specific script types
