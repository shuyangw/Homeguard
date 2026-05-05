# RAMP Re-Evaluation - 2026-05-04

## Summary
Ran a diagnostic re-evaluation of the RAMP strategy using production parameters on 2017-2026 data. The 0.846 OOS baseline (2022-2024) was confirmed at 0.823, but the truly-OOS 2025-2026 period showed material decay (Sharpe 0.074, CAGR -1.5%).

## Changes Made
- **scripts/backtest_scripts/ramp_re_eval_20260504.py**: New working copy of the walk-forward validation script; fixed universe path, extended to three periods (IS/OOS/EXTENDED-OOS), added regime breakdown for EXTENDED-OOS, saves JSON summary
- **docs/reports/ramp/20260504_re_evaluation.md**: Full re-evaluation report with results table, baseline comparison, alpha decay assessment, regime breakdown, and recommendation

## Commits
- `6de6e60` docs(ramp): re-evaluate baseline on 2017-2026 data

## Known Issues / Remaining Work
- BEAR regime is the primary drag in 2025-2026 (Sharpe -2.17 over 64 days); needs investigation before Phase B proceeds
- WEAK_BULL (43.6% of 2025-2026 time) also slightly negative; may indicate the momentum formula is not adapting to current macro environment
- Phase B (z-score infrastructure swap) is on hold pending human review of BEAR regime behavior
- The regime detector's classification of 2025 tariff-shock events has not been verified -- may warrant spot-checking

## Validation
- yfinance connectivity verified before run
- REGIME_PARAMS in script confirmed identical to production `ramp_strategy.py:44-50` (line-by-line check)
- Original `ramp_walk_forward_validation.py` confirmed unchanged (still has wrong path)
- Script ran successfully: 503 symbols, 2447 days loaded in 11.4s
- JSON summary saved to `logs/backtesting/results/20260504_233944_ramp_reeval_summary.json`
