# FxCarrySeatbelt (#16 + #19) - 2026-07-07

## Summary
Built and gated the enhanced FX carry strategy `FxCarrySeatbelt` (research #16 Carry-Momentum Double Filter + #19 Carry-Unwind Detector), the first ENHANCED-form strategy of the 60-strategy FX catalog after four naive screens failed. Success bar was pre-registered as RELATIVE: OOS Sharpe must beat the S&P 500 over the same OOS dates. Result: both daily and weekly cadences FAIL (daily OOS Sharpe -0.75, weekly -0.11, vs S&P 0.68 over 2014-2026). Honest negative result, corroborated by negative in-sample Sharpe and correct cost monotonicity; not a bug. Merged to main via subagent-driven TDD (6 tasks, per-task reviews, whole-branch opus review = READY TO MERGE).

## Changes Made
- **src/backtesting/signals/carry_unwind.py** (new): reusable carry-unwind composite risk-off score `compute_unwind_score(close_panel, z_window=252)` = z(JPY strength delta) + z(CHF strength delta) + z(AUDJPY 5d realized vol) + z(XAUUSD 3d return), all causal trailing z-scores. Designed as a shared risk-off brain for #15/#16/#18/#42.
- **src/strategies/advanced/fx_carry_seatbelt.py** (new): `FxCarrySeatbelt` forecast_panel strategy. Long a G10 pair only when rate-diff carry > +2% AND price > EMA(50) with positive 10d slope (+10 Carver, never short for carry); defensive veto zeroes all longs when score >= 1.0 (hysteresis, clears after 3 consecutive days < 1.0); offensive short (-5) on AUDJPY/NZDJPY when score > 2.5 and pair < prior 20d low.
- **src/strategies/registry.py**: registered `FxCarrySeatbelt`.
- **config/backtesting/fx_carry_seatbelt_{daily,weekly}.yaml** (new): 22-pair G10, vol_target 0.03, leverage_cap 4, idm on. Identical except rebalance cadence.
- **src/backtesting/benchmark.py** (new): S&P 500 helpers -- `load_sp500_daily_returns`, `sp500_sharpe_over_dates`, `sp500_aligned_count`, `correlation_over_dates`, `information_ratio_vs_sp500`. SPX cache populated via equity_index_yfinance (4086 rows, 2010-2026, keyless).
- **scripts/backtest_scripts/run_fx_carry_seatbelt_walkforward.py** (new): walk-forward runner (36/12/12, both cost legs, both cadences) evaluating the S&P relative gate + PSR/DSR/PBO/IS-OOS/correlation/IR diagnostics + Aug-2024/Mar-2020 episode attribution, wrapped in RunStatus.
- **docs/reports/fx/FX_CARRY_SEATBELT_WALK_FORWARD.md** (new): the readiness report.
- **docs/reports/fx/20260706_carry_seatbelt_prereg.md** (new): pre-registration (committed before any result).
- Tests: test_carry_unwind.py (8), test_fx_carry_seatbelt.py (5), test_benchmark.py (5), test_fx_carry_seatbelt_configs.py (2). 26 pass across the feature surface.

## Commits
- `f7b2111` pre-registration note
- `6f2a9cd` + `91c527d` carry_unwind score + CHF sign test
- `5de8c4b` + `972c937` FxCarrySeatbelt strategy + zscore-floor fix
- `ca1ceb0` daily/weekly configs + smoke
- `8543dc4` + `c41f351` S&P benchmark helpers + cross-module Sharpe pin
- `2a0b5fb` walk-forward runner + report
- `8b94f05` final-review fix wave (S&P day-count transparency, stable dedup, is_oos_ratio guard)
- `0c540da` tracker: #16/#19 FAIL-enh

## Result (real, both cadences FAIL the S&P bar)
| Cadence | OOS Sharpe 1x | OOS Sharpe 1.5x | S&P Sharpe | Beats S&P | DSR | PBO | S&P n_days |
|---|---|---|---|---|---|---|---|
| daily | -0.7498 | worse | 0.6842 | False | 0.00 | 0.2171 | 3080 / 3196 |
| weekly | -0.1123 | worse | 0.6842 | False | 0.00 | 0.4199 | 3080 / 3196 |

Corroboration it is a real FAIL, not a bug: IS Sharpe also negative (daily -0.54, weekly -0.09); cost monotone (1.5x worse than 1x); veto not stuck-on; offensive short earned +1.43% in the Aug-2024 yen unwind (sign convention verified correct); S&P correlation ~0.07.

## Known Issues / Remaining Work
- Per the pre-registration (no absolute kill), ONE deferred variant remains before shelving carry on daily FX: #16 mod-a (EMA(50) replaced by 12-month TSMOM sign as the momentum leg) or mod-b (graded sizing). Decision pending: run the deferred variant vs. shelve carry vs. pivot to the intraday engine.
- Interpretation: the daily-frequency carry edge in this G10 universe looks genuinely weak across naive (#15) and enhanced (#16/#19) forms over the 2014-2026 USD-strength regime. The reusable carry_unwind score and the S&P benchmark harness are the durable assets.
- Minor (non-blocking, logged): boundary-dedup now uses sort_index(kind="stable"); is_oos_ratio guarded to NaN on non-positive Sharpe.

## Validation
- 26 tests pass across the feature surface (carry_unwind, strategy, benchmark, configs, fx_strategies).
- Whole-feature opus review: READY TO MERGE, no Critical findings; causality clean end-to-end, S&P comparison apples-to-apples, negative result corroborated.
- Walk-forward run reproduced twice with identical verdict.
