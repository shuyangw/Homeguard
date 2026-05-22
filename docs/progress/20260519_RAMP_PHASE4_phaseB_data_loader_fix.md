# RAMP Phase 4 Phase B - Data Loader Fix + V01/V03 Parity - 2026-05-19

## Summary

Reworked the Phase B harness data loader to use the actual on-disk Alpaca SIP
cache (`H:/Stock_Data/equities_daily_cache.parquet`, long-form, 2017-01-03 ->
2025-12-04, 3435 symbols) instead of the non-existent `daily/1day/<SYM>.parquet`
layout it was originally written for. Re-ran V01 and V03 end-to-end over the
full available window and produced the V01-vs-V03 parity report.

## Changes Made

- **`src/research/ramp_phase4/data.py`**: full rewrite of `_read_closes_from_parquet`
  - Reads the single long-form Parquet at `<storage>/equities_daily_cache.parquet`.
  - Pivots to wide (`index=trade_date`, `columns=symbol`, `values=close`).
  - `_read_universe_symbols` accepts any capitalization of the `symbol` header
    (the production CSV `config/universes/sp500-2025.csv` uses `Symbol`).
  - VIX is not present in the equities cache; added `_fetch_vix_yfinance` to
    fetch ^VIX once per call from yfinance. yfinance is used **only** for VIX;
    equities data is always Alpaca SIP per the no-yfinance-equities rule.
- **`scripts/backtest_scripts/_make_parity_report.py`**: new one-shot driver
  that re-runs V01 + V03 at a single cost tier and emits the parity Markdown.
- **`docs/reports/ramp/20260519_phase4_v01.md`**: V01 baseline, 4 cost tiers.
- **`docs/reports/ramp/20260519_phase4_v03.md`**: V03 target-weight-correct, 4 cost tiers.
- **`docs/reports/ramp/20260519_phase4_v01_vs_v03_parity.md`**: side-by-side at 5 bps.

## Key Result (5 bps per side, 2017-01-01 -> 2025-12-04)

| Metric | V01 | V03 | Delta (V03 - V01) |
|---|---:|---:|---:|
| Sharpe | 0.554 | 0.620 | +0.066 |
| CAGR | 129.22% | 95.04% | -34.18 ppts |
| Max DD | -66.84% | -44.43% | +22.40 ppts |
| Avg turnover | 91.39% | 72.53% | -18.85 ppts |
| Cost drag | 33.17% | 30.61% | -2.56 ppts |

**Conclusion: Option 1 -- V03 wins net.** Risk-adjusted return is meaningfully
better; CAGR is lower because crash exposure correctly halves gross during the
2020 and 2022 drawdowns. Advance to Wave 1 turnover-control on V03 base.

## Commits

- `958f964` fix(research): data loader reads equities_daily_cache.parquet with case-insensitive symbol column
- `f705dbc` report(ramp): Phase 4 V01 baseline (Alpaca SIP daily cache, 2017-2025-12-04)
- `9c83ee7` report(ramp): Phase 4 V03 target-weight-correct (Alpaca SIP daily cache, 2017-2025-12-04)
- (this commit) report(ramp): Phase 4 V01 vs V03 parity finding + session log

## Known Issues / Remaining Work

- **Survivorship bias**: universe is current S&P 500 membership, not
  point-in-time. Acknowledged in report headers; do not treat the absolute
  numbers as production-realistic until a PIT membership pipeline is in place.
- **No PIT delisting handling beyond NaN-driven forced exits**: the engine
  treats a missing close as a forced exit but the universe itself is static.
- **Cost drag is 30-33% of CAGR at the 5 bps tier** for both variants; turnover
  control is the next lever (Wave 1).
- **Cache stale by ~6 months**: `equities_daily_cache.parquet` covers through
  2025-12-04. Live verification beyond that window requires a refresh.
- **VIX fetched via yfinance** (single round-trip per panel load, ~1 s). For
  reproducibility we should consider snapshotting VIX into the FRED alt_data
  tree so the harness has no network dependency.

## Validation

- `pytest tests/research/ramp_phase4/test_data.py -v` -- 3/3 passed.
- Smoke load (June 2024, full S&P 500 universe + SPY + VIX): 19 trading days x
  505 columns in 1.0 s; no NaN in SPY or VIX over the smoke window.
- Smoke variant run (2023-01-01 -> 2024-03-31 V01): 311 daily records, first
  traded date 2024-01-03 STRONG_BULL, final PV $128,018 from $100,000 start.
- Full V01 and V03 runs over 2017-01-01 -> 2025-12-04 each completed in well
  under the 2-min-per-tier budget (4 tiers each).
