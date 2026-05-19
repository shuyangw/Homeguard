# RAMP Phase 4 Phase B - Fresh SIP Data Loader + V01/V03 Re-run - 2026-05-19

## Summary

Extended the Phase B harness data loader to aggregate the fresh 1-min Alpaca
SIP split tree (`H:/Stock_Data/equities_1min_sip_split/`) to a daily panel,
cached the result, and re-ran V01 and V03 over the full window
2017-01-01..2026-05-16. The new data revealed that the legacy
`equities_daily_cache.parquet` we were using before stores UNADJUSTED close
prices, which were silently inflating both variants' returns by treating
every stock split as a real ~67-80% crash and the strategy mechanically
"buying the dip" the next day at the post-split price. With properly
split-adjusted prices, V01 is ~+4% CAGR / 0.28 Sharpe at 5 bps and V03 is
slightly negative, flipping the prior parity conclusion.

## Changes Made

- **`src/research/ramp_phase4/data.py`**:
  - Added `_aggregate_symbol_daily(symbol_dir, start, end)` that walks one
    symbol's `year=Y/month=M/data.parquet` partitions, filters minute bars
    to RTH (09:30-16:00 ET), and groups by trade date taking the LAST
    close. Falls back to all-bars / last-per-day if RTH window is empty
    (rare malformed month).
  - Added `_aggregate_to_daily_from_sip_split(symbols, start, end)` that
    invokes the per-symbol aggregator across the requested universe and
    returns a wide daily-close DataFrame. Symbols whose `symbol=<SYM>`
    partition is absent get an all-NaN column.
  - Added `_load_or_build_sip_daily_cache(symbols, start, end)` that
    persists the aggregated long-form panel to
    `H:/Stock_Data/cache/ramp_phase4/equities_daily_from_sip.parquet`
    (1.14M rows, 504 symbols). Cache invalidation: rebuild only if
    `max_date < end - 7 days` (slack for weekends/holidays so a Friday
    close is good for the following weekend).
  - Renamed the prior loader to `_read_closes_from_legacy_cache` and kept
    it as a fallback that only fires if SIP aggregation throws or yields
    an empty panel.
  - `_read_closes_from_parquet` now resolves source order: FRESH SIP ->
    legacy stale cache, then joins VIX from yfinance as before.
- **`tests/research/ramp_phase4/test_data.py`**: added
  `test_aggregate_symbol_daily_groups_by_rth_close` (tmp-path partition with
  pre-market, RTH, post-market bars; asserts the last RTH bar wins and
  post-market bars are ignored). All 4 tests pass.
- **`docs/reports/ramp/20260519_phase4_v01.md`**: regenerated with FRESH
  SIP daily panel, full window 2017-01-01..2026-05-16, 4 cost tiers.
- **`docs/reports/ramp/20260519_phase4_v03.md`**: same, V03.
- **`docs/reports/ramp/20260519_phase4_v01_vs_v03_parity.md`**: regenerated
  side-by-side at 5 bps.

## Why the loader was updated

The user re-downloaded SIP data over 24h into the 1-min split tree
(`equities_1min_sip_split/symbol=<SYM>/year=<Y>/month=<M>/data.parquet`),
fresh through May 2026. The prior loader read
`equities_daily_cache.parquet` which was stale at 2025-12-04 and missing
the entire EXT-OOS Q1+Q2 2026 window. Beyond freshness, the SIP minute
data is split-adjusted (verified below); the legacy daily cache was NOT.

## Performance

- **Cache build (cold)**: 504 symbols x ~10 years of 1-min data
  aggregated to daily in **~287s (~4.8 min)** on the user's machine.
  Aggregation logs `progress: N/504` every 50 symbols.
- **Cache hit (warm)**: full panel load (2355 dates x 505 cols including
  SPY+VIX) in **~1.0s**.
- **Cache size**: 1,136,050 long-form rows, 504 unique symbols, single
  Parquet file at `H:/Stock_Data/cache/ramp_phase4/equities_daily_from_sip.parquet`.

## Critical finding: legacy cache was NOT split-adjusted

Spot check on AAPL 4-for-1 split (2020-08-31):

| trade_date | legacy close | SIP daily close | legacy / SIP |
|---|---:|---:|---:|
| 2020-08-28 | 501.10 | 124.71 | 4.018 |
| 2020-08-31 | 129.33 | 128.85 | 1.004 |

And TSLA 3-for-1 split (2022-08-25):

| trade_date | legacy close | SIP daily close | legacy / SIP |
|---|---:|---:|---:|
| 2022-08-24 | 892.60 | 297.15 | 3.004 |
| 2022-08-25 | 295.70 | 296.11 | 0.999 |

The legacy `equities_daily_cache.parquet` stores RAW unadjusted closes,
so every stock-split day shows a ~67-80% one-day "loss" followed by a
~300-400% "recovery". The regime detector flagged these days as BEAR,
the momentum scorer then identified the post-split stocks as deeply
oversold + showing huge prior gains, and the strategy bought them at
the new low price -- producing the prior phantom 129% CAGR / Sharpe
0.55 baseline. **None of those returns are real.** The SIP minute data
is split-adjusted at source (confirmed across both example splits) and
gives the corrected numbers below.

## New V01 / V03 metrics (FRESH SIP, 2017-01-01..2026-05-16)

### V01 at 5 bps per side

| Metric | Value |
|---|---:|
| CAGR | 3.74% |
| Sharpe | 0.282 |
| Max DD | -79.88% |
| Avg daily turnover | 90.64% |
| Cost drag | 75.28% |

### V03 at 5 bps per side

| Metric | Value |
|---|---:|
| CAGR | -0.84% |
| Sharpe | 0.077 |
| Max DD | -66.76% |
| Avg daily turnover | 72.67% |
| Cost drag | 110.88% |

### Parity at 5 bps (V03 - V01)

| Metric | V01 | V03 | Delta |
|---|---:|---:|---:|
| Sharpe | 0.282 | 0.077 | -0.204 |
| CAGR | 3.74% | -0.84% | -4.58 ppts |
| Max DD | -79.88% | -66.76% | +13.12 ppts |
| Avg turnover | 90.64% | 72.67% | -17.98 ppts |
| Cost drag | 75.28% | 110.88% | +35.60 ppts |

## Updated parity conclusion

Was Option 1 ("V03 wins net"). Now **Option 2: V03 wins gross-of-cost
drawdown but loses Sharpe and CAGR net of cost to V01.** V03's correct
crash-exposure halving cuts gross WEAK_BULL returns (the dominant
contributor) roughly in half without proportionally cutting turnover-cost,
so the cost drag explodes to >100% of gross return. The "right" thing
to do at the variant-correctness level still hurts net performance at
realistic costs.

Both variants are sub-viable at 5 bps: V01 PSR is far from significant
at Sharpe ~0.28, V03 is breakeven/negative. EXT-OOS Q1+Q2 2026 is now
present in the panel but the cost-of-trading dwarfs any per-regime edge.
**Turnover control is no longer an "advance" -- it's a precondition for
ANY net-positive variant.** Wave 1 work should re-prioritize that
accordingly.

## Commits

- (this branch) `fix(research): aggregate 1-min SIP to daily for fresh end-of-window coverage`
- (this branch) `report(ramp): Phase 4 V01/V03 + parity on FRESH SIP daily (2017-01-01 to 2026-05-16)`
- (this branch) `docs(progress): Phase B data-fresh recovery session log`

## Known Issues / Remaining Work

- **10 universe symbols missing from SIP tree**: BRK.B, MMC, FI, K,
  CTRA, HOLX, BF.B, DAY, WBA, IPG (mostly dot-bearing tickers and
  recent index additions). They are present in the legacy cache. Engine
  treats them as NaN -> forced exit. Worth a follow-up to either source
  these from the legacy cache as a per-symbol fallback or download them
  fresh.
- **Survivorship bias remains** -- universe is current SP500 membership,
  not point-in-time. Reported numbers should not be treated as
  production-realistic until a PIT membership pipeline is in place.
- **Cost drag dominates** at 5 bps for both variants. Turnover control
  is now a precondition rather than an optimization.
- **Both variants' Max DD blown out** -- V01 -80%, V03 -67%. The
  previously reported -65/-44% values were artifacts of the unadjusted
  data inflating cumulative returns so denominator was larger.
- **VIX still yfinance** -- single round-trip per panel load. Same
  reproducibility caveat as the prior session log.

## Validation

- `pytest tests/research/ramp_phase4/ -v` -- 41/41 passed.
- Cold aggregation wall-clock: 287s for 504 symbols x ~10 years.
- Warm cache load: 1.0s for the full 2017-2026-05-16 panel.
- AAPL/TSLA split-adjustment spot check confirms SIP daily is correctly
  split-adjusted while the legacy cache is not.
- V01 and V03 each ran all 4 cost tiers over the full window in well
  under 5 minutes per backtest (in parallel).
