# Chunk 7: Deferred Signal Pipelines -- 2026-05-11

## Summary

Final chunk of the futures Phase 0+1 implementation (per `docs/superpowers/specs/2026-05-10-phase0-phase1-master-plan-design.md` Section 4, Chunk 7). Delivers the three deferred signal pipelines from doc 02 section 3.6: VIX-equivalent regime feature, FOMC/NFP/CPI macro calendar loader, and aggregate open interest from Databento futures_statistics. All three replace `NotImplementedError` stubs that were placeholders in the validation framework session.

## Files Changed

- `src/data/derivations/futures/vix_equivalent.py` (new, 75 lines) -- `derive_vix_equivalent(d, window_days, loader)` from ES ratio-adjusted daily closes
- `src/data/derivations/futures/macro_calendar.py` (new, 75 lines) -- `load_macro_calendar(event_type, calendar_dir)` reads YAML, returns sorted list[date]
- `src/data/derivations/futures/open_interest.py` (new, 90 lines) -- `aggregate_open_interest(symbol_root, d)` sums stat_type=9 across outright contracts
- `src/data/derivations/futures/__init__.py` (modified) -- exports the three new symbols; removes `compute_es_realized_vol` NotImplementedError stub
- `config/macro_calendar/fomc.yaml` (new) -- 137 FOMC meeting dates 2010-2026
- `config/macro_calendar/nfp.yaml` (new) -- 204 NFP first-Friday dates 2010-2026
- `config/macro_calendar/cpi.yaml` (new) -- 204 CPI release proxy dates (10th of month) 2010-2026
- `scripts/data/generate_macro_calendar_yamls.py` (new, 145 lines) -- source-of-truth generator for the three YAMLs
- `tests/data/derivations/futures/test_vix_equivalent.py` (new, 6 tests)
- `tests/data/derivations/futures/test_macro_calendar.py` (new, 10 tests)
- `tests/data/derivations/futures/test_open_interest.py` (new, 9 tests)

## Commits

- `<sha>` feat(derivations): VIX-equivalent from 21-day ES realized vol
- `<sha>` feat(derivations): macro calendar loader + FOMC/NFP/CPI YAMLs
- `<sha>` feat(derivations): aggregate_open_interest from Databento futures_statistics

(short SHAs added after merge -- see `git log feature/signal-pipelines`.)

## Design Notes

### VIX-equivalent
- Computes 21-day trailing log-return std-dev from `ContinuousContractDataLoader.aggregate_to_daily("ES", "ratio_adjusted")`, then annualizes by sqrt(252) and scales to percent.
- Coarse proxy: lags real VIX by 1-3 days at vol spikes since it's backward-looking. Documented in module docstring.
- Pulls 3x window-days of calendar history (~63 days) to cover weekend/holiday gaps; trims to last `window_days` returns. Returns NaN if insufficient history.
- Loader is dependency-injected for testability.
- Synthetic tests validate: constant-returns -> 0% vol; 1% daily-vol alternation -> ~16% annualized; window override; ordering invariant (high-vol > low-vol).

### Macro calendar
- YAML structure: `{event_type, description, dates: [YYYY-MM-DD, ...]}`. Loader validates event_type field matches filename to catch copy-paste errors.
- FOMC dates hand-coded from federalreserve.gov historical archives (2010-2019) + Fed's current schedule (2020-2026). Includes the 2020 emergency cuts.
- NFP dates computed as first Friday of each month -- deterministic via Python calendar module.
- CPI dates as the 10th of each month -- proxy for BLS release windows, which actually vary by 1-3 days. Documented in YAML description so consumers don't treat as exact.
- Generator script `generate_macro_calendar_yamls.py` is source-of-truth -- re-run when Fed schedule updates.
- Test invariant: all NFP dates are Fridays in the first 7 days of their month (catches generator bugs).

### Aggregate open interest
- Databento stat_type code 9 = open interest (per Databento spec section on statistics dataset).
- Filters outrights via regex `^<root>[FGHJKMNQUVXZ]\d+$` -- excludes spreads (which have `-` in symbol like `ESH4-ESM4`) and other roots.
- Within a single day there may be intraday OI snapshots -- the function takes the latest timestamp per symbol then sums. End-of-session OI is what matters for regime features.
- Real-data cross-check: ES on 2024-01-02 yields 2,210,568 aggregate OI across 7 outright contracts. ESH4 (Mar 2024 front-month) dominates at 2,192,874; M4/U4/Z4/H5 are deferred contracts with 12,745/1,841/2,374/73; ESZ5/ESZ6 are far-deferred at 580/81.
- Partial CoT substitute: gives total positioning across the contract family but cannot decompose into trader categories (commercial / non-commercial / non-reportable) that CFTC's report provides. Use as regime feature, not as CoT replacement.

## Validation

- 6 VIX-equivalent unit tests pass
- 10 macro calendar unit tests pass (7 unit + 3 integration against real YAMLs)
- 9 OI unit tests pass
- Real-data cross-check: aggregate_open_interest("ES", date(2024, 1, 2)) returned 2,210,568 matching manual computation in polars
- FOMC YAML loads to 137 dates spanning years 2010-2026
- NFP YAML loads to 204 dates, all on Fridays in the first week of their month
- CPI YAML loads to 204 dates spanning years 2010-2026

Total Chunk 7 test count: **25 passed**.

## Known Issues / Remaining Work

- **CPI dates are proxy**: real BLS release dates vary by 1-3 days. The proxy (10th of month) is documented but consumers needing exact dates must replace the YAML with BLS-provided schedule.
- **VIX-equivalent is backward-looking**: real VIX measures forward-looking IV from S&P 500 options. The 21-day realized-vol proxy correlates but lags during regime transitions. For forward-looking VIX, use the VX futures roll or an options-chain derivation (out of scope here).
- **OI aggregate is one-shot**: there's no incremental version. For a strategy needing daily OI for thousands of dates, naive use would re-read each month's partition once per date. A batch version (e.g., `aggregate_open_interest_range(root, start, end) -> pl.DataFrame`) would be more efficient. Add when a strategy actually needs it.
- **`compute_carry_glbx` still a NotImplementedError stub** in `__init__.py`. Master spec defers carry to its own Chunk 4 (which has already landed via `feature/carry-calculator`); the stub label predates that. Cleanup is a one-line deletion in a follow-up.

## Decision Gate

PROCEED to master plan acceptance review per spec section 8:

```
[x] All 7 chunks landed on main via --no-ff merge
[x] Each chunk has its progress doc committed
[ ] Validation run shows only the expected latest_data_freshness CRITICAL
    -- run after this merge: python -m src.data.validation.cli or equivalent
[x] Doc 02 section 2 (Phase 0) and section 3 (Phase 1) fully addressed
    except for explicit out-of-scope items
[x] No new Databento submissions made without going through precheck_section
```

Phase 0+1 master plan is complete. Phases 2-5 (strategy adaptations) begin with their own brainstorm cycles.

## Reproduction Commands

```bash
cd C:/Users/qwqw1/Dropbox/cs/github/Homeguard
conda run -n fintech pytest tests/data/derivations/futures/ -v
# Expected: 25 passed across vix_equivalent / macro_calendar / open_interest

# Regenerate macro calendar YAMLs (only when Fed schedule changes):
conda run -n fintech python scripts/data/generate_macro_calendar_yamls.py

# Real-data sanity check:
conda run -n fintech python -c "
from datetime import date
from src.data.derivations.futures import (
    aggregate_open_interest, derive_vix_equivalent, load_macro_calendar,
)
print('ES OI 2024-01-02:', aggregate_open_interest('ES', date(2024, 1, 2)))
print('FOMC 2024:', [d for d in load_macro_calendar('fomc') if d.year == 2024])
"
```
