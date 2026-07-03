# Bond Carry (CTD/yield-based) - Design

**Date:** 2026-07-03 · **Status:** approved, pre-plan · **Depends on:** merged futures pipeline (`main` @ 3de106a), `CarryCalculator`, FRED rates data on disk.

## Goal

Replace `CarryCalculator`'s `return 0.0` v1 fallback for the 6 price-traded bond futures
(ZT/ZF/ZN/TN/ZB/UB) with real, full-history bond carry, so futures carry runs on all 33
roots instead of ~27. The 6 rate roots are currently INERT (cached carry all-zero), which
narrows effective breadth and inflates carry's concentration (PBO 0.63 / kurtosis 33.5).

## Context (verified -- data is on hand)

- `src/data/carry_calculator.py::compute` bond branch (line 133): price-traded bond futures
  (ZT/ZF/ZN/TN/ZB/UB) hit `return 0.0` because "yield not directly available without CTD
  conversion factor." The yield-quoted MICRO_YIELD_ROOTS (2YY/5YY/10Y/30Y) already compute
  `duration * (front_yield - funding) / 100` correctly (front close IS the yield), but those
  micro contracts launched ~2021 and are NOT in the 33-root universe.
- `DURATION_BY_ROOT` already has ZT=2, ZF=5, ZN=9, TN=9, ZB=17, UB=22.
- **FRED data is downloaded, full history** (`alt_data/fred/{series}/daily.parquet`, schema
  `date,value`): `DGS2/DGS5/DGS10/DGS30` (constant-maturity Treasury yields) 1995-2026, and
  `DFF` (effective fed funds) 1995-2026. `SOFR` exists only 2018+. `derive_sofr` (SR1) is
  2018+ only -- so FRED `DFF` is the full-history funding source.

## Architecture (Approach A: FRED CMT yield - funding)

For each price-traded bond root: `carry = duration * (CMT_yield(tenor, d) - DFF(d)) / 100`
-- the SAME formula the micro-yield path uses, sourcing the yield from the FRED
constant-maturity series (full history) instead of the unavailable futures yield. The
micro-yield path (2YY/5YY/10Y/30Y) is UNTOUCHED.

Rejected: (B) micro-yield sibling futures -- only 2021+, most of history inert; (C)
futures calendar-slope -- bond-futures calendar spreads are dominated by CTD/delivery-option
noise, not clean carry.

## Components

1. **FRED rates reader** (`src/data/rates/fred_reader.py` or reuse an existing
   `alt_data/fred/` reader if one exists -- the `fred_rates.py` plugin is write-only, so a
   read helper is likely new): `get_fred_series(series_id, d) -> float` reading
   `get_local_storage_dir()/alt_data/fred/{series_id}/daily.parquet`, returning the value
   as of the latest date `<= d` (forward-fill weekends/holidays; the value is point-in-time,
   never revised for DGS/DFF). Raise if `d` precedes the series start.

2. **Tenor map + funding in `carry_calculator.py`:** a constant map
   `_BOND_CMT_TENOR = {"ZT":"DGS2","ZF":"DGS5","ZN":"DGS10","TN":"DGS10","ZB":"DGS30","UB":"DGS30"}`
   (TN ultra-10y ~ 10Y, UB ultra-bond ~ 30Y). Funding series = `"DFF"`.

3. **`CarryCalculator.compute` bond branch:** replace the `return 0.0` fallback (line 133)
   with:
   `cmt = get_fred_series(_BOND_CMT_TENOR[root], d); funding = get_fred_series("DFF", d);`
   `return duration * (cmt - funding) / 100.0`.
   Keep the `MICRO_YIELD_ROOTS` branch above it unchanged.

## Data Flow

`compute(ZN, "bond", d)` -> `get_fred_series("DGS10", d)` (yield) and `get_fred_series("DFF", d)`
(funding) -> `9 * (yield - funding) / 100`. `compute_history` walks days as today; the carry
cache builder then produces non-zero `{root}.parquet` for the 6 bonds.

## Point-in-Time / No Lookahead

FRED `DGS*`/`DFF` are published as-of-date and NOT revised, so reading the value on date `d`
(latest `<= d`) uses only information known that day. Forward-fill across non-print days
(weekends/holidays) is causal (carries the last known value forward, never backward). No
future data enters the carry on date `d`.

## Error Handling

- `d` before a series' start (all our series start 1995, before the 2010 backtest) -> the
  reader raises `ValueError`; `compute_history` already skips days that raise (logged debug).
- Missing FRED parquet for a mapped series -> `FileNotFoundError` (fail loud; the series ARE
  on disk, so this signals a real data regression).
- Sign: yield > funding -> positive carry -> long the bond future (upward-sloping/positive-
  carry regime). Inverted curve (funding > yield, e.g. 2023-24) -> negative carry -> short.

## Testing

- FRED reader: known-value read (a specific date's DGS10 matches the parquet), latest-`<=d`
  forward-fill on a weekend date, raise before series start.
- Bond carry known-value: on a date where DGS10 ~ 4.2 and DFF ~ 5.3, `compute("ZN","bond",d)`
  ~ `9*(4.2-5.3)/100 = -0.099` (negative -- inverted curve, correct sign). A positive-curve
  date (e.g. 2013, DGS10 > DFF) gives positive carry.
- All 6 price-traded roots return NONZERO (not 0.0) over 2010-2026.
- No-lookahead: `compute` for date `d` reads only FRED rows with date `<= d` (assert the
  reader never returns a value dated after `d`).
- Micro-yield path (2YY/5YY/10Y/30Y) unchanged (regression).

## Acceptance (execution, controller-run after implementation)

1. Rebuild the carry cache for the 6 bond roots (`build_carry_cache --jobs 6 --roots ZT ZF
   ZN TN ZB UB ...`) + confirm GC/CL (rebuilding now) -> all 33 roots non-zero/complete.
2. Re-run the broad carry walk-forward on the COMPLETE 33-root cache -> the corrected
   `CARRY_BROAD_READINESS.md`. Compare PBO/kurtosis/Sharpe to the 27-root 0.88/PBO-0.63
   baseline: quantify how much concentration was a data hole vs intrinsic. Also resolve the
   `nan` windows W11/W12 (diagnose whether they were caused by the same cache gaps).

## Scope / Out of Scope

- IN: FRED reader, the bond-branch fix, tenor map, tests, the rebuild + re-baseline.
- OUT: the XS-carry / IDM / combine strategy variants (separate efforts, measured AGAINST
  this corrected baseline). No change to the micro-yield path, the simulator, sizing, or the
  gate.
- Parameter-free: durations + tenor map + funding series are FIXED doctrine (no fitting) ->
  carry stays trial_count=1.
