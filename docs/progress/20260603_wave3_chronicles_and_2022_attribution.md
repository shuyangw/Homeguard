# Wave-3 Section-12 Chronicling + 2022 Bear Attribution -- 2026-06-03

## Summary

Two deliverables completed in one session: (A) Section-12-compliant trade/holdings
chronicling added to the research harness; (B) 2022 bear-year attribution for V28, V31,
V26-robust vs V11, establishing the mechanism behind V11's 2022 outperformance.

## Changes Made

**Part A -- Chronicling (engine + runner):**
- `src/research/ramp_phase4/engine.py`: Added `trades: List[Dict]` field to `DailyRecord`
  (default empty list). Both timing branches (near_close, one_day_lag) and SAFE_MODE branch
  now populate it with the `compute_trades()` output for that day.
- `scripts/backtest_scripts/ramp_phase4_wave3_readiness.py`: Added `_write_chronicles()`
  (atomic gzip CSV write for holdings + trade ledger), `_should_chronicle()` filter, and
  `--no-chronicles` / `--chronicles-filter` CLI args. Default: near_close:5.0 only.
- `tests/research/ramp_phase4/test_chronicles.py`: 12 new tests covering trades field
  population, turnover consistency, and CSV schema/row-count validation.

**Part B -- Attribution runs and analysis:**
- Ran V11, V28, V31, V26-robust (near_close, 5bps, full window 2017-2026-05-16) with
  chronicles on; produced 8 chronicle files in `docs/reports/ramp/holdings/`.
- `scripts/scratch/ramp_2022_bear_attribution.py`: Attribution script using
  `load_universe_panel` for price data; computes set-difference unique holdings,
  per-name contributions, trailing beta, and exposure check.
- `docs/reports/ramp/20260603_wave3_2022_bear_attribution.md` + `.json`: Full report.

## Key Attribution Findings

2022 calendar returns (registry, authoritative):
- V11: -16.5%, V26-robust: -19.3%, V28: -20.0%, V31: -26.0%

Exposure mechanism: ALL variants ~100% invested in 2022 (avg gross 0.997-1.012).
The 2022 gap is a pure SELECTION effect -- not a cash/exposure effect.

H6/H8 beta mechanism: PARTIAL. Unique-vs-V11 picks have avg beta 1.04-1.06 (same as
V11's 1.06). Overall portfolio beta modestly higher for V28 (1.138) and V31 (1.094).
The dominant mechanism is CONCENTRATION (V28: 13.5 names, V31: 15.4) and slow rotation
vs V11 (21.6 names, 10325% AnnTO vs V28's 5264%).

Specific drivers: V28 suffers from MU (-48%) and EPAM (-49%) each at ~9% weight (shared
names, not unique). V28's unique picks are mostly 2022 winners (+14.1% avg). V31's worst
hit is NCLH (-45%) at 9% weight (unique). V26-robust is closest to V11 with 20.9 names.

Hybrid implication: a concentration constraint (position count floor ~V11's 20+) plus
turnover floor may recover more 2022 resilience than beta-dampening alone.

## Commits

- `96a22768` `feat(research): Part A -- Section-12 trade chronicling in engine + readiness runner`
- `fa11ddd` `feat(research): Part B -- 2022 bear-year attribution V28/V31/V26-robust vs V11`

## Known Issues / Remaining Work

- Attribution contribution proxy (sum of weight_T * return_T) underestimates absolute
  returns by ~2x vs registry. Cause: cost drag not modeled + whole-share residuals.
  Rankings are directionally reliable; absolute magnitudes should not be used directly.
- The scripts/scratch attribution script is gitignored; not committed to branch.
- Chronicle files (holdings/trades CSVs) are not committed; they're outputs to disk only.
- V28's MU/EPAM concentration mechanism warrants a forward test: if a position-count
  constraint (e.g., min 20 names) is applied to V28, does the 2022 gap close?

## Validation

- Suite baseline: 219 passing. After Part A: 231 passing (12 new tests, 0 regressions).
- All four variants re-ran fresh (new git_sha 96a22768, registry confirmed fresh runs).
- 2022 returns verified against registry return_streams (not from chronicle data).
