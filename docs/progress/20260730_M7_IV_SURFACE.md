# M7 Smoothed IV Surface -- 2026-07-30

## Summary

Built M7, the per-expiry raw-SVI implied-volatility surface over
`options_chain_eod`, unblocking Wave 1 (OPT-047 and the D-047/030 integrity
diagnostic). Materialized across 100% of buildable scope -- 270 partitions, SPY
2017-01..2025-12 and QQQ 2012-06..2025-12, zero failures. **The registered
D-047/030 gate FAILS on both roots**; the registered consequence (OPT-047/030
need a different mark source) is applied as written. This was a data/model
build: no backtest, no P&L, zero trials consumed.

## Changes Made

- **`src/data/options/iv_surface.py`** (new): raw-SVI fitter in total variance
  against `k = log(K/F)`. Parity-implied forward, FRED discount curve, weighted
  fit, closed-form Jacobian, vectorized Black-76 IV inversion, Gatheral
  butterfly / calendar checks, and nine refusal reason codes.
- **`src/data/options/iv_surface_build.py`** (new): materializes two
  hive-partitioned join tables, `options_iv_surface` (session x expiry: params,
  forward, diagnostics, reason) and `options_iv_smooth` (contract x session:
  `iv_smooth`, `delta_smooth`, provenance, extrapolation flag).
  `options_chain_eod` is not mutated.
- **`scripts/data/build_m7_surface.py`** (new): parallel driver with
  `--jobs`/`--shard`/`--overwrite`, `RunStatus`-wrapped.
- **`scripts/data/validate_m7_surface.py`** (new): the four registered checks
  plus the D-047/030 gate and a supplementary strike-selection diagnostic.
- **`tests/data/test_options/test_iv_surface.py`** (new): 32 tests, TDD,
  including negative controls (an arbitrageable slice must FAIL the butterfly
  check; a decreasing-total-variance pair must FAIL the calendar check).
- **`docs/architecture/infra_patterns.md`**: registered the M7 tables and
  recorded the two `OptionsDataLoader` footguns.

## Key decisions

- **Raw SVI over a spline**, because SVI's wings are linear in `k` by
  construction with a Lee-bounded slope. A spline's wing is governed by knot
  placement and can manufacture unbounded curvature at exactly the 0.05-delta
  strikes OPT-047 reads.
- **Dividend-awareness via the put-call-parity forward, NOT a yfinance dividend
  series.** The task brief proposed fetching realized dividends; using a
  later-paid dividend to build an as-of-`t` forward is lookahead and violates
  spec v2 Section 1.3. The parity forward is point-in-time and validates itself:
  implied `q` converges to +1.09..+1.20% on SPY Jan-2024 against an actual
  yield of ~1.3%. No new data source was added.
- **Discount curve** reuses `src/data/rates/fred_reader.py` (DGS1MO..DGS2,
  already on disk). Found two places hardcoding `RISK_FREE_RATE = 0.05`
  (`thetadata_adapter.py:59`, `csp/mark_to_market.py:48,75`) -- flagged, not
  changed.
- **"Diverges materially" was undefined** in the entire doc chain. Operationalized
  and committed to git BEFORE the first fit ran, and flagged as a researcher
  degree of freedom.

## Commits

- `6271424` docs(m7): pre-register parameterization, refusals, D-047/030 threshold
- `d4457e5` feat(m7): raw-SVI fitter with refusal codes
- `be1326f` feat(m7): materializer, parallel driver, vectorized inversion
- `7f661b3` feat(m7): validation battery
- `328e402` test(m7): move tests under tests/data/test_options/
- `922ed00` docs(architecture): register the M7 tables
- `06d01ad` fix(m7): do not depend on caller row ordering
- `eb7445b` perf(m7): make the validation battery scale to the full store
- `288fe20` feat(m7): calendar violations by DTE and vol-point size
- `a718509` feat(m7): strike-selection agreement diagnostic
- `65db919` fix(m7): structural refusals outrank forward refusals in the reason code
- `5b642e2` docs(m7): build + validation report

## Results

- Coverage: **SPY 84.65%**, **QQQ 74.58%** of session-expiry slices fitted.
  Best where the slate lives (SPY 61-180 DTE: 98%+). Worst in early QQQ
  (2012-2016: 41.7-54.4%) because those chains are genuinely thin -- median 22
  valid contracts per session-expiry vs SPY's 103.
- Fit residuals: median **0.08-0.25 vol points**, essentially unbiased. But the
  `|delta| < 0.05` bucket has a p99 of **17-24 vol points** and ~31%
  extrapolated contracts.
- Butterfly violations: **0.02-0.04%** near ATM, **1.3-1.8%** in the deepest
  OTM bucket. Not repaired.
- Calendar: **33-36%** of adjacent expiry pairs cross somewhere -- 57%/55% at
  <= 7 DTE falling to **17%/10% at 91+ DTE** (worst ~1.7 vol points). M7 fits
  expiries independently, so nothing enforces calendar consistency.
- Stability: ATM is stable and its >5-vol-point jumps coincide with a median
  2.61% underlying move (vs 0.58% baseline) -- real vol events. The wing is
  not: 10.6% of sessions jump >5 vol points with only a 0.71% median move.
- **D-047/030 gate: FAIL both roots.** Inside-band 27.5% (SPY) / 55.1% (QQQ) vs
  a registered 90% floor; median divergence 0.154 / 0.122 vol points vs a 1.5
  ceiling. The median quoted IV band at 0.05 delta is only 0.09-0.35 vol points
  -- tighter than a 5-parameter fit can thread.

## Known Issues / Remaining Work

- **OPT-006 is blocked**: its 12-18 month long leg exceeds the registered
  400-day `MAX_DTE` cap, so 1,873 QQQ LEAPS slices have no surface. Needs an
  explicit registered amendment, not a silent widening.
- **`IMPLAUSIBLE_FORWARD` is mis-calibrated at short DTE** -- the implied-`q`
  bound is `1/T`-amplified noise below ~7 DTE. Registered blind, deliberately
  NOT retuned. Should be re-registered as a DTE-aware bound on `F/S`.
- **`ARB_VIOLATION` is 14-16% of SPY 1-30 DTE slices** -- raw SVI cannot
  represent some steep short-dated smiles arbitrage-free. SSVI with a
  calendar-consistent `theta` would recover some.
- No calendar coupling across expiries; no cross-vendor validation (ORATS
  deferred); single names out of scope pending the V7 split-adjustment layer.
- **`D-037` should be re-run** now that M7 exists (wave-0 flagged this).
- `20260728_options_wave0_diagnostics.md` is **not committed to `main`** -- it
  exists only in the shared working tree. Someone should commit it.
- A stale `RUNNING` sentinel remains at
  `output/run_status/m7_surface_shard0of2_20260730_083649.json` from a shard I
  deliberately killed to restart with corrected reason codes. That is
  `RunStatus` behaving as designed.

## Validation

- 32 new tests, all passing; `tests/data/test_options/` 109 passing;
  `tests/data/test_options/ + tests/backtesting/vol/` 120 passing. No regressions.
- Two performance rewrites (vectorized IV inversion, analytic Jacobian) were
  each verified to leave results identical before adoption; the defensive
  `k`-sort was verified a bit-exact no-op (max delta 0.0 across 11 param
  columns) against already-built data.
- Full store rebuilt after the reason-code precedence fix: 270/270 partitions,
  0 failures, verified no stale files remain.
- Validation artifacts: `docs/strategies/research/options-slate/m7_validation/`
  and `output/m7_validation/`.
