# FX Fill-Logging Revalidation + Generic-Runner Boundary Dedup - 2026-07-20

## Summary
Revalidated the OOS-fill-slicing fix on a second strategy/runner (FxTrend via the
generic `run_fx_walkforward.py`) and, in doing so, found a pre-existing latent bug
in that runner's OOS return stitching: it used `np.concatenate` with no dedup of
the calendar day shared by adjacent windows (`test_end_N == test_start_{N+1}`),
unlike the seatbelt runner which dedups. Fixed it by adding a tested
`_stitch_oos_dedup` helper and wiring the generic runner to it. Merged to main
(FF) and pushed. Also cleaned up demo scratch run dirs.

## Revalidation findings (all independently verified against the bytes)
- **FxTrend (generic runner) fills are clean post-fix.** `trades_oos` = 1234 rows,
  span 2021-01-04..2024-01-01, 0 pre-test_start rows, 0 duplicate `(date,pair)`;
  per-window files still full (2018-2020 starts); my independent OOS-slice
  reconstruction (394+416+424) matched 1234 exactly. The fix generalizes.
- **The vectorbt `WalkForwardValidator.validate()` path is DORMANT** -- the only
  `.validate(` caller in the codebase is an unrelated discord config check.
  Nothing invokes it with a `fill_sink`, so the Task-10/11 optimizer-probe
  logging instruments a path nothing runs (no live risk, but also logs nothing
  in practice).
- **The generic runner's boundary double-count does NOT manifest for year-aligned
  windows.** Reported `n_oos_days=781`; sum of per-window inclusive OOS days = 781;
  unique = 781; shared boundary trading days = []. Reason: the boundaries are
  Jan-1, always a market holiday, so adjacent windows share no trading day. The
  bug only bites if a boundary lands on a trading day (non-year-aligned config).
- The demo run was registry-safe: `append_run` monkeypatched to a no-op, `run_id`
  None, `experiments.duckdb` md5 unchanged -- no trial appended (catalog stays
  CLOSED).

## Changes Made
- **src/backtesting/walkforward_common.py**: added `_stitch_oos_dedup(per_window:
  List[pd.Series]) -> np.ndarray` -- concat, stable sort, drop
  `index.duplicated(keep="first")`, return bare float array. Mirrors the seatbelt
  runner's inline dedup.
- **scripts/backtest_scripts/run_fx_walkforward.py**: `process_window` now returns
  DATED OOS series (`_oos_returns_dated`); `walk_forward_fx` stitches via
  `_stitch_oos_dedup`; every numeric consumer (`_annualized_sharpe`,
  `_compute_pbo`) receives `.to_numpy(dtype=float)` so numerics are byte-identical
  for the no-shared-boundary case and only the deduped-stitch changes when a
  boundary trading day is shared. Seatbelt runner untouched.
- **tests/backtesting/test_walkforward_common.py** (new): 3 tests -- shared
  boundary keep-first, disjoint keeps-all-sorted, empty.
- Deleted demo scratch dirs `output/backtests/{FxCarrySeatbelt,FxTrend}/runs/`
  (untracked; confirmed no tracked files before removal).

## Commits (feat/fx-generic-boundary-dedup, FF-merged to main; base d384faf)
- `07e1894` fix(fx): generic FX walk-forward dedups shared boundary day in
  stitched OOS returns (helper + 3 tests + runner wiring)

## Known Issues / Remaining Work
- **`scripts/backtest_scripts/run_carver_walkforward.py` has the SAME latent
  boundary double-count** (bare `_oos_returns` + `np.concatenate`, no dedup) --
  found by the review, out of scope here, worth the same `_stitch_oos_dedup` fix.
- Still open from prior features: futures walk-forward runner not sink-wired;
  `GridSearchOptimizer.optimize_parallel` probes unlogged; vectorbt validate path
  dormant; `strategy_lead_gate` hook substring-matches filenames/commit messages
  (repeated false positives this session).

## Validation
- `tests/backtesting/test_walkforward_common.py` 3/3 pass (RED first). py_compile
  clean on runner + walkforward_common. Whole-branch/opus review: PASS,
  merge-ready, 0 Critical/0 Important; every numeric consumer confirmed to receive
  a numpy array (no ddof drift), no-shared-boundary equivalence confirmed
  byte-identical, keep-first matches seatbelt, seatbelt untouched.
- No real backtest run for the fix (unit-tested helper proves the dedup; the
  double-count is calendar-dependent and does not manifest for year-aligned
  windows, so a real re-run would show no change).
- Merged via fast-forward ref-update (no `checkout`); pushed to origin/main =
  07e1894.
