# FX Walk-Forward OOS Fill Slicing - 2026-07-20

## Summary
Fixed a data-contamination defect found by validating the fill-logging-everywhere
demo: `trades_oos.csv.gz` for the FX walk-forward runners was the concatenation
of each window's FULL `[train_start, test_end]` fills, so it was ~39% in-sample
plus 502 duplicate rows from overlapping windows -- not the out-of-sample fills
its docs claimed. The fix slices each window's logged fills to its
`[test_start, test_end)` OOS range before concatenating, while leaving the
per-window files full-window on disk. Verified on a real re-run: `trades_oos`
dropped from 2237 rows (864 in-sample, 502 dup) to 1099 rows (0 in-sample, 0
dup), matching the gated OOS window. Merged to main (FF) and pushed.

## How the defect was found
The fill-logging-everywhere feature (merged earlier today, 630f622) shipped
`trades_oos.csv.gz` documented as "the actual gated-verdict fills." A validation
of the demo run (`ultrathink` request) showed it started at 2018-02-27 -- but the
36m-train/12m-test walk-forward's OOS begins ~2021. Independent inspection with a
scratch script quantified it: 864/2237 rows (39%) dated before 2021-01-01
(guaranteed in-sample), 502 duplicate `(date, pair)` rows from adjacent windows
sharing 3 of 4 years, and `leverage_utilization` spans confirming each per-window
run was the full 4-year `[train_start, test_end]`. Every per-task and whole-branch
review had verified the MECHANICAL concat but not the SEMANTIC (that the content
was OOS) -- the fills were structurally valid, just the wrong subset.

## Changes Made
- **src/backtesting/engine/fill_sink.py**: added `self._oos_ranges` +
  `set_oos_range(window, start, end)`; `finalize` now slices each window file to
  its recorded `[test_start, test_end)` on the `date` column (half-open per
  window; inclusive only at the single global-max `test_end`, so the last OOS day
  is retained and no boundary day is double-counted) before building
  `trades_oos.csv.gz`. No range recorded, or no `date` column -> no slicing
  (back-compat: existing `finalize` tests, the vectorbt validator path, and the
  sweep path are all unaffected). Per-window files on disk are never rewritten.
- **scripts/backtest_scripts/run_fx_carry_seatbelt_walkforward.py** and
  **run_fx_walkforward.py**: each records `set_oos_range(window, test_start,
  test_end)` for every window before `sink.finalize`.
- **Docs** (.claude/rules/strategy-pipeline.md, docs/methodology/backtesting.md
  Section 12, .claude/agents/strategy-lead.md): clarified that per-window
  `wNN_<leg>_trades.csv.gz` files are full-window runs and `trades_oos.csv.gz` is
  the OOS-sliced concat -- so the prior "trades_oos = gated fills" claim is now
  accurate.

## Design decisions
- **Option A** (user choice): keep full-window per-window files (honors "log every
  simulated run"); slice only the concat.
- Slice in `finalize` driven by runner-recorded ranges (keeps slicing generic,
  date-column based; runner change is one loop each).
- Boundary: half-open `[test_start, test_end)`, global-max end inclusive ->
  non-overlapping, no `(date,pair)` dedup needed in the generic sink.
- Known, user-approved consequence (Minor): the carry runner dedups its OOS
  RETURN series keep-first (earlier window owns the shared boundary day) while the
  fill slice attributes it to the later window -- one diagnostic day per boundary,
  no gate impact (the gate runs on the untouched return series).

## Commits (feat/fx-oos-fill-slicing, FF-merged to main; base 6714322)
- `a0464b2` FillSink slices trades_oos to per-window OOS range (+ 4 tests)
- `92d0d31` both WF runners record per-window OOS range
- `ebf7dfe` docs clarification
- spec `d60f3b9`, plan `6714322` (on main pre-branch)

## Known Issues / Remaining Work
- Cosmetic: one test comment says "half-open" though the single-window case is
  effectively inclusive (no behavior impact).
- Carried over from the parent feature (still open, out of this branch): futures
  walk-forward runner not sink-wired; `GridSearchOptimizer.optimize_parallel`
  probes unlogged; `run_fx_wave2_gate.py` untracked/unwired (verdict path covered
  via `walk_forward_fx`); `strategy_lead_gate` hook substring-matches
  filenames/commit messages (false positives worked around with `git commit -F`).

## Validation
- FillSink suite 16/16 (12 pre-existing + 4 new). Whole-branch review (opus):
  READY TO MERGE, 0 Critical / 0 Important; all 6 cross-task integration checks
  confirmed (OOS-only + non-overlapping, back-compat, per-window files untouched,
  manifest consistent, boundary edge, docs accurate).
- End-to-end re-run via strategy-lead (demonstration-only, register=False, no
  verdict) + independent scratch-script check on the fresh artifacts
  (`output/backtests/FxCarrySeatbelt/runs/20260721T030437Z_ec42e6/`):
  `trades_oos` = 1099 rows, 0 pre-2021 rows, 0 duplicate `(date,pair)`, all rows
  within `[2021-01-01, 2024-01-01]`, and my independent OOS-slice reconstruction
  summed to exactly 1099. Per-window c1x files still span 2018-2023 (full).
  Manifest `oos_concat` count matches the file.
- Merged via fast-forward ref-update (no `checkout`, per the macOS/Dropbox git
  hazard); pushed to origin/main = ebf7dfe.
