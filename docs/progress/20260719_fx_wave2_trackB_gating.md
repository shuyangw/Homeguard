# FX Wave 2 Track B Gating (#35, #37, #30) + Wave 2 Resolution - 2026-07-19

## Summary

Gated the last 3 of 6 FX Catalog Wave 2 strategies (Track B: the market-neutral
relative-value spread strategies #35 AudNzdPairs, #37 CointScanner, #30
VolRatioPair). Built the missing walk-forward + combined-gate harness for the
beta-weighted spread engine (which was already implemented and merged to
`main` prior to this session), ran all 3 gates under the strategy-lead
sentinel, and confirmed the pre-registered stopping rule resolves: all 6 Wave
2 strategies (Track A + Track B) fail the combined statistical gate. Declared
the FX catalog exhausted per the pre-registration -- recommend STOP, no Wave 3,
no ML meta-labeling harness build.

## Changes Made

- **`scripts/backtest_scripts/run_fx_spread_walkforward.py`** (new): walk-forward
  + combined statistical gate (methodology Section 2.5) harness for the spread
  strategies. Mirrors `run_fx_walkforward.py` / `run_fx_carry_seatbelt_walkforward.py`:
  rolls 36m/12m/12m non-overlapping OOS windows, re-runs the spread assembly
  per window over `[train_start, test_end]`, keeps the OOS-dated tail, stitches.
  Optimization over the naive "call run_spread_backtest twice per window":
  builds the spread book ONCE per window (cointegration/vol-ratio scan is
  cost-independent) and simulates both cost legs (1.0x, 1.5x) against it,
  avoiding a redundant second scan -- material for #37's monthly Engle-Granger
  scan over ~200 candidate pairs. Computes Sharpe/PSR/DSR (honest growing
  project-wide N via `get_campaign_trial_distribution()`)/PBO, S&P
  correlation/IR/marginal-contribution-proxy as book-level context, and
  registers exactly one `runs` row per strategy via `append_run`.
- **`scripts/backtest_scripts/run_fx_spread_backtest.py`**: added a `cost_mult`
  parameter (threaded to `FxSpreadPortfolioSimulator`), needed for the 1.5x
  cost-sensitivity leg used by the new walk-forward harness.
- Data-coverage check: verified all 6 of #30's declared legs (EURNOK, EURSEK,
  AUDUSD, NZDUSD, XAUUSD, XAGUSD) are present in the daily cache for the full
  2011-2026 range before trusting the gate, per the build review's explicit
  requirement; the harness also tracks per-window `present_universe` and the
  report would show a data-coverage note if any leg/set were ever dropped
  (none were).
- Ran all 3 gates sequentially via
  `python -m scripts.backtest_scripts.run_fx_spread_walkforward --config <cfg> --name <name>`
  under the fintech conda env with `PYTHONPATH=$(pwd)`, in strategy order
  #35 -> #37 -> #30 so the honest project-wide trial count incremented
  correctly and sequentially (109 -> 110 -> 111).
- Wrote reports: `docs/reports/fx/fx_audnzd_pairs_wave2_gate.md`,
  `docs/reports/fx/fx_coint_scanner_wave2_gate.md`,
  `docs/reports/fx/fx_vol_ratio_pair_wave2_gate.md`.
- Wrote durable results copy:
  `docs/strategies/research/20260719_fx_wave2_trackB_results.md`, and the
  combined Wave 2 resolution doc:
  `docs/strategies/research/20260719_fx_wave2_resolution.md`.
- Updated `docs/strategies/FX_60_CATALOG_TRACKER.md`: rows for #30/#35/#37
  (BT/WF/Gate columns + notes), the "Beta-weighted spread execution" unblock
  roadmap row (now BUILT), and a Wave 2 resolution note at the top of the file.
- Updated `TODO.md`: corrected Track A's #39/#42 status (they were actually
  complete from a prior session but left marked `[~]`; results docs already
  existed) to `[x]` with full verdicts, and added the Track B block (also
  `[x]` complete) with the Wave 2 resolution summary.

## Registry hygiene incident (caught and corrected before any real gate ran)

While validating the new harness end-to-end, a smoke test (a small
2011-2015/12m-6m-6m AudNzdPairs configuration used only to sanity-check window
stitching and gate math) called `walk_forward_fx_spread`, which -- as
designed -- registers a `runs` row via `append_run`. This wrote a bogus,
non-pre-registered row to `output/experiments.duckdb` (run_id
`21093b52-ffd0-4a75-b2b2-1fb2dfdc38ac`). Caught immediately after the smoke
test's own printed output showed a `run_id`; deleted the row directly via
DuckDB (`DELETE FROM runs WHERE run_id = '...'`) BEFORE any real Track B gate
ran, and re-confirmed the honest project-wide trial count afterward. No real
gate's DSR was computed against the contaminated count. For future
harness-validation smoke tests, call the internal per-window worker
(`_run_window_spread`, which never touches the registry) directly instead of
the full gate function.

## Verdicts

| # | Strategy | OOS Sharpe (1x/1.5x) | PSR | DSR | PBO | N | S&P corr | Verdict |
|---|---|---|---|---|---|---|---|---|
| 35 | AUD/NZD pairs | -0.24 / -0.30 | 0.00 | 0.00 | 0.82 | 109 | 0.04 | REJECT |
| 37 | Cointegration scanner | -0.24 / -0.31 | 0.00 | 0.00 | 0.45 | 110 | -0.01 | REJECT |
| 30 | Vol-ratio pair (XAU/XAG) | -0.48 / -0.54 | 0.00 | 0.00 | 0.43 | 111 | 0.14 | REJECT |

All 3 decisively fail (negative OOS Sharpe at 1x, DSR exactly 0.0000 -- not
"genuinely close" per the pre-registered stopping rule).

## WAVE 2 RESOLUTION

All 6 Wave 2 strategies (#33/#39/#42 Track A + #35/#37/#30 Track B) fail the
combined statistical gate. Per the pre-registered stopping rule (Section 6,
`docs/superpowers/specs/2026-07-19-fx-wave2-selection-design.md`), the campaign
has now tested 8+ distinct mechanisms across Wave 1 + Wave 2, all failing net
of realistic costs. **Declared: the retail G10 FX catalog is exhausted under
this cost regime. STOP -- no Wave 3, no ML meta-labeling harness build
(#48-53).** Full resolution and reasoning:
`docs/strategies/research/20260719_fx_wave2_resolution.md`.

## Commits

See `git log` for the commit(s) made in this session (new walk-forward
harness, `cost_mult` param, reports, tracker, results docs, resolution doc,
TODO.md, this session log).

## Known Issues / Remaining Work

- **Registry duplicate rows (pre-existing, not from this session):** the
  registry contains exact-duplicate rows for `FxRoroRegimeSpread` and
  `FxPcaDollarResidual` (2 each, from Track A) and `FxCarrySeatbelt` (4, from
  an earlier session). These bias the honest trial count N upward (harder
  gate), the safe direction per the North Star, so left uncorrected. A future
  hygiene pass could add idempotency/dedup-on-identical-spec to
  `append_run` call sites.
- **Deferred, not tested:** #36 Scandi triangle (needs Brent oil data on top
  of the now-built spread engine) and #40 correlation-breakdown (partial
  spread dependency) were explicitly out of Wave 2's scope and remain
  untested. Per the resolution doc, the stopping rule's judgment is that 8+
  already-tested mechanisms is decisive; further neighbors in the same
  exhausted style space are not automatically justified.
- **Next step (for the user/orchestrator, not auto-actioned):** per the
  resolution, redirect research effort away from the retail G10 FX catalog
  (no Wave 3) rather than continuing to search the same style space.

## Validation

- Both cost legs and the full 36m/12m/12m walk-forward ran cleanly for all 3
  strategies (confirmed via captured stdout, not just exit code); each
  produced 13 usable OOS windows and ~3,180-3,200 stitched OOS days.
- Registry integrity verified by direct DuckDB query against
  `output/experiments.duckdb` (`runs` table): exactly one
  `fx-spread-walkforward` row per strategy, no duplicates from this session,
  honest trial count grew monotonically 109 -> 110 -> 111 (one increment per
  gate call).
- Data-coverage verified directly (cache probe + harness per-window tracking)
  for #30's 6 legs before trusting its verdict, per the build review's
  explicit requirement.
- `.claude/.strategy-lead-active` sentinel created at session start and
  removed at session end.
