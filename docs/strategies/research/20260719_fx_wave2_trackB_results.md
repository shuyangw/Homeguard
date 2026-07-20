# FX Catalog Wave 2, Track B -- Results (#35, #37, #30)

**Date:** 2026-07-19
**Pre-registration:** `docs/superpowers/specs/2026-07-19-fx-wave2-selection-design.md`
(Wave 2 roadmap + pre-registered stopping rule) and
`docs/superpowers/specs/2026-07-19-fx-spread-engine-design.md` (Track B engine +
strategy design).
**Generator:** `scripts/backtest_scripts/run_fx_spread_walkforward.py` (new; built
this session, mirrors `run_fx_walkforward.py` / `run_fx_carry_seatbelt_walkforward.py`).
**Gate:** methodology Section 2.5 combined statistical gate (PSR>=0.95, DSR>=0.95
using the honest project-wide growing trial count, PBO<0.25, 1.5x cost-sensitivity
survival). S&P correlation / IR / marginal contribution are book-level context,
non-gating, per the Wave 2 pre-registration.

## Summary

| # | Strategy | OOS Sharpe (1x) | OOS Sharpe (1.5x) | PSR | DSR | PBO | Trials (N) | S&P corr | Verdict |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 35 | AUD/NZD pairs | -0.2362 | -0.3016 | 0.0000 | 0.0000 | 0.8242 | 109 | 0.0436 | REJECT |
| 37 | Cointegration scanner | -0.2381 | -0.3078 | 0.0000 | 0.0000 | 0.4517 | 110 | -0.0075 | REJECT |
| 30 | Vol-ratio pair (XAU/XAG) | -0.4795 | -0.5392 | 0.0000 | 0.0000 | 0.4320 | 111 | 0.1433 | REJECT |

**All 3 Track B strategies decisively REJECT** -- every one has a negative OOS
Sharpe at 1x cost (which only worsens at 1.5x), so DSR is exactly 0.0000 for all
three; none is a marginal or cost-fragile case, all are outright unprofitable
out-of-sample. Full reports: `docs/reports/fx/fx_audnzd_pairs_wave2_gate.md`,
`docs/reports/fx/fx_coint_scanner_wave2_gate.md`,
`docs/reports/fx/fx_vol_ratio_pair_wave2_gate.md`.

Walk-forward design (all 3): 36m/12m/12m rolling OOS windows, 2011-01-01 to
2026-04-01 (full available data range, matching Track A and the rest of the FX
campaign), 13 windows each, ~3,180-3,200 OOS trading days. Rebalance weekly,
vol_target 0.10, $100,000 initial capital, leverage_cap 4.0x (matches each
strategy's pre-registered config).

## Infrastructure built this session

Per the Track B design spec, gating these 3 strategies required a beta-weighted
2-leg spread-execution engine that did not exist before this campaign
(`FxSpreadPortfolioSimulator`, the 3 strategy implementations, and their configs
were already merged to `main` prior to this session -- see spec Section 3). What
was missing and built THIS session, per the brief's explicit instruction to
strategy-lead ("Build a walk-forward around this"):

- **`scripts/backtest_scripts/run_fx_spread_walkforward.py`** (new): the
  walk-forward + combined-gate harness for spread strategies, mirroring
  `run_fx_walkforward.py` / `run_fx_carry_seatbelt_walkforward.py`. Rolls
  non-overlapping 36m/12m/12m OOS windows; each window re-runs the spread
  assembly over `[train_start, test_end]` (train segment is signal warm-up
  only), keeps the OOS-dated tail, and stitches across windows. Both cost legs
  (1.0x, 1.5x) share ONE spread-book computation per window (the book --
  cointegration/vol-ratio scan and entry/exit state machine -- is
  cost-independent; only the simulator's `cost_mult` differs), avoiding a
  redundant second scan for the cost-sensitivity leg. Computes Sharpe / PSR /
  DSR (honest growing project-wide N) / PBO on the stitched 1x series, S&P
  correlation / IR / marginal-contribution-proxy as book-level context, and
  registers exactly one `runs` row per strategy via `append_run`.
- **`scripts/backtest_scripts/run_fx_spread_backtest.py`**: added a `cost_mult`
  parameter (threaded through to `FxSpreadPortfolioSimulator`), needed for the
  1.5x cost-sensitivity leg. The walk-forward harness does not call this
  function directly (it inlines the same book-once/simulate-twice logic for
  performance -- see below) but the parameter is also useful standalone.

**Correctness validation before any real gate ran:** smoke-tested the new
harness end-to-end on a small 2011-2015 / 12m-6m-6m AudNzdPairs configuration
(sanity-checking window stitching, Sharpe/PSR/DSR/PBO computation, and registry
append). That smoke run DID call `append_run` and wrote a bogus row to
`output/experiments.duckdb` (run_id `21093b52-ffd0-4a75-b2b2-1fb2dfdc38ac`,
strategy `AudNzdPairs`, non-standard windowing) -- this was NOT a pre-registered
specification and was deleted from the registry immediately after being
identified, before any real Track B gate ran, so it does not contaminate the
honest trial count used below. Lesson applied for future harness-smoke-testing:
call the internal per-window worker (`_run_window_spread`, which never touches
the registry) directly for wiring checks, not the full gate function.

**Runtime:** all 3 strategies ran in well under a minute each (AudNzdPairs
~0.1s/window, VolRatioPair ~0.3s/window, CointScanner ~9s/window for the
heaviest single-window timing probe used to size the run before committing to
it) -- far below the multi-hour budget flagged as a risk in the design review.

## #35 AUD/NZD pairs -- REJECT

120-day rolling OLS residual-z spread on AUDUSD/NZDUSD, entry `|z|>2`, exit
`z<0.5` (target) / `|z|>3.25` (stop) / 20 days (time), RBA/RBNZ blackout. OOS
Sharpe -0.24 (1x) / -0.30 (1.5x), PSR/DSR 0.0000, PBO 0.82 (badly overfit --
window ranking is unstable, consistent with a mechanism with no real edge), S&P
corr 0.04 (near-zero, as expected for a beta-weighted market-neutral spread).
Non-positive OOS Sharpe: no edge to deflate or gate.

## #37 Cointegration scanner -- REJECT

Monthly Engle-Granger scan over all non-triangular pairs among the 22-pair G10
universe (`cointegration.test_pair`, ADF p<0.05, OU half-life in [5,25]d, spread
vol clearing 2x round-trip cost at 1.5-sigma), top-5 tradeable set, entry
`|z|>2`, exit `z<0.5` / `|z|>3.5` (stop) / `2*half_life` (time) / structural
ADF-degradation (rolling p-value worse than baseline by >0.2 for 10 consecutive
days). Gated the FIXED pre-registered config only -- no optimizer round, per
the build's whole-branch review note that #37's free-parameter budget
(entry_z/target_z/stop_z) is already at the methodology 5.4 cap of 3; the scan
mechanics (scan_window/half_life_range/adf_max/top_n) were NOT varied. OOS
Sharpe -0.24 (1x) / -0.31 (1.5x), PSR/DSR 0.0000, PBO 0.45, S&P corr -0.01.
Non-positive OOS Sharpe: no edge to deflate or gate.

**Scan warm-up floor confirmed satisfied by construction:** the review flagged
that the strategy's `_MIN_SCAN_DAYS=30` floor is far short of its intended
`scan_window=250` day trailing window, so early-window short-history ADF
selections could leak into a graded OOS window if a walk-forward window's
`test_start` fell too close to that window's own `train_start`. With the
standard 36-month training segment (~756 trading days) preceding every window's
`test_start`, this is never binding -- by the time any window's OOS period
begins, its own re-run already has 3x the 250-day scan window of history. No
special handling was needed beyond the standard windowing.

## #30 Vol-ratio pair (XAU/XAG) -- REJECT

Weekly symmetric vol-ratio reversion on 3 coupled sets ({EURNOK,EURSEK},
{AUDUSD,NZDUSD}, {XAUUSD,XAGUSD}): z-score of `ln(RV_10d(A)/RV_10d(B))` vs its
trailing 2-year distribution, entry `|z|>2` (short high-vol leg / long low-vol
leg, beta-weighted), exit `|z|<1`. **All 6 declared legs (EURNOK, EURSEK,
AUDUSD, NZDUSD, XAUUSD, XAGUSD) were confirmed present in the daily cache for
the full 2011-2026 range AND in every walk-forward window** (verified both via
a direct full-range cache probe before running the gate, and via the harness's
own per-window `present_universe` tracking -- the report shows no
"data-coverage note", meaning no coupled set was ever dropped). The verdict is
on all 3 coupled sets, not a subset. OOS Sharpe -0.48 (1x) / -0.54 (1.5x),
PSR/DSR 0.0000, PBO 0.43, S&P corr 0.14. Non-positive OOS Sharpe (the worst of
the 3 Track B strategies): no edge to deflate or gate.

## Trial-count integrity check

Verified against `output/experiments.duckdb`: exactly one registered
`fx-spread-walkforward` run per strategy (`AudNzdPairs`, `CointScanner`,
`VolRatioPair` -- one `run_id` each, no duplicates). The honest project-wide
trial count grew monotonically and by exactly one per gate call: 109 -> 110 ->
111 (continuing from Track A's N=106 plus 3 additional registry-logged trials
from unrelated intervening FX campaign work between Track A's close and this
session -- the mechanism counts every registry row with a numeric `oos_sharpe`,
which is the documented, honest, growing-search behavior; it does not need to
match a manually-tracked prior expectation). Per the North Star and the Wave 2
pre-registration Section 5, this growing N is the load-bearing protection
behind the DSR gate.

**Non-blocking hygiene observation (not corrected, out of scope for this
session):** the registry also contains exact-duplicate rows for two of Track
A's own strategies (`FxRoroRegimeSpread` and `FxPcaDollarResidual` each have 2
rows with the identical `oos_sharpe`, apparently from a re-run during Track A's
session) and 4 rows for `FxCarrySeatbelt` from an earlier session. These
duplicates bias N upward (a harder gate), which is the safe direction per the
North Star ("never compute a gate over an undercounted search"), so they were
left as-is rather than retroactively edited -- deleting historical rows without
being certain which is the "true" one risks a worse error (undercounting).
Flagged for a future hygiene pass on `append_run` call sites to add
idempotency/dedup-on-identical-spec.

## Does Track B trigger the pre-registered stopping rule?

**Yes.** All 3 Track B strategies decisively FAIL (negative OOS Sharpe, DSR
0.0000, none within any reasonable distance of the "genuinely close" bar --
which requires a meaningfully POSITIVE deflated Sharpe). Combined with Track
A's 3/3 FAIL (`docs/strategies/research/20260719_fx_wave2_trackA_results.md`),
**all 6 Wave 2 strategies fail the combined statistical gate**, resolving the
pre-registered stopping rule. See
`docs/strategies/research/20260719_fx_wave2_resolution.md` for the full Wave 2
resolution statement.
