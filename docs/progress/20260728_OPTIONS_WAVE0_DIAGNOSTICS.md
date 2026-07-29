# Options Slate Wave 0 Diagnostics - 2026-07-28

## Summary

Executed Wave 0 of the equity-options slate: 18 data-property diagnostics (Group A on
underlying bars, Group B on the derived option-chain tables) against the gates
pre-registered in handoff spec v2 Section 5. **Zero trials consumed** -- no strategy
backtest, no P&L, no positions or fills. Five candidates were killed outright, two are
blocked on missing infrastructure, four cleared, two split, and the rest are descriptive
framing. Materializing the EOD chain surfaced two apparatus bugs and three previously
undocumented data boundaries.

## Changes Made

- **`src/data/options/canonical.py`**: `canonicalize_frame` assumed `expiration` was
  String; 100 of 4,510 partitions ship it as `date32` -- including **SPY 2017-2018**, the
  exact window the slate is bounded to -- so materialization raised `SchemaError`. Added
  `_expiry_expr()` accepting both, failing loud otherwise. Apparatus correction.
- **`src/utils/run_status.py`**: status destination path was `<name>_<YYYYmmdd_HHMMSS>.json`,
  second-resolution only, so parallel workers launched in the same second shared one
  destination and their atomic `replace()` calls fought (WinError 5). **Killed 9 of 276
  parallel build jobs.** The 2026-07-27 fix had made only the *tmp* path unique. Added pid
  plus a short random suffix.
- **`src/backtesting/diagnostics/`** (new package): `session_bars.py` (session marks +
  `session_close_schedule()`), `wave0_group_a.py`, `options_iv_state.py`,
  `options_chain_structure.py`.
- **`scripts/data/build_eod_chain_parallel.sh`** (new): parallel driver for
  `options_chain_eod` materialization, 8 jobs, resumable (skips existing).
- **`scripts/backtest_scripts/`**: four Wave 0 drivers (force-added; dir is gitignored).
- **Data**: materialized `options_chain_eod` for **SPY 2017-2025 (108) + QQQ 2012-2025
  (162) = 270 partitions**, 100% of what is buildable. It previously held one
  proof-of-concept partition (SPY 2024-01).
- **Deliverable**: `docs/strategies/research/options-slate/20260728_options_wave0_diagnostics.md`

## Commits

- `ebb5d06` fix(options): canonicalize accepts date32 `expiration` variant
- `1f85516` test+feat(diagnostics): Wave-0 Group A data-property measurement primitives
- `32d01db` feat(diagnostics): Wave-0 Group A driver + persisted measurement artifacts
- `a54a281` fix(run_status): unique status path per run, not per second
- `b73bd2e` feat(options): Wave 0 diagnostics -- Groups A and B, zero trials

## Key Decisions

- **D-047/030 reported BLOCKED rather than proxied.** The registered diagnostic compares a
  *smoothed* surface against raw quotes; module M7 (`iv_smooth`) does not exist anywhere in
  the repo and ORATS was deferred 2026-07-27. A raw-quote census was produced but labelled
  explicitly as NOT the gate. **This puts M7 on the Wave-1 critical path** -- OPT-047 is a
  Wave-1 candidate whose 0.05-delta strikes trigger P1's low-delta rule.
- **D-011 reported BLOCKED.** Its registered universe (bottom-decile momentum S&P 500 single
  names) is not on disk; the ThetaData top-up route is dead and ORATS deferred. An
  index-level analogue was measured but labelled supplementary, not the census.
- **D-029 passed a fortiori.** Registered universe is "10 megacaps + SPY" but only SPY/QQQ
  are materialized; SPY alone clears 6 entries/yr, and adding megacaps can only add entries,
  so the PASS is valid for the registered universe. A SPY-only shortfall would have been
  INCONCLUSIVE, not FAIL.
- **Splits reported rather than resolved.** D-048 (SPY KEEP 0.2337 / QQQ DROP 0.1936 against
  a 0.20 bar fixed in advance) and D-049 (22 episodes on the literal next-day read PASS / 12
  on the horizon-matched read FAIL) both straddle their gates. Neither was resolved by taking
  the passing read.
- **"post-2023" computed both ways** (2023+ and 2024+) with the gate applied to each, chosen
  before seeing numbers, since the source docs never defined it.
- **Gate read on QQQ for D-033** because it is the only root with full-history usable greeks,
  which is the window the gate text names. SPY's count is a strict undercount.

## Known Issues / Remaining Work

- **M7 (smoothed IV surface) unbuilt** -- blocks D-047/030 and gates OPT-047/030/037's
  low-delta marks. Decision logged 2026-07-27 was BUILD. Wave-1 critical path.
- **D-037 must be re-run once M7 lands**; 93.5% of its far-wing selections sit below the
  0.10 delta floor where P1 mandates a smoothed surface.
- **Float dtype inconsistency** across `options_chain_eod` partitions (Float32 vs Float64)
  breaks naive multi-month `pl.concat`. Worked around with a widening cast; should be
  normalized in the canonical writer.
- **Honest lifetime `N` still unreconstructed** (`combinations_project` in
  `output/experiments.duckdb` is empty; directionally low hundreds, not 9). Blocks Wave 1
  *grading*, not building. Wave 0 spent no trials so N is unchanged.
- **Megacap EOD chains unbuilt** -- D-011 blocked, D-029/D-043 ran on a narrowed universe.
- **D-043 is structurally confounded** and cannot be de-confounded: OpEx-week DTE mean 1.97
  vs control 17.25, and a DTE-matched control has n=0 by construction. Verdict stands (the
  entry-day reading is a flat null) but the instrument is weaker than Section 5 assumed.

## Data boundaries discovered (not in the doc chain)

- **SPY 2017 H1 is effectively absent** from `options_combined`: 107 of 2,262 calendar
  sessions missing, ~90 in 2017 H1 (2017-01 holds 2 of 20 sessions). Partitions exist and
  their greeks are `OK` -- the sessions simply are not in them. Delays SPY's first valid 2y
  IV percentile to 2019-06-05 and cost D-050b 62 of 167 trigger events.
- **QQQ 2015-11 partition does not exist** -- a genuine hole. QQQ has 162 of 163 possible
  partitions, not the 168 assumed.
- **Underlying 1m bars (2016+) bind QQQ harder than the greeks boundary** on any diagnostic
  needing realized vol (D-005, D-037, D-043).
- **Pre-2015 monthly expiries are dated SATURDAY**, not Friday (OCC change Feb 2015). A
  Friday-only rule silently discarded all QQQ 2012-2013 term-slope rows.

## Validation

- **99 tests pass** across `tests/backtesting/test_diagnostics/`, `tests/utils/test_run_status.py`,
  `tests/data/test_options/test_canonical.py` (fintech env).
- Both apparatus fixes landed with regression tests written first (TDD).
- Headline gate numbers were **re-read directly from the persisted artifacts**, not taken
  from subagent summaries: D-040b ratios, D-043 direction/medians, D-014 per-slice nets.
- EOD chain coverage verified programmatically against the source store (270/270 buildable;
  all 6 "failures" confirmed as genuine source gaps, not code faults).
- All numeric outputs persisted under `output/wave0/{groupA,groupB1,groupB2,derived}/` so
  every result is re-readable without a re-run.
