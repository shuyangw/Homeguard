# Futures SP-A: Daily Signal Wrappers - 2026-07-07

## Summary
Shipped sub-project A of the futures strategy-testability campaign: reusable
daily-forecast signal templates plus a priority subset of concrete strategies,
all on the EXISTING `forecast_panel` walk-forward engine (no engine change).
About 20 catalog strategies are now testable. Built via subagent-driven
development (fresh implementer + reviewer per task, 9 tasks, final Opus
whole-subproject review returned MERGE-READY with zero Critical/Important).

## Context
Campaign umbrella design: `docs/superpowers/specs/2026-07-07-futures-strategy-testability-campaign-design.md`
(five sub-projects A-E; only SP-B intraday is a genuinely new engine; A/C/D/E
route through the existing daily engine). SP-A plan:
`docs/superpowers/plans/2026-07-07-futures-sp-a-daily-signal-wrappers.md`.

## Changes Made
- **Pre-registration gate** (`src/backtesting/utils/pre_registration.py`,
  wired into `src/backtesting/engine/futures_backtest.py`): `run_futures_backtest`
  now refuses a config lacking a non-empty `pre_registration` block
  (construction, expected_sign in {long,short,long_short,neutral}, hypothesis).
  Backfilled all 10 existing futures configs + 11 pre-existing engine tests.
- **Three reusable base templates** (`src/strategies/advanced/futures_signal_base.py`):
  `CrossSectionalRankStrategy` (within-group demean -> same-day z -> scale 10 ->
  clip +/-20; all-NaN rows -> 0), `CalendarMaskStrategy` (daily on/off hold,
  sign*cap), `ConditioningOverlayStrategy` (gate/combine a base + conditioning
  forecast via registry).
- **Concrete strategies**: #3 `FuturesXSMomentum` (12-1, commodity block),
  #23 `FuturesReversal` (-z 5d, index), #16 `FuturesTurnOfMonth` (calendar),
  #15 `FuturesSameMonthSeasonality` (prior-years same-month mean, causal),
  #13 `FuturesCarryTrend` (trend gated by carry sign). #10 curve-slope XS is a
  config on the refactored `FuturesCarryXS` over a commodity-only universe.
- **FuturesCarryXS refactored** onto `CrossSectionalRankStrategy` (behavior
  preserved; regression-tested).
- Configs in `config/backtesting/` (6 new + backfills), trial ledger
  `docs/strategies/research/20260707_FUTURES_SP_A_TRIALS.md`, tests in
  `tests/strategies/futures/`.

## Commits (on main via merge 9de7228)
- `c6b47fe` pre-registration gate; `05e142a` backfill remaining configs
- `b079334` CrossSectionalRankStrategy base + FuturesCarryXS refactor
- `81a8268` #3 XS momentum; `9c71f34` base-class warmup zero-fill + cleanup
- `48835fd` #23 reversal
- `162be30` CalendarMaskStrategy base + #16 turn-of-month
- Task 6 (#15 seasonality) files: landed inside `27d85c1` (see Known Issues)
- `d0cbdcd` ConditioningOverlayStrategy base + #13 carry-trend
- `9c7c4cb` #10 curve-slope config + trial ledger
- `550eb62` registry sanity test
- `9de7228` merge (integrated concurrent FX session-clock work already on main)

## Known Issues / Remaining Work
- **Concurrent-session entanglement (resolved at merge):** a parallel
  "FX session-clock" SDD session committed to the SAME working tree/HEAD, so its
  commits interleaved onto the feature branch AND onto main directly, and my
  Task 6 (#15 seasonality) files were swept into the FX commit `27d85c1` (correct
  content, misleading message). FX code was byte-identical on both refs, so the
  merge into main was conflict-free. LESSON: run concurrent Homeguard sessions in
  separate git worktrees (superpowers using-git-worktrees), never a shared tree.
- **#9 multi-horizon carry DEFERRED**: needs a multi-horizon carry cache
  (~3mo/~6mo horizons from per-contract data) -- a data-pipeline task adjacent to
  SP-E, not signal code. Tracked in the trial ledger.
- **No walk-forward runs done yet**: SP-A builds the CAPABILITY. The gated runs
  (PSR/DSR/PBO verdicts) are the next step, one pre-registered trial at a time.
- **Next campaign sub-projects**: E (external data), then B (intraday engine),
  C (spread engine), D (options-IV).
- Cosmetic minors (accepted by final review): unused `base` var in the
  seasonality test; a redundant `.copy()`; pre-existing pydantic deprecation warning.

## Validation
- Per-task: fresh reviewer gate each task; 2 fix waves (Task 1 config backfill,
  Task 3 base-class NaN handling) both re-reviewed clean.
- Two real bugs caught that the plan's code would have shipped: base-class
  NaN-warmup handling (moved into the base, gated on all-NaN rows) and a silent
  index-dtype mismatch in seasonality that zeroed all positions (fixed by
  restoring the original close_panel.index).
- Tests green: `tests/strategies/futures/` 26 passed; Task-1-touched engine tests
  13 passed; FX session tests 16 passed (merged main).
- Smoke tests (real engine, short window, register=False): momentum 0.476/3464d,
  reversal 3467d, turn-of-month 3467d, seasonality 0.304/4082d, carry-trend 3467d.
- Final Opus whole-subproject review: MERGE-READY, zero Critical/Important.
- Merged to origin/main as fast-forward `8a7d169..9de7228` (no force-push).
