# Futures SP-C: Multi-Leg Spread Engine - 2026-07-10

## Summary
Built SP-C of the Futures Strategy Testability Campaign: a shared spread-construction
layer plus two return-stream engines (continuous forecast + convergence state machine)
that make Family E (spread / relative-value strategies #31-#36) testable in the
walk-forward PSR/DSR/PBO gate. Ran all six strategies. Honest bottom line: NOTHING in
Family E beats carry_idm (OOS Sharpe 0.765), and nothing passes the gate. The headline
finding is that the calendar-spread's apparent edge was almost entirely roll-jump
contamination that vanished under masking. Built via subagent-driven development in an
isolated git worktree; merged to main (a07336e) as a clean fast-forward.

## Changes Made
- **Shared gate relocation** (`src/backtesting/walkforward_common.py`): moved
  `gate_return_stream` + `_oos_windows` here (byte-for-byte from `vix_rolldown_eval.py`)
  so the new `spreads` package does not depend on the `vix` package; VIX suite unchanged.
- **Yield/RTY daily data**: aggregated 2YY/5YY/10Y/30Y + RTY from databento 1min to
  `daily_raw`. Found 5YY is data-degraded (~440 rows, sparse from 2023).
- **Front/next builder** (`src/data/futures/front_next.py` + `front_next_dir()`): per-root
  daily F1/F2 settle series reusing `CarryCalculator._find_front_second_close`. Made it
  month-batched (read each monthly parquet once) after a review caught a 24s/month cost
  -> ~10s/year; added a cache coverage-check + merge-on-write.
- **Construction layer** (`src/backtesting/spreads/construction.py`): `SpreadLeg`,
  `SpreadSeries`, `build_spread(additive|multiplicative)`, `round_trip_cost_usd`.
- **Continuous engine** (`continuous.py`): causal z-MR + 12-1 momentum forecasts,
  vol-scaled net-of-cost return stream.
- **Convergence engine** (`convergence.py`): `SpreadTrade` state machine (enter |z|>=2
  fade-the-stretch, exit converge/time/structural, asymmetric short-side stop) + entry
  guard + force-close-at-end.
- **Six strategies** (`src/strategies/advanced/spread_*.py`): #35 steepener (+ segments,
  build-time sign check), #36 inter-market RV, #31 calendar MR (with roll-day masking),
  #32/#33 crack/crush, #34 gold/silver ratio. Run via `run_*`, not registered.
- **Trial ledger** (`docs/strategies/research/20260710_FUTURES_SP_C_TRIALS.md`): all
  verdicts, tracked.

## Verdicts (real, gated)
- **#35 steepener**: UNGRADEABLE (n_windows=0). CME Micro Yield futures too new (2YY from
  ~2021; 3yr z-window leaves ~1.5yr usable); 5YY degraded. Sign confirmed +0.0247, not flipped.
- **#36 inter-market RV**: NQ/ES 0.329 (< carry), RTY/ES -0.280 REJECT. book_corr not run
  (no RAMP daily-return series found) -- follow-up.
- **#31 calendar MR** (roll-masked): CL 0.394 / NG -0.150 / ZC 0.174 / ZS 0.358 / ZW 0.263,
  ALL fail PBO. Pre-mask Sharpes (1.0-1.18, NG nominally > carry) were roll-jump contamination.
  NG REJECT provisional (PBO 0.320 near threshold; volume-rank F1/F2 over-masks).
- **#32/#33 crack/crush**: crack RB -0.116, crack HO -0.215 REJECT; crush 0.136 (PBO 0.109
  clean but Sharpe trivial) MARGINAL. Research-grade (no micro product legs).
- **#34 gold/silver**: 0.269 / PBO 0.674 REJECT (weak-anchor, as pre-registered).

## Commits
Merged to main as fast-forward, 15 commits `e7c68b1..a07336e`. Key:
- `e7c68b1` canonical gate + yield/RTY data; `bd91618`/`0969662` yield test + import cleanup
- `db201b8`/`5c833c2` front/next builder + month-batched perf
- `546aed4` construction; `018c2c8` continuous engine; `8b615ed` convergence engine
- `a04d5e3` #35 steepener; `846f68e` #36 inter-market
- `37b6d43` #31 calendar; `127f364` #31 roll-jump masking fix
- `d7927e3` #32/#33 crack/crush; `c1040e9` #34 ratio
- `c388e03`/`a07336e` trial ledger + gate-wording/lint cleanup

## Known Issues / Remaining Work (SP-C2 candidates)
- **#35 rerun when yield-futures history matures** (or a shorter pre-registered window is
  justified by theory, not fit). 5YY needs a better data source.
- **#36 book-correlation check** vs the equity-momentum sleeve (the make-or-break re-expression
  test) still owed -- needs a loadable RAMP daily-return series.
- **#31 NG**: RollCalendar-based F1/F2 (instead of volume-rank) to stop over-masking; could move
  NG's provisional verdict.
- **#32/#33**: per-contract front-series or return-space construction to remove the additive
  ratio-adjusted-continuous kurtosis artifact.
- **Shared-gate PBO=NaN** when the shortest OOS window < 16 rows (methodology follow-up;
  affects VIX/SP-B too -- explains VIX #26's "PBO NaN").
- **Full-sample vol scaling** in `continuous_return_stream` -> trailing vol before promoting any
  continuous spread (same as SP-B I1).
- Campaign status: SP-A + SP-E + SP-B + SP-C done (4 of 5). Remaining: SP-D (options-IV #28),
  plus the VIX #26 deflation and SP-B2 intraday remainder.

## Validation
- Subagent-driven: fresh implementer + reviewer per task in an isolated worktree; per-task
  reviews caught real defects offline tests missed -- most importantly the #31 roll-jump
  contamination (Critical), plus a front/next cache-window Critical and a perf defect.
- Final whole-branch review (opus): MERGE-READY, no Critical/Important; all Minors triaged as
  accepted / SP-C2 follow-ups.
- Tests: 33 SP-C + touched-suite tests pass (VIX regression green, so the gate relocation is
  behavior-preserving). Every strategy persisted returns.csv + gate.json (+ trades.csv for the
  convergence strategies) under output/backtests/futures/sp_c_*.
