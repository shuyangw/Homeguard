# Futures SP-B1: Intraday Session Engine + Gate Verdicts - 2026-07-09..10

## Summary
Two pieces of the futures testability campaign this session: (1) ran the actual
walk-forward gate verdicts on the SP-A/SP-E strategies that have data, and (2)
built SP-B1, the intraday session engine (the campaign's one genuinely new
simulator) plus overnight drift (#21/#25) and pre-FOMC (#39). Both via
subagent-driven development; SP-B1 in an isolated git worktree (no collision).
Honest bottom line: almost nothing clears the gate -- carry_idm (0.765) remains
the best deployable futures book.

## Part 1 -- Gate verdicts (2026-07-09)
Ran `walk_forward_carver` (PSR/DSR/PBO + 1.5x cost gate) on the now-testable
strategies. Results recorded in the SP-A/SP-E trial ledgers
(`docs/strategies/research/20260707_FUTURES_SP_{A,E}_TRIALS.md`):
- **Only #13 carry-trend PASSES** (0.357) -- but weaker than carry (0.765) and a
  carry+trend re-expression.
- #10 curve-slope highest raw Sharpe (0.846 > carry) but **PBO 0.690** = overfit
  (the XS-carry failure mode). #3/#15/#23 WEAK. #37 CoT / #16 turn-of-month REJECT.
- #26 VIX roll-down +0.564 (from SP-E). NONE beats carry_idm.
- **Regression fixed:** SP-A's pre-registration gate had silently broken the
  walk-forward runner (`run_futures_backtest` gate over-fired on the per-window
  configs the walk-forward builds internally). Fixed with a `validate_prereg`
  flag (internal machinery opts out) + regression test (commit 723304a).

## Part 2 -- SP-B1 intraday session engine
Return-stream engine (NO contract/margin sim), gated via the shared PSR/DSR/PBO
path exactly as the VIX sleeve. Spec:
`docs/superpowers/specs/2026-07-10-futures-sp-b-intraday-session-engine-design.md`.
Plan: `docs/superpowers/plans/2026-07-10-futures-sp-b-intraday-session-engine.md`.

### Changes Made
- **Session clock** (`src/backtesting/sessions/equity_index_clock.py`): ET session
  times + `et_to_utc` (DST-aware via `fx_clock`).
- **Session-bars cache** (`src/backtesting/session/session_bars.py`): extract the
  ratio-adjusted 1-min closes at the boundary ET times per root; all-NaN (Sunday)
  rows dropped so the trading-day index is correct. ES/NQ caches built (4044 dates).
- **Simulator** (`session_simulator.py`): `SessionTrade` + `simulate_session_returns`
  -> net-of-cost per-trade returns; NaN close skips.
- **Aggregation + gate** (`session_walkforward.py`): vol-normalized multi-root
  aggregation (returns a DatetimeIndex) + `gate_session_stream`.
- **Strategies**: #21/#25 overnight drift + hour-slice variant
  (`overnight_drift_strategy.py`), #39 pre-FOMC (`prefomc_strategy.py`).

### Real results (session trial ledger `docs/strategies/research/20260710_FUTURES_SP_B_TRIALS.md`)
- **#21/#25 overnight drift**: 8086 trades, OOS Sharpe 1x 0.792 / 1.5x 0.671, but
  **PBO 0.513** -> WEAK (a real positive premium, but window-unstable, fails the gate).
- **#21 hour-slice** (02:00-05:00 ET): -0.023 / -0.277, PBO 0.87 -> REJECT (the
  drift is NOT concentrated in this window; the full overnight is stronger).
  CAVEAT: the window is an unverified approximation of NY-Fed SR-917.
- **#39 pre-FOMC**: 252 trades; the walk-forward gate returns n_windows=0 (all-NaN)
  -- an ~8-events/yr stream never fills a 12-month/10-sample window, so the
  return-stream gate CANNOT judge sparse events (architectural, disclosed, not a
  bug). Decay split (pre-2015 0.25 / post-2015 6.54) is small-n noise (and
  sqrt(252) annualization overstates the sparse-stream Sharpe). UNGRADEABLE.
- NONE clears the gate. The engine works; the strategies are honest negatives/marginals.

## Commits (on main, ff to 66bf019)
Verdicts: `723304a` gate-regression fix; `4dba5f7` verdict ledgers.
SP-B (10 commits `76938dd..66bf019`): session clock; session-bars cache (+ fix
b586bce drop Sunday rows); simulator; aggregation+gate; overnight drift (+ fix
62bc131 central DatetimeIndex); hour-slice; pre-FOMC; trial ledger.

## Known Issues / Remaining Work
- **SP-B2 hardening (from the final review, 2 Important, deferred -- they change no
  current verdict since all strats fail the gate):** (I1) `aggregate_returns`
  vol-normalizes over the FULL sample (mild in-sample); switch to trailing/expanding
  vol before promoting any session strategy. (I2) the cost model uses
  `regular_hours=True` (half-tick slippage) for the hour-slice's 02:00-05:00 ET
  off-hours window (understates cost); thread a per-window liquidity flag when
  off-hours strategies are added.
- Minors: short-side sign untested (symmetric); simulator return dict keyed by
  exit_date only (safe as used -- per-root filtered -- but a footgun for a future
  >1-trade/day/root strategy); returns.csv aggregated not per-fill (VIX precedent).
- **Two review-caught bugs during SP-B** (both fixed): all-NaN Sunday rows polluted
  the trading-day index (would have dropped the Friday->Monday weekend overnight);
  a date-vs-Timestamp mismatch between the cache and the gate (centralized in
  aggregate_returns).
- **Campaign status**: SP-A + SP-E + SP-B done (3 of 5). Remaining: SP-C
  (multi-leg spread engine, incl. the #1-ranked yield steepener) and SP-D
  (options-IV, #28, with the correlated-re-expression check vs #26).
- No session strategy is deployment-ready; all failed the gate. carry_idm (0.765)
  stays the best futures book.

## Validation
- SDD per-task reviews caught 2 real bugs SP-B's offline tests missed (Sunday-row
  index pollution; date/Timestamp gate mismatch), both fixed + re-reviewed.
- Causality verified end-to-end by the final review: entry/exit closes are the
  first 1-min bar at/after the ET timestamp (no lookahead); overnight uses the
  true next trading day (weekend overnight representable); signs long, never flipped
  despite marginal/negative smokes.
- Tests: 51 passing (session engine + SP-A/SP-E regression + SP-B strategies).
- Worktree isolation worked (no FX-session collision). Merged as a clean
  fast-forward; no force-push.
