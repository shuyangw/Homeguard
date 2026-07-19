# FX Intraday Engine + #20 London Open Breakout - 2026-07-19

## Summary
Built the general intraday minute-bar order engine (the reusable core for ~24 intraday FX strategies) and ran research strategy #20 London Open Breakout through it end-to-end, producing the campaign's FIRST gated intraday result. #20 FAILS the S&P bar: OOS Sharpe -1.60 vs S&P 0.68 (IS -0.99, DSR 0). The daily London-open breakout dies after realistic spread. Built via subagent-driven TDD in an isolated git worktree; opus whole-branch review judged the negative verdict trustworthy (reconciled to spread bleed to first order). Merged to main.

## Changes Made
- **src/backtesting/data/fx_intraday_loader.py** (new): `load_fx_1min` (tz-aware UTC 1m bars from the Dukascopy cache) + `resample_ohlc`.
- **src/backtesting/engine/intraday_order_engine.py** (new): general minute-bar order engine. Order book (stop/limit/OCO/bracket), partial fills, trailing stops, time controls, no-lookahead loop (order armed at bar_t first eligible bar_{t+1}), conservative fills (buy-stop at max(trigger,open) gap-through + half round-trip spread), adverse both-in-one-bar rule, OCO atomicity within a bar. 15 tests.
- **src/strategies/advanced/fx_london_breakout.py** (new): #20. Incremental Asian range (00:00-07:00 London), 0.25-0.8x ATR(14,D1) width filter, tier-1 EUR/GBP event skip (win_end 12:01 for BOE noon), 08:00-09:30 OCO entry cancel-at-09:30, bracket exit (take half at 1x range + trail 1x ATR(15m) + flat 16:00), fixed-fractional 0.5% risk. Emits `day_r` (R-multiple, qty/pip-independent: full-stop = -1 R). 8 tests.
- **scripts/backtest_scripts/run_fx_london_breakout_walkforward.py** (new): per-FX-day engine runs -> aggregate `risk_frac * day_r` equal-weight across 4 pairs -> daily return series -> existing walk-forward + S&P gate (same-dates), RunStatus-wrapped.
- **config/backtesting/fx_london_breakout.yaml**, **docs/reports/fx/20260719_london_breakout_prereg.md** (pre-registration), **docs/reports/fx/FX_LONDON_BREAKOUT_WALK_FORWARD.md** (report).
- **docs/strategies/FX_60_CATALOG_TRACKER.md**: #20 -> FAIL-enh with the intraday verdict.

## Commits (branch feat/fx-london-breakout, FF-merged to main 3fad9c2..8e57a81)
- `4be5b7f` 1m loader + resampler
- `b9edf38` engine core (stop/limit fills, no-lookahead)
- `51ecd9c` + `1426977` OCO/bracket positions + exit-math fix wave (same-bar trail breach, ratchet guard, tp=1 close, exit-reason label)
- `db67f41` bar loop run()
- `61a9491` + `792e756` #20 strategy + fix wave (R-multiple day_r, day-leak guard, 09:30-cancel robustness, OCO same-bar bias)
- `91f208f` walk-forward runner + S&P gate (first verdict OOS -1.28)
- `8e57a81` final-review wave (london 1.2x spread, OCO engine atomicity, eod flatten, same-dates gate, drop dead asian_range; re-run -> OOS -1.60)

## Result (real, FAIL vs S&P)
| Metric | Value |
|---|---|
| OOS Sharpe (net, 1.2x London spread) | -1.5995 |
| S&P Sharpe (same 3064 OOS dates) | 0.6767 |
| Beats S&P | False |
| IS Sharpe | -0.99 |
| DSR / PBO | 0.00 / 0.67 |
| n_oos_days | 3064 (2014-2026, 13 windows) |

Trustworthy per opus whole-branch review: sign/scaling provably correct, no lookahead, spread charged once, magnitude reconciles with spread bleed, IS also negative. Honest "mechanism dies after costs."

## Known Issues / Remaining Work
- **fx_clock DST-on-intraday bug (on main, needs hardening):** `fx_trading_day`'s `DateOffset(hours=7)` raises NonExistentTimeError on 1m data crossing the spring-forward gap (only ever exercised on daily data). The runner works around it with a tz-naive +7h shift (verified byte-identical on non-gap data). Harden `fx_clock` itself in a follow-up before more intraday strategies rely on it.
- EURGBP/GBPJPY daily ATR(14) derived from 1m bars (absent from the fx_daily cache); disclosed in the report.
- Single cost leg reported (R-multiple nets a fixed spread; no clean 1.5x rescale) -- moot since it fails at 1x.
- Per the pre-registration: one bounded #20 modification (Judas-swing / first-15m filter / retest-limit) may be tried, OR move to the next intraday strategy (#21-25) now that the engine exists. Decision pending.

## Validation
- 26 tests across the 2b surface pass (25 confirmed on main post-merge; runner test needs the local 1m data path).
- Opus whole-branch review: READY TO MERGE, -1.60 FAIL judged trustworthy.
- Walk-forward re-run after the cost fix; verdict reproduced (more negative, as expected).
- Built in an isolated worktree (`.worktrees/fx-london-breakout`), FF-merged, worktree + branch cleaned up.
