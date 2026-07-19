# Intraday Order Engine + #20 London Open Breakout Design Spec

**Date:** 2026-07-19
**Status:** Approved (brainstorm), pending implementation plan
**Context:** Sub-project 2b of the intraday FX engine, the vertical slice that produces the FIRST gated intraday result of the 60-strategy campaign. Builds on the merged FX session clock (`src/backtesting/sessions/fx_clock.py`) and the tier-1 event calendar (`src/data/macro_calendar_tier1.py`, sub-project 2a). Delivers, in one spec: a general minute-bar order engine, a 1-minute bar loader, the #20 London Open Breakout strategy, and its walk-forward gate vs the S&P. The engine is deliberately general (reusable by later intraday strategies #21-25) even though only #20 rides it now.

## 1. Purpose

Answer the first intraday question of the campaign: does a filtered London-open breakout (research #20) beat the S&P out-of-sample, net of realistic intraday fills and costs? To do that honestly, build a minute-bar order engine whose fill model is conservative (no lookahead, gap-through realism, spread slippage), run #20 through it on real 1-minute data, aggregate to daily returns, and apply the same pre-registered "beat the S&P OOS Sharpe" gate used for FxCarrySeatbelt.

## 2. Constraints and inputs

- 1-minute bars are on disk at `fx/massive/1min/symbol=<PAIR>/**` (canonical 8-col schema, tz-aware UTC, 2011-2026) for GBPUSD, EURUSD, EURGBP, GBPJPY (all confirmed present).
- Reuse: `fx_clock` (`in_session_mask`, `session_window_utc`, `fx_trading_day`, `EXCHANGE_TZ`), `macro_calendar_tier1.tier1_release_in_window`, the FX cost model (`src/backtesting/costs/fx.py`: `fx_round_trip_pips`, `_pip_size`), the daily FX panel loader (`fx_backtest_loader.load_fx_daily_panel`) for daily ATR, the walk-forward helpers (`walkforward_common`), and the S&P benchmark (`src/backtesting/benchmark.py`). Mirror the FxCarrySeatbelt walk-forward runner.
- fintech conda env; ASCII-only, no em dashes, no emojis, no print() (use `src.utils.logger`).
- Every fill must be causal: an order placed reacting to bar_t may only fill on bar_{t+1} or later. This is THE correctness property of the engine.

## 3. Architecture (components)

Clean module boundaries within this one spec:

1. `src/backtesting/data/fx_intraday_loader.py` -- load 1m bars for a pair list over [start, end] as a tz-aware UTC frame; resample helper to arbitrary freq (15-minute for the trail ATR). Reads the parquet cache with the same pattern as the daily loader.
2. `src/backtesting/engine/intraday_order_engine.py` -- the general minute-bar order engine (reusable core). Order book with stop / limit / OCO / bracket orders, partial fills (scale-out), trailing stops, time-based cancels and exits. Instrument-agnostic; drives a strategy via an `on_bar(bar, api)` callback.
3. `src/strategies/advanced/fx_london_breakout.py` -- the #20 strategy that drives the engine.
4. `config/backtesting/fx_london_breakout.yaml` -- the run config.
5. `scripts/backtest_scripts/run_fx_london_breakout_walkforward.py` -- aggregates intraday trades to daily returns and runs the existing walk-forward + S&P gate.

## 4. The order engine

### 4.1 No-lookahead loop
Per instrument, bars in time order:
```
for bar in bars:
    engine.match_resting_orders(bar)   # orders placed on PRIOR bars fill against this bar
    strategy.on_bar(bar, api)          # strategy reacts to bar.close, places/cancels orders for NEXT bar
```
An order created inside `on_bar(bar_t)` is added to the book AFTER matching for bar_t, so it is first eligible on bar_{t+1}. This structurally prevents same-bar (lookahead) fills.

### 4.2 Fill model (conservative)
- Buy-stop with trigger T: triggered on a bar when `bar.high >= T`; fill price = `max(T, bar.open)` (models gap-through: if the bar opened above T you fill at the worse open, not T).
- Sell-stop with trigger T: triggered when `bar.low <= T`; fill price = `min(T, bar.open)`.
- Limit orders: buy-limit at L fills when `bar.low <= L` at `min(L, bar.open)`; sell-limit at L fills when `bar.high >= L` at `max(L, bar.open)`.
- Slippage/cost: add half the round-trip spread (from `fx_round_trip_pips` for the pair's tier, converted to price via `_pip_size`) to the fill in the adverse direction, on both entry and exit. (Round-trip spread is applied once across the entry+exit pair.)

### 4.3 Order types
- `OCO(a, b)`: two resting orders; when one fills, the other is cancelled.
- `Bracket(entry, stop_loss, take_profit)`: on entry fill, arm a protective stop and a take-profit as an OCO exit pair. Take-profit may be partial (`tp_fraction`, e.g. 0.5): filling it reduces the position and converts the remainder's protective stop into a trailing stop.
- Trailing stop: ratchets in the favorable direction by a fixed price distance (set from an ATR at arm time); never loosens.
- Time-based: `cancel_unfilled_at(t)` (drop resting entry orders at a wall-clock session time) and `flatten_at(t)` (market-exit any open position at a session time), both expressed via `fx_clock` session times.

### 4.4 Output
Per instrument, a trade log (entry/exit fills, timestamps, pips, position fraction) and a per-FX-day realized P&L. No lookahead, no same-bar fills, deterministic.

## 5. #20 London Open Breakout strategy

Per pair, per FX day (`fx_trading_day`):

1. Asian range: high/low of the bars in `in_session_mask(utc_index, "ASIAN_RANGE")` (00:00-07:00 London) for that FX day. `range_width = high - low`.
2. Width filter: trade only if `0.25 * atr_d1 <= range_width <= 0.80 * atr_d1`, where `atr_d1 = ATR(14)` on daily bars (prior day's value, no lookahead) from the daily FX cache.
3. Event skip: if `tier1_release_in_window(day, time(9,30), time(12,1), exchange="LONDON", currencies=("EUR","GBP"))` is True, stand down for that day (note win_end=12:01 to include a BOE noon decision, per the 2a hand-off).
4. Entry: during 08:00-09:30 London, place an OCO pair: buy-stop at `high + 3*pip`, sell-stop at `low - 3*pip` (`pip = _pip_size(pair)`). Cancel unfilled entry orders at 09:30 London.
5. Exit (bracket on the filled side): protective stop at the opposite range side; take-profit at `1 * range_width` beyond entry with `tp_fraction = 0.5` (take half); trail the remainder at `1 * ATR(15m)` (15-minute ATR(14) at arm time). Flatten any open position at 16:00 London.
6. Sizing: fixed-fractional risk -- size each entry so that being stopped at the protective stop loses a fixed fraction of current equity (`risk_frac`, default 0.005 = 0.5%). This is the natural sizing for a stop-defined breakout, not vol-target.

## 6. The gate (same bar as FxCarrySeatbelt)

- Aggregate each pair's per-FX-day P&L into a combined daily return series (equal risk weight across the 4 pairs; a day with no trade contributes 0).
- Run the existing walk-forward (36m train / 12m test / 12m step, both 1.0x and 1.5x cost legs) over the combined daily series, and evaluate: strategy OOS Sharpe (1x) > S&P 500 Sharpe over the same OOS dates. Report PSR/DSR/PBO, IS/OOS ratio, correlation and IR vs S&P as diagnostics (non-gating), plus the S&P aligned day count.
- Pre-register the criterion (a dated note under `docs/reports/fx/`) BEFORE the run, same discipline as FxCarrySeatbelt. This run increments the project-wide DSR trial count.
- Because the strategy is fixed-parameter (no in-run search), the walk-forward is a single configuration; the daily-frequency gate machinery is reused unchanged since it operates on the aggregated daily return series.

## 7. Known limitations (documented)

1. 1-minute OHLC bars do not capture intrabar path: when both the stop and the target lie within one bar's range, the engine cannot know which was hit first. Resolution: assume the ADVERSE outcome (stop before target) for any bar that spans both -- conservative, documented, and rare at 1-minute resolution.
2. Fills assume the modeled worst-of price; real slippage on a fast London-open break can exceed it. The half-spread slippage is a floor, not a guarantee; the 1.5x cost leg is the stress check.
3. The tier-1 event calendar is approximate (2a limitation); event-skip days may be off by a day occasionally.
4. Spread is modeled per-tier from the cost model, not from measured intraday quotes; the London-open session multiplier applies.

## 8. Testing plan

Engine (heaviest coverage, it is the reusable core):
1. Buy-stop fills at `max(trigger, open)`; sell-stop at `min(trigger, open)`; gap-through case (bar opens beyond trigger) fills at the open.
2. No-lookahead: an order placed in `on_bar(bar_t)` does not fill against bar_t even if bar_t's range spans the trigger; it fills on bar_{t+1}.
3. OCO: filling one leg cancels the sibling in the same match step.
4. Bracket partial take + trailing: TP at fraction 0.5 halves the position and arms a trailing stop on the remainder; the trail ratchets favorably and never loosens; an adverse move hits the trailed stop.
5. Time cancel at 09:30 drops unfilled entry orders; time-flat at 16:00 market-exits an open position.
6. Both-in-one-bar rule: a bar spanning both stop and target resolves to the stop (adverse).

Strategy (synthetic day):
7. Width filter: a range at 0.5x ATR trades; 0.1x and 0.9x stand down.
8. Event-skip: a day where `tier1_release_in_window` is True places no orders.
9. A clean upside break fills the buy-stop, cancels the sell-stop, and produces a bracket with the expected stop/target prices.

Gate runner:
10. Trades aggregate into a daily return series of the right length; the runner produces the readiness report with the S&P comparison and all diagnostics.

## 9. Files

- Create `src/backtesting/data/fx_intraday_loader.py`
- Create `src/backtesting/engine/intraday_order_engine.py`
- Create `src/strategies/advanced/fx_london_breakout.py`
- Modify `src/strategies/registry.py` (register the strategy if it goes through the registry; otherwise the runner instantiates it directly)
- Create `config/backtesting/fx_london_breakout.yaml`
- Create `scripts/backtest_scripts/run_fx_london_breakout_walkforward.py`
- Create tests under `tests/backtesting/engine/`, `tests/backtesting/data/`, `tests/strategies/`
- Create `docs/reports/fx/YYYYMMDD_london_breakout_prereg.md` (pre-registration, before the run)

## 10. Out of scope (deferred)

- Strategies #21-25 and beyond (they will reuse this engine).
- The Judas-swing / retest-limit / first-15-minute modifications of #20 (mods a-d); the base form is gated first, one bounded improvement round only if it lands marginal.
- Tick-level fills, real intraday quote spreads, and a live-trading adapter for the engine.
- Multi-instrument simultaneous position interaction (each pair is simulated independently; correlation shows up only at the daily-return aggregation).
