# FX Carry Seatbelt (#16 + #19) Design Spec

**Date:** 2026-07-06
**Status:** Approved (brainstorm), pending implementation plan
**Supersedes:** the naive `FxCarry` strategy (`src/strategies/advanced/fx_strategies.py::FxCarryStrategy`), which failed the walk-forward gate (OOS Sharpe -0.327, DSR 0.00, PBO 0.73). This is not a parameter tweak of that strategy; it is a different, filtered construction.

## 1. Purpose

The naive carry factor failed because it held every pair continuously through every crash (SNB 2015, COVID 2020, the Aug 2024 yen-carry unwind) with zero crash protection. The research is emphatic that carry's edge "is not in the idea but in survival engineering." This strategy adds the two survival filters the research prescribes:

- Research strategy #16 (Carry-Momentum Double Filter): only hold a carry pair when price momentum agrees with the carry direction. "Carry with a seatbelt."
- Research strategy #19 (Carry-Unwind Detector): a composite risk-off score that (defensively) flattens the carry book when a liquidation cascade ignites, and (offensively) shorts the crowded carry pairs into the cascade.

Both derive from the same #19 composite score, so they are built as one strategy sharing one score.

## 2. Architecture

A single strategy class `FxCarrySeatbelt` exposing `forecast_panel(close_panel: pd.DataFrame) -> pd.DataFrame`, matching the existing continuous-forecast FX engine contract (`src/backtesting/engine/fx_backtest.py:85` calls `strategy.forecast_panel(close)`; `FxSpotPortfolioSimulator` samples the returned daily forecast at the configured rebalance cadence).

The forecast for each (date, pair) is assembled from three components, all derivable from the daily close panel (which includes all 22 pairs, XAUUSD, and the JPY/CHF crosses) plus the FRED rate panel loaded internally (exactly as the naive `FxCarry` does via `src/data/fx_rates.py::load_fx_rate_panel`).

Universe: the existing 22-pair G10 cache
`[EURUSD, USDJPY, USDCHF, EURJPY, EURCHF, CHFJPY, XAUUSD, XAGUSD, GBPUSD, USDCAD, AUDUSD, NZDUSD, AUDNZD, AUDJPY, NZDJPY, EURNOK, EURSEK, USDNOK, USDSEK, NOKSEK, NOKJPY, SEKJPY]`.

### Component A: Long carry book (#16)

For each pair, on each date, eligible to go LONG only if BOTH gates pass:

1. Carry gate: annualized rate differential (base currency rate minus quote currency rate, the carry proxy) > +0.02 (i.e. +2% annualized). Rate panel from `load_fx_rate_panel` / `build_rate_diff_panel` (same source as naive `FxCarry`).
2. Momentum gate: close > EMA(50, daily) AND EMA(50) slope positive over the trailing 10 days (EMA(50)_t > EMA(50)_{t-10}).

Forecast when both gates pass: a fixed Carver-scale forecast of +10.0 (10 = 1x vol-target position; the engine's per-instrument vol-target sizing equalizes risk across eligible pairs). Otherwise 0.0. Binary, not graded, to keep degrees of freedom low for DSR. Never short for carry: when the gates disagree the pair is flat (flat is a position).

The +2% carry gate naturally excludes thin-carry legs (e.g. EURSEK, whose differential is well under 2%), which is the research's "SEK special case" handled without special-casing.

### Component B: Carry-unwind composite score (#19), reusable

A standalone, independently testable function in a new module `src/backtesting/signals/carry_unwind.py`:

```
compute_unwind_score(close_panel: pd.DataFrame, z_window: int = 252) -> pd.Series
```

Returns a daily score indexed like `close_panel`, higher = more risk-off / cascade-like:

```
score_t = z(delta_JPY_strength, inverted)
        + z(delta_CHF_strength)
        + z(AUDJPY 5-day realized vol)
        + z(XAUUSD 3-day return)
```

- JPY / CHF strength: per-currency strength via the same `aggregate_currency_returns` logic used by the `currency_strength` artifact (`src/data/artifacts/currency_strength.py`), using the 3-day change in cumulative strength. JPY strengthening is risk-off, so its contribution is inverted (strengthening -> positive score).
- AUDJPY 5-day realized vol: rolling std of AUDJPY daily returns over 5 days.
- XAUUSD 3-day return: gold bid is a risk-off tell.
- All four terms are converted to causal trailing z-scores over `z_window` (default 252 trading days): mean and std use only data up to and including date t. No lookahead. If any input is unavailable for a date, that term contributes 0 for that date.

This module is separate from the strategy because the research designates the score a shared risk-off "brain" consumed by #15/#16/#18/#42 (three-plus future consumers), meeting the bar for a reusable unit.

### Component C: Defensive veto + offensive short (#19)

Using the score from Component B:

- Defensive veto: when the veto is engaged, ALL Component-A long forecasts are set to 0.0 (flatten the carry book). The veto engages on any day `score >= 1.0`, and disengages only after 3 consecutive days with `score < 1.0` (per the research's "carry books stay flat until score < 1.0 for 3 consecutive days"). This is a stateful trailing condition computed causally over the daily index.
- Offensive short: when `score > 2.5` AND AUDJPY close < its trailing 20-day low, emit a short forecast of -5.0 (half of the +10 full size) on AUDJPY and NZDJPY. The short exits naturally when the condition clears (forecast returns to 0). This is "trading with the cascade, deliberately late and fast," not predicting it.

The strategy returns one daily forecast DataFrame: Component-A longs (>= 0), zeroed on veto days, plus Component-C shorts (< 0) on AUDJPY/NZDJPY when the cascade condition fires. The engine samples this at the rebalance cadence, so under weekly rebalance the veto/short only act on week boundaries (blunted); under daily rebalance they act within a day. Measuring that difference is an explicit goal (see Section 4).

## 3. Rebalance cadences

Two configs, identical except for the rebalance field:

- `config/backtesting/fx_carry_seatbelt_daily.yaml` (rebalance: daily)
- `config/backtesting/fx_carry_seatbelt_weekly.yaml` (rebalance: weekly)

Shared backtest settings mirror the other FX strategies: `vol_target_per_instrument: 0.03`, `leverage_cap: 4.0`, `idm: true`, `idm_cap: 2.5`, `initial_capital: 100000`, `save_trades: true`, dates 2011-01-01 to 2026-04-01, full 22-pair universe.

Daily rebalance is faithful to #19's instant-flatten intent but costs more turnover; weekly is cheap but reacts to a cascade up to 5 days late. Running both directly answers whether fast crash-reaction pays for its cost.

## 4. Success criterion (pre-registered, locked before any backtest)

Primary bar is RELATIVE to the S&P 500, not an absolute statistical threshold.

- Gate mechanics: the existing walk-forward harness (36-month train / 12-month test / 12-month step, purge + embargo, both 1.0x and 1.5x cost legs), run on BOTH cadences. Log all five combined-gate legs from methodology Section 2.5 (Sharpe, PSR, DSR, PBO, and 1.5x cost sensitivity), plus the IS/OOS Sharpe ratio, fixing the earlier subset-reporting gap. The two new configs increment the project-wide DSR trial count in `output/experiments.duckdb`.
- Benchmark: S&P 500 buy-and-hold annualized Sharpe computed over the EXACT same stitched OOS dates the strategy trades (rf = 0, same convention as the strategy's Sharpe), net. Source `^GSPC` / SPY daily closes (available in the framework's equity data / `equity_index_yfinance` plugin). Align to the intersection of the strategy's OOS return dates and available S&P dates.
- PASS: strategy OOS Sharpe (1.0x cost) > S&P OOS Sharpe over those dates. Also reported under 1.5x cost for robustness.
- No absolute kill threshold. DSR / PSR / PBO are computed and reported as honesty diagnostics but do not gate the pass/fail decision; the S&P comparison does.
- Also report (do not gate on): the strategy's correlation to the S&P over the OOS dates and its information ratio versus the S&P. A beat with low correlation is a materially stronger result (diversification value) than a beat with high correlation.

Caveat acknowledged, not adjusted for: the OOS window (~2014-2026) spans the 2010s equity bull run, so the S&P OOS Sharpe is a demanding bar (likely ~0.6-0.8). Clearing it with an uncorrelated FX book is a real result.

## 5. Known limitations (reported, not silently absorbed)

1. Swap = policy-rate-differential proxy. We have no broker swap tables. The +2% carry gate uses the FRED rate differential. The research explicitly warns "broker swap rates, not policy differentials, are your P&L," with retail markups of 0.5-1.5%. This is an optimism bias in the carry gate and must be stated in the report, not hidden.
2. Offensive short = existence proof, not statistics. There are only ~4-6 unwind episodes in the sample. The combined strategy receives a DSR, but the short leg's contribution is reported as per-episode P&L attribution (Aug 2024 yen unwind, Mar 2020 COVID, Jan 2019 flash if data reaches), explicitly labeled as existence proofs, not a statistical claim.

## 6. File structure

- Create `src/backtesting/signals/carry_unwind.py`: `compute_unwind_score(close_panel, z_window=252)` plus small causal helpers (trailing z-score, per-currency strength). Pure functions, no I/O.
- Create `src/strategies/advanced/fx_carry_seatbelt.py`: `FxCarrySeatbelt` class. Loads the rate panel internally; imports `compute_unwind_score`.
- Modify `src/strategies/registry.py`: register `FxCarrySeatbelt`.
- Create `config/backtesting/fx_carry_seatbelt_daily.yaml` and `..._weekly.yaml`.
- Create `tests/backtesting/signals/test_carry_unwind.py`: score correctness in isolation (JPY-strengthening + gold-bid + high-vol day -> high score; calm day -> low score; causality: score at t independent of data after t; NaN handling).
- Modify `tests/strategies/test_fx_strategies.py` (or a new `test_fx_carry_seatbelt.py`): `forecast_panel` gate logic (carry+momentum both pass -> +10; carry alone -> 0; momentum alone -> 0; veto zeroes longs when score high; short fires only on score > 2.5 AND 20-day low; causality; NaN handling).
- Add a short pre-registration note (the Section 4 criterion, dated, before results) to the experiment log / a dated doc under `docs/reports/fx/`.

## 7. Testing plan

1. Unit tests on `compute_unwind_score` (isolated, synthetic panels): a constructed risk-off day scores high, a calm day scores low, causality holds (truncating future rows does not change past scores), missing inputs degrade gracefully to 0.
2. Unit tests on `FxCarrySeatbelt.forecast_panel` (synthetic close + monkeypatched rates, following the existing `test_fx_carry_sign_convention` pattern): each gate in isolation, the veto, the short trigger, causality, NaN handling, forecast bounds.
3. Walk-forward gate on both cadences via the existing FX walk-forward runner (extended to emit the S&P benchmark comparison and the five gate legs), producing the readiness report with per-episode attribution.

## 8. Out of scope (explicitly deferred)

- Graded sizing (#16 mod b) and the 12-month TSMOM momentum leg (#16 mod a): reserved as the single pre-registered follow-up variant, allowed only if the primary form lands marginal (beats or nearly beats S&P but we want to strengthen it). Not built now (keeps degrees of freedom low).
- The hourly offensive-short refinement, ATR trailing stops, and time stops: require intraday data / an event-driven engine. The daily approximation above is what we build now.
- Wiring the `carry_unwind` score into other strategies (#15/#18/#42): the module is built reusable, but only this strategy consumes it now.
