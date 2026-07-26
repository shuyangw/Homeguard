# FX OHLC / Range-Based Wave: Pre-Registration

**Date:** 2026-07-25 | **Status:** LOCKED 2026-07-25 (approved; no post-hoc edits to specs/params/universe/gate) | **Owner:** main-loop -> strategy-lead for the verdict

Pre-registration per the North Star: hypotheses, universe, specs, params, gate,
trial count and PASS/FAIL fixed BEFORE any backtest. Once locked this set IS the
search.

## 1. Why these are new

Every FX spec gated so far used CLOSE prices only, because the engine discarded
open/high/low before calling the strategy (fixed 2026-07-25 via `wants_ohlc`).
This wave tests signals that require the intraday RANGE and therefore could not
previously be expressed:

- **True range / ATR** -- volatility measured from the high-low span, not
  close-to-close. Bands built on ATR widen and narrow with intraday activity in
  a way a close-based standard deviation does not.
- **ADX / directional movement** -- trend STRENGTH, which has no close-only
  equivalent.
- **Parkinson volatility** -- a high-low range estimator, ~5x more efficient
  than close-to-close for the same sample.

This distinction matters for honesty: the close-only z-score mean-reversion we
already ran and failed (EM-MEANREV) is a DEGRADED PROXY for the band-based
mechanisms below, not the same test. Where a spec IS close to something already
tested, Section 3 says so explicitly.

## 2. Universe (ONE universe for every spec -- no per-spec selection)

**G10-22**: the full validated 22-pair G10 daily cache. Deliberately identical
across all four specs. Choosing a mechanism-friendly universe per spec (e.g.
low-volatility crosses for mean reversion, majors for trend) would be a
researcher degree of freedom; if a mechanism needs a specific universe that is a
SEPARATE pre-registered spec, not a tweak to this one.

Range 2011-2026. Standard taker costs, `execution_lag=1`, hardened apparatus as
of 2026-07-25 (fixed PBO, publication-lagged rates, silent-skip fixed,
unit-correct PSR/DSR).

## 3. The four specs (params FIXED a priori, textbook defaults, no sweep)

All are continuous Carver-scale forecasts, vol-targeted 0.03/instrument, IDM on,
weekly rebalance. All set `wants_ohlc = True`.

1. **OHLC-KELTNER (#12) -- GENUINELY NEW.** Mean reversion against ATR bands.
   Center = EMA(20) of close; band = 2.0 x ATR(10). Forecast = `-(close - center)
   / (2 * ATR10)`, clipped +-2, scaled. Distinct from EM-MEANREV because the
   band width tracks the intraday RANGE, not close-to-close dispersion.
2. **OHLC-SQUEEZE (#27) -- GENUINELY NEW.** Volatility-compression breakout.
   Squeeze is ON when the Bollinger(20, 2.0) band sits INSIDE the Keltner(20,
   1.5 x ATR10) band. Forecast = 0 while squeezed; on release, take the sign of
   the 20d close change. No close-only equivalent (the test compares a
   close-based band to a range-based band).
3. **OHLC-VOLSPIKE (#29) -- GENUINELY NEW.** Fade after a range-volatility
   spike. Parkinson RV(10) z-scored over 252d; when z > 2, forecast =
   `-sign(20d return)`, else 0. Uses the high-low estimator, not close-to-close.
4. **OHLC-ADX-TREND (#6) -- AN ENHANCEMENT, LABELLED AS SUCH.** The existing
   FxTrend (Carver EWMAC) forecast, gated to zero when ADX(14) < 25. This is
   NOT a new mechanism: it is a filter on #3/FxTrend, which already FAILED (OOS
   -0.02, DSR 0.20). Included because ADX is the pre-registered catalog entry
   and the gate is genuinely unexpressible without high/low, but the honest
   prior is low and it must not be read as a fresh mechanism. Counted as a full
   trial.

Fixed params: EMA 20, ATR 10, Keltner 2.0x / 1.5x, Bollinger 20/2.0, Parkinson
10, z-window 252, ADX 14 threshold 25, breakout/fade horizon 20d. **No sweep.**
If a spec fails we do NOT try another ATR multiple or ADX threshold; that is the
specification search this document exists to prevent.

**Trial count = 4** (N 137 -> 141).

## 4. Explicitly EXCLUDED from this wave (and why)

- **#28 ATR-regime switch** -- an allocation OVERLAY. Overlays modulate a base
  strategy and we have no profitable base; testing one measures the base.
- **#1 Dual MA + ATR trail** -- the new part is a stateful TRAILING STOP, which
  the continuous-forecast engine does not express (it has no position state or
  exit machinery). That is a build gap, not a signal to gate here.
- **#8 Bollinger** -- its base form is a close-based z-score, i.e. materially
  what EM-MEANREV already tested and failed. Only its ADX filter is new, and
  that is covered by spec 4.
- **#47 Silver beta amplification** -- a substitution layer on #43 GoldSilver,
  which already failed (OOS -0.31).

## 5. Gate (pre-committed)

Combined gate, methodology Section 2.5: OOS Sharpe > 0 AND positive at 1.5x cost
AND PSR >= 0.95 AND **DSR >= 0.95** AND PBO < 0.5, plus the S&P benchmark /
marginal-contribution check. PSR/DSR computed with `periods_per_year=252` (the
unit bug fixed 2026-07-25). Mandatory run-scoped fills. Deflation at the
project-wide cumulative N (137 -> 141); N is never reduced to help a spec pass.

## 6. Registered prediction

At N~137 the deflated bar is approximately **1.13** annualized Sharpe (using the
realized cross-trial spread v=0.4278 measured in the Tier B wave). The
campaign's best genuine OOS Sharpe is +0.05 (TOT-OIL, cost-surviving but
trivial). **Predicted outcome: all four FAIL.** Registered in advance.

The wave is still worth running: the marginal deflation cost across 4 trials is
under 0.01 Sharpe, the range-based mechanisms are genuinely unexpressible before
this week's engine change, and a scoped negative closes the OHLC family.

## 7. Stopping rule

All four fail -> the range-based (ATR/ADX/Parkinson) family is unproductive FOR
THIS daily-spot-taker construction. Record it scoped exactly that way, STOP the
family (no sweep, no ML variant), and the remaining catalog blockers are
INTRADAY (21) and ML (6), both substantial builds. Any pass -> book-level
marginal-contribution evaluation before any deployment.

## 8. Build tasks (NOT verdicts)

1. `src/features/range_indicators.py`: causal `true_range`, `atr`, `adx`,
   `parkinson_rv` over a (pair, field) OHLC panel. Trailing windows only.
2. `FxOhlcStrategy` base with `wants_ohlc = True` + four registered subclasses;
   4 configs under `config/backtesting/ohlc/`.
3. Unit tests: each indicator against a hand-computed fixture, plus a
   no-lookahead assertion (perturbing bar t+1 must not change the value at t).
