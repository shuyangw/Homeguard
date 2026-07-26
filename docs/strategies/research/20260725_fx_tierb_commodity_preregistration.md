# FX Tier B -- Commodity Terms-of-Trade Wave: Pre-Registration

**Date:** 2026-07-25 | **Status:** DRAFT (awaiting approval to lock) | **Owner:** main-loop -> strategy-lead for the verdict

Pre-registration per the North Star: hypotheses, universe, specs, signs, gate,
trial count and PASS/FAIL fixed BEFORE any backtest. Once locked this set IS the
search.

## 1. Mechanism and why it is genuinely new

Every prior FX wave tested signals derived from FX ITSELF (price, rate
differentials, carry) or from FX POSITIONING (COT). This wave tests an
**exogenous** signal family: the price of the COMMODITY a country exports.

Economic rationale (long-standing and well documented): for a commodity-exporting
economy, its export price IS its terms of trade. A rise in oil raises Canada's and
Norway's national income, improves the trade balance, and is transmitted to CAD
and NOK; the same holds for gold and AUD. The channel is macroeconomic, not a
price pattern in the currency, so this is a distinct hypothesis rather than a
re-roll of a failed FX factor.

## 2. Universe and data

- **Brent** (`alt_data/oil/BRENT/daily.parquet`, fetched 2026-07-25 keyless via
  yfinance BZ=F): 4,133 rows 2010-01-04..2026-07-24, 3,883 in the FX era, close
  19.33-127.98 (sane Brent levels).
- **Gold**: `XAUUSD` from the validated daily cache.
- **Currency legs**: USDCAD, USDNOK (oil); AUDUSD, NZDUSD (gold/commodity).
- Backtest range 2011-2026 (FX cache floor). Standard taker costs; USDNOK is a
  G10 cross, USDCAD a major.

**Publication/alignment control:** commodity closes and FX closes are aligned on
the FX trading date; the signal computed from a commodity close at date d is
executed at d+1 by the engine's `execution_lag=1` (added 2026-07-25). No
same-bar fills.

## 3. The three specs (signs FIXED a priori)

Shared: vol-target 0.03/instrument, IDM on, weekly rebalance, leverage cap,
daily `forecast_panel` engine, cost_mults (1.0, 1.5).

1. **TOT-OIL.** Forecast on USDCAD and USDNOK = `-z(oil momentum)`, i.e. oil
   rallying implies a STRONGER commodity currency implies a LOWER USDxxx. Sign
   fixed NEGATIVE on the USD-quoted pair. Momentum = 63d return of Brent,
   z-scored on a trailing 252d window, clipped +-2 -> Carver scale.
2. **TOT-GOLD.** Forecast on AUDUSD and NZDUSD = `+z(gold momentum)`, same
   construction on XAUUSD. Sign fixed POSITIVE (gold up -> AUD up -> AUDUSD up).
3. **TOT-XS.** Cross-sectional: rank the four commodity currencies by their OWN
   commodity's momentum z, long the top / short the bottom, market-neutral
   within the basket (sign inherited from specs 1-2 per leg).

Fixed params, no sweep: momentum 63d, z-window 252d, clip +-2.
**Trial count for this wave = 3** (N 134 -> 137).

## 4. Gate (pre-committed, unchanged)

Walk-forward (purge 0 -- correct here: nothing is fitted on the training segment,
it is warmup only), combined gate per methodology Section 2.5: OOS Sharpe > 0 AND
positive at 1.5x cost AND PSR >= 0.95 AND **DSR >= 0.95** AND PBO < 0.5, plus the
S&P benchmark/marginal-contribution check. Mandatory run-scoped fills.

Note the DSR wording is stated correctly here (>= 0.95). Earlier
pre-registrations wrote "DSR > 0", which a probability trivially satisfies; that
phrasing is retired.

## 5. Registered prediction and the bar

At N~134 the deflated bar is approximately **1.05 annualized Sharpe** (v=0.40).
The campaign's best OOS Sharpe to date is +0.42, and that was on the pre-fix
apparatus (same-bar fills, 65-day PBO); the best genuine candidate was +0.06.
**Predicted outcome: all three FAIL.** Registered so the result cannot be
retrofitted to whatever appears.

The wave is still worth running: the marginal deflation cost is small (the bar
moves from 1.053 to 1.059 across these 3 trials), the mechanism is genuinely
untested, and a scoped negative closes the last locally-testable signal family.

## 6. Stopping rule

All three fail -> the commodity terms-of-trade family is unproductive FOR THIS
daily-spot-taker construction (scoped exactly that way, NOT "commodity currencies
have no edge"); STOP this family, no sweep, no ML variant. Any pass -> book-level
marginal-contribution evaluation before any deployment.

## 7. Build tasks (subagent-driven; NOT verdicts)

1. Commodity loader: read `alt_data/oil/BRENT` and XAUUSD onto the FX date index
   (causal ffill, no interpolation of missing commodity days).
2. `FxTermsOfTrade` strategy with a `form` param (oil / gold / xs) implementing
   the three forecasts with per-leg signs from the map above; register it; 3
   configs under `config/backtesting/tierb/`.
3. Verify the four legs load and the portfolio vol cap is on.

## 8. Trial accounting

Project-wide cumulative N = 134 before this wave, 137 after. DSR is deflated at
the updated N. N is never reduced to help a spec pass.
