# FX Tier B -- Commodity Terms-of-Trade Wave: RESULTS

**Date:** 2026-07-25 | **Status:** CLOSED -- all 3 pre-registered trials FAIL |
**Pre-registration:** `docs/strategies/research/20260725_fx_tierb_commodity_preregistration.md` (LOCKED 2026-07-25)
**Working copy:** `docs/reports/fx/tierb_commodity_gate.md`

## 1. Verdict

**All three pre-registered specs FAIL the pre-committed gate.** This matches the
registered prediction in pre-registration Section 5 exactly. No spec, parameter,
sign, universe, or gate threshold was changed at any point.

## 2. Setup

Universe: USDCAD, USDNOK (oil legs); AUDUSD, NZDUSD (gold legs). Backtest window
2011-01-01..2026-04-30; FX panel actually spans 2011-01-03..2026-04-22 (3,979 rows).
Walk-forward: anchored 36m-train / 12m-test / 12m-step, **13 OOS windows**, OOS
calendar 2014-01-06..2026-04-20. Data frequency: daily spot FX with a daily
commodity close, causally forward-filled onto the FX date index. Cost model:
standard taker (`src/backtesting/costs/fx.py`); the 1.5x leg is the mandatory
sensitivity gate. Vol target 0.03/instrument, IDM on, weekly rebalance, leverage
cap 10, `execution_lag=1` (no same-bar fills). Fixed params, no sweep: momentum
63d, z-window 252d, clip +-2.

Commodity data: Brent `alt_data/oil/BRENT/daily.parquet`, verified at run time --
4,133 rows 2010-01-04..2026-07-24, close 19.33..127.98 (sane Brent levels), exactly
as pre-registered. Gold = XAUUSD from the validated FX daily cache.

## 3. Sign and alignment pre-check (run on real data BEFORE the gate)

The pre-registration fixes the signs a priori in
`src/data/commodities.py::COMMODITY_LEGS`. Measured on the real panel:

| Check | Required | Measured | Result |
|---|---|---|---|
| corr(oil 63d momentum, USDCAD forecast) | NEGATIVE (oil up -> CAD stronger -> USDCAD down) | **-0.7937** | PASS |
| corr(gold 63d momentum, AUDUSD forecast) | POSITIVE (gold up -> AUD stronger -> AUDUSD up) | **+0.8114** | PASS |
| TOT-XS row-sum across the 4 legs | 0 (market-neutral) | max abs **5.3e-15** | PASS |
| Commodity alignment | forward-fill only, no interpolation | ffill only, confirmed in loader | PASS |

The signs are transmitted correctly and the cross-sectional form is
market-neutral to floating-point precision.

**On the XS form specifically:** with only TWO commodities spread across FOUR
legs, TOT-XS is a **2-group relative tilt** (gold currencies vs oil currencies),
**NOT a 4-way cross-sectional rank**. Both legs within a group carry an identical
forecast. This is a structural property of the pre-registered universe, stated
here so the result is not over-read as a general cross-sectional commodity-FX
test.

Minor: the gold legs carry 2.69% NaN commodity values (~107 leading days where
the XAUUSD cache starts after the FX floor). These fall entirely inside the
315-day warmup (252 z-window + 63 momentum) and never reach a traded forecast.

## 4. Results

| # | Trial | Strategy | Sign (fixed a priori) | OOS Sharpe 1x | OOS Sharpe 1.5x | PSR (runner) | PSR (corrected units) | DSR | PBO | N (before -> after) | S&P corr | IR vs S&P | n_win | n_oos | Verdict |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | TOT-OIL | `FxTotOil` | NEGATIVE on USDCAD/USDNOK | **+0.0505** | **+0.0385** | 0.9979 | 0.5712 | **0.0000** | 0.4927 | 134 -> 135 | -0.1500 | -0.7075 | 13 | 3180 | **FAIL** |
| 2 | TOT-GOLD | `FxTotGold` | POSITIVE on AUDUSD/NZDUSD | **-0.4903** | **-0.5238** | 1.9e-112 | 0.0380 | **0.0000** | 0.1713 | 135 -> 136 | +0.0807 | -0.7634 | 13 | 3200 | **FAIL** |
| 3 | TOT-XS | `FxTotXS` | inherited per leg, market-neutral | **-0.1702** | **-0.1971** | 1.3e-20 | 0.2722 | **0.0000** | 0.1097 | 136 -> 137 | -0.0293 | -0.7092 | 13 | 3200 | **FAIL** |

Gate (pre-committed, unchanged): OOS Sharpe > 0 AND positive at 1.5x cost AND
PSR >= 0.95 AND **DSR >= 0.95** AND PBO < 0.5, plus the S&P benchmark /
marginal-contribution check.

### Per-trial FAIL reasons

- **TOT-OIL** -- the only spec with a positive Sharpe, and it does clear the
  sign, 1.5x-cost, and PBO legs. It fails on **DSR = 0.0000**, decisively. The
  deflated bar at N=134 is **SR_zero = 1.126** annualized; the observed +0.0505
  is ~1/22nd of it. It also fails PSR once the units bug in Section 6 is
  corrected (0.571, not 0.998). The honest reading: the oil -> CAD/NOK
  transmission is real (pre-check corr -0.79) but at this frequency and
  construction its tradable content is **economically trivial, not
  cost-destroyed** -- the edge survives 1.5x costs (+0.0385) simply because
  there is almost no edge for costs to consume. PBO 0.4927 sits just under the
  0.5 threshold, consistent with an absent rather than an unstable signal.
- **TOT-GOLD** -- non-positive OOS Sharpe (-0.4903) that **widens negative at
  1.5x cost** (-0.5238). REJECT on the non-positive-Sharpe short circuit; PSR/DSR
  follow. PBO 0.1713 is low, which here means the failure is *consistent* across
  windows, not noisy: the pre-registered positive sign is reliably wrong over
  this sample.
- **TOT-XS** -- non-positive OOS Sharpe (-0.1702), widening to -0.1971 at 1.5x
  cost. The market-neutral 2-group tilt inherits the gold legs' negative
  contribution. FAIL on the same short circuit.

### S&P benchmark / marginal-contribution check

All three are effectively uncorrelated with the S&P (|corr| 0.03 to 0.15), so
they are genuinely diversifying in *direction* -- but the information ratio vs
S&P is approximately **-0.71 for all three**, i.e. each would have subtracted
from the book. Low correlation cannot rescue a sleeve with no standalone edge:
marginal contribution is negative for all three. **No candidate proceeds to
book-level evaluation.**

## 5. Trial accounting (cumulative N)

Project-wide cumulative trial count, sourced live from
`get_campaign_trial_distribution()` over `output/experiments.duckdb`:

- **N = 134** immediately before this wave -- matches pre-registration Section 8.
- Each trial's DSR was deflated at the N prevailing when it ran: TOT-OIL at
  **134**, TOT-GOLD at **135**, TOT-XS at **136** (each run registers itself
  immediately after its DSR is computed, growing N for the next trial).
- **N = 137 after the wave**, verified post-run. Exactly the 134 -> 137 stated in
  the pre-registration. N was never reduced.
- Trial-Sharpe dispersion v = 0.4278; deflated bar SR_zero = 1.126 (N=134) rising
  to 1.129 (N=137). The pre-registration estimated ~1.05 using v=0.40; the
  realized bar is slightly **higher**, i.e. more conservative than predicted.
- Registry rows went 489 -> 492 (three runs, one per trial). A post-run recompute
  used to capture the dated OOS streams had its registry append **deliberately
  disabled** so it could not inflate N to 140; verified 492 after.

**Explicitly not done:** no sign was flipped after seeing results. TOT-GOLD's
-0.4903 is not license to test +0.4903 -- that would be HARKing and a new trial.
For the record, the counterfactual fails anyway: DSR(+0.4903, N=135) = 4.4e-109,
still ~1/2.3rd of the 1.127 bar. There is no hidden pass being left on the table.

## 6. Apparatus concern found during this wave (does NOT change the verdict)

**PSR is computed in mismatched units in `scripts/backtest_scripts/run_fx_walkforward.py:212`.**
The call is `psr(oos_sharpe, 0.0, n, skew, kurt)` where `oos_sharpe` is
**annualized** (`_annualized_sharpe`) but `n` is the count of **daily**
observations. The formula's `(SR_hat - SR*) * sqrt(n - 1)` term requires SR in
per-period units, so passing an annualized SR inflates the z-score by
**sqrt(252) ~ 15.9**. `src/backtesting/statistics/psr.py` itself is correct and
its docstring warns the units must match; the defect is in the caller.

Effect here: TOT-OIL's headline PSR of 0.9979 is an artifact. On a Sharpe of
+0.0505 the correct PSR is **0.5712** -- i.e. barely better than a coin flip that
the true Sharpe exceeds zero, which is the economically honest statement. Both
values are reported in the table above rather than silently substituting the
corrected one.

**Why the verdict is unaffected:** DSR is the binding constraint and it fails at
0.0000 for all three. DSR is computed through the same inflated path, and because
every observed Sharpe is *below* SR_zero, the inflation pushes DSR *toward* zero;
correcting it can only move these results further from passing, never toward it.
Recomputing TOT-OIL's DSR in correct per-period units still yields ~0. The
direction of the bug is such that it can manufacture a false PSR **pass**, so it
must be fixed before any future near-miss candidate is gated -- but it cannot
have manufactured a false pass in this wave, because nothing passed.

This is filed as remaining work, not silently patched: changing gate code
mid-verdict on a wave that is being adjudicated would itself be a researcher
degree of freedom.

## 7. Conclusion -- scoped exactly

**The commodity terms-of-trade signal family is unproductive FOR THIS
daily-spot-taker construction.** Per pre-registration Section 6 this family now
STOPS: no parameter sweep, no alternative momentum/z-window, no ML variant.

What was actually shown to fail is one specific slice: **exogenous commodity-price
momentum (63d return, 252d z-score, +-2 clip), transmitted to four G10 commodity
currencies with a priori fixed signs, traded daily on spot as a spread-TAKER at
retail cost, weekly-rebalanced, over 2014-2026 OOS.**

This does **NOT** establish that commodity currencies have no edge, nor that
terms-of-trade is not a real macroeconomic channel -- the pre-check confirms the
transmission is present and correctly signed (corr -0.79 / +0.81). It establishes
that this channel, sampled at daily frequency through a momentum-z construction
and executed as a taker, does not yield a deflated edge against a search that has
now spent 137 trials. Untested and still live in principle: the same terms-of-trade
channel at other frequencies, expressed as a liquidity provider rather than a
taker, or via a non-momentum functional form (e.g. terms-of-trade *level* vs a
fair-value anchor rather than momentum). Those are separate pre-registrations, not
extensions of this one.

## 8. Artifacts

- Gate JSON: `output/tierb/tot_oil.json`, `tot_gold.json`, `tot_xs.json`
- Gate table: `output/tierb/gate_table.csv`
- Dated OOS return streams: `output/tierb/oos_stream_tot_{oil,gold,xs}.csv`
- Fills (mandatory, verified non-empty before the verdict was accepted):
  - `output/backtests/FxTotOil/runs/20260726T014143Z_ce74b2/` -- `trades_oos.csv.gz` 1,220 fills, `manifest.csv` 53 rows
  - `output/backtests/FxTotGold/runs/20260726T014151Z_e30f67/` -- `trades_oos.csv.gz` 1,247 fills, `manifest.csv` 53 rows
  - `output/backtests/FxTotXS/runs/20260726T014158Z_3a11ca/` -- `trades_oos.csv.gz` 2,457 fills, `manifest.csv` 53 rows
  - All OOS fill ranges 2014-01-06..2026-04-20, matching the gated OOS return series.
- Registry: `output/experiments.duckdb` -- 3 runs + 9,580 return-stream rows appended.
