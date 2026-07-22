# FX Positioning / COT Signal Wave -- Pre-Registration

**Date:** 2026-07-22 | **Status:** LOCKED 2026-07-22 (approved; no post-hoc edits to specs/params/gate) | **Owner:** main-loop -> strategy-lead for verdicts

Pre-registration per the North Star: hypotheses, universe, exact specs, gate, trial
count, and PASS/FAIL fixed BEFORE any backtest runs. Once locked, the set below IS
the search; no post-hoc additions or sign-flips. Each spec is a counted trial.

## 1. Motivation -- a genuinely NEW signal family (not a re-roll)

The retail daily/session catalog (G10 + EM, ~19 specs) tested PRICE and RATE
factors executed as a spread-taker, and that slice dies after costs. Per the North
Star principle "a negative bounds the specification tested, not the asset class,"
the untested edge-bearing families remain live. This wave tests the SIGNAL-FAMILY
axis: speculative POSITIONING (CFTC Commitments of Traders), a non-price signal
that reflects who is crowded / who is flowing, on the SAME trusted daily spot
engine + honest taker cost model. This is the family we can test rigorously with
free data now (the maker/liquidity-provision and microstructure-frequency axes are
PARKED -- they need tick/L2 data to model adverse selection honestly, and a naive
version on minute bars would be a fake PASS).

Economic basis: speculative (non-commercial) positioning has two documented FX
effects with OPPOSITE signs -- (a) crowded positioning LEVELS mean-revert
(contrarian sentiment), (b) positioning CHANGES reflect informed flow that trends
(momentum). We pre-register both as SEPARATE trials with signs fixed a priori, so
choosing the winning sign after the fact is impossible.

## 2. Universe and data

**COT8 = EURUSD, USDJPY, GBPUSD, USDCAD, USDCHF, AUDUSD, NZDUSD, USDMXN** -- the 8
CME FX futures with a COT series that map to a validated spot pair. Signal built,
signed "bullish-the-pair" net-spec %OI:
`net%OI = sign * (noncomm_long - noncomm_short) / open_interest`, where sign flips
the USD-per-foreign futures (JPY/CAD/CHF/MXN futures -> short the USDxxx pair).

Data: `alt_data/cot/cot_fx.parquet` (fetched 2026-07-22, CFTC Socrata legacy
futures-only, 10,531 weekly rows, 2000-2026). Spot from the validated daily cache.
Backtest range = COT/price overlap = **2011-2026** (price cache floor).

**Lookahead control (critical):** COT is a TUESDAY snapshot (date D) published the
following FRIDAY (D+3) ~15:30 ET. A reading is applied to trading days on or after
**D+7 calendar days** (a conservative buffer past the D+3 publication), forward-
filled to daily. No bar ever sees a COT value before it was public.

## 3. The wave -- 3 pre-registered specs (signs fixed a priori)

Shared: vol-target 0.03/instrument, IDM on, portfolio vol cap (positioning book is
USD-correlated), leverage cap 10, weekly rebalance (COT frequency). Existing daily
`forecast_panel` + `FxSpotPortfolioSimulator`; standard taker costs (major tier +
EM bps for USDMXN); cost_mults (1.0, 1.5).

1. **COT-CONTRARIAN-TS** (primary). Per-pair time-series contrarian:
   `forecast_i = -zscore(net%OI_i, rolling 156w)`, clipped. Fade crowded
   positioning; long the pair when specs are washed-out short, short when crowded
   long. Sign FIXED negative (contrarian).
2. **COT-MOMENTUM-TS** (distinct opposite mechanism). Per-pair positioning-flow
   momentum: `forecast_i = +zscore(net%OI_i - net%OI_i[4w ago], rolling 156w)`,
   clipped. Follow the direction specs are adding. Sign FIXED positive (momentum).
3. **COT-CONTRARIAN-XS** (distinct construction). Cross-sectional contrarian: each
   week rank COT8 by net%OI; long the least-crowded / short the most-crowded
   (cross-sectional z of net%OI, sign negative), market-neutral within COT8.

Fixed params (no sweep): z-window 156w (3y), momentum horizon 4w, clip +-2 z ->
Carver scale. **Trial count for this wave = 3.** These are all the trials.

## 4. Validation gate (authoritative: docs/methodology/backtesting.md)

- Walk-forward purge + embargo (Sec 3), full 2011-2026 range, IDM on, via the FX WF
  runner. Mandatory run-scoped FillSink -> trades_oos.csv.gz.
- Combined gate (Sec 2): Sharpe, PSR>0.95, DSR>0 deflated by the PROJECT-WIDE
  cumulative trial count (currently 120; +3 here), PBO<0.5.
- Cost: standard taker (major tier + USDMXN EM bps) + mandatory 1.5x cost gate.
- Benchmark: OOS Sharpe vs S&P buy-hold (~0.68) + market-neutrality/S&P-corr; a
  market-neutral book is judged on deflated marginal book contribution.

## 5. Pre-committed PASS / FAIL

- PASS (all): OOS Sharpe > 0 AND positive at 1.5x cost AND PSR>0.95 AND DSR>0
  (deflated for cumulative N) AND PBO<0.5 AND (beats S&P bar OR positive deflated
  marginal book contribution at low S&P corr).
- FAIL: any of -- non-positive OOS Sharpe, sign-flip at 1.5x cost, DSR~0, PBO>=0.5.
  A near-miss needing any post-hoc degree of freedom is a FAIL.

## 6. Stopping rule

If all 3 FAIL: the COT/positioning signal family is unproductive FOR THIS
daily-spot-taker application -- record it (scoped, per the SCOPE principle: this is
NOT "positioning has no edge," only that this daily construction does not clear the
gate) and move to the next family. If any PASS: book-level marginal-contribution
evaluation before any deployment. No optional stopping either way.

## 7. Prerequisite BUILD (subagent-driven; NOT verdicts)

1. COT loader: `alt_data/cot/cot_fx.parquet` -> daily-aligned, publication-lagged
   net%OI panel (a tracked module; the fetch scratch script becomes a proper
   acquisition step).
2. `FxCotPositioning` strategy with a `form` param (contrarian_ts / momentum_ts /
   contrarian_xs) implementing the 3 forecasts; register it; 3 configs under
   `config/backtesting/cot/`.
3. Verify COT8 loads through the WF runner; portfolio vol cap on.

## 8. Trial accounting

Prior project cumulative N = 120 (12 G10 + 7 EM + baseline). This wave adds 3.
DSR for each COT spec is deflated against the updated cumulative N; N is never
undercounted to help a spec pass.
