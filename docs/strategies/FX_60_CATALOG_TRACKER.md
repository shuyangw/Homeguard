# FX 60-Strategy Catalog Tracker

Living tracker for the 60-strategy FX catalog (research docs: `~/Downloads/compass_artifact_*.md`,
`~/Downloads/fx_strategy_deep_dive.md`). Tracks, per strategy: current viability, what blocks it,
and test progress. Update this file as strategies are tested or unblocked.

Last updated: 2026-07-25 (OHLC range-based wave closed)

**OHLC WAVE RESOLUTION -- RANGE-BASED SIGNALS (2026-07-25): all 4 pre-registered
trials (OHLC-KELTNER #12, OHLC-SQUEEZE #27, OHLC-VOLSPIKE #29, OHLC-ADX-TREND #6)
FAIL the pre-registered gate.** This wave tested the first FX signals in the campaign
that require the intraday RANGE -- true range / ATR, ADX directional movement, and
Parkinson high-low volatility -- which were literally unexpressible before the
`wants_ohlc` engine change of 2026-07-25 (the engine previously discarded open/high/low
before calling the strategy). Specs #12/#27/#29 are GENUINELY NEW mechanisms; #6 is
explicitly an ENHANCEMENT of the already-failed #3/FxTrend and is recorded as such,
not as a fresh mechanism. The indicators are demonstrably correct on real data
(ATR(10) = 0.00712, ~71 pips EURUSD; ADX(14) = 27.8; Parkinson RV = 5.5% annualized)
and no-lookahead was confirmed end-to-end by perturbation -- yet none survives the
gate. Three of four are directionally NEGATIVE and widen further negative at 1.5x cost
(Keltner -0.4762 -> -0.6621; Squeeze -0.4783 -> -0.6573; ADX-trend -0.3089 -> -0.4234).
OHLC-VOLSPIKE is the only positive spec (+0.0873 OOS Sharpe, still +0.0114 at 1.5x
cost) and fails decisively on DSR = 0.0000 against a deflated bar of SR_zero = 1.129
at N=137: the edge is economically TRIVIAL rather than cost-destroyed, built from only
594 OOS fills (it gates on Parkinson-RV z > 2 and fires on ~13% of days by design) with
per-window Sharpe spanning -1.90 to +1.93 -- i.e. noise, and it is NOT reported as a
lead. The ADX(14) < 25 gate made the trend mechanism WORSE, not better (-0.31 vs the
-0.02 FxTrend baseline in row #3, DSR 0.20 -> 0.0000); its PBO improvement (0.85 ->
0.27) reflects suppressed exposure, not performance, and per the pre-registration we do
NOT try another ADX threshold. All four are near-uncorrelated with the S&P (|corr|
0.11-0.14) but carry IR vs S&P of -0.60 to -0.88 against an S&P Sharpe of 0.6842 over
the same OOS dates, so marginal book contribution is negative for all four; none
proceeds to book-level evaluation. Cumulative N went 137 -> 141. Two apparatus defects
were filed during this wave (`run_fx_walkforward.py` omits the S&P benchmark leg that
`run_fx_wave2_gate.py` computes, so it cannot adjudicate the full gate alone; and its
`--report` default clobbers the shared FxTrend baseline report path); neither can have
manufactured a false pass since nothing passed, but both should be fixed. Per the
pre-registration stopping rule
(`docs/strategies/research/20260725_fx_ohlc_wave_preregistration.md` Section 7), this
scoped slice STOPS: no sweep, no alternative ATR multiple / ADX threshold / z-window,
no ML variant. This is NOT a claim that range-based signals have no edge or that
ATR/ADX/Parkinson are uninformative (see SCOPE banner below) -- only that these four
constructions, daily spot, spread-TAKER, weekly-rebalanced on G10-22, do not clear the
gate. The frequency objection bites hardest here of any wave so far: a range sampled
once per DAY discards essentially all of the intraday path the mechanism is about.
With this wave the remaining catalog blockers are INTRADAY (21) and ML (6), both
substantial builds. See
`docs/strategies/research/20260725_fx_ohlc_wave_results.md` and
`docs/reports/fx/ohlc_wave_gate.md`.

**TIER B RESOLUTION -- COMMODITY TERMS-OF-TRADE (2026-07-25): all 3 pre-registered
trials (TOT-OIL, TOT-GOLD, TOT-XS) FAIL the pre-registered gate.** This wave tested
the first genuinely EXOGENOUS signal family in the campaign -- not a pattern in FX
itself, nor FX positioning, but the price of the commodity a country EXPORTS (Brent
for CAD/NOK, gold for AUD/NZD), with per-leg signs fixed a priori. The macroeconomic
transmission is demonstrably present and correctly signed on real data (pre-check:
corr(oil momentum, USDCAD forecast) = -0.7937; corr(gold momentum, AUDUSD forecast) =
+0.8114; the XS form is market-neutral to 5.3e-15) -- yet it does not survive the
gate. TOT-OIL is the only positive spec (+0.0505 OOS Sharpe, still +0.0385 at 1.5x
cost) and fails decisively on DSR = 0.0000 against a deflated bar of SR_zero = 1.126
at N=134: the edge is economically TRIVIAL rather than cost-destroyed. TOT-GOLD
(-0.4903) and TOT-XS (-0.1702) are non-positive and widen negative at 1.5x cost. All
three are near-uncorrelated with the S&P (|corr| 0.03-0.15) but carry IR vs S&P of
approximately -0.71, so marginal book contribution is negative for all three; none
proceeds to book-level evaluation. Cumulative N went 134 -> 137. NOTE: no sign was
flipped post-hoc; for the record the counterfactual flipped-gold (+0.4903) also fails
(DSR = 4.4e-109). Per the pre-registration stopping rule
(`docs/strategies/research/20260725_fx_tierb_commodity_preregistration.md` Section 6),
this scoped slice STOPS: no sweep, no alternative momentum/z-window, no ML variant.
This is NOT a claim that commodity currencies have no edge or that terms-of-trade is
not a real channel (see SCOPE banner below) -- only that this 63d-momentum/252d-z
construction, daily spot, spread-TAKER, weekly-rebalanced, does not clear the gate.
An apparatus defect was found and filed during this wave (PSR computed in mismatched
units in `run_fx_walkforward.py:212`, inflating z by sqrt(252)); it cannot have
manufactured a false pass here since nothing passed, but it must be fixed before any
future near-miss is gated. See
`docs/strategies/research/20260725_fx_tierb_commodity_results.md` and
`docs/reports/fx/tierb_commodity_gate.md`.

**COT WAVE RESOLUTION (2026-07-22): all 3 pre-registered COT/positioning trials
(COT-CONTRARIAN-TS, COT-MOMENTUM-TS, COT-CONTRARIAN-XS) FAIL the pre-registered
gate.** Both documented COT mechanisms (crowded-positioning mean-reversion and
positioning-flow momentum), fixed-sign a priori, plus both a per-pair time-series
and a cross-sectional construction of the contrarian leg, all produce non-positive
OOS Sharpe (-0.10 to -0.13) that WIDENS negative at 1.5x cost stress -- a genuine
cost-sensitivity failure, not a marginal edge nudged under by friction. This is the
first NON-price-factor signal family tested against the daily-spot-taker engine
(distinct from the price/rate/carry factors already exhausted by Wave 1/2 and the
EM wave). Per the pre-registration's stopping rule
(`docs/strategies/research/20260722_fx_cot_positioning_preregistration.md` Section 6),
this scoped slice STOPS: no further COT specs, no parameter sweep, no ML variant.
This is NOT a claim that CFTC positioning data has no predictive value in FX (see
SCOPE banner below) -- only that this weekly-net%OI-z-score construction, D+7
publication-lagged, on the daily-spot-taker engine, does not clear the gate. See
`docs/strategies/research/20260722_fx_cot_wave_results.md` and
`docs/reports/fx/cot_wave_gate.md`.

**SCOPE OF THE NEGATIVE FINDINGS (read before quoting "exhausted").** Every "FAIL"
and "RESOLUTION" below is bounded by the SPECIFICATION tested, not by the FX asset
class (see CLAUDE.md North Star, "A negative bounds the specification you tested,
not the asset class"). What has actually been shown to die after realistic costs
is one specific slice: RETAIL-accessible, DAILY/session frequency, SPOT, standard
PRICE/RATE + CARRY factor signals, executed as a SPREAD-TAKER. That is the corner
LEAST likely to hold edge, and it does not. It does NOT establish that FX has no
edge. As of 2026-07-25, the same slice has also been tested for TWO non-price
signal families -- speculative COT positioning (weekly net%OI, D+7-lagged; see COT
WAVE RESOLUTION above) and EXOGENOUS commodity terms-of-trade (Brent/gold momentum
transmitted to CAD/NOK/AUD/NZD; see TIER B RESOLUTION above) -- and both failed in
the same daily-taker construction; those are now tested-and-failed corners too,
scoped identically. The terms-of-trade result is the sharpest illustration of the
scoping rule on record: the economic channel was VERIFIED present and correctly
signed (corr -0.79 / +0.81) and STILL produced no deflated edge, which bounds the
daily-momentum-taker EXPRESSION of the channel, not the channel itself. Untested
families that are where much real FX edge lives -- earning the spread as a liquidity PROVIDER/maker
(adverse-selection-modeled, needs tick/L2 data), MICROSTRUCTURE frequency, and
other non-price SIGNAL families (order-flow, options-implied risk-reversals,
cross-venue/triangular) -- remain LIVE hypotheses, not "exhausted." Read "catalog
exhausted" below as "this retail
daily/session taker-factor slice is exhausted."

**EM WAVE RESOLUTION (2026-07-21): all 7 pre-registered EM7 trials (EM-CARRY
weekly+daily, EM-CARRY-SEATBELT, EM-TSMOM, EM-XSMOM, EM-CARRY-MOM (blend of
#15/#3), EM-MEANREV -- EM-universe variants of #18/#16/#19/#3/#4/#8) FAIL the
pre-registered gate.**
Every trial fails on at least two independent legs (sign at 1.5x EM cost, PSR,
DSR, or PBO). EM's larger carry differentials and structurally different
trend/reversion dynamics do NOT survive realistic EM transaction costs (MXN
3bp/ZAR 6bp/PLN 4bp/HUF 5bp/CNH 5bp/TRY 15bp/INR 8bp half-spread, x1.5
sensitivity) or crash risk. Per the pre-registration's stopping rule
(`docs/strategies/research/20260721_fx_em_wave_preregistration.md` Section 6),
the EM extension of the retail daily/taker-factor slice is declared exhausted:
STOP that slice (no wave-2 EM daily factors, no ML on it) -- NOT a claim that EM FX
has no edge (see SCOPE banner above). See
`docs/strategies/research/20260721_fx_em_wave_results.md` and
`docs/reports/fx/em_wave_gate.md`.

**WAVE 2 RESOLUTION (2026-07-19): all 6 Wave 2 strategies (#33/#39/#42 Track A +
#30/#35/#37 Track B) FAIL the combined statistical gate.** Per the pre-registered
stopping rule (`docs/superpowers/specs/2026-07-19-fx-wave2-selection-design.md`
Section 6), across Wave 1 + Wave 2 the campaign has now tested 8+ distinct
mechanisms (trend, cross-sectional momentum, carry, filtered carry, session
breakout, spread-RV, statistical residual, macro-regime, seasonal, metals
ratio-reversion) spanning the DAILY/session price-factor style space (NOT the full
frequency spectrum -- microstructure and maker/liquidity-provision were never
tested), all failing after realistic taker costs. The campaign DECLARES the finding
and STOPS this slice: no Wave 3 of daily price factors, no ML on it (#48-53). This
is NOT a claim FX has no edge (see SCOPE banner above). See
`docs/strategies/research/20260719_fx_wave2_resolution.md`.

## Status legend (what a strategy needs before it can be tested)

| Tag | Meaning |
|---|---|
| `READY` | Runs on the current daily `forecast_panel` engine + clean G10 data. Testable now. |
| `OHLC` | Needs OHLC passed into `forecast_panel` (cache+loader already carry it; backtest passes only `close`). Trivial change. |
| `SPREAD` | Needs beta-weighted spread execution (rolling hedge-ratio legs). cointegration artifact ready. |
| `INTRADAY` | Needs the intraday engine (minute/hourly loop, session/DST clocks, entry/exit). |
| `BRACKET` | Needs stop/target/OCO order primitives (usually with INTRADAY). |
| `ML` | Needs the ML meta-label harness (triple-barrier + feature pipeline). CPCV/DSR ready. |
| `DATA` | Needs data we do not yet have (EM spot pairs, oil, equity indices, full CB calendar). |

## Test-progress legend

`-` not started | `BT` in-sample backtest | `WF` walk-forward + gate | Gate: `PASS` / `WEAK` / `FAIL-naive` (naive form fails gate; enhanced form untested) / `FAIL-enh` (enhanced form tested, fails gate) / `REJECT` (idea killed across forms)

## Infrastructure status (2026-07-06)

- Universe: 22-pair G10 daily cache, gap-free, cross-vendor-validated (Polygon + Dukascopy), Mon-Fri clean.
- Carry rates: FRED IR3TIB01 (3M interbank) for all non-USD/EUR legs, current through 2026-05.
- Engine: daily `forecast_panel` (continuous forecast + periodic rebalance) + `FxSpotPortfolioSimulator`
  (MTM + calendar-day carry + leverage cap + bankruptcy floor). Spike-clean + weekday filter on load.
- Artifacts built: spread_model, vol_surface, currency_strength, pca_dollar, cointegration, regime,
  event_registries. Validation: CPCV + combined DSR/PBO gate.
- Sizing calibration (2026-07-06): vol_target ~0.03/instrument -> ~17% portfolio vol for the 22-pair book
  (8-pair configs' 0.20 over-leverages the correlated FX book -> blowup). FX pairs are highly correlated;
  a portfolio-level vol cap is a candidate infra improvement.
- NOT built: OHLC-into-forecast_panel wiring; intraday engine; spread execution; ML harness.
- EM daily cache BUILT + VALIDATED (2026-07-21): 8 USD-EM pairs in `fx_daily/` (MXN/ZAR/CNH/TRY/BRL/PLN/HUF/INR,
  2011-2026). 6 are G10-grade gap-free via Dukascopy backfill (99.3-100% cov, 0 significant gaps). Validated
  vs yfinance + FRED H.10 (indep. of both feeds): corr 0.997-0.99996, med|d| 0.08-0.60%. ZAR had 25 sprinkled
  bad Massive closes (2023-2025) -> FIXED by Dukascopy re-fetch (now 1 residual). MXN/CNH/TRY/PLN/HUF/INR clean.
  BRL usable-with-caveat (Massive-only holiday thin prints, not backfillable). See
  `docs/progress/20260721_fx_em_cache_backfill.md`. EM carry still needs a FRED EM-rate check.
- oil (Brent) one keyless `fetch_brent()` away; equity-index + macro-calendar feeds already cached on disk.

## Summary counts

| Status | Count | Strategies |
|---|---|---|
| READY (test now) | 20 | 3,4,6,12,15,16,17,18,19,27,29,31,33,34,39,40,42,43,44,46 |
| OHLC (trivial change) | 4 | 1,8,28,47 |
| SPREAD | 4 | 30,35,36,37 |
| BRACKET | 3 | 2,26,60 |
| ML | 6 | 48,49,50,51,52,53 |
| DATA | 1 | 55 |
| INTRADAY | 22 | 5,7,9,10,11,13,14,20,21,22,23,24,25,32,38,41,45,54,56,57,58,59 |

The 2026-07-25 `wants_ohlc` engine change unblocked the OHLC category, and the OHLC
wave gated 4 of them (#6, #12, #27, #29 -- all FAIL, moved to READY with Gate=FAIL).
The remaining 4 were EXPLICITLY EXCLUDED from that wave with stated reasons
(pre-registration Section 4), NOT merely left untested: **#28** ATR-regime switch is an
allocation OVERLAY and there is no profitable base to modulate; **#1** Dual MA + ATR
trail needs a stateful TRAILING STOP the continuous-forecast engine cannot express (a
build gap, not a signal); **#8** Bollinger's base form is a close-based z-score, i.e.
materially what EM-MEANREV already tested and failed, and its only new part (the ADX
filter) is what #6 tested; **#47** Silver beta amplification is a substitution layer on
#43 GoldSilver, which already failed (OOS -0.31). Per the OHLC stopping rule none of
these is a live daily-spot lead.

---

## Category A -- Momentum / Trend (1-7)

| # | Name | Status | Blocks / needs | BT | WF | Gate | Notes |
|---|---|---|---|---|---|---|---|
| 1 | Dual MA + ATR trail | OHLC | ATR trail needs OHLC; daily EMA-cross ~= existing FxTrend | - | - | - | |
| 2 | Donchian breakout | BRACKET | OHLC channel + stop-entry/pyramid | - | - | - | |
| 3 | TSMOM portfolio | READY | -- | BT | WF | FAIL-naive | OOS -0.02, DSR 0.20, PBO 0.85 (IDM on, 13 win); naive sign form fails gate. Enhanced/param-sweep untested. **2026-07-19 cost re-gate: 0.5x-cost (IBKR-optimistic) OOS Sharpe +0.075 vs -0.02 base -- point estimate flips sign but PSR/DSR/PBO unchanged (still far outside gate); NOT a robustness rescue.** **EM7 variant (EM-TSMOM, 2026-07-21): OOS Sharpe -0.31 (1x) / -0.52 (1.5x), PSR/DSR 0, PBO 0.52 -- FAIL, worse than G10 form. See `docs/strategies/research/20260721_fx_em_wave_results.md`.** |
| 4 | Cross-sectional momentum | READY | currency_strength artifact | BT | WF | FAIL-naive | OOS -0.05, DSR 0.01, PBO 0.66; naive 63d form fails gate. **2026-07-19 cost re-gate: 0.5x-cost OOS Sharpe +0.058 vs -0.05 base -- same caveat as #3, gate metrics unchanged, NOT a rescue.** **EM7 variant (EM-XSMOM, 2026-07-21): OOS Sharpe -1.12 (1x) / -1.48 (1.5x), PSR/DSR 0, PBO 0.54 -- FAIL, the worst result of the EM wave. See `docs/strategies/research/20260721_fx_em_wave_results.md`.** |
| 5 | Breakout-pullback | INTRADAY | M15 triggers | - | - | - | |
| 6 | ADX-gated trend | READY | -- (unblocked 2026-07-25 by `wants_ohlc`) | BT | WF | FAIL | **OHLC wave 2026-07-25 (OHLC-ADX-TREND, N=140):** OOS Sharpe -0.3089 (1x) / -0.4234 (1.5x), PSR 0.1357, DSR 0.0000, PBO 0.2727, 13 win, S&P corr -0.118, IR vs S&P -0.605. FAIL. This is an ENHANCEMENT of #3/FxTrend, not a new mechanism: gating the Carver EWMAC forecast to zero when ADX(14) < 25 made it WORSE (-0.31 vs the -0.02 baseline; DSR 0.20 -> 0.0000). PBO improved 0.85 -> 0.27 only because suppressed exposure stabilizes the window ranking. Baseline CITED not re-run (different apparatus vintage, so not a controlled A/B). Per pre-reg, no alternative ADX threshold will be tried. See `docs/strategies/research/20260725_fx_ohlc_wave_results.md`. |
| 7 | Multi-TF momentum | INTRADAY | D1/H4/M15 stack | - | - | - | |

## Category B -- Mean Reversion (8-14)

| # | Name | Status | Blocks / needs | BT | WF | Gate | Notes |
|---|---|---|---|---|---|---|---|
| 8 | Bollinger reversion | OHLC | daily z-reversion runs; ADX/news extra | - | - | - | test on EURCHF/EURJPY/CHFJPY. **EM7 close-only variant (EM-MEANREV, 2026-07-21): OOS Sharpe -0.69 (1x) / -1.01 (1.5x), PSR/DSR 0, PBO 0.48 -- FAIL. See `docs/strategies/research/20260721_fx_em_wave_results.md`.** |
| 9 | RSI(2) fade | INTRADAY | H1 | - | - | - | |
| 10 | Asian range fade | INTRADAY | session | - | - | - | |
| 11 | Hourly z-reversion | INTRADAY | H1 + vol filter (vol_surface ready) | - | - | - | |
| 12 | Keltner reversion | READY | -- (unblocked 2026-07-25 by `wants_ohlc`) | BT | WF | FAIL | **OHLC wave 2026-07-25 (OHLC-KELTNER, N=137):** OOS Sharpe -0.4762 (1x) / -0.6621 (1.5x), PSR 0.0459, DSR 0.0000, PBO 0.6806 (worst of the wave), 13 win, S&P corr +0.141, IR vs S&P -0.877. FAIL -- directionally negative and widening under cost stress, negative in 10 of 13 windows. GENUINELY NEW mechanism (EMA(20) center, 2.0 x ATR(10) bands; the band width tracks the intraday RANGE, unlike the close-only EM-MEANREV). Per pre-reg, no alternative ATR multiple will be tried. See `docs/strategies/research/20260725_fx_ohlc_wave_results.md`. |
| 13 | EOD reversal | INTRADAY | session | - | - | - | |
| 14 | Weekend gap fade | INTRADAY | Sunday-open + intraday | - | - | - | |

## Category C -- Carry (15-19)

| # | Name | Status | Blocks / needs | BT | WF | Gate | Notes |
|---|---|---|---|---|---|---|---|
| 15 | Vol-targeted carry basket | READY | G10 FRED rates current | BT | WF | FAIL-naive | OOS -0.33, DSR 0, PBO 0.73; CONTINUOUS rate-diff form (not the ranked top-3 + crash-filter basket) fails gate. **2026-07-19 cost re-gate: 0.5x-cost OOS Sharpe -0.295 vs -0.33 base -- stays negative, not a rescue.** |
| 16 | Carry-momentum filter | READY | -- | BT | WF | FAIL-enh | Built as FxCarrySeatbelt (#16 swap+EMA50 filter + #19 veto/short), broad G10, daily+weekly. FAIL S&P bar: daily OOS -0.75 / weekly -0.11 vs S&P 0.68 (2014-2026, 3196d). DSR 0. Report FX_CARRY_SEATBELT_WALK_FORWARD.md; 1 deferred variant (12mo-TSMOM leg / graded sizing) remains per pre-reg. **2026-07-19 cost re-gate: 0.5x-cost OOS Sharpe daily -0.491 / weekly +0.020 vs S&P 0.684 -- both still FAIL the S&P bar; weekly flips sign but stays ~34x below benchmark. Not a rescue.** |
| 17 | Swap-aware swing bias | READY | overlay/tilt | - | - | - | carry PnL modeled |
| 18 | EM carry (EM7: MXN/ZAR/PLN/HUF/CNH/TRY/INR) | READY | FRED EM rates wired + validated (2026-07-21) | BT | WF | FAIL | **2026-07-21 gate (EM-CARRY-weekly + EM-CARRY-daily): weekly OOS Sharpe 0.0245 (1x) / -0.0774 (1.5x), PSR 0.916, DSR 0, PBO 0.136; daily OOS Sharpe 0.0586 (1x) / -0.0988 (1.5x), PSR 0.9995, DSR 0, PBO 0.101. Both sign-flip negative at the mandatory 1.5x EM-cost leg -> FAIL despite passing PSR on the daily cadence.** EM's larger rate differentials do not survive EM's wider (3-15bp) transaction costs. See `docs/strategies/research/20260721_fx_em_wave_results.md`. |
| 19 | Carry-unwind detector | READY | AUD/NZD/JPY/CHF/XAU all present | BT | WF | FAIL-enh | Built as the veto + offensive-short leg of FxCarrySeatbelt. Short earned +1.4% in the Aug-2024 yen unwind (existence proof, N~4-6). Combined strategy FAILs S&P bar (see #16). carry_unwind score reusable (src/backtesting/signals/carry_unwind.py). See #16 for the 2026-07-19 cost re-gate (same combined strategy). **EM7 variant (EM-CARRY-SEATBELT, 2026-07-21): the unwind score's JPY/CHF/AUDJPY/XAUUSD terms are ALL absent from EM7, so `compute_unwind_score` returns identically 0.0 across the full history -- the crash filter never generalized to EM and never engaged (0/3993 nonzero days). The trial ran as a degenerate long-only carry+momentum-gate book (106 fills OOS, monthly win rate 11.5%): OOS Sharpe 0.0775 (1x, near-zero-activity artifact) / -0.0025 (1.5x), PSR 1.0, PBO 0.5633 (>0.5) -- FAIL. A genuine EM crash filter needs EM-native risk-off proxies, not reused G10 JPY/CHF/AUDJPY/XAUUSD terms. See `docs/strategies/research/20260721_fx_em_wave_results.md`.** |

## Category D -- Session / Time-of-day (20-25)

| # | Name | Status | Blocks / needs | BT | WF | Gate | Notes |
|---|---|---|---|---|---|---|---|
| 20 | London open breakout | READY | intraday engine BUILT | BT | WF | FAIL (cost-robust) | First gated INTRADAY result. Filtered Asian-range break (0.25-0.8x ATR width, tier-1 event skip), OCO bracket, conservative 1m fills, 1.2x London spread. FAIL S&P: OOS Sharpe -1.60 vs 0.68 (IS -0.99, DSR 0, 3064 same-dates OOS, 2014-2026). Dies after spread. Report FX_LONDON_BREAKOUT_WALK_FORWARD.md. Intraday engine (src/backtesting/engine/intraday_order_engine.py) now REUSABLE for #21-25. **2026-07-19 cost re-gate: 0.5x-pip/side (IBKR-optimistic) OOS Sharpe -0.748 vs -1.60 base and S&P 0.677 -- big point improvement, still a hard FAIL. Not a rescue.** **2026-07-26 RE-GATE on corrected apparatus (measured hour-of-week spread + fixed trial count N=142/SR_zero 1.1372, vs the old flat hour-blind pip charge deflated against a degenerate SR_zero 0): OOS Sharpe -1.866 (1.0x) / -2.710 (1.5x stress) vs S&P 0.677; PSR 1.8e-11, DSR 7.6e-27, PBO 0.432; CAGR -4.13%, MaxDD -40.8%, Calmar -0.101, monthly win 31.8%, PF 0.713, 6162 trades, avg hold 4.2h. WORSE than -1.60, NOT better as expected: the corrected model cut GBPUSD/EURUSD cost 1.9x/3.7x but EURGBP and GBPJPY are absent from the measured tables and take the conservative 4.0bps unmeasured fallback, making them 1.6x/2.9x MORE expensive -- book-weighted cost rose 30% (1.967 -> 2.553 bps) on realized fills. DECISIVE: a ZERO-COST bound leg (cost_mult 0.0, an upper bound no execution can beat) gives OOS Sharpe -0.138, PF 0.975, monthly win 51.4%, gross pre-cost +43.9R over 6162 trades -- the mechanism has NO gross edge, so the FAIL is invariant to the entire cost model and no spread refinement can rescue it. Re-run adjudicated as an apparatus correction, NOT a new trial (N unchanged at 141; runner applied a stricter 142). Fills log verified on disk with the Section 11.9 exit schema 100% populated on 8676 exit rows; `trade_id` found DEGENERATE (constant 4 -- per-day OrderEngine resets the counter), diagnostic-only, verdict unaffected, fix pending. See `docs/strategies/research/20260726_fx20_london_breakout_regate.md`.** |
| 21 | NY continuation | INTRADAY | session | - | - | - | |
| 22 | Tokyo JPY-cross MR | INTRADAY | session + synth decomposition | - | - | - | |
| 23 | WMR 16:00 fix | INTRADAY | intraday fix window | - | - | - | |
| 24 | Friday squaring fade | INTRADAY | session | - | - | - | |
| 25 | Session-transition vol | INTRADAY | vol_surface ready; needs intraday | - | - | - | |

## Category E -- Volatility (26-30)

| # | Name | Status | Blocks / needs | BT | WF | Gate | Notes |
|---|---|---|---|---|---|---|---|
| 26 | NR7 squeeze | BRACKET | OHLC range + OCO bracket | - | - | - | |
| 27 | Bandwidth squeeze | READY | -- (unblocked 2026-07-25 by `wants_ohlc`) | BT | WF | FAIL | **OHLC wave 2026-07-25 (OHLC-SQUEEZE, N=138):** OOS Sharpe -0.4783 (1x) / -0.6573 (1.5x), PSR 0.0437, DSR 0.0000, PBO 0.6352, 13 win, S&P corr -0.105, IR vs S&P -0.783. FAIL -- negative in 9 of 13 windows and widening under cost stress. GENUINELY NEW mechanism (Bollinger(20,2.0) inside Keltner(20, 1.5 x ATR10) = squeeze ON, forecast 0 while squeezed, sign of 20d close change on release). Compression does precede expansion, but the DIRECTION of the release is not predicted by trailing 20d sign -- a coin flip paid for with spread. See `docs/strategies/research/20260725_fx_ohlc_wave_results.md`. |
| 28 | ATR-regime switch | OHLC | OHLC + regime artifact (ready) | - | - | - | allocation overlay |
| 29 | Vol-spike fade | READY | -- (unblocked 2026-07-25 by `wants_ohlc`) | BT | WF | FAIL | **OHLC wave 2026-07-25 (OHLC-VOLSPIKE, N=139):** OOS Sharpe +0.0873 (1x) / +0.0114 (1.5x), PSR 0.6235, DSR 0.0000, PBO 0.2448, 13 win, S&P corr +0.116, IR vs S&P -0.684. FAIL -- the ONLY positive spec of the wave and it does survive 1.5x cost, but PSR 0.62 < 0.95 and DSR 0.0000 against a bar of SR_zero = 1.131: the edge is economically TRIVIAL, not cost-destroyed. Built from only 594 OOS fills (gates on Parkinson-RV(10) z > 2 over 252d, fires ~13% of days by design) with per-window Sharpe from -1.90 to +1.93 and one undefined window -- noise, explicitly NOT a lead. GENUINELY NEW mechanism (high-low range estimator, not close-to-close). Per pre-reg, no alternative z-window will be tried. See `docs/strategies/research/20260725_fx_ohlc_wave_results.md`. |
| 30 | Relative-vol pair | SPREAD | beta-weighted spread engine BUILT | BT | WF | REJECT | Wave 2 Track B. VolRatioPair, symmetric vol-ratio reversion on {EURNOK,EURSEK}/{AUDUSD,NZDUSD}/{XAUUSD,XAGUSD}, all 6 legs present in every window. OOS Sharpe -0.48 (1.5x: -0.54), PSR 0, DSR 0 (N=111), PBO 0.43, S&P corr 0.14. Non-positive OOS Sharpe -- no edge to deflate. Report: docs/reports/fx/fx_vol_ratio_pair_wave2_gate.md |

## Category F -- Seasonal / Calendar (31-34)

| # | Name | Status | Blocks / needs | BT | WF | Gate | Notes |
|---|---|---|---|---|---|---|---|
| 31 | Day-of-week | READY | overlay/tilt | - | - | - | |
| 32 | Month-end fix | INTRADAY | equity-index data + intraday fix | - | - | - | needs equity feed |
| 33 | Turn-of-month USD | READY | AUDUSD present | BT | WF | REJECT | Wave 2 Track A. FxTurnOfMonth, USD-major seasonal. OOS Sharpe -0.28 (1.5x: -0.36), PSR 0, DSR 0 (N=104), PBO 0.84, S&P corr 0.03. Non-positive OOS Sharpe -- no edge to deflate. Report: docs/reports/fx/fx_turn_of_month_wave2_gate.md |
| 34 | Holiday / thin-liquidity | READY | holiday calendar (lib ready) | - | - | - | overlay |

## Category G -- Stat-arb / Relative Value (35-42)

| # | Name | Status | Blocks / needs | BT | WF | Gate | Notes |
|---|---|---|---|---|---|---|---|
| 35 | AUD/NZD pairs | SPREAD | beta-weighted spread engine BUILT | BT | WF | REJECT | Wave 2 Track B. AudNzdPairs, 120d OLS residual-z, RBA/RBNZ blackout. OOS Sharpe -0.24 (1.5x: -0.30), PSR 0, DSR 0 (N=109), PBO 0.82, S&P corr 0.04. Non-positive OOS Sharpe -- no edge to deflate. Report: docs/reports/fx/fx_audnzd_pairs_wave2_gate.md. FOLLOW-UP SCOPING DIAGNOSTIC (2026-07-25, adversarial-reviewed, no lookahead): tested whether the -0.24 was an artifact of the STATIC hedge-ratio estimator by swapping in a causal Kalman/regularized dynamic beta (delta=1e-4, same universe/entry-exit/costs/windows) -- AudNzdPairsKalman OOS Sharpe +0.42 (1.5x: +0.35), PSR 1.00, DSR ~0 (9.96e-186, N=133), PBO 0.89 -- FAILS the combined gate (DSR and PBO each independently decisive; PSR/Sharpe pass alone). The swing is explained (not a leak): Arm A's rolling-OLS beta is a near-unidentified near-collinear fit (range [0.01,1.66]); the Kalman filter at this delta acts as shrinkage/regularization (range [0.50,0.93]), not genuine time-variation. Result: hedge-ratio mis-specification is ELIMINATED as the explanation for this pair's Wave 2 failure -- both a static and a regularized dynamic hedge ratio fail net of costs. Scope unchanged from the banner above (RETAIL/DAILY/SPOT/SPREAD-TAKER, zero execution lag, one pair, one delta); does NOT extend to other pairs, other deltas, other frequencies, or maker/liquidity-provision execution. Reports: docs/reports/fx/kalman_hedge_ratio_gate.md, docs/strategies/research/20260722_fx_kalman_hedge_ratio_results.md |
| 36 | Scandi triangle | SPREAD | spread engine BUILT; still needs Brent oil | - | - | - | NOK/SEK present |
| 37 | Cointegration scanner | SPREAD | beta-weighted spread engine BUILT | BT | WF | REJECT | Wave 2 Track B. CointScanner, monthly Engle-Granger scan over 22-pair G10, top-5 tradeable set, structural ADF-degradation exit. OOS Sharpe -0.24 (1.5x: -0.31), PSR 0, DSR 0 (N=110), PBO 0.45, S&P corr -0.01. Non-positive OOS Sharpe -- no edge to deflate. Report: docs/reports/fx/fx_coint_scanner_wave2_gate.md |
| 38 | Synthetic cross divergence | INTRADAY | minute/diagnostic | - | - | - | |
| 39 | PCA dollar-factor residual | READY | pca_dollar artifact + USD panel | BT | WF | REJECT | Wave 2 Track A. FxPcaDollarResidual, 22-pair weekly residual rank, major-tier tradeable. OOS Sharpe -0.12 (1.5x: -0.22), PSR 0, DSR 0 (N=105), PBO 0.38, S&P corr 0.02. Non-positive OOS Sharpe. Report: docs/reports/fx/fx_pca_dollar_residual_wave2_gate.md |
| 40 | Correlation-breakdown | READY | daily rolling corr; overlay + pair | - | - | - | pair leg needs SPREAD |
| 41 | HF lead-lag | INTRADAY | minute | - | - | - | |
| 42 | RORO regime spread | READY | AUDJPY/CHFJPY/XAU present | BT | WF | WEAK | Wave 2 Track A. FxRoroRegimeSpread, AUDJPY/CHFJPY beta-weighted, XAUUSD score-only. OOS Sharpe +0.06 (1.5x: -0.03 -- FAILS cost sensitivity), PSR 0.9993, DSR 0 (N=106), PBO 0.17, S&P corr 0.002 (genuinely market-neutral). Positive 1x edge does not survive realistic cost stress and shows no statistical evidence of skill once deflated for the 106-trial search. Report: docs/reports/fx/fx_roro_regime_spread_wave2_gate.md |

## Category H -- Metals (43-47)

| # | Name | Status | Blocks / needs | BT | WF | Gate | Notes |
|---|---|---|---|---|---|---|---|
| 43 | Gold/silver ratio | READY | XAU/XAG clean | BT | WF | FAIL-naive | OOS -0.31, DSR 0, PBO 0.49; plain 756d z-reversion (no momentum-brake/vol-band) fails gate. **2026-07-19 cost re-gate: 0.5x-cost OOS Sharpe -0.299 vs -0.31 base -- stays negative, not a rescue.** |
| 44 | Non-USD gold momentum | READY | synth XAU crosses from FX legs | - | - | - | |
| 45 | Metals-implied FX | INTRADAY | minute/diagnostic | - | - | - | |
| 46 | Gold as risk filter | READY | overlay, computable | - | - | - | |
| 47 | Silver beta amplification | OHLC | substitution layer; XAU/XAG present | - | - | - | |

## Category I -- ML / AI (48-53)

| # | Name | Status | Blocks / needs | BT | WF | Gate | Notes |
|---|---|---|---|---|---|---|---|
| 48 | Gradient-boosted signal filter | ML | meta-label harness | - | - | - | |
| 49 | HMM regime router | ML | HMM + regime artifact | - | - | - | |
| 50 | Triple-barrier meta-labeling | ML | the labeling framework itself | - | - | - | build first of Cat I |
| 51 | Genetic programming | ML | GP + CPCV (ready) | - | - | - | dev-time tool |
| 52 | LLM event/sentiment | ML | deferred (no LLM key this phase) | - | - | - | shelved |
| 53 | Sequence models | ML | neural pipeline | - | - | - | last-priority |

## Category J -- Novel (54-60)

| # | Name | Status | Blocks / needs | BT | WF | Gate | Notes |
|---|---|---|---|---|---|---|---|
| 54 | Cross-sectional intraday strength | INTRADAY | currency_strength ready; needs intraday | - | - | - | |
| 55 | USDCNH PBOC fix | DATA | USDCNH daily cache BUILT 2026-07-21 (G10-grade); still needs PBOC fix history | - | - | - | spot unblocked; fix-history feed still missing |
| 56 | EM local-open effects | INTRADAY | EM pairs + session | - | - | - | |
| 57 | NOKSEK microstructure | INTRADAY | intraday | - | - | - | |
| 58 | Vol-spillover network | INTRADAY | minute VAR | - | - | - | |
| 59 | Hour-of-week vol surface | INTRADAY | vol_surface ready; needs intraday | - | - | - | |
| 60 | Scheduled-news straddle | BRACKET | event OCO + calendar | - | - | - | |

---

## Unblock roadmap (the 4 engine subsystems)

| Subsystem | Unblocks | Prereqs met | Status |
|---|---|---|---|
| OHLC-into-forecast_panel | 8 (Cat: 1,6,8,12,27,28,29,47) | all | not started (trivial) |
| Beta-weighted spread execution | 4 (30,35,36,37) + #40 pair leg | cointegration artifact | BUILT (2026-07-19, `FxSpreadPortfolioSimulator`); gated #30/#35/#37 (all REJECT); #36 (needs Brent oil) and #40's pair leg still unbuilt |
| Intraday engine | 22 | minute data, spread_model, vol_surface | not started (large) |
| ML meta-label harness | 6 (48-53) | CPCV/DSR | not started |
| EM spot pairs (data) | 2 (18,55) | on disk (Massive/Dukascopy) | BUILT 2026-07-21: 6 pairs G10-grade daily cache. #18 gated 2026-07-21 (EM-CARRY + EM-CARRY-SEATBELT + EM-TSMOM + EM-XSMOM + EM-CARRY-MOM + EM-MEANREV, all 7 trials FAIL -- see EM WAVE RESOLUTION above); #55 still needs PBOC fix history. |

## Findings

2026-07-06: 4 READY strategies got a QUICK IN-SAMPLE SCREEN only (one arbitrary config each, IDM off,
weekly, NO walk-forward, NO PSR/DSR/PBO gate). In-sample Sharpe -0.01 to -0.28. These are NOT verdicts --
per methodology a real verdict needs a parameter sweep + walk-forward (purge/embargo) + the combined gate,
IDM on, and cost sensitivity. Suggestive that naive daily factors are weak, but not conclusive; re-evaluate
properly before rejecting. Sizing note: vol_target 0.03 -> ~17% book vol.

## Current focus

Validate the 15 READY strategies through backtest -> walk-forward -> combined gate before
committing to engine builds. Survivors determine which unblock subsystem gets priority.
Record verdicts in the tables above (BT / WF / Gate columns) as each is tested.
