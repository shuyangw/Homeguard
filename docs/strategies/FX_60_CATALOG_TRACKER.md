# FX 60-Strategy Catalog Tracker

Living tracker for the 60-strategy FX catalog (research docs: `~/Downloads/compass_artifact_*.md`,
`~/Downloads/fx_strategy_deep_dive.md`). Tracks, per strategy: current viability, what blocks it,
and test progress. Update this file as strategies are tested or unblocked.

Last updated: 2026-07-19

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
- NOT built: OHLC-into-forecast_panel wiring; intraday engine; spread execution; ML harness;
  oil/equity/full-CB-calendar feeds; EM spot pairs (USDMXN/USDZAR/USDCNH on disk, not in daily cache).

## Summary counts

| Status | Count | Strategies |
|---|---|---|
| READY (test now) | 15 | 3,4,15,16,17,19,31,33,34,39,40,42,43,44,46 |
| OHLC (trivial change) | 8 | 1,6,8,12,27,28,29,47 |
| SPREAD | 4 | 30,35,36,37 |
| BRACKET | 3 | 2,26,60 |
| ML | 6 | 48,49,50,51,52,53 |
| DATA | 2 | 18,55 |
| INTRADAY | 22 | 5,7,9,10,11,13,14,20,21,22,23,24,25,32,38,41,45,54,56,57,58,59 |

---

## Category A -- Momentum / Trend (1-7)

| # | Name | Status | Blocks / needs | BT | WF | Gate | Notes |
|---|---|---|---|---|---|---|---|
| 1 | Dual MA + ATR trail | OHLC | ATR trail needs OHLC; daily EMA-cross ~= existing FxTrend | - | - | - | |
| 2 | Donchian breakout | BRACKET | OHLC channel + stop-entry/pyramid | - | - | - | |
| 3 | TSMOM portfolio | READY | -- | BT | WF | FAIL-naive | OOS -0.02, DSR 0.20, PBO 0.85 (IDM on, 13 win); naive sign form fails gate. Enhanced/param-sweep untested. **2026-07-19 cost re-gate: 0.5x-cost (IBKR-optimistic) OOS Sharpe +0.075 vs -0.02 base -- point estimate flips sign but PSR/DSR/PBO unchanged (still far outside gate); NOT a robustness rescue.** |
| 4 | Cross-sectional momentum | READY | currency_strength artifact | BT | WF | FAIL-naive | OOS -0.05, DSR 0.01, PBO 0.66; naive 63d form fails gate. **2026-07-19 cost re-gate: 0.5x-cost OOS Sharpe +0.058 vs -0.05 base -- same caveat as #3, gate metrics unchanged, NOT a rescue.** |
| 5 | Breakout-pullback | INTRADAY | M15 triggers | - | - | - | |
| 6 | ADX-gated trend | OHLC | ADX needs high/low | - | - | - | |
| 7 | Multi-TF momentum | INTRADAY | D1/H4/M15 stack | - | - | - | |

## Category B -- Mean Reversion (8-14)

| # | Name | Status | Blocks / needs | BT | WF | Gate | Notes |
|---|---|---|---|---|---|---|---|
| 8 | Bollinger reversion | OHLC | daily z-reversion runs; ADX/news extra | - | - | - | test on EURCHF/EURJPY/CHFJPY |
| 9 | RSI(2) fade | INTRADAY | H1 | - | - | - | |
| 10 | Asian range fade | INTRADAY | session | - | - | - | |
| 11 | Hourly z-reversion | INTRADAY | H1 + vol filter (vol_surface ready) | - | - | - | |
| 12 | Keltner reversion | OHLC | ATR | - | - | - | |
| 13 | EOD reversal | INTRADAY | session | - | - | - | |
| 14 | Weekend gap fade | INTRADAY | Sunday-open + intraday | - | - | - | |

## Category C -- Carry (15-19)

| # | Name | Status | Blocks / needs | BT | WF | Gate | Notes |
|---|---|---|---|---|---|---|---|
| 15 | Vol-targeted carry basket | READY | G10 FRED rates current | BT | WF | FAIL-naive | OOS -0.33, DSR 0, PBO 0.73; CONTINUOUS rate-diff form (not the ranked top-3 + crash-filter basket) fails gate. **2026-07-19 cost re-gate: 0.5x-cost OOS Sharpe -0.295 vs -0.33 base -- stays negative, not a rescue.** |
| 16 | Carry-momentum filter | READY | -- | BT | WF | FAIL-enh | Built as FxCarrySeatbelt (#16 swap+EMA50 filter + #19 veto/short), broad G10, daily+weekly. FAIL S&P bar: daily OOS -0.75 / weekly -0.11 vs S&P 0.68 (2014-2026, 3196d). DSR 0. Report FX_CARRY_SEATBELT_WALK_FORWARD.md; 1 deferred variant (12mo-TSMOM leg / graded sizing) remains per pre-reg. **2026-07-19 cost re-gate: 0.5x-cost OOS Sharpe daily -0.491 / weekly +0.020 vs S&P 0.684 -- both still FAIL the S&P bar; weekly flips sign but stays ~34x below benchmark. Not a rescue.** |
| 17 | Swap-aware swing bias | READY | overlay/tilt | - | - | - | carry PnL modeled |
| 18 | EM carry (MXN) | DATA | USDMXN not in daily cache (on disk; MXN rate ready) | - | - | - | buildable via Dukascopy/Massive |
| 19 | Carry-unwind detector | READY | AUD/NZD/JPY/CHF/XAU all present | BT | WF | FAIL-enh | Built as the veto + offensive-short leg of FxCarrySeatbelt. Short earned +1.4% in the Aug-2024 yen unwind (existence proof, N~4-6). Combined strategy FAILs S&P bar (see #16). carry_unwind score reusable (src/backtesting/signals/carry_unwind.py). See #16 for the 2026-07-19 cost re-gate (same combined strategy). |

## Category D -- Session / Time-of-day (20-25)

| # | Name | Status | Blocks / needs | BT | WF | Gate | Notes |
|---|---|---|---|---|---|---|---|
| 20 | London open breakout | READY | intraday engine BUILT | BT | WF | FAIL-enh | First gated INTRADAY result. Filtered Asian-range break (0.25-0.8x ATR width, tier-1 event skip), OCO bracket, conservative 1m fills, 1.2x London spread. FAIL S&P: OOS Sharpe -1.60 vs 0.68 (IS -0.99, DSR 0, 3064 same-dates OOS, 2014-2026). Dies after spread. Report FX_LONDON_BREAKOUT_WALK_FORWARD.md. Intraday engine (src/backtesting/engine/intraday_order_engine.py) now REUSABLE for #21-25. **2026-07-19 cost re-gate: 0.5x-pip/side (IBKR-optimistic) OOS Sharpe -0.748 vs -1.60 base and S&P 0.677 -- big point improvement, still a hard FAIL. Not a rescue.** |
| 21 | NY continuation | INTRADAY | session | - | - | - | |
| 22 | Tokyo JPY-cross MR | INTRADAY | session + synth decomposition | - | - | - | |
| 23 | WMR 16:00 fix | INTRADAY | intraday fix window | - | - | - | |
| 24 | Friday squaring fade | INTRADAY | session | - | - | - | |
| 25 | Session-transition vol | INTRADAY | vol_surface ready; needs intraday | - | - | - | |

## Category E -- Volatility (26-30)

| # | Name | Status | Blocks / needs | BT | WF | Gate | Notes |
|---|---|---|---|---|---|---|---|
| 26 | NR7 squeeze | BRACKET | OHLC range + OCO bracket | - | - | - | |
| 27 | Bandwidth squeeze | OHLC | Keltner/ATR (D1 form) | - | - | - | |
| 28 | ATR-regime switch | OHLC | OHLC + regime artifact (ready) | - | - | - | allocation overlay |
| 29 | Vol-spike fade | OHLC | daily RV form | - | - | - | |
| 30 | Relative-vol pair | SPREAD | bracket + coupled legs | - | - | - | {XAU,XAG} set available |

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
| 35 | AUD/NZD pairs | SPREAD | beta-weighted spread; cointegration artifact ready | - | - | - | AUD/NZD present |
| 36 | Scandi triangle | SPREAD | spread + Brent oil | - | - | - | NOK/SEK present |
| 37 | Cointegration scanner | SPREAD | spread engine; cointegration artifact ready | - | - | - | |
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
| 55 | USDCNH PBOC fix | DATA | USDCNH spot + fix history | - | - | - | on disk, not built |
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
| Beta-weighted spread execution | 4 (30,35,36,37) + #40 pair leg | cointegration artifact | not started |
| Intraday engine | 22 | minute data, spread_model, vol_surface | not started (large) |
| ML meta-label harness | 6 (48-53) | CPCV/DSR | not started |
| EM spot pairs (data) | 2 (18,55) | on disk (Massive/Dukascopy) | not started |

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
