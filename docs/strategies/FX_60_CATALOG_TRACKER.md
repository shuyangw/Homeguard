# FX 60-Strategy Catalog Tracker

Living tracker for the 60-strategy FX catalog (research docs: `~/Downloads/compass_artifact_*.md`,
`~/Downloads/fx_strategy_deep_dive.md`). Tracks, per strategy: current viability, what blocks it,
and test progress. Update this file as strategies are tested or unblocked.

Last updated: 2026-07-06

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

`-` not started | `BT` backtested | `WF` walk-forward done | `PASS`/`WEAK`/`REJECT` gate verdict

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
| 3 | TSMOM portfolio | READY | -- | BT | - | REJECT | Sharpe -0.156 @17% vol (2011-26); FX trend decayed post-2010 |
| 4 | Cross-sectional momentum | READY | currency_strength artifact | BT | - | WEAK | Sharpe -0.066 (flat); naive XS-mom no edge on G10 |
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
| 15 | Vol-targeted carry basket | READY | G10 FRED rates current | BT | - | WEAK | Sharpe -0.013 (flat); G10 carry thin (no EM, no crash filter) |
| 16 | Carry-momentum filter | READY | -- | - | - | - | |
| 17 | Swap-aware swing bias | READY | overlay/tilt | - | - | - | carry PnL modeled |
| 18 | EM carry (MXN) | DATA | USDMXN not in daily cache (on disk; MXN rate ready) | - | - | - | buildable via Dukascopy/Massive |
| 19 | Carry-unwind detector | READY | AUD/NZD/JPY/CHF/XAU all present | - | - | - | daily composite z-score |

## Category D -- Session / Time-of-day (20-25)

| # | Name | Status | Blocks / needs | BT | WF | Gate | Notes |
|---|---|---|---|---|---|---|---|
| 20 | London open breakout | INTRADAY | session + OCO | - | - | - | |
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
| 33 | Turn-of-month USD | READY | AUDUSD present | - | - | - | calendar forecast |
| 34 | Holiday / thin-liquidity | READY | holiday calendar (lib ready) | - | - | - | overlay |

## Category G -- Stat-arb / Relative Value (35-42)

| # | Name | Status | Blocks / needs | BT | WF | Gate | Notes |
|---|---|---|---|---|---|---|---|
| 35 | AUD/NZD pairs | SPREAD | beta-weighted spread; cointegration artifact ready | - | - | - | AUD/NZD present |
| 36 | Scandi triangle | SPREAD | spread + Brent oil | - | - | - | NOK/SEK present |
| 37 | Cointegration scanner | SPREAD | spread engine; cointegration artifact ready | - | - | - | |
| 38 | Synthetic cross divergence | INTRADAY | minute/diagnostic | - | - | - | |
| 39 | PCA dollar-factor residual | READY | pca_dollar artifact + USD panel | - | - | - | weekly residual rank |
| 40 | Correlation-breakdown | READY | daily rolling corr; overlay + pair | - | - | - | pair leg needs SPREAD |
| 41 | HF lead-lag | INTRADAY | minute | - | - | - | |
| 42 | RORO regime spread | READY | AUDJPY/CHFJPY/XAU present | - | - | - | spread as net forecast |

## Category H -- Metals (43-47)

| # | Name | Status | Blocks / needs | BT | WF | Gate | Notes |
|---|---|---|---|---|---|---|---|
| 43 | Gold/silver ratio | READY | XAU/XAG clean | BT | - | REJECT | Sharpe -0.284, -7% DD; naive z-reversion anti-predictive |
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

2026-07-06: naive/canonical versions of 4 READY strategies (#3 TSMOM, #4 XS-mom, #15 carry,
#43 gold/silver) all show NO edge on clean G10 daily (Sharpe -0.01 to -0.28). Matches the research:
FX trend decayed post-2010; G10 carry is thin without EM + crash management; simple ratio reversion
is anti-predictive. Survivors (if any) will be the ENHANCED variants (#16 carry+trend, #19 carry-unwind,
regime-gated) or the intraday half. Sizing note: vol_target 0.03 -> ~17% book vol.

## Current focus

Validate the 15 READY strategies through backtest -> walk-forward -> combined gate before
committing to engine builds. Survivors determine which unblock subsystem gets priority.
Record verdicts in the tables above (BT / WF / Gate columns) as each is tested.
