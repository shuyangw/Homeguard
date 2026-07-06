# FX Data Acquisition + Computation Layer - Design

**Date:** 2026-07-06
**Status:** Approved (design), pending implementation plan
**Scope:** Foundational data + compute layer for the 60-strategy FX catalog
**Companion research:** `~/Downloads/compass_artifact_wf-4265ee05...md` (report), `~/Downloads/fx_strategy_deep_dive.md` (deep dive)

## Purpose

Build the complete data-acquisition and derived-artifact layer required to research the
60-strategy FX catalog on top of the already-merged spot-FX daily backtesting vertical
(`asset_class: fx`). The layer acquires every external feed the catalog needs (keyless
wherever possible) and computes the shared research artifacts many strategies reuse.

Guiding directive: acquire keyless where a keyless source exists; flag anything that
needs an API key. The resolved posture is 100% keyless except one optional item (the
OANDA broker swap table), which has a working FRED-derived proxy and therefore blocks
no strategy.

## Context (verified on disk, 2026-07-06)

- FX prices: all 80 pairs present at minute resolution under
  `Stock_Data/fx/massive/1min/symbol=<PAIR>/year=/month=/data.parquet` (canonical 8-col
  schema, 2011+). No FX price data needs acquisition.
- Daily cache: `Stock_Data/fx_daily/` built for 8 pairs only
  (EURUSD, USDJPY, USDCHF, EURJPY, EURCHF, CHFJPY, XAUUSD, XAGUSD) and it is
  CLOSE-ONLY (columns `fx_date`, `close`); open/high/low were discarded at build time.
- FX quotes (bid/ask spread): present for 5 majors under
  `Stock_Data/fx/massive/quotes_minute_aggregated/` (AUDUSD, EURUSD, GBPUSD, USDCAD, USDJPY).
- FRED rates: 4 series present (`Stock_Data/alt_data/fred/`): DFF (USD), ECBDFR (EUR),
  IRSTCI01CHM156N (CHF), IRSTCI01JPM156N (JPY).
- Macro calendar: partial US-only yaml present (`config/macro_calendar/{cpi,fomc,nfp}.yaml`).
- No local equity-index, oil, VIX, or DXY data.
- Acquisition registry exists: `src/data/acquisition/` with `BaseDownloader` (ABC),
  `manager`, `manifest`, `schemas`, `status_tracker`, and plugins including `fred_rates`,
  `alpaca_equities`, `alpaca_news`, `massive_fx_flat`, `massive_fx_quotes_flat`,
  `databento_futures`, `cftc_cot`. `BaseDownloader` is symbol-partitioned time-series
  oriented (per-symbol API fetch -> hive parquet, atomic `.tmp` + `os.replace`).
- Backtest engine reads the daily cache: `src/backtesting/engine/fx_backtest.py`
  passes ONLY the close panel to `strategy.forecast_panel(close)`. Idiom is
  continuous-forecast + periodic-rebalance (no stop/target/bracket primitives).

## Scope

### In scope
- G10 daily-cache build (14 pairs) + optional breadth crosses (6).
- Daily OHLC cache rebuild (add open/high/low; fixes close-only limitation).
- FRED rate pulls for G10 (GBP, CAD, AUD, NZD, NOK, SEK) and EM (MXN, ZAR, SGD).
- External feeds: Brent oil, equity indices (S&P/EuroStoxx/Nikkei), holiday calendars,
  CB/econ calendar (curated yaml).
- Spread cost model (real quotes for 5 majors, synthetic for the rest).
- Computed artifacts: vol surface, currency strength, PCA dollar factor, cointegration
  scan, regime layer, event registries.
- CPCV + Deflated-Sharpe validation harness (extends existing PSR/DSR/PBO gate).

### Out of scope (deferred)
- LLM event/sentiment layer (strategy #52) and the LLM modifications of #29 and #40.
  No Anthropic key is used. #29 and #40 remain in the catalog with rule-based baselines only.
- OANDA broker swap-table archiver: interface designed and key-flagged, but not activated.
  The FRED-derived carry proxy (current behavior) is the default.
- USDCNH official PBOC fix feed (strategy #55): stubbed, deferred.
- Intraday backtest engine, stop/bracket primitives, beta-weighted spread execution,
  and the ML strategy implementations themselves. This spec builds DATA + ARTIFACTS,
  not the strategy engines that consume them.

## Architecture

Two registered families under the existing `src/data/acquisition/` infrastructure,
driven by one CLI.

### Family 1: Acquisition plugins (external feeds)
Symbol/series feeds subclass the existing `BaseDownloader`. Non-symbol feeds (calendar,
holidays) use a thin `FeedFetcher` sibling with the same manifest/atomic-write contract.
Each declares a class-level `REQUIRES_KEY` attribute (env var name or `None`) that the
CLI surfaces so key status is visible before any run.

### Family 2: Artifact builders (derived data)
New lightweight base `ArtifactBuilder`:
- `inputs() -> list[str]` names upstream feeds/builders.
- `build() -> Path` computes and writes the artifact.
- `output_subdir` conventional path under `artifacts/fx/`.
The manager topologically resolves `inputs()` so deep chains
(daily_ohlc_cache -> vol_surface -> regime) build in order without a separate DAG engine.

### CLI
`python -m src.data.fx_pipeline`:
- `list` shows every feed + builder with its key status and up-to-date state.
- `build <name>` or `--all` builds by name, resolving dependencies.
- `--skip-existing` no-ops via the existing manifest.

## Key posture (the core directive)

### Keyless, build now
| Feed / artifact | Source |
|---|---|
| FRED short rates (G10 + EM) | `fred_rates` plugin, fredgraph CSV (keyless) |
| Brent crude | yfinance `BZ=F` |
| Equity indices | yfinance `^GSPC`, `^STOXX50E`, `^N225` |
| Holiday calendars | `holidays` / `pandas-market-calendars` library (no API) |
| CB / econ calendar | curated yaml, API-ready loader |
| All 8 computed artifacts | local compute |

### Keyed and FLAGGED (not activated in this spec)
| Feed | Needs | Impact if absent |
|---|---|---|
| OANDA broker swap table | OANDA v20 account + key | Falls back to FRED carry proxy (current behavior). Archive-forward only. Blocks nothing. |
| USDCNH PBOC fix | vendor/scrape | Only #55 affected; deferred as stub. |

No Anthropic, Alpaca, or paid-calendar keys are required. The single external-key gap is
the optional OANDA swap table.

## Storage layout

External feeds -> `local_storage_dir/alt_data/<feed>/`:
- `alt_data/fred/<SERIES_ID>/` (existing pattern, extended)
- `alt_data/oil/BRENT/`, `alt_data/equity_index/{SPX,STOXX50E,N225}/`
- `alt_data/holidays/` (cached generation)
- `config/macro_calendar/*.yaml` (existing; extended with CB decisions)

Price caches -> `local_storage_dir/`:
- `fx_daily/symbol=<PAIR>/...` rebuilt to carry o/h/l/c, extended to G10+ pairs.

Computed artifacts -> `local_storage_dir/artifacts/fx/<name>/`:
- `spread_model/`, `vol_surface/`, `currency_strength/`, `pca_dollar/`,
  `cointegration/`, `regime/`, `event_registries/`.

Validation harness -> code in `src/backtesting/validation/` (no data artifact).

All writes atomic (`.tmp` + `os.replace`, per `BaseDownloader._save_partitioned`) and
recorded to the existing manifest.

## Feeds (detail)

| Feed | Impl | Source | Output | Serves |
|---|---|---|---|---|
| FRED rates | extend `fred_rates` plugin | fredgraph CSV; add GBP/CAD/AUD/NZD/NOK/SEK (+ MXN/ZAR/SGD) | `alt_data/fred/<SERIES>/` | #15, #16, #19 |
| Brent oil | new `oil_yfinance` | yfinance `BZ=F` daily | `alt_data/oil/BRENT/` | #36, #57 |
| Equity indices | new `equity_index_yfinance` | yfinance `^GSPC`,`^STOXX50E`,`^N225` | `alt_data/equity_index/` | #32, #33 |
| Holiday calendars | new `holidays_builder` | `holidays`/`pandas-market-calendars` | `alt_data/holidays/` | #34, #25 |
| CB/econ calendar | curated yaml + loader | extend `config/macro_calendar/` | `config/macro_calendar/*.yaml` | blackouts, #60 |

### FRED series-ID validation gate (design risk)
An invalid FRED series ID previously (CHF: IRSTCB01CHM156N) returned an HTML error page
and silently zeroed carry. Every new series gets a fetch-time gate: reject empty / HTML
error page, require values in a plausible policy-rate range, require coverage across the
backtest window. Fail loud, never silently zero. Series IDs for the new currencies must
be chosen and verified during implementation (do not assume).

## Artifact builders (detail)

### Foundation
- daily_ohlc_cache - inputs: fx minute. Aggregate minute -> daily o/h/l/c for the 8
  built + 14 G10 (+ optional crosses). Output `fx_daily/`. Fixes the close-only cache.
- spread_model - inputs: fx quotes (5 majors) + minute. Per-pair, per-hour spread;
  synthetic model for the ~75 non-quote pairs (tier x time-of-day multipliers anchored to
  the 5 real ones) with rollover and Sunday-gap widening. Output
  `artifacts/fx/spread_model/`. The cost gate for every strategy.

### Shared computes
- vol_surface - inputs: minute + daily_ohlc_cache. Per-pair hour-of-week realized-vol
  surface. Serves #11, #25, #34, #57, #59.
- currency_strength - inputs: daily_ohlc_cache. Per-currency strength vectors from the
  panel. Serves #4, #44, #54, #58.

### Factor / stat
- pca_dollar - inputs: daily_ohlc_cache (USD pairs). PC1 dollar factor + residuals.
  Serves #39, #54.
- cointegration - inputs: daily_ohlc_cache (cross set). Rolling Engle-Granger scan + OU
  half-life + tradeable-pair registry. Serves #35, #36, #37.

### Regime / registries
- regime - inputs: daily_ohlc_cache + vol_surface + rates. ATR-ratio -> optional HMM ->
  gold-state composite. Serves #6, #28, #42, #46, #49.
- event_registries - inputs: daily_ohlc_cache + rates + calendar. Labeled unwinds /
  vol-spikes / correlation-breaks. Serves #19, #29, #40, #49.

### Validation
- CPCV/DSR harness - code in `src/backtesting/validation/`, extends the existing
  PSR/DSR/PBO gate with combinatorial purged cross-validation and deflated Sharpe using
  the project-wide cumulative trial count. Serves all.

### Resolved build order (topological, from inputs())
```
FRED / oil / equity / calendar / holidays   (leaf feeds)
minute -> daily_ohlc_cache -> { vol_surface, currency_strength, pca_dollar, cointegration }
daily_ohlc_cache + vol_surface + rates      -> regime
daily_ohlc_cache + rates + calendar         -> event_registries
quotes + minute                             -> spread_model
```

## Pairs

G10 core build set (14, all on disk at minute; none in daily cache yet):
GBPUSD, USDCAD, AUDUSD, NZDUSD, AUDNZD, AUDJPY, NZDJPY, EURNOK, EURSEK, USDNOK, USDSEK,
NOKSEK, NOKJPY, SEKJPY.

Optional breadth crosses (6): EURGBP, GBPJPY, GBPCHF, EURAUD, EURCAD, CADJPY.

Non-G10 (separate, for the flagged/EM strategies): USDMXN, USDZAR, USDCNH.

FRED short rates to pull for carry legs: GBP, CAD, AUD, NZD, NOK, SEK (+ MXN, ZAR, SGD
for EM). Metals carry = 0 (no rate needed).

## Testing

- Feeds: mock HTTP/lib response; assert parse + validation gate (FRED HTML-error
  rejection, plausible-range, window coverage). Key-flag surfaced correctly.
- Builders: small synthetic input -> known output (minute->daily aggregation;
  vol_surface hour-of-week bucketing; PCA on a toy 2-factor panel; cointegration on a
  constructed cointegrated series; regime transitions on a scripted vol series).
- Cache-rebuild regression: new OHLC `fx_daily/` reproduces the existing 8-pair close
  series exactly (only o/h/l added), so nothing already validated shifts.
- Determinism: reproducible builders; seed anything stochastic (HMM).
- Integration: end-to-end pipeline on the 8-pair set builds every artifact without error;
  `--skip-existing` no-ops on rerun.

## Build-order phasing (feeds the implementation plan)

| Phase | Delivers | Unblocks |
|---|---|---|
| 1 Foundation | daily_ohlc_cache (OHLC) + G10 pair build + FRED G10 rate pulls | Bucket R/O + G10 daily strategies |
| 2 Cost honesty | spread_model | validity of every strategy |
| 3 External feeds | oil, equity indices, holidays, CB calendar yaml | #32, #33, #34, #36, #57 |
| 4 Shared computes | vol_surface, currency_strength | #4, #11, #25, #44, #54, #58, #59 |
| 5 Factor/stat | pca_dollar, cointegration | #35, #36, #37, #39 |
| 6 Regime/registries | regime, event_registries | #6, #19, #28, #29, #40, #42, #46, #49 |
| 7 Validation | CPCV/DSR harness | all |
| Deferred | OANDA swap archiver (key), CNH fix (stub), LLM layer (shelved) | #15/#17 fidelity, #55, #52 |

Phases 1-2 alone make the ~17 currently-runnable strategies trustworthy.

## Open risks / notes

- Kurtosis-437 data spike: the existing daily cache carries a thin-minute-data bad-close
  artifact. The OHLC rebuild + spread_model + event_registries (vol-spike labeling) should
  surface and let us quarantine it before strategy verdicts. Any strategy inherits it until
  then.
- Known v1.1 engine limitation (out of scope here): gap-spanning MTM move dropped on
  reopen (2020-10/11 EURUSD).
- FRED series-ID selection for new currencies is unverified and must be validated at
  implementation (see validation gate).
- Synthetic spread model for the 75 non-quote pairs is an approximation; calibrate against
  the 5 real-quote majors and document the residual uncertainty per tier.
