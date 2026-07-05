# Spot FX Backtesting Platform - Design

Date: 2026-07-05
Status: Approved (brainstorming), pending implementation plan
Author: Homeguard strategy pipeline

## Summary

Build a reusable spot-FX backtesting vertical, structurally parallel to the
existing futures vertical, routed by `asset_class: fx`. The deliverable is the
platform (daily loader, USD-conversion + rate panels, spot simulator with carry
accrual, notional sizing, pip/bps costs, walk-forward + statistical gate), with
two price-only reference strategies (trend EWMAC, nominal value) shipped to
prove and exercise it. Any FX forecast strategy can then be dropped in via a
YAML config.

The futures path is left untouched. Everything asset-agnostic is reused
(walk-forward harness, PSR/DSR/PBO gate, IDM weighting, volatility features,
Carver/Asness forecast logic, StandardReportGenerator, experiment registry,
RunStatus, parallel_map).

## Decisions (from brainstorming)

| Decision | Choice |
|---|---|
| Goal | Reusable FX research platform (infra/breadth; strategies pluggable) |
| Universe | Full cross-rate support up front (general quote-currency -> USD conversion) |
| Reference strategies | FX trend (EWMAC) + FX cross-sectional value (nominal) |
| Carry accrual | Modeled now via FRED interest-rate differentials |
| Value signal | Nominal long-horizon reversal now; interface accommodates PPP later |
| Simulator | Dedicated `FxSpotPortfolioSimulator` (futures sim untouched) |
| Metals (XAU/XAG) | Included, priced with a bps-of-notional cost line |

## Why a dedicated vertical

Spot FX does not fit the equity `PortfolioSimulator` (notional/leverage, not
shares) nor the futures simulator (no contracts/multiplier/SPAN margin, and it
must accrue carry that futures get for free via the futures curve). The futures
module already established the pattern of a separate asset-class path routed off
the raw YAML `asset_class` key; spot FX follows it. A dedicated
`FxSpotPortfolioSimulator` keeps the working, gate-passing futures code at zero
regression risk and keeps each file focused.

## Architecture and module map

### New modules

| Module | Responsibility |
|---|---|
| `scripts/data/build_fx_daily_cache.py` | Resample `fx_1min/` -> `fx_daily/` parquet once (17:00-ET anchor). Mirrors `build_daily_raw_cache.py`. Avoids re-reading 383M rows per backtest. |
| `src/backtesting/data/fx_backtest_loader.py` | `load_fx_daily_panel(pairs, start, end)` -> `(pair, field)` MultiIndex daily panel; builds the USD-conversion panel and the rate-differential panel. |
| `src/data/fx_rates.py` | currency -> FRED short-rate series map; daily forward-filled rate panel used for carry accrual. |
| `src/backtesting/engine/fx_spot_portfolio_simulator.py` | `FxSpotPortfolioSimulator`: daily MTM + carry accrual + leverage cap + bankruptcy floor; `run_sized(...)`. |
| `src/backtesting/utils/position_sizer_fx.py` | Vol-target notional sizing from forecast (Carver analog, units of base currency). Reuses `compute_div_mult`. |
| `src/backtesting/engine/fx_backtest.py` | `run_fx_backtest(config, register, log_trades)` orchestration. Mirrors `futures_backtest.py`. |
| `scripts/backtest_scripts/run_fx_walkforward.py` | Non-overlapping OOS walk-forward + 1x/1.5x cost legs + PSR/DSR/PBO gate. Mirrors `run_carver_walkforward.py`. |
| `config/backtesting/fx_trend.yaml`, `config/backtesting/fx_value.yaml` | Reference strategy configs. |
| `config/universes/fx_spot-2026.csv` | v1 spot universe (G10 USD-legged + selected crosses + XAU/XAG). |

### Edits to existing files (additive, minimal)

- `src/backtest_runner.py`: add an `asset_class: fx` branch next to the futures
  branch (~line 1202) routing to `run_fx_backtest`. No change to
  `_resolve_costs` - the FX runner injects its own `cost_fn` (as futures does),
  leaving the equity/crypto cost path untouched.
- `src/strategies/registry.py`: register FX strategy names (trend, value).
- `src/backtesting/costs/fx.py`: add `fx_round_trip_usd(...)`; keep existing
  `fx_round_trip_pips` as its core.

## Data layer

### Daily bars (17:00-ET anchor)

FX is 24/5; the market convention day boundary is 17:00 ET (New York close,
Sunday 17:00 -> Friday 17:00). `build_fx_daily_cache.py` reads `fx_1min/`
(UTC timestamps), converts to ET, assigns each minute to the FX trading day
whose boundary is 17:00 ET, and takes the last minute close before the boundary
as the daily `close`. Result cached to
`fx_daily/symbol={SYM}/year={YYYY}/month={M}/data.parquet`.

For a daily forecast-rebalance strategy this fully replaces a 24/5 intraday
calendar - no session calendar is needed in v1.

### USD-conversion panel

For cross-rate PnL and USD notional, every currency needs a daily C -> USD rate,
derived from the USD legs present on disk:

- USD -> USD = 1
- Quote is USD (EURUSD, GBPUSD, AUDUSD, XAUUSD): C -> USD = the pair price
- Base is USD (USDJPY, USDCAD, USDCHF): C -> USD = 1 / pair price
- True cross (e.g. EURPLN): convert its quote-currency PnL (PLN) via USDPLN

If a required USD leg for a held pair is missing on a given day, the loader
raises at load time. No silent mis-conversion.

### Rate-differential panel

`fx_rates.py` maps each currency to a FRED short-rate series (USD: EFFR/SOFR;
EUR: ECB MRO; JPY: BoJ overnight; GBP: BoE Bank Rate; and the other G10 policy
rates already ingested in `alt_data/fred/`), daily forward-filled to step
functions. `rate_diff[pair, t] = r_base(t) - r_quote(t)`.

### Gap policy

Known-thin months (2019-09 cross-asset; 2020-10 and 2020-11 EURUSD outage in the
Polygon archive) yield NaN close. The simulator forward-holds the last price
(no MTM that day; carry still accrues) and logs it. Reports flag any window
overlapping a known gap. If an entire pair is empty in-window, fail loud.

## Simulator

`FxSpotPortfolioSimulator`, per pair p holding signed `units_p` of base currency.
Daily loop:

```
base_to_usd[p,t] = px[p,t] * quote_to_usd[p,t]              # 1 base unit in USD
mtm_usd   = sum_p  units_p * (px[p,t] - px[p,t-1]) * quote_to_usd[p,t]
carry_usd = sum_p  (units_p * base_to_usd[p,t]) * rate_diff[p,t] / 365
costs_usd = sum_p  fx_round_trip_usd(p, d_units_p, quote_to_usd[p,t], tier, session)
equity[t] = equity[t-1] + mtm_usd + carry_usd - costs_usd
```

- Costs are charged on rebalance days only, on the traded delta.
- Carry sign convention: long the base currency earns +(r_base - r_quote) on USD
  notional (long the high-yielder is positive carry).
- Leverage cap: gross notional sum_p |units_p * base_to_usd[p,t]| <=
  leverage_cap * equity[t] (default 10x). If exceeded, scale the whole target
  book down. Same interface shape as `MarginModel.check_and_scale`, simpler
  internals (no SPAN offsets).
- Bankruptcy floor: reuse the futures pattern - if equity <= 0, liquidate and
  floor at 0 to prevent negative-equity pct_change blowups that corrupt Sharpe
  and tail statistics.
- Rebalance cadence: daily / weekly / monthly (mirrors futures).

Result object:
- `equity_curve` (pd.Series)
- `trades` (DataFrame, one row per position change: date, pair, units, cost)
- `leverage_utilization` (pd.Series) - the FX analog of `margin_utilization`

## Sizing

`position_sizer_fx.py`, Carver analog in notional terms:

```
units_p = (forecast_p / 10) * capital * vol_target * div_mult_p
          / (base_to_usd_p * ann_ret_vol_p)
```

- `ann_ret_vol_p` from `close_to_close_rv` on daily returns
  (`src/features/volatility.py`, reused).
- `div_mult_p` from `compute_div_mult` (`src/backtesting/utils/idm_weights.py`,
  reused as-is; asset-agnostic).
- Denominator `base_to_usd_p * ann_ret_vol_p` is the USD standard deviation of
  holding one base unit, so the sizer targets equal USD risk per instrument.

## Costs

`fx_round_trip_usd(pair, units_traded, quote_to_usd, tier, session)`:

- Currency pairs: `fx_round_trip_pips(tier, session) * pip_size(pair)
  * abs(units_traded) * quote_to_usd`, with `pip_size = 0.01` for JPY-quoted
  pairs and `0.0001` otherwise.
- Metals (XAU/XAG): `notional_usd_traded * (metals_bps / 10_000)`, with a
  `_METALS_BPS` default (~4 bps), overridable in config. This is the correct,
  scale-invariant unit for metals, which have no standard pip.

Session defaults to a fixed conservative value for daily rebalance (no intraday
timing). Cost model round-trip semantics are unchanged from
`docs/methodology/backtesting.md` Section 4.3.

## Strategy interface and reference strategies

Adopt the futures `SupportsForecastPanel` protocol:
`forecast_panel(close_panel) -> forecast_panel`. Both reference strategies are
price-only and reuse existing forecast logic configured with FX-appropriate
scalars:

- FX trend: multi-speed EWMAC (Carver combined forecast).
- FX value: nominal 5yr-to-1yr reversal (skip recent 252d, lookback 1260d),
  cross-sectionally demeaned (Asness), mirroring the futures value strategy.

Extension seam for PPP value (v2): the protocol permits an optional
`forecast_panel(close, context=None)` where `context` carries rate/CPI panels.
Base strategies ignore it; a future PPP-value strategy consumes it. No CPI
wiring in v1.

## Runner, config, output

`run_fx_backtest(config, register, log_trades)` mirrors `futures_backtest.py`:
load daily panel -> build USD-conversion and rate-diff panels -> strategy
forecast -> daily vol -> `FxSpotPortfolioSimulator.run_sized(...)` ->
StandardReportGenerator -> `append_run(asset_class="fx",
data_frequency="daily")`.

Trade log (gated on `log_trades`, mandatory for the representative run per the
strategy-pipeline rules) to
`output/backtests/fx/<strategy>/<start>_to_<end>/{trades,equity,leverage_utilization}.csv`.

Config YAML fields: `asset_class: fx`, `strategy{name, universe, params}`,
`dates{start, end}`, `backtest{initial_capital, vol_target_per_instrument,
rebalance, cost_mult, leverage_cap, tier, idm, idm_cap}`.

## Evaluation

`run_fx_walkforward.py` mirrors `run_carver_walkforward.py`:

- Non-overlapping OOS windows (default train 36m / test 12m / step 12m); train
  segment is signal warm-up only; keep the OOS-dated equity; stitch per-window
  OOS return series.
- Run 1x and 1.5x cost legs per window.
- Apply the reused `_verdict` gate: PSR >= 0.95, DSR >= 0.95, PBO < 0.25, plus
  the 1.5x cost gate (sharpe_1.5x > 0 and >= 0.5 * sharpe_1x). Trial count = 1
  for parameter-free strategies (honest DSR).
- `RunStatus` heartbeat; `parallel_map` parallelization.
- Reports flag windows overlapping known data gaps.

Metrics recorded per the strategy-pipeline rules; append to the experiment
registry (`output/experiments.duckdb`) with `asset_class="fx"`.

## Testing (TDD - tests first)

Unit:
- USD-rate resolver: JPY = 1/rate, cross via USD leg, missing-leg raises.
- Carry accrual: sign and magnitude on synthetic known rates.
- PnL conversion golden cases: USD-quote, USD-base, true-cross.
- Sizer vol-target.
- `fx_round_trip_usd`: pip path (JPY vs non-JPY) and metals bps path.
- Leverage-cap scaling.
- Bankruptcy floor.

Golden carry test: hold long EURUSD one year at flat price, r_base - r_quote =
+2% -> equity rises ~= 2% * notional. Deterministic, no external data.

E2E: tiny synthetic 3-pair panel (one USD-quote, one USD-base, one cross)
through `run_fx_backtest` producing the three CSVs. Mirrors
`test_futures_backtest_e2e.py`.

Walk-forward test: mirrors `test_futures_walkforward.py`.

## Milestones

- M1 - Data layer: daily cache builder + loader + USD-conversion panel + FRED
  rate map, with unit tests.
- M2 - Simulator core: `FxSpotPortfolioSimulator` (MTM + carry accrual +
  leverage cap + bankruptcy floor) + FX sizer + `fx_round_trip_usd`, TDD. Golden
  carry test passes here.
- M3 - Runner + first strategy: routing + config + registry + trade log + FX
  trend -> first end-to-end equity curve on real data.
- M4 - Value + evaluation: nominal-value strategy + walk-forward + gate +
  reporting -> first honest OOS verdicts for trend and value.

## Out of scope (v1) - seams left for v2

- PPP/CPI value signal (interface seam left via optional `context`).
- Empirical-spread costs from `fx_quotes_minute_aggregated/` for the 5 G7
  majors (v1 uses modeled pip tiers).
- 24/5 intraday session calendar and intraday strategies.
- Live FX trading / broker integration (no live FX broker exists in Homeguard).
- FX options.

## Risks and open items

- Carry-data coverage (v1 scope decision, 2026-07-05): only USD/EUR/CHF short
  rates (+ a JPY long-rate proxy) are on disk in `alt_data/fred/`. GBP/CAD/AUD/NZD
  short rates are not downloaded, so v1 is restricted to USD/EUR/CHF/JPY-legged
  pairs + metals (EURUSD, USDJPY, USDCHF, EURJPY, EURCHF, CHFJPY, XAUUSD, XAGUSD).
  Extending to the full G10 requires pulling the missing FRED foreign short rates
  (keyless pandas-datareader suffices; a FRED API key would ease discovery) on the
  machine that holds the data. This Mac has no `[macos]` storage config, so all
  real-data runs (Task 11, cache build, walk-forward) execute on the EC2/Windows box.
- v1 implementation notes (2026-07-05, post-final-review): (1) carry accrues per
  CALENDAR day (`rate_diff * (d - prev_d).days / 365`), so weekend rollover is
  captured -- an early per-trading-day-bar `/365` draft understated carry ~31% and
  was fixed. (2) Cost tier is per-pair: USD-leg pairs = major, in-universe crosses
  (EURJPY/EURCHF/CHFJPY) = minor, metals via bps. (3) FxTrend tolerates a
  partial cache (strategy built on the present-pairs set).
- KNOWN LIMITATION (v1.1 candidate): a position forward-held across a MULTI-DAY
  data gap does NOT realize the gap-spanning price move on reopen -- `prev_close`
  holds the NaN row, so the accumulated move is dropped, not caught up. Material
  for the 2020-10/11 EURUSD multi-week outage. Any walk-forward window crossing a
  known gap must be read with this caveat; the fix (forward-fill `prev_close` to
  realize the move on reopen) is deferred to v1.1.
- ACCEPTANCE RUN PENDING: Task 11 (build_fx_daily_cache -> run_fx_backtest ->
  run_fx_walkforward, value with `--train-months 72`) has NOT been executed -- it
  requires the EC2/Windows machine that holds the fx_1min + FRED data. No FX code
  has been run against real data yet; the first real OOS gate verdict comes from
  that run.
- Carry-signal fidelity: FRED short rates are step functions and some foreign
  rates lag; G10 coverage is adequate, EM is thinner. The academically correct
  carry input is FX forward points, which we do not have for spot; IR
  differential is a defensible v1 proxy. Validate that any carry-based signal
  survives the DSR gate before trusting it (not shipped in v1, but the accrual
  uses the same data).
- 17:00-ET anchor: confirm the anchor choice does not inject artificial daily
  vol versus a midnight-UTC anchor when computing sizing vol.
- Data gaps (2019-09, 2020-10/11) dent any backtest crossing those months;
  flagged in reports, never silently spanned.
- Metals carry: gold carry is pure USD funding (r_quote only); handled naturally
  by the accrual since XAU has no interest rate (r_base = 0).
