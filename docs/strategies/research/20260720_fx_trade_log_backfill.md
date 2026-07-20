# FX fills-level trade-log backfill -- 2026-07-20

## Scope

Integrity/completeness remediation only. The FX 60-strategy catalog campaign
is CLOSED (see `docs/strategies/research/20260719_fx_catalog_campaign_synthesis.md`);
all 12 gated strategies FAILED the combined statistical gate net of costs.
This backfill does NOT re-gate, re-derive, or change any verdict. It
persists the missing fills-level `trades.csv` artifact (methodology Section
12 / `strategy-pipeline.md` "Trade logging -- MANDATORY") for the 8 FX
strategies whose backtests previously only persisted daily return-streams.
Each strategy's verdict remains FAIL, as closed on 2026-07-19.

No walk-forward, PSR/DSR/PBO, or S&P-relative gate was run. Each strategy
received exactly ONE primary/representative full-window backtest with trade
logging enabled, over the full available data range (2011-01-01 to
2026-04-01 daily / 1-minute FX cache).

## Results

| # | Strategy | trades.csv path | Rows (fills) | Schema | Notes |
|---|---|---|---:|---|---|
| 16/19 | FxCarrySeatbelt (daily cadence) | `output/backtests/fx/FxCarrySeatbelt/2011-01-01_to_2026-04-01/trades.csv` | 5177 | date,pair,units,cost | Primary cadence only (weekly not re-run; both cadences already FAILED and share no distinguishing output path) |
| 20 | LondonBreakout | `output/backtests/fx/LondonBreakout/2011-01-01_to_2026-04-01/trades.csv` | 6162 | date,pair,ts,side,price,qty,day_r | Entry fills only -- see Known Limitation below |
| 33 | FxTurnOfMonth | `output/backtests/fx/FxTurnOfMonth/2011-01-01_to_2026-04-01/trades.csv` | 1679 | date,pair,units,cost | |
| 39 | FxPcaDollarResidual | `output/backtests/fx/FxPcaDollarResidual/2011-01-01_to_2026-04-01/trades.csv` | 4296 | date,pair,units,cost | |
| 42 | FxRoroRegimeSpread | `output/backtests/fx/FxRoroRegimeSpread/2011-01-01_to_2026-04-01/trades.csv` | 1410 | date,pair,units,cost | Red flag: only 2 of 4 universe pairs (AUDJPY, CHFJPY) ever trade; last fill 2024-09-10, over a year before window end. Consistent with closed FAIL verdict; reported honestly, not "fixed" |
| 35 | AudNzdPairs | `output/backtests/fx/AudNzdPairs/2011-01-01_to_2026-04-01/trades.csv` | 360 | date,pair,units,cost | |
| 37 | CointScanner | `output/backtests/fx/CointScanner/2011-01-01_to_2026-04-01/trades.csv` | 1076 | date,pair,units,cost | |
| 30 | VolRatioPair | `output/backtests/fx/VolRatioPair/2011-01-01_to_2026-04-01/trades.csv` | 706 | date,pair,units,cost | |

All 8 files independently verified on disk (row counts, schema, and
irregular per-pair/per-fill row structure -- not one row per calendar day of
aggregate return) by the orchestrating session, not just trusted from
subagent summaries.

## Mechanism per group

**Group A (Seatbelt + Track A daily engine: FxCarrySeatbelt, FxTurnOfMonth,
FxPcaDollarResidual, FxRoroRegimeSpread)** -- no code change. All 4 already
route through `run_fx_backtest()` in `src/backtesting/engine/fx_backtest.py`,
which has a `log_trades: bool` parameter that was simply never passed as
`True` by the walk-forward/gate runners. Backfilled by calling
`run_fx_backtest(cfg, register=False, log_trades=True)` once per config.
`register=False` is load-bearing: it prevents this backfill from appending a
new row to the project-wide DSR trial registry (`output/experiments.duckdb`)
-- this is an artifact re-emission, not a new specification trial.

**Group B (spread engine Track B: AudNzdPairs, CointScanner, VolRatioPair)**
-- one small code change. `FxSpreadPortfolioSimulator.run_spreads()` already
built a `trades` DataFrame internally but `run_spread_backtest()` in
`scripts/backtest_scripts/run_fx_spread_backtest.py` discarded it, returning
only the equity curve's pct-change series. Added an optional
`log_trades: bool = False` parameter (default off, no behavior change for
existing callers including the walk-forward gate) that persists
`trades.csv`/`equity.csv` via a new `_write_trade_log` helper mirroring
`fx_backtest.py`'s convention.

**Group C (LondonBreakout intraday)** -- one small additive code change.
`_pair_daily_returns()` in
`scripts/backtest_scripts/run_fx_london_breakout_walkforward.py` already
drives `OrderEngine` per FX trading day but discards `eng.fills` after
aggregating `strat.day_r`. Added `_pair_trade_log()` / `build_trade_log()`,
additive functions that capture each day's `eng.fills` (the OCO breakout
entry order match) tagged with that entry day's resulting R-multiple, plus a
`--trade-log` CLI flag that short-circuits before the walk-forward gate (so
it cannot affect the PSR/DSR/PBO path). The existing `run()` gate function
and `_pair_daily_returns()` are untouched.

## Known limitation -- LondonBreakout entry-only fills

`OrderEngine` (`src/backtesting/engine/intraday_order_engine.py`) only
appends to `self.fills` on ENTRY (via `match_resting_orders` matching a
resting stop/limit order). Exit events -- stop-loss, profit-target, trailing
stop, and EOD flatten -- are computed inside `_update_position` /
`_flatten`, which return event lists that the existing `OrderEngine.run()`
loop and `LondonBreakoutStrategy.on_bar()` both discard; only the net effect
survives, via `Position.realized_pips` consumed once by
`LondonBreakoutStrategy._book()` into the aggregate `day_r`. Reconstructing
full entry+exit fills would require modifying the shared intraday order
engine (used by the walk-forward gate and potentially other intraday
strategies) -- assessed as out of scope and higher risk for a pure artifact
backfill. The delivered `trades.csv` therefore contains one row per genuine
timestamped ENTRY fill (real `ts`, `price`, `qty`, `side` -- not a daily
return-stream), tagged with the resulting `day_r` for that entry's day. This
satisfies "fills / position changes, not daily P&L" but does not itemize
exits separately. Flagged here for future review if exit-level MAE/MFE
diagnostics are ever needed for this strategy (moot while it remains FAIL).

## Code changes (all reviewed and independently verified by the
orchestrating session before commit)

- `scripts/backtest_scripts/run_fx_spread_backtest.py` -- added `Path`
  import, `_write_trade_log` helper, optional `log_trades` param on
  `run_spread_backtest()`, `--log-trades` CLI flag.
- `scripts/backtest_scripts/run_fx_london_breakout_walkforward.py` -- added
  `_pair_trade_log()`, `build_trade_log()`, `_run_trade_log_backfill()`,
  `--trade-log` CLI flag. Confirmed via import check that the existing gate
  path (`run()`, `_pair_daily_returns()`) is unmodified and still imports
  cleanly.

## Commits (all local on `main`, none pushed)

| Hash | Description |
|---|---|
| `f3adc78` | chore(fx): backfill fills-level trades.csv for 4 closed-catalog strategies (Group A artifacts) |
| `f95029f` | fix(fx): wire trade-log persistence into run_spread_backtest (Group B code) |
| `18a18aa` | data(fx): backfill fills-level trades.csv for 3 FX spread strategies (Group B artifacts) |
| `5345f03` | feat(fx): add fills-level trade-log backfill mode to London Breakout runner (Group C code) |
| `b411aa0` | data(fx): backfill fills-level trades.csv for LondonBreakout (Group C artifact) |

`output/` is gitignored; all `trades.csv` files were force-added
(`git add -f`) individually -- no `equity.csv`/`leverage_utilization.csv`
side artifacts were committed except where already covered by Group A's
`run_fx_backtest` write path (Group A's `_write_trade_log` writes all three
files together, so `equity.csv`/`leverage_utilization.csv` exist on disk for
those 4 strategies but are also gitignored and were not separately
force-added).

## Verification performed

- Every `trades.csv` read from disk directly (not trusted from subagent
  summaries) -- row count, header schema, and irregular per-fill row shape
  confirmed for all 8 files.
- Every commit's `--stat` / diff reviewed directly before or after landing.
- Governance-sensitive paths (`TODO.md`, `docs/reports/fx/`, `settings.ini`,
  `.claude/.strategy-lead-active`, `output/experiments.duckdb`) confirmed
  untouched via scoped `git status`.
- No `git push` performed; all 5 commits remain local on `main`, ahead of
  `origin/main`.
