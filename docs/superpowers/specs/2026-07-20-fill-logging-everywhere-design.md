# Fill Logging Everywhere -- Design

**Date:** 2026-07-20
**Status:** Approved (brainstorming), pending implementation plan
**Author:** main-loop orchestrator + user

## Problem

The trade-log mandate (strategy-pipeline.md:71, methodology Section 12) currently
carves out an exception: "Validation-harness internals (e.g. per-window
walk-forward runs) may suppress logging, but the primary/representative backtest
for a strategy MUST produce one." That carve-out means the EXACT out-of-sample
fills behind every gated Sharpe/DSR/PBO number are discarded. The FX trade-log
backfill (2026-07-20) could therefore only reconstruct representative
single-full-window logs, not the real gated walk-forward fills. This proves the
gap: we simulate fills we never persist.

User decision: log fills on EVERY simulated run, no exceptions. That includes
every walk-forward window, every optimization-sweep config, and every inner
GridSearchOptimizer probe, across ALL asset classes.

## Goal

Every simulated backtest run in the framework persists its fills to disk, in a
consistent, high-volume-safe, asset-class-agnostic layout, enforced by
strategy-lead. Retire the duplicate single-run trade-log writers in favor of one
shared sink.

## Non-Goals

- Not changing any verdict, gate, or metric. This is logging plumbing only.
- Not re-running any closed campaign. Applies going forward (and any future
  re-gate).
- Not introducing a new file format or dependency. Gzipped CSV only.

## Key Decisions (from brainstorming)

1. **Format/layout:** gzipped CSV everywhere for high-volume internals; a
   run-scoped directory tree; a per-run manifest. (Chosen over Parquet for
   simplicity and zero new dependency; over plain CSV for disk cost at scale.)
2. **Coverage:** all four paths wired now -- the 3 FX walk-forward runners, the
   sweep runner, the vectorbt walk_forward.py validator, and the intraday
   runner. Plus inner GridSearchOptimizer probes.
3. **Cross-asset:** the sink is asset-class-agnostic by construction (two entry
   points matching the only two fill shapes in the repo). Futures and FX
   single-run writers delegate to it; crypto/CSCM verified and wired if bespoke.
4. **Top-level verdict artifact stays plain:** the single-window
   `output/backtests/<strategy>/<start>_to_<end>/trades.csv` remains uncompressed
   (human-browsable, back-compat with the 12 existing logs and the strategy-lead
   hook). Only the run-scoped bulk internals are gzipped.

## Architecture

### Core unit: `FillSink` (`src/backtesting/engine/fill_sink.py`)

One new module owns everything volume-related so no caller reinvents it: the
run-scoped directory, gzip, file naming, and the run manifest.

```
class FillSink:
    def __init__(self, strategy: str, run_id: str, meta: dict,
                 root: Path = Path("output/backtests")):
        # creates output/backtests/<strategy>/runs/<run_id>/, writes meta.json

    def write_window(self, trades_df, window: int, cfg_hash: str | None = None,
                     extras: dict[str, pd.DataFrame] | None = None) -> Path:
        # DataFrame-shape fills (FX spot/spread, futures). extras -> sidecar
        # gz files (e.g. margin_utilization, leverage_utilization).

    def write_portfolio(self, portfolio, window: int, cfg_hash: str | None = None,
                        symbol: str = "") -> Path:
        # vectorbt-shape fills (equity, crypto, sweep, vectorbt WF).
        # Delegates to TradeLogger.export_trades_csv with a .csv.gz path
        # (pandas auto-gzips by extension -- no new dependency).

    def finalize(self, oos_windows: list[int] | None = None) -> Path:
        # concatenates the named windows' fills into trades_oos.csv.gz
        # (the gated-verdict fills) and writes manifest.csv.
```

`run_id = "<UTC-timestamp>_<cfg-hash>"` (e.g. `20260720T014530Z_a1b2c3`). Computed
in normal python runtime (datetime is available; this is not a Workflow script).

### Directory layout

```
output/backtests/<strategy>/
  <start>_to_<end>/trades.csv          # single-window verdict -- unchanged, PLAIN
  runs/<run_id>/
    meta.json                          # strategy, kind (verdict|walkforward|sweep),
                                        # git SHA, config SHA, window spec, cost model,
                                        # start/end, n_windows, created_by
    trades_oos.csv.gz                  # concatenated OOS fills (the gated-verdict fills)
    w01_trades.csv.gz ...              # per-window (walk-forward)
    w03_a1b2c3_trades.csv.gz ...       # per-window-per-config (sweep / probes)
    w03_a1b2c3_margin.csv.gz ...       # optional extras sidecars (futures margin, fx leverage)
    manifest.csv                       # every file + row count + kind + window + cfg_hash
```

### The two fill shapes (why two entry points)

| Shape | Paths that produce it | Sink entry point |
|---|---|---|
| `res.trades` DataFrame | FX spot, FX spread, futures | `write_window` |
| vectorbt portfolio | equity, crypto/CSCM, sweep_runner, vectorbt walk_forward.py | `write_portfolio` |

## Components and wiring

### 1. Shared sink replaces duplicate writers (DRY)
`fx_backtest.py:_write_trade_log` and `futures_backtest.py:_write_trade_log`
(plus its margin_utilization sidecar) are retired; both delegate to
`FillSink.write_window(..., extras=...)`. Equity `TradeLogger` calls in the
single-run path route through `FillSink.write_portfolio`. Single writer, every
asset class. Behavior for the top-level verdict `trades.csv` is preserved.

### 2. FX walk-forward runners (3)
`run_fx_walkforward.py`, `run_fx_carry_seatbelt_walkforward.py`,
`run_fx_wave2_gate.py`: replace `run_fx_backtest(..., log_trades=False)` with
capturing each window's `res.trades` into a per-run `FillSink`, one file per
window. `finalize()` concatenates the OOS windows into `trades_oos.csv.gz`. The
per-window `run_fx_backtest` call must NOT write to the fixed single-window path
(would collide across windows); it returns its trades DataFrame to the runner,
which routes to the sink.

### 3. Sweep runner
`sweep_runner.py` already logs per-symbol via `TradeLogger`. Re-point it at the
`FillSink` so its output uses the run-scoped gz layout, cfg_hash naming, and
manifest -- consistent with every other path.

### 4. vectorbt walk_forward.py validator
After each window: log the OOS `test_portfolio` and the IS best portfolio via
`write_portfolio`. Additionally, `GridSearchOptimizer` (invoked per training
window) logs every inner probe's portfolio. This is the highest-volume path
(windows x gridsize files) -- see Performance below.

### 5. Intraday runner
`run_fx_london_breakout_walkforward.py`: log ALL `OrderEngine.fills` (entries AND
bracket/OCO exits), fixing the entry-only limitation the backfill exposed. Where
an exit is an EOD/time close not routed through the engine as a fill, synthesize
the exit row from the engine's position-close event so every round trip has an
exit.

### 6. Crypto/CSCM
Verify the CSCM backtest path's fill shape; if it produces a vectorbt portfolio
it is already covered by `write_portfolio` and only needs the call wired. If it
has a bespoke path, wiring it is a fast-follow.

## Docs and enforcement

- Rewrite `strategy-pipeline.md:71`: delete the suppression carve-out; state that
  every simulated run persists fills -- per-window, per-config, per-probe -- across
  equity, crypto, futures, and FX.
- Update methodology Section 12 to match.
- `strategy-lead`'s 3106b9c verification checks the run manifest exists (and is
  non-empty) for a gated strategy, not just a single `trades.csv`.

## Performance and known tradeoffs

Logging every inner GridSearchOptimizer probe means gz-writing inside a hot loop.
Sweeps and walk-forward optimization will be materially slower and produce
`windows x gridsize` files. This is the accepted cost of literal no-exceptions.
The sink keeps per-write cost low (single `to_csv` with gzip, no re-open of the
manifest per write -- manifest rows are buffered and flushed in `finalize`), but
the writes are not free. This is a documented characteristic, not a regression.

## Error handling

- `TradeLogger.export_trades_csv` already writes an error CSV on failure rather
  than raising; the sink preserves that (a probe that fails to log leaves an
  error file, never silently vanishes).
- A run with zero trades writes an empty (header-only) gz file and a manifest row
  with row_count 0 -- an explicit "this ran and traded nothing," not a missing
  artifact (which would read as "never ran").
- `finalize()` is idempotent; re-running a run_id overwrites its own dir.

## Testing

Heaviest coverage on `FillSink`:
- run-scoped dir + meta.json creation
- `write_window` DataFrame round-trip through gzip
- `write_portfolio` via TradeLogger to a .csv.gz path
- `extras` sidecar files (margin_utilization, leverage_utilization)
- `finalize` OOS concatenation correctness (only named windows, ordered)
- manifest completeness (one row per file, correct row counts)
- zero-trade run -> header-only gz + row_count 0

Per-path smoke (each proves fills land in the sink):
- one FX WF runner emits per-window gz + trades_oos.csv.gz
- sweep_runner emits per-config gz + manifest
- vectorbt walk_forward emits OOS + IS + probe files
- intraday runner emits BOTH entry and exit fills (regression test for the
  backfill's entry-only gap)

## File inventory

Create:
- `src/backtesting/engine/fill_sink.py`
- `tests/backtesting/engine/test_fill_sink.py`

Modify:
- `src/backtesting/engine/fx_backtest.py` (delegate _write_trade_log)
- `src/backtesting/engine/futures_backtest.py` (delegate _write_trade_log + margin)
- `src/backtesting/optimization/sweep_runner.py`
- `src/backtesting/chunking/walk_forward.py` (+ GridSearchOptimizer probe logging)
- `scripts/backtest_scripts/run_fx_walkforward.py`
- `scripts/backtest_scripts/run_fx_carry_seatbelt_walkforward.py`
- `scripts/backtest_scripts/run_fx_wave2_gate.py`
- `scripts/backtest_scripts/run_fx_london_breakout_walkforward.py`
- `.claude/rules/strategy-pipeline.md` (line 71 carve-out)
- `docs/methodology/backtesting.md` (Section 12)
- `.claude/agents/strategy-lead.md` (manifest check)

## Governance

- Superpowers implementation runs in an isolated git worktree (repo CLAUDE.md).
- Any phase that RUNS a walk-forward/gate/smoke to VERIFY the wiring end-to-end
  on a real strategy is a verdict-adjacent run and goes through strategy-lead
  (the unit tests + fixture-based smokes in the build plan do not; a real-data
  end-to-end validation does).
- Commit by explicit path; orchestrator owns pushes.
