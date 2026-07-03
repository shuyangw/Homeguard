# Parallel Execution Foundation - Design (P)

**Date:** 2026-07-03 · **Status:** approved, pre-plan · **Depends on:** merged futures pipeline (`main` @ 88c8002).

## Goal

One reusable, correct parallel-map primitive, applied to the two expensive futures hot
spots (per-root cache builds, per-window walk-forward), replacing the ad-hoc `&`/`wait`
shell hack used for the carry cache. Correctness (right process/thread choice,
determinism, no registry-write races, bounded workers) is guaranteed and tested, not
hoped for. This is the foundation for W1 (carry IDM), W2 (combine), W3 (new signals) --
every later walk-forward becomes fast.

## Context (verified -- reuse, do not reinvent)

- `src/backtesting/optimization/data_loader.py::get_default_workers()` returns 80% of CPU
  threads (min 1) -- the canonical worker cap. Reuse it.
- `src/backtesting/optimization/grid_search.py` already uses `ProcessPoolExecutor` +
  `as_completed` with pickleable dataclass configs and a top-level worker function
  (`_test_single_params`). That machinery is tied to the equity `BacktestEngine`, so it is
  NOT directly reusable for the futures path -- but its PATTERN is the template.
- `src/experiments.make_trial_callback` + the `TrialCallback` note in grid_search document
  the canonical race-free approach: optimizers stay registry-agnostic; the PARENT records
  results. We apply the same principle.
- `run_futures_backtest` writes to the DuckDB experiment registry via `append_run` in a
  try/except (`src/backtesting/engine/futures_backtest.py:97-111`).
- `walk_forward_carver` (`scripts/backtest_scripts/run_carver_walkforward.py`) runs ~13
  windows SERIALLY; per window it loads the daily panel to find roots-with-data, then calls
  `_run_window` twice (1x cost line ~224, 1.5x cost line ~230), slices the OOS segment, and
  appends. The parent writes ONE final `append_run` (line ~287). Carver/carry are
  parameter-free with NO RNG.

## Architecture

A single generic `parallel_map` over `ProcessPoolExecutor`, used in three call sites (two
built now, one noted). Processes (not threads) because the per-window backtest is
GIL-bound Python. Results returned in INPUT order (determinism). Registry writes happen
only in the parent (race-free).

## Components

### 1. `src/backtesting/parallel.py::parallel_map(fn, items, max_workers=None)`

- Signature: `parallel_map(fn: Callable[[T], R], items: Sequence[T], max_workers: int | None = None) -> list[R]`.
- `max_workers` defaults to `min(get_default_workers(), len(items))`; `max_workers <= 1`
  runs SERIALLY in-process (debugging + the determinism test + tiny item lists).
- Uses `ProcessPoolExecutor`; submits each item with its index; collects via `as_completed`;
  **always returns results reassembled in input order** (the determinism guarantee -- no
  unordered mode; YAGNI until something needs it).
- **Fail-fast:** the first worker exception is re-raised in the parent (repo fail-loud
  stance); remaining futures are cancelled.
- `fn` and every item must be picklable (top-level functions + plain dict/tuple specs).

### 2. `register` flag on `run_futures_backtest`

- `run_futures_backtest(config, register: bool = True)`. When `register is False`, SKIP the
  `append_run` block entirely (return `run_id=None`). Default True preserves all existing
  callers (single-pass backtests, the runner) byte-for-byte.
- Purpose: parallel per-window workers pass `register=False` so they perform NO registry
  writes -- eliminating concurrent single-writer DuckDB contention. The parent still writes
  the single final walk-forward entry.

### 3. Parallelize `walk_forward_carver` by window

- Extract the per-window body into a top-level, picklable
  `_process_window(spec: dict) -> dict` where `spec` carries `{train_start, test_start,
  test_end, universe, capital, vol_target, strategy_name, strategy_params}`. It loads the
  panel, computes `window_universe` (roots with data), runs `_run_window` at 1x and 1.5x
  cost with `register=False`, slices the OOS segments, and returns
  `{window_universe, oos_1x, oos_1_5x, sharpe, ...}` -- everything the aggregation needs.
- Replace the serial `for window in windows` loop with:
  `results = parallel_map(_process_window, window_specs, max_workers=max_workers)`.
- The AGGREGATION after the map (stitch OOS-1x and OOS-1.5x segments IN WINDOW ORDER,
  compute Sharpe/PSR/DSR/PBO, build the result dict, write the single parent `append_run`)
  is UNCHANGED. Because `parallel_map` returns in input order and the strategies are
  RNG-free, the stitched series and every gate metric are identical to the serial path.
- `walk_forward_carver` gains `max_workers: int | None = None`; `main()` may expose
  `--jobs`. Default parallel; `--jobs 1` forces serial.
- Empty/insufficient-data windows are handled exactly as today (skipped) -- `_process_window`
  returns a sentinel the aggregator drops, preserving current behavior.

### 4. Parallelize `build_carry_cache.py` by root

- `build_carry_cache(roots, start, end, max_workers=None)` maps each root through a
  top-level `_build_one(spec) -> str | None` via `parallel_map`. Each root writes its own
  `{root}.parquet` (embarrassingly parallel, no aggregation, no shared state). Add a
  `--jobs N` CLI arg (default `get_default_workers()`).

## Correctness Commitments

- **Process, not thread:** ProcessPoolExecutor -- true parallelism for GIL-bound backtest
  loops.
- **Deterministic == serial:** input-order results + RNG-free strategies => stitched series
  and gate metrics byte-identical to serial. Enforced by an equality test.
- **Race-free registry:** parallel workers use `register=False`; only the parent writes.
- **Bounded:** `get_default_workers()` cap, further capped at item count.

## Testing

- `parallel_map`: (a) returns input order under concurrency (map a shuffled-latency fn,
  assert order); (b) `max_workers=1` serial path returns identical results; (c) worker
  exception propagates to the parent; (d) worker cap respected (<= min(default, n_items)).
- `run_futures_backtest(register=False)`: returns `run_id=None` and performs NO registry
  write (monkeypatch `append_run` to raise if called); `register=True` default unchanged.
- **Walk-forward determinism (the key test):** a small 2-window walk-forward run with
  `max_workers=1` vs `max_workers=2` -> assert identical `oos_sharpe`, `psr`, `dsr`, `pbo`,
  `oos_sharpe_1_5x_cost`, `n_windows` (uses the small cached slice; fast).
- `build_carry_cache(--jobs 2)` on 2 monkeypatched roots -> same parquets as serial.

## Scope / YAGNI

- The `parallel_map` primitive makes "multi-strategy/config parallel runs" (for W3) a
  trivial future one-liner -- NOTED, not built now.
- No new dependency (stdlib `concurrent.futures`).
- No change to the simulator, sizing, strategies, the gate math, or the equity/crypto path
  -- only WHERE per-window backtests run and WHEN the registry is written.
- The optimization framework's own parallelism is untouched (it already has its own).

## Out of Scope (later workstreams)

- W1 carry attribution + IDM; W2 carry+momentum combine; W3 new signals. Each its own
  spec/plan; each benefits from P's fast walk-forward.
