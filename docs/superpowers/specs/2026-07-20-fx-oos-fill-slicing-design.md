# FX Walk-Forward OOS Fill Slicing -- Design

**Date:** 2026-07-20
**Status:** Approved (brainstorming), pending implementation plan
**Author:** main-loop orchestrator + user

## Problem

Independent validation of a demo FxCarrySeatbelt walk-forward run
(`output/backtests/FxCarrySeatbelt/runs/20260721T021244Z_ec42e6/`) found that
`trades_oos.csv.gz` -- documented as "the actual gated-verdict fills" -- is not
the out-of-sample fills:

- spans 2018-02-27 to 2023-12-21 (the gate's OOS is ~2021-2024, 782 days),
- 864 of 2237 rows (39%) are dated before 2021-01-01 (guaranteed in-sample),
- 502 duplicate `(date, pair)` rows from overlapping adjacent windows.

Root cause: each walk-forward window runs `run_fx_backtest` over
`[train_start, test_end]` (line 59-66 of the seatbelt runner) because the
strategy needs warm-up history, and logs `res.trades` for that whole range. The
gate, by contrast, scores only `_oos_returns_dated(equity, dates, test_start)`
(returns from `test_start` onward). Neither the runner nor `FillSink.finalize`
slices the logged fills to `[test_start, test_end]` before concatenating. So the
per-window fill files are full-window runs and `trades_oos` is their raw
concatenation -- training-contaminated and overlap-duplicated.

## Goal

Make `trades_oos.csv.gz` for the FX walk-forward runners contain exactly the
out-of-sample fills (each window sliced to its `[test_start, test_end]` OOS
segment, non-overlapping), so it matches the gated OOS return series. Keep the
per-window files as the full-window runs (nothing discarded).

## Non-Goals

- No change to any verdict, gate metric, or the OOS return computation. This
  only fixes what the fill LOG contains.
- No re-run of any closed campaign. Applies to future runs (and the demo can be
  re-run to confirm).
- No change to the vectorbt validator path (it already logs OOS-only
  `test_portfolio`; verified, left alone).

## Key decisions (from brainstorming)

1. **Keep full-window per-window files; slice only the concat** (user choice).
   `wNN_<leg>_trades.csv.gz` stays the full `[train_start, test_end]` run
   (honors "log every simulated run" -- in-sample warm-up preserved, nothing
   discarded). Only `trades_oos.csv.gz` is sliced to OOS.
2. **Slice in `FillSink.finalize`**, driven by per-window OOS ranges the runner
   records via a new `set_oos_range`. Keeps the slicing generic (date-column
   based) and the runner change minimal.
3. **Half-open `[test_start, test_end)` per window**, with the single global
   maximum `test_end` (the final window's end) included, so adjacent windows are
   non-overlapping (the shared boundary day belongs to exactly one window) and
   the last OOS day is not dropped. No `(date, pair)` dedup needed -- the slice
   is non-overlapping by construction, keeping `finalize` free of FX-specific
   column assumptions.
4. **Vectorbt path unaffected.** It logs `test_portfolio` (run on
   `load_symbols(test_start, test_end)` -- OOS-only) under `cfg_hash="oos"` and
   records no OOS range, so `finalize` concatenates it as-is. Confirmed OOS by
   inspection; no change.

## Architecture

### FillSink (`src/backtesting/engine/fill_sink.py`)

Add per-window OOS-range tracking and apply it in `finalize`.

```
def __init__(self, ...):
    ...
    self._oos_ranges: dict[int, tuple] = {}   # window -> (start, end) as Timestamps

def set_oos_range(self, window: int, start, end) -> None:
    # Normalize start/end to pd.Timestamp; store keyed by window.
    self._oos_ranges[window] = (pd.Timestamp(start), pd.Timestamp(end))
```

In `finalize(oos_windows=None, oos_cfg_hash=None)`, when building the OOS concat,
slice each window file to its recorded range (if any):

```
global_max_end = max((e for (_, e) in self._oos_ranges.values()), default=None)
frames = []
for w in sorted(oos_windows):
    wpath = self.run_dir / f"w{w:02d}{suffix}_trades.csv.gz"
    if not wpath.exists():
        continue
    df = pd.read_csv(wpath)
    rng = self._oos_ranges.get(w)
    if rng is not None and "date" in df.columns:
        lo, hi = rng
        d = pd.to_datetime(df["date"])
        if hi == global_max_end:
            mask = (d >= lo) & (d <= hi)      # final window: inclusive end
        else:
            mask = (d >= lo) & (d < hi)       # half-open [lo, hi)
        df = df[mask]
    frames.append(df)
```

`set_oos_range` is parent-side state; `finalize` runs in the parent, so no
multiprocessing concern (workers only write full-window files; the jsonl
manifest already handles worker rows). When no range is recorded for a window
(vectorbt path, or any current caller), behavior is unchanged -- concat as-is.

### FX walk-forward runners

`scripts/backtest_scripts/run_fx_carry_seatbelt_walkforward.py` and
`scripts/backtest_scripts/run_fx_walkforward.py`: after building `specs` (which
already carry `test_start`/`test_end` per window) and before `finalize`, record
each window's OOS range on the sink:

```
for s in specs:
    sink.set_oos_range(s["window"], s["test_start"], s["test_end"])
```

`finalize(oos_windows=..., oos_cfg_hash=_leg_tag(1.0))` is otherwise unchanged.
The implementer must READ `run_fx_walkforward.py` to confirm its window/spec
variable names (it was wired identically in Task 8 but may differ in detail) and
apply the same `set_oos_range` calls; if its structure differs, adapt and note
it.

### Docs

Clarify (do not reverse) the Task-14 wording in
`.claude/rules/strategy-pipeline.md`, `docs/methodology/backtesting.md`
Section 12, and `.claude/agents/strategy-lead.md`: `trades_oos.csv.gz` is the
OOS-sliced concatenation (each window cut to `[test_start, test_end)`), and the
per-window `wNN_<leg>_trades.csv.gz` files are the full-window runs (train +
test). After this fix the existing "trades_oos == the actual gated-verdict
fills" claim becomes accurate; the clarification just states the per-window vs
OOS-concat distinction explicitly.

## Data flow

1. Runner builds windows -> per-window `run_fx_backtest([train_start, test_end],
   fill_sink, window, fill_cfg_hash)` writes full-window `wNN_<leg>_trades.csv.gz`
   (workers).
2. Runner records `set_oos_range(window, test_start, test_end)` per window
   (parent).
3. `finalize(oos_cfg_hash="c1x")` reads each `wNN_c1x_trades.csv.gz`, slices to
   its OOS range, concatenates -> `trades_oos.csv.gz` (OOS-only, non-overlapping).
4. Manifest records both the per-window files (full-window row counts) and the
   `oos_concat` row (sliced row count).

## Error handling / edge cases

- **No range recorded** for a window -> no slicing (back-compat; vectorbt path
  and existing finalize tests unaffected).
- **No `date` column** in the window file -> no slicing (portfolio-based files).
- **Boundary day**: half-open `[lo, hi)` per window + inclusive at the global max
  end -> each OOS calendar day counted once, no duplication, last day retained.
- **Empty OOS slice** for a window (no fills in its test segment) -> contributes
  no rows; if all are empty, `trades_oos.csv.gz` is header-only (consistent with
  the zero-trade handling already in the sink).
- Row counts in the manifest for per-window files remain FULL-window counts; only
  the `oos_concat` row reflects the sliced total.

## Testing

FillSink unit tests (`tests/backtesting/engine/test_fill_sink.py`):
- `set_oos_range` + `finalize` slices a window file to `[start, end)`; rows
  outside the range are excluded from `trades_oos.csv.gz` but the per-window file
  on disk is unchanged (still full-window).
- Two adjacent windows sharing a boundary day: the shared day appears exactly
  once in `trades_oos` (non-overlapping); the final window's end day is included.
- No range recorded -> `finalize` concatenates full files unchanged (back-compat;
  existing `oos_cfg_hash` tests still pass).
- Window file without a `date` column + a recorded range -> no slice, no crash.

Integration / validation:
- Re-run the FxCarrySeatbelt demo (via strategy-lead, demonstration-only, no
  verdict) and re-run the validation script: assert `trades_oos.csv.gz` now
  starts at the first `test_start` (~2021), has zero rows before it, zero
  duplicate `(date, pair)` rows, and a row count consistent with the OOS
  segments -- versus the pre-fix 864 in-sample rows + 502 duplicates.

## Files touched

Modify:
- `src/backtesting/engine/fill_sink.py` (`set_oos_range` + `finalize` slicing)
- `tests/backtesting/engine/test_fill_sink.py` (slicing tests)
- `scripts/backtest_scripts/run_fx_carry_seatbelt_walkforward.py` (record ranges)
- `scripts/backtest_scripts/run_fx_walkforward.py` (record ranges)
- `.claude/rules/strategy-pipeline.md`, `docs/methodology/backtesting.md`,
  `.claude/agents/strategy-lead.md` (clarify per-window vs OOS-concat semantics)

## Governance

- Superpowers implementation in an isolated git worktree.
- The re-run demo/validation is a backtest -> routed through strategy-lead
  (sets the sentinel), demonstration-only, no verdict, no registry trial.
- Commit by explicit path; orchestrator owns pushes. macOS/Dropbox git hazard:
  targeted git only (no checkout/bare status/reset).
