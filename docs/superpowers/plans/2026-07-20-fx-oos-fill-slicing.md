# FX Walk-Forward OOS Fill Slicing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `trades_oos.csv.gz` contain only out-of-sample fills by slicing each walk-forward window's logged fills to its `[test_start, test_end)` range before concatenation, while leaving the per-window files as full-window runs.

**Architecture:** `FillSink` gains a per-window OOS-range registry (`set_oos_range`) that `finalize` applies when building the OOS concat (slice on the `date` column, half-open per window, inclusive at the single global-max end). The two FX walk-forward runners record each window's `[test_start, test_end]` into the sink. No range recorded -> no slicing (back-compat + vectorbt path untouched).

**Tech Stack:** Python 3, pandas (gzip via `.csv.gz`), pytest.

## Global Constraints

- Slice on the `date` column, half-open `[test_start, test_end)` per window; the single window whose `test_end` equals the global maximum recorded `test_end` is inclusive `[test_start, test_end]` (so the last OOS day is not dropped).
- Per-window files on disk stay FULL-window (unchanged); only `trades_oos.csv.gz` is sliced.
- No range recorded for a window, or no `date` column in the file -> no slicing (concat as-is). This preserves back-compat for existing `finalize` tests and the vectorbt validator path.
- ASCII-only, no em dashes, no `print()` (use `from src.utils import logger`).
- Env: fintech conda; tests via `PYTHONPATH=$(pwd) pytest ...`.
- Commit by explicit path only; do NOT push (orchestrator owns pushes). git hazard: only `git add <paths>`/`git commit`/`git log`; never `checkout`/bare `status`/`diff`/`reset`. If a commit is hook-blocked on a trigger word in the message, use `git commit -F <file>`. Runner scripts live under gitignored `scripts/backtest_scripts/` -> stage with `git add -f`.
- Do NOT run any real backtest/walk-forward (hook-blocked + verdict-adjacent). Runner changes are verified with `python -m py_compile` only. The re-run demo + validation is delegated to strategy-lead post-merge, NOT part of this plan.
- Implementation runs in an isolated git worktree.

---

### Task 1: FillSink OOS-range slicing

**Files:**
- Modify: `src/backtesting/engine/fill_sink.py` (`__init__`, add `set_oos_range`, modify `finalize`)
- Test: `tests/backtesting/engine/test_fill_sink.py`

**Interfaces:**
- Consumes: existing `FillSink.__init__` (sets `self.run_dir`, `self._manifest_rows`, `self._manifest_path`), `write_window`, `_record`, and the existing `finalize(oos_windows=None, oos_cfg_hash=None)` whose OOS block reads `w{w:02d}{suffix}_trades.csv.gz` and concatenates.
- Produces: `set_oos_range(self, window: int, start, end) -> None` (stores `(pd.Timestamp(start), pd.Timestamp(end))` in `self._oos_ranges`); `finalize` unchanged signature but slices per recorded range.

- [ ] **Step 1: Write the failing tests**

```python
# add to tests/backtesting/engine/test_fill_sink.py
def test_set_oos_range_slices_only_the_concat(tmp_path):
    sink = FillSink("FxDemo", "rid", {}, root=tmp_path)
    df = pd.DataFrame({"date": ["2018-06-01", "2021-03-01", "2021-09-01"],
                       "pair": ["EURUSD", "EURUSD", "EURUSD"], "units": [1.0, 2.0, 3.0]})
    sink.write_window(df, window=1, cfg_hash="c1x")
    sink.set_oos_range(1, "2021-01-01", "2022-01-01")
    sink.finalize(oos_windows=[1], oos_cfg_hash="c1x")
    # per-window file on disk stays full (3 rows)
    full = pd.read_csv(sink.run_dir / "w01_c1x_trades.csv.gz")
    assert len(full) == 3
    # OOS concat sliced to [2021-01-01, 2022-01-01) -> only the 2 2021 rows
    oos = pd.read_csv(sink.run_dir / "trades_oos.csv.gz")
    assert list(oos["units"]) == [2.0, 3.0]
    assert (pd.to_datetime(oos["date"]) >= pd.Timestamp("2021-01-01")).all()


def test_adjacent_windows_boundary_counted_once_and_last_end_inclusive(tmp_path):
    sink = FillSink("FxDemo", "rid", {}, root=tmp_path)
    # window 1 OOS [2021-01-01, 2022-01-01): the 2022-01-01 row must NOT come from w1
    w1 = pd.DataFrame({"date": ["2021-06-01", "2022-01-01"], "pair": ["EURUSD", "EURUSD"], "units": [1.0, 99.0]})
    # window 2 OOS [2022-01-01, 2023-01-01] (global-max end -> inclusive): owns 2022-01-01 and 2023-01-01
    w2 = pd.DataFrame({"date": ["2022-01-01", "2023-01-01"], "pair": ["EURUSD", "EURUSD"], "units": [2.0, 3.0]})
    sink.write_window(w1, window=1, cfg_hash="c1x")
    sink.write_window(w2, window=2, cfg_hash="c1x")
    sink.set_oos_range(1, "2021-01-01", "2022-01-01")
    sink.set_oos_range(2, "2022-01-01", "2023-01-01")
    sink.finalize(oos_windows=[1, 2], oos_cfg_hash="c1x")
    oos = pd.read_csv(sink.run_dir / "trades_oos.csv.gz")
    # 2022-01-01 appears exactly once (from window 2, half-open excludes it from w1),
    # 2023-01-01 included (global-max end inclusive); the w1 99.0 row is excluded.
    assert sorted(oos["units"]) == [1.0, 2.0, 3.0]
    dup = oos.duplicated(subset=["date", "pair"]).sum()
    assert dup == 0


def test_no_range_recorded_concats_full(tmp_path):
    sink = FillSink("FxDemo", "rid", {}, root=tmp_path)
    df = pd.DataFrame({"date": ["2018-06-01", "2021-03-01"], "pair": ["EURUSD", "EURUSD"], "units": [1.0, 2.0]})
    sink.write_window(df, window=1, cfg_hash="c1x")
    sink.finalize(oos_windows=[1], oos_cfg_hash="c1x")  # no set_oos_range
    oos = pd.read_csv(sink.run_dir / "trades_oos.csv.gz")
    assert len(oos) == 2  # unchanged, full concat


def test_range_but_no_date_column_does_not_crash(tmp_path):
    sink = FillSink("FxDemo", "rid", {}, root=tmp_path)
    df = pd.DataFrame({"pair": ["EURUSD", "EURUSD"], "units": [1.0, 2.0]})  # no "date"
    sink.write_window(df, window=1, cfg_hash="c1x")
    sink.set_oos_range(1, "2021-01-01", "2022-01-01")
    sink.finalize(oos_windows=[1], oos_cfg_hash="c1x")
    oos = pd.read_csv(sink.run_dir / "trades_oos.csv.gz")
    assert len(oos) == 2  # no date column -> no slice
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=$(pwd) pytest tests/backtesting/engine/test_fill_sink.py -k "oos_range or boundary or no_range or no_date_column" -v`
Expected: FAIL (`AttributeError: 'FillSink' object has no attribute 'set_oos_range'`).

- [ ] **Step 3: Implement**

In `__init__`, after `self._manifest_path = ...`, add:

```python
        self._oos_ranges: dict[int, tuple] = {}
```

Add the method (place it near `_record`/`_stem`):

```python
    def set_oos_range(self, window, start, end):
        self._oos_ranges[window] = (pd.Timestamp(start), pd.Timestamp(end))
```

In `finalize`, replace the OOS-concat frame-building loop so each window file is sliced to its recorded range. The current block reads each `wpath` and appends `pd.read_csv(wpath)`; change it to:

```python
    def finalize(self, oos_windows=None, oos_cfg_hash=None):
        if oos_windows:
            suffix = f"_{oos_cfg_hash}" if oos_cfg_hash else ""
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
                    if global_max_end is not None and hi == global_max_end:
                        df = df[(d >= lo) & (d <= hi)]
                    else:
                        df = df[(d >= lo) & (d < hi)]
                frames.append(df)
            if frames:
                oos = pd.concat(frames, ignore_index=True)
                oos.to_csv(self.run_dir / "trades_oos.csv.gz", index=False, compression="gzip")
                self._record({"file": "trades_oos.csv.gz", "kind": "oos_concat",
                              "window": -1, "cfg_hash": "", "row_count": len(oos)})
        # (manifest-building block below stays exactly as-is: read jsonl, dedup by file, write manifest.csv)
```

Leave the manifest-from-jsonl block (reading `manifest_rows.jsonl`, dedup-by-file, writing `manifest.csv`, the `logger.info`, and `return manifest_path`) unchanged.

- [ ] **Step 4: Run tests to verify they pass, then the whole FillSink suite**

Run: `PYTHONPATH=$(pwd) pytest tests/backtesting/engine/test_fill_sink.py -v`
Expected: PASS -- the 4 new tests plus all pre-existing FillSink tests (the pre-existing `finalize`/`oos_cfg_hash` tests must remain green, since they set no range).

- [ ] **Step 5: Commit**

```bash
git add src/backtesting/engine/fill_sink.py tests/backtesting/engine/test_fill_sink.py
git commit -m "feat(backtest): FillSink slices trades_oos to per-window OOS range"
```

---

### Task 2: Record OOS ranges in both FX walk-forward runners

**Files:**
- Modify: `scripts/backtest_scripts/run_fx_carry_seatbelt_walkforward.py` (in `run`, `specs` built ~lines 91-98, `sink.finalize` ~line 102)
- Modify: `scripts/backtest_scripts/run_fx_walkforward.py` (in `walk_forward_fx`, `specs` built ~lines 160-166, `sink.finalize` ~line 178)

**Interfaces:**
- Consumes: `FillSink.set_oos_range(window, start, end)` from Task 1. In both runners each `specs` element already has keys `"window"`, `"test_start"`, `"test_end"`.

- [ ] **Step 1: Read both runners to confirm the `specs` variable and the line just before `sink.finalize(...)`**

Read `scripts/backtest_scripts/run_fx_carry_seatbelt_walkforward.py` and `scripts/backtest_scripts/run_fx_walkforward.py`. Confirm `specs` is a list of dicts with `"window"`, `"test_start"`, `"test_end"`, and that `sink.finalize(oos_windows=..., oos_cfg_hash=_leg_tag(1.0))` is called after `parallel_map`. If a runner's key names differ, use the real names and note it in the report.

- [ ] **Step 2: Add the range-recording loop before `finalize` in EACH runner**

In `run_fx_carry_seatbelt_walkforward.py`, immediately before `sink.finalize(...)` (~line 102):

```python
    for s in specs:
        sink.set_oos_range(s["window"], s["test_start"], s["test_end"])
```

In `run_fx_walkforward.py::walk_forward_fx`, immediately before `sink.finalize(...)` (~line 178):

```python
    for s in specs:
        sink.set_oos_range(s["window"], s["test_start"], s["test_end"])
```

- [ ] **Step 3: Syntax-check both runners (do NOT execute them)**

Run: `python -m py_compile scripts/backtest_scripts/run_fx_carry_seatbelt_walkforward.py scripts/backtest_scripts/run_fx_walkforward.py`
Expected: exit 0, no output. (Running the runners is hook-blocked and verdict-adjacent; py_compile is the only verification here.)

- [ ] **Step 4: Commit**

```bash
git add -f scripts/backtest_scripts/run_fx_carry_seatbelt_walkforward.py scripts/backtest_scripts/run_fx_walkforward.py
git commit -m "feat(fx): WF runners record per-window OOS range for trades_oos slicing"
```

---

### Task 3: Docs clarification

**Files:**
- Modify: `.claude/rules/strategy-pipeline.md`, `docs/methodology/backtesting.md` (Section 12), `.claude/agents/strategy-lead.md`

**Interfaces:** none (documentation).

- [ ] **Step 1: Read the three trade-logging passages**

Read the every-run-mandate paragraph in `.claude/rules/strategy-pipeline.md` (added at the prior feature's Task 14), the Section 12 addition in `docs/methodology/backtesting.md`, and the fills-verification row/paragraph in `.claude/agents/strategy-lead.md`. Each currently describes `trades_oos.csv.gz` as the walk-forward OOS concatenation.

- [ ] **Step 2: Add the per-window vs OOS-concat clarification (one sentence in each)**

In each of the three files, add (or fold into the existing sentence) wording equivalent to:

```
For walk-forward runs, the per-window files (wNN_<leg>_trades.csv.gz) are the
FULL-window runs (train + test), and trades_oos.csv.gz is those windows sliced
to their out-of-sample [test_start, test_end) segments and concatenated -- so
trades_oos is the fills matching the gated OOS return series.
```

Match each file's existing style and phrasing; do not reformat unrelated content. ASCII only, no em dashes.

- [ ] **Step 3: Verify ASCII-clean**

Run: `grep -nP "[^\x00-\x7F]" .claude/rules/strategy-pipeline.md docs/methodology/backtesting.md .claude/agents/strategy-lead.md && echo FOUND || echo clean`
Expected: `clean` (any pre-existing non-ASCII outside your edited lines is out of scope; do not touch it -- if the grep reports pre-existing hits, confirm none are on lines you added).

- [ ] **Step 4: Commit**

```bash
git add .claude/rules/strategy-pipeline.md docs/methodology/backtesting.md .claude/agents/strategy-lead.md
git commit -m "docs: clarify per-window (full) vs trades_oos (OOS-sliced) fill artifacts"
```

---

## Final Validation (whole-branch)

- [ ] Run the FillSink suite: `PYTHONPATH=$(pwd) pytest tests/backtesting/engine/test_fill_sink.py -v` -- all pass (new slicing tests + all pre-existing).
- [ ] Whole-branch review (most capable model): confirm slicing is applied ONLY to `trades_oos` (per-window files unchanged), back-compat holds (no-range -> full concat), boundary rule is non-overlapping with the last day retained, and both runners record ranges for every window.
- [ ] POST-MERGE (delegated to strategy-lead, NOT this plan): re-run the FxCarrySeatbelt demo and the validation script; expect `trades_oos.csv.gz` to start at the first `test_start` (~2021), zero rows before it, zero duplicate `(date,pair)` rows -- versus the pre-fix 864 in-sample rows + 502 duplicates.
