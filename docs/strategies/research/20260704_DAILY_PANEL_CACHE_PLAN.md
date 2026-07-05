# Daily-Panel Cache (OOM fix / A1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Steps use `- [ ]` checkboxes.

**Goal:** Eliminate the per-window 5.6 GB memory spike (and redundant work) by caching daily
RAW OHLCV per root once, then ratio-adjusting on the small daily series instead of re-loading and
ratio-adjusting the full 1-min history every walk-forward window/worker. RESULT-IDENTICAL.

**Architecture:** `aggregate_to_daily(root, "ratio_adjusted", start, end)` currently calls
`load()` which does `pl.concat([read_parquet(f) for f in ALL 1-min files])` (the 5.6 GB), then
ratio-adjusts at 1-min, then aggregates to daily. New: cache daily RAW OHLCV per root; a new
`ratio_adjust_daily()` applies the per-date roll factor to the DAILY close/OHLC. This is
byte-identical to the current path because the ratio factor is uniform within a date, so
`last(raw_1min * factor) == last(raw_1min) * factor == raw_daily_close * factor`. Per-window
anchoring is preserved (roll dates detected over the same sliced date range).

**Tech Stack:** polars, existing ContinuousContractDataLoader.

## Global Constraints
- Python tests: `/c/Users/qwqw1/anaconda3/envs/fintech/python.exe -m pytest <files> -q`. `PYTHONPATH=.` for scripts.
- ASCII only, no `print()`. Branch `feat/futures-daily-panel-cache` -- do NOT switch.
- **RESULT-PRESERVING is the hard gate:** cached-path ratio-adjusted daily close MUST equal the
  current on-the-fly path to the float, and carry_idm walk-forward MUST reproduce OOS Sharpe 0.76
  byte-identical. Any divergence = REJECT.
- Cache is OPT-IN with graceful fallback: if a root's daily-raw cache file is absent, fall back to
  the current on-the-fly aggregation (no behavior change when cache not built).
- Storage via `get_local_storage_dir()` -- never hardcode paths.

## Files
- Modify: `src/data/futures/paths.py` (add `daily_raw_dir()`)
- Modify: `src/data/continuous_contract_loader.py` (add `ratio_adjust_daily`; wire cache into `aggregate_to_daily`)
- Create: `scripts/data/build_daily_raw_cache.py`
- Tests: `tests/data/test_daily_raw_cache.py`, `tests/data/test_ratio_adjust_daily_equivalence.py`

---

## Task 1: daily_raw_dir path helper + cache builder

**Files:** Modify `src/data/futures/paths.py`; Create `scripts/data/build_daily_raw_cache.py`; Test `tests/data/test_daily_raw_cache.py`

**Interfaces:** `daily_raw_dir() -> Path` (e.g. `<storage>/futures/daily_raw`); `build_daily_raw_cache(roots, max_workers=None) -> list[str]`.

- [ ] Step 1: failing test -- `tests/data/test_daily_raw_cache.py::test_builder_writes_raw_daily` (skipif futures store absent, mirror existing skip guards): build cache for one real root (e.g. "ES"), assert `daily_raw_dir()/"ES.parquet"` exists and its rows equal `ContinuousContractDataLoader().aggregate_to_daily("ES", method="raw")` (same date+close).
- [ ] Step 2: run -> FAIL.
- [ ] Step 3: add `daily_raw_dir()` to `paths.py` mirroring `carry_dir()`. Write `build_daily_raw_cache.py` mirroring `scripts/data/build_carry_cache.py`: for each root, `df = ContinuousContractDataLoader().aggregate_to_daily(root, method="raw")`; if non-empty, `df.write_parquet(daily_raw_dir()/f"{root}.parquet")`. Use `parallel_map`, `--roots`, `--jobs`.
- [ ] Step 4: run -> PASS. Step 5: commit `feat(futures): daily-raw OHLCV cache builder + daily_raw_dir`.

---

## Task 2: ratio_adjust_daily + equivalence gate

**Files:** Modify `src/data/continuous_contract_loader.py`; Test `tests/data/test_ratio_adjust_daily_equivalence.py`

**Interfaces:** `ContinuousContractDataLoader.ratio_adjust_daily(daily_raw: pl.DataFrame, root: str, start=None, end=None) -> pl.DataFrame` -- takes daily RAW OHLCV (columns timestamp/open/high/low/close/volume), applies the same reverse-roll ratio factor used in `load()`'s `ratio_adjusted` branch, but on the daily series. Returns daily ratio-adjusted OHLCV.

- [ ] Step 1: failing EQUIVALENCE test -- `test_ratio_adjust_daily_equivalence.py` (skipif store absent): for a real root with rolls (e.g. "ES", "CL"), compare (a) current `aggregate_to_daily(root, "ratio_adjusted", start, end)` close vs (b) `ratio_adjust_daily(aggregate_to_daily(root,"raw",start,end), root, start, end)` close. Assert close series EQUAL to the float (`np.allclose(a, b, rtol=0, atol=1e-9)`) on aligned dates, for at least two windows (a full-range and a sub-window, to check anchoring).
- [ ] Step 2: run -> FAIL (method missing).
- [ ] Step 3: implement `ratio_adjust_daily` by factoring the existing `ratio_adjusted` branch logic (lines ~204-251) to operate on the daily df's per-date close (`close_map` = daily close), reusing `detect_roll_dates(root, data_start, data_end)` where data_start/end come from the daily df. Apply `factor` to open/high/low/close per date. (The panama path is out of scope -- only ratio_adjusted.)
- [ ] Step 4: run -> PASS (byte-identical). Step 5: commit `feat(futures): ratio_adjust_daily (result-identical daily-series roll adjustment)`.

---

## Task 3: wire cache into aggregate_to_daily (opt-in, graceful fallback)

**Files:** Modify `src/data/continuous_contract_loader.py`; Test `tests/data/test_daily_cache_wiring.py`

- [ ] Step 1: failing tests: (a) `test_uses_cache_when_present` -- monkeypatch `load()` to raise (proving it's NOT called), pre-place a small daily-raw parquet in `daily_raw_dir()` for a fake root, assert `aggregate_to_daily(root, "ratio_adjusted", ...)` returns the ratio-adjusted daily WITHOUT calling `load()`. (b) `test_falls_back_when_absent` -- for a root with NO cache file, `aggregate_to_daily` still works via `load()` (monkeypatch load to return a tiny known raw df; assert it was used).
- [ ] Step 2: run -> FAIL.
- [ ] Step 3: in `aggregate_to_daily`, for `method == "ratio_adjusted"`: if `daily_raw_dir()/f"{root}.parquet"` exists, read it, filter [start,end], and return `self.ratio_adjust_daily(raw_daily, root, start, end)`. Else fall through to the current `self.load(...)` path. `method in ("raw","panama_adjusted")` unchanged. (Only the ratio_adjusted daily path -- the one `load_daily_panel` uses -- is cache-accelerated.)
- [ ] Step 4: run -> PASS. Also run `tests/backtesting/data/test_load_daily_panel_errors.py` (no regression). Step 5: commit `feat(futures): aggregate_to_daily reads daily-raw cache when present (5.6GB->MB), fallback intact`.

---

## Task 4: build cache + acceptance gate (CONTROLLER-RUN)

Not TDD. After Tasks 1-3 merged-ready.
- [ ] Step 1: build the daily-raw cache for all 35 roots (33 macro + BTC/ETH), 8-thread capped:
  `... build_daily_raw_cache.py --roots ES NQ ... BTC ETH --jobs 8`.
- [ ] Step 2: re-run `carry_idm_broad` walk-forward (now cache-accelerated) and confirm OOS Sharpe
  **0.7646 / PBO 0.1887 byte-identical** to the pre-cache result (the hard acceptance gate).
- [ ] Step 3: re-measure per-window peak RSS (mem_probe) with cache present -- expect a few hundred MB,
  not 5.6 GB. Record the before/after. Step 4: re-run the 35-root carry+crypto combination at
  `--jobs 8` (previously OOM'd) and confirm it completes without BrokenProcessPool.

---

## Self-Review
- Coverage: cache builder (T1), result-identical daily ratio-adjust (T2), opt-in wiring + fallback
  (T3), build + equivalence gate + memory re-measure + OOM-resolution proof (T4). Matches design.
- Placeholders: none -- exact functions/paths/tests specified.
- Types: `daily_raw_dir()->Path`; `build_daily_raw_cache(list,int)->list[str]`;
  `ratio_adjust_daily(pl.DataFrame,str,date,date)->pl.DataFrame`; `aggregate_to_daily` signature unchanged.
- Result-preserving gate (T2 float-equivalence + T4 0.76 byte-identical) is the non-negotiable acceptance.
