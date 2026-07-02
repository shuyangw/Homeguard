# Broad-Basket Carver TSMOM Walk-Forward Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run Carver TSMOM through the walk-forward statistical gate on a ~33-root, full-size, $10M diversified futures basket (2010-2026), replacing the 3-instrument WEAK baseline with a trustworthy gate-checked result.

**Architecture:** Two small deliverables plus one execution run. (1) A new futures config YAML for the broad basket. (2) Make `run_carver_walkforward.py` config-driven (read universe/capital/vol-target/dates from a `--config` YAML; write to a `--report` path), and fix report prose that hardcodes capital and "12-instrument" micro assumptions so the report is accurate for any basket. (3) Controller runs the multi-hour walk-forward and records the verdict. No harness-core changes.

**Tech Stack:** Python 3.13, pandas, numpy, pyyaml, pytest. Conda env `fintech`.

## Global Constraints

- **Python execution:** ALWAYS `/c/Users/qwqw1/anaconda3/envs/fintech/python.exe -m pytest <args>`. Scripts that import `scripts/` need `PYTHONPATH=.`. Never system Python.
- **ASCII only** in all code/docs (Windows cp1252). No `print()` (use `src.utils.logger`). Homeguard logger uses f-strings, not `%s`.
- **Base branch:** `feat/carver-broad-basket` (already checked out, off `main` @ a855ae2). Do NOT switch branches.
- **Do NOT touch harness core:** `run_futures_backtest`, the simulator, sizing, loader, or `carver_*` strategy code. This effort only adds a config and edits the ONE walk-forward script.
- **Preserve the baseline:** never overwrite `docs/reports/futures/CARVER_TSMOM_READINESS.md` (the 3-instrument baseline). The no-argument default behavior of `run_carver_walkforward.py` must stay byte-for-byte equivalent to today.
- **Universe (33 roots), verbatim:** ES, NQ, YM, ZT, ZF, ZN, TN, ZB, UB, 6E, 6J, 6B, 6A, 6C, 6S, 6M, 6N, CL, BZ, NG, HO, RB, GC, SI, HG, PL, ZC, ZW, ZS, ZL, ZM, LE, HE.
- **Params:** initial_capital 10_000_000; vol_target_per_instrument 0.20; rebalance weekly; cost_mult 1.0; dates 2010-06-07 .. 2026-02-20; walk-forward train 36m / test 12m / step 12m (script literals, not in the YAML).

---

## Task 1: Broad-basket config YAML

**Files:**
- Create: `config/backtesting/carver_tsmom_broad.yaml`
- Test: `tests/backtesting/config/test_carver_broad_config.py`

**Interfaces:**
- Produces: a futures-asset-class config consumed by `src.backtest_runner` (single pass) and by `run_carver_walkforward.py --config` (Task 2). Keys mirror `config/backtesting/carver_tsmom.yaml`.

- [ ] **Step 1: Write the failing test**

```python
# tests/backtesting/config/test_carver_broad_config.py
from pathlib import Path
import yaml
from src.data.futures.contract_specs import SPECS

CONFIG = Path("config/backtesting/carver_tsmom_broad.yaml")
EXPECTED = {
    "ES", "NQ", "YM", "ZT", "ZF", "ZN", "TN", "ZB", "UB",
    "6E", "6J", "6B", "6A", "6C", "6S", "6M", "6N",
    "CL", "BZ", "NG", "HO", "RB", "GC", "SI", "HG", "PL",
    "ZC", "ZW", "ZS", "ZL", "ZM", "LE", "HE",
}

def test_broad_config_shape_and_roots():
    cfg = yaml.safe_load(CONFIG.read_text())
    assert cfg["asset_class"] == "futures"
    universe = set(cfg["strategy"]["universe"])
    assert len(cfg["strategy"]["universe"]) == 33
    assert universe == EXPECTED
    assert universe <= set(SPECS.keys())  # no typos; every root is speced
    assert cfg["backtest"]["initial_capital"] == 10_000_000
    assert cfg["backtest"]["vol_target_per_instrument"] == 0.20
    assert cfg["backtest"]["rebalance"] == "weekly"
    assert cfg["backtest"]["cost_mult"] == 1.0
    assert cfg["dates"]["start"] == "2010-06-07"
    assert cfg["dates"]["end"] == "2026-02-20"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/c/Users/qwqw1/anaconda3/envs/fintech/python.exe -m pytest tests/backtesting/config/test_carver_broad_config.py -v`
Expected: FAIL (config file does not exist -> yaml read raises FileNotFoundError).

- [ ] **Step 3: Create the config**

```yaml
# config/backtesting/carver_tsmom_broad.yaml
# Broad-basket Carver multi-speed TSMOM (parameter-free) over ~33 full-size roots.
# Usage (single pass): python -m src.backtest_runner --config config/backtesting/carver_tsmom_broad.yaml
# Usage (walk-forward): python scripts/backtest_scripts/run_carver_walkforward.py \
#     --config config/backtesting/carver_tsmom_broad.yaml \
#     --report docs/reports/futures/CARVER_TSMOM_BROAD_READINESS.md
#
# asset_class: futures is detected on the raw YAML dict in src/backtest_runner.py
# and routed to src.backtesting.engine.futures_backtest.run_futures_backtest.

asset_class: futures

strategy:
  universe:
    - ES
    - NQ
    - YM
    - ZT
    - ZF
    - ZN
    - TN
    - ZB
    - UB
    - 6E
    - 6J
    - 6B
    - 6A
    - 6C
    - 6S
    - 6M
    - 6N
    - CL
    - BZ
    - NG
    - HO
    - RB
    - GC
    - SI
    - HG
    - PL
    - ZC
    - ZW
    - ZS
    - ZL
    - ZM
    - LE
    - HE

dates:
  start: "2010-06-07"
  end: "2026-02-20"

backtest:
  initial_capital: 10000000
  vol_target_per_instrument: 0.20
  rebalance: weekly
  cost_mult: 1.0
```

Note: the FX/currency roots (`6E` etc.) MUST be quoted or bare-listed such that YAML reads them as strings, not numbers. As list items with a leading digit they parse as strings already, but if you inline them (`[6E, 6J]`) confirm they stay strings. The block-list form above is safe.

- [ ] **Step 4: Run test to verify it passes**

Run: `/c/Users/qwqw1/anaconda3/envs/fintech/python.exe -m pytest tests/backtesting/config/test_carver_broad_config.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add config/backtesting/carver_tsmom_broad.yaml tests/backtesting/config/test_carver_broad_config.py
git commit -m "feat(futures): broad-basket Carver config (33 full-size roots, 10M, 2010-2026)"
```

---

## Task 2: Config-driven walk-forward + basket-accurate report

**Files:**
- Modify: `scripts/backtest_scripts/run_carver_walkforward.py`
- Test: `tests/backtesting/test_carver_walkforward_config.py`

**Interfaces:**
- Consumes: `config/backtesting/carver_tsmom_broad.yaml` (Task 1).
- Produces: new pure helper `_config_to_kwargs(config: dict) -> dict` returning `{"universe": list[str], "capital": float, "vol_target": float, "start": str, "end": str}`; a `main()` that accepts `--config <yaml>` and `--report <path>`; `walk_forward_carver`'s result dict gains `capital` and `vol_target` keys; `_write_readiness_report` gains a `report_path` param and interpolates actual capital/vol-target/instrument-count.

**Context — current state (verified):**
- `walk_forward_carver(train_months, test_months, step_months, start, end, universe=None, capital=_DEFAULT_CAPITAL, vol_target=_DEFAULT_VOL_TARGET)` is already parametrized; it internally runs both 1x and 1.5x cost and hardcodes weekly rebalance in `_run_window`. Good - Task 2 only wires config -> these existing params.
- The result dict (lines 238-254) lacks `capital`/`vol_target`.
- `_write_readiness_report(result, train_months, test_months, step_months, start, end)` writes to module const `_REPORT_PATH` and interpolates `_DEFAULT_CAPITAL` (line ~349), `_DEFAULT_VOL_TARGET` (line ~350), and a hardcoded "fixed 12-instrument universe" micro sentence (lines ~353-355) - all wrong for a 33-root full-size basket.
- `main()` (lines 436-453) hardcodes `_DEFAULT_UNIVERSE`, 36/12/12, and dates 2010-06-07..2025-02-01.

- [ ] **Step 1: Write the failing tests**

```python
# tests/backtesting/test_carver_walkforward_config.py
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "wf", "scripts/backtest_scripts/run_carver_walkforward.py")
wf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(wf)


def test_config_to_kwargs_extracts_params():
    cfg = {
        "asset_class": "futures",
        "strategy": {"universe": ["ES", "GC", "6E"]},
        "dates": {"start": "2010-06-07", "end": "2026-02-20"},
        "backtest": {"initial_capital": 10_000_000,
                     "vol_target_per_instrument": 0.20,
                     "rebalance": "weekly", "cost_mult": 1.0},
    }
    kw = wf._config_to_kwargs(cfg)
    assert kw["universe"] == ["ES", "GC", "6E"]
    assert kw["capital"] == 10_000_000
    assert kw["vol_target"] == 0.20
    assert kw["start"] == "2010-06-07"
    assert kw["end"] == "2026-02-20"


def test_report_interpolates_actual_capital_and_count(tmp_path):
    # Minimal fake result covering everything _write_readiness_report reads.
    result = {
        "oos_sharpe": 0.3, "psr": 1.0, "dsr": 1.0, "pbo": 0.25,
        "oos_sharpe_1_5x_cost": 0.2, "n_windows": 2, "n_oos_days": 500,
        "window_sharpes": [0.3, 0.4], "trial_count": 1,
        "skew": -0.2, "kurtosis_pearson": 5.0,
        "universe": ["ES", "GC", "6E"], "window_universes": [["ES", "GC"], ["ES", "GC", "6E"]],
        "window_start": __import__("datetime").date(2013, 6, 7),
        "window_end": __import__("datetime").date(2026, 2, 20),
        "capital": 10_000_000, "vol_target": 0.20,
    }
    out = tmp_path / "BROAD.md"
    wf._write_readiness_report(result, train_months=36, test_months=12,
                               step_months=12, start="2010-06-07", end="2026-02-20",
                               report_path=str(out))
    text = out.read_text()
    assert "$10,000,000" in text          # actual capital, not the default
    assert "12-instrument" not in text    # stale micro prose removed
    assert "0.20" in text                 # actual vol target
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/c/Users/qwqw1/anaconda3/envs/fintech/python.exe -m pytest tests/backtesting/test_carver_walkforward_config.py -v`
Expected: FAIL — `_config_to_kwargs` does not exist (AttributeError); `_write_readiness_report` has no `report_path` kwarg (TypeError).

- [ ] **Step 3: Add `_config_to_kwargs` helper**

Add near the top of the module (after the existing `_as_date` helper):

```python
def _config_to_kwargs(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract walk_forward_carver kwargs from a futures backtest YAML dict."""
    strat = config.get("strategy", {})
    dates = config.get("dates", {})
    bt = config.get("backtest", {})
    return {
        "universe": list(strat["universe"]),
        "capital": float(bt.get("initial_capital", _DEFAULT_CAPITAL)),
        "vol_target": float(bt.get("vol_target_per_instrument", _DEFAULT_VOL_TARGET)),
        "start": str(dates["start"]),
        "end": str(dates["end"]),
    }
```

- [ ] **Step 4: Thread `capital`/`vol_target` into the result dict**

In `walk_forward_carver`, add two keys to the `result` dict (lines 238-254), after `"universe": universe,`:

```python
        "universe": universe,
        "capital": capital,
        "vol_target": vol_target,
        "window_universes": window_universes,
```

- [ ] **Step 5: Make `_write_readiness_report` basket-accurate and path-parametric**

Change the signature:

```python
def _write_readiness_report(result: Dict[str, Any], train_months: int, test_months: int,
                             step_months: int, start: str, end: str,
                             report_path: str = _REPORT_PATH) -> str:
```

In the report body, replace the capital/vol-target line (currently uses `_DEFAULT_CAPITAL` / `_DEFAULT_VOL_TARGET`) with the actual run values and the instrument count:

```python
Requested universe ({len(result['universe'])} roots): {result['universe']}.
Initial capital: ${result['capital']:,.0f}. Vol target per instrument:
{result['vol_target']:.2f}. Rebalance: weekly.
```

Replace the stale micro/"fixed 12-instrument universe" sentence (lines ~353-355) with a generic phase-in sentence:

```python
**Per-window data-availability filtering.** Instruments phase in over time
(micro contracts launched 2019+, SOFR 2018, some roots later), so a fixed
universe cannot have full history back to {start} for every root.
`load_daily_panel` (`src/backtesting/data/futures_backtest_loader.py`)
gracefully excludes any root with no usable data for a window's
[train_start, test_end] range (logged as a WARNING, never silently).
```

At the end of the function, build `out_path` from the new parameter instead of the module constant:

```python
    out_path = Path(report_path)
```
(Keep the rest of the write/return unchanged.)

- [ ] **Step 6: Rewrite `main()` to be config-driven (default behavior preserved)**

```python
def main() -> None:
    import argparse
    import yaml

    parser = argparse.ArgumentParser(description="Carver TSMOM walk-forward + gate")
    parser.add_argument("--config", default=None,
                        help="Futures backtest YAML; drives universe/capital/vol-target/dates")
    parser.add_argument("--report", default=_REPORT_PATH,
                        help="Output readiness-report path (defaults to the baseline path)")
    args = parser.parse_args()

    if args.config is not None:
        cfg = yaml.safe_load(Path(args.config).read_text())
        kw = _config_to_kwargs(cfg)
    else:
        kw = {"universe": list(_DEFAULT_UNIVERSE), "capital": _DEFAULT_CAPITAL,
              "vol_target": _DEFAULT_VOL_TARGET, "start": "2010-06-07", "end": "2025-02-01"}

    result = walk_forward_carver(
        train_months=36, test_months=12, step_months=12,
        start=kw["start"], end=kw["end"],
        universe=kw["universe"], capital=kw["capital"], vol_target=kw["vol_target"],
    )
    report_path = _write_readiness_report(
        result, train_months=36, test_months=12, step_months=12,
        start=kw["start"], end=kw["end"], report_path=args.report,
    )
    logger.info(
        f"[walk_forward_carver] wrote {report_path}; "
        f"oos_sharpe={result['oos_sharpe']:.4f} psr={result['psr']:.4f} "
        f"dsr={result['dsr']:.4f} pbo={result['pbo']} "
        f"oos_sharpe_1_5x_cost={result['oos_sharpe_1_5x_cost']:.4f} "
        f"n_windows={result['n_windows']}"
    )
```

Confirm `from pathlib import Path` is already imported at module top (it is used for `out_path`); if not, add it.

- [ ] **Step 7: Run the unit tests to verify they pass**

Run: `/c/Users/qwqw1/anaconda3/envs/fintech/python.exe -m pytest tests/backtesting/test_carver_walkforward_config.py -v`
Expected: PASS (both tests).

- [ ] **Step 8: Tiny end-to-end plumbing smoke (real data, 2 roots, fast)**

Run:
```bash
PYTHONPATH=. /c/Users/qwqw1/anaconda3/envs/fintech/python.exe -c "
import importlib.util
s=importlib.util.spec_from_file_location('wf','scripts/backtest_scripts/run_carver_walkforward.py')
wf=importlib.util.module_from_spec(s); s.loader.exec_module(wf)
r=wf.walk_forward_carver(train_months=12,test_months=6,step_months=6,
    start='2021-01-01',end='2022-12-31',universe=['6E','GC'],capital=1_000_000)
import math
assert r['n_windows']>=1, r['n_windows']
assert math.isfinite(r['oos_sharpe']), r['oos_sharpe']
assert 'capital' in r and r['capital']==1_000_000
print('SMOKE OK n_windows=',r['n_windows'],'oos_sharpe=%.4f'%r['oos_sharpe'])
"
```
Expected: prints `SMOKE OK n_windows=...` with a finite Sharpe (a couple minutes; 2 roots over 2 years). This proves config-driven kwargs flow end-to-end without launching the multi-hour full run.

- [ ] **Step 9: Confirm no-arg default behavior is unchanged (report path)**

Run:
```bash
/c/Users/qwqw1/anaconda3/envs/fintech/python.exe -c "
import importlib.util
s=importlib.util.spec_from_file_location('wf','scripts/backtest_scripts/run_carver_walkforward.py')
wf=importlib.util.module_from_spec(s); s.loader.exec_module(wf)
print('default report path:', wf._REPORT_PATH)
assert wf._REPORT_PATH == 'docs/reports/futures/CARVER_TSMOM_READINESS.md'
"
```
Expected: prints the baseline path; confirms the default target is still the baseline file (so a no-arg invocation would not touch a broad-report file).

- [ ] **Step 10: Commit**

```bash
git add scripts/backtest_scripts/run_carver_walkforward.py tests/backtesting/test_carver_walkforward_config.py
git commit -m "feat(futures): config-driven walk-forward universe + basket-accurate report"
```

---

## Task 3: Execution and Acceptance (CONTROLLER-run, not a TDD/subagent task)

This task is NOT test-first and NOT delegated to an implementer subagent. The controller runs the multi-hour experiment in the background and records the verdict. Tasks 1-2 must be complete and committed first.

**Files:**
- Produces: `docs/reports/futures/CARVER_TSMOM_BROAD_READINESS.md` (new, written by the run)
- Verifies: `docs/reports/futures/CARVER_TSMOM_READINESS.md` (baseline) is UNCHANGED.

- [ ] **Step 1: Launch the full broad walk-forward in the background**

```bash
cd "C:/Users/qwqw1/Dropbox/cs/github/Homeguard"
PYTHONPATH=. /c/Users/qwqw1/anaconda3/envs/fintech/python.exe \
  scripts/backtest_scripts/run_carver_walkforward.py \
  --config config/backtesting/carver_tsmom_broad.yaml \
  --report docs/reports/futures/CARVER_TSMOM_BROAD_READINESS.md \
  > .superpowers/sdd/broad_walkforward.log 2>&1
```
Run with `run_in_background: true`. Expect multi-hour runtime (33 roots x ~15.7y x ~14 windows). Do NOT poll in a sleep loop; wait for the completion notification.

- [ ] **Step 2: On completion, verify the run and the report**

```bash
tail -20 .superpowers/sdd/broad_walkforward.log
grep -iE "oos_sharpe|psr|dsr|pbo|skew|kurt|verdict" docs/reports/futures/CARVER_TSMOM_BROAD_READINESS.md | head
git status --porcelain docs/reports/futures/CARVER_TSMOM_READINESS.md   # MUST be empty (baseline untouched)
```
Acceptance checks:
- The log ends with the `wrote ...; oos_sharpe=... psr=... pbo=... n_windows=...` line (non-error exit).
- `CARVER_TSMOM_BROAD_READINESS.md` exists, has the metrics table + per-window table, interpolates `$10,000,000` and the actual instrument count, and has sane tail stats (skew O(1), kurtosis single/low-double digit; 1.5x-cost Sharpe below the 1x figure).
- The baseline report shows no diff.

- [ ] **Step 3: Record the verdict**

Read the verdict line in the broad report. Summarize the gate outcome (OOS Sharpe, PSR/DSR/PBO, 1.5x-cost Sharpe, n_windows, verdict) for the user. If it clears the combined gate -> Carver viable on a real basket (candidate for paper deployment); if still WEAK -> naive Carver fairly exhausted, proceed to B/C. Either is a valid, reportable result.

- [ ] **Step 4: Commit the broad report**

```bash
git add docs/reports/futures/CARVER_TSMOM_BROAD_READINESS.md
git commit -m "docs(futures): broad-basket (33-root, 10M) Carver walk-forward results"
```

---

## Self-Review

- **Spec coverage:** Task 1 = broad config (spec component 1). Task 2 = config-driven walk-forward `--config`/`--report` + basket-accurate report (spec components 2 and 3, incl. sibling-report path and baseline preservation). Task 3 = the actual run + verdict (spec success criteria). All covered.
- **Placeholder scan:** none. Config is fully enumerated (33 roots); the helper, report edits, and `main()` are shown in full; every test has real assertions.
- **Type consistency:** `_config_to_kwargs` returns `{universe: list[str], capital: float, vol_target: float, start: str, end: str}`, matching the keys `main()` reads and the params `walk_forward_carver` accepts. `_write_readiness_report` gains `report_path: str`; result dict gains `capital: float`, `vol_target: float`, both read back in the report body.
- **Baseline safety:** `--report` defaults to `_REPORT_PATH` (baseline); the broad run passes an explicit sibling path; Task 2 Step 9 asserts the default target is unchanged; Task 3 Step 2 asserts the baseline file has no diff.
- **Isolation:** no harness-core files touched; only a new config, a new-and-existing test file, and the one walk-forward script. Keeps A independent of B (pluggable runner) and C (carry).
