# RAMP Research Consolidation -- Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rename the RAMP backtest harness to a function-descriptive module, make its variant registry the single source of truth (descriptive ids + aliases + two new decision-relevant variants), archive the dead investigation scripts, and prepare the daily turnover/cost experiment -- without changing a single numeric output in the refactor.

**Architecture:** Five sequential PRs. PRs 1-4 are code (mechanical rename, CLI surface, variant registry, archival) and are covered here as task-by-task steps. PR 5a is a data-dependent research-execution checklist that runs on the operator's machine (the SIP data lives on the H: drive, unreachable from an agent sandbox). PR 6 (weekly rebalance) is gated on PR 5a firing "Branch 2" and gets its own plan when triggered.

**Tech Stack:** Python 3.13, pytest (mock-based unit tests via monkeypatch), pandas, git. Conda env `fintech`. Spec: `docs/planning/20260530_ramp_research_consolidation_plan.md`.

---

## Preconditions (do once, before Task 1)

- [ ] **P.1: Confirm you are on `origin/main` (not a stale local main).**

Run:
```bash
cd /c/Users/qwqw1/Homeguard-main
git fetch origin main
git rev-list --count HEAD..origin/main   # expect 0
```
Expected: `0`. If non-zero, `git checkout main && git merge --ff-only origin/main` first. (The spec exists because the draft was authored against a tree 5 commits behind; do not repeat that.)

- [ ] **P.2: Confirm the load-bearing state is present.**

Run:
```bash
grep -n "variant" config/trading/strategy_toggle.yaml        # expect ramp: v11, others v01
test -f src/settings/data_paths.py && echo "data_paths OK"   # expect: data_paths OK
ls src/research/ramp_phase4/ tests/research/ramp_phase4/      # expect 8 + 8 files
```

- [ ] **P.3: Work on the existing feature branch.**

Run:
```bash
git checkout feature/ramp-research-consolidation || git checkout -b feature/ramp-research-consolidation origin/main
git branch --show-current   # expect feature/ramp-research-consolidation
```

- [ ] **P.4: Establish the env.** All `pytest`/`python` commands below run in the `fintech` conda env: `conda activate fintech` (or prefix with `conda run -n fintech`).

**Sandbox vs box:** every `pytest tests/research/...` command below is mock-based and runs anywhere. Every command that invokes `scripts/backtest_scripts/*_backtest.py` / `run_momentum_variant.py` against real dates reads H: data and must run on the operator's box -- those are marked **[BOX-ONLY]**.

---

## File Structure

**PR 1 -- rename (mechanical, zero numeric change):**
- Move: `src/research/ramp_phase4/` -> `src/research/regime_momentum_lab/` (8 files)
- Move: `tests/research/ramp_phase4/` -> `tests/research/regime_momentum_lab/` (8 files)
- Modify: `src/research/regime_momentum_lab/data.py` (cache path + SIP source path)
- Modify: `scripts/backtest_scripts/ramp_phase4_backtest.py` (import lines only)
- Modify: `scripts/backtest_scripts/_make_parity_report.py` (import lines only)

**PR 2 -- CLI surface:**
- Move: `scripts/backtest_scripts/ramp_phase4_backtest.py` -> `scripts/backtest_scripts/run_momentum_variant.py`
- Modify: that file (add `--rebalance-frequency` fail-loud guard + turnover line; both via small testable helpers)
- Create: `tests/research/regime_momentum_lab/test_cli_helpers.py`

**PR 3 -- variant registry:**
- Modify: `src/research/regime_momentum_lab/variants.py` (aliases + `resolve()` + rename funcs + `bear_to_cash` thread + `plain`/`bear_cash`)
- Modify: `tests/research/regime_momentum_lab/test_variants.py` (rewrite assertions, add alias/behavior tests)
- Modify: `scripts/backtest_scripts/_make_parity_report.py` (use `resolve()`)
- Modify: `scripts/backtest_scripts/run_momentum_variant.py` (use `resolve()`, drop hard `choices=`)

**PR 4 -- archive:**
- Move: 4 dated scripts -> `scripts/backtest_scripts/_archived/`
- Create: `scripts/backtest_scripts/_archived/README.md`
- (No edit to `strategy_toggle.yaml` or `strategy_state_manager.py` -- see spec Appendix B)

---

## PR 1: Rename module `ramp_phase4` -> `regime_momentum_lab`

> Mechanical move, zero numeric change. The "test" is the existing 41 mock-based tests staying green, plus a [BOX-ONLY] numeric-identity diff. No new behavior, so no new failing test is written here.

### Task 1.1: Capture the numeric-identity baseline **[BOX-ONLY]**

**Files:** none (produces `/tmp/v03_before.md`).

- [ ] **Step 1: On the operator box, run the OLD CLI before any rename.**

Run:
```bash
python scripts/backtest_scripts/ramp_phase4_backtest.py --variant V03 \
  --start 2025-01-01 --end 2026-04-30 --cost-bps 0,5 --output /tmp/v03_before.md
```
Expected: `wrote /tmp/v03_before.md`. Keep this file; Task 1.6 diffs against it.
If you are in a sandbox without H: data, **skip this task** and leave a `NUMERIC-IDENTITY DEFERRED` note in the PR description instructing the operator to run Task 1.1 (old tree) and Task 1.6 (new tree) manually.

### Task 1.2: Move the source directory and fix internal imports

**Files:**
- Move: `src/research/ramp_phase4/` -> `src/research/regime_momentum_lab/`
- Modify: internal sibling imports in the moved files

- [ ] **Step 1: Move the directory (history follows).**

Run:
```bash
git mv src/research/ramp_phase4 src/research/regime_momentum_lab
```

- [ ] **Step 2: Find internal references to the old path.**

Run:
```bash
grep -rln 'ramp_phase4' src/research/regime_momentum_lab/
```
Expected: hits in `engine.py`, `reports.py`, possibly `data.py` docstrings.

- [ ] **Step 3: Replace import paths (NOT logic).**

In each file grep surfaced, replace `from src.research.ramp_phase4.` with `from src.research.regime_momentum_lab.`. Do not touch function names, signatures, or bodies.

Run (sed-style, then re-grep to confirm):
```bash
grep -rl 'src.research.ramp_phase4' src/research/regime_momentum_lab/ \
  | xargs sed -i 's/src\.research\.ramp_phase4/src.research.regime_momentum_lab/g'
grep -rn 'ramp_phase4' src/research/regime_momentum_lab/*.py   # expect: no import/code hits
```

### Task 1.3: Decouple cache path AND reconcile SIP source path in `data.py`

**Files:**
- Modify: `src/research/regime_momentum_lab/data.py`

- [ ] **Step 1: Re-read the path constants before editing.**

Run:
```bash
grep -n "SIP_DAILY_CACHE_REL\|SIP_SPLIT_REL\|LEGACY_DAILY_CACHE_REL\|get_local_storage_dir\|equities_1min_sip_split" src/research/regime_momentum_lab/data.py
```
Confirm the two constants exist (locators ~44-46). Also confirm the canonical helper:
```bash
grep -n "get_equities_sip_split_1min_dir\|EQUITIES_SIP_SPLIT_1MIN" src/settings/data_paths.py
```

- [ ] **Step 2: Change the cache path to drop the module name.**

In `data.py`, change:
```python
SIP_DAILY_CACHE_REL = 'cache/ramp_phase4/equities_daily_from_sip.parquet'
```
to:
```python
SIP_DAILY_CACHE_REL = 'cache/regime_momentum/equities_daily_from_sip.parquet'
```
Update the docstring references (the `cache/ramp_phase4/...` mentions) to `cache/regime_momentum/...`.

- [ ] **Step 3: Reconcile the SIP SOURCE path to the canonical post-reorg location.**

The reorg moved the source tree's canonical location to `equities/sip_split/1min`; `data.py` still hardcodes the legacy `equities_1min_sip_split`. Add the import and resolve via the canonical helper. Near the top imports of `data.py`:
```python
from src.settings.data_paths import get_equities_sip_split_1min_dir
```
Then, where the source root is built (currently `root = storage / SIP_SPLIT_REL` in `_aggregate_to_daily_from_sip_split`), replace the `storage / SIP_SPLIT_REL` join with the canonical resolver and fail loud if absent:
```python
root = get_equities_sip_split_1min_dir()
if not root.exists():
    raise RuntimeError(
        f'SIP split source tree not found at canonical location {root}. '
        f'The data-reorg moved it from the legacy equities_1min_sip_split layout; '
        f'verify the H: migration ran (VERIFY SOURCE PATH).')
```
Keep `SIP_SPLIT_REL` defined for reference but stop using the bare-string join. (If `get_equities_sip_split_1min_dir()` is unavailable in your env, fall back to `get_data_dir('equities/sip_split/1min')`.)

- [ ] **Step 4: Run the unit tests (mock-based, no H: needed).**

Run:
```bash
pytest tests/research/ramp_phase4/ -q 2>&1 | tail -5
```
Expected: still collects from the OLD test dir (moved in Task 1.4) -- if you have not moved tests yet this command targets the old path. Defer the green check to Task 1.4 Step 3. (This step exists to confirm `data.py` still imports cleanly: `python -c "import src.research.regime_momentum_lab.data"` -> no ImportError.)

Run:
```bash
python -c "import src.research.regime_momentum_lab.data; print('import OK')"
```
Expected: `import OK`.

### Task 1.4: Move the test directory and fix monkeypatch strings

**Files:**
- Move: `tests/research/ramp_phase4/` -> `tests/research/regime_momentum_lab/`
- Modify: imports + monkeypatch string literals in the 8 test files

- [ ] **Step 1: Move the directory.**

Run:
```bash
git mv tests/research/ramp_phase4 tests/research/regime_momentum_lab
```

- [ ] **Step 2: Fix `from`-imports.**

Run:
```bash
grep -rl 'src.research.ramp_phase4' tests/research/regime_momentum_lab/ \
  | xargs sed -i 's/src\.research\.ramp_phase4/src.research.regime_momentum_lab/g'
```

- [ ] **Step 3 (CRITICAL): Confirm zero string-literal monkeypatch misses.**

Monkeypatch targets are strings, not imports -- a miss breaks the mock silently. The sed in Step 2 catches `src.research.ramp_phase4` in any quoted string too, but verify:
```bash
grep -rn 'ramp_phase4' tests/research/regime_momentum_lab/
```
Expected: **zero hits**. Known locations were `test_data.py` (`'...data._read_closes_from_parquet'`) and `test_engine.py` (`'...engine.load_universe_panel'`).

- [ ] **Step 4: Run the full moved suite.**

Run:
```bash
pytest tests/research/regime_momentum_lab/ -v 2>&1 | tail -15
```
Expected: **41 passed** (same count as before the move). `test_variants.py` still asserts `V01`/`V03` -- those change in PR 3, not here.

- [ ] **Step 5: Spot-check a mock actually binds (guards against a silent patch miss).**

Pick one patched test and confirm it fails when the real function is sabotaged. Temporarily edit `src/research/regime_momentum_lab/data.py` `_read_closes_from_parquet` to `raise RuntimeError('sabotage')`, then:
```bash
pytest tests/research/regime_momentum_lab/test_data.py -q 2>&1 | tail -5
```
Expected: tests that patch `_read_closes_from_parquet` still PASS (mock intercepts); any test that uses the real function FAILS with `sabotage`. Revert the sabotage edit immediately (`git checkout src/research/regime_momentum_lab/data.py` -- but only if you have NOT yet made the Task 1.3 edits committed; otherwise re-apply the raise removal by hand). If patched tests fail with `sabotage`, a monkeypatch string is wrong -- fix it.

### Task 1.5: Update the two importing scripts' import lines

**Files:**
- Modify: `scripts/backtest_scripts/ramp_phase4_backtest.py` (imports only; file rename is PR 2)
- Modify: `scripts/backtest_scripts/_make_parity_report.py` (imports only)

- [ ] **Step 1: Replace import paths in both scripts.**

Run:
```bash
sed -i 's/src\.research\.ramp_phase4/src.research.regime_momentum_lab/g' \
  scripts/backtest_scripts/ramp_phase4_backtest.py \
  scripts/backtest_scripts/_make_parity_report.py
grep -rn 'ramp_phase4' scripts/backtest_scripts/ramp_phase4_backtest.py scripts/backtest_scripts/_make_parity_report.py
```
Expected: no `src.research.ramp_phase4` hits (the filename `ramp_phase4_backtest.py` still contains the string in its own name -- that's fine, renamed in PR 2).

- [ ] **Step 2: Confirm both scripts import cleanly.**

Run:
```bash
python -c "import ast; ast.parse(open('scripts/backtest_scripts/ramp_phase4_backtest.py').read()); ast.parse(open('scripts/backtest_scripts/_make_parity_report.py').read()); print('parse OK')"
```
Expected: `parse OK`.

### Task 1.6: Numeric-identity check **[BOX-ONLY]** and commit

**Files:** none new.

- [ ] **Step 1: Re-run the same variant on the NEW tree and diff.** **[BOX-ONLY]**

Run:
```bash
python scripts/backtest_scripts/ramp_phase4_backtest.py --variant V03 \
  --start 2025-01-01 --end 2026-04-30 --cost-bps 0,5 --output /tmp/v03_after.md
diff /tmp/v03_before.md /tmp/v03_after.md
```
Expected: only the git-sha line differs. Anything else means the rename (almost certainly the Task 1.3 source-path change) altered the data -- STOP and confirm the canonical source location holds byte-identical data to the legacy one.

- [ ] **Step 2: Confirm the rename is complete.**

Run:
```bash
grep -rn 'ramp_phase4' src/ tests/ --include='*.py' | grep -vE '^Binary'
test -d src/research/ramp_phase4 && echo "OLD DIR STILL EXISTS (bad)" || echo "old dir gone (good)"
```
Expected: zero `.py` import/patch hits; `old dir gone (good)`.

- [ ] **Step 3: Commit.**

```bash
git add -A
git commit -m "refactor(research): rename ramp_phase4 -> regime_momentum_lab; reconcile cache+SIP source paths

Mechanical module rename (history preserved via git mv), all imports and
monkeypatch string targets updated, cache path decoupled from the module name
(cache/regime_momentum/), and the SIP source path reconciled to the post-reorg
canonical location (equities/sip_split/1min via get_equities_sip_split_1min_dir).
Zero numeric-output change (V03 report byte-identical modulo git-sha)."
```

---

## PR 2: Rename CLI + `--rebalance-frequency` (fail-loud) + turnover line

> TDD applies to the two new behaviors (the fail-loud guard and the turnover-line formatter), which are extracted into small pure helpers so they unit-test without H: data.

### Task 2.1: Rename the CLI file

**Files:**
- Move: `scripts/backtest_scripts/ramp_phase4_backtest.py` -> `scripts/backtest_scripts/run_momentum_variant.py`

- [ ] **Step 1: Move and update the docstring.**

Run:
```bash
git mv scripts/backtest_scripts/ramp_phase4_backtest.py scripts/backtest_scripts/run_momentum_variant.py
```
Then edit line 2 of `run_momentum_variant.py` from:
```python
"""CLI to run a Phase 4 variant against Alpaca SIP data and emit a Markdown report."""
```
to:
```python
"""Run a regime-momentum backtest variant against SIP data and emit a Markdown report."""
```

- [ ] **Step 2: Confirm it still parses.**

Run:
```bash
python -c "import ast; ast.parse(open('scripts/backtest_scripts/run_momentum_variant.py').read()); print('parse OK')"
```
Expected: `parse OK`.

### Task 2.2: Add the fail-loud rebalance-frequency guard (TDD)

**Files:**
- Create: `tests/research/regime_momentum_lab/test_cli_helpers.py`
- Modify: `scripts/backtest_scripts/run_momentum_variant.py`

- [ ] **Step 1: Write the failing test.**

Create `tests/research/regime_momentum_lab/test_cli_helpers.py`:
```python
"""Tests for run_momentum_variant CLI helpers (no market data needed)."""
import importlib.util
from pathlib import Path

import pytest

_CLI_PATH = Path('scripts/backtest_scripts/run_momentum_variant.py')


def _load_cli():
    spec = importlib.util.spec_from_file_location('run_momentum_variant', _CLI_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_daily_rebalance_is_allowed():
    cli = _load_cli()
    # Should not raise for the implemented cadence.
    cli._validate_rebalance_frequency('daily')


@pytest.mark.parametrize('freq', ['weekly_friday', 'weekly_wednesday'])
def test_weekly_rebalance_raises_not_implemented(freq):
    cli = _load_cli()
    with pytest.raises(NotImplementedError) as exc:
        cli._validate_rebalance_frequency(freq)
    assert 'PR 6' in str(exc.value)
```

- [ ] **Step 2: Run the test to verify it fails.**

Run:
```bash
pytest tests/research/regime_momentum_lab/test_cli_helpers.py -v 2>&1 | tail -10
```
Expected: FAIL with `AttributeError: module 'run_momentum_variant' has no attribute '_validate_rebalance_frequency'`.

- [ ] **Step 3: Implement the guard + add the arg.**

In `run_momentum_variant.py`, add the helper above `_parse_args`:
```python
def _validate_rebalance_frequency(freq: str) -> None:
    """Fail loud on cadences the engine does not yet honor (PR 6 builds weekly)."""
    if freq != 'daily':
        raise NotImplementedError(
            f'rebalance_frequency={freq!r} is not implemented in the engine yet '
            f'(see PR 6, gated on the daily cost verdict). Only "daily" is supported. '
            f'The HarnessConfig field exists but engine.run_variant does not branch on it.')
```
In `_parse_args`, add the argument:
```python
    p.add_argument('--rebalance-frequency',
                   choices=['daily', 'weekly_friday', 'weekly_wednesday'],
                   default='daily',
                   help='Rebalance cadence. Only "daily" is implemented today; '
                        'weekly_* is built in PR 6 (gated on the daily cost verdict).')
```
In `main()`, immediately after `args = _parse_args()`:
```python
    _validate_rebalance_frequency(args.rebalance_frequency)
```
And pass it into each `HarnessConfig(...)` construction:
```python
            rebalance_frequency=args.rebalance_frequency,
```

- [ ] **Step 4: Run the test to verify it passes.**

Run:
```bash
pytest tests/research/regime_momentum_lab/test_cli_helpers.py -v 2>&1 | tail -10
```
Expected: PASS (3 tests).

### Task 2.3: Add the turnover-summary line (TDD)

**Files:**
- Modify: `tests/research/regime_momentum_lab/test_cli_helpers.py`
- Modify: `scripts/backtest_scripts/run_momentum_variant.py`

- [ ] **Step 1: Write the failing test.**

Append to `tests/research/regime_momentum_lab/test_cli_helpers.py`:
```python
def test_format_turnover_line_shape():
    cli = _load_cli()
    # avg_daily_turnover is monkeypatched indirectly: pass records the metric can read.
    # Use a stub that the formatter routes through the real metric.
    line = cli._format_turnover_line('prod', 5.0, _turnover=0.1873)
    assert line.startswith('[turnover] prod @ 5.0bps:')
    assert 'avg_daily_turnover=0.1873' in line
    assert '18.73% of portfolio/day' in line
```

- [ ] **Step 2: Run the test to verify it fails.**

Run:
```bash
pytest tests/research/regime_momentum_lab/test_cli_helpers.py::test_format_turnover_line_shape -v 2>&1 | tail -8
```
Expected: FAIL with `AttributeError: ... has no attribute '_format_turnover_line'`.

- [ ] **Step 3: Implement the formatter + wire it into `main()`.**

In `run_momentum_variant.py`, add:
```python
def _format_turnover_line(variant: str, bps: float, records=None, _turnover: float | None = None) -> str:
    """One-line realized-turnover summary. Pass _turnover directly in tests; in
    production pass `records` and let it route through the metric."""
    if _turnover is None:
        from src.research.regime_momentum_lab.metrics import avg_daily_turnover
        _turnover = avg_daily_turnover(records)
    return (f'[turnover] {variant} @ {bps}bps: '
            f'avg_daily_turnover={_turnover:.4f} ({_turnover * 100:.2f}% of portfolio/day)')
```
In `main()`, after the `records_by_tier` loop completes and before/after writing the report:
```python
    for bps, records in records_by_tier.items():
        print(_format_turnover_line(args.variant, bps, records=records))
```

- [ ] **Step 4: Run the test to verify it passes.**

Run:
```bash
pytest tests/research/regime_momentum_lab/test_cli_helpers.py -v 2>&1 | tail -10
```
Expected: PASS (4 tests).

- [ ] **Step 5: Confirm `--help` shows the new arg and weekly fails loud.**

Run:
```bash
python scripts/backtest_scripts/run_momentum_variant.py --help 2>&1 | grep -A2 rebalance-frequency
python scripts/backtest_scripts/run_momentum_variant.py --variant V03 \
  --rebalance-frequency weekly_friday --start 2025-01-01 --end 2026-04-30 \
  --cost-bps 5 --output /tmp/should_not_write.md 2>&1 | tail -3
```
Expected: help shows `--rebalance-frequency`; the weekly invocation raises `NotImplementedError` and exits non-zero (no file written). This works WITHOUT H: data because the guard fires before any data access.

- [ ] **Step 6: Commit.**

```bash
git add scripts/backtest_scripts/run_momentum_variant.py tests/research/regime_momentum_lab/test_cli_helpers.py
git commit -m "feat(research): rename CLI to run_momentum_variant; fail-loud --rebalance-frequency + turnover line

Renamed ramp_phase4_backtest.py -> run_momentum_variant.py. Added
--rebalance-frequency that raises NotImplementedError on weekly_* (the engine
does not honor it yet; PR 6 builds it) instead of silently running daily. Added
a [turnover] stdout line per cost tier via avg_daily_turnover. Guard + formatter
are pure helpers with unit tests (no market data needed)."
```

---

## PR 3: Variant registry -- descriptive ids, aliases, `plain` + `bear_cash`

> Real TDD. Rewrite `test_variants.py` first (red), then rebuild `variants.py` (green). The renamed `prod`/`prod_no_crash` keep the old `V03`/`V01` bodies, so their numbers are unchanged.

### Task 3.1: Rewrite `test_variants.py` for the new model (red)

**Files:**
- Modify: `tests/research/regime_momentum_lab/test_variants.py`

- [ ] **Step 1: Replace the registry/id tests and add alias + behavior tests.**

Replace the entire body of `tests/research/regime_momentum_lab/test_variants.py` with:
```python
"""Tests for variants.py: registry ids, aliases, resolve(), and the four plan_fns."""
from datetime import datetime
import numpy as np
import pandas as pd
import pytest

from src.research.regime_momentum_lab.variants import REGISTRY, VariantSpec, resolve


def _calm_panel(n=300):
    idx = pd.date_range('2023-01-02', periods=n, freq='B')
    return pd.DataFrame({
        'AAA': 100 + np.arange(n) * 0.05,
        'BBB': 110 + np.arange(n) * 0.04,
        'CCC': 90 + np.arange(n) * 0.06,
        'SPY': 400 + np.arange(n) * 0.1,   # uptrend -> STRONG_BULL
        'VIX': np.full(n, 12.0),           # low vol
    }, index=idx)


def _crash_panel(n=300):
    idx = pd.date_range('2023-01-02', periods=n, freq='B')
    spy_path = np.concatenate([400 + np.arange(n - 30) * 0.1, np.linspace(430, 380, 30)])
    vix_path = np.concatenate([np.full(n - 30, 12.0), np.linspace(20, 35, 30)])
    return pd.DataFrame({
        'AAA': 100 + np.arange(n) * 0.05,
        'BBB': 110 + np.arange(n) * 0.04,
        'CCC': 90 + np.arange(n) * 0.06,
        'SPY': spy_path,
        'VIX': vix_path,
    }, index=idx)


def _call(variant_id, panel):
    spec = REGISTRY[variant_id]
    state = type('S', (), {'positions': {}, 'cash_usd': 100000.0})()
    cfg = type('C', (), {})()
    return spec.plan_fn(panel.index[-1].to_pydatetime(), state, panel, cfg)


def test_registry_has_four_canonical_ids():
    assert set(REGISTRY) == {'plain', 'prod', 'prod_no_crash', 'bear_cash'}
    for spec in REGISTRY.values():
        assert isinstance(spec, VariantSpec)


def test_resolve_aliases():
    assert resolve('V03').id == 'prod'
    assert resolve('V0').id == 'prod'
    assert resolve('V01').id == 'prod_no_crash'
    assert resolve('V1').id == 'plain'
    assert resolve('V8').id == 'bear_cash'
    # Canonical ids resolve to themselves.
    assert resolve('prod').id == 'prod'


def test_resolve_unknown_raises():
    with pytest.raises(KeyError):
        resolve('nonsense')
    # v11 is a toggle value, NOT a registry variant (spec Appendix B).
    with pytest.raises(KeyError):
        resolve('v11')


def test_prod_no_crash_full_gross_in_calm():
    plan = _call('prod_no_crash', _calm_panel())
    body = {k: v for k, v in plan.items() if k != '__regime__'}
    assert abs(sum(body.values()) - 1.0) < 0.01


def test_prod_applies_crash_exposure_in_crash():
    plan = _call('prod', _crash_panel())
    body = {k: v for k, v in plan.items() if k != '__regime__'}
    assert sum(body.values()) <= 0.6   # 0.5 with epsilon


def test_prod_and_prod_no_crash_identical_in_calm():
    p_nc = _call('prod_no_crash', _calm_panel())
    p_pr = _call('prod', _calm_panel())
    assert set(p_nc) - {'__regime__'} == set(p_pr) - {'__regime__'}
    for sym in set(p_nc) - {'__regime__'}:
        assert abs(p_nc[sym] - p_pr[sym]) < 1e-6


def test_plain_ignores_regime_keeps_full_gross_in_crash():
    """plain records the regime but does not act on it -> full gross even in crash."""
    plain = _call('plain', _crash_panel())
    prod = _call('prod', _crash_panel())
    plain_body = {k: v for k, v in plain.items() if k != '__regime__'}
    prod_body = {k: v for k, v in prod.items() if k != '__regime__'}
    assert abs(sum(plain_body.values()) - 1.0) < 0.01   # full exposure regardless of regime
    assert sum(prod_body.values()) <= 0.6               # prod cuts exposure
    assert '__regime__' in plain                        # regime still recorded


@pytest.mark.skip(reason="needs deterministic BEAR fixture; see spec PR 3.3")
def test_bear_cash_goes_to_cash_in_bear():
    """When the detector classifies BEAR, bear_cash holds no positions (all cash)."""
    plan = _call('bear_cash', _crash_panel())   # _crash_panel may not reach BEAR
    if plan.get('__regime__') != 'BEAR':
        pytest.skip('panel did not classify BEAR; needs a dedicated BEAR fixture')
    body = {k: v for k, v in plan.items() if k != '__regime__'}
    assert abs(sum(body.values())) < 1e-9
```

- [ ] **Step 2: Run to verify it fails.**

Run:
```bash
pytest tests/research/regime_momentum_lab/test_variants.py -v 2>&1 | tail -15
```
Expected: FAIL/ERROR at import -- `ImportError: cannot import name 'resolve'` (and the new ids don't exist yet).

### Task 3.2: Rebuild `variants.py` (green)

**Files:**
- Modify: `src/research/regime_momentum_lab/variants.py`

- [ ] **Step 1: Add `aliases` to `VariantSpec` and the `resolve()` function.**

In `variants.py`, change the dataclass:
```python
@dataclass(frozen=True)
class VariantSpec:
    id: str
    description: str
    plan_fn: PlanFn
    aliases: tuple[str, ...] = ()
```
Add, after the dataclass (before `_DETECTOR`):
```python
def resolve(name: str) -> "VariantSpec":
    """Look up a variant by canonical id or any registered alias.

    Lets historical report labels (V0, V01, V03, V1, V8) keep resolving after
    the rename. Raises KeyError with the full id list on miss. Note: v11 is a
    toggle value, not a registry variant (see spec Appendix B).
    """
    if name in REGISTRY:
        return REGISTRY[name]
    for spec in REGISTRY.values():
        if name in spec.aliases:
            return spec
    raise KeyError(
        f'Unknown variant {name!r}. Known ids: {sorted(REGISTRY)}; '
        f'aliases: {sorted(a for s in REGISTRY.values() for a in s.aliases)}')
```

- [ ] **Step 2: Thread `bear_to_cash` through `_compute_plan_from_panel`.**

Change the signature:
```python
def _compute_plan_from_panel(t: datetime, panel: pd.DataFrame, bear_to_cash: bool = False) -> "RampPlan":
```
and the `compute_plan(...)` call at the end -- add the kwarg:
```python
    return compute_plan(
        as_of=t,
        regime=regime,
        regime_confidence=confidence,
        regime_scores=regime_scores,
        top_n=top_n,
        momentum_scores=momentum,
        current_positions={},
        vix=float(vix_slice.iloc[-1]),
        spy_drawdown=spy_dd,
        bear_to_cash=bear_to_cash,
    )
```

- [ ] **Step 3: Rename the two plan_fns (bodies unchanged) and add the two new ones + `V1_PARAMS`.**

Rename `_variant_v01` -> `_variant_prod_no_crash` and `_variant_v03` -> `_variant_prod` (keep their bodies exactly). Then add, before `REGISTRY`:
```python
# Vanilla-momentum fixed params. Origin: ramp_root_cause_20260505.py (archived).
V1_PARAMS = {'long_p': 21, 'short_p': 5, 'long_w': 0.3, 'pen_w': 5.0, 'top_n': 10}


def _variant_bear_cash(t: datetime, state, panel: pd.DataFrame, cfg) -> Dict[str, float]:
    """prod, but BEAR regime -> 100% cash (bear_to_cash=True)."""
    plan = _compute_plan_from_panel(t, panel, bear_to_cash=True)
    if plan is None:
        return {'__regime__': 'SAFE_MODE'}
    targets = list(plan.targets.keys())
    if not targets:
        # BEAR cashed out: empty targets -> engine sells existing positions to zero.
        return {'__regime__': plan.regime}
    per_weight = float(plan.exposure_pct) / len(targets)
    out: Dict[str, float] = {sym: per_weight for sym in targets}
    out['__regime__'] = plan.regime
    return out


def _variant_plain(t: datetime, state, panel: pd.DataFrame, cfg) -> Dict[str, float]:
    """Vanilla momentum: fixed params, equal-weight top-N, full exposure.

    The regime is recorded for forensics but has zero effect on selection,
    sizing, or exposure (no overlay).
    """
    spy = panel['SPY'].dropna()
    vix = panel['VIX'].dropna()
    if t not in spy.index or t not in vix.index:
        return {'__regime__': 'SAFE_MODE'}
    spy_slice = spy.loc[:t]
    vix_slice = vix.loc[:t]
    if len(spy_slice) < 252 or len(vix_slice) < 252:
        return {'__regime__': 'SAFE_MODE'}
    # Regime label for forensics only.
    spy_df = pd.DataFrame({'close': spy_slice, 'open': spy_slice, 'high': spy_slice,
                           'low': spy_slice, 'volume': 1e6})
    vix_df = pd.DataFrame({'close': vix_slice})
    try:
        regime, _ = _DETECTOR.classify_regime(spy_df, vix_df, t)
    except Exception:
        regime = 'SAFE_MODE'
    universe_cols = [c for c in panel.columns if c not in ('SPY', 'VIX')]
    prices_slice = panel.loc[:t, universe_cols]
    ramp = RAMPSignals(symbols=universe_cols)
    ramp._current_params = V1_PARAMS
    momentum = ramp.calculate_momentum_scores(prices_slice)
    if momentum is None or len(momentum) == 0:
        return {'__regime__': regime}
    top = list(momentum.index[:V1_PARAMS['top_n']])
    if not top:
        return {'__regime__': regime}
    per_weight = 1.0 / len(top)   # full exposure, equal weight, no crash multiplier
    out: Dict[str, float] = {sym: per_weight for sym in top}
    out['__regime__'] = regime
    return out
```

- [ ] **Step 4: Rebuild `REGISTRY`.**

Replace the `REGISTRY` dict with:
```python
REGISTRY: Dict[str, VariantSpec] = {
    'plain': VariantSpec(
        id='plain',
        description='Vanilla momentum, fixed params (pen_w=5.0, top_n=10), no regime overlay; '
                    'regime recorded for forensics only',
        plan_fn=_variant_plain,
        aliases=('V1',),
    ),
    'prod': VariantSpec(
        id='prod',
        description='Production RAMP: regime overlay + per-regime params + 0.5x crash multiplier',
        plan_fn=_variant_prod,
        aliases=('V03', 'V0'),
    ),
    'prod_no_crash': VariantSpec(
        id='prod_no_crash',
        description='Overlay + per-regime params, crash multiplier ignored (parity-test baseline)',
        plan_fn=_variant_prod_no_crash,
        aliases=('V01',),
    ),
    'bear_cash': VariantSpec(
        id='bear_cash',
        description='Production overlay but BEAR regime -> 100% cash (bear_to_cash=True)',
        plan_fn=_variant_bear_cash,
        aliases=('V8',),
    ),
}
```

- [ ] **Step 5: Run the tests to verify they pass.**

Run:
```bash
pytest tests/research/regime_momentum_lab/test_variants.py -v 2>&1 | tail -20
```
Expected: PASS (the `bear_cash` BEAR test is skipped; all others pass). If `test_plain_ignores_regime_keeps_full_gross_in_crash` fails, `_variant_plain` is wrongly routing through the overlay -- recheck Step 3.

### Task 3.3: Point `_make_parity_report.py` at `resolve()`

**Files:**
- Modify: `scripts/backtest_scripts/_make_parity_report.py`

- [ ] **Step 1: Re-read how it references the ids.**

Run:
```bash
grep -n "REGISTRY\|V01\|V03\|resolve" scripts/backtest_scripts/_make_parity_report.py
```

- [ ] **Step 2: Replace `REGISTRY['V01']`/`REGISTRY['V03']` with `resolve('V01')`/`resolve('V03')`.**

Update the import to include `resolve`:
```python
from src.research.regime_momentum_lab.variants import resolve
```
and replace each `REGISTRY['V01']` -> `resolve('V01')`, `REGISTRY['V03']` -> `resolve('V03')`. Leave the `v01_records`/`v03_records` internal parameter names as-is (changing them means touching `reports.py` + `test_reports.py` together -- out of scope).

- [ ] **Step 3: Confirm it parses and resolves.**

Run:
```bash
python -c "import ast; ast.parse(open('scripts/backtest_scripts/_make_parity_report.py').read()); print('parse OK')"
python -c "from src.research.regime_momentum_lab.variants import resolve; print(resolve('V01').id, resolve('V03').id)"
```
Expected: `parse OK` then `prod_no_crash prod`.

### Task 3.4: Make the CLI accept canonical ids and aliases via `resolve()`

**Files:**
- Modify: `scripts/backtest_scripts/run_momentum_variant.py`

- [ ] **Step 1: Use `resolve()` and drop the hard `choices=`.**

Change the import:
```python
from src.research.regime_momentum_lab.variants import REGISTRY, resolve
```
In `_parse_args`, change the `--variant` line from:
```python
    p.add_argument('--variant', required=True, choices=list(REGISTRY.keys()))
```
to:
```python
    p.add_argument('--variant', required=True,
                   help='Variant id or legacy alias. Known ids: '
                        'plain, prod, prod_no_crash, bear_cash (aliases: V0/V01/V03/V1/V8).')
```
In `main()`, change:
```python
    spec = REGISTRY[args.variant]
```
to:
```python
    spec = resolve(args.variant)
```
And change the report's `variant_id=args.variant` to `variant_id=spec.id` so the report header shows the canonical id.

- [ ] **Step 2: Confirm alias + canonical id both resolve at the CLI layer.**

Run:
```bash
pytest tests/research/regime_momentum_lab/test_cli_helpers.py -v 2>&1 | tail -6
python scripts/backtest_scripts/run_momentum_variant.py --help 2>&1 | grep -A2 -- '--variant'
```
Expected: CLI helper tests still pass; help shows the known-ids text.

### Task 3.5: Numeric-identity for renamed variants **[BOX-ONLY]** and commit

- [ ] **Step 1: Confirm `prod`/`prod_no_crash` match the old `V03`/`V01`.** **[BOX-ONLY]**

Run:
```bash
python scripts/backtest_scripts/run_momentum_variant.py --variant prod \
  --start 2025-01-01 --end 2026-04-30 --cost-bps 0,5 --output /tmp/prod.md
# metric tables in /tmp/prod.md must match /tmp/v03_after.md from PR 1 (modulo the id header)
diff <(grep -v '^- \*\*Variant' /tmp/prod.md) <(grep -v '^- \*\*Variant' /tmp/v03_after.md) | head
```
Expected: no metric-table differences (only the variant-id header line, already filtered). Also run `plain`, `bear_cash`, and the `V03` alias end-to-end to confirm they produce reports. **Sanity (not a hard gate):** at 0% cost, ordering should be `bear_cash > plain > prod`. If it inverts, a plan_fn is wrong -- investigate before PR 5a.

If in a sandbox, leave a `NUMERIC-IDENTITY DEFERRED` note for the operator.

- [ ] **Step 2: Confirm no `V0n` survives as a registry key.**

Run:
```bash
grep -n "'V0" src/research/regime_momentum_lab/variants.py | grep -i "VariantSpec\|REGISTRY" || echo "no V0n keys (good)"
```
Expected: `no V0n keys (good)` (V0/V01/V03 appear only inside `aliases=(...)`).

- [ ] **Step 3: Commit.**

```bash
git add src/research/regime_momentum_lab/variants.py \
        tests/research/regime_momentum_lab/test_variants.py \
        scripts/backtest_scripts/_make_parity_report.py \
        scripts/backtest_scripts/run_momentum_variant.py
git commit -m "feat(research): variant registry -- descriptive ids, aliases, resolve(), plain + bear_cash

Renamed V01/V03 -> prod_no_crash/prod (bodies unchanged, byte-identical output).
Added resolve() mapping legacy labels V0/V01/V03/V1/V8 to canonical ids, the
plain variant (vanilla momentum, fixed V1_PARAMS, regime recorded but not acted
on), and bear_cash (bear_to_cash=True). CLI + parity report use resolve()."
```

---

## PR 4: Archive dated scripts; toggle field left inert (documented)

> No variant code or config changes (spec Appendix B). The `ramp.variant` field stays untouched; this PR only archives dead scripts.

### Task 4.1: Confirm nothing imports the dated scripts

**Files:** none.

- [ ] **Step 1: Verify the exact filenames.**

Run:
```bash
ls scripts/backtest_scripts/ramp_*.py
```
Expected: `ramp_root_cause_20260505.py`, `ramp_phase3a_variants_20260505.py`, `ramp_phase3b_bear_optimizer_20260505.py`, `ramp_re_eval_20260504.py` (re-verify the dated suffixes).

- [ ] **Step 2: Confirm zero imports of them.**

Run:
```bash
grep -rn "ramp_root_cause\|ramp_phase3a\|ramp_phase3b\|ramp_re_eval" --include='*.py' src/ scripts/ tests/
```
Expected: **zero import hits** (they are entrypoint scripts). If any are imported, STOP -- archival becomes extract-then-move.

### Task 4.2: Move the scripts and add the archive README

**Files:**
- Move: 4 scripts -> `scripts/backtest_scripts/_archived/`
- Create: `scripts/backtest_scripts/_archived/README.md`

- [ ] **Step 1: Create the archive and move (re-verify names from Task 4.1).**

Run:
```bash
mkdir -p scripts/backtest_scripts/_archived
git mv scripts/backtest_scripts/ramp_root_cause_20260505.py            scripts/backtest_scripts/_archived/
git mv scripts/backtest_scripts/ramp_phase3a_variants_20260505.py      scripts/backtest_scripts/_archived/
git mv scripts/backtest_scripts/ramp_phase3b_bear_optimizer_20260505.py scripts/backtest_scripts/_archived/
git mv scripts/backtest_scripts/ramp_re_eval_20260504.py               scripts/backtest_scripts/_archived/
```

- [ ] **Step 2: Write the README.**

Create `scripts/backtest_scripts/_archived/README.md`:
```markdown
# Archived RAMP investigation scripts

These dated scripts performed the RAMP alpha-decay root-cause investigation
(2026-05). Their FINDINGS are preserved in docs/reports/ramp/20260505_*.md.
Their CODE is archived (not deleted) for audit. Each reimplemented its own
backtest loop, metrics, and data loader; that functionality now lives in the
tested harness at src/research/regime_momentum_lab/. Variants V0/V01/V03/V1/V8
referenced here map to the registry ids prod/prod_no_crash/plain/bear_cash via
variants.resolve(). Excluded from pytest discovery via norecursedirs in pytest.ini.

Do not re-activate by importing. To reproduce a finding, run the equivalent
registry variant through scripts/backtest_scripts/run_momentum_variant.py.
```

- [ ] **Step 3: Confirm pytest does not collect the archive.**

Run:
```bash
pytest --collect-only -q 2>&1 | grep -c _archived
```
Expected: `0`.

- [ ] **Step 4: Full suite still green.**

Run:
```bash
pytest tests/research/regime_momentum_lab/ -q 2>&1 | tail -5
```
Expected: all pass (41 original minus any moved + the new CLI/variant tests; the bear_cash BEAR test skipped).

- [ ] **Step 5: Commit.**

```bash
git add -A
git commit -m "chore(research): archive dated RAMP investigation scripts under _archived/

Moved ramp_root_cause/phase3a/phase3b/re_eval scripts to _archived/ (excluded
from pytest via norecursedirs) with a README mapping their V0n labels to the
registry. No variant code or strategy_toggle.yaml change -- the ramp.variant
field is present-but-inert and deferred to the post-PR-5 deploy decision (spec
Appendix B)."
```

---

## PR 5a: Daily turnover + cost verdict **[BOX-ONLY -- research execution, not TDD]**

> This is not a code PR and has no failing test. It runs the harness on the operator's box (needs H: SIP data) and writes a report. An agent's role is to prepare commands, reuse prior outputs, and scaffold the report; the data-dependent runs are handed off.

### Task 5a.0: Reuse/cross-check existing outputs first

- [ ] **Step 1: Read the prior phase4 reports before regenerating.**

Run:
```bash
ls docs/reports/ramp/20260519_phase4_v01*.md docs/progress/20260519_RAMP_PHASE4_phaseB_*.md
```
Read `docs/reports/ramp/20260519_phase4_v01.md` and `..._v01_vs_v03_parity.md`. They likely already contain `V01`/`V03` (now `prod_no_crash`/`prod`) turnover and parity numbers over an overlapping window. Reuse where they answer 5a; cross-check any regenerated number against them (large divergence on the same variant/window is a finding -- the rename was numerically identical). Only regenerate what they don't cover -- notably `plain` and `bear_cash`.

### Task 5a.1: Source-path smoke-check **[BOX-ONLY]**

- [ ] **Step 1: Confirm the SIP source resolves before any run.**

Run (on the box):
```bash
python -c "from src.settings.data_paths import get_equities_sip_split_1min_dir as g; p=g(); print(p, p.exists())"
```
Expected: a path under the H: storage root ending `equities/sip_split/1min` and `True`. If `False`, the migration has not run there -- resolve before proceeding (the rebuild will fail "SIP split tree not found").

### Task 5a.2: Measure daily turnover + cost sensitivity **[BOX-ONLY]**

- [ ] **Step 1: Turnover at 0 bps, full span + EXT-OOS.**

```bash
mkdir -p docs/reports/ramp/_scratch
for V in plain prod bear_cash; do
  python scripts/backtest_scripts/run_momentum_variant.py --variant $V \
    --start 2017-01-01 --end 2026-04-30 --cost-bps 0 --rebalance-frequency daily \
    --output docs/reports/ramp/_scratch/${V}_full_daily.md
  python scripts/backtest_scripts/run_momentum_variant.py --variant $V \
    --start 2025-01-01 --end 2026-04-30 --cost-bps 0 --rebalance-frequency daily \
    --output docs/reports/ramp/_scratch/${V}_ext_daily.md
done
```
Record the `[turnover]` line for each (the headline `T`).

- [ ] **Step 2: Daily cost sweep.**

```bash
for V in plain prod bear_cash; do
  python scripts/backtest_scripts/run_momentum_variant.py --variant $V \
    --start 2025-01-01 --end 2026-04-30 --cost-bps 0,2.5,5,7.5 --rebalance-frequency daily \
    --output docs/reports/ramp/_scratch/${V}_ext_costsweep_daily.md
done
```

### Task 5a.3: Write the findings report with the PRE-REGISTERED 3-branch verdict

**Files:**
- Create: `docs/reports/ramp/<YYYYMMDD>_turnover_cost_sensitivity.md`

- [ ] **Step 1: Write the verdict thresholds FIRST (before pasting numbers) and commit that skeleton.**

Create the report with Context / Methodology / (empty) Results / and the pre-committed decision tree, then commit it *before* filling results, so commit order proves the gate predates the numbers:
```markdown
## Pre-registered verdict (thresholds fixed before results)
Best variant over EXT-OOS, target cost tier = 5 bps. SE on ~331 days ~= 0.17, so
treat |Sharpe| < ~0.2 as indistinguishable from zero.
- Branch 1 (clears costs daily): best variant NET Sharpe @5bps > ~0.2 and net CAGR > 0
  -> keep daily, NO PR 6.
- Branch 2 (cost-bound): best variant GROSS Sharpe @0bps > ~0.2 (gross CAGR > 0) BUT
  net @5bps <= ~0 -> ONLY PR 6 trigger.
- Branch 3 (alpha-dead): best variant GROSS Sharpe @0bps <= ~0.2 -> retire/restructure,
  NO PR 6 (weekly cannot manufacture gross return).
```
```bash
git add docs/reports/ramp/*_turnover_cost_sensitivity.md
git commit -m "report(ramp): pre-register turnover/cost verdict thresholds (pre-results)"
```

- [ ] **Step 2: Fill in the results and evaluate the branch.**

Populate: realized `T` per variant vs the assumed 1.0; net Sharpe/CAGR/MaxDD at 0/2.5/5/7.5 bps daily; the forensics (on days `plain` beats `prod`, the recorded regime); the statistical-honesty caveat; and state explicitly which branch fired and whether PR 6 is triggered. Commit.

- [ ] **Step 3: Update the RAMP report index** with a pointer + the `T` number.

### Validation (internal consistency)

- [ ] cost_drag at 5 bps ~= `2 * 0.0005 * T * days * mean_portfolio_value` for a spot-checked variant (else the cost model and turnover metric disagree).

---

## PR 6 (gated): Weekly rebalance -- separate plan when triggered

**Do not build unless PR 5a fires Branch 2.** When triggered, write a dedicated plan (`docs/planning/<date>_ramp_weekly_rebalance_plan.md`) covering:
- Engine: branch `run_variant` on `cfg.rebalance_frequency`; on non-rebalance days carry positions + mark-to-market with zero turnover; resolve the holiday rule explicitly ("rebalance on the last trading day on-or-before the target weekday each week") and document it in the engine docstring.
- Tests (`test_engine.py`): trades fire only on the target weekday; ~1/5 turnover vs daily over a multi-week window; full-week carry preserves positions.
- Remove the PR 2 `NotImplementedError` guard.
- Weekly cost sweep + report addendum + final cross-cadence verdict (soften the CSCM analogy per spec).

Building it for Branch 1 (already clears) or Branch 3 (alpha dead) is wasted work -- weekly only cuts cost, it cannot create gross return.

---

## Self-Review (completed by plan author)

- **Spec coverage:** PR 1 (rename + 1.2b source path) -> Tasks 1.1-1.6. PR 2 (CLI + fail-loud + turnover) -> 2.1-2.3. PR 3 (registry/aliases/resolve/plain/bear_cash) -> 3.1-3.5. PR 4 (archive + inert toggle) -> 4.1-4.2. PR 5a (daily verdict, reuse 36eb91b, source smoke-check, pre-registered 3-branch) -> 5a.0-5a.3. PR 6 -> gated stub. All spec sections mapped.
- **Placeholders:** none -- every code step shows complete code; the only `<YYYYMMDD>` is a deliberate execution-time report date.
- **Type consistency:** `VariantSpec.aliases: tuple[str,...]`, `resolve()->VariantSpec`, plan_fns return `Dict[str,float]` with `__regime__` sentinel, `_compute_plan_from_panel(t, panel, bear_to_cash=False)` -- consistent across Tasks 3.1-3.4 and the tests.
- **Run-location:** [BOX-ONLY] steps (numeric-identity diffs, PR 5a runs) clearly separated from sandbox-runnable mock tests.
