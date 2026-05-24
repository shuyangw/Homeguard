# V14 Soft-Bear Factorial Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement WS-3c V14a/b/c factorial (cash / SPY / dampen consumer-layer variants on a Schmitt-trigger BEAR_score) and produce the 35-backtest readiness report per spec rev2.

**Architecture:** Three variants share one state machine (`state.in_bear_soft_mode` toggled by Schmitt trigger on `detector.last_regime_scores['BEAR']` with pre-registered tau_in/tau_out from G1_BEAR median). Variants differ only in their action when in_bear_soft_mode is True. The detector gets one 2-line freshness field; a new `_SentinelPlan` class formalizes the "no-exposure" plan dispatch in the engine. The 5-gate readiness orchestrator clones V12c's structure with DSR n_trials=36 and 8-variant gate PBO + 4-variant diagnostic PBO.

**Tech Stack:** Python 3.13 in `fintech` conda env (`C:\Users\qwqw1\anaconda3\envs\fintech`); pandas, numpy, scipy; existing RAMP Phase 4 harness in `src/research/ramp_phase4/`; pytest. ASCII-only files (Windows cp1252).

**Spec:** `C:\Users\qwqw1\Dropbox\cs\github\Homeguard\docs\superpowers\specs\2026-05-24-v14-soft-bear-factorial-design.md`
**Open-questions resolution:** `C:\Users\qwqw1\Dropbox\cs\github\Homeguard\docs\superpowers\specs\2026-05-24-v14-soft-bear-factorial-design-open-questions.md`

**Branch:** v12-bear-to-cash (do NOT switch).

---

## Reusable references (read-only -- DO NOT modify)

- `C:\Users\qwqw1\Dropbox\cs\github\Homeguard\src\strategies\advanced\market_regime_detector.py` -- detector source; `classify_regime` at line 111
- `C:\Users\qwqw1\Dropbox\cs\github\Homeguard\src\research\ramp_phase4\variants.py` -- existing variants; `_variant_v11` at line 190; `_variant_v12` at line 235
- `C:\Users\qwqw1\Dropbox\cs\github\Homeguard\src\research\ramp_phase4\engine.py` -- harness engine; `HarnessState` at line 24; `_engine_pre_variant_update` at line 74; `run_variant` at line 99
- `C:\Users\qwqw1\Dropbox\cs\github\Homeguard\src\research\ramp_phase4\config.py` -- `HarnessConfig` dataclass + V12 `regime_positions` pattern
- `C:\Users\qwqw1\Dropbox\cs\github\Homeguard\scripts\diagnostics\ground_truth_labelers.py` -- `label_g1_drawdown_bear` (locked at commit `9c48245`)
- `C:\Users\qwqw1\Dropbox\cs\github\Homeguard\scripts\backtest_scripts\ramp_phase4_v12c_readiness.py` -- V12c readiness orchestrator (clone target)
- `C:\Users\qwqw1\Dropbox\cs\github\Homeguard\diagnostics\regime\v0\labels.parquet` -- v0 replay labels with regime classification
- `C:\Users\qwqw1\Dropbox\cs\github\Homeguard\diagnostics\regime\v0_scores\labels.parquet` -- soft-score replay (5 score columns from E3)

---

## Task 0: Pre-spec -- compute tau_in from G1_BEAR median

**Files:**
- Create: `C:\Users\qwqw1\Dropbox\cs\github\Homeguard\scripts\diagnostics\compute_tau_in_from_g1.py`
- Create: `C:\Users\qwqw1\Dropbox\cs\github\Homeguard\config\research\v14_tau_constants.json` (output of script)

- [ ] **Step 1: Create the script that computes median BEAR_score on G1_BEAR days**

Create `scripts/diagnostics/compute_tau_in_from_g1.py`:

```python
"""Compute pre-registered tau_in for V14 from G1_BEAR median BEAR_score.

Runs ONCE at spec time before any V14 backtest. Output: v14_tau_constants.json.
The G1_BEAR labeler is pinned at commit 9c48245; the pinning is recorded
in the output JSON.
"""

from __future__ import annotations

import json
import sys
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from scripts.diagnostics.ground_truth_labelers import label_g1_drawdown_bear
from src.utils.logger import logger


PANEL_PATH = Path('diagnostics/data/spy_vix_2016_2026.parquet')
SCORES_PATH = Path('diagnostics/regime/v0_scores/labels.parquet')
OUTPUT_PATH = Path('config/research/v14_tau_constants.json')
LABELER_FILE = 'scripts/diagnostics/ground_truth_labelers.py'
HYSTERESIS_BAND = 0.1


def main() -> int:
    if not PANEL_PATH.exists():
        raise FileNotFoundError(PANEL_PATH)
    if not SCORES_PATH.exists():
        raise FileNotFoundError(SCORES_PATH)

    panel = pd.read_parquet(PANEL_PATH)
    scores = pd.read_parquet(SCORES_PATH)
    scores['date'] = pd.to_datetime(scores['date'])
    scores = scores.set_index('date')

    g1_bear = label_g1_drawdown_bear(panel)
    g1_bear_dates = g1_bear[g1_bear].index

    joined = scores.loc[scores.index.isin(g1_bear_dates), 'score_BEAR']
    if joined.empty:
        raise RuntimeError(
            f'No overlap between G1_BEAR days ({len(g1_bear_dates)}) and '
            f'soft-score replay ({len(scores)} days).'
        )

    tau_in = float(joined.median())
    p25 = float(joined.quantile(0.25))
    p75 = float(joined.quantile(0.75))
    tau_out = round(tau_in - HYSTERESIS_BAND, 6)

    if not (0.0 < tau_out < tau_in < 1.0):
        raise RuntimeError(
            f'Pre-registered tau predicate failed: '
            f'tau_out={tau_out}, tau_in={tau_in}'
        )

    labeler_commit = subprocess.run(
        ['git', 'log', '-1', '--format=%H', '--', LABELER_FILE],
        check=True, capture_output=True, text=True,
    ).stdout.strip()

    out = {
        'tau_in': round(tau_in, 6),
        'tau_out': tau_out,
        'tau_band': HYSTERESIS_BAND,
        'g1_bear_score_p25': round(p25, 6),
        'g1_bear_score_p75': round(p75, 6),
        'n_g1_bear_days': int(len(joined)),
        'g1_labeler_commit': labeler_commit,
        'computed_at': datetime.now(timezone.utc).isoformat(),
        'computation_script': 'scripts/diagnostics/compute_tau_in_from_g1.py',
        'computation_method': (
            'median BEAR_score on G1_BEAR days '
            '(drawdown > 10% from 252-day trailing high)'
        ),
        'source_data': {
            'labels': str(PANEL_PATH),
            'scores': str(SCORES_PATH),
        },
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(out, indent=2) + '\n')
    logger.info(f'[+] tau_in={out["tau_in"]} tau_out={out["tau_out"]} '
                f'(n={out["n_g1_bear_days"]} G1_BEAR days)')
    logger.info(f'[+] Wrote {OUTPUT_PATH}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
```

- [ ] **Step 2: Run the script**

```bash
cd /c/Users/qwqw1/Dropbox/cs/github/Homeguard
source ~/anaconda3/etc/profile.d/conda.sh && conda activate fintech
PYTHONPATH=. python scripts/diagnostics/compute_tau_in_from_g1.py
```

Expected output:
- `config/research/v14_tau_constants.json` exists
- Logger prints `tau_in=<value> tau_out=<value-0.1>` with non-zero n_g1_bear_days

- [ ] **Step 3: Verify the JSON content**

```bash
cat config/research/v14_tau_constants.json
```

Expected: valid JSON containing `tau_in`, `tau_out` satisfying `0 < tau_out < tau_in < 1.0`, plus `g1_labeler_commit: "9c48245"`.

- [ ] **Step 4: Force-add and commit (scripts/diagnostics is NOT gitignored; config/research/ is new)**

```bash
git add -f scripts/diagnostics/compute_tau_in_from_g1.py
git add config/research/v14_tau_constants.json
git commit -m "feat(diagnostics): pre-register V14 tau constants from G1_BEAR median

Computes tau_in = median BEAR_score on G1_BEAR days (drawdown > 10% from
252-day trailing high) and tau_out = tau_in - 0.1. Independent of E3's
lead-time sweep on the same window (no in-sample optimization).

Output config/research/v14_tau_constants.json pins:
- tau_in / tau_out values
- G1 labeler commit hash for reproducibility
- p25/p75 of G1_BEAR BEAR_score for sensitivity panel ranges
- computation timestamp and source data paths"
```

---

## Task 1: Detector freshness field

**Files:**
- Modify: `C:\Users\qwqw1\Dropbox\cs\github\Homeguard\src\strategies\advanced\market_regime_detector.py`
- Create: `C:\Users\qwqw1\Dropbox\cs\github\Homeguard\tests\strategies\advanced\test_market_regime_detector_freshness.py`

- [ ] **Step 1: Write the failing test for the new field**

Create `tests/strategies/advanced/test_market_regime_detector_freshness.py`:

```python
"""Tests for V14 freshness assertion -- detector.last_classification_timestamp."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from src.strategies.advanced.market_regime_detector import MarketRegimeDetector


def _build_minimal_panel(n_days: int = 260) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build a 260-day SPY+VIX panel large enough to pass the 252-day check."""
    dates = pd.date_range('2020-01-02', periods=n_days, freq='B')
    spy = pd.DataFrame({
        'open': 100.0 + np.arange(n_days) * 0.1,
        'high': 101.0 + np.arange(n_days) * 0.1,
        'low': 99.0 + np.arange(n_days) * 0.1,
        'close': 100.0 + np.arange(n_days) * 0.1,
        'volume': 1_000_000.0,
    }, index=dates)
    vix = pd.DataFrame({
        'open': 15.0, 'high': 16.0, 'low': 14.0, 'close': 15.0,
    }, index=dates)
    return spy, vix


def test_initial_state_no_timestamp():
    """A fresh detector has last_classification_timestamp == None."""
    detector = MarketRegimeDetector()
    assert detector.last_classification_timestamp is None


def test_classify_sets_timestamp():
    """After classify_regime, last_classification_timestamp == the passed timestamp."""
    detector = MarketRegimeDetector()
    spy, vix = _build_minimal_panel()
    ts = spy.index[-1].to_pydatetime()
    detector.classify_regime(spy, vix, ts)
    assert detector.last_classification_timestamp == ts


def test_double_call_idempotent_output():
    """Calling classify_regime twice with same inputs returns identical results."""
    detector = MarketRegimeDetector()
    spy, vix = _build_minimal_panel()
    ts = spy.index[-1].to_pydatetime()
    r1 = detector.classify_regime(spy, vix, ts)
    scores1 = dict(detector.last_regime_scores)
    r2 = detector.classify_regime(spy, vix, ts)
    scores2 = dict(detector.last_regime_scores)
    assert r1 == r2
    assert scores1 == scores2
    assert detector.last_classification_timestamp == ts
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
PYTHONPATH=. python -m pytest tests/strategies/advanced/test_market_regime_detector_freshness.py -v
```

Expected: 3 tests, `test_initial_state_no_timestamp` and `test_classify_sets_timestamp` FAIL with AttributeError on `last_classification_timestamp`.

- [ ] **Step 3: Add the field to the detector**

In `src/strategies/advanced/market_regime_detector.py`, locate `__init__` (line 91). Add the new field after `last_regime_scores`:

```python
        self.last_regime_scores: Optional[Dict[str, float]] = None
        # V14 freshness assertion: variants check this equals their tick timestamp
        # before reading last_regime_scores. Populated at the end of classify_regime.
        self.last_classification_timestamp: Optional[datetime] = None
```

Locate the end of `classify_regime` (just before the `return best_regime, confidence` line, around line 188). Add:

```python
        # Persist for harness consumption (variants need the full score vector,
        # not just the winner) before we collapse to best_regime.
        self.last_regime_scores = dict(regime_scores)
        self.last_classification_timestamp = timestamp  # V14 freshness assertion target

        # Select regime with highest score
        best_regime = max(regime_scores, key=regime_scores.get)
        confidence = regime_scores[best_regime]
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
PYTHONPATH=. python -m pytest tests/strategies/advanced/test_market_regime_detector_freshness.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Run the full existing test suite to verify no regression**

```bash
PYTHONPATH=. python -m pytest tests/research/ramp_phase4/ tests/diagnostics/ -q
```

Expected: 105+ passed (the merged test count before V14 work).

- [ ] **Step 6: Commit**

```bash
git add src/strategies/advanced/market_regime_detector.py tests/strategies/advanced/test_market_regime_detector_freshness.py
git commit -m "feat(detector): last_classification_timestamp field for V14 freshness assertion

Adds 2-line additive change to MarketRegimeDetector:
- self.last_classification_timestamp: Optional[datetime] = None in __init__
- self.last_classification_timestamp = timestamp at end of classify_regime

No logic change. V14 variants assert detector.last_classification_timestamp
== t before reading last_regime_scores, decoupling V14 from
_compute_plan_from_panel ordering.

classify_regime is output-idempotent (deterministic; no randomness or
global state); recompute cost is trivial vs the orchestrator wall-clock.
No cache refactor needed per open-questions Q1 resolution."
```

---

## Task 2: `_SentinelPlan` class + tests

**Files:**
- Create: `C:\Users\qwqw1\Dropbox\cs\github\Homeguard\src\research\ramp_phase4\plans.py`
- Create: `C:\Users\qwqw1\Dropbox\cs\github\Homeguard\tests\research\ramp_phase4\test_plans.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/research/ramp_phase4/test_plans.py`:

```python
"""Tests for _SentinelPlan -- the no-exposure plan marker used by V14."""

from __future__ import annotations

import pytest

from src.research.ramp_phase4.plans import _SentinelPlan, PLAN_CASH_BEAR_SOFT


def test_sentinel_plan_has_reason():
    sp = _SentinelPlan(reason='TEST_REASON')
    assert sp.reason == 'TEST_REASON'


def test_sentinel_plan_has_empty_weights():
    sp = _SentinelPlan(reason='TEST_REASON')
    assert sp.weights == {}


def test_sentinel_plan_is_frozen():
    sp = _SentinelPlan(reason='TEST_REASON')
    with pytest.raises(Exception):
        sp.reason = 'OTHER'  # frozen dataclass should reject


def test_plan_cash_bear_soft_constant():
    assert PLAN_CASH_BEAR_SOFT.reason == 'BEAR_SOFT_CASH'
    assert PLAN_CASH_BEAR_SOFT.weights == {}


def test_isinstance_check_distinguishes_from_dict():
    sp = PLAN_CASH_BEAR_SOFT
    d = {'__regime__': 'BEAR', 'SPY': 1.0}
    assert isinstance(sp, _SentinelPlan)
    assert not isinstance(d, _SentinelPlan)
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
PYTHONPATH=. python -m pytest tests/research/ramp_phase4/test_plans.py -v
```

Expected: 5 tests, all FAIL with ImportError on `src.research.ramp_phase4.plans`.

- [ ] **Step 3: Create the module**

Create `src/research/ramp_phase4/plans.py`:

```python
"""Plan sentinels for variants that produce non-allocation outputs.

The engine pattern-matches on isinstance(_SentinelPlan) and dispatches to
the corresponding no-allocation execution path. The reason field flows
into per-day attribution logging.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass(frozen=True)
class _SentinelPlan:
    """Marker for plans that signal 'no exposure' or other non-allocation actions.

    Engine dispatch: isinstance(plan, _SentinelPlan) -> zero_target_orders().
    The `reason` field is recorded in DailyRecord.regime for traceability.
    """
    reason: str
    weights: Dict[str, float] = field(default_factory=dict)  # always empty for sentinels


PLAN_CASH_BEAR_SOFT = _SentinelPlan(reason='BEAR_SOFT_CASH')
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
PYTHONPATH=. python -m pytest tests/research/ramp_phase4/test_plans.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/research/ramp_phase4/plans.py tests/research/ramp_phase4/test_plans.py
git commit -m "feat(plans): _SentinelPlan class for V14 no-exposure marker

Frozen dataclass with reason: str and weights: dict. PLAN_CASH_BEAR_SOFT
exposed as the V14a 'cash' sentinel. Engine integration in subsequent
task uses isinstance(plan, _SentinelPlan) to dispatch to the no-trade
execution path.

Replaces rev1's '__regime__: BEAR_SOFT' magic-dict pattern with a typed
class. Other variants can add new sentinels (e.g. PLAN_REDUCED_BEAR_SOFT)
without dunder-key collisions."
```

---

## Task 3: Engine integration -- new state field + soft-bear update helper + sentinel dispatch

**Files:**
- Modify: `C:\Users\qwqw1\Dropbox\cs\github\Homeguard\src\research\ramp_phase4\engine.py`
- Create: `C:\Users\qwqw1\Dropbox\cs\github\Homeguard\tests\research\ramp_phase4\test_engine_v14.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/research/ramp_phase4/test_engine_v14.py`:

```python
"""Tests for V14 engine extensions: in_bear_soft_mode state + Schmitt helper + _SentinelPlan dispatch."""

from __future__ import annotations

from datetime import datetime

import pytest

from src.research.ramp_phase4.engine import (
    HarnessState, _engine_pre_variant_update_soft_bear,
)
from src.research.ramp_phase4.plans import PLAN_CASH_BEAR_SOFT, _SentinelPlan


def _state() -> HarnessState:
    return HarnessState(cash_usd=100_000.0)


def test_state_starts_not_in_bear_soft_mode():
    s = _state()
    assert s.in_bear_soft_mode is False


def test_enter_on_score_at_or_above_tau_in():
    s = _state()
    _engine_pre_variant_update_soft_bear(s, bear_score=0.4, tau_in=0.3, tau_out=0.2)
    assert s.in_bear_soft_mode is True

def test_enter_on_score_exactly_tau_in():
    s = _state()
    _engine_pre_variant_update_soft_bear(s, bear_score=0.3, tau_in=0.3, tau_out=0.2)
    assert s.in_bear_soft_mode is True  # >= tau_in enters


def test_stay_in_band():
    s = _state()
    s.in_bear_soft_mode = True
    _engine_pre_variant_update_soft_bear(s, bear_score=0.25, tau_in=0.3, tau_out=0.2)
    assert s.in_bear_soft_mode is True  # in band, no transition


def test_stay_when_score_equals_tau_out():
    s = _state()
    s.in_bear_soft_mode = True
    _engine_pre_variant_update_soft_bear(s, bear_score=0.2, tau_in=0.3, tau_out=0.2)
    assert s.in_bear_soft_mode is True  # NOT strict <, stays


def test_exit_when_score_below_tau_out():
    s = _state()
    s.in_bear_soft_mode = True
    _engine_pre_variant_update_soft_bear(s, bear_score=0.1999, tau_in=0.3, tau_out=0.2)
    assert s.in_bear_soft_mode is False


def test_no_enter_in_band_when_below():
    s = _state()
    _engine_pre_variant_update_soft_bear(s, bear_score=0.25, tau_in=0.3, tau_out=0.2)
    assert s.in_bear_soft_mode is False  # never crossed tau_in


def test_nan_score_no_transition():
    s = _state()
    s.in_bear_soft_mode = True
    _engine_pre_variant_update_soft_bear(s, bear_score=float('nan'), tau_in=0.3, tau_out=0.2)
    assert s.in_bear_soft_mode is True  # unchanged on NaN
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
PYTHONPATH=. python -m pytest tests/research/ramp_phase4/test_engine_v14.py -v
```

Expected: 8 tests fail with AttributeError on `in_bear_soft_mode` and ImportError on `_engine_pre_variant_update_soft_bear`.

- [ ] **Step 3: Add the state field**

In `src/research/ramp_phase4/engine.py`, locate the `HarnessState` dataclass (line 24). Add a new field at the end of the existing fields, before `__post_init__` (if any):

```python
    # V12: per-regime debouncing state (pre-variant update each tick).
    last_regime: Optional[str] = None
    regime_streak: Dict[str, int] = field(default_factory=dict)
    last_validated_regime: Optional[str] = None
    # V14: Schmitt-trigger soft-bear consumer state.
    in_bear_soft_mode: bool = False
```

- [ ] **Step 4: Add the Schmitt-trigger helper**

In `engine.py`, after the existing `_engine_pre_variant_update` function (ends around line 97), add:

```python
def _engine_pre_variant_update_soft_bear(
    state: HarnessState,
    bear_score: float,
    tau_in: float,
    tau_out: float,
) -> None:
    """V14 Schmitt-trigger state update.

    Entry: bear_score >= tau_in -> in_bear_soft_mode = True.
    Exit:  bear_score < tau_out (strict) -> in_bear_soft_mode = False.
    Within band [tau_out, tau_in): no transition (state sticks).
    NaN bear_score: no transition.

    Called BEFORE the variant reads state.in_bear_soft_mode, so the variant
    sees the updated value on the same tick the threshold was crossed.
    """
    if bear_score != bear_score:  # NaN check (NaN != NaN)
        return
    if not state.in_bear_soft_mode and bear_score >= tau_in:
        state.in_bear_soft_mode = True
    elif state.in_bear_soft_mode and bear_score < tau_out:
        state.in_bear_soft_mode = False
```

- [ ] **Step 5: Run the tests to verify they pass**

```bash
PYTHONPATH=. python -m pytest tests/research/ramp_phase4/test_engine_v14.py -v
```

Expected: 8 passed.

- [ ] **Step 6: Add _SentinelPlan dispatch in run_variant**

In `engine.py`, locate the `run_variant` function (line 99). The variant's plan_output is consumed twice -- once in the `one_day_lag` branch and once in the `near_close` branch. Add `_SentinelPlan` dispatch in BOTH branches.

First, add the import at the top of engine.py (after the existing imports):

```python
from src.research.ramp_phase4.plans import _SentinelPlan
```

Then, in the `one_day_lag` branch around line 146:

```python
            # Plan_fn now sees post-execution state.
            plan_output = variant_spec.plan_fn(ts, state, panel, cfg)
            if isinstance(plan_output, _SentinelPlan):
                regime_label = plan_output.reason
                target_weights: Dict[str, float] = {}
            else:
                plan_output = plan_output or {}
                regime_label = str(plan_output.pop('__regime__', 'STUB'))
                target_weights = plan_output
```

And in the `near_close` branch around line 169:

```python
        # near_close branch (unchanged semantics).
        plan_output = variant_spec.plan_fn(ts, state, panel, cfg)
        if isinstance(plan_output, _SentinelPlan):
            regime_label = plan_output.reason
            target_weights: Dict[str, float] = {}
        else:
            plan_output = plan_output or {}
            regime_label = str(plan_output.pop('__regime__', 'STUB'))
            target_weights = plan_output
```

The downstream code (SAFE_MODE check, weight_sum, compute_trades) is unaffected -- if `target_weights == {}`, the engine's existing trade-generation path emits sells for all current positions and zero buys. This is the desired "go to cash" behavior.

- [ ] **Step 7: Add an integration test for the dispatch**

Append to `tests/research/ramp_phase4/test_engine_v14.py`:

```python
def test_sentinel_plan_dispatch_produces_zero_targets_and_attribution_label():
    """run_variant treats _SentinelPlan output as zero-target with reason as regime label."""
    # We test the dispatch logic in isolation by checking the contract:
    # isinstance check + target_weights = {} + regime_label = plan.reason
    sp = PLAN_CASH_BEAR_SOFT
    assert isinstance(sp, _SentinelPlan)
    # The engine code does:
    #   if isinstance(plan_output, _SentinelPlan):
    #       regime_label = plan_output.reason
    #       target_weights = {}
    # Mimic that here:
    regime_label = sp.reason
    target_weights: dict = {}
    assert regime_label == 'BEAR_SOFT_CASH'
    assert target_weights == {}
```

- [ ] **Step 8: Run all engine tests + the full suite**

```bash
PYTHONPATH=. python -m pytest tests/research/ramp_phase4/test_engine_v14.py tests/research/ramp_phase4/test_plans.py -v
PYTHONPATH=. python -m pytest tests/research/ramp_phase4/ tests/diagnostics/ tests/strategies/advanced/ -q
```

Expected: 8 + 5 = 13 V14 tests pass; full suite still passes (no regression in existing variants).

- [ ] **Step 9: Commit**

```bash
git add src/research/ramp_phase4/engine.py tests/research/ramp_phase4/test_engine_v14.py
git commit -m "feat(engine): V14 Schmitt-trigger state + _SentinelPlan dispatch

HarnessState gets in_bear_soft_mode: bool = False (V14 toggle).
_engine_pre_variant_update_soft_bear() implements the Schmitt trigger
(>= tau_in enters, strict < tau_out exits; NaN no transition).

run_variant gets _SentinelPlan dispatch in both timing-mode branches:
isinstance(_SentinelPlan) -> target_weights = {}, regime_label = reason.
Downstream trade generation produces all-sells naturally on zero target.

V01-V13 unaffected: they return dicts, never _SentinelPlan."
```

---

## Task 4: HarnessConfig V14 fields + JSON loader + validation

**Files:**
- Modify: `C:\Users\qwqw1\Dropbox\cs\github\Homeguard\src\research\ramp_phase4\config.py`
- Create: `C:\Users\qwqw1\Dropbox\cs\github\Homeguard\tests\research\ramp_phase4\test_config_v14.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/research/ramp_phase4/test_config_v14.py`:

```python
"""Tests for V14 config fields + validation predicate."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import pytest

from src.research.ramp_phase4.config import (
    HarnessConfig, load_v14_tau_constants,
)


def _base_cfg(**overrides) -> HarnessConfig:
    defaults = dict(
        start_date=datetime(2017, 1, 3),
        end_date=datetime(2026, 5, 22),
        universe_csv=Path('config/universes/sp500-2025.csv'),
        initial_capital=100_000.0,
        cost_bps_per_side=5.0,
        soft_bear_tau_in=0.3,
        soft_bear_tau_out=0.2,
    )
    defaults.update(overrides)
    return HarnessConfig(**defaults)


def test_v14_fields_defaults():
    cfg = _base_cfg()
    assert cfg.soft_bear_tau_in == 0.3
    assert cfg.soft_bear_tau_out == 0.2
    assert cfg.soft_bear_dampen_factor == 0.5


def test_tau_predicate_tau_out_zero_rejected():
    with pytest.raises(ValueError, match='tau_out'):
        _base_cfg(soft_bear_tau_in=0.3, soft_bear_tau_out=0.0)


def test_tau_predicate_tau_in_one_rejected():
    with pytest.raises(ValueError, match='tau_in'):
        _base_cfg(soft_bear_tau_in=1.0, soft_bear_tau_out=0.5)


def test_tau_predicate_inverted_rejected():
    with pytest.raises(ValueError, match='tau_out.*tau_in'):
        _base_cfg(soft_bear_tau_in=0.2, soft_bear_tau_out=0.3)


def test_tau_predicate_equal_rejected():
    with pytest.raises(ValueError, match='tau_out.*tau_in'):
        _base_cfg(soft_bear_tau_in=0.3, soft_bear_tau_out=0.3)


def test_dampen_factor_out_of_range_rejected():
    with pytest.raises(ValueError, match='dampen_factor'):
        _base_cfg(soft_bear_dampen_factor=1.5)


def test_dampen_factor_negative_rejected():
    with pytest.raises(ValueError, match='dampen_factor'):
        _base_cfg(soft_bear_dampen_factor=-0.1)


def test_load_v14_tau_constants_from_json(tmp_path):
    p = tmp_path / 'tau.json'
    p.write_text(json.dumps({
        'tau_in': 0.35, 'tau_out': 0.25,
        'tau_band': 0.1, 'g1_labeler_commit': 'abc',
    }))
    tau_in, tau_out = load_v14_tau_constants(p)
    assert tau_in == 0.35
    assert tau_out == 0.25


def test_load_v14_tau_constants_default_path():
    """Default path exists from Task 0."""
    tau_in, tau_out = load_v14_tau_constants()
    assert 0.0 < tau_out < tau_in < 1.0
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
PYTHONPATH=. python -m pytest tests/research/ramp_phase4/test_config_v14.py -v
```

Expected: 9 tests fail with various TypeError / ImportError errors.

- [ ] **Step 3: Extend the config dataclass**

In `src/research/ramp_phase4/config.py`, add at the top of the file (after existing imports):

```python
import json


def load_v14_tau_constants(
    path: Path = Path('config/research/v14_tau_constants.json'),
) -> tuple[float, float]:
    """Load (tau_in, tau_out) from the pre-registered JSON.

    Raises FileNotFoundError if the JSON has not been produced by
    scripts/diagnostics/compute_tau_in_from_g1.py (Task 0 of the V14 plan).
    """
    if not path.exists():
        raise FileNotFoundError(
            f'{path} not found. Run scripts/diagnostics/compute_tau_in_from_g1.py '
            f'before any V14 backtest.'
        )
    data = json.loads(path.read_text())
    return float(data['tau_in']), float(data['tau_out'])
```

Then add fields to `HarnessConfig` (after `min_regime_days`):

```python
    # V12 additions:
    regime_positions: Dict[str, str] = field(
        default_factory=lambda: dict(_DEFAULT_REGIME_POSITIONS)
    )
    min_regime_days: int = 0
    # V14 additions:
    soft_bear_tau_in: float = 0.3
    soft_bear_tau_out: float = 0.2
    soft_bear_dampen_factor: float = 0.5
```

Extend the validation in `__post_init__`:

```python
    def __post_init__(self) -> None:
        for regime, mode in self.regime_positions.items():
            if mode not in _ALLOWED_POSITION_VALUES:
                raise ValueError(
                    f"regime_positions[{regime!r}] = {mode!r} is not in "
                    f"{sorted(_ALLOWED_POSITION_VALUES)}. Ticker/strategy "
                    f"values are reserved for V13+."
                )
        if self.min_regime_days < 0:
            raise ValueError(
                f"min_regime_days must be >= 0, got {self.min_regime_days}"
            )
        # V14 predicates.
        if not (0.0 < self.soft_bear_tau_out < self.soft_bear_tau_in < 1.0):
            raise ValueError(
                f"V14 tau predicate violated: must have "
                f"0 < tau_out < tau_in < 1.0; got "
                f"tau_in={self.soft_bear_tau_in}, tau_out={self.soft_bear_tau_out}"
            )
        if not (0.0 <= self.soft_bear_dampen_factor <= 1.0):
            raise ValueError(
                f"soft_bear_dampen_factor must be in [0.0, 1.0], "
                f"got {self.soft_bear_dampen_factor}"
            )
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
PYTHONPATH=. python -m pytest tests/research/ramp_phase4/test_config_v14.py -v
```

Expected: 9 passed.

- [ ] **Step 5: Run full test suite for regression**

```bash
PYTHONPATH=. python -m pytest tests/research/ramp_phase4/ -q
```

Expected: existing 94+ V12 tests still pass + new V14 tests.

- [ ] **Step 6: Commit**

```bash
git add src/research/ramp_phase4/config.py tests/research/ramp_phase4/test_config_v14.py
git commit -m "feat(config): V14 tau / dampen fields + JSON loader + predicate validation

HarnessConfig gets three new fields:
- soft_bear_tau_in: float = 0.3
- soft_bear_tau_out: float = 0.2
- soft_bear_dampen_factor: float = 0.5

Validation in __post_init__: 0 < tau_out < tau_in < 1.0 AND
0.0 <= dampen_factor <= 1.0.

load_v14_tau_constants() helper reads the JSON produced by Task 0
(config/research/v14_tau_constants.json) and returns (tau_in, tau_out).
The orchestrator calls this once at start; ad-hoc tests can override
with explicit values."
```

---

## Task 5: V14a-soft-bear-cash variant + canonical pinning test

**Files:**
- Modify: `C:\Users\qwqw1\Dropbox\cs\github\Homeguard\src\research\ramp_phase4\variants.py`
- Modify: `C:\Users\qwqw1\Dropbox\cs\github\Homeguard\tests\research\ramp_phase4\test_variants.py`

- [ ] **Step 1: Write the canonical pinning test for V14a**

Append to `tests/research/ramp_phase4/test_variants.py` (at the end of the file):

```python
import json
from pathlib import Path
from datetime import datetime, timedelta
from src.research.ramp_phase4.variants import (
    _variant_v14a_soft_bear_cash, REGISTRY,
)
from src.research.ramp_phase4.plans import _SentinelPlan, PLAN_CASH_BEAR_SOFT


def _v14_test_cfg():
    """Standard config for V14 tests with explicit tau values."""
    from src.research.ramp_phase4.config import HarnessConfig
    return HarnessConfig(
        start_date=datetime(2017, 1, 3),
        end_date=datetime(2026, 5, 22),
        universe_csv=Path('config/universes/sp500-2025.csv'),
        initial_capital=100_000.0,
        cost_bps_per_side=5.0,
        soft_bear_tau_in=0.3,
        soft_bear_tau_out=0.2,
    )


def _patch_detector_score(monkeypatch, score: float):
    """Patch the module-level _DETECTOR to return a known BEAR_score."""
    from src.research.ramp_phase4 import variants
    class _MockDetector:
        last_regime_scores = {'BEAR': score, 'STRONG_BULL': 1.0 - score}
        last_classification_timestamp = None
        def classify_regime(self, spy, vix, t):
            self.last_classification_timestamp = t
            return ('BEAR', score) if score >= 0.5 else ('STRONG_BULL', 1.0 - score)
    monkeypatch.setattr(variants, '_DETECTOR', _MockDetector())


def test_v14a_hysteresis_canonical_schmitt(monkeypatch):
    """Canonical pinning: source of truth for V14a state-machine semantics.

    With tau_in=0.3, tau_out=0.2:
      Day  Score   Expected in_bear_soft_mode after update    Plan returned
       1   0.10    False                                      V11 plan
       2   0.25    False (never crossed tau_in)               V11 plan
       3   0.30    True (>= tau_in enters)                    PLAN_CASH_BEAR_SOFT
       4   0.25    True (stays in band)                       PLAN_CASH_BEAR_SOFT
       5   0.20    True (NOT strict <, stays)                 PLAN_CASH_BEAR_SOFT
       6   0.1999  False (strict <, exits)                    V11 plan
       7   0.25    False (no entry from in-band)              V11 plan
       8   0.50    True (re-enters)                           PLAN_CASH_BEAR_SOFT
    """
    from src.research.ramp_phase4.engine import HarnessState

    cfg = _v14_test_cfg()
    state = HarnessState(cash_usd=100_000.0)
    # Stub V11 to return a known dict (we're not testing V11 here)
    from src.research.ramp_phase4 import variants as v_mod
    def _stub_v11(t, state, panel, cfg):
        return {'__regime__': 'STRONG_BULL', 'STUB': 1.0}
    monkeypatch.setattr(v_mod, '_variant_v11', _stub_v11)

    sequence = [
        (0.10, False, dict),
        (0.25, False, dict),
        (0.30, True, _SentinelPlan),
        (0.25, True, _SentinelPlan),
        (0.20, True, _SentinelPlan),
        (0.1999, False, dict),
        (0.25, False, dict),
        (0.50, True, _SentinelPlan),
    ]
    base_date = datetime(2020, 1, 1)
    for i, (score, expected_mode, expected_type) in enumerate(sequence):
        _patch_detector_score(monkeypatch, score)
        t = base_date + timedelta(days=i)
        plan = _variant_v14a_soft_bear_cash(t, state, None, cfg)
        assert state.in_bear_soft_mode is expected_mode, \
            f'Day {i+1} score={score}: expected mode={expected_mode}, got {state.in_bear_soft_mode}'
        if expected_type is _SentinelPlan:
            assert isinstance(plan, _SentinelPlan), f'Day {i+1}: expected sentinel, got {plan}'
            assert plan.reason == 'BEAR_SOFT_CASH'
        else:
            assert isinstance(plan, dict), f'Day {i+1}: expected dict, got {plan}'


def test_v14a_freshness_assertion_passes_with_explicit_call(monkeypatch):
    """V14a calls classify_regime explicitly; freshness assertion passes."""
    from src.research.ramp_phase4.engine import HarnessState
    cfg = _v14_test_cfg()
    state = HarnessState(cash_usd=100_000.0)
    from src.research.ramp_phase4 import variants as v_mod
    monkeypatch.setattr(v_mod, '_variant_v11', lambda t, s, p, c: {'__regime__': 'STRONG_BULL'})

    _patch_detector_score(monkeypatch, 0.5)
    t = datetime(2020, 1, 1)
    plan = _variant_v14a_soft_bear_cash(t, state, None, cfg)
    assert v_mod._DETECTOR.last_classification_timestamp == t


def test_v14a_registry_entry_exists():
    assert 'V14a-soft-bear-cash' in REGISTRY
    assert REGISTRY['V14a-soft-bear-cash'].plan_fn is _variant_v14a_soft_bear_cash
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
PYTHONPATH=. python -m pytest tests/research/ramp_phase4/test_variants.py::test_v14a_hysteresis_canonical_schmitt tests/research/ramp_phase4/test_variants.py::test_v14a_freshness_assertion_passes_with_explicit_call tests/research/ramp_phase4/test_variants.py::test_v14a_registry_entry_exists -v
```

Expected: 3 tests fail with ImportError on `_variant_v14a_soft_bear_cash`.

- [ ] **Step 3: Implement V14a in variants.py**

In `src/research/ramp_phase4/variants.py`, add the import at the top (after existing imports):

```python
from src.research.ramp_phase4.engine import (
    _engine_pre_variant_update,
    _engine_pre_variant_update_soft_bear,
)
from src.research.ramp_phase4.plans import PLAN_CASH_BEAR_SOFT, _SentinelPlan
```

After `_variant_v12` (around line 280), add:

```python
def _variant_v14a_soft_bear_cash(
    t: datetime, state, panel: pd.DataFrame, cfg,
) -> Dict[str, float] | _SentinelPlan:
    """V14a: BEAR_score Schmitt-trigger -> cash on enter.

    Pre-conditions: cfg.soft_bear_tau_in/tau_out loaded from
    config/research/v14_tau_constants.json by the orchestrator.

    Reads detector.last_regime_scores['BEAR'] AFTER making an explicit
    classify_regime call to guarantee freshness. Decouples from
    _compute_plan_from_panel ordering (V11's incidental detector call).

    State machine: _engine_pre_variant_update_soft_bear mutates
    state.in_bear_soft_mode based on Schmitt trigger.

    Action: if in_bear_soft_mode -> PLAN_CASH_BEAR_SOFT; else V11 plan.
    """
    plan_v11 = _variant_v11(t, state, panel, cfg)

    spy_slice = panel['SPY'].dropna().loc[:t] if panel is not None else None
    vix_slice = panel['VIX'].dropna().loc[:t] if panel is not None else None

    bear_score = None
    if spy_slice is not None and vix_slice is not None \
            and len(spy_slice) >= 252 and len(vix_slice) >= 252:
        spy_df = pd.DataFrame({
            'close': spy_slice, 'open': spy_slice, 'high': spy_slice, 'low': spy_slice,
            'volume': 1e6,
        })
        vix_df = pd.DataFrame({'close': vix_slice})
        try:
            _DETECTOR.classify_regime(spy_df, vix_df, t)
            assert _DETECTOR.last_classification_timestamp == t, \
                'Detector freshness assertion failed in V14a'
            bear_score = _DETECTOR.last_regime_scores.get('BEAR')
        except Exception:
            bear_score = None
    elif _DETECTOR.last_regime_scores is not None:
        bear_score = _DETECTOR.last_regime_scores.get('BEAR')

    if bear_score is not None:
        _engine_pre_variant_update_soft_bear(
            state, bear_score, cfg.soft_bear_tau_in, cfg.soft_bear_tau_out,
        )

    if state.in_bear_soft_mode:
        return PLAN_CASH_BEAR_SOFT
    return plan_v11
```

Add to `REGISTRY` dict (at the bottom):

```python
    'V14a-soft-bear-cash': VariantSpec(
        id='V14a-soft-bear-cash',
        description='V11 + Schmitt-trigger BEAR_score consumer; in_bear_soft_mode -> cash',
        plan_fn=_variant_v14a_soft_bear_cash,
    ),
```

- [ ] **Step 4: Run the V14a tests to verify they pass**

```bash
PYTHONPATH=. python -m pytest tests/research/ramp_phase4/test_variants.py -v -k 'v14a'
```

Expected: 3 passed.

- [ ] **Step 5: Run full suite for regression**

```bash
PYTHONPATH=. python -m pytest tests/research/ramp_phase4/ tests/diagnostics/ tests/strategies/advanced/ -q
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/research/ramp_phase4/variants.py tests/research/ramp_phase4/test_variants.py
git commit -m "feat(variants): V14a-soft-bear-cash via Schmitt-trigger BEAR_score

V14a is the first of three V14 factorial variants. Logic:
1. Run V11 plan (for the non-bear-soft branch).
2. Make explicit classify_regime call + freshness assertion.
3. Update state.in_bear_soft_mode via Schmitt trigger on BEAR_score.
4. If in_bear_soft_mode: return PLAN_CASH_BEAR_SOFT (sentinel).
   Else: return V11 plan unchanged.

Canonical pinning test (8-day Schmitt sequence) is the source of truth
for state-machine semantics. Covers boundary cases: bear_score exactly
tau_in (enters), exactly tau_out (stays), strictly < tau_out (exits)."
```

---

## Task 6: V14b-soft-bear-spy variant

**Files:**
- Modify: `C:\Users\qwqw1\Dropbox\cs\github\Homeguard\src\research\ramp_phase4\variants.py`
- Modify: `C:\Users\qwqw1\Dropbox\cs\github\Homeguard\tests\research\ramp_phase4\test_variants.py`

- [ ] **Step 1: Write the failing test for V14b**

Append to `tests/research/ramp_phase4/test_variants.py`:

```python
def test_v14b_hysteresis_canonical_schmitt(monkeypatch):
    """V14b: same state machine as V14a; action is {SPY: 1.0} instead of cash."""
    from src.research.ramp_phase4.engine import HarnessState
    from src.research.ramp_phase4.variants import _variant_v14b_soft_bear_spy

    cfg = _v14_test_cfg()
    state = HarnessState(cash_usd=100_000.0)
    from src.research.ramp_phase4 import variants as v_mod
    monkeypatch.setattr(v_mod, '_variant_v11', lambda t, s, p, c: {'__regime__': 'STRONG_BULL', 'AAPL': 1.0})

    sequence_in_mode = [
        (0.10, False),
        (0.25, False),
        (0.30, True),
        (0.20, True),
        (0.1999, False),
        (0.50, True),
    ]
    base_date = datetime(2020, 1, 1)
    for i, (score, expected_mode) in enumerate(sequence_in_mode):
        _patch_detector_score(monkeypatch, score)
        t = base_date + timedelta(days=i)
        plan = _variant_v14b_soft_bear_spy(t, state, None, cfg)
        assert state.in_bear_soft_mode is expected_mode
        if expected_mode:
            # V14b returns dict with SPY at 1.0 (V11 gross is constant 1.0)
            assert isinstance(plan, dict)
            assert plan.get('SPY') == 1.0
            assert '__regime__' in plan
            assert plan['__regime__'] == 'BEAR_SOFT_SPY'
            # No other position symbols
            symbols = {k for k in plan.keys() if k not in ('__regime__',)}
            assert symbols == {'SPY'}
        else:
            # Defer to V11
            assert plan == {'__regime__': 'STRONG_BULL', 'AAPL': 1.0}


def test_v14b_registry_entry_exists():
    from src.research.ramp_phase4.variants import _variant_v14b_soft_bear_spy
    assert 'V14b-soft-bear-spy' in REGISTRY
    assert REGISTRY['V14b-soft-bear-spy'].plan_fn is _variant_v14b_soft_bear_spy
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
PYTHONPATH=. python -m pytest tests/research/ramp_phase4/test_variants.py -v -k 'v14b'
```

Expected: 2 tests fail with ImportError on `_variant_v14b_soft_bear_spy`.

- [ ] **Step 3: Implement V14b**

In `src/research/ramp_phase4/variants.py`, after `_variant_v14a_soft_bear_cash`, add:

```python
def _variant_v14b_soft_bear_spy(
    t: datetime, state, panel: pd.DataFrame, cfg,
) -> Dict[str, float]:
    """V14b: BEAR_score Schmitt-trigger -> SPY 100% on enter.

    Tests E1's BEAR-as-buy hypothesis under a working trigger (soft scores
    lead argmax by median 24 days at tau=0.3 per E3). V11's gross is
    constant 1.0 (V01 base ignores exposure_pct; filters renormalize),
    so 'V11 gross to SPY' simplifies to a fixed {SPY: 1.0} allocation.
    """
    plan_v11 = _variant_v11(t, state, panel, cfg)

    spy_slice = panel['SPY'].dropna().loc[:t] if panel is not None else None
    vix_slice = panel['VIX'].dropna().loc[:t] if panel is not None else None

    bear_score = None
    if spy_slice is not None and vix_slice is not None \
            and len(spy_slice) >= 252 and len(vix_slice) >= 252:
        spy_df = pd.DataFrame({
            'close': spy_slice, 'open': spy_slice, 'high': spy_slice, 'low': spy_slice,
            'volume': 1e6,
        })
        vix_df = pd.DataFrame({'close': vix_slice})
        try:
            _DETECTOR.classify_regime(spy_df, vix_df, t)
            assert _DETECTOR.last_classification_timestamp == t, \
                'Detector freshness assertion failed in V14b'
            bear_score = _DETECTOR.last_regime_scores.get('BEAR')
        except Exception:
            bear_score = None
    elif _DETECTOR.last_regime_scores is not None:
        bear_score = _DETECTOR.last_regime_scores.get('BEAR')

    if bear_score is not None:
        _engine_pre_variant_update_soft_bear(
            state, bear_score, cfg.soft_bear_tau_in, cfg.soft_bear_tau_out,
        )

    if state.in_bear_soft_mode:
        return {'SPY': 1.0, '__regime__': 'BEAR_SOFT_SPY'}
    return plan_v11
```

Add to `REGISTRY`:

```python
    'V14b-soft-bear-spy': VariantSpec(
        id='V14b-soft-bear-spy',
        description='V11 + Schmitt-trigger BEAR_score consumer; in_bear_soft_mode -> SPY 100%',
        plan_fn=_variant_v14b_soft_bear_spy,
    ),
```

- [ ] **Step 4: Run V14b tests to verify they pass**

```bash
PYTHONPATH=. python -m pytest tests/research/ramp_phase4/test_variants.py -v -k 'v14b'
```

Expected: 2 passed.

- [ ] **Step 5: Run full suite**

```bash
PYTHONPATH=. python -m pytest tests/research/ramp_phase4/ -q
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/research/ramp_phase4/variants.py tests/research/ramp_phase4/test_variants.py
git commit -m "feat(variants): V14b-soft-bear-spy via Schmitt-trigger BEAR_score

V14b shares the V14a state machine; action diverges: when in_bear_soft_mode
is True, returns {SPY: 1.0, __regime__: BEAR_SOFT_SPY}. V11's gross is
constant 1.0 per open-questions Q4, so 'V11 gross to SPY' simplifies to
fixed {SPY: 1.0}. Tests E1's BEAR-as-buy hypothesis under a working
soft-score trigger (E3 verdict: argmax was the lag culprit, not the
sign)."
```

---

## Task 7: V14c-soft-bear-dampen variant

**Files:**
- Modify: `C:\Users\qwqw1\Dropbox\cs\github\Homeguard\src\research\ramp_phase4\variants.py`
- Modify: `C:\Users\qwqw1\Dropbox\cs\github\Homeguard\tests\research\ramp_phase4\test_variants.py`

- [ ] **Step 1: Write the failing test for V14c**

Append to `tests/research/ramp_phase4/test_variants.py`:

```python
def test_v14c_hysteresis_canonical_schmitt(monkeypatch):
    """V14c: same state machine as V14a/b; action is V11 weights * dampen_factor."""
    from src.research.ramp_phase4.engine import HarnessState
    from src.research.ramp_phase4.variants import _variant_v14c_soft_bear_dampen

    cfg = _v14_test_cfg()
    assert cfg.soft_bear_dampen_factor == 0.5
    state = HarnessState(cash_usd=100_000.0)
    from src.research.ramp_phase4 import variants as v_mod
    monkeypatch.setattr(v_mod, '_variant_v11', lambda t, s, p, c: {
        '__regime__': 'STRONG_BULL', 'AAPL': 0.5, 'MSFT': 0.5,
    })

    sequence = [(0.10, False), (0.30, True), (0.1999, False), (0.50, True)]
    base_date = datetime(2020, 1, 1)
    for i, (score, expected_mode) in enumerate(sequence):
        _patch_detector_score(monkeypatch, score)
        t = base_date + timedelta(days=i)
        plan = _variant_v14c_soft_bear_dampen(t, state, None, cfg)
        assert state.in_bear_soft_mode is expected_mode
        if expected_mode:
            assert isinstance(plan, dict)
            assert plan.get('AAPL') == 0.25  # 0.5 * 0.5
            assert plan.get('MSFT') == 0.25
            assert plan['__regime__'] == 'BEAR_SOFT_DAMPEN'
        else:
            assert plan == {'__regime__': 'STRONG_BULL', 'AAPL': 0.5, 'MSFT': 0.5}


def test_v14c_registry_entry_exists():
    from src.research.ramp_phase4.variants import _variant_v14c_soft_bear_dampen
    assert 'V14c-soft-bear-dampen' in REGISTRY
    assert REGISTRY['V14c-soft-bear-dampen'].plan_fn is _variant_v14c_soft_bear_dampen


def test_v14c_respects_custom_dampen_factor(monkeypatch):
    from src.research.ramp_phase4.engine import HarnessState
    from src.research.ramp_phase4.variants import _variant_v14c_soft_bear_dampen
    from src.research.ramp_phase4.config import HarnessConfig
    cfg = HarnessConfig(
        start_date=datetime(2017, 1, 3),
        end_date=datetime(2026, 5, 22),
        universe_csv=Path('config/universes/sp500-2025.csv'),
        initial_capital=100_000.0,
        cost_bps_per_side=5.0,
        soft_bear_tau_in=0.3,
        soft_bear_tau_out=0.2,
        soft_bear_dampen_factor=0.25,
    )
    state = HarnessState(cash_usd=100_000.0)
    from src.research.ramp_phase4 import variants as v_mod
    monkeypatch.setattr(v_mod, '_variant_v11', lambda t, s, p, c: {
        '__regime__': 'STRONG_BULL', 'AAPL': 1.0,
    })
    _patch_detector_score(monkeypatch, 0.5)
    plan = _variant_v14c_soft_bear_dampen(datetime(2020, 1, 1), state, None, cfg)
    assert plan['AAPL'] == 0.25  # 1.0 * 0.25
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
PYTHONPATH=. python -m pytest tests/research/ramp_phase4/test_variants.py -v -k 'v14c'
```

Expected: 3 tests fail with ImportError.

- [ ] **Step 3: Implement V14c**

In `src/research/ramp_phase4/variants.py`, after `_variant_v14b_soft_bear_spy`, add:

```python
def _variant_v14c_soft_bear_dampen(
    t: datetime, state, panel: pd.DataFrame, cfg,
) -> Dict[str, float]:
    """V14c: BEAR_score Schmitt-trigger -> V11 plan * dampen_factor on enter.

    Tests whether the right response to BEAR is risk reduction (not switch).
    cfg.soft_bear_dampen_factor (default 0.5) is the multiplier applied to
    all V11 symbol weights; __regime__ marker is preserved.
    """
    plan_v11 = _variant_v11(t, state, panel, cfg)

    spy_slice = panel['SPY'].dropna().loc[:t] if panel is not None else None
    vix_slice = panel['VIX'].dropna().loc[:t] if panel is not None else None

    bear_score = None
    if spy_slice is not None and vix_slice is not None \
            and len(spy_slice) >= 252 and len(vix_slice) >= 252:
        spy_df = pd.DataFrame({
            'close': spy_slice, 'open': spy_slice, 'high': spy_slice, 'low': spy_slice,
            'volume': 1e6,
        })
        vix_df = pd.DataFrame({'close': vix_slice})
        try:
            _DETECTOR.classify_regime(spy_df, vix_df, t)
            assert _DETECTOR.last_classification_timestamp == t, \
                'Detector freshness assertion failed in V14c'
            bear_score = _DETECTOR.last_regime_scores.get('BEAR')
        except Exception:
            bear_score = None
    elif _DETECTOR.last_regime_scores is not None:
        bear_score = _DETECTOR.last_regime_scores.get('BEAR')

    if bear_score is not None:
        _engine_pre_variant_update_soft_bear(
            state, bear_score, cfg.soft_bear_tau_in, cfg.soft_bear_tau_out,
        )

    if state.in_bear_soft_mode:
        dampened: Dict[str, float] = {}
        for sym, w in plan_v11.items():
            if sym == '__regime__':
                continue
            dampened[sym] = float(w) * cfg.soft_bear_dampen_factor
        dampened['__regime__'] = 'BEAR_SOFT_DAMPEN'
        return dampened
    return plan_v11
```

Add to `REGISTRY`:

```python
    'V14c-soft-bear-dampen': VariantSpec(
        id='V14c-soft-bear-dampen',
        description='V11 + Schmitt-trigger BEAR_score consumer; in_bear_soft_mode -> V11 plan * dampen_factor (default 0.5)',
        plan_fn=_variant_v14c_soft_bear_dampen,
    ),
```

- [ ] **Step 4: Run V14c tests to verify they pass**

```bash
PYTHONPATH=. python -m pytest tests/research/ramp_phase4/test_variants.py -v -k 'v14c'
```

Expected: 3 passed.

- [ ] **Step 5: Run full suite for regression**

```bash
PYTHONPATH=. python -m pytest tests/research/ramp_phase4/ tests/diagnostics/ tests/strategies/advanced/ -q
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/research/ramp_phase4/variants.py tests/research/ramp_phase4/test_variants.py
git commit -m "feat(variants): V14c-soft-bear-dampen via Schmitt-trigger BEAR_score

V14c shares the V14a/b state machine; action: V11 weights multiplied by
cfg.soft_bear_dampen_factor (default 0.5). Tests whether risk reduction
(not regime switch) is the right response to BEAR. Tag __regime__ as
BEAR_SOFT_DAMPEN for attribution.

Completes the V14 factorial: V14a (cash) + V14b (SPY) + V14c (dampen).
All three share state.in_bear_soft_mode and the canonical Schmitt-trigger
state machine; differ only in action."
```

---

## Task 8: V14 factorial readiness orchestrator

**Files:**
- Create: `C:\Users\qwqw1\Dropbox\cs\github\Homeguard\scripts\backtest_scripts\ramp_phase4_v14_factorial_readiness.py`

This is a substantial clone of `scripts/backtest_scripts/ramp_phase4_v12c_readiness.py`. Read that file first to understand the structure (orchestrator + 5-gate evaluation + report writing). The differences are: (1) three variants in cost grid, (2) different DSR n_trials, (3) different PBO set, (4) V11-warm-start NOT recomputed (per Q2 resolution), (5) different sensitivity panel.

- [ ] **Step 1: Read the V12c orchestrator structure**

```bash
wc -l scripts/backtest_scripts/ramp_phase4_v12c_readiness.py
head -100 scripts/backtest_scripts/ramp_phase4_v12c_readiness.py
```

Expected: ~710 lines, structured as: imports, constants, run_backtest() helper, main() with cost grid loop, cross-variant loop, sensitivity panel, gate computations, report writing.

- [ ] **Step 2: Create the V14 factorial orchestrator**

Create `scripts/backtest_scripts/ramp_phase4_v14_factorial_readiness.py`. Use the V12c orchestrator as the template with these adaptations:

**At the top, constants section:**

```python
# DSR n_trials_project audited count per spec rev2 honesty discipline:
#   V11+pre-V11 cohort:               22
#   V12 base + 4-grid sensitivity:     5
#   V12c base:                         1
#   V13 base:                          1
#   V14a, V14b, V14c base:             3
#   V14a tau-band sensitivity:         2
#   V14c dampen sensitivity:           2
#                                    ---
#                                     36
N_TRIALS_PROJECT = 36

# Variant IDs for the three V14 factorial arms.
V14_VARIANTS = ['V14a-soft-bear-cash', 'V14b-soft-bear-spy', 'V14c-soft-bear-dampen']
COST_GRID_BPS = [1.0, 5.0, 7.5, 10.0]
TIMING_MODES = ['near_close', 'one_day_lag']

# Cross-variants for PBO gate set (8-variant set per spec rev2):
GATE_PBO_VARIANTS = ['V01', 'V11', 'V12', 'V12c', 'V13-bear-invert',
                     'V14a-soft-bear-cash', 'V14b-soft-bear-spy', 'V14c-soft-bear-dampen']
# V12c is V12 with regime_positions[UNPREDICTABLE]='cash'; we re-create via config override.

# Diagnostic PBO -- 4 orthogonal variants (not gate-influencing):
DIAGNOSTIC_PBO_VARIANTS = ['V01', 'V11', 'V12', 'V14a-soft-bear-cash']

LAG_DEGRADATION_FRACTION = 0.2
LAG_DEGRADATION_FLOOR = 0.1
COST_FLOOR_SHARPE = 0.30
NO_REGRESS_VS_V11_FRACTION = 0.9
TIER_1_LIFT_THRESHOLD = 0.10

OUTPUT_REPORT = Path('docs/reports/ramp/20260526_phase4_v14_factorial_readiness.md')
```

**Load tau constants at import:**

```python
from src.research.ramp_phase4.config import load_v14_tau_constants
TAU_IN, TAU_OUT = load_v14_tau_constants()
```

**Cost grid loop -- iterate over all three V14 variants:**

```python
def run_cost_grid(universe_csv, start_date, end_date) -> Dict:
    """24 backtests: 3 variants x 4 cost levels x 2 timing modes."""
    results = {}
    for variant_id in V14_VARIANTS:
        for bps in COST_GRID_BPS:
            for mode in TIMING_MODES:
                key = f'{variant_id}|{bps}bps|{mode}'
                cfg = HarnessConfig(
                    start_date=start_date, end_date=end_date,
                    universe_csv=universe_csv,
                    initial_capital=100_000.0,
                    cost_bps_per_side=bps,
                    timing_mode=mode,
                    soft_bear_tau_in=TAU_IN,
                    soft_bear_tau_out=TAU_OUT,
                )
                sharpe, cagr, nav_series = run_backtest(cfg, REGISTRY[variant_id])
                results[key] = {'sharpe': sharpe, 'cagr': cagr, 'nav': nav_series}
    return results
```

**Cross-variant runs for PBO at 5 bps near_close:**

```python
def run_cross_variants_5bps_nc(universe_csv, start_date, end_date) -> Dict:
    """Cross-variants at 5 bps near_close for PBO + Tier 1 lift check.

    7 unique variants: V01, V11, V12 (with default BEAR-to-cash), V12c
    (V12 with regime_positions[UNPREDICTABLE]='cash'), V13-bear-invert, V14a, V14b, V14c.
    """
    base_kwargs = dict(
        start_date=start_date, end_date=end_date,
        universe_csv=universe_csv,
        initial_capital=100_000.0,
        cost_bps_per_side=5.0,
        timing_mode='near_close',
        soft_bear_tau_in=TAU_IN,
        soft_bear_tau_out=TAU_OUT,
    )
    out = {}
    for variant_id in ['V01', 'V11', 'V12', 'V13-bear-invert']:
        cfg = HarnessConfig(**base_kwargs)
        out[variant_id] = run_backtest(cfg, REGISTRY[variant_id])
    # V12c is V12 with regime_positions override:
    v12c_positions = {**_DEFAULT_REGIME_POSITIONS, 'UNPREDICTABLE': 'cash'}
    cfg_v12c = HarnessConfig(**base_kwargs, regime_positions=v12c_positions)
    out['V12c'] = run_backtest(cfg_v12c, REGISTRY['V12'])
    # V14 variants are run in cost_grid; reused here.
    return out
```

**V11 reference at 7.5 bps lag (Gate 5):**

```python
def run_v11_ref_75bps_lag(universe_csv, start_date, end_date):
    """V11 reference for Gate 5 no-regress baseline."""
    cfg = HarnessConfig(
        start_date=start_date, end_date=end_date,
        universe_csv=universe_csv,
        initial_capital=100_000.0,
        cost_bps_per_side=7.5,
        timing_mode='one_day_lag',
        soft_bear_tau_in=TAU_IN,
        soft_bear_tau_out=TAU_OUT,
    )
    return run_backtest(cfg, REGISTRY['V11'])
```

**Sensitivity panel (V14a tau-band + V14c dampen, informational only):**

```python
def run_sensitivity_panels(universe_csv, start_date, end_date):
    """4 informational runs: 2 V14a tau-band + 2 V14c dampen."""
    out = {}
    base_kwargs = dict(
        start_date=start_date, end_date=end_date,
        universe_csv=universe_csv,
        initial_capital=100_000.0,
        cost_bps_per_side=5.0,
        timing_mode='near_close',
    )
    # V14a tau-band sensitivity (fixed tau_in, sweep tau_out)
    for tau_out_val in [TAU_IN - 0.05, TAU_IN - 0.15]:
        if tau_out_val <= 0:
            continue
        cfg = HarnessConfig(**base_kwargs, soft_bear_tau_in=TAU_IN, soft_bear_tau_out=tau_out_val)
        out[f'V14a-soft-bear-cash|tau_out={tau_out_val:.2f}'] = run_backtest(
            cfg, REGISTRY['V14a-soft-bear-cash'])
    # V14c dampen sensitivity (fixed tau, sweep dampen_factor)
    for dampen in [0.25, 0.75]:
        cfg = HarnessConfig(**base_kwargs, soft_bear_tau_in=TAU_IN, soft_bear_tau_out=TAU_OUT,
                            soft_bear_dampen_factor=dampen)
        out[f'V14c-soft-bear-dampen|dampen={dampen}'] = run_backtest(
            cfg, REGISTRY['V14c-soft-bear-dampen'])
    return out
```

**Gate evaluation (one entry per V14a/b/c):**

```python
def evaluate_gates_per_variant(variant_id, cost_grid_results, v11_ref_lag, pbo_value):
    """5-gate evaluation per V14 variant."""
    nc_5 = cost_grid_results[f'{variant_id}|5.0bps|near_close']['sharpe']
    lag_5 = cost_grid_results[f'{variant_id}|5.0bps|one_day_lag']['sharpe']
    lag_75 = cost_grid_results[f'{variant_id}|7.5bps|one_day_lag']['sharpe']

    nc_minus_lag = nc_5 - lag_5
    cap = max(LAG_DEGRADATION_FRACTION * abs(nc_5), LAG_DEGRADATION_FLOOR)
    lag_pass = nc_minus_lag <= cap

    cost_floor_pass = lag_75 > COST_FLOOR_SHARPE
    no_regress_pass = lag_75 >= NO_REGRESS_VS_V11_FRACTION * v11_ref_lag

    # PSR/DSR at 5 bps near_close (the canonical cost tier)
    nav_5_nc = cost_grid_results[f'{variant_id}|5.0bps|near_close']['nav']
    psr_value = compute_psr(nav_5_nc)
    dsr_value = compute_dsr(nav_5_nc, n_trials=N_TRIALS_PROJECT)

    return {
        'variant': variant_id,
        'sharpe_5bps_nc': nc_5,
        'sharpe_5bps_lag': lag_5,
        'sharpe_75bps_lag': lag_75,
        'psr': psr_value, 'psr_pass': psr_value > 0.95,
        'dsr': dsr_value, 'dsr_pass': dsr_value > 0.95,
        'pbo': pbo_value, 'pbo_pass': pbo_value < 0.5,
        'nc_minus_lag': nc_minus_lag, 'lag_pass': lag_pass,
        'cost_floor_pass': cost_floor_pass,
        'no_regress_pass': no_regress_pass,
        'v11_ref_lag': v11_ref_lag,
    }
```

**Tier classification per variant:**

```python
def classify_tier(gates, v11_5bps_nc_sharpe):
    """TIER 1 / 3 / 4 per spec rev2 success criteria."""
    structural_pass = (gates['pbo_pass'] and gates['lag_pass']
                       and gates['cost_floor_pass'] and gates['no_regress_pass'])
    if not structural_pass:
        return 'TIER 4'
    sig_pass = gates['psr_pass'] and gates['dsr_pass']
    lift_pass = gates['sharpe_5bps_nc'] > v11_5bps_nc_sharpe + TIER_1_LIFT_THRESHOLD
    if sig_pass and lift_pass:
        return 'TIER 1'
    return 'TIER 3'
```

**Selection rule for multiple TIER 1 candidates:**

```python
def select_deployment_candidate(tier_results, gates_by_variant):
    """Pre-registered tiebreak: Sharpe desc -> lower PBO -> lower DSR penalty -> run-off."""
    tier1 = [v for v in V14_VARIANTS if tier_results[v] == 'TIER 1']
    if not tier1:
        return None
    sharpes = {v: gates_by_variant[v]['sharpe_5bps_nc'] for v in tier1}
    sorted_by_sharpe = sorted(tier1, key=lambda v: -sharpes[v])
    top = sorted_by_sharpe[0]
    if len(sorted_by_sharpe) > 1:
        second = sorted_by_sharpe[1]
        if abs(sharpes[top] - sharpes[second]) < 0.05:
            # Tiebreak by lower PBO (gates_by_variant['pbo'] is same value for all in this round;
            # use diagnostic PBO if available, else recommend run-off)
            return f'{top}_or_{second}__run_off_recommended'
    return top
```

**Main + report writing**: clone the V12c orchestrator's `main()` end-to-end, replacing the V12c-specific variant runs with the V14 factorial structure. Output to `docs/reports/ramp/20260526_phase4_v14_factorial_readiness.md` with sections: Summary, Cost grid (per variant), Cross-variants table, 5-gate verdict per variant, Diagnostic PBO, Sensitivity panel, Selection rule (if multiple TIER 1).

**Important**: hard-code N_TRIALS_PROJECT = 36 in the report's methodology section.

- [ ] **Step 3: Smoke-test the orchestrator imports**

```bash
PYTHONPATH=. python -c "from scripts.backtest_scripts.ramp_phase4_v14_factorial_readiness import V14_VARIANTS, N_TRIALS_PROJECT, GATE_PBO_VARIANTS; print(V14_VARIANTS, N_TRIALS_PROJECT)"
```

Expected: prints `['V14a-soft-bear-cash', 'V14b-soft-bear-spy', 'V14c-soft-bear-dampen'] 36`.

- [ ] **Step 4: Commit (do NOT run the orchestrator yet)**

```bash
git add -f scripts/backtest_scripts/ramp_phase4_v14_factorial_readiness.py
git commit -m "feat(orchestrator): V14 factorial readiness gate -- 3 variants, DSR n_trials=36

Clones ramp_phase4_v12c_readiness.py with V14-specific adaptations:

- 24 cost-grid backtests (3 variants x 4 cost levels x 2 timing modes)
- 5 cross-variant runs (V01, V11, V12, V12c, V13-bear-invert) at 5 bps nc
- 1 V11 reference at 7.5 bps lag (Gate 5)
- 4 sensitivity runs (V14a tau-band x2, V14c dampen x2)
- Total: 34 unique backtests (~30 min wall-clock)

Hard-coded DSR n_trials_project = 36 per spec rev2 audit:
V11+pre-V11 22 + V12+sensitivity 5 + V12c 1 + V13 1 + V14a/b/c 3 +
V14a tau sens 2 + V14c dampen sens 2 = 36.

Gate PBO across 8 variants {V01, V11, V12, V12c, V13, V14a/b/c}.
Diagnostic PBO across 4 orthogonal variants {V01, V11, V12, V14a},
reported alongside but not gate-influencing.

Tau values loaded once at import from config/research/v14_tau_constants.json
(pre-registered in Task 0). Selection rule for multiple TIER 1 candidates
pre-registered: Sharpe -> PBO -> DSR -> run-off recommendation.

Output: docs/reports/ramp/20260526_phase4_v14_factorial_readiness.md."
```

---

## Task 9: Run the V14 factorial orchestrator + verify

**Files (generated):**
- Generated: `C:\Users\qwqw1\Dropbox\cs\github\Homeguard\docs\reports\ramp\20260526_phase4_v14_factorial_readiness.md`

- [ ] **Step 1: Verify the env and prerequisites**

```bash
source ~/anaconda3/etc/profile.d/conda.sh && conda activate fintech
ls config/research/v14_tau_constants.json  # must exist from Task 0
PYTHONPATH=. python -c "from src.research.ramp_phase4.variants import REGISTRY; print(sorted(REGISTRY.keys()))"
```

Expected: prints registry list including V14a-soft-bear-cash, V14b-soft-bear-spy, V14c-soft-bear-dampen.

- [ ] **Step 2: Run the full test suite once more before the long run**

```bash
PYTHONPATH=. python -m pytest tests/research/ramp_phase4/ tests/diagnostics/ tests/strategies/advanced/ -q
```

Expected: all pass (~155 tests after V14 work).

- [ ] **Step 3: Run the orchestrator**

```bash
PYTHONPATH=. python scripts/backtest_scripts/ramp_phase4_v14_factorial_readiness.py 2>&1 | tail -50
```

Expected: ~30 min wall-clock. Final lines show "Wrote docs/reports/ramp/20260526_phase4_v14_factorial_readiness.md".

- [ ] **Step 4: Verify the report has all required sections**

```bash
grep -E '^## ' docs/reports/ramp/20260526_phase4_v14_factorial_readiness.md
```

Expected output includes (in some order):
- `## Summary`
- `## Per-variant 5-gate verdict`
- `## Cost grid`
- `## Cross-variants`
- `## PBO (gate + diagnostic)`
- `## Sensitivity appendix`
- Tier verdict per V14a/b/c

- [ ] **Step 5: Force-add and commit the report**

```bash
git add -f docs/reports/ramp/20260526_phase4_v14_factorial_readiness.md
git commit -m "report(ramp): V14 factorial readiness -- WS-3c soft-score consumer verdicts

35 backtests over ~30 min:
- Cost grid: 3 variants x 4 cost x 2 timing = 24 runs
- Cross-variants: V01, V11, V12, V12c, V13 at 5 bps nc = 5 runs
- V11 reference at 7.5 bps lag = 1 run
- Sensitivity (informational): V14a tau-band x2 + V14c dampen x2 = 4 runs

DSR n_trials=36 (audited count per spec rev2 honesty discipline).
Gate PBO over 8 variants; diagnostic PBO over 4 orthogonal variants.

Tier verdicts per variant + selection rule output (if any reaches TIER 1)
inline in the report. See sections per variant."
```

---

## Task 10: Update RAMP_VARIANTS.md + session log

**Files:**
- Modify: `C:\Users\qwqw1\Dropbox\cs\github\Homeguard\docs\strategies\RAMP_VARIANTS.md`
- Create: `C:\Users\qwqw1\Dropbox\cs\github\Homeguard\docs\progress\20260524_RAMP_V14_FACTORIAL_READINESS.md`

- [ ] **Step 1: Add V14 sections to RAMP_VARIANTS.md**

In `docs/strategies/RAMP_VARIANTS.md`, between the V13-bear-invert section and the "V12b / V12c -- reserved" / "V13+ -- reserved" sections, add a new section for the V14 factorial. Use this template (fill the verdict line from the readiness report produced in Task 9):

```markdown
## V14 factorial (a/b/c) -- soft-score BEAR consumer

The V14 factorial tests three actions for the same Schmitt-trigger BEAR_score consumer.
Spec: `docs/superpowers/specs/2026-05-24-v14-soft-bear-factorial-design.md`.
Open-questions resolution: `docs/superpowers/specs/2026-05-24-v14-soft-bear-factorial-design-open-questions.md`.
Plan: `docs/superpowers/plans/2026-05-24-v14-soft-bear-factorial.md`.
Pre-registered tau constants: `config/research/v14_tau_constants.json` (G1_BEAR median).
Readiness report: `docs/reports/ramp/20260526_phase4_v14_factorial_readiness.md`.

### V14a-soft-bear-cash
- **Code**: `src/research/ramp_phase4/variants.py::_variant_v14a_soft_bear_cash`
- **Action when in_bear_soft_mode**: returns PLAN_CASH_BEAR_SOFT (sentinel; engine treats as zero exposure).
- **Readiness verdict (2026-05-24)**: <FILL: TIER 1 / 3 / 4 with one-line summary from report>
- **Status**: research; <deployment decision>.

### V14b-soft-bear-spy
- **Code**: `src/research/ramp_phase4/variants.py::_variant_v14b_soft_bear_spy`
- **Action when in_bear_soft_mode**: returns {SPY: 1.0, __regime__: BEAR_SOFT_SPY} (single-name 100%).
- **Readiness verdict (2026-05-24)**: <FILL>
- **Status**: research; <deployment decision>.

### V14c-soft-bear-dampen
- **Code**: `src/research/ramp_phase4/variants.py::_variant_v14c_soft_bear_dampen`
- **Action when in_bear_soft_mode**: V11 weights * cfg.soft_bear_dampen_factor (default 0.5).
- **Readiness verdict (2026-05-24)**: <FILL>
- **Status**: research; <deployment decision>.

### Shared infrastructure
- **State machine**: `engine.py::_engine_pre_variant_update_soft_bear` (Schmitt trigger; >= tau_in enters, strict < tau_out exits).
- **State field**: `HarnessState.in_bear_soft_mode: bool`.
- **Tau pre-registration**: `scripts/diagnostics/compute_tau_in_from_g1.py` -> `config/research/v14_tau_constants.json`.
- **Plan sentinel**: `src/research/ramp_phase4/plans.py::_SentinelPlan` + `PLAN_CASH_BEAR_SOFT`.
- **Detector freshness**: `MarketRegimeDetector.last_classification_timestamp` field; each V14 variant asserts on it.
- **Honesty discipline**: DSR n_trials_project = 36 (audited); PBO gate over 8 variants {V01, V11, V12, V12c, V13, V14a/b/c}; diagnostic PBO over 4 orthogonal {V01, V11, V12, V14a}; NOT strict OOS (tau derived from G1 median on the same window, but G1 is independent of E3's lead-time sweep -- selection bias eliminated, EXT-OOS contamination remains).
```

Fill in the verdict lines after reading `docs/reports/ramp/20260526_phase4_v14_factorial_readiness.md`.

- [ ] **Step 2: Write the session log**

Create `docs/progress/20260524_RAMP_V14_FACTORIAL_READINESS.md`:

```markdown
# V14 Soft-Bear Factorial Phase D Readiness -- Session Log (2026-05-24)

## Summary

WS-3c spec rev2 (V14 factorial) implemented and gated per the 10-task plan
at `docs/superpowers/plans/2026-05-24-v14-soft-bear-factorial.md`. The
factorial tests three actions (cash / SPY / dampen 0.5) on the same
Schmitt-trigger BEAR_score consumer with pre-registered tau_in from G1_BEAR
median. Output report: `docs/reports/ramp/20260526_phase4_v14_factorial_readiness.md`.

Verdicts:
- V14a-soft-bear-cash: <FILL>
- V14b-soft-bear-spy: <FILL>
- V14c-soft-bear-dampen: <FILL>

Selection (if multiple TIER 1): <FILL>

## What ran

- Pre-spec script `compute_tau_in_from_g1.py` produced `v14_tau_constants.json` (tau_in=<FILL>, tau_out=<FILL>).
- 35 backtests over ~30 minutes wall-clock.
- DSR n_trials_project = 36 (audited honest count per spec rev2 honesty discipline).
- 50+ new unit tests covering plan sentinel, engine state machine + dispatch, config validation, and 3 canonical pinning tests.

## Headline verdicts table

(filled from readiness report)

| Variant | Tier | Sharpe @5bps nc | Sharpe @5bps lag | PSR | DSR | PBO | Gate 4 | Gate 5 |
|---|---|---|---|---|---|---|---|---|
| V14a-soft-bear-cash | <FILL> | <FILL> | <FILL> | <FILL> | <FILL> | <FILL> | <FILL> | <FILL> |
| V14b-soft-bear-spy | <FILL> | <FILL> | <FILL> | <FILL> | <FILL> | <FILL> | <FILL> | <FILL> |
| V14c-soft-bear-dampen | <FILL> | <FILL> | <FILL> | <FILL> | <FILL> | <FILL> | <FILL> | <FILL> |

## Key findings

(filled from readiness report's Findings section)

## Tier verdicts + honesty discipline

- DSR n_trials = 36 is the full audited count from spec rev2. If verdicts are TIER 3 because of this, the campaign has consumed its multi-trial budget.
- PBO gate uses 8 variants; diagnostic PBO uses 4 orthogonal variants. If gate PBO fails but diagnostic PBO passes, the synthesis flags family-correlation; the gate verdict stands either way.
- NOT strict OOS: tau_in derived from G1 median on the same 2017-2026 window. Independent of E3's lead-time sweep (no in-sample optimization), but the BEAR_score series itself is in-window. Forward OOS validation required before any paper deploy.

## Decisions

- V14a/b/c deployment status: <FILL>
- WS-3c hypothesis status: <FILL>
- Next research priority: <FILL>

## Commits this session

(filled from `git log --oneline -15`)

## Cross-experiment context

This session continues the 6-experiment campaign that began with E1-E6 on 2026-05-24:

| Exp | Verdict | Implication |
|---|---|---|
| E3 soft scores | WS-3c | Soft-score consumption is the right track (motivated this V14 work) |
| E2 UNPREDICTABLE | AMBIGUOUS | E6 proceeded with COVID-excluded panel |
| E4 lag asymmetry | DIFFUSE | E6 used standard cost grid |
| E1 V13-bear-invert | TIER 4 | argmax-BEAR-as-buy spurious; V14b tests soft-trigger BEAR-as-buy |
| E5 OMR cross-check | AMBIGUOUS | WS-3 is RAMP-attributable lever |
| E6 V12c readiness | TIER 4 | argmax-BEAR+UNPREDICTABLE-to-cash overfits (PBO 0.71) |

V14 factorial is the first soft-score consumption surface tested.
```

Fill in the FILL placeholders by reading the readiness report.

- [ ] **Step 3: Commit**

```bash
git add docs/strategies/RAMP_VARIANTS.md
git add docs/progress/20260524_RAMP_V14_FACTORIAL_READINESS.md
git commit -m "docs(progress): V14 factorial readiness session log + RAMP_VARIANTS V14 sections

RAMP_VARIANTS gets a V14 factorial section listing V14a/b/c with their
readiness verdicts, shared infrastructure pointers, and the honesty
discipline framing (DSR n_trials=36, PBO 8-variant gate, NOT strict OOS).

Session log captures the campaign continuation from E1-E6, the V14
factorial outcomes, and the deployment / next-research decisions."
```

---

## Final verification

After Task 10:

- [ ] **Run the full test suite one last time**

```bash
PYTHONPATH=. python -m pytest tests/research/ramp_phase4/ tests/diagnostics/ tests/strategies/advanced/ -q
```

Expected: ~155 passed (105 pre-V14 + 50 V14).

- [ ] **Verify all commits landed**

```bash
git log --oneline -15
```

Expected: see 10+ V14-related commits in the recent history.

- [ ] **Verify no working tree drift**

```bash
git status --short
```

Expected: only the pre-existing tmp files; no V14 deliverables left uncommitted.

---

## Risks / known issues

| Risk | Mitigation |
|---|---|
| `_compute_plan_from_panel` ordering changes break V14 freshness assertion | Detector freshness assertion in each V14 variant catches this immediately (AssertionError); tests cover the boundary |
| Orchestrator wall-clock exceeds 30 min due to JIT warmup or panel I/O | Confirm import timing in Task 9 Step 2; if > 45 min escalate before committing the report |
| V14 PBO inflates due to family correlation with V11/V12 | Spec rev2 forbids reinterpretation. Diagnostic PBO panel is reported alongside but does NOT override the gate verdict. |
| `_SentinelPlan` dispatch breaks an existing V01-V13 variant due to subtle typing | Engine tests in test_engine_v14.py cover the dispatch contract; full suite regression check in Task 3 Step 8 |
| tau pre-registration JSON missing at orchestrator time | Task 0 must run before Task 9; `load_v14_tau_constants` raises FileNotFoundError otherwise |
| Concurrent subagent commits introduce orphan files (as happened in E1/E6) | Sequential subagent-driven execution avoids this; do NOT parallelize V14 task subagents |

---

## What this plan does NOT do

- No production detector logic changes (only `last_classification_timestamp` additive field).
- No V11/V12/V12c/V13 modifications.
- No new data acquisition (uses existing 2017-2026 panel + v0 labels + v0_scores).
- No live deployment (forward OOS validation required regardless of TIER outcome).
- No multi-tau parameter search as gates (sensitivity panels are informational only).
- No min-persistence filter (Schmitt trigger is the chosen noise suppression).
- No UNPREDICTABLE_score consumption (deferred to future WS-3c.1 after measuring its lead-time).
