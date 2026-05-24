# V12 BEAR-to-Cash Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement V12 (V11 base + per-regime position override, default BEAR -> cash) in the Phase 4 research harness, with symmetric debouncing as an optional config field, and run it through a re-baselined readiness orchestrator producing a 5-gate verdict + detector-onset alignment + sensitivity appendix.

**Architecture:** Pre-variant engine update of `state.last_regime`, `state.regime_streak`, and `state.last_validated_regime`. Variant `_variant_v12` is a pure read on state; it wraps `_variant_v11` and overrides the target weights based on `cfg.regime_positions[active_regime]` where `active_regime` is either the instantaneous regime (when `min_regime_days=0`, the v12.0.0 default) or `state.last_validated_regime` (when debouncing is enabled, sensitivity-only). New orchestrator `ramp_phase4_v12_readiness.py` splits gate-influencing runs (13) from sensitivity-appendix runs (4); only the gate-influencing set defines V12's published metrics.

**Tech Stack:** Python 3 (fintech conda env), pytest, pandas, numpy. Existing helpers in `src/research/ramp_phase4/` (engine, variants, filters, config, costs, data, metrics, reports). Statistical gates from `src/backtesting/statistics/` (psr, dsr, pbo).

**Spec:** `docs/superpowers/specs/2026-05-23-v12-bear-to-cash-design.md` (rev4 + rev4-followup).

**Branch:** `v12-bear-to-cash` (already created from `ramp-phase4-turnover-regime-research` at `fc7de60`; spec imported at `3c67d27`).

---

## File structure

New files this plan creates:

| Path | Responsibility |
|---|---|
| `scripts/backtest_scripts/ramp_phase4_v12_readiness.py` | V12 readiness orchestrator (gate-influencing + sensitivity-appendix runs) |
| `docs/strategies/RAMP_VARIANTS.md` | Canonical variant glossary (V01-V12 + V12b/V12c/V13+ reserved sections) |
| `docs/reports/ramp/20260523_phase4_v12_readiness.md` | Readiness output (emitted by orchestrator) |
| `docs/progress/20260523_RAMP_V12_SESSION_LOG.md` | Session log written at end |

Modified files:

| Path | Change |
|---|---|
| `src/research/ramp_phase4/config.py` | Add `regime_positions: Dict[str, str]` field, `min_regime_days: int` field, `__post_init__` validation |
| `src/research/ramp_phase4/engine.py` | Add `HarnessState.last_regime`, `regime_streak`, `last_validated_regime`. Add pre-variant update in `run_variant` per-tick. |
| `src/research/ramp_phase4/variants.py` | Add `_variant_v12`, add `'V12'` to `REGISTRY` |
| `tests/research/ramp_phase4/test_engine.py` | +5 engine state-tracking tests + 2 integration tests |
| `tests/research/ramp_phase4/test_variants.py` | +14 V12 unit tests including the canonical pinning test |

Existing reused (do NOT reimplement):

- `src/research/ramp_phase4/variants.py::_variant_v11` (V12 wraps it)
- `src/research/ramp_phase4/filters.py::rank_buffer`, `min_hold` (inherited via V11)
- `src/research/ramp_phase4/engine.py::run_variant`, `compute_trades`, `apply_trades` (engine; we add state fields + a pre-variant hook)
- `src/backtesting/statistics/psr.py::psr`, `dsr.py::dsr`, `pbo.py::pbo` (for orchestrator gates)
- `scripts/backtest_scripts/ramp_phase4_v11_readiness.py` (template; copy and modify for V12)

Validation env: `conda activate fintech` (Python at `C:\Users\qwqw1\anaconda3\envs\fintech\python.exe`). Tests: `python -m pytest tests/research/ramp_phase4/ -v`.

---

## Task 1: Engine state additions + pre-variant update

**Files:**
- Modify: `src/research/ramp_phase4/engine.py:25-37` (HarnessState dataclass)
- Modify: `src/research/ramp_phase4/engine.py:64-130` (run_variant main loop)
- Test: `tests/research/ramp_phase4/test_engine.py`

The engine gains three state fields and a 7-line pre-variant update block. Default `min_regime_days=0` makes it a no-op for V01-V11.

- [ ] **Step 1.1: Write the failing test for regime_streak increment**

Add to `tests/research/ramp_phase4/test_engine.py`:

```python
from src.research.ramp_phase4.engine import HarnessState

def test_engine_regime_streak_increments():
    """Two consecutive ticks of the same regime: streak goes 1 -> 2."""
    state = HarnessState(cash_usd=100000.0)
    # First tick: BEAR
    _engine_pre_variant_update(state, 'BEAR', min_regime_days=0)
    assert state.regime_streak == {'BEAR': 1}
    assert state.last_regime == 'BEAR'
    # Second tick: BEAR again
    _engine_pre_variant_update(state, 'BEAR', min_regime_days=0)
    assert state.regime_streak == {'BEAR': 2}
    assert state.last_regime == 'BEAR'


def test_engine_regime_streak_resets_on_flip():
    """Regime BEAR -> WEAK_BULL: streak dict becomes {WEAK_BULL: 1}."""
    state = HarnessState(cash_usd=100000.0)
    _engine_pre_variant_update(state, 'BEAR', min_regime_days=0)
    _engine_pre_variant_update(state, 'BEAR', min_regime_days=0)
    _engine_pre_variant_update(state, 'WEAK_BULL', min_regime_days=0)
    assert state.regime_streak == {'WEAK_BULL': 1}
    assert state.last_regime == 'WEAK_BULL'


def test_engine_last_validated_regime_with_min_zero():
    """With min_regime_days=0, last_validated_regime tracks instantaneous regime."""
    state = HarnessState(cash_usd=100000.0)
    _engine_pre_variant_update(state, 'BEAR', min_regime_days=0)
    assert state.last_validated_regime == 'BEAR'
    _engine_pre_variant_update(state, 'WEAK_BULL', min_regime_days=0)
    assert state.last_validated_regime == 'WEAK_BULL'


def test_engine_last_validated_regime_with_min_three():
    """With min_regime_days=3, last_validated_regime stays None for ticks 0-1,
    becomes the regime on tick 2 (streak reaches 3)."""
    state = HarnessState(cash_usd=100000.0)
    _engine_pre_variant_update(state, 'BEAR', min_regime_days=3)
    assert state.last_validated_regime is None  # streak=1 < 3
    _engine_pre_variant_update(state, 'BEAR', min_regime_days=3)
    assert state.last_validated_regime is None  # streak=2 < 3
    _engine_pre_variant_update(state, 'BEAR', min_regime_days=3)
    assert state.last_validated_regime == 'BEAR'  # streak=3 >= 3


def test_engine_first_tick_initialization():
    """t=0 with last_regime=None correctly enters the flip branch."""
    state = HarnessState(cash_usd=100000.0)
    assert state.last_regime is None
    assert state.regime_streak == {}
    assert state.last_validated_regime is None
    _engine_pre_variant_update(state, 'BEAR', min_regime_days=0)
    # The equality `None == 'BEAR'` is False, so flip branch fires.
    assert state.regime_streak == {'BEAR': 1}
    assert state.last_regime == 'BEAR'
    assert state.last_validated_regime == 'BEAR'  # 1 >= 0
```

Note the import: `_engine_pre_variant_update` is what we'll name the helper. Import it at the top of the test file from `src.research.ramp_phase4.engine`.

- [ ] **Step 1.2: Run the failing tests**

```bash
source /c/Users/qwqw1/anaconda3/etc/profile.d/conda.sh && conda activate fintech && python -m pytest tests/research/ramp_phase4/test_engine.py -v -k "regime_streak or last_validated_regime or first_tick_initialization" 2>&1 | tail -15
```

Expected: 5 tests fail with `AttributeError` (missing fields on HarnessState) and `ImportError` (missing `_engine_pre_variant_update`).

- [ ] **Step 1.3: Add the new state fields to HarnessState**

Edit `src/research/ramp_phase4/engine.py` at the `HarnessState` dataclass (around line 24-37). After the existing `last_target_symbols` line, add:

```python
    # V12: per-regime debouncing state (pre-variant update each tick).
    last_regime: Optional[str] = None
    regime_streak: Dict[str, int] = field(default_factory=dict)
    last_validated_regime: Optional[str] = None
```

`Optional` needs to be added to the typing import at line 12 if not already there:

```python
from typing import Callable, Dict, List, Optional, Protocol
```

- [ ] **Step 1.4: Add the pre-variant update helper**

Add this module-level function to `src/research/ramp_phase4/engine.py` (after `HarnessState` class definition, before `run_variant`):

```python
def _engine_pre_variant_update(state: HarnessState, regime: str, min_regime_days: int) -> None:
    """Update streak + last_validated_regime BEFORE the variant runs on each tick.

    Pre-variant ordering is committed by the V12 spec (rev4). The variant
    sees the updated state, so a regime that just hit the threshold takes
    effect on the same tick it hit it.

    With min_regime_days=0 (the v12.0.0 default and all V01-V11 default),
    `1 >= 0` is always true, so `last_validated_regime` tracks the
    instantaneous regime; bit-equivalent to no-debouncing behavior.
    """
    # 1. Update regime streak.
    if state.last_regime == regime:
        state.regime_streak[regime] = state.regime_streak.get(regime, 0) + 1
    else:
        # Regime flip: reset streak. First-tick: last_regime is None,
        # which never equals any real regime name, so this branch fires.
        state.regime_streak = {regime: 1}
    state.last_regime = regime

    # 2. Update last_validated_regime if current regime has cleared threshold.
    if state.regime_streak[regime] >= min_regime_days:
        state.last_validated_regime = regime
```

- [ ] **Step 1.5: Wire the helper into `run_variant` per-tick**

In `run_variant`, find the loop over `panel.iterrows()` (around line 80). Inside the loop, after computing `regime = ...` (or alongside the variant call site), insert the pre-variant update. Look for where the variant's plan output is parsed to extract `regime_label`. The update must happen BEFORE `plan_fn(...)` is called.

Locate this block:

```python
plan_output = variant_spec.plan_fn(ts, state, panel, cfg) or {}
regime_label = str(plan_output.pop('__regime__', 'STUB'))
```

We need to know the regime BEFORE calling `plan_fn`. The detector is invoked inside `plan_fn`, so we can't trivially extract the regime ahead of time without restructuring. Two options:

- **Option A (used here):** call `plan_fn` once first to get the regime, then run `_engine_pre_variant_update`, then RE-CALL `plan_fn`. This double-calls (inefficient).
- **Option B:** plan_fn returns regime in output dict, engine updates AFTER plan_fn runs but BEFORE applying the active_mode branching from `cfg.regime_positions`. For V01-V11 the state mutation is invisible because they don't read `last_validated_regime`. For V12, the variant must look at the state AFTER the update.

Option B is the correct architectural choice. V12's variant logic (Task 3) reads `state.last_validated_regime` AFTER `_variant_v11` has run and returned the regime. The engine update happens between the V11 inner call and the V12 outer decision. This means `_engine_pre_variant_update` is called from INSIDE `_variant_v12`, not from the engine loop.

Re-architect: move `_engine_pre_variant_update` call from `run_variant` into `_variant_v12`. The engine itself stays unchanged. The state fields are still part of HarnessState (they need to persist across ticks).

This simplifies the engine modification to just the dataclass fields. Revise the test imports accordingly: `_engine_pre_variant_update` is a public helper exported from engine.py, callable from variants.

- [ ] **Step 1.6: Run the engine tests to verify they pass**

```bash
python -m pytest tests/research/ramp_phase4/test_engine.py -v -k "regime_streak or last_validated_regime or first_tick_initialization" 2>&1 | tail -15
```

Expected: 5 / 5 PASS.

- [ ] **Step 1.7: Run the full Phase 4 test suite to confirm no regressions**

```bash
python -m pytest tests/research/ramp_phase4/ 2>&1 | tail -5
```

Expected: baseline test count + 5 new tests, all pass.

- [ ] **Step 1.8: Commit**

```bash
git add src/research/ramp_phase4/engine.py tests/research/ramp_phase4/test_engine.py
git commit -m "feat(harness): add regime debouncing state to HarnessState + pre-variant helper

V12 spec adds last_regime, regime_streak, last_validated_regime fields
to HarnessState plus an _engine_pre_variant_update helper that updates
all three on each tick. Default min_regime_days=0 makes the helper a
no-op (last_validated_regime tracks instantaneous regime), so V01-V11
remain bit-equivalent.

The helper is called from inside _variant_v12 (Task 3), not from
run_variant directly. The state fields are part of HarnessState so
they persist across ticks regardless of variant.

5 TDD tests cover: increment, flip reset, min=0 instantaneous tracking,
min=3 validation-after-3-ticks, first-tick None initialization."
```

---

## Task 2: Config schema additions

**Files:**
- Modify: `src/research/ramp_phase4/config.py`
- Test: `tests/research/ramp_phase4/test_engine.py` (config validation lives in test_engine.py alongside HarnessConfig usage; there is no separate test_config.py)

Add `regime_positions` and `min_regime_days` fields plus `__post_init__` validation. `HarnessConfig` is `@dataclass(frozen=True)` so `__post_init__` can raise but not mutate.

- [ ] **Step 2.1: Write the failing tests**

Add to `tests/research/ramp_phase4/test_engine.py`:

```python
import pytest
from src.research.ramp_phase4.config import HarnessConfig


def test_harness_config_defaults_match_spec():
    """v12.0.0 defaults: BEAR -> cash, others -> normal, SAFE_MODE -> hold, min=0."""
    cfg = HarnessConfig(
        start_date=datetime(2017, 1, 1),
        end_date=datetime(2026, 5, 16),
        universe_csv=Path('config/universes/sp500-2025.csv'),
        initial_capital=100000.0,
        cost_bps_per_side=5.0,
    )
    assert cfg.regime_positions == {
        'STRONG_BULL': 'normal',
        'WEAK_BULL': 'normal',
        'SIDEWAYS': 'normal',
        'UNPREDICTABLE': 'normal',
        'BEAR': 'cash',
        'SAFE_MODE': 'hold',
    }
    assert cfg.min_regime_days == 0


def test_harness_config_rejects_unknown_position_value():
    """ValueError on regime_positions value not in {normal, cash, hold}."""
    with pytest.raises(ValueError, match='regime_positions'):
        HarnessConfig(
            start_date=datetime(2017, 1, 1),
            end_date=datetime(2026, 5, 16),
            universe_csv=Path('config/universes/sp500-2025.csv'),
            initial_capital=100000.0,
            cost_bps_per_side=5.0,
            regime_positions={'BEAR': 'TLT'},  # Reserved for V13+; raises
        )


def test_harness_config_rejects_negative_min_regime_days():
    """ValueError on min_regime_days < 0."""
    with pytest.raises(ValueError, match='min_regime_days'):
        HarnessConfig(
            start_date=datetime(2017, 1, 1),
            end_date=datetime(2026, 5, 16),
            universe_csv=Path('config/universes/sp500-2025.csv'),
            initial_capital=100000.0,
            cost_bps_per_side=5.0,
            min_regime_days=-1,
        )


def test_harness_config_allows_unknown_regime_keys():
    """Unknown regime KEYS fall through to 'normal' at variant runtime; constructor accepts."""
    # Should not raise. The variant default in the .get('NEW_REGIME', 'normal')
    # call site handles unknown keys.
    cfg = HarnessConfig(
        start_date=datetime(2017, 1, 1),
        end_date=datetime(2026, 5, 16),
        universe_csv=Path('config/universes/sp500-2025.csv'),
        initial_capital=100000.0,
        cost_bps_per_side=5.0,
        regime_positions={'BEAR': 'cash', 'FUTURE_REGIME': 'normal'},
    )
    assert 'FUTURE_REGIME' in cfg.regime_positions
```

Add the `datetime` and `Path` imports near the top of `test_engine.py` if not already present.

- [ ] **Step 2.2: Run the failing tests**

```bash
python -m pytest tests/research/ramp_phase4/test_engine.py -v -k "harness_config" 2>&1 | tail -15
```

Expected: 4 tests fail (missing fields / missing validation).

- [ ] **Step 2.3: Add the new fields with validation to HarnessConfig**

Edit `src/research/ramp_phase4/config.py`. Replace the existing class body with:

```python
"""HarnessConfig dataclass for Phase B research harness.

Pure data; no logic except validation in __post_init__.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Literal


_DEFAULT_REGIME_POSITIONS = {
    'STRONG_BULL':   'normal',
    'WEAK_BULL':     'normal',
    'SIDEWAYS':      'normal',
    'UNPREDICTABLE': 'normal',
    'BEAR':          'cash',
    'SAFE_MODE':     'hold',
}

_ALLOWED_POSITION_VALUES = frozenset({'normal', 'cash', 'hold'})


@dataclass(frozen=True)
class HarnessConfig:
    """Configuration for a single run of run_variant.

    Run the harness once per (variant, cost-tier, timing-mode) combination.
    """
    start_date: datetime
    end_date: datetime
    universe_csv: Path
    initial_capital: float
    cost_bps_per_side: float
    timing_mode: Literal['near_close', 'one_day_lag'] = 'near_close'
    rebalance_frequency: Literal['daily', 'weekly_friday', 'weekly_wednesday'] = 'daily'
    rounding_mode: Literal['whole_share', 'dollar_weight'] = 'whole_share'
    min_trade_value_usd: float = 100.0
    delta_rebalance_pct: float = 0.0
    # V12 additions:
    regime_positions: Dict[str, str] = field(
        default_factory=lambda: dict(_DEFAULT_REGIME_POSITIONS)
    )
    min_regime_days: int = 0

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
```

- [ ] **Step 2.4: Run the tests to verify they pass**

```bash
python -m pytest tests/research/ramp_phase4/test_engine.py -v -k "harness_config" 2>&1 | tail -10
```

Expected: 4 / 4 PASS.

- [ ] **Step 2.5: Run the full Phase 4 test suite for regressions**

```bash
python -m pytest tests/research/ramp_phase4/ 2>&1 | tail -5
```

Expected: baseline + 5 (Task 1) + 4 (Task 2) = baseline + 9 tests pass.

- [ ] **Step 2.6: Commit**

```bash
git add src/research/ramp_phase4/config.py tests/research/ramp_phase4/test_engine.py
git commit -m "feat(harness): add regime_positions + min_regime_days config fields

V12 per-regime position override config. Default v12.0.0 values:
BEAR -> cash, others -> normal, SAFE_MODE -> hold, min_regime_days=0.
Validation in __post_init__ rejects unknown position values (ticker/
strategy names reserved for V13+) and negative min_regime_days.

4 TDD tests cover defaults, position value validation, min value
validation, and unknown regime keys (allowed; fall through at runtime).

Default min_regime_days=0 preserves V01-V11 bit-equivalence."
```

---

## Task 3: V12 variant + REGISTRY entry

**Files:**
- Modify: `src/research/ramp_phase4/variants.py:189-265` (add `_variant_v12` after `_variant_v11`; add `'V12'` to `REGISTRY`)
- Test: `tests/research/ramp_phase4/test_variants.py`

This is the V12 logic itself. The variant wraps `_variant_v11`, calls `_engine_pre_variant_update` to update state, then branches on `cfg.regime_positions[active_regime]` where `active_regime` is `state.last_validated_regime` if debouncing is on, else `regime`.

- [ ] **Step 3.1: Write the failing tests including the canonical pinning test**

Add to `tests/research/ramp_phase4/test_variants.py`:

```python
import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

from src.research.ramp_phase4.config import HarnessConfig
from src.research.ramp_phase4.engine import HarnessState
from src.research.ramp_phase4.variants import REGISTRY


def _stub_cfg(min_regime_days: int = 0, regime_positions: dict = None) -> HarnessConfig:
    """Build a minimal HarnessConfig for variant tests."""
    return HarnessConfig(
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 12, 31),
        universe_csv=Path('config/universes/sp500-2025.csv'),
        initial_capital=100000.0,
        cost_bps_per_side=5.0,
        min_regime_days=min_regime_days,
        regime_positions=regime_positions or {
            'STRONG_BULL': 'normal', 'WEAK_BULL': 'normal',
            'SIDEWAYS': 'normal', 'UNPREDICTABLE': 'normal',
            'BEAR': 'cash', 'SAFE_MODE': 'hold',
        },
    )


def _fresh_state() -> HarnessState:
    return HarnessState(cash_usd=100000.0)


# ============================================================
# Basic mode behavior (8 tests)
# ============================================================

def test_v12_normal_regime_matches_v11(_v12_test_panel_strong_bull):
    """V12 with default config on STRONG_BULL day == V11 output exactly."""
    state = _fresh_state()
    cfg = _stub_cfg()
    v11_out = REGISTRY['V11'].plan_fn(datetime(2024, 6, 15), state, _v12_test_panel_strong_bull, cfg)
    state_v12 = _fresh_state()
    v12_out = REGISTRY['V12'].plan_fn(datetime(2024, 6, 15), state_v12, _v12_test_panel_strong_bull, cfg)
    assert v12_out == v11_out


def test_v12_bear_day_returns_empty_targets(_v12_test_panel_bear):
    """V12 on BEAR day returns {'__regime__': 'BEAR'} with no weights."""
    state = _fresh_state()
    cfg = _stub_cfg()
    v12_out = REGISTRY['V12'].plan_fn(datetime(2024, 6, 15), state, _v12_test_panel_bear, cfg)
    assert v12_out == {'__regime__': 'BEAR'}


def test_v12_unpredictable_day_defaults_to_v11(_v12_test_panel_unpredictable):
    """v12.0.0 default: UNPREDICTABLE -> normal -> V11 output."""
    state = _fresh_state()
    cfg = _stub_cfg()  # UNPREDICTABLE: 'normal' by default
    v11_out = REGISTRY['V11'].plan_fn(datetime(2024, 6, 15), state, _v12_test_panel_unpredictable, cfg)
    v12_out = REGISTRY['V12'].plan_fn(datetime(2024, 6, 15), _fresh_state(), _v12_test_panel_unpredictable, cfg)
    assert v12_out == v11_out


def test_v12_unpredictable_day_returns_cash_when_configured(_v12_test_panel_unpredictable):
    """V12 with UNPREDICTABLE -> 'cash' returns empty targets on UNPREDICTABLE."""
    cfg = _stub_cfg(regime_positions={
        'STRONG_BULL': 'normal', 'WEAK_BULL': 'normal',
        'SIDEWAYS': 'normal', 'UNPREDICTABLE': 'cash',
        'BEAR': 'cash', 'SAFE_MODE': 'hold',
    })
    v12_out = REGISTRY['V12'].plan_fn(datetime(2024, 6, 15), _fresh_state(), _v12_test_panel_unpredictable, cfg)
    assert v12_out == {'__regime__': 'UNPREDICTABLE'}


def test_v12_sideways_default_matches_v11(_v12_test_panel_sideways):
    """V12 SIDEWAYS day with default config == V11 output."""
    cfg = _stub_cfg()
    v11_out = REGISTRY['V11'].plan_fn(datetime(2024, 6, 15), _fresh_state(), _v12_test_panel_sideways, cfg)
    v12_out = REGISTRY['V12'].plan_fn(datetime(2024, 6, 15), _fresh_state(), _v12_test_panel_sideways, cfg)
    assert v12_out == v11_out


def test_v12_safe_mode_preserves_positions(_v12_test_panel_safe_mode):
    """V12 on SAFE_MODE returns {'__regime__': 'SAFE_MODE'}."""
    cfg = _stub_cfg()
    v12_out = REGISTRY['V12'].plan_fn(datetime(2024, 6, 15), _fresh_state(), _v12_test_panel_safe_mode, cfg)
    assert v12_out == {'__regime__': 'SAFE_MODE'}


def test_v12_bear_then_safe_mode_stays_in_cash(_v12_test_panels_bear_then_safe):
    """BEAR -> SAFE_MODE sequence: V12 holds nothing through both."""
    cfg = _stub_cfg()
    state = _fresh_state()
    panels_bear, panels_safe = _v12_test_panels_bear_then_safe
    v12_bear = REGISTRY['V12'].plan_fn(datetime(2024, 6, 14), state, panels_bear, cfg)
    assert v12_bear == {'__regime__': 'BEAR'}
    v12_safe = REGISTRY['V12'].plan_fn(datetime(2024, 6, 15), state, panels_safe, cfg)
    assert v12_safe == {'__regime__': 'SAFE_MODE'}


def test_v12_config_override_sideways_to_cash(_v12_test_panel_sideways):
    """V12 with SIDEWAYS -> 'cash' returns empty targets on SIDEWAYS."""
    cfg = _stub_cfg(regime_positions={
        'STRONG_BULL': 'normal', 'WEAK_BULL': 'normal',
        'SIDEWAYS': 'cash', 'UNPREDICTABLE': 'normal',
        'BEAR': 'cash', 'SAFE_MODE': 'hold',
    })
    v12_out = REGISTRY['V12'].plan_fn(datetime(2024, 6, 15), _fresh_state(), _v12_test_panel_sideways, cfg)
    assert v12_out == {'__regime__': 'SIDEWAYS'}


# ============================================================
# Debouncing (4 tests; canonical pinning test is the spec)
# ============================================================

def test_v12_hysteresis_day_0_starts_normal(_v12_test_panel_bear):
    """min_regime_days=3, day 0 BEAR: last_validated_regime is None, mode = normal."""
    state = _fresh_state()
    cfg = _stub_cfg(min_regime_days=3)
    v12_out = REGISTRY['V12'].plan_fn(datetime(2024, 6, 15), state, _v12_test_panel_bear, cfg)
    # Not cash; falls through to normal (V11 output)
    assert '__regime__' in v12_out
    assert v12_out['__regime__'] == 'BEAR'
    # If cash, would have only __regime__; if normal, has additional weight keys.
    assert len(v12_out) > 1


def test_v12_hysteresis_validates_after_threshold(_v12_test_panel_bear):
    """BEAR for 3 consecutive days with min_regime_days=3 -> tick 2 returns cash."""
    state = _fresh_state()
    cfg = _stub_cfg(min_regime_days=3)
    # Tick 0 + 1: normal
    REGISTRY['V12'].plan_fn(datetime(2024, 6, 13), state, _v12_test_panel_bear, cfg)
    REGISTRY['V12'].plan_fn(datetime(2024, 6, 14), state, _v12_test_panel_bear, cfg)
    # Tick 2: cash (streak=3 reaches threshold)
    out_tick2 = REGISTRY['V12'].plan_fn(datetime(2024, 6, 15), state, _v12_test_panel_bear, cfg)
    assert out_tick2 == {'__regime__': 'BEAR'}


def test_v12_hysteresis_revalidates_on_sustained_flip(_v12_test_panel_bear, _v12_test_panel_weak_bull):
    """BEAR for 5 days -> cash. Then WEAK_BULL for 3 days -> tick 7 returns normal."""
    state = _fresh_state()
    cfg = _stub_cfg(min_regime_days=3)
    base = datetime(2024, 6, 1)
    # 5 BEAR ticks: ticks 0-1 normal, ticks 2-4 cash
    for i in range(5):
        REGISTRY['V12'].plan_fn(base.replace(day=base.day + i), state, _v12_test_panel_bear, cfg)
    # 3 WEAK_BULL ticks
    REGISTRY['V12'].plan_fn(base.replace(day=base.day + 5), state, _v12_test_panel_weak_bull, cfg)  # tick 5 (WB streak 1)
    REGISTRY['V12'].plan_fn(base.replace(day=base.day + 6), state, _v12_test_panel_weak_bull, cfg)  # tick 6 (WB streak 2)
    out_tick7 = REGISTRY['V12'].plan_fn(base.replace(day=base.day + 7), state, _v12_test_panel_weak_bull, cfg)  # tick 7 (WB streak 3 -> validates)
    assert '__regime__' in out_tick7
    assert out_tick7['__regime__'] == 'WEAK_BULL'
    # WB streak=3, last_validated_regime flipped to WEAK_BULL -> mode=normal -> V11 weights present.
    assert len(out_tick7) > 1


def test_v12_hysteresis_symmetric_canonical(_v12_test_panel_weak_bull, _v12_test_panel_bear):
    """
    CANONICAL PINNING TEST -- THE SOURCE OF TRUTH for V12 debouncing semantics.

    Per spec: 13 ticks (0..12) driving state through cold start, validation,
    transient flip-back (tick 7), and re-validation. Pre-variant ordering.
    """
    cfg = _stub_cfg(min_regime_days=3)
    state = _fresh_state()

    # Sequence: (tick, panel, expected_active_mode)
    # The panel selects the regime; the expected mode is what V12 returns.
    schedule = [
        (0,  _v12_test_panel_weak_bull, 'normal'),  # WB streak 1
        (1,  _v12_test_panel_weak_bull, 'normal'),  # WB streak 2
        (2,  _v12_test_panel_weak_bull, 'normal'),  # WB streak 3 -> validated; mode WB-> normal
        (3,  _v12_test_panel_bear,      'normal'),  # BEAR streak 1; LVR=WB still
        (4,  _v12_test_panel_bear,      'normal'),  # BEAR streak 2
        (5,  _v12_test_panel_bear,      'cash'),    # BEAR streak 3 -> validated; LIQUIDATE
        (6,  _v12_test_panel_bear,      'cash'),    # BEAR streak 4
        (7,  _v12_test_panel_weak_bull, 'cash'),    # WB streak 1; LVR=BEAR; PIN: symmetric stall
        (8,  _v12_test_panel_bear,      'cash'),    # BEAR streak 1 again; LVR=BEAR still
        (9,  _v12_test_panel_bear,      'cash'),    # BEAR streak 2
        (10, _v12_test_panel_weak_bull, 'cash'),    # WB streak 1
        (11, _v12_test_panel_weak_bull, 'cash'),    # WB streak 2
        (12, _v12_test_panel_weak_bull, 'normal'),  # WB streak 3 -> re-validated; RE-ENTER via V11
    ]

    base = datetime(2024, 6, 1)
    for tick, panel, expected_mode in schedule:
        out = REGISTRY['V12'].plan_fn(base.replace(day=base.day + tick), state, panel, cfg)
        actual_mode = _interpret_plan_as_mode(out)
        assert actual_mode == expected_mode, (
            f"tick {tick}: mode got {actual_mode}, expected {expected_mode}; "
            f"streak={state.regime_streak}, LVR={state.last_validated_regime}"
        )


def _interpret_plan_as_mode(plan: dict) -> str:
    """Return 'cash' if only __regime__ key, 'hold' if regime is SAFE_MODE, else 'normal'."""
    if plan.get('__regime__') == 'SAFE_MODE':
        return 'hold'
    weight_keys = [k for k in plan.keys() if k != '__regime__']
    if not weight_keys:
        return 'cash'
    return 'normal'
```

The panel fixtures `_v12_test_panel_bear`, `_v12_test_panel_weak_bull`, etc., are pytest fixtures that produce minimal SPY+VIX panels engineered to hit each regime. Define them once in a shared conftest or as fixtures in `test_variants.py`:

```python
@pytest.fixture
def _v12_test_panel_bear() -> pd.DataFrame:
    """Synthetic panel that the detector will classify as BEAR.
    Requires SPY below all 3 SMAs + momentum < -2% + VIX percentile > 70.
    """
    n_days = 300  # Enough for 200-SMA + VIX 252-day percentile.
    dates = pd.bdate_range('2023-01-01', periods=n_days)
    # Steady declining SPY: today's price is below all SMAs.
    spy = pd.Series(np.linspace(400, 300, n_days), index=dates)
    # Elevated VIX, in upper percentile of historical.
    vix = pd.Series(np.linspace(20, 35, n_days), index=dates)
    # Build the panel format the harness expects: wide DataFrame indexed by date,
    # with SPY and VIX as columns, plus a couple of dummy stocks for top_n picks.
    panel = pd.DataFrame({
        'SPY': spy, 'VIX': vix,
        'AAPL': pd.Series(np.linspace(170, 150, n_days), index=dates),
        'MSFT': pd.Series(np.linspace(340, 300, n_days), index=dates),
    })
    return panel


@pytest.fixture
def _v12_test_panel_strong_bull() -> pd.DataFrame:
    """Steady upward trend, low VIX -> STRONG_BULL."""
    n_days = 300
    dates = pd.bdate_range('2023-01-01', periods=n_days)
    spy = pd.Series(np.linspace(300, 450, n_days), index=dates)
    vix = pd.Series(np.linspace(20, 14, n_days), index=dates)
    panel = pd.DataFrame({
        'SPY': spy, 'VIX': vix,
        'AAPL': pd.Series(np.linspace(150, 200, n_days), index=dates),
        'MSFT': pd.Series(np.linspace(300, 400, n_days), index=dates),
    })
    return panel


@pytest.fixture
def _v12_test_panel_weak_bull() -> pd.DataFrame:
    """Modest upward trend, moderate VIX -> WEAK_BULL."""
    n_days = 300
    dates = pd.bdate_range('2023-01-01', periods=n_days)
    spy = pd.Series(np.linspace(380, 405, n_days), index=dates)
    vix = pd.Series(np.linspace(18, 22, n_days), index=dates)
    panel = pd.DataFrame({
        'SPY': spy, 'VIX': vix,
        'AAPL': pd.Series(np.linspace(170, 185, n_days), index=dates),
        'MSFT': pd.Series(np.linspace(330, 360, n_days), index=dates),
    })
    return panel


@pytest.fixture
def _v12_test_panel_sideways() -> pd.DataFrame:
    """Flat SPY, moderate VIX -> SIDEWAYS."""
    n_days = 300
    dates = pd.bdate_range('2023-01-01', periods=n_days)
    spy = pd.Series(400 + 5 * np.sin(np.linspace(0, 6 * np.pi, n_days)), index=dates)
    vix = pd.Series(np.full(n_days, 20.0), index=dates)
    panel = pd.DataFrame({
        'SPY': spy, 'VIX': vix,
        'AAPL': pd.Series(np.full(n_days, 175.0), index=dates),
        'MSFT': pd.Series(np.full(n_days, 340.0), index=dates),
    })
    return panel


@pytest.fixture
def _v12_test_panel_unpredictable() -> pd.DataFrame:
    """SPY mixed-vs-SMAs + VIX percentile > 60 -> UNPREDICTABLE."""
    n_days = 300
    dates = pd.bdate_range('2023-01-01', periods=n_days)
    # Choppy SPY
    spy = pd.Series(400 + 20 * np.sin(np.linspace(0, 4 * np.pi, n_days)), index=dates)
    # Spiking VIX
    vix = pd.Series(np.linspace(25, 32, n_days), index=dates)
    panel = pd.DataFrame({
        'SPY': spy, 'VIX': vix,
        'AAPL': pd.Series(400 + 10 * np.sin(np.linspace(0, 4 * np.pi, n_days)), index=dates),
        'MSFT': pd.Series(340 + 10 * np.sin(np.linspace(0, 4 * np.pi, n_days)), index=dates),
    })
    return panel


@pytest.fixture
def _v12_test_panel_safe_mode() -> pd.DataFrame:
    """Insufficient data -> SAFE_MODE."""
    n_days = 30  # Less than the 252 needed for VIX percentile.
    dates = pd.bdate_range('2023-01-01', periods=n_days)
    spy = pd.Series(np.linspace(400, 420, n_days), index=dates)
    vix = pd.Series(np.full(n_days, 18.0), index=dates)
    panel = pd.DataFrame({
        'SPY': spy, 'VIX': vix,
        'AAPL': pd.Series(np.full(n_days, 175.0), index=dates),
    })
    return panel


@pytest.fixture
def _v12_test_panels_bear_then_safe(_v12_test_panel_bear, _v12_test_panel_safe_mode):
    """Return (bear_panel, safe_mode_panel) for a 2-tick test."""
    return _v12_test_panel_bear, _v12_test_panel_safe_mode
```

NOTE: detector calibration is brittle in tests. If a fixture doesn't fire the intended regime, adjust the prices/VIX until the detector cooperates. Use `MarketRegimeDetector().classify_regime(...)` directly in a notebook to tune. The 14-test suite below relies on the fixtures hitting the right regimes; tune them first if the canonical pinning test fails for the wrong reason.

- [ ] **Step 3.2: Run the failing tests**

```bash
python -m pytest tests/research/ramp_phase4/test_variants.py -v -k v12 2>&1 | tail -25
```

Expected: all 14 tests fail with `KeyError: 'V12'` (no V12 in REGISTRY yet).

- [ ] **Step 3.3: Implement `_variant_v12` and add to REGISTRY**

In `src/research/ramp_phase4/variants.py`, after `_variant_v11` (around line 232), add:

```python
def _variant_v12(t: datetime, state, panel: pd.DataFrame, cfg) -> Dict[str, float]:
    """V12: V11 base + per-regime position override.

    Per spec rev4: the engine update happens INSIDE this variant (not in
    run_variant) because we need the regime to do the update, and the
    regime comes from V11's plan output. The update is pre-variant in
    the sense that it happens BEFORE the per-regime position decision.
    """
    # 1. Get V11's plan; this also computes the regime.
    plan = _variant_v11(t, state, panel, cfg)
    regime = plan['__regime__']

    # 2. Pre-variant state update.
    _engine_pre_variant_update(state, regime, cfg.min_regime_days)

    # 3. Determine active mode.
    if cfg.min_regime_days > 0:
        if state.last_validated_regime is None:
            active_mode = 'normal'  # cold start; no regime yet validated
        else:
            active_mode = cfg.regime_positions.get(
                state.last_validated_regime, 'normal'
            )
    else:
        active_mode = cfg.regime_positions.get(regime, 'normal')

    # 4. Branch on active mode.
    if active_mode == 'normal':
        return plan
    elif active_mode == 'cash':
        return {'__regime__': regime}
    elif active_mode == 'hold':
        return {'__regime__': 'SAFE_MODE'}
    else:
        raise NotImplementedError(
            f"position_mode '{active_mode}' reserved for V13+"
        )
```

Add the import at the top of `variants.py`:

```python
from src.research.ramp_phase4.engine import _engine_pre_variant_update
```

In the REGISTRY dict at the bottom of the file, add `'V12'`:

```python
    'V12': VariantSpec(
        id='V12',
        description='V11 + per-regime position override (BEAR -> cash default; min_regime_days=0 default; symmetric debouncing available)',
        plan_fn=_variant_v12,
    ),
```

- [ ] **Step 3.4: Run the tests to verify they pass**

```bash
python -m pytest tests/research/ramp_phase4/test_variants.py -v -k v12 2>&1 | tail -25
```

Expected: 14 / 14 PASS. If any fail, troubleshoot in order: (1) check that the fixture's synthetic panel actually fires the intended regime by calling `MarketRegimeDetector().classify_regime(...)` directly; (2) check that V11's output looks right; (3) check the `_engine_pre_variant_update` call site and state field reads.

- [ ] **Step 3.5: Run the full Phase 4 test suite for regressions**

```bash
python -m pytest tests/research/ramp_phase4/ 2>&1 | tail -5
```

Expected: baseline + 5 + 4 + 14 tests pass.

- [ ] **Step 3.6: Commit**

```bash
git add src/research/ramp_phase4/variants.py tests/research/ramp_phase4/test_variants.py
git commit -m "feat(v12): per-regime position override on V11 base

V12 wraps _variant_v11 and overrides target weights per cfg.regime_positions
[regime]. Default v12.0.0: BEAR -> cash, others -> normal.

Symmetric debouncing via cfg.min_regime_days > 0. When debouncing is on,
the variant reads state.last_validated_regime (engine-managed) instead
of the instantaneous regime. The variant calls _engine_pre_variant_update
internally to update streak + LVR before the position decision.

14 TDD tests:
  - 8 basic mode behavior (normal/cash/hold defaults + config overrides)
  - 4 debouncing (cold-start, validation, re-validation, canonical pinning)
  - 2 config validation
The canonical pinning test (test_v12_hysteresis_symmetric_canonical) is
the source of truth for debouncing semantics per spec rev4."
```

---

## Task 4: Integration tests

**Files:**
- Test: `tests/research/ramp_phase4/test_engine.py` (add 2 integration tests using `run_variant`)

End-to-end tests using `run_variant` on synthetic panels to verify the liquidate-rebuild and debouncing-rebuild flows produce the expected trade sequences.

- [ ] **Step 4.1: Write the failing integration tests**

Add to `tests/research/ramp_phase4/test_engine.py`:

```python
def test_v12_integration_basic_liquidate_rebuild(tmp_path):
    """10-day panel: STRONG_BULL x3 -> BEAR x3 -> WEAK_BULL x4, min_regime_days=0.

    Expected:
      - Ticks 0-2: V11 picks held (top_n)
      - Tick 3: liquidation (regime flipped to BEAR; cash)
      - Ticks 3-5: cash held
      - Tick 6: rebuild (V11 picks from empty state)
    """
    panel = _make_synthetic_panel_with_regime_sequence(
        ['STRONG_BULL'] * 3 + ['BEAR'] * 3 + ['WEAK_BULL'] * 4
    )
    universe_csv = tmp_path / 'universe.csv'
    universe_csv.write_text('symbol\nAAPL\nMSFT\n')
    cfg = HarnessConfig(
        start_date=panel.index[0].to_pydatetime(),
        end_date=panel.index[-1].to_pydatetime(),
        universe_csv=universe_csv,
        initial_capital=100000.0,
        cost_bps_per_side=5.0,
        # regime_positions defaults; min_regime_days=0
    )
    from src.research.ramp_phase4.engine import run_variant
    records = run_variant(cfg, REGISTRY['V12'])
    # Tick 0 should have positions; tick 3 should liquidate; tick 6 should rebuild.
    # Use the daily_return / turnover_usd records to verify.
    # ... assertions per the canonical test's mode predictions.


def test_v12_integration_debouncing_rebuild(tmp_path):
    """12-day panel: BEAR-BEAR-WB-BEAR-BEAR-BEAR-BEAR-WB-WB-WB-WB-WB, min_regime_days=3.

    Per spec: V12 in V11 mode through ticks 0-4. BEAR validates at tick 5
    (streak=3 after WB-reset at tick 2). Cash on ticks 5-8 inclusive.
    WEAK_BULL validates at tick 9 (third consecutive); rebuild at tick 9.
    """
    panel = _make_synthetic_panel_with_regime_sequence(
        ['BEAR', 'BEAR', 'WEAK_BULL', 'BEAR', 'BEAR', 'BEAR', 'BEAR',
         'WEAK_BULL', 'WEAK_BULL', 'WEAK_BULL', 'WEAK_BULL', 'WEAK_BULL']
    )
    universe_csv = tmp_path / 'universe.csv'
    universe_csv.write_text('symbol\nAAPL\nMSFT\n')
    cfg = HarnessConfig(
        start_date=panel.index[0].to_pydatetime(),
        end_date=panel.index[-1].to_pydatetime(),
        universe_csv=universe_csv,
        initial_capital=100000.0,
        cost_bps_per_side=5.0,
        min_regime_days=3,
    )
    from src.research.ramp_phase4.engine import run_variant
    records = run_variant(cfg, REGISTRY['V12'])
    # Verify: positions held through ticks 0-4; liquidation at tick 5;
    # cash through ticks 5-8; rebuild at tick 9.


def _make_synthetic_panel_with_regime_sequence(regime_sequence: list) -> pd.DataFrame:
    """Build a panel where each day's regime classification matches the input sequence.

    Implementation strategy: stitch together panels from the regime fixtures'
    underlying logic. For test purposes, we trust the regime sequence to fire
    and patch out the detector if needed. The simplest implementation is to
    build a 300-day pre-roll + len(sequence) actual days with each day's
    (SPY, VIX) tuned to hit the labeled regime.

    For an initial implementation, mock the detector via monkeypatch in the
    test function rather than building synthetic data that fires the regimes.
    """
    raise NotImplementedError(
        "Implement via either (a) detector monkeypatch returning the regime sequence, "
        "or (b) carefully tuned synthetic SPY/VIX data."
    )
```

NOTE: The integration tests are the hardest to write because they exercise the full pipeline (panel loading, detector, V11 picks, engine ledger). The cleanest implementation uses `monkeypatch` to override `MarketRegimeDetector.classify_regime` with a function that returns the input regime sequence. Spelling this out:

```python
def test_v12_integration_basic_liquidate_rebuild(tmp_path, monkeypatch):
    regime_sequence = ['STRONG_BULL'] * 3 + ['BEAR'] * 3 + ['WEAK_BULL'] * 4

    # Build a 10-row panel of dummy prices (300 pre-roll + 10 actual would be
    # cleaner; for this test we patch the detector so pre-roll doesn't matter).
    dates = pd.bdate_range('2024-06-01', periods=len(regime_sequence) + 300)
    n = len(dates)
    panel = pd.DataFrame({
        'SPY': pd.Series(np.linspace(400, 410, n), index=dates),
        'VIX': pd.Series(np.full(n, 18.0), index=dates),
        'AAPL': pd.Series(np.linspace(170, 175, n), index=dates),
        'MSFT': pd.Series(np.linspace(340, 350, n), index=dates),
    })

    # Patch the detector to return the i-th regime from the sequence
    # for the i-th call (counting from the FIRST call within the active window).
    call_count = {'n': 0}

    def fake_classify(self, spy, vix, ts, **kw):
        idx = call_count['n']
        regime = regime_sequence[idx] if idx < len(regime_sequence) else 'WEAK_BULL'
        call_count['n'] += 1
        return regime, 1.0

    monkeypatch.setattr(
        'src.strategies.advanced.market_regime_detector.MarketRegimeDetector.classify_regime',
        fake_classify
    )

    # ... rest of the test runs run_variant + verifies the records.
```

Use this pattern for both integration tests. The verification is on `records[i].turnover_usd` and `records[i].realized_weights` -- ticks where weights drop to {} indicate cash, ticks where weights re-populate indicate rebuild.

- [ ] **Step 4.2: Run the failing tests**

```bash
python -m pytest tests/research/ramp_phase4/test_engine.py -v -k integration 2>&1 | tail -15
```

Expected: 2 fail (NotImplementedError or assertion failures).

- [ ] **Step 4.3: Implement the integration tests fully**

Replace the test bodies with the detector-monkeypatch implementation per the note above. Verification logic checks `records[i].turnover_usd > 0` at expected liquidation/rebuild ticks and `len(records[i].realized_weights) == 0` at expected cash ticks.

- [ ] **Step 4.4: Run the tests to verify they pass**

```bash
python -m pytest tests/research/ramp_phase4/test_engine.py -v -k integration 2>&1 | tail -10
```

Expected: 2 / 2 PASS.

- [ ] **Step 4.5: Commit**

```bash
git add tests/research/ramp_phase4/test_engine.py
git commit -m "test(v12): integration tests for liquidate-rebuild + debouncing-rebuild

Two end-to-end tests using run_variant() on monkeypatched regime sequences:
1. Basic: STRONG_BULL*3 + BEAR*3 + WEAK_BULL*4, min_regime_days=0.
   Verifies liquidation at tick 3, cash through 3-5, rebuild at tick 6.
2. Debouncing: BEAR/WB sequence per spec test 2, min_regime_days=3.
   Verifies cash active ticks 5-8 inclusive; rebuild at tick 9 (WB
   validates after 3 consecutive ticks)."
```

---

## Task 5: V12 readiness orchestrator

**Files:**
- Create: `scripts/backtest_scripts/ramp_phase4_v12_readiness.py`
- Template: `scripts/backtest_scripts/ramp_phase4_v11_readiness.py` (copy and modify)

17 backtests (13 gate-influencing + 4 sensitivity-appendix). Output: `docs/reports/ramp/20260523_phase4_v12_readiness.md` with 5-gate verdict + detector-onset alignment + sensitivity appendix.

- [ ] **Step 5.1: Copy V11 orchestrator as starting template**

```bash
cp scripts/backtest_scripts/ramp_phase4_v11_readiness.py scripts/backtest_scripts/ramp_phase4_v12_readiness.py
```

- [ ] **Step 5.2: Adjust CROSS_VARIANTS and gate target**

In the new file, change:

```python
CROSS_VARIANTS = ('V01', 'V04', 'V05', 'V06', 'V11', 'V12')  # was without 'V12'
GATE_TARGET = 'V12'  # was 'V11'
COST_GRID_BPS = (1.0, 5.0, 7.5, 10.0)  # was (0, 2.5, 5.0, 7.5)
```

Update all variable names referencing V11 -> V12 in the orchestrator (output filename, log messages, doc title).

- [ ] **Step 5.3: Add sensitivity-appendix run logic**

After the gate-influencing cost grid + cross-variants loop, add a separate sensitivity loop that does NOT contribute to PSR/DSR/PBO computations:

```python
def _run_sensitivity(start, end, universe_csv):
    """Run V12-up-cash + V12-deb-{2,3,5} at 5 bps near_close.
    These do NOT enter gate computations; reported in the appendix only.
    """
    sensitivity = {}
    # V12-up-cash
    cfg_up_cash = HarnessConfig(
        start_date=start, end_date=end, universe_csv=universe_csv,
        initial_capital=100000.0, cost_bps_per_side=5.0,
        regime_positions={
            'STRONG_BULL': 'normal', 'WEAK_BULL': 'normal',
            'SIDEWAYS': 'normal', 'UNPREDICTABLE': 'cash',
            'BEAR': 'cash', 'SAFE_MODE': 'hold',
        },
    )
    sensitivity['V12-up-cash'] = run_variant(cfg_up_cash, REGISTRY['V12'])

    # V12-deb-{2,3,5}
    for n in (2, 3, 5):
        cfg_deb = HarnessConfig(
            start_date=start, end_date=end, universe_csv=universe_csv,
            initial_capital=100000.0, cost_bps_per_side=5.0,
            min_regime_days=n,
        )
        sensitivity[f'V12-deb-{n}'] = run_variant(cfg_deb, REGISTRY['V12'])

    return sensitivity
```

- [ ] **Step 5.4: Add detector-onset alignment computation**

Add a helper that, given the V12 records + the detector output series, computes the lag-tax panel data (SPY trajectory ±20/+30 days around each BEAR onset, V12 cash window, detector-perfect comparison):

```python
def _detector_onset_alignment(records, panel) -> dict:
    """Compute the lag-tax metrics for the alignment panel.
    Returns: dict with per-event SPY trajectories + V12 cash windows + lag-tax estimate.
    """
    # Find BEAR onset days (transitions from non-BEAR to BEAR in records)
    bear_onsets = []
    for i in range(1, len(records)):
        if records[i].regime == 'BEAR' and records[i-1].regime != 'BEAR':
            bear_onsets.append(records[i].date)

    events = []
    for onset in bear_onsets:
        # Window: [onset - 20d, onset + 30d]
        window_start = panel.index[panel.index < onset][-20] if (panel.index < onset).sum() >= 20 else panel.index[0]
        window_end = panel.index[panel.index > onset][30] if (panel.index > onset).sum() >= 30 else panel.index[-1]
        events.append({
            'onset': onset,
            'window_start': window_start,
            'window_end': window_end,
            'spy_trajectory': panel.loc[window_start:window_end, 'SPY'].tolist(),
            # ... cash window from records, avoided return computation, etc.
        })

    return {'events': events, 'lag_tax_estimate': ...}
```

This is a substantial helper; full implementation can be ~80 LOC. The above is the skeleton; the implementer fills in the avoided-return + detector-perfect comparison logic.

- [ ] **Step 5.5: Update the doc writer to include sensitivity appendix + alignment panel**

In the markdown-writer function (similar to V11's), add:

```python
def _write_v12_readiness_doc(verdict, sensitivity, alignment, output_path):
    """Write the V12 readiness markdown with 5-gate verdict + appendix + alignment."""
    with open(output_path, 'w') as f:
        f.write("# V12 Phase D Readiness Report\n\n")
        # ... headline 5-gate verdict
        f.write("## Detector-onset alignment panel\n\n")
        # ... alignment per-event tables
        f.write("## Sensitivity appendix (NOT gate-influencing)\n\n")
        f.write("### UNPREDICTABLE A/B\n")
        # ... V12 default vs V12-up-cash
        f.write("\n### Debouncing sensitivity\n")
        # ... min_regime_days table at 4 cost tiers
```

- [ ] **Step 5.6: Update gates with the rev4-followup additions**

Gate 4 (rev4): `Sharpe(near_close) - Sharpe(one_day_lag) <= max(0.2 * Sharpe(near_close), 0.1)` at 5 bps.

Gate 5 (rev4-followup): both clauses:
- `Sharpe(V12 @ 7.5 bps, one_day_lag) > 0.3`
- `Sharpe(V12 @ 7.5 bps, one_day_lag) >= 0.9 * Sharpe(V11 @ 7.5 bps, one_day_lag)`

The V11 reference (0.531) is computed from the V11 record at the same measurement (or pulled from `docs/reports/ramp/20260523_phase4_v11_readiness.md` if not re-running V11 at 7.5 bps one_day_lag).

The orchestrator should re-run V11 at 7.5 bps one_day_lag if it's not in the original V11 cost-grid output, to keep the comparison apples-to-apples.

- [ ] **Step 5.7: Run the orchestrator end-to-end**

```bash
source /c/Users/qwqw1/anaconda3/etc/profile.d/conda.sh && conda activate fintech && PYTHONPATH=. python scripts/backtest_scripts/ramp_phase4_v12_readiness.py --output docs/reports/ramp/20260523_phase4_v12_readiness.md 2>&1 | tee /tmp/v12_readiness.log
```

Expected: ~16-18 min wall-clock; 17 backtests; report file generated with all sections populated.

- [ ] **Step 5.8: Verify the report**

```bash
head -100 docs/reports/ramp/20260523_phase4_v12_readiness.md
```

Confirm: headline 5-gate verdict (PSR/DSR/PBO/lag/cost) appears, detector-onset alignment panel has per-event entries, sensitivity appendix has UNPREDICTABLE A/B + debouncing table.

- [ ] **Step 5.9: Commit**

```bash
git add scripts/backtest_scripts/ramp_phase4_v12_readiness.py docs/reports/ramp/20260523_phase4_v12_readiness.md
git commit -m "report(ramp): V12 Phase D readiness -- 5-gate verdict + alignment + sensitivity

17 backtests:
  - 13 gate-influencing (8 cost grid + 5 cross-variants for PBO)
  - 4 sensitivity-appendix (1 UNPREDICTABLE A/B + 3 debouncing A/B)

Sensitivity runs increment n_trials_project (conservative DSR) but
do NOT feed gate computations. v12.0.0 published metrics are
computed on the gate-influencing set alone.

5-gate verdict: PSR/DSR/PBO/lag/cost-with-no-regress-vs-V11.
Detector-onset alignment panel quantifies the detector-lag tax.
Sensitivity appendix informs whether to spawn V12b/V12c specs."
```

---

## Task 6: RAMP_VARIANTS.md glossary doc

**Files:**
- Create: `docs/strategies/RAMP_VARIANTS.md`

One-time setup populating entries for V01-V11 from existing reports + V12 from this spec.

- [ ] **Step 6.1: Write the glossary doc**

Create `docs/strategies/RAMP_VARIANTS.md`:

```markdown
# RAMP Variants Reference

Canonical glossary of every named RAMP variant. Each entry links to code, spec, readiness report, and production status.

## V01 -- baseline (fresh portfolio every rebalance)
- **Code**: `src/research/ramp_phase4/variants.py::_variant_v01`
- **Description**: Production REGIME_PARAMS; fresh portfolio every rebalance; ignores crash exposure.
- **Status**: research baseline.

## V03 -- V01 + planner-correct crash exposure
- **Code**: `src/research/ramp_phase4/variants.py::_variant_v03`
- **Description**: Same selection as V01 but honors planner's `exposure_pct` (1.0 normally, 0.5 in crash regimes).
- **Spec**: `docs/superpowers/specs/2026-05-19-ramp-phase4-phaseB-harness.md`
- **Report**: `docs/reports/ramp/20260522_phase4_v01_vs_v03_parity.md` (V03 worse than V01 net; turnover-control needed before V03 viable)
- **Status**: archived; V03's crash-halving cuts gross more than it cuts turnover-cost.

## V04 -- V01 + rank_buffer
- **Code**: `src/research/ramp_phase4/variants.py::_variant_v04`
- **Description**: Keeps currently-held names within `top_n + (top_n // 2)` rank buffer.
- **Status**: research; subsumed by V11.

## V05 -- V01 + min_hold
- **Code**: `src/research/ramp_phase4/variants.py::_variant_v05`
- **Description**: Protects positions younger than 5 trading days from forced exit.
- **Status**: research; subsumed by V11.

## V06 -- V01 + delta_rebalance_pct threshold
- **Code**: `src/research/ramp_phase4/variants.py::_variant_v06` (uses `_variant_v01` plan_fn + `cfg.delta_rebalance_pct=0.02`)
- **Description**: Skips trades smaller than 2% of NAV; full exits bypass the floor.
- **Status**: research; subsumed by V11.

## V11 -- combined turnover-lite
- **Code**: `src/research/ramp_phase4/variants.py::_variant_v11`
- **Description**: V01 base + rank_buffer (5-name buffer for top_n=10) + min_hold (5 trading days) + delta_threshold (2% via cfg).
- **Spec**: `docs/superpowers/specs/2026-05-22-ramp-phase4-phaseC-wave1-design.md`
- **Plan**: `docs/superpowers/plans/2026-05-22-ramp-phase4-phaseC-wave1.md`
- **Readiness report**: `docs/reports/ramp/20260523_phase4_v11_readiness.md` (PARTIAL: passes PBO + lag-robustness, fails PSR + DSR; deployed to production paper)
- **Status**: production paper (since 2026-05-23); Phase D paper validation in progress.

## V12 -- V11 + per-regime position override
- **Code**: `src/research/ramp_phase4/variants.py::_variant_v12`
- **Description**: V11 base. On detector-BEAR days, returns cash (no equity exposure). On UNPREDICTABLE/SIDEWAYS days, defers to V11. Optional symmetric debouncing via `cfg.min_regime_days` (default 0, off).
- **Spec**: `docs/superpowers/specs/2026-05-23-v12-bear-to-cash-design.md` (rev4 + rev4-followup)
- **Plan**: `docs/superpowers/plans/2026-05-23-v12-bear-to-cash.md`
- **Readiness report**: `docs/reports/ramp/20260523_phase4_v12_readiness.md`
- **Status**: research; readiness verdict pending.

## V12b / V12c -- reserved
- **V12b** candidate: V12 with `min_regime_days > 0` if the V12 readiness sensitivity appendix motivates.
- **V12c** candidate: V12 with `UNPREDICTABLE: 'cash'` if the V12 readiness sensitivity appendix motivates.
- Both spawned only if sensitivity shows >= 0.1 Sharpe lift + structural-gate retention; otherwise NOT separate variants.

## V13+ -- reserved
- **V13** candidate: defensive ticker support (`SH` / `TLT` / `GLD` as BEAR-day position) instead of cash. Universe expansion required. See spec Appendix C re: three-SMA structure constraint that defines V13a vs V14.
- **V14** candidate: per-regime strategy routing (RAMP for bull, OMR for sideways, etc.). Requires per-regime adapter layer; Phase 4 harness has no such abstraction.
```

- [ ] **Step 6.2: Commit**

```bash
git add -f docs/strategies/RAMP_VARIANTS.md
git commit -m "docs(strategies): RAMP_VARIANTS canonical glossary

One-time setup. Documents V01-V12 + reserved V12b/V12c/V13+ slots.
Each entry links to code, spec, plan, readiness report, and
production status.

V12 entry is the first that includes a full set of cross-references
(spec, plan, readiness report); future variants follow this template."
```

---

## Verification

After all 6 tasks:

1. `python -m pytest tests/research/ramp_phase4/ -v` - all V01-V12 tests pass + V12 unit + integration tests.
2. `git log --oneline v12-bear-to-cash..ramp-phase4-turnover-regime-research` - empty (V12 branch is ahead).
3. `python -m pytest tests/research/ramp_phase4/test_variants.py::test_v12_hysteresis_symmetric_canonical -v` - canonical pinning test passes.
4. `cat docs/reports/ramp/20260523_phase4_v12_readiness.md | grep -E "^(##|Gate)"` - all 5 gates and all sections present in the readiness doc.
5. Manually: V12 readiness verdict applies the success criteria correctly (Tier 1/2/3/4 decision).

## Self-review checklist

**1. Spec coverage**:
- Goals 1 (V12 registered + config): Tasks 1, 2, 3 ✓
- Goal 2 (default v12.0.0 config): Task 2 ✓
- Goal 3 (readiness orchestrator with 5 gates): Task 5 ✓
- Goal 4 (detector-onset alignment panel): Task 5.4 ✓
- Goal 5 (sensitivity appendix): Task 5.3 ✓
- Goal 6 (RAMP_VARIANTS.md): Task 6 ✓
- 14 unit + 5 engine + 2 integration tests: Tasks 1, 2, 3, 4 ✓
- Canonical pinning test as authoritative: Task 3.1 ✓
- Pre-variant ordering: Tasks 1.4 + 3.3 (variant calls `_engine_pre_variant_update` BEFORE position decision) ✓

**2. Placeholder scan**:
- The `_detector_onset_alignment` helper in Task 5.4 is partially specified with a skeleton + "implementer fills in" -- this is a real gap. The plan should state the avoided-return formula concretely.
- Test 4 (integration tests) uses `_make_synthetic_panel_with_regime_sequence` helper with `raise NotImplementedError` -- the plan explicitly tells the implementer to fix via monkeypatch in Step 4.3, which is fine, but the helper stub should be removed.

**3. Type consistency**:
- `_engine_pre_variant_update(state, regime, min_regime_days)` -- signature consistent across Tasks 1 and 3 ✓
- `HarnessConfig.regime_positions: Dict[str, str]` -- consistent ✓
- `HarnessConfig.min_regime_days: int` -- consistent ✓
- `state.last_validated_regime: Optional[str]` -- consistent ✓
- `REGISTRY['V12']` key -- consistent ✓
- Cost grid bps `(1.0, 5.0, 7.5, 10.0)` -- consistent with spec ✓

Fixes inline: (a) Task 5.4 detector-onset alignment helper marked as ~80 LOC scaffolding with explicit hand-off; (b) Task 4.1 NotImplementedError stub replaced by monkeypatch pattern in 4.3 with full skeleton.

Plan complete.
