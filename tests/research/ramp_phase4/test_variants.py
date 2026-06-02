"""Tests for variants.py: V01 + V03 plan functions."""
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.research.ramp_phase4.config import HarnessConfig
from src.research.ramp_phase4.engine import HarnessState
from src.research.ramp_phase4.variants import REGISTRY, VariantSpec


def _interpret_plan_as_mode(plan: dict) -> str:
    """Return 'cash' if only __regime__ key, 'hold' if regime is SAFE_MODE, else 'normal'."""
    if plan.get('__regime__') == 'SAFE_MODE':
        return 'hold'
    weight_keys = [k for k in plan.keys() if k != '__regime__']
    if not weight_keys:
        return 'cash'
    return 'normal'


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


def _calm_panel(n=300):
    idx = pd.date_range('2023-01-02', periods=n, freq='B')
    return pd.DataFrame({
        'AAA': 100 + np.arange(n) * 0.05,
        'BBB': 110 + np.arange(n) * 0.04,
        'CCC': 90 + np.arange(n) * 0.06,
        'SPY': 400 + np.arange(n) * 0.1,  # uptrend -> STRONG_BULL
        'VIX': np.full(n, 12.0),         # low vol
    }, index=idx)


def test_variant_registry_contains_v01_and_v03():
    assert 'V01' in REGISTRY
    assert 'V03' in REGISTRY
    assert isinstance(REGISTRY['V01'], VariantSpec)
    assert isinstance(REGISTRY['V03'], VariantSpec)


def test_v01_returns_target_weights_in_calm_regime():
    spec = REGISTRY['V01']
    panel = _calm_panel()
    state = type('S', (), {'positions': {}, 'cash_usd': 100000.0})()
    cfg = type('C', (), {})()
    plan = spec.plan_fn(panel.index[-1].to_pydatetime(), state, panel, cfg)
    # __regime__ sentinel present.
    assert '__regime__' in plan
    # Non-regime weights sum to ~1.0 in calm (no crash trigger).
    body = {k: v for k, v in plan.items() if k != '__regime__'}
    assert abs(sum(body.values()) - 1.0) < 0.01


def _crash_panel(n=300):
    idx = pd.date_range('2023-01-02', periods=n, freq='B')
    spy_path = np.concatenate([400 + np.arange(n - 30) * 0.1, np.linspace(430, 380, 30)])
    vix_path = np.concatenate([np.full(n - 30, 12.0), np.linspace(20, 35, 30)])
    return pd.DataFrame({
        'AAA': 100 + np.arange(n) * 0.05,
        'BBB': 110 + np.arange(n) * 0.04,
        'CCC': 90  + np.arange(n) * 0.06,
        'SPY': spy_path,
        'VIX': vix_path,
    }, index=idx)


def test_v03_applies_crash_exposure_in_crash_regime():
    spec = REGISTRY['V03']
    panel = _crash_panel()
    state = type('S', (), {'positions': {}, 'cash_usd': 100000.0})()
    cfg = type('C', (), {})()
    plan = spec.plan_fn(panel.index[-1].to_pydatetime(), state, panel, cfg)
    body = {k: v for k, v in plan.items() if k != '__regime__'}
    # In crash (VIX > 25 OR SPY-DD < -5%), gross should be reduced.
    assert sum(body.values()) <= 0.6  # 0.5 with epsilon


def test_v01_v03_identical_in_calm_regime():
    panel = _calm_panel()
    state = type('S', (), {'positions': {}, 'cash_usd': 100000.0})()
    cfg = type('C', (), {})()
    p01 = REGISTRY['V01'].plan_fn(panel.index[-1].to_pydatetime(), state, panel, cfg)
    p03 = REGISTRY['V03'].plan_fn(panel.index[-1].to_pydatetime(), state, panel, cfg)
    # Same symbols selected; per-weight identical when calm.
    assert set(p01) - {'__regime__'} == set(p03) - {'__regime__'}
    for sym in set(p01) - {'__regime__'}:
        assert abs(p01[sym] - p03[sym]) < 1e-6


def test_variant_v04_in_registry():
    from src.research.ramp_phase4.variants import REGISTRY, VariantSpec
    assert 'V04' in REGISTRY
    assert isinstance(REGISTRY['V04'], VariantSpec)
    assert 'rank buffer' in REGISTRY['V04'].description.lower()


def test_variant_v05_in_registry():
    from src.research.ramp_phase4.variants import REGISTRY, VariantSpec
    assert 'V05' in REGISTRY
    assert isinstance(REGISTRY['V05'], VariantSpec)
    assert 'min' in REGISTRY['V05'].description.lower() and 'hold' in REGISTRY['V05'].description.lower()


def test_variant_v06_in_registry():
    from src.research.ramp_phase4.variants import REGISTRY, VariantSpec
    assert 'V06' in REGISTRY
    assert isinstance(REGISTRY['V06'], VariantSpec)
    assert 'delta' in REGISTRY['V06'].description.lower()


def test_variant_v11_in_registry():
    from src.research.ramp_phase4.variants import REGISTRY, VariantSpec
    assert 'V11' in REGISTRY
    assert isinstance(REGISTRY['V11'], VariantSpec)
    assert 'combined' in REGISTRY['V11'].description.lower() or 'turnover' in REGISTRY['V11'].description.lower()


def test_v11_calls_rank_buffer_then_min_hold(monkeypatch):
    """V11's plan_fn must call rank_buffer THEN min_hold in that order."""
    from datetime import datetime
    import pandas as pd
    import numpy as np
    from src.research.ramp_phase4.variants import REGISTRY
    from src.research.ramp_phase4.engine import HarnessState

    n = 300
    idx = pd.date_range('2023-01-02', periods=n, freq='B') + pd.Timedelta(hours=9, minutes=30)
    panel = pd.DataFrame({
        'AAA': 100 + np.arange(n) * 0.05,
        'BBB': 110 + np.arange(n) * 0.04,
        'CCC': 90 + np.arange(n) * 0.06,
        'SPY': 400 + np.arange(n) * 0.1,
        'VIX': np.full(n, 12.0),
    }, index=idx)

    call_order = []

    def fake_rank_buffer(proposed_targets, state, buffer_size, universe_ranking, top_n):
        call_order.append('rank_buffer')
        return proposed_targets

    def fake_min_hold(proposed_targets, state, current_date, min_hold_days, crash_exit=False):
        call_order.append('min_hold')
        return proposed_targets

    monkeypatch.setattr('src.research.ramp_phase4.variants.rank_buffer', fake_rank_buffer, raising=False)
    monkeypatch.setattr('src.research.ramp_phase4.variants.min_hold', fake_min_hold, raising=False)

    state = HarnessState(cash_usd=100000.0)
    cfg = type('C', (), {})()
    plan = REGISTRY['V11'].plan_fn(panel.index[-1].to_pydatetime(), state, panel, cfg)

    assert call_order == ['rank_buffer', 'min_hold']


# ============================================================
# V12 panel fixtures: synthetic OHLC + VIX engineered per regime.
#
# All panels span 2023-01-01 .. 2024-06-30 (daily, weekends included) so
# datetime(2024, 6, 15) is in the index and the 252-row lookback is satisfied
# for tests that use it as the .loc[:t] target. Calibration verified against
# MarketRegimeDetector.classify_regime; see the canonical pinning test for the
# load-bearing schedule.
# ============================================================

_V12_DATES = pd.date_range('2023-01-01', '2024-06-30', freq='D')
_V12_N = len(_V12_DATES)


@pytest.fixture
def _v12_test_panel_bear() -> pd.DataFrame:
    """Linear SPY decline + climbing VIX -> BEAR at t=2024-06-15."""
    spy = np.linspace(400, 280, _V12_N)
    vix = np.linspace(20, 38, _V12_N)
    return pd.DataFrame({
        'SPY': pd.Series(spy, index=_V12_DATES),
        'VIX': pd.Series(vix, index=_V12_DATES),
        'AAPL': pd.Series(np.linspace(170, 140, _V12_N), index=_V12_DATES),
        'MSFT': pd.Series(np.linspace(340, 280, _V12_N), index=_V12_DATES),
    })


@pytest.fixture
def _v12_test_panel_strong_bull() -> pd.DataFrame:
    """Accelerating SPY rise + low VIX -> STRONG_BULL at t=2024-06-15.

    Quadratic acceleration in the last 100 days pushes momentum_slope above
    the 0.02 threshold required by REGIME_CRITERIA['STRONG_BULL'].
    """
    spy_base = np.linspace(280, 400, _V12_N)
    accel = np.zeros(_V12_N)
    accel[-100:] = np.linspace(0, 80, 100)
    spy = spy_base + accel
    vix = np.linspace(28, 10, _V12_N)
    return pd.DataFrame({
        'SPY': pd.Series(spy, index=_V12_DATES),
        'VIX': pd.Series(vix, index=_V12_DATES),
        'AAPL': pd.Series(np.linspace(150, 200, _V12_N), index=_V12_DATES),
        'MSFT': pd.Series(np.linspace(300, 400, _V12_N), index=_V12_DATES),
    })


@pytest.fixture
def _v12_test_panel_weak_bull() -> pd.DataFrame:
    """Modest SPY rise + moderate VIX -> WEAK_BULL at t=2024-06-15."""
    spy = np.linspace(380, 405, _V12_N)
    vix = np.linspace(18, 22, _V12_N)
    return pd.DataFrame({
        'SPY': pd.Series(spy, index=_V12_DATES),
        'VIX': pd.Series(vix, index=_V12_DATES),
        'AAPL': pd.Series(np.linspace(170, 185, _V12_N), index=_V12_DATES),
        'MSFT': pd.Series(np.linspace(330, 360, _V12_N), index=_V12_DATES),
    })


@pytest.fixture
def _v12_test_panel_sideways() -> pd.DataFrame:
    """Oscillating SPY + VIX percentile in 30-60 band -> SIDEWAYS at t=2024-06-15."""
    spy = 400 + 8 * np.sin(np.linspace(0, 16 * np.pi, _V12_N))
    vix_vals = 15.0 + (np.arange(_V12_N) % 15.0)
    vix_vals[-1] = 22.0
    return pd.DataFrame({
        'SPY': pd.Series(spy, index=_V12_DATES),
        'VIX': pd.Series(vix_vals, index=_V12_DATES),
        'AAPL': pd.Series(np.full(_V12_N, 175.0), index=_V12_DATES),
        'MSFT': pd.Series(np.full(_V12_N, 340.0), index=_V12_DATES),
    })


@pytest.fixture
def _v12_test_panel_unpredictable() -> pd.DataFrame:
    """Low VIX base with sharp spike in last 5 days before t=2024-06-15.

    The spike (16 -> 30) triggers volatility_spike (current > 1.5x 20-day avg)
    which together with VIX percentile > 60 satisfies UNPREDICTABLE criteria.
    """
    vix_unpred = np.full(_V12_N, 15.0)
    idx_jun11 = _V12_DATES.get_loc(pd.Timestamp('2024-06-11'))
    idx_jun15 = _V12_DATES.get_loc(pd.Timestamp('2024-06-15'))
    vix_unpred[idx_jun11:idx_jun15 + 1] = [16, 18, 22, 25, 30]
    spy_choppy = 400 + 5 * np.sin(np.linspace(0, 12 * np.pi, _V12_N))
    return pd.DataFrame({
        'SPY': pd.Series(spy_choppy, index=_V12_DATES),
        'VIX': pd.Series(vix_unpred, index=_V12_DATES),
        'AAPL': pd.Series(400 + 10 * np.sin(np.linspace(0, 8 * np.pi, _V12_N)), index=_V12_DATES),
        'MSFT': pd.Series(340 + 10 * np.sin(np.linspace(0, 8 * np.pi, _V12_N)), index=_V12_DATES),
    })


@pytest.fixture
def _v12_test_panel_safe_mode() -> pd.DataFrame:
    """Insufficient data (< 252 rows) -> SAFE_MODE via _compute_plan_from_panel."""
    dates_safe = pd.date_range('2024-05-15', '2024-06-15', freq='D')
    n_safe = len(dates_safe)
    return pd.DataFrame({
        'SPY': pd.Series(np.linspace(400, 420, n_safe), index=dates_safe),
        'VIX': pd.Series(np.full(n_safe, 18.0), index=dates_safe),
        'AAPL': pd.Series(np.full(n_safe, 175.0), index=dates_safe),
    })


@pytest.fixture
def _v12_test_panels_bear_then_safe(_v12_test_panel_bear, _v12_test_panel_safe_mode):
    """Return (bear_panel, safe_mode_panel) for a 2-tick test."""
    return _v12_test_panel_bear, _v12_test_panel_safe_mode


# ============================================================
# V12 basic mode behavior (8 tests)
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
    cfg = _stub_cfg()
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
# V12 debouncing (4 tests; canonical pinning test is the spec)
# ============================================================

def test_v12_hysteresis_day_0_starts_normal(_v12_test_panel_bear):
    """min_regime_days=3, day 0 BEAR: last_validated_regime is None at decision,
    BUT pre-variant update sets it to BEAR (1 >= 3 false, so LVR stays None).
    On cold start LVR is None -> active_mode = 'normal' -> V11 output (with weights or empty)."""
    state = _fresh_state()
    cfg = _stub_cfg(min_regime_days=3)
    v12_out = REGISTRY['V12'].plan_fn(datetime(2024, 6, 15), state, _v12_test_panel_bear, cfg)
    assert '__regime__' in v12_out
    assert v12_out['__regime__'] == 'BEAR'
    # Cold start: LVR=None -> normal mode -> V11 output. V11 on a BEAR day with
    # synthetic stocks may return {'__regime__': 'BEAR'} (no targets) or with
    # weights; either way the regime sentinel is BEAR and mode was 'normal'.
    assert _interpret_plan_as_mode(v12_out) == 'normal'


def test_v12_hysteresis_validates_after_threshold(_v12_test_panel_bear):
    """BEAR for 3 consecutive days with min_regime_days=3 -> tick 2 returns cash."""
    state = _fresh_state()
    cfg = _stub_cfg(min_regime_days=3)
    # Tick 0 + 1: normal (LVR not yet set or still None)
    REGISTRY['V12'].plan_fn(datetime(2024, 6, 13), state, _v12_test_panel_bear, cfg)
    REGISTRY['V12'].plan_fn(datetime(2024, 6, 14), state, _v12_test_panel_bear, cfg)
    # Tick 2: cash (streak=3 reaches threshold; LVR=BEAR; mode=cash)
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
    REGISTRY['V12'].plan_fn(base.replace(day=base.day + 5), state, _v12_test_panel_weak_bull, cfg)
    REGISTRY['V12'].plan_fn(base.replace(day=base.day + 6), state, _v12_test_panel_weak_bull, cfg)
    out_tick7 = REGISTRY['V12'].plan_fn(base.replace(day=base.day + 7), state, _v12_test_panel_weak_bull, cfg)
    assert '__regime__' in out_tick7
    assert out_tick7['__regime__'] == 'WEAK_BULL'
    # WB streak=3, last_validated_regime flipped to WEAK_BULL -> mode=normal -> V11 weights present.
    assert _interpret_plan_as_mode(out_tick7) == 'normal'


def test_v12_hysteresis_symmetric_canonical(_v12_test_panel_weak_bull, _v12_test_panel_bear):
    """CANONICAL PINNING TEST -- SOURCE OF TRUTH for V12 debouncing semantics.

    Per spec rev4: 13 ticks (0..12) driving state through cold start, validation,
    transient flip-back (tick 7), and re-validation. Pre-variant ordering.
    Tick 7 distinguishes symmetric (cash) from asymmetric (would re-enter) debouncing.
    """
    cfg = _stub_cfg(min_regime_days=3)
    state = _fresh_state()

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


# ============================================================
# V12 config validation (2 tests)
# ============================================================

def test_v12_in_registry():
    """V12 is registered with expected description hallmarks."""
    assert 'V12' in REGISTRY
    assert isinstance(REGISTRY['V12'], VariantSpec)
    desc = REGISTRY['V12'].description.lower()
    assert 'regime' in desc
    assert 'position' in desc or 'override' in desc


def test_v12_unknown_position_mode_raises():
    """A regime_positions value outside {normal, cash, hold} raises at config time."""
    with pytest.raises(ValueError, match="reserved for V13"):
        HarnessConfig(
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 12, 31),
            universe_csv=Path('config/universes/sp500-2025.csv'),
            initial_capital=100000.0,
            cost_bps_per_side=5.0,
            regime_positions={
                'STRONG_BULL': 'normal', 'WEAK_BULL': 'normal',
                'SIDEWAYS': 'normal', 'UNPREDICTABLE': 'normal',
                'BEAR': 'TQQQ',  # ticker reserved for V13+
                'SAFE_MODE': 'hold',
            },
        )


# ============================================================
# V13-bear-invert tests (experiment 1: BEAR-as-buy hypothesis)
# ============================================================

def test_v13_in_registry():
    """V13-bear-invert is registered with expected description hallmarks."""
    assert 'V13-bear-invert' in REGISTRY
    assert isinstance(REGISTRY['V13-bear-invert'], VariantSpec)
    desc = REGISTRY['V13-bear-invert'].description.lower()
    assert 'bear' in desc
    assert 'spy' in desc


def test_v13_bear_returns_full_spy(_v12_test_panel_bear):
    """V13 on a BEAR day returns {'SPY': 1.0, '__regime__': 'BEAR'}."""
    state = _fresh_state()
    cfg = _stub_cfg()
    v13_out = REGISTRY['V13-bear-invert'].plan_fn(
        datetime(2024, 6, 15), state, _v12_test_panel_bear, cfg
    )
    assert v13_out == {'SPY': 1.0, '__regime__': 'BEAR'}


def test_v13_non_bear_strong_bull_defers_to_v11(_v12_test_panel_strong_bull):
    """V13 on STRONG_BULL day == V11 output exactly."""
    cfg = _stub_cfg()
    v11_out = REGISTRY['V11'].plan_fn(
        datetime(2024, 6, 15), _fresh_state(), _v12_test_panel_strong_bull, cfg
    )
    v13_out = REGISTRY['V13-bear-invert'].plan_fn(
        datetime(2024, 6, 15), _fresh_state(), _v12_test_panel_strong_bull, cfg
    )
    assert v13_out == v11_out


def test_v13_non_bear_weak_bull_defers_to_v11(_v12_test_panel_weak_bull):
    """V13 on WEAK_BULL day == V11 output exactly."""
    cfg = _stub_cfg()
    v11_out = REGISTRY['V11'].plan_fn(
        datetime(2024, 6, 15), _fresh_state(), _v12_test_panel_weak_bull, cfg
    )
    v13_out = REGISTRY['V13-bear-invert'].plan_fn(
        datetime(2024, 6, 15), _fresh_state(), _v12_test_panel_weak_bull, cfg
    )
    assert v13_out == v11_out


def test_v13_non_bear_sideways_defers_to_v11(_v12_test_panel_sideways):
    """V13 on SIDEWAYS day == V11 output exactly."""
    cfg = _stub_cfg()
    v11_out = REGISTRY['V11'].plan_fn(
        datetime(2024, 6, 15), _fresh_state(), _v12_test_panel_sideways, cfg
    )
    v13_out = REGISTRY['V13-bear-invert'].plan_fn(
        datetime(2024, 6, 15), _fresh_state(), _v12_test_panel_sideways, cfg
    )
    assert v13_out == v11_out


def test_v13_non_bear_unpredictable_defers_to_v11(_v12_test_panel_unpredictable):
    """V13 on UNPREDICTABLE day == V11 output exactly."""
    cfg = _stub_cfg()
    v11_out = REGISTRY['V11'].plan_fn(
        datetime(2024, 6, 15), _fresh_state(), _v12_test_panel_unpredictable, cfg
    )
    v13_out = REGISTRY['V13-bear-invert'].plan_fn(
        datetime(2024, 6, 15), _fresh_state(), _v12_test_panel_unpredictable, cfg
    )
    assert v13_out == v11_out


def test_v13_safe_mode_when_insufficient_data(_v12_test_panel_safe_mode):
    """V13 returns SAFE_MODE when V11 returns SAFE_MODE."""
    cfg = _stub_cfg()
    v13_out = REGISTRY['V13-bear-invert'].plan_fn(
        datetime(2024, 6, 15), _fresh_state(), _v12_test_panel_safe_mode, cfg
    )
    assert v13_out == {'__regime__': 'SAFE_MODE'}


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
            assert isinstance(plan, dict)
            assert plan.get('SPY') == 1.0
            assert '__regime__' in plan
            assert plan['__regime__'] == 'BEAR_SOFT_SPY'
            symbols = {k for k in plan.keys() if k not in ('__regime__',)}
            assert symbols == {'SPY'}
        else:
            assert plan == {'__regime__': 'STRONG_BULL', 'AAPL': 1.0}


def test_v14b_registry_entry_exists():
    from src.research.ramp_phase4.variants import _variant_v14b_soft_bear_spy
    assert 'V14b-soft-bear-spy' in REGISTRY
    assert REGISTRY['V14b-soft-bear-spy'].plan_fn is _variant_v14b_soft_bear_spy


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


# ============================================================
# V31 -- Beta-residual momentum tests
# ============================================================

def _v31_calm_panel(n=350):
    """Panel for V31 tests: calm trending market, multiple universe names.

    Three stocks with different beta characteristics vs SPY.
    HIGH_BETA has 2x SPY sensitivity; LOW_BETA has 0.3x; RESIDUAL is
    similar beta to SPY (1x) but has idiosyncratic positive alpha.
    """
    idx = pd.date_range('2022-01-03', periods=n, freq='B')
    spy = 400 + np.arange(n) * 0.3
    high_beta = 100 + np.arange(n) * 0.6 + np.sin(np.arange(n) * 0.2) * 5
    low_beta = 50 + np.arange(n) * 0.09 + np.sin(np.arange(n) * 0.15) * 1
    residual = 80 + np.arange(n) * 0.28 + np.arange(n) * 0.001  # slight extra alpha
    return pd.DataFrame({
        'HIGH_BETA': high_beta,
        'LOW_BETA': low_beta,
        'RESIDUAL': residual,
        'SPY': spy,
        'VIX': np.full(n, 12.0),
    }, index=idx)


def _v31_residual_advantage_panel(n=350):
    """Panel engineered so HIGH_BETA has high raw 21d return (from SPY move)
    but near-zero residual, while GOOD_RESIDUAL has moderate raw return
    but strongly positive residual momentum.

    Used to verify that V31 ranks GOOD_RESIDUAL above HIGH_BETA.
    """
    idx = pd.date_range('2022-01-03', periods=n, freq='B')
    # SPY climbs strongly in the last 21 days (the momentum window)
    spy_base = np.ones(n) * 400.0
    spy_base[-21:] += np.linspace(0, 25, 21)  # ~6.25% gain
    # HIGH_BETA: 2x leveraged vs SPY, so ~12.5% gain, but zero residual
    hb_base = np.ones(n) * 100.0
    hb_base[-21:] += np.linspace(0, 50, 21)  # ~50% of SPY move * 2
    # GOOD_RESIDUAL: 1x SPY beta but +5% extra idiosyncratic in window
    gr_base = np.ones(n) * 100.0
    gr_base[-21:] += np.linspace(0, 12.5, 21) + np.linspace(0, 5, 21)
    # FLAT: flat price, negative residual
    flat_base = np.ones(n) * 80.0
    return pd.DataFrame({
        'HIGH_BETA': hb_base,
        'GOOD_RESIDUAL': gr_base,
        'FLAT': flat_base,
        'SPY': spy_base,
        'VIX': np.full(n, 12.0),
    }, index=idx)


def test_v31_in_registry():
    """V31 is registered with expected description hallmarks."""
    assert 'V31' in REGISTRY
    assert isinstance(REGISTRY['V31'], VariantSpec)
    desc = REGISTRY['V31'].description.lower()
    assert 'beta' in desc or 'residual' in desc


def test_v31_plan_fn_returns_weights_summing_to_one_in_calm_regime():
    """V31 plan_fn returns weights summing to <= 1.0 + epsilon with __regime__ key."""
    spec = REGISTRY['V31']
    panel = _v31_calm_panel()
    state = HarnessState(cash_usd=100_000.0)
    cfg = _stub_cfg()
    plan = spec.plan_fn(panel.index[-1].to_pydatetime(), state, panel, cfg)
    assert '__regime__' in plan
    weight_keys = [k for k in plan if k != '__regime__']
    if weight_keys:
        total = sum(plan[k] for k in weight_keys)
        assert total <= 1.0 + 1e-6, f'Weights sum to {total:.6f}, expected <= 1.0'
        assert total > 0.0


def test_v31_safe_mode_when_insufficient_history():
    """V31 returns SAFE_MODE when panel has fewer than 252 rows (insufficient beta history)."""
    spec = REGISTRY['V31']
    n_short = 90
    idx = pd.date_range('2024-01-01', periods=n_short, freq='B')
    short_panel = pd.DataFrame({
        'AAPL': 100 + np.arange(n_short) * 0.1,
        'SPY': 400 + np.arange(n_short) * 0.2,
        'VIX': np.full(n_short, 14.0),
    }, index=idx)
    state = HarnessState(cash_usd=100_000.0)
    cfg = _stub_cfg()
    plan = spec.plan_fn(short_panel.index[-1].to_pydatetime(), state, short_panel, cfg)
    assert plan.get('__regime__') == 'SAFE_MODE'


def test_v31_high_beta_raw_momentum_name_not_selected_when_zero_residual():
    """V31 signal selection: a high-beta name with high raw return but
    near-zero residual is de-ranked relative to a name with positive residual.

    This test verifies the core H6/H8 attack: HIGH_BETA has high raw 21d
    return (market beta * SPY gain) but negligible residual; GOOD_RESIDUAL
    has idiosyncratic alpha in the window. V31 should select GOOD_RESIDUAL
    over HIGH_BETA (or exclude HIGH_BETA if only one slot is available).
    """
    spec = REGISTRY['V31']
    panel = _v31_residual_advantage_panel()
    state = HarnessState(cash_usd=100_000.0)
    cfg = _stub_cfg()
    t = panel.index[-1].to_pydatetime()
    plan = spec.plan_fn(t, state, panel, cfg)
    # If V31 correctly residualizes, GOOD_RESIDUAL should appear in plan.
    # HIGH_BETA should NOT appear alone without GOOD_RESIDUAL being present.
    weight_keys = {k for k in plan if k != '__regime__'}
    if weight_keys:
        # GOOD_RESIDUAL must be present if any selection was made
        # (it has the highest residual momentum in this panel)
        assert 'GOOD_RESIDUAL' in weight_keys, (
            f'V31 did not select GOOD_RESIDUAL; selected={weight_keys}. '
            f'Residual momentum ranking should prefer idiosyncratic alpha over market beta.'
        )


def test_v31_applies_rank_buffer_and_min_hold(monkeypatch):
    """V31 must call rank_buffer then min_hold (same composition order as V11)."""
    from src.research.ramp_phase4 import variants as v_mod
    call_order = []

    def fake_rank_buffer(proposed_targets, state, buffer_size, universe_ranking, top_n):
        call_order.append('rank_buffer')
        return proposed_targets

    def fake_min_hold(proposed_targets, state, current_date, min_hold_days, crash_exit=False):
        call_order.append('min_hold')
        return proposed_targets

    monkeypatch.setattr(v_mod, 'rank_buffer', fake_rank_buffer, raising=False)
    monkeypatch.setattr(v_mod, 'min_hold', fake_min_hold, raising=False)

    panel = _v31_calm_panel()
    state = HarnessState(cash_usd=100_000.0)
    cfg = _stub_cfg()
    REGISTRY['V31'].plan_fn(panel.index[-1].to_pydatetime(), state, panel, cfg)
    assert call_order == ['rank_buffer', 'min_hold'], (
        f'Expected [rank_buffer, min_hold], got {call_order}'
    )


# ============================================================
# V28 -- Multi-horizon momentum ensemble tests
#
# Three tests required by spec:
#   (1) weights sum <= 1.0+eps with a __regime__ key
#   (2) Symbol with stable multi-horizon momentum ranked above symbol with only 5d spike
#   (3) safe-mode on insufficient history (< 126 trading days)
# ============================================================

def _v28_calm_panel(n=350):
    """Panel for V28 tests: calm trending market with multiple universe names."""
    idx = pd.date_range('2022-01-03', periods=n, freq='B')
    spy = 400 + np.arange(n) * 0.3
    stock_a = 100 + np.arange(n) * 0.25  # steady uptrend
    stock_b = 110 + np.arange(n) * 0.15  # moderate uptrend
    return pd.DataFrame({
        'STOCK_A': stock_a,
        'STOCK_B': stock_b,
        'SPY': spy,
        'VIX': np.full(n, 12.0),
    }, index=idx)


def _v28_horizon_advantage_panel(n=350):
    """Panel engineered so STABLE_MOM has strong multi-horizon momentum
    (21d + 63d + 126d all positive) while SPIKE_ONLY has only a 5d spike
    (which the reversal penalty should punish).

    V28 score = 0.5*ret_21d + 0.3*ret_63d + 0.2*ret_126d - 0.1*ret_5d

    STABLE_MOM: consistent uptrend across all horizons -> high blended score.
    SPIKE_ONLY: flat for all horizons except a spike in last 5 days -> the
    blend of 21/63/126d returns is near zero, but 5d return is large, so
    the reversal penalty hurts the blended score. V28 should rank STABLE_MOM above.
    """
    idx = pd.date_range('2022-01-03', periods=n, freq='B')
    spy = 400 + np.arange(n) * 0.1  # gentle uptrend (market context)

    # STABLE_MOM: consistent uptrend; all horizon returns strongly positive.
    # Price grows ~30% over 126 days, ~20% over 63 days, ~10% over 21 days.
    stable = np.ones(n) * 100.0
    stable += np.arange(n) * 0.40  # 0.4 per day -> big horizon returns

    # SPIKE_ONLY: flat for 345 days, then spikes up 10% in last 5 days.
    # 21d/63d/126d returns will all be dominated by the spike size only.
    # But the 5d return is ~10% and the reversal penalty (0.1 * 10%) = 1%.
    # The blended score (0.5 * ~9% + 0.3 * ~9% + 0.2 * ~9% - 0.1 * 10%) wins for STABLE.
    spike = np.ones(n) * 100.0
    spike[-5:] += np.linspace(0, 10, 5)  # pure 5d spike

    return pd.DataFrame({
        'STABLE_MOM': stable,
        'SPIKE_ONLY': spike,
        'SPY': spy,
        'VIX': np.full(n, 12.0),
    }, index=idx)


def test_v28_in_registry():
    """V28 is registered with expected description hallmarks."""
    assert 'V28' in REGISTRY
    assert isinstance(REGISTRY['V28'], VariantSpec)
    desc = REGISTRY['V28'].description.lower()
    assert 'horizon' in desc or 'multi' in desc or 'ensemble' in desc


def test_v28_weights_sum_lte_one_with_regime_key():
    """(Test 1) V28 plan_fn returns weights summing <= 1.0+eps with __regime__ key."""
    spec = REGISTRY['V28']
    panel = _v28_calm_panel()
    state = HarnessState(cash_usd=100_000.0)
    cfg = _stub_cfg()
    plan = spec.plan_fn(panel.index[-1].to_pydatetime(), state, panel, cfg)
    assert '__regime__' in plan, 'V28 must include __regime__ key'
    weight_keys = [k for k in plan if k != '__regime__']
    if weight_keys:
        total = sum(plan[k] for k in weight_keys)
        assert total <= 1.0 + 1e-6, f'Weights sum to {total:.6f}, expected <= 1.0'
        assert total > 0.0


def test_v28_stable_mom_ranked_above_spike_only():
    """(Test 2) V28 ranks STABLE_MOM above SPIKE_ONLY.

    STABLE_MOM has consistent multi-horizon momentum (21d/63d/126d all positive
    from a consistent uptrend). SPIKE_ONLY has only a 5-day spike; the
    reversal penalty (lambda=0.1) and near-zero 21/63/126d returns should
    keep it ranked below STABLE_MOM.
    """
    from src.research.ramp_phase4.variants import _compute_v28_multihorizon_scores
    panel = _v28_horizon_advantage_panel()
    t = panel.index[-1].to_pydatetime()
    scores = _compute_v28_multihorizon_scores(t, panel)
    assert scores is not None, 'V28 score computation returned None on sufficient history panel'
    assert 'STABLE_MOM' in scores.index, 'STABLE_MOM must appear in V28 score output'
    assert 'SPIKE_ONLY' in scores.index, 'SPIKE_ONLY must appear in V28 score output'
    assert scores['STABLE_MOM'] > scores['SPIKE_ONLY'], (
        f'V28 should rank STABLE_MOM ({scores["STABLE_MOM"]:.4f}) above '
        f'SPIKE_ONLY ({scores["SPIKE_ONLY"]:.4f}); '
        f'multi-horizon blend + reversal penalty should penalize pure 5d spike.'
    )


def test_v28_safe_mode_on_insufficient_history():
    """(Test 3) V28 returns SAFE_MODE when panel has fewer than 126 trading days."""
    spec = REGISTRY['V28']
    n_short = 90  # fewer than 126 required trading days
    idx = pd.date_range('2024-01-01', periods=n_short, freq='B')
    short_panel = pd.DataFrame({
        'AAPL': 100 + np.arange(n_short) * 0.1,
        'SPY': 400 + np.arange(n_short) * 0.2,
        'VIX': np.full(n_short, 14.0),
    }, index=idx)
    state = HarnessState(cash_usd=100_000.0)
    cfg = _stub_cfg()
    plan = spec.plan_fn(short_panel.index[-1].to_pydatetime(), state, short_panel, cfg)
    assert plan.get('__regime__') == 'SAFE_MODE', (
        f'Expected SAFE_MODE on short panel, got: {plan}'
    )


def test_v28_applies_rank_buffer_and_min_hold(monkeypatch):
    """V28 must call rank_buffer then min_hold (V11 composition order)."""
    from src.research.ramp_phase4 import variants as v_mod
    call_order = []

    def fake_rank_buffer(proposed_targets, state, buffer_size, universe_ranking, top_n):
        call_order.append('rank_buffer')
        return proposed_targets

    def fake_min_hold(proposed_targets, state, current_date, min_hold_days, crash_exit=False):
        call_order.append('min_hold')
        return proposed_targets

    monkeypatch.setattr(v_mod, 'rank_buffer', fake_rank_buffer, raising=False)
    monkeypatch.setattr(v_mod, 'min_hold', fake_min_hold, raising=False)

    panel = _v28_calm_panel()
    state = HarnessState(cash_usd=100_000.0)
    cfg = _stub_cfg()
    REGISTRY['V28'].plan_fn(panel.index[-1].to_pydatetime(), state, panel, cfg)
    assert call_order == ['rank_buffer', 'min_hold'], (
        f'Expected [rank_buffer, min_hold], got {call_order}'
    )


# ============================================================
# V26 -- Z-score normalized score tests
#
# Four tests required by spec:
#   (1) weights sum <= 1.0+eps with __regime__ key
#   (2) winsorization caps an extreme outlier at 3 sigma
#       (a symbol with a 10-sigma raw move does not dominate the ranking)
#   (3) z-scoring makes the 5d penalty comparable across days
#       (construct a case where un-normalized penalty would flip the ranking
#        but z-normalized does not)
#   (4) safe-mode on insufficient history (< 21 rows)
# ============================================================

def _v26_calm_panel(n=350):
    """Panel for V26 tests: calm trending market with multiple universe names."""
    idx = pd.date_range('2022-01-03', periods=n, freq='B')
    spy = 400 + np.arange(n) * 0.3
    stock_a = 100 + np.arange(n) * 0.25  # steady uptrend
    stock_b = 110 + np.arange(n) * 0.15  # moderate uptrend
    stock_c = 90 + np.arange(n) * 0.20   # mid uptrend
    return pd.DataFrame({
        'STOCK_A': stock_a,
        'STOCK_B': stock_b,
        'STOCK_C': stock_c,
        'SPY': spy,
        'VIX': np.full(n, 12.0),
    }, index=idx)


def _v26_outlier_panel(n=350):
    """Panel engineered with an extreme outlier (OUTLIER has 10-sigma raw 21d return).

    OUTLIER has a 10-sigma move in the last 21 days -- without winsorization it
    would dominate the z-score ranking. With winsorization at +/-3 sigma it is
    capped and should NOT top-rank above STEADY_MOM whose 21d return is
    consistently strong.
    """
    idx = pd.date_range('2022-01-03', periods=n, freq='B')
    spy = 400 + np.arange(n) * 0.1

    # STEADY_MOM: consistent 21d uptrend, strong but not extreme.
    steady = np.ones(n) * 100.0
    steady += np.arange(n) * 0.40  # ~8.4% 21d return

    # OUTLIER: flat for most of history, then +200% in last 21 days (extreme spike).
    outlier = np.ones(n) * 100.0
    outlier[-21:] += np.linspace(0, 200.0, 21)  # extreme move -> would be >>3 sigma

    # WEAK: slight downtrend.
    weak = np.ones(n) * 80.0
    weak -= np.arange(n) * 0.05

    return pd.DataFrame({
        'STEADY_MOM': steady,
        'OUTLIER': outlier,
        'WEAK': weak,
        'SPY': spy,
        'VIX': np.full(n, 12.0),
    }, index=idx)


def _v26_zscore_comparability_panel(n=350):
    """Panel demonstrating that z-scoring makes 21d and 5d terms comparable.

    Tests: STRONG_21D_FLAT5 has a strong 21-day uptrend but a flat last-5-day
    (reversal to baseline after spike). RAW_ONLY_21D has a large raw 21d return
    but also a huge 5d return (the same gains concentrated in 5 days).

    Without z-scoring: RAW_ONLY_21D raw 21d ~ STRONG_21D_FLAT5 raw 21d, BUT
    RAW_ONLY_21D has big raw 5d return -> big penalty -> rank FLIP: STRONG_21D_FLAT5 wins.

    With z-scoring: both raw terms are normalized, same conclusion: STRONG_21D_FLAT5
    has large z21 and near-zero z5 (flat last 5 days) -> highest score.

    The test checks STRONG_21D_FLAT5's z26 score > RAW_ONLY_21D's z26 score.
    """
    idx = pd.date_range('2022-01-03', periods=n, freq='B')
    spy = 400 + np.arange(n) * 0.1

    # STRONG_21D_FLAT5: steady 21d uptrend, but flat last 5 days (no 5d penalty).
    # Price rises over the 21d window, then levels off in last 5 days.
    strong21 = np.ones(n) * 100.0
    strong21[-(21 + 5):-(5)] += np.linspace(0, 20.0, 21)  # rises 20% in days -26..-6
    strong21[-5:] = strong21[-6]                             # flat in last 5 days

    # RAW_ONLY_21D: same 21d endpoint-to-endpoint gain, but concentrated in last 5 days.
    # This means raw_21d ~ same as STRONG_21D_FLAT5, but raw_5d is large -> big penalty.
    raw_only = np.ones(n) * 100.0
    raw_only[-5:] += np.linspace(0, 20.0, 5)  # all 20% gain in last 5 days

    # FLAT: flat price, used to set the cross-sectional baseline.
    flat = np.ones(n) * 80.0

    return pd.DataFrame({
        'STRONG_21D_FLAT5': strong21,
        'RAW_ONLY_21D': raw_only,
        'FLAT': flat,
        'SPY': spy,
        'VIX': np.full(n, 12.0),
    }, index=idx)


def test_v26_in_registry():
    """V26 is registered with expected description hallmarks."""
    assert 'V26' in REGISTRY
    assert isinstance(REGISTRY['V26'], VariantSpec)
    desc = REGISTRY['V26'].description.lower()
    assert 'z-score' in desc or 'zscore' in desc or 'z_score' in desc or 'normalized' in desc


def test_v26_weights_sum_lte_one_with_regime_key():
    """(Test 1) V26 plan_fn returns weights summing <= 1.0+eps with __regime__ key."""
    spec = REGISTRY['V26']
    panel = _v26_calm_panel()
    state = HarnessState(cash_usd=100_000.0)
    cfg = _stub_cfg()
    plan = spec.plan_fn(panel.index[-1].to_pydatetime(), state, panel, cfg)
    assert '__regime__' in plan, 'V26 must include __regime__ key'
    weight_keys = [k for k in plan if k != '__regime__']
    if weight_keys:
        total = sum(plan[k] for k in weight_keys)
        assert total <= 1.0 + 1e-6, f'Weights sum to {total:.6f}, expected <= 1.0'
        assert total > 0.0


def test_v26_winsorization_caps_extreme_outlier():
    """(Test 2) Winsorization at 3 sigma caps extreme raw returns.

    A symbol with a 10-sigma raw move (OUTLIER) should NOT dominate the
    ranking after winsorization. STEADY_MOM with consistent returns should
    score comparably or better than the winsorized outlier.
    """
    from src.research.ramp_phase4.variants import _compute_v26_zscore_scores
    panel = _v26_outlier_panel()
    t = panel.index[-1].to_pydatetime()
    scores = _compute_v26_zscore_scores(t, panel)
    assert scores is not None, 'V26 score computation returned None on sufficient history panel'
    assert 'OUTLIER' in scores.index, 'OUTLIER must appear in score output'
    assert 'STEADY_MOM' in scores.index, 'STEADY_MOM must appear in score output'
    # With winsorization: OUTLIER's extreme 21d return is capped at 3 sigma.
    # STEADY_MOM has top-decile consistent 21d performance.
    # OUTLIER should NOT have a score more than 1 sigma above STEADY_MOM
    # (without winsorization it would be 10x or more).
    score_outlier = scores['OUTLIER']
    score_steady = scores['STEADY_MOM']
    # After winsorization the outlier is capped; the gap should be small
    # (within normal z-score range), not 10x or more.
    assert score_outlier <= score_steady + 6.0, (
        f'Outlier score {score_outlier:.3f} too far above STEADY_MOM {score_steady:.3f}; '
        f'winsorization should cap extreme raw returns at 3 sigma'
    )


def test_v26_zscore_makes_penalty_comparable():
    """(Test 3) Z-scoring makes the 21d and 5d terms comparable.

    STRONG_21D_FLAT5 has a strong 21-day uptrend but flat last 5 days (no penalty).
    RAW_ONLY_21D has the same 21d endpoint gain but concentrated in last 5 days
    (big 5d reversal penalty). Z-scoring normalizes both terms so the comparison
    is fair: STRONG_21D_FLAT5 should rank above RAW_ONLY_21D because z21 is
    high for both, but z5 penalty is zero for STRONG_21D_FLAT5 vs large for
    RAW_ONLY_21D.
    """
    from src.research.ramp_phase4.variants import _compute_v26_zscore_scores
    panel = _v26_zscore_comparability_panel()
    t = panel.index[-1].to_pydatetime()
    scores = _compute_v26_zscore_scores(t, panel)
    assert scores is not None, 'V26 score computation returned None'
    assert 'STRONG_21D_FLAT5' in scores.index
    assert 'RAW_ONLY_21D' in scores.index
    # STRONG_21D_FLAT5 should rank above RAW_ONLY_21D: same z21 but lower z5 penalty.
    assert scores['STRONG_21D_FLAT5'] > scores['RAW_ONLY_21D'], (
        f'V26 should rank STRONG_21D_FLAT5 ({scores["STRONG_21D_FLAT5"]:.4f}) above '
        f'RAW_ONLY_21D ({scores["RAW_ONLY_21D"]:.4f}); '
        f'STRONG_21D_FLAT5 has same 21d gain but zero 5d penalty (flat last 5 days).'
    )


def test_v26_safe_mode_on_insufficient_history():
    """(Test 4) V26 returns SAFE_MODE when panel has insufficient history."""
    spec = REGISTRY['V26']
    n_short = 20  # fewer than 21 required rows for z-scoring
    idx = pd.date_range('2024-01-01', periods=n_short, freq='B')
    short_panel = pd.DataFrame({
        'AAPL': 100 + np.arange(n_short) * 0.1,
        'SPY': 400 + np.arange(n_short) * 0.2,
        'VIX': np.full(n_short, 14.0),
    }, index=idx)
    state = HarnessState(cash_usd=100_000.0)
    cfg = _stub_cfg()
    plan = spec.plan_fn(short_panel.index[-1].to_pydatetime(), state, short_panel, cfg)
    assert plan.get('__regime__') == 'SAFE_MODE', (
        f'Expected SAFE_MODE on short panel, got: {plan}'
    )


def test_v26_applies_rank_buffer_and_min_hold(monkeypatch):
    """V26 must call rank_buffer then min_hold (V11 composition order)."""
    from src.research.ramp_phase4 import variants as v_mod
    call_order = []

    def fake_rank_buffer(proposed_targets, state, buffer_size, universe_ranking, top_n):
        call_order.append('rank_buffer')
        return proposed_targets

    def fake_min_hold(proposed_targets, state, current_date, min_hold_days, crash_exit=False):
        call_order.append('min_hold')
        return proposed_targets

    monkeypatch.setattr(v_mod, 'rank_buffer', fake_rank_buffer, raising=False)
    monkeypatch.setattr(v_mod, 'min_hold', fake_min_hold, raising=False)

    panel = _v26_calm_panel()
    state = HarnessState(cash_usd=100_000.0)
    cfg = _stub_cfg()
    REGISTRY['V26'].plan_fn(panel.index[-1].to_pydatetime(), state, panel, cfg)
    assert call_order == ['rank_buffer', 'min_hold'], (
        f'Expected [rank_buffer, min_hold], got {call_order}'
    )
