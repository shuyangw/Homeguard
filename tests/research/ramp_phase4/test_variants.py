"""Tests for variants.py: V01 + V03 plan functions."""
from datetime import datetime
import numpy as np
import pandas as pd
import pytest

from src.research.ramp_phase4.variants import REGISTRY, VariantSpec


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
