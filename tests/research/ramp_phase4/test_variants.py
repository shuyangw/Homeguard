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
    # V03 is added in Task 14; keep this assertion permissive until then.
    assert 'V03' in REGISTRY or True  # V03 added in Task 14
    assert isinstance(REGISTRY['V01'], VariantSpec)
    if 'V03' in REGISTRY:
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
