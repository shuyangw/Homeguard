"""Tests for variants.py: registry ids, aliases, resolve(), and the four plan_fns."""
from datetime import datetime
import numpy as np
import pandas as pd
import pytest

from src.research.regime_momentum_lab.variants import REGISTRY, VariantSpec, V1_PARAMS, resolve
from src.strategies.advanced.ramp_strategy import RAMPSignals


def _calm_panel(n=300):
    idx = pd.date_range('2023-01-02', periods=n, freq='B')
    return pd.DataFrame({
        'AAA': 100 + np.arange(n) * 0.05,
        'BBB': 110 + np.arange(n) * 0.04,
        'CCC': 90 + np.arange(n) * 0.06,
        'SPY': 400 + np.arange(n) * 0.1,
        'VIX': np.full(n, 12.0),
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
    assert resolve('prod').id == 'prod'


def test_resolve_unknown_raises():
    with pytest.raises(KeyError):
        resolve('nonsense')
    with pytest.raises(KeyError):
        resolve('v11')


def test_prod_no_crash_full_gross_in_calm():
    plan = _call('prod_no_crash', _calm_panel())
    body = {k: v for k, v in plan.items() if k != '__regime__'}
    assert abs(sum(body.values()) - 1.0) < 0.01


def test_prod_applies_crash_exposure_in_crash():
    plan = _call('prod', _crash_panel())
    body = {k: v for k, v in plan.items() if k != '__regime__'}
    assert sum(body.values()) <= 0.6


def test_prod_and_prod_no_crash_identical_in_calm():
    p_nc = _call('prod_no_crash', _calm_panel())
    p_pr = _call('prod', _calm_panel())
    assert set(p_nc) - {'__regime__'} == set(p_pr) - {'__regime__'}
    for sym in set(p_nc) - {'__regime__'}:
        assert abs(p_nc[sym] - p_pr[sym]) < 1e-6


def test_plain_ignores_regime_keeps_full_gross_in_crash():
    plain = _call('plain', _crash_panel())
    prod = _call('prod', _crash_panel())
    plain_body = {k: v for k, v in plain.items() if k != '__regime__'}
    prod_body = {k: v for k, v in prod.items() if k != '__regime__'}
    assert abs(sum(plain_body.values()) - 1.0) < 0.01
    assert sum(prod_body.values()) <= 0.6
    assert '__regime__' in plain


def test_plain_selects_by_momentum_not_column_order():
    """Regression: _variant_plain must rank by momentum score, not column order.

    The momentum formula is: 0.3 * return_21d - 5.0 * return_5d.
    With pen_w=5.0 dominating, LOWER price slope -> LOWER penalty -> HIGHER score.
    We arrange slopes to DECREASE with column index (S00 fastest, S12 slowest),
    so the highest-momentum names are the LAST columns.

    Column-order top-10 picks S00..S09 (fast slope, low momentum).
    Sorted top-10 picks S03..S12 (slow slope, high momentum). Different sets.

    The naive bug (index[:top_n]) picks the wrong set; sort_values fixes it.
    """
    n = 300
    top_n = V1_PARAMS['top_n']  # 10
    n_syms = 13  # > top_n so column-order != momentum-order
    idx = pd.date_range('2023-01-02', periods=n, freq='B')
    t = idx[-1].to_pydatetime()

    # Slopes DECREASE with column index: S00 fastest (0.13), S12 slowest (0.01).
    # Momentum formula penalizes fast short-term returns heavily (pen_w=5.0),
    # so slower slope -> higher score -> highest-momentum names are LAST columns.
    data = {}
    for i in range(n_syms):
        sym = f'S{i:02d}'
        slope = 0.01 * (n_syms - i)  # S00: 0.13, S12: 0.01
        data[sym] = 100.0 + np.arange(n) * slope
    data['SPY'] = 400.0 + np.arange(n) * 0.1
    data['VIX'] = np.full(n, 12.0)
    panel = pd.DataFrame(data, index=idx)

    universe_cols = [c for c in panel.columns if c not in ('SPY', 'VIX')]
    prices_slice = panel.loc[:t, universe_cols]

    # Reference ranking using the same path as _variant_plain.
    ramp = RAMPSignals(symbols=universe_cols)
    ramp._current_params = V1_PARAMS
    momentum = ramp.calculate_momentum_scores(prices_slice)
    assert momentum is not None and len(momentum) > 0

    expected_top = set(momentum.sort_values(ascending=False).index[:top_n])
    naive_top = set(momentum.index[:top_n])

    # Sanity: confirm the test actually exercises the ordering distinction.
    assert expected_top != naive_top, (
        'Expected and naive sets are identical -- adjust slopes so column order '
        'differs from momentum order.'
    )

    # Run _variant_plain and check it returns the momentum-sorted set.
    state = type('S', (), {'positions': {}, 'cash_usd': 100000.0})()
    cfg = type('C', (), {})()
    result = REGISTRY['plain'].plan_fn(t, state, panel, cfg)
    selected = {k for k in result if k != '__regime__'}

    assert selected == expected_top, (
        f'plain selected {sorted(selected)} but expected momentum top-{top_n}: '
        f'{sorted(expected_top)}'
    )


@pytest.mark.skip(reason="needs deterministic BEAR fixture; see plan PR 3.3")
def test_bear_cash_goes_to_cash_in_bear():
    plan = _call('bear_cash', _crash_panel())
    if plan.get('__regime__') != 'BEAR':
        pytest.skip('panel did not classify BEAR; needs a dedicated BEAR fixture')
    body = {k: v for k, v in plan.items() if k != '__regime__'}
    assert abs(sum(body.values())) < 1e-9
