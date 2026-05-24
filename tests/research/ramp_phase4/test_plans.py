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
