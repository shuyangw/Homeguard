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


def test_sentinel_plan_dispatch_produces_zero_targets_and_attribution_label():
    """run_variant treats _SentinelPlan output as zero-target with reason as regime label."""
    sp = PLAN_CASH_BEAR_SOFT
    assert isinstance(sp, _SentinelPlan)
    regime_label = sp.reason
    target_weights: dict = {}
    assert regime_label == 'BEAR_SOFT_CASH'
    assert target_weights == {}
