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
