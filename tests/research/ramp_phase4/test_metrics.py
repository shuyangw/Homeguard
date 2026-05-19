"""Tests for metrics module."""
import numpy as np
import pandas as pd
import pytest

from src.research.ramp_phase4.metrics import (
    sharpe_ratio, cagr, max_drawdown,
)


def test_sharpe_ratio_known_series():
    np.random.seed(42)
    rets = pd.Series(np.random.normal(0.001, 0.01, 252))
    s = sharpe_ratio(rets)
    expected = (rets.mean() * 252) / (rets.std(ddof=1) * np.sqrt(252))
    assert abs(s - expected) < 1e-9


def test_sharpe_ratio_zero_when_returns_constant():
    rets = pd.Series([0.001] * 252)
    s = sharpe_ratio(rets)
    assert s == 0.0


def test_cagr_known_curve():
    idx = pd.date_range('2024-01-02', periods=252, freq='B')
    equity = pd.Series(np.linspace(100.0, 110.0, 252), index=idx)
    c = cagr(equity)
    assert abs(c - 0.10) < 0.01


def test_max_drawdown_known_curve():
    equity = pd.Series([100, 110, 105, 95, 100, 90, 95])
    dd = max_drawdown(equity)
    assert abs(dd - (-20 / 110)) < 1e-9


def test_max_drawdown_zero_for_monotonic_rising():
    equity = pd.Series([100, 101, 102, 103])
    assert max_drawdown(equity) == 0.0
