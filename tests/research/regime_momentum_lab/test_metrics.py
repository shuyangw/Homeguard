"""Tests for metrics module."""
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from src.research.regime_momentum_lab.metrics import (
    sharpe_ratio, cagr, max_drawdown,
    avg_daily_turnover, cost_drag_pct, regime_attribution,
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


@dataclass
class FakeRecord:
    date: datetime
    regime: str
    portfolio_value: float
    daily_return: float
    turnover_usd: float
    cost_usd: float


def _records(seq):
    return [
        FakeRecord(
            date=datetime(2024, 1, i + 1),
            regime=row[0],
            portfolio_value=row[1],
            daily_return=row[2],
            turnover_usd=row[3],
            cost_usd=row[4],
        )
        for i, row in enumerate(seq)
    ]


def test_avg_daily_turnover_normalizes_by_portfolio_value():
    recs = _records([
        ('STRONG_BULL', 100000.0, 0.0, 10000.0, 5.0),
        ('STRONG_BULL', 100100.0, 0.001, 20000.0, 10.0),
    ])
    t = avg_daily_turnover(recs)
    assert abs(t - 0.1499) < 0.001


def test_cost_drag_pct_zero_when_no_costs():
    recs = _records([
        ('STRONG_BULL', 100000.0, 0.005, 0.0, 0.0),
        ('STRONG_BULL', 100500.0, 0.005, 0.0, 0.0),
    ])
    assert cost_drag_pct(recs) == 0.0


def test_cost_drag_pct_proportional_to_gross_return():
    recs = _records([
        ('STRONG_BULL', 100000.0, 0.005, 0.0, 100.0),
        ('STRONG_BULL', 100500.0, 0.005, 0.0, 100.0),
    ])
    drag = cost_drag_pct(recs)
    assert 0.15 < drag < 0.25


def test_regime_attribution_sums_to_total_return():
    recs = _records([
        ('STRONG_BULL', 100000.0, 0.01, 0.0, 0.0),
        ('WEAK_BULL',   101000.0, -0.005, 0.0, 0.0),
        ('STRONG_BULL', 100495.0, 0.02, 0.0, 0.0),
    ])
    attr = regime_attribution(recs)
    total = (1 + 0.01) * (1 - 0.005) * (1 + 0.02) - 1
    assert abs(sum(row['net_return'] for row in attr.values()) - total) < 1e-3
