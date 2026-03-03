from datetime import date

import pytest

import pandas as pd

from src.strategies.options.csp.metrics import compute_csp_metrics, compute_sharpe, compute_max_drawdown
from src.strategies.options.csp.position import CSPTrade


def _make_trade(pnl_direction: float = 1.0, **kwargs) -> CSPTrade:
    defaults = dict(
        symbol="AAPL",
        strike=150.0,
        expiry=date(2024, 7, 19),
        entry_date=date(2024, 6, 20),
        exit_date=date(2024, 7, 5),
        entry_price=2.50,
        exit_price=2.50 - (1.30 * pnl_direction),
        num_contracts=1,
        exit_reason="profit_target",
        regime_at_entry="STRONG_BULL",
        regime_at_exit="STRONG_BULL",
        momentum_rank_at_entry=3,
    )
    defaults.update(kwargs)
    return CSPTrade(**defaults)


class TestComputeCspMetrics:
    def test_win_rate(self):
        trades = [
            _make_trade(pnl_direction=1.0),
            _make_trade(pnl_direction=1.0),
            _make_trade(pnl_direction=-1.0),
        ]
        result = compute_csp_metrics(trades)
        assert result["total_trades"] == 3
        assert result["winning_trades"] == 2
        assert result["losing_trades"] == 1
        assert result["win_rate"] == pytest.approx(2.0 / 3.0)

    def test_avg_return_on_collateral(self):
        trade = _make_trade(pnl_direction=1.0)
        result = compute_csp_metrics([trade])
        expected_roc = trade.return_on_collateral
        assert expected_roc > 0
        assert result["avg_return_on_collateral"] == pytest.approx(expected_roc)

    def test_pnl_by_exit_reason(self):
        t1 = _make_trade(pnl_direction=1.0, exit_reason="profit_target")
        t2 = _make_trade(pnl_direction=-1.0, exit_reason="stop_loss")
        result = compute_csp_metrics([t1, t2])
        assert "profit_target" in result["pnl_by_exit_reason"]
        assert "stop_loss" in result["pnl_by_exit_reason"]
        assert result["pnl_by_exit_reason"]["profit_target"] == pytest.approx(
            t1.realized_pnl
        )
        assert result["pnl_by_exit_reason"]["stop_loss"] == pytest.approx(
            t2.realized_pnl
        )
        assert result["count_by_exit_reason"]["profit_target"] == 1
        assert result["count_by_exit_reason"]["stop_loss"] == 1

    def test_empty_trades(self):
        result = compute_csp_metrics([])
        assert result["total_trades"] == 0
        assert result["win_rate"] == 0.0
        assert result["winning_trades"] == 0
        assert result["losing_trades"] == 0
        assert result["avg_premium"] == 0.0
        assert result["avg_return_on_collateral"] == 0.0
        assert result["avg_holding_days"] == 0.0
        assert result["total_pnl"] == 0.0
        assert result["pnl_by_exit_reason"] == {}
        assert result["count_by_exit_reason"] == {}
        assert result["pnl_by_regime"] == {}

    def test_avg_holding_days(self):
        trade = _make_trade(
            entry_date=date(2024, 6, 20),
            exit_date=date(2024, 7, 5),
        )
        result = compute_csp_metrics([trade])
        assert result["avg_holding_days"] == 15.0


class TestComputeSharpe:
    def test_positive_sharpe(self):
        equity = pd.Series(
            [100.0, 101.0, 102.0, 103.0, 104.0],
            index=pd.date_range("2024-01-01", periods=5),
        )
        sharpe = compute_sharpe(equity)
        assert sharpe > 0

    def test_none_returns_zero(self):
        assert compute_sharpe(None) == 0.0

    def test_single_value_returns_zero(self):
        equity = pd.Series([100.0], index=pd.date_range("2024-01-01", periods=1))
        assert compute_sharpe(equity) == 0.0

    def test_flat_equity_returns_zero(self):
        equity = pd.Series(
            [100.0, 100.0, 100.0],
            index=pd.date_range("2024-01-01", periods=3),
        )
        assert compute_sharpe(equity) == 0.0


class TestComputeMaxDrawdown:
    def test_drawdown_from_decline(self):
        equity = pd.Series(
            [100.0, 110.0, 99.0, 105.0],
            index=pd.date_range("2024-01-01", periods=4),
        )
        dd = compute_max_drawdown(equity)
        assert dd == pytest.approx(11.0 / 110.0)

    def test_none_returns_zero(self):
        assert compute_max_drawdown(None) == 0.0

    def test_monotonic_increase_returns_zero(self):
        equity = pd.Series(
            [100.0, 101.0, 102.0],
            index=pd.date_range("2024-01-01", periods=3),
        )
        assert compute_max_drawdown(equity) == pytest.approx(0.0)
