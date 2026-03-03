from datetime import date

import pytest

from src.strategies.options.csp.metrics import compute_csp_metrics
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
