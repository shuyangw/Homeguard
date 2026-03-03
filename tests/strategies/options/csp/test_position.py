from datetime import date

import pytest

from src.strategies.options.csp.position import CSPPosition, CSPTrade


class TestCSPPosition:

    def _make_position(self, **overrides) -> CSPPosition:
        defaults = dict(
            symbol="AAPL",
            strike=150.0,
            expiry=date(2026, 7, 18),
            entry_date=date(2026, 6, 20),
            entry_price=2.50,
            num_contracts=1,
            collateral=15000.0,
        )
        defaults.update(overrides)
        return CSPPosition(**defaults)

    def test_premium_collected(self):
        pos = self._make_position(entry_price=2.50, num_contracts=2)
        assert pos.premium_collected == pytest.approx(500.0)

    def test_unrealized_pnl_profit(self):
        pos = self._make_position(entry_price=2.50, num_contracts=1)
        pos.current_price = 1.25
        assert pos.unrealized_pnl == pytest.approx(125.0)

    def test_unrealized_pnl_loss(self):
        pos = self._make_position(entry_price=2.50, num_contracts=1)
        pos.current_price = 5.00
        assert pos.unrealized_pnl == pytest.approx(-250.0)

    def test_pnl_pct_of_premium(self):
        pos = self._make_position(entry_price=2.50, num_contracts=1)
        pos.current_price = 1.25
        assert pos.pnl_pct_of_premium == pytest.approx(0.5)

    def test_pnl_pct_of_premium_zero_premium(self):
        pos = self._make_position(entry_price=0.0, num_contracts=1)
        assert pos.pnl_pct_of_premium == pytest.approx(0.0)


class TestCSPTrade:

    def _make_trade(self, **overrides) -> CSPTrade:
        defaults = dict(
            symbol="AAPL",
            strike=150.0,
            expiry=date(2026, 7, 18),
            entry_date=date(2026, 6, 20),
            exit_date=date(2026, 7, 5),
            entry_price=2.50,
            exit_price=1.20,
            num_contracts=1,
            exit_reason="profit_target",
            regime_at_entry="STRONG_BULL",
            regime_at_exit="STRONG_BULL",
            momentum_rank_at_entry=5,
        )
        defaults.update(overrides)
        return CSPTrade(**defaults)

    def test_realized_pnl(self):
        trade = self._make_trade(entry_price=2.50, exit_price=1.20, num_contracts=1)
        assert trade.realized_pnl == pytest.approx(130.0)

    def test_holding_days(self):
        trade = self._make_trade(
            entry_date=date(2026, 6, 20),
            exit_date=date(2026, 7, 5),
        )
        assert trade.holding_days == 15

    def test_return_on_collateral(self):
        trade = self._make_trade(
            entry_price=2.50,
            exit_price=1.20,
            strike=150.0,
            num_contracts=1,
        )
        assert trade.return_on_collateral == pytest.approx(130.0 / 15000.0)

    def test_return_on_collateral_zero_strike(self):
        trade = self._make_trade(strike=0.0)
        assert trade.return_on_collateral == pytest.approx(0.0)

    def test_realized_pnl_includes_fees(self):
        trade = self._make_trade(
            entry_price=2.50, exit_price=1.20, num_contracts=2,
            entry_fee=0.04, exit_fee=0.04,
        )
        gross = (2.50 - 1.20) * 100 * 2  # 260.0
        assert trade.realized_pnl == pytest.approx(gross - 0.04 - 0.04)
