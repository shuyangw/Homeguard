"""
Unit tests for DemoPortfolioState.

Tests position tracking, P&L calculations, and state persistence.
"""

import json
import pytest
import tempfile
from decimal import Decimal
from pathlib import Path
from datetime import datetime

from src.trading.demo.portfolio_state import DemoPortfolioState, DemoPosition


class TestDemoPosition:
    """Tests for DemoPosition class."""

    def test_create_position(self):
        """Create position with basic attributes."""
        pos = DemoPosition(
            symbol="BTC/USD",
            quantity=Decimal("0.1"),
            avg_entry_price=50000.0,
        )
        assert pos.symbol == "BTC/USD"
        assert pos.quantity == Decimal("0.1")
        assert pos.avg_entry_price == 50000.0

    def test_cost_basis_auto_calculated(self):
        """Cost basis calculated from qty and price."""
        pos = DemoPosition(
            symbol="BTC/USD",
            quantity=Decimal("0.1"),
            avg_entry_price=50000.0,
        )
        assert pos.cost_basis == 5000.0  # 0.1 * 50000

    def test_market_value(self):
        """Market value calculation."""
        pos = DemoPosition(
            symbol="BTC/USD",
            quantity=Decimal("0.1"),
            avg_entry_price=50000.0,
        )
        assert pos.market_value(55000.0) == 5500.0

    def test_unrealized_pnl_profit(self):
        """Unrealized P&L when in profit."""
        pos = DemoPosition(
            symbol="BTC/USD",
            quantity=Decimal("0.1"),
            avg_entry_price=50000.0,
        )
        assert pos.unrealized_pnl(55000.0) == 500.0  # 5500 - 5000

    def test_unrealized_pnl_loss(self):
        """Unrealized P&L when in loss."""
        pos = DemoPosition(
            symbol="BTC/USD",
            quantity=Decimal("0.1"),
            avg_entry_price=50000.0,
        )
        assert pos.unrealized_pnl(45000.0) == -500.0  # 4500 - 5000

    def test_unrealized_pnl_pct(self):
        """Unrealized P&L percentage."""
        pos = DemoPosition(
            symbol="BTC/USD",
            quantity=Decimal("0.1"),
            avg_entry_price=50000.0,
        )
        assert pos.unrealized_pnl_pct(55000.0) == 10.0  # 500/5000 * 100

    def test_to_dict(self):
        """Position serializes to dict."""
        pos = DemoPosition(
            symbol="BTC/USD",
            quantity=Decimal("0.1"),
            avg_entry_price=50000.0,
        )
        d = pos.to_dict()
        assert d["symbol"] == "BTC/USD"
        assert d["quantity"] == "0.1"  # Decimal -> str
        assert d["avg_entry_price"] == 50000.0

    def test_from_dict(self):
        """Position deserializes from dict."""
        data = {
            "symbol": "BTC/USD",
            "quantity": "0.1",
            "avg_entry_price": 50000.0,
            "cost_basis": 5000.0,
            "opened_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
        }
        pos = DemoPosition.from_dict(data)
        assert pos.symbol == "BTC/USD"
        assert pos.quantity == Decimal("0.1")

    def test_quantity_auto_converted_to_decimal(self):
        """Float quantity converted to Decimal."""
        pos = DemoPosition(
            symbol="BTC/USD",
            quantity=0.1,  # float, not Decimal
            avg_entry_price=50000.0,
        )
        assert isinstance(pos.quantity, Decimal)


class TestDemoPortfolioState:
    """Tests for DemoPortfolioState class."""

    @pytest.fixture
    def temp_state_dir(self):
        """Create temporary directory for state files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def portfolio(self, temp_state_dir):
        """Create portfolio with temp state directory."""
        return DemoPortfolioState(
            initial_cash=10000.0,
            state_dir=temp_state_dir,
            auto_save=False,
        )

    def test_initial_cash(self, portfolio):
        """Portfolio starts with initial cash."""
        assert portfolio.cash == 10000.0

    def test_initial_no_positions(self, portfolio):
        """Portfolio starts with no positions."""
        assert len(portfolio.get_all_positions()) == 0

    def test_add_position_buy(self, portfolio):
        """Adding position via buy."""
        portfolio.add_or_update_position(
            symbol="BTC/USD",
            quantity_delta=Decimal("0.1"),
            price=50000.0,
        )
        pos = portfolio.get_position("BTC/USD")
        assert pos is not None
        assert pos.quantity == Decimal("0.1")
        assert pos.avg_entry_price == 50000.0

    def test_update_position_average_in(self, portfolio):
        """Multiple buys average entry price."""
        # Buy 0.1 @ 50000
        portfolio.add_or_update_position("BTC/USD", Decimal("0.1"), 50000.0)
        # Buy 0.1 @ 52000
        portfolio.add_or_update_position("BTC/USD", Decimal("0.1"), 52000.0)

        pos = portfolio.get_position("BTC/USD")
        assert pos.quantity == Decimal("0.2")
        assert pos.avg_entry_price == 51000.0  # (5000 + 5200) / 0.2

    def test_sell_position_partial(self, portfolio):
        """Selling partial position."""
        portfolio.add_or_update_position("BTC/USD", Decimal("0.2"), 50000.0)
        portfolio.add_or_update_position("BTC/USD", Decimal("-0.1"), 55000.0)

        pos = portfolio.get_position("BTC/USD")
        assert pos.quantity == Decimal("0.1")

    def test_sell_position_full_removes(self, portfolio):
        """Selling full position removes it."""
        portfolio.add_or_update_position("BTC/USD", Decimal("0.1"), 50000.0)
        portfolio.add_or_update_position("BTC/USD", Decimal("-0.1"), 55000.0)

        assert portfolio.get_position("BTC/USD") is None

    def test_sell_realizes_pnl(self, portfolio):
        """Selling realizes P&L."""
        portfolio.add_or_update_position("BTC/USD", Decimal("0.1"), 50000.0)
        portfolio.add_or_update_position("BTC/USD", Decimal("-0.1"), 55000.0)

        # P&L = 0.1 * (55000 - 50000) = 500
        assert portfolio.realized_pnl == 500.0

    def test_cannot_sell_without_position(self, portfolio):
        """Selling without position raises error."""
        with pytest.raises(ValueError, match="no position exists"):
            portfolio.add_or_update_position("BTC/USD", Decimal("-0.1"), 50000.0)

    def test_cannot_oversell(self, portfolio):
        """Cannot sell more than held."""
        portfolio.add_or_update_position("BTC/USD", Decimal("0.1"), 50000.0)
        with pytest.raises(ValueError, match="Cannot sell"):
            portfolio.add_or_update_position("BTC/USD", Decimal("-0.2"), 50000.0)

    def test_adjust_cash_positive(self, portfolio):
        """Add cash to balance."""
        portfolio.adjust_cash(1000.0)
        assert portfolio.cash == 11000.0

    def test_adjust_cash_negative(self, portfolio):
        """Subtract cash from balance."""
        portfolio.adjust_cash(-2000.0)
        assert portfolio.cash == 8000.0

    def test_get_total_value(self, portfolio):
        """Total value includes cash and positions."""
        portfolio.add_or_update_position("BTC/USD", Decimal("0.1"), 50000.0)
        # Cash is still 10000 (we don't auto-deduct in add_or_update_position)

        prices = {"BTC/USD": 55000.0}
        total = portfolio.get_total_value(prices)
        # Cash + 0.1 * 55000 = 10000 + 5500 = 15500
        assert total == 15500.0

    def test_get_unrealized_pnl(self, portfolio):
        """Unrealized P&L calculated correctly."""
        portfolio.add_or_update_position("BTC/USD", Decimal("0.1"), 50000.0)
        prices = {"BTC/USD": 55000.0}
        unrealized = portfolio.get_unrealized_pnl(prices)
        assert unrealized == 500.0  # 0.1 * (55000 - 50000)


class TestPortfolioStatePersistence:
    """Tests for state persistence."""

    @pytest.fixture
    def temp_state_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_save_and_load(self, temp_state_dir):
        """State survives save/load cycle."""
        # Create and populate portfolio
        p1 = DemoPortfolioState(
            initial_cash=10000.0, state_dir=temp_state_dir, auto_save=False
        )
        p1.add_or_update_position("BTC/USD", Decimal("0.1"), 50000.0)
        p1.adjust_cash(-5000.0)  # Simulate spending
        p1.save()

        # Create new portfolio from saved state
        p2 = DemoPortfolioState(
            initial_cash=99999.0,  # Should be ignored
            state_dir=temp_state_dir,
            auto_save=False,
        )

        assert p2.cash == 5000.0
        pos = p2.get_position("BTC/USD")
        assert pos is not None
        assert pos.quantity == Decimal("0.1")

    def test_fresh_state_if_no_file(self, temp_state_dir):
        """Fresh portfolio if no saved state."""
        p = DemoPortfolioState(
            initial_cash=25000.0, state_dir=temp_state_dir, auto_save=False
        )
        assert p.cash == 25000.0
        assert len(p.get_all_positions()) == 0

    def test_auto_save_on_change(self, temp_state_dir):
        """Changes auto-saved when auto_save=True."""
        p1 = DemoPortfolioState(
            initial_cash=10000.0, state_dir=temp_state_dir, auto_save=True
        )
        p1.add_or_update_position("BTC/USD", Decimal("0.1"), 50000.0)

        # Verify file exists and has content
        state_file = temp_state_dir / "portfolio_state.json"
        assert state_file.exists()

        with open(state_file) as f:
            data = json.load(f)
        assert len(data["positions"]) == 1

    def test_reset_clears_everything(self, temp_state_dir):
        """Reset clears positions and realized P&L."""
        p = DemoPortfolioState(
            initial_cash=10000.0, state_dir=temp_state_dir, auto_save=False
        )
        p.add_or_update_position("BTC/USD", Decimal("0.1"), 50000.0)
        p.add_or_update_position("BTC/USD", Decimal("-0.1"), 55000.0)  # Realize P&L

        p.reset(initial_cash=20000.0)

        assert p.cash == 20000.0
        assert p.realized_pnl == 0.0
        assert len(p.get_all_positions()) == 0


class TestPortfolioSnapshot:
    """Tests for portfolio snapshot generation."""

    @pytest.fixture
    def portfolio_with_positions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            p = DemoPortfolioState(
                initial_cash=10000.0, state_dir=Path(tmpdir), auto_save=False
            )
            p.add_or_update_position("BTC/USD", Decimal("0.1"), 50000.0)
            p.add_or_update_position("ETH/USD", Decimal("1.0"), 3000.0)
            p.adjust_cash(-8000.0)  # Spent on positions
            yield p

    def test_snapshot_contains_cash(self, portfolio_with_positions):
        """Snapshot includes cash."""
        snapshot = portfolio_with_positions.get_portfolio_snapshot(
            {"BTC/USD": 50000.0, "ETH/USD": 3000.0}
        )
        assert "cash" in snapshot
        assert snapshot["cash"] == 2000.0

    def test_snapshot_contains_positions(self, portfolio_with_positions):
        """Snapshot includes all positions."""
        snapshot = portfolio_with_positions.get_portfolio_snapshot(
            {"BTC/USD": 55000.0, "ETH/USD": 3500.0}
        )
        assert "positions" in snapshot
        assert "BTC/USD" in snapshot["positions"]
        assert "ETH/USD" in snapshot["positions"]

    def test_snapshot_position_details(self, portfolio_with_positions):
        """Snapshot positions have correct details."""
        snapshot = portfolio_with_positions.get_portfolio_snapshot(
            {"BTC/USD": 55000.0, "ETH/USD": 3000.0}
        )
        btc_pos = snapshot["positions"]["BTC/USD"]
        assert btc_pos["quantity"] == 0.1
        assert btc_pos["current_price"] == 55000.0
        assert btc_pos["unrealized_pnl"] == 500.0  # 0.1 * (55000 - 50000)

    def test_snapshot_total_value(self, portfolio_with_positions):
        """Snapshot includes total value."""
        snapshot = portfolio_with_positions.get_portfolio_snapshot(
            {"BTC/USD": 55000.0, "ETH/USD": 3500.0}
        )
        # Cash 2000 + BTC 5500 + ETH 3500 = 11000
        assert snapshot["total_value"] == 11000.0

    def test_snapshot_has_timestamp(self, portfolio_with_positions):
        """Snapshot includes timestamp."""
        snapshot = portfolio_with_positions.get_portfolio_snapshot({})
        assert "timestamp" in snapshot


class TestFractionalQuantities:
    """Tests for fractional quantity support."""

    @pytest.fixture
    def portfolio(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield DemoPortfolioState(
                initial_cash=10000.0, state_dir=Path(tmpdir), auto_save=False
            )

    def test_very_small_quantity(self, portfolio):
        """Support very small quantities."""
        portfolio.add_or_update_position(
            "BTC/USD", Decimal("0.00001"), 50000.0
        )
        pos = portfolio.get_position("BTC/USD")
        assert pos.quantity == Decimal("0.00001")
        assert pos.cost_basis == pytest.approx(0.5, rel=0.01)

    def test_precise_averaging(self, portfolio):
        """Precise averaging with fractional quantities."""
        portfolio.add_or_update_position("BTC/USD", Decimal("0.00001"), 50000.0)
        portfolio.add_or_update_position("BTC/USD", Decimal("0.00002"), 51000.0)

        pos = portfolio.get_position("BTC/USD")
        assert pos.quantity == Decimal("0.00003")
        # Avg = (0.00001 * 50000 + 0.00002 * 51000) / 0.00003
        # = (0.5 + 1.02) / 0.00003 = 50666.67
        assert pos.avg_entry_price == pytest.approx(50666.67, rel=0.01)
