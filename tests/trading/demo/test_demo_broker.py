"""
Unit tests for DemoBroker.

Tests order placement, position management, and integration.
"""

import pytest
import tempfile
from decimal import Decimal
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch

from src.trading.brokers.interfaces.base import (
    OrderSide,
    OrderType,
    InvalidOrderError,
    InsufficientFundsError,
    NoPositionError,
)
from src.trading.demo.demo_broker import DemoBroker
from src.trading.demo.portfolio_state import DemoPortfolioState
from src.trading.demo.trade_logger import DemoTradeLogger
from src.streaming.types import Bar


def create_isolated_broker(initial_cash: float = 10000.0, **kwargs) -> DemoBroker:
    """Create a broker with isolated temp storage for test isolation."""
    import tempfile
    tmpdir = tempfile.mkdtemp()
    broker = DemoBroker(initial_cash=initial_cash, **kwargs)
    broker._portfolio = DemoPortfolioState(
        initial_cash=initial_cash,
        state_dir=Path(tmpdir),
        auto_save=False,
    )
    broker._trade_logger = DemoTradeLogger(log_dir=Path(tmpdir) / "trades")
    return broker


class TestDemoBrokerInit:
    """Tests for DemoBroker initialization."""

    def test_init_default_cash(self):
        """Default initial cash is 10000."""
        broker = create_isolated_broker()
        assert broker.cash == 10000.0

    def test_init_custom_cash(self):
        """Custom initial cash."""
        broker = create_isolated_broker(initial_cash=50000.0)
        assert broker.cash == 50000.0

    def test_init_no_streaming(self):
        """Streaming not started by default."""
        broker = create_isolated_broker()
        assert broker.is_streaming() is False

    def test_init_no_positions(self):
        """No positions initially."""
        broker = create_isolated_broker()
        assert len(broker.get_crypto_positions()) == 0


class TestOrderPlacement:
    """Tests for placing orders."""

    @pytest.fixture
    def broker_with_price(self):
        """Broker with price data for BTC."""
        broker = create_isolated_broker(initial_cash=100000.0)
        # Add a bar to the buffer so we have a price
        bar = Bar(
            symbol="BTC/USD",
            timestamp=datetime.now(),
            open=50000.0,
            high=50100.0,
            low=49900.0,
            close=50000.0,
            volume=100.0,
        )
        broker._bar_buffer.add_bar(bar)
        return broker

    def test_buy_order_creates_position(self, broker_with_price):
        """Buy order creates new position."""
        order = broker_with_price.place_crypto_order(
            symbol="BTC/USD",
            quantity=Decimal("0.1"),
            side=OrderSide.BUY,
        )
        pos = broker_with_price.get_crypto_position("BTC/USD")
        assert pos is not None
        assert pos["quantity"] == Decimal("0.1")

    def test_buy_order_deducts_cash(self, broker_with_price):
        """Buy order deducts cash including fees."""
        initial_cash = broker_with_price.cash
        broker_with_price.place_crypto_order(
            symbol="BTC/USD",
            quantity=Decimal("0.1"),
            side=OrderSide.BUY,
        )
        # Should have less cash (price * qty + fees)
        assert broker_with_price.cash < initial_cash

    def test_sell_order_adds_cash(self, broker_with_price):
        """Sell order adds cash minus fees."""
        # First buy
        broker_with_price.place_crypto_order(
            symbol="BTC/USD",
            quantity=Decimal("0.1"),
            side=OrderSide.BUY,
        )
        cash_after_buy = broker_with_price.cash

        # Then sell
        broker_with_price.place_crypto_order(
            symbol="BTC/USD",
            quantity=Decimal("0.1"),
            side=OrderSide.SELL,
        )
        assert broker_with_price.cash > cash_after_buy

    def test_order_returns_dict(self, broker_with_price):
        """Order returns dict with order details."""
        order = broker_with_price.place_crypto_order(
            symbol="BTC/USD",
            quantity=Decimal("0.1"),
            side=OrderSide.BUY,
        )
        assert "order_id" in order
        assert "symbol" in order
        assert "quantity" in order
        assert "side" in order
        assert "status" in order
        assert "filled_avg_price" in order

    def test_order_has_execution_details(self, broker_with_price):
        """Order includes execution details."""
        order = broker_with_price.place_crypto_order(
            symbol="BTC/USD",
            quantity=Decimal("0.1"),
            side=OrderSide.BUY,
        )
        assert "fees" in order
        assert "slippage_cost" in order
        assert "latency_ms" in order

    def test_insufficient_funds_rejected(self, broker_with_price):
        """Order rejected if insufficient funds."""
        with pytest.raises(InsufficientFundsError):
            broker_with_price.place_crypto_order(
                symbol="BTC/USD",
                quantity=Decimal("100.0"),  # Way more than we can afford
                side=OrderSide.BUY,
            )

    def test_sell_without_position_rejected(self, broker_with_price):
        """Sell without position raises NoPositionError."""
        with pytest.raises(NoPositionError):
            broker_with_price.place_crypto_order(
                symbol="BTC/USD",
                quantity=Decimal("0.1"),
                side=OrderSide.SELL,
            )

    def test_oversell_rejected(self, broker_with_price):
        """Cannot sell more than held."""
        broker_with_price.place_crypto_order(
            symbol="BTC/USD",
            quantity=Decimal("0.1"),
            side=OrderSide.BUY,
        )
        with pytest.raises(InvalidOrderError):
            broker_with_price.place_crypto_order(
                symbol="BTC/USD",
                quantity=Decimal("0.2"),  # More than we have
                side=OrderSide.SELL,
            )

    def test_zero_quantity_rejected(self, broker_with_price):
        """Zero quantity order rejected."""
        with pytest.raises(InvalidOrderError):
            broker_with_price.place_crypto_order(
                symbol="BTC/USD",
                quantity=Decimal("0"),
                side=OrderSide.BUY,
            )

    def test_negative_quantity_rejected(self, broker_with_price):
        """Negative quantity order rejected."""
        with pytest.raises(InvalidOrderError):
            broker_with_price.place_crypto_order(
                symbol="BTC/USD",
                quantity=Decimal("-0.1"),
                side=OrderSide.BUY,
            )

    def test_no_price_rejected(self):
        """Order rejected if no price available."""
        broker = DemoBroker(initial_cash=100000.0)
        # No bars in buffer
        with pytest.raises(InvalidOrderError, match="No price available"):
            broker.place_crypto_order(
                symbol="BTC/USD",
                quantity=Decimal("0.1"),
                side=OrderSide.BUY,
            )


class TestPositionManagement:
    """Tests for position management."""

    @pytest.fixture
    def broker_with_positions(self):
        """Broker with multiple positions."""
        broker = create_isolated_broker(initial_cash=100000.0)

        # Add prices
        for symbol, price in [("BTC/USD", 50000.0), ("ETH/USD", 3000.0)]:
            bar = Bar(
                symbol=symbol,
                timestamp=datetime.now(),
                open=price,
                high=price * 1.01,
                low=price * 0.99,
                close=price,
                volume=100.0,
            )
            broker._bar_buffer.add_bar(bar)

        # Buy positions
        broker.place_crypto_order("BTC/USD", Decimal("0.1"), OrderSide.BUY)
        broker.place_crypto_order("ETH/USD", Decimal("1.0"), OrderSide.BUY)

        return broker

    def test_get_positions_empty(self):
        """get_crypto_positions returns empty list when no positions."""
        broker = create_isolated_broker()
        assert broker.get_crypto_positions() == []

    def test_get_positions_with_holdings(self, broker_with_positions):
        """get_crypto_positions returns all positions."""
        positions = broker_with_positions.get_crypto_positions()
        assert len(positions) == 2
        symbols = {p["symbol"] for p in positions}
        assert symbols == {"BTC/USD", "ETH/USD"}

    def test_get_single_position(self, broker_with_positions):
        """get_crypto_position returns specific position."""
        pos = broker_with_positions.get_crypto_position("BTC/USD")
        assert pos is not None
        assert pos["symbol"] == "BTC/USD"
        assert pos["quantity"] == Decimal("0.1")

    def test_get_missing_position(self, broker_with_positions):
        """get_crypto_position returns None for missing symbol."""
        pos = broker_with_positions.get_crypto_position("SOL/USD")
        assert pos is None

    def test_close_position_full(self, broker_with_positions):
        """close_crypto_position closes entire position."""
        broker_with_positions.close_crypto_position("BTC/USD")
        pos = broker_with_positions.get_crypto_position("BTC/USD")
        assert pos is None

    def test_close_position_partial(self, broker_with_positions):
        """close_crypto_position can close partial."""
        broker_with_positions.close_crypto_position("BTC/USD", Decimal("0.05"))
        pos = broker_with_positions.get_crypto_position("BTC/USD")
        assert pos is not None
        assert pos["quantity"] == Decimal("0.05")

    def test_close_all_positions(self, broker_with_positions):
        """close_all_crypto_positions closes everything."""
        results = broker_with_positions.close_all_crypto_positions()
        assert len(results) == 2
        assert len(broker_with_positions.get_crypto_positions()) == 0


class TestQuotesAndBalances:
    """Tests for quotes and account balance."""

    @pytest.fixture
    def broker_with_data(self):
        broker = create_isolated_broker(initial_cash=10000.0)
        bar = Bar(
            symbol="BTC/USD",
            timestamp=datetime.now(),
            open=50000.0,
            high=50100.0,
            low=49900.0,
            close=50000.0,
            volume=100.0,
        )
        broker._bar_buffer.add_bar(bar)
        return broker

    def test_get_quote_has_data(self, broker_with_data):
        """get_crypto_quote returns quote data."""
        quote = broker_with_data.get_crypto_quote("BTC/USD")
        assert quote["symbol"] == "BTC/USD"
        assert quote["last"] == 50000.0
        assert quote["available"] is True

    def test_get_quote_has_bid_ask(self, broker_with_data):
        """Quote includes bid/ask spread."""
        quote = broker_with_data.get_crypto_quote("BTC/USD")
        assert "bid" in quote
        assert "ask" in quote
        assert quote["bid"] < quote["last"] < quote["ask"]

    def test_get_quote_missing_symbol(self, broker_with_data):
        """Quote for missing symbol has available=False."""
        quote = broker_with_data.get_crypto_quote("UNKNOWN/USD")
        assert quote["available"] is False
        assert quote["bid"] == 0.0

    def test_get_balance(self, broker_with_data):
        """get_crypto_balance returns balance info."""
        balance = broker_with_data.get_crypto_balance()
        assert "total" in balance
        assert "available" in balance
        assert "held" in balance
        assert balance["available"] == 10000.0


class TestOrderTracking:
    """Tests for order history tracking."""

    @pytest.fixture
    def broker_with_orders(self):
        broker = create_isolated_broker(initial_cash=100000.0)
        bar = Bar(
            symbol="BTC/USD",
            timestamp=datetime.now(),
            open=50000.0,
            high=50100.0,
            low=49900.0,
            close=50000.0,
            volume=100.0,
        )
        broker._bar_buffer.add_bar(bar)

        # Place some orders
        broker.place_crypto_order("BTC/USD", Decimal("0.1"), OrderSide.BUY)
        broker.place_crypto_order("BTC/USD", Decimal("0.05"), OrderSide.SELL)
        return broker

    def test_get_order_by_id(self, broker_with_orders):
        """get_order retrieves order by ID."""
        orders = broker_with_orders.get_orders()
        order_id = orders[0]["order_id"]
        order = broker_with_orders.get_order(order_id)
        assert order["order_id"] == order_id

    def test_get_orders_list(self, broker_with_orders):
        """get_orders returns all orders."""
        orders = broker_with_orders.get_orders()
        assert len(orders) == 2

    def test_cancel_order_noop(self, broker_with_orders):
        """cancel_order returns False (market orders fill immediately)."""
        orders = broker_with_orders.get_orders()
        result = broker_with_orders.cancel_order(orders[0]["order_id"])
        assert result is False


class TestAccountSummary:
    """Tests for account summary and portfolio value."""

    @pytest.fixture
    def broker_with_position(self):
        broker = create_isolated_broker(initial_cash=10000.0)
        bar = Bar(
            symbol="BTC/USD",
            timestamp=datetime.now(),
            open=50000.0,
            high=50100.0,
            low=49900.0,
            close=50000.0,
            volume=100.0,
        )
        broker._bar_buffer.add_bar(bar)
        broker.place_crypto_order("BTC/USD", Decimal("0.1"), OrderSide.BUY)
        return broker

    def test_get_portfolio_value(self, broker_with_position):
        """get_portfolio_value calculates total value."""
        value = broker_with_position.get_portfolio_value()
        # Cash (~5000) + position value (~5000) = ~10000
        assert value > 0

    def test_get_portfolio_snapshot(self, broker_with_position):
        """get_portfolio_snapshot returns complete state."""
        snapshot = broker_with_position.get_portfolio_snapshot()
        assert "cash" in snapshot
        assert "positions" in snapshot
        assert "total_value" in snapshot

    def test_get_account_summary(self, broker_with_position):
        """get_account_summary includes all details."""
        summary = broker_with_position.get_account_summary()
        assert "streaming" in summary
        assert "bars_buffered" in summary
        assert "orders_placed" in summary
        assert "execution_config" in summary


class TestFractionalQuantities:
    """Tests for fractional quantity support."""

    @pytest.fixture
    def broker(self):
        broker = create_isolated_broker(initial_cash=100000.0)
        bar = Bar(
            symbol="BTC/USD",
            timestamp=datetime.now(),
            open=50000.0,
            high=50100.0,
            low=49900.0,
            close=50000.0,
            volume=100.0,
        )
        broker._bar_buffer.add_bar(bar)
        return broker

    def test_buy_fractional_btc(self, broker):
        """Can buy 0.001 BTC."""
        order = broker.place_crypto_order(
            symbol="BTC/USD",
            quantity=Decimal("0.001"),
            side=OrderSide.BUY,
        )
        pos = broker.get_crypto_position("BTC/USD")
        assert pos["quantity"] == Decimal("0.001")

    def test_very_small_quantity(self, broker):
        """Can trade 0.00001 BTC."""
        order = broker.place_crypto_order(
            symbol="BTC/USD",
            quantity=Decimal("0.00001"),
            side=OrderSide.BUY,
        )
        pos = broker.get_crypto_position("BTC/USD")
        assert pos["quantity"] == Decimal("0.00001")


class TestSymbolNormalization:
    """Tests for symbol normalization."""

    def test_normalize_btc_usd(self):
        broker = DemoBroker()
        assert broker.normalize_symbol("BTC/USD") == "BTC/USD"

    def test_normalize_btc_dash_usd(self):
        broker = DemoBroker()
        assert broker.normalize_symbol("BTC-USD") == "BTC/USD"

    def test_normalize_lowercase(self):
        broker = DemoBroker()
        assert broker.normalize_symbol("btc/usd") == "BTC/USD"

    def test_normalize_no_slash(self):
        broker = DemoBroker()
        # BTCUSD ends with USD, so no /USD is added
        assert broker.normalize_symbol("BTCUSD") == "BTCUSD"

    def test_normalize_no_slash_no_usd(self):
        broker = DemoBroker()
        # BTC doesn't end with USD, so /USD is added
        assert broker.normalize_symbol("BTC") == "BTC/USD"


class TestReset:
    """Tests for portfolio reset."""

    def test_reset_clears_positions(self):
        broker = create_isolated_broker(initial_cash=100000.0)
        bar = Bar(
            symbol="BTC/USD",
            timestamp=datetime.now(),
            open=50000.0,
            high=50100.0,
            low=49900.0,
            close=50000.0,
            volume=100.0,
        )
        broker._bar_buffer.add_bar(bar)
        broker.place_crypto_order("BTC/USD", Decimal("0.1"), OrderSide.BUY)

        broker.reset_portfolio(initial_cash=20000.0)

        assert broker.cash == 20000.0
        assert len(broker.get_crypto_positions()) == 0

    def test_reset_clears_orders(self):
        broker = create_isolated_broker(initial_cash=100000.0)
        bar = Bar(
            symbol="BTC/USD",
            timestamp=datetime.now(),
            open=50000.0,
            high=50100.0,
            low=49900.0,
            close=50000.0,
            volume=100.0,
        )
        broker._bar_buffer.add_bar(bar)
        broker.place_crypto_order("BTC/USD", Decimal("0.1"), OrderSide.BUY)

        broker.reset_portfolio()

        assert len(broker.get_orders()) == 0
