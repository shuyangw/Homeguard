"""
End-to-end integration tests for demo trading system.

Tests complete trading scenarios with simulated market data,
verifying data flow, P&L calculations, and state persistence.
"""

import pytest
import tempfile
from decimal import Decimal
from pathlib import Path
from datetime import datetime, timedelta

from src.trading.brokers.interfaces.base import OrderSide
from src.trading.demo import (
    DemoBroker,
    DemoPortfolioState,
    ExecutionSimulator,
    DemoTradeLogger,
)
from src.streaming.crypto_buffer import CryptoBarBuffer
from src.streaming.types import Bar


def create_isolated_broker(initial_cash: float = 10000.0, **kwargs) -> DemoBroker:
    """Create a broker with isolated temp storage for test isolation."""
    tmpdir = tempfile.mkdtemp()
    broker = DemoBroker(initial_cash=initial_cash, **kwargs)
    broker._portfolio = DemoPortfolioState(
        initial_cash=initial_cash,
        state_dir=Path(tmpdir),
        auto_save=False,
    )
    broker._trade_logger = DemoTradeLogger(log_dir=Path(tmpdir) / "trades")
    return broker


class TestBuyHoldSellCycle:
    """Tests for complete buy-hold-sell trading cycle."""

    @pytest.fixture
    def demo_broker(self):
        """Create broker with mock market data."""
        broker = create_isolated_broker(initial_cash=10000.0, slippage_bps=5, fee_bps=10)

        # Add initial price
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

    def test_buy_hold_sell_profit(self, demo_broker):
        """
        1. Buy 0.01 BTC at $50,000
        2. Verify position created
        3. Price rises to $55,000
        4. Sell position
        5. Verify realized P&L positive
        """
        # 1. Buy
        order = demo_broker.place_crypto_order(
            symbol="BTC/USD",
            quantity=Decimal("0.01"),
            side=OrderSide.BUY,
        )
        assert order["status"] == "filled"

        # 2. Verify position
        pos = demo_broker.get_crypto_position("BTC/USD")
        assert pos is not None
        assert pos["quantity"] == Decimal("0.01")

        # 3. Update price (simulate price rise)
        new_bar = Bar(
            symbol="BTC/USD",
            timestamp=datetime.now(),
            open=55000.0,
            high=55100.0,
            low=54900.0,
            close=55000.0,
            volume=100.0,
        )
        demo_broker._bar_buffer.add_bar(new_bar)

        # Check unrealized P&L before selling
        pos = demo_broker.get_crypto_position("BTC/USD")
        assert pos["unrealized_pnl"] > 0

        # 4. Sell
        sell_order = demo_broker.close_crypto_position("BTC/USD")
        assert sell_order["status"] == "filled"

        # 5. Verify P&L
        assert demo_broker.realized_pnl > 0
        assert demo_broker.get_crypto_position("BTC/USD") is None

    def test_buy_hold_sell_loss(self, demo_broker):
        """Complete cycle with loss scenario."""
        # Buy
        demo_broker.place_crypto_order(
            symbol="BTC/USD",
            quantity=Decimal("0.01"),
            side=OrderSide.BUY,
        )

        # Price drops
        new_bar = Bar(
            symbol="BTC/USD",
            timestamp=datetime.now(),
            open=45000.0,
            high=45100.0,
            low=44900.0,
            close=45000.0,
            volume=100.0,
        )
        demo_broker._bar_buffer.add_bar(new_bar)

        # Sell at loss
        demo_broker.close_crypto_position("BTC/USD")

        # Verify loss (considering fees too)
        assert demo_broker.realized_pnl < 0


class TestMultipleBuysAveraging:
    """Tests for multiple buys and price averaging."""

    @pytest.fixture
    def demo_broker(self):
        broker = create_isolated_broker(initial_cash=100000.0, slippage_bps=0, fee_bps=0)
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

    def test_multiple_buys_average_price(self, demo_broker):
        """
        1. Buy 0.01 BTC at $50,000
        2. Buy 0.01 BTC at $52,000
        3. Verify avg_entry_price = $51,000
        4. Verify quantity = 0.02 BTC
        """
        # Buy at 50000
        demo_broker.place_crypto_order("BTC/USD", Decimal("0.01"), OrderSide.BUY)

        # Update price
        bar = Bar(
            symbol="BTC/USD",
            timestamp=datetime.now(),
            open=52000.0,
            high=52100.0,
            low=51900.0,
            close=52000.0,
            volume=100.0,
        )
        demo_broker._bar_buffer.add_bar(bar)

        # Buy at 52000
        demo_broker.place_crypto_order("BTC/USD", Decimal("0.01"), OrderSide.BUY)

        pos = demo_broker.get_crypto_position("BTC/USD")
        assert pos["quantity"] == Decimal("0.02")
        assert pos["avg_entry_price"] == pytest.approx(51000.0, rel=0.01)


class TestPartialPositionClose:
    """Tests for partial position closing."""

    @pytest.fixture
    def demo_broker(self):
        broker = create_isolated_broker(initial_cash=100000.0, slippage_bps=0, fee_bps=0)
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

    def test_partial_position_close(self, demo_broker):
        """
        1. Buy 0.1 BTC
        2. Sell 0.05 BTC
        3. Verify 0.05 BTC remaining
        4. Verify partial P&L realized
        """
        demo_broker.place_crypto_order("BTC/USD", Decimal("0.1"), OrderSide.BUY)

        # Price rises
        bar = Bar(
            symbol="BTC/USD",
            timestamp=datetime.now(),
            open=55000.0,
            high=55100.0,
            low=54900.0,
            close=55000.0,
            volume=100.0,
        )
        demo_broker._bar_buffer.add_bar(bar)

        # Sell half
        demo_broker.close_crypto_position("BTC/USD", Decimal("0.05"))

        # Verify remaining
        pos = demo_broker.get_crypto_position("BTC/USD")
        assert pos["quantity"] == Decimal("0.05")

        # Verify P&L realized (0.05 * (55000 - 50000) = 250)
        assert demo_broker.realized_pnl == pytest.approx(250.0, rel=0.01)


class TestMultiAssetScenarios:
    """Tests for multi-asset portfolio scenarios."""

    @pytest.fixture
    def demo_broker(self):
        broker = create_isolated_broker(initial_cash=100000.0, slippage_bps=5, fee_bps=10)

        # Add prices for multiple symbols
        symbols_prices = [
            ("BTC/USD", 50000.0),
            ("ETH/USD", 3000.0),
            ("SOL/USD", 100.0),
        ]
        for symbol, price in symbols_prices:
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

        return broker

    def test_multiple_positions(self, demo_broker):
        """
        1. Buy BTC, ETH, SOL
        2. Verify 3 separate positions
        """
        demo_broker.place_crypto_order("BTC/USD", Decimal("0.1"), OrderSide.BUY)
        demo_broker.place_crypto_order("ETH/USD", Decimal("1.0"), OrderSide.BUY)
        demo_broker.place_crypto_order("SOL/USD", Decimal("10.0"), OrderSide.BUY)

        positions = demo_broker.get_crypto_positions()
        assert len(positions) == 3

        symbols = {p["symbol"] for p in positions}
        assert symbols == {"BTC/USD", "ETH/USD", "SOL/USD"}

    def test_close_one_keep_others(self, demo_broker):
        """Close ETH only, BTC and SOL remain."""
        demo_broker.place_crypto_order("BTC/USD", Decimal("0.1"), OrderSide.BUY)
        demo_broker.place_crypto_order("ETH/USD", Decimal("1.0"), OrderSide.BUY)
        demo_broker.place_crypto_order("SOL/USD", Decimal("10.0"), OrderSide.BUY)

        demo_broker.close_crypto_position("ETH/USD")

        positions = demo_broker.get_crypto_positions()
        assert len(positions) == 2
        symbols = {p["symbol"] for p in positions}
        assert symbols == {"BTC/USD", "SOL/USD"}


class TestFinancialAccuracy:
    """Tests for financial calculation accuracy."""

    @pytest.fixture
    def demo_broker(self):
        """Broker with fixed slippage and fees for predictable testing."""
        broker = create_isolated_broker(
            initial_cash=10000.0,
            slippage_bps=5,  # 0.05%
            fee_bps=10,  # 0.10%
        )
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

    def test_pnl_calculation_accuracy(self, demo_broker):
        """
        Scenario:
        - Buy 0.1 BTC at $50,000 (notional: $5,000)
        - Slippage: 5bps = $2.50 (fill at $50,025)
        - Fee: 10bps on fill = $5.00
        - Actual cost: $5,002.50 + $5.00 = $5,007.50
        """
        order = demo_broker.place_crypto_order(
            symbol="BTC/USD",
            quantity=Decimal("0.1"),
            side=OrderSide.BUY,
        )

        # Verify fill price includes slippage
        assert order["avg_fill_price"] == pytest.approx(50025.0, rel=0.001)

        # Verify fees
        assert order["fees"] == pytest.approx(5.0025, rel=0.01)

        # Verify cash deducted correctly
        # 0.1 * 50025 + 5.0025 = 5007.5025
        expected_cash = 10000.0 - 5007.5025
        assert demo_broker.cash == pytest.approx(expected_cash, rel=0.01)

    def test_portfolio_value_accuracy(self, demo_broker):
        """
        - Cash: $5,000
        - Position: 0.1 BTC at current price $50,000 = $5,000
        - Total value: ~$10,000 (minus fees)
        """
        demo_broker.place_crypto_order("BTC/USD", Decimal("0.1"), OrderSide.BUY)

        value = demo_broker.get_portfolio_value()
        # Should be close to 10000, minus the fees and slippage costs
        assert value < 10000.0  # Lost money to fees
        assert value > 9980.0  # But not much

    def test_fractional_precision(self, demo_broker):
        """Very small quantities don't have floating point errors."""
        demo_broker.place_crypto_order(
            symbol="BTC/USD",
            quantity=Decimal("0.00001"),
            side=OrderSide.BUY,
        )

        pos = demo_broker.get_crypto_position("BTC/USD")
        assert pos["quantity"] == Decimal("0.00001")


class TestSlippageVerification:
    """Tests to verify slippage is applied correctly."""

    def test_buy_slippage_increases_price(self):
        """Buy orders fill above market price."""
        broker = create_isolated_broker(initial_cash=100000.0, slippage_bps=10, fee_bps=0)
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
        # Use fixed slippage by accessing simulator directly
        broker._execution_sim.randomize_slippage = False

        order = broker.place_crypto_order("BTC/USD", Decimal("0.1"), OrderSide.BUY)

        # 10 bps = 0.1%, so fill = 50000 * 1.001 = 50050
        assert order["avg_fill_price"] == pytest.approx(50050.0, rel=0.001)

    def test_sell_slippage_decreases_price(self):
        """Sell orders fill below market price."""
        broker = create_isolated_broker(initial_cash=100000.0, slippage_bps=10, fee_bps=0)
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
        broker._execution_sim.randomize_slippage = False

        # Buy first (at higher price due to slippage)
        broker.place_crypto_order("BTC/USD", Decimal("0.1"), OrderSide.BUY)

        # Then sell
        sell_order = broker.place_crypto_order("BTC/USD", Decimal("0.1"), OrderSide.SELL)

        # 10 bps = 0.1%, so fill = 50000 * 0.999 = 49950
        assert sell_order["avg_fill_price"] == pytest.approx(49950.0, rel=0.001)


class TestStatePersistence:
    """Tests for state persistence across restarts."""

    def test_state_survives_restart(self):
        """
        1. Create broker, buy position
        2. Save state
        3. Create new broker instance
        4. Load state
        5. Verify position exists
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            state_dir = Path(tmpdir)

            # First broker session
            broker1 = DemoBroker(initial_cash=10000.0)
            broker1._portfolio = DemoPortfolioState(
                initial_cash=10000.0, state_dir=state_dir, auto_save=True
            )

            # Add price and buy
            bar = Bar(
                symbol="BTC/USD",
                timestamp=datetime.now(),
                open=50000.0,
                high=50100.0,
                low=49900.0,
                close=50000.0,
                volume=100.0,
            )
            broker1._bar_buffer.add_bar(bar)
            broker1.place_crypto_order("BTC/USD", Decimal("0.1"), OrderSide.BUY)

            # Check position exists
            pos1 = broker1.get_crypto_position("BTC/USD")
            assert pos1 is not None

            # Simulate restart - create new broker
            broker2 = DemoBroker(initial_cash=99999.0)  # Different initial
            broker2._portfolio = DemoPortfolioState(
                initial_cash=99999.0, state_dir=state_dir, auto_save=True
            )

            # Position should be loaded from saved state
            pos2 = broker2._portfolio.get_position("BTC/USD")
            assert pos2 is not None
            assert pos2.quantity == Decimal("0.1")


class TestTradeLogging:
    """Tests for trade log functionality."""

    def test_trades_logged(self):
        """Trades are written to log file."""
        with tempfile.TemporaryDirectory() as tmpdir:
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

            # Execute trades
            broker.place_crypto_order("BTC/USD", Decimal("0.1"), OrderSide.BUY)
            broker.place_crypto_order("BTC/USD", Decimal("0.05"), OrderSide.SELL)

            # Check trades logged
            trades = broker._trade_logger.get_trades()
            assert len(trades) == 2

            # Verify trade details
            buy_trade = next(t for t in trades if t["side"] == "buy")
            assert buy_trade["symbol"] == "BTC/USD"
            assert "portfolio_snapshot" in buy_trade


class TestDataIntegrity:
    """Tests for data flow integrity."""

    def test_buffer_to_quote_consistency(self):
        """Price from buffer matches quote."""
        broker = create_isolated_broker(initial_cash=10000.0)

        bar = Bar(
            symbol="BTC/USD",
            timestamp=datetime.now(),
            open=50000.0,
            high=50100.0,
            low=49900.0,
            close=50050.0,  # Specific close price
            volume=100.0,
        )
        broker._bar_buffer.add_bar(bar)

        quote = broker.get_crypto_quote("BTC/USD")
        assert quote["last"] == 50050.0

    def test_position_value_matches_quote(self):
        """Position market value uses current quote."""
        broker = create_isolated_broker(initial_cash=100000.0, slippage_bps=0, fee_bps=0)

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

        # Update price
        new_bar = Bar(
            symbol="BTC/USD",
            timestamp=datetime.now(),
            open=55000.0,
            high=55100.0,
            low=54900.0,
            close=55000.0,
            volume=100.0,
        )
        broker._bar_buffer.add_bar(new_bar)

        pos = broker.get_crypto_position("BTC/USD")
        # 0.1 * 55000 = 5500
        assert pos["market_value"] == pytest.approx(5500.0, rel=0.01)


class TestEdgeCases:
    """Tests for edge cases and error conditions."""

    @pytest.fixture
    def demo_broker(self):
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

    def test_buy_exact_cash_amount(self, demo_broker):
        """Can buy exactly as much as cash allows."""
        # Cash = 10000, price ~50000, so max qty ~0.2 (minus fees)
        # With fees, need to leave some buffer
        order = demo_broker.place_crypto_order(
            symbol="BTC/USD",
            quantity=Decimal("0.19"),  # Slightly under max
            side=OrderSide.BUY,
        )
        assert order["status"] == "filled"

    def test_sell_exact_position(self, demo_broker):
        """Selling exact position quantity works."""
        demo_broker.place_crypto_order("BTC/USD", Decimal("0.1"), OrderSide.BUY)

        # Get exact quantity (might be slightly different due to Decimal)
        pos = demo_broker.get_crypto_position("BTC/USD")
        exact_qty = pos["quantity"]

        # Sell exact amount
        demo_broker.place_crypto_order("BTC/USD", exact_qty, OrderSide.SELL)

        # Position should be gone
        assert demo_broker.get_crypto_position("BTC/USD") is None

    def test_many_small_trades(self, demo_broker):
        """System handles many small trades."""
        for _ in range(20):
            demo_broker.place_crypto_order("BTC/USD", Decimal("0.001"), OrderSide.BUY)

        pos = demo_broker.get_crypto_position("BTC/USD")
        assert pos["quantity"] == Decimal("0.02")

        orders = demo_broker.get_orders()
        assert len(orders) == 20
