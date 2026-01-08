"""
Unit tests for ExecutionSimulator.

Tests slippage, fee, and latency simulation.
"""

import pytest
from decimal import Decimal
from datetime import datetime

from src.trading.brokers.interfaces.base import OrderSide
from src.trading.demo.execution_simulator import ExecutionSimulator, ExecutionResult


class TestExecutionResult:
    """Tests for ExecutionResult dataclass."""

    def test_total_cost(self):
        """Total cost sums slippage and fees."""
        result = ExecutionResult(
            fill_price=50025.0,
            fill_qty=Decimal("0.1"),
            fees=5.0,
            slippage_cost=2.5,
            latency_ms=100,
            timestamp=datetime.now(),
            market_price=50000.0,
        )
        assert result.total_cost == 7.5

    def test_effective_price(self):
        """Effective price includes fees."""
        result = ExecutionResult(
            fill_price=50025.0,
            fill_qty=Decimal("0.1"),
            fees=5.0,
            slippage_cost=2.5,
            latency_ms=100,
            timestamp=datetime.now(),
            market_price=50000.0,
        )
        # 50025 + (5 / 0.1) = 50025 + 50 = 50075
        assert result.effective_price == 50075.0


class TestExecutionSimulator:
    """Tests for ExecutionSimulator class."""

    @pytest.fixture
    def simulator(self):
        """Create simulator with fixed (non-random) slippage."""
        return ExecutionSimulator(
            slippage_bps=5.0,
            fee_bps=10.0,
            latency_min_ms=50,
            latency_max_ms=200,
            randomize_slippage=False,
        )

    def test_buy_slippage_increases_price(self, simulator):
        """Buy orders fill at higher price."""
        result = simulator.simulate_execution(
            side=OrderSide.BUY,
            quantity=Decimal("0.1"),
            market_price=50000.0,
        )
        assert result.fill_price > 50000.0

    def test_sell_slippage_decreases_price(self, simulator):
        """Sell orders fill at lower price."""
        result = simulator.simulate_execution(
            side=OrderSide.SELL,
            quantity=Decimal("0.1"),
            market_price=50000.0,
        )
        assert result.fill_price < 50000.0

    def test_slippage_magnitude_5bps(self, simulator):
        """5 bps slippage = 0.05% price impact."""
        result = simulator.simulate_execution(
            side=OrderSide.BUY,
            quantity=Decimal("0.1"),
            market_price=50000.0,
        )
        # 5 bps = 0.0005, so fill = 50000 * 1.0005 = 50025
        assert result.fill_price == pytest.approx(50025.0, rel=0.0001)

    def test_fee_calculation(self, simulator):
        """Fees calculated on fill notional."""
        result = simulator.simulate_execution(
            side=OrderSide.BUY,
            quantity=Decimal("0.1"),
            market_price=50000.0,
        )
        # Notional at fill = 0.1 * 50025 = 5002.5
        # Fee = 5002.5 * 10bps = 5002.5 * 0.001 = 5.0025
        assert result.fees == pytest.approx(5.0025, rel=0.01)

    def test_slippage_cost_calculated(self, simulator):
        """Slippage cost = difference from market price."""
        result = simulator.simulate_execution(
            side=OrderSide.BUY,
            quantity=Decimal("0.1"),
            market_price=50000.0,
        )
        # Slippage = (50025 - 50000) * 0.1 = 2.5
        assert result.slippage_cost == pytest.approx(2.5, rel=0.01)

    def test_latency_in_range(self, simulator):
        """Latency within configured range."""
        result = simulator.simulate_execution(
            side=OrderSide.BUY,
            quantity=Decimal("0.1"),
            market_price=50000.0,
        )
        assert 50 <= result.latency_ms <= 200

    def test_fill_qty_matches_input(self, simulator):
        """Fill quantity matches requested quantity."""
        qty = Decimal("0.12345")
        result = simulator.simulate_execution(
            side=OrderSide.BUY,
            quantity=qty,
            market_price=50000.0,
        )
        assert result.fill_qty == qty

    def test_market_price_recorded(self, simulator):
        """Market price recorded in result."""
        result = simulator.simulate_execution(
            side=OrderSide.BUY,
            quantity=Decimal("0.1"),
            market_price=50000.0,
        )
        assert result.market_price == 50000.0

    def test_timestamp_present(self, simulator):
        """Timestamp is present and recent."""
        result = simulator.simulate_execution(
            side=OrderSide.BUY,
            quantity=Decimal("0.1"),
            market_price=50000.0,
        )
        assert isinstance(result.timestamp, datetime)


class TestZeroSlippageAndFees:
    """Tests for zero slippage/fee configuration."""

    def test_zero_slippage(self):
        """Zero slippage means fill at market price."""
        sim = ExecutionSimulator(
            slippage_bps=0.0,
            fee_bps=10.0,
            randomize_slippage=False,
        )
        result = sim.simulate_execution(
            side=OrderSide.BUY,
            quantity=Decimal("0.1"),
            market_price=50000.0,
        )
        assert result.fill_price == 50000.0
        assert result.slippage_cost == 0.0

    def test_zero_fees(self):
        """Zero fees means no fee charged."""
        sim = ExecutionSimulator(
            slippage_bps=5.0,
            fee_bps=0.0,
            randomize_slippage=False,
        )
        result = sim.simulate_execution(
            side=OrderSide.BUY,
            quantity=Decimal("0.1"),
            market_price=50000.0,
        )
        assert result.fees == 0.0


class TestRandomSlippage:
    """Tests for randomized slippage."""

    def test_randomize_slippage_varies(self):
        """Randomized slippage produces varying fills."""
        sim = ExecutionSimulator(
            slippage_bps=10.0,
            fee_bps=10.0,
            randomize_slippage=True,
        )

        fill_prices = set()
        for _ in range(20):
            result = sim.simulate_execution(
                side=OrderSide.BUY,
                quantity=Decimal("0.1"),
                market_price=50000.0,
            )
            fill_prices.add(round(result.fill_price, 2))

        # Should have multiple different fill prices
        assert len(fill_prices) > 1

    def test_randomize_slippage_bounded(self):
        """Random slippage stays within bounds."""
        sim = ExecutionSimulator(
            slippage_bps=10.0,
            fee_bps=0.0,
            randomize_slippage=True,
        )

        for _ in range(50):
            result = sim.simulate_execution(
                side=OrderSide.BUY,
                quantity=Decimal("0.1"),
                market_price=50000.0,
            )
            # Max slippage of 10 bps = 50000 * 1.001 = 50050
            assert 50000.0 <= result.fill_price <= 50050.0


class TestEstimateExecution:
    """Tests for estimate_execution method."""

    @pytest.fixture
    def simulator(self):
        return ExecutionSimulator(
            slippage_bps=5.0,
            fee_bps=10.0,
            randomize_slippage=False,
        )

    def test_estimate_returns_dict(self, simulator):
        """Estimate returns dict with expected keys."""
        estimate = simulator.estimate_execution(
            side=OrderSide.BUY,
            quantity=Decimal("0.1"),
            market_price=50000.0,
        )
        assert "estimated_fill_price" in estimate
        assert "estimated_slippage_cost" in estimate
        assert "estimated_fees" in estimate
        assert "estimated_total_cost" in estimate
        assert "notional_value" in estimate

    def test_estimate_matches_simulate(self, simulator):
        """Estimate matches simulate for fixed slippage."""
        estimate = simulator.estimate_execution(
            side=OrderSide.BUY,
            quantity=Decimal("0.1"),
            market_price=50000.0,
        )
        result = simulator.simulate_execution(
            side=OrderSide.BUY,
            quantity=Decimal("0.1"),
            market_price=50000.0,
        )
        assert estimate["estimated_fill_price"] == pytest.approx(
            result.fill_price, rel=0.01
        )


class TestGetConfig:
    """Tests for get_config method."""

    def test_get_config_returns_settings(self):
        """get_config returns all settings."""
        sim = ExecutionSimulator(
            slippage_bps=7.5,
            fee_bps=12.5,
            latency_min_ms=25,
            latency_max_ms=150,
            randomize_slippage=True,
        )
        config = sim.get_config()
        assert config["slippage_bps"] == 7.5
        assert config["fee_bps"] == 12.5
        assert config["latency_min_ms"] == 25
        assert config["latency_max_ms"] == 150
        assert config["randomize_slippage"] is True


class TestEdgeCases:
    """Tests for edge cases."""

    @pytest.fixture
    def simulator(self):
        return ExecutionSimulator(
            slippage_bps=5.0,
            fee_bps=10.0,
            randomize_slippage=False,
        )

    def test_very_small_quantity(self, simulator):
        """Handles very small quantities."""
        result = simulator.simulate_execution(
            side=OrderSide.BUY,
            quantity=Decimal("0.00001"),
            market_price=50000.0,
        )
        assert result.fill_qty == Decimal("0.00001")
        # Notional = 0.00001 * 50025 = 0.50025
        # Fee = 0.50025 * 0.001 = 0.00050025
        assert result.fees == pytest.approx(0.0005, rel=0.01)

    def test_large_quantity(self, simulator):
        """Handles large quantities."""
        result = simulator.simulate_execution(
            side=OrderSide.BUY,
            quantity=Decimal("100.0"),
            market_price=50000.0,
        )
        assert result.fill_qty == Decimal("100.0")
        # Notional = 100 * 50025 = 5,002,500
        # Fee = 5,002,500 * 0.001 = 5002.5
        assert result.fees == pytest.approx(5002.5, rel=0.01)

    def test_string_quantity_converted(self, simulator):
        """String quantity converted to Decimal."""
        result = simulator.simulate_execution(
            side=OrderSide.BUY,
            quantity="0.1",  # String
            market_price=50000.0,
        )
        assert result.fill_qty == Decimal("0.1")
