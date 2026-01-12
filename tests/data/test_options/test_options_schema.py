"""Tests for options data schema."""

import pytest
from datetime import date, datetime

from src.data.options.options_schema import (
    OptionSnapshot,
    OptionsChain,
    StrikeGEX,
    DailyGEXSummary,
    OptionRight,
)


class TestOptionSnapshot:
    """Tests for OptionSnapshot dataclass."""

    def test_basic_creation(self):
        """Test creating a basic option snapshot."""
        snapshot = OptionSnapshot(
            symbol="SPY",
            date=date(2024, 1, 15),
            expiry=date(2024, 1, 19),
            strike=475.0,
            option_type=OptionRight.CALL,
            open_interest=5000,
        )

        assert snapshot.symbol == "SPY"
        assert snapshot.strike == 475.0
        assert snapshot.is_call is True
        assert snapshot.is_put is False
        assert snapshot.days_to_expiry == 4

    def test_option_type_string_conversion(self):
        """Test that string option types are converted to enum."""
        snapshot = OptionSnapshot(
            symbol="SPY",
            date=date(2024, 1, 15),
            expiry=date(2024, 1, 19),
            strike=475.0,
            option_type="C",  # String input
            open_interest=5000,
        )

        assert snapshot.option_type == OptionRight.CALL
        assert snapshot.is_call is True

    def test_datetime_to_date_conversion(self):
        """Test that datetime inputs are converted to date."""
        snapshot = OptionSnapshot(
            symbol="SPY",
            date=datetime(2024, 1, 15, 12, 0, 0),  # datetime input
            expiry=datetime(2024, 1, 19, 16, 0, 0),  # datetime input
            strike=475.0,
            option_type=OptionRight.PUT,
            open_interest=3000,
        )

        assert isinstance(snapshot.date, date)
        assert isinstance(snapshot.expiry, date)
        assert snapshot.date == date(2024, 1, 15)

    def test_mid_price_calculation(self):
        """Test mid price between bid and ask."""
        snapshot = OptionSnapshot(
            symbol="SPY",
            date=date(2024, 1, 15),
            expiry=date(2024, 1, 19),
            strike=475.0,
            option_type=OptionRight.CALL,
            open_interest=5000,
            bid=2.50,
            ask=2.60,
        )

        assert snapshot.mid_price == 2.55

    def test_mid_price_no_bid(self):
        """Test mid price when no bid."""
        snapshot = OptionSnapshot(
            symbol="SPY",
            date=date(2024, 1, 15),
            expiry=date(2024, 1, 19),
            strike=475.0,
            option_type=OptionRight.CALL,
            open_interest=5000,
            bid=0.0,
            ask=2.60,
        )

        assert snapshot.mid_price == 2.60

    def test_moneyness(self):
        """Test moneyness calculation."""
        # ITM call
        snapshot = OptionSnapshot(
            symbol="SPY",
            date=date(2024, 1, 15),
            expiry=date(2024, 1, 19),
            strike=470.0,
            option_type=OptionRight.CALL,
            open_interest=5000,
            underlying_price=475.0,
        )

        assert snapshot.moneyness == pytest.approx(475.0 / 470.0, rel=0.001)

    def test_moneyness_zero_strike(self):
        """Test moneyness with zero strike returns 0."""
        snapshot = OptionSnapshot(
            symbol="SPY",
            date=date(2024, 1, 15),
            expiry=date(2024, 1, 19),
            strike=0.0,
            option_type=OptionRight.CALL,
            open_interest=5000,
            underlying_price=475.0,
        )

        assert snapshot.moneyness == 0.0


class TestStrikeGEX:
    """Tests for StrikeGEX dataclass."""

    def test_basic_creation(self):
        """Test creating a StrikeGEX."""
        gex = StrikeGEX(
            strike=475.0,
            call_gex=1000000.0,
            put_gex=-500000.0,
            net_gex=500000.0,
            call_oi=5000,
            put_oi=3000,
        )

        assert gex.strike == 475.0
        assert gex.net_gex == 500000.0
        assert gex.total_oi == 8000  # Computed from call_oi + put_oi

    def test_total_oi_explicit(self):
        """Test that explicit total_oi is preserved."""
        gex = StrikeGEX(
            strike=475.0,
            call_gex=1000000.0,
            put_gex=-500000.0,
            net_gex=500000.0,
            call_oi=5000,
            put_oi=3000,
            total_oi=10000,  # Explicit value
        )

        assert gex.total_oi == 10000  # Preserved


class TestOptionsChain:
    """Tests for OptionsChain dataclass."""

    @pytest.fixture
    def sample_contracts(self):
        """Create sample option contracts."""
        return [
            OptionSnapshot(
                symbol="SPY",
                date=date(2024, 1, 15),
                expiry=date(2024, 1, 19),
                strike=470.0,
                option_type=OptionRight.CALL,
                open_interest=5000,
                volume=1000,
            ),
            OptionSnapshot(
                symbol="SPY",
                date=date(2024, 1, 15),
                expiry=date(2024, 1, 19),
                strike=470.0,
                option_type=OptionRight.PUT,
                open_interest=3000,
                volume=500,
            ),
            OptionSnapshot(
                symbol="SPY",
                date=date(2024, 1, 15),
                expiry=date(2024, 1, 19),
                strike=475.0,
                option_type=OptionRight.CALL,
                open_interest=8000,
                volume=2000,
            ),
            OptionSnapshot(
                symbol="SPY",
                date=date(2024, 1, 15),
                expiry=date(2024, 1, 19),
                strike=475.0,
                option_type=OptionRight.PUT,
                open_interest=4000,
                volume=800,
            ),
        ]

    def test_aggregates_computed(self, sample_contracts):
        """Test that aggregates are computed from contracts."""
        chain = OptionsChain(
            symbol="SPY",
            date=date(2024, 1, 15),
            expiry=date(2024, 1, 19),
            underlying_price=473.0,
            contracts=sample_contracts,
        )

        assert chain.total_call_oi == 13000  # 5000 + 8000
        assert chain.total_put_oi == 7000   # 3000 + 4000
        assert chain.total_call_volume == 3000  # 1000 + 2000
        assert chain.total_put_volume == 1300   # 500 + 800
        assert chain.put_call_ratio == pytest.approx(7000 / 13000, rel=0.001)

    def test_days_to_expiry(self, sample_contracts):
        """Test days to expiry calculation."""
        chain = OptionsChain(
            symbol="SPY",
            date=date(2024, 1, 15),
            expiry=date(2024, 1, 19),
            underlying_price=473.0,
            contracts=sample_contracts,
        )

        assert chain.days_to_expiry == 4

    def test_get_calls_puts(self, sample_contracts):
        """Test filtering calls and puts."""
        chain = OptionsChain(
            symbol="SPY",
            date=date(2024, 1, 15),
            expiry=date(2024, 1, 19),
            underlying_price=473.0,
            contracts=sample_contracts,
        )

        calls = chain.get_calls()
        puts = chain.get_puts()

        assert len(calls) == 2
        assert len(puts) == 2
        assert all(c.is_call for c in calls)
        assert all(p.is_put for p in puts)

    def test_get_strikes(self, sample_contracts):
        """Test getting unique strikes."""
        chain = OptionsChain(
            symbol="SPY",
            date=date(2024, 1, 15),
            expiry=date(2024, 1, 19),
            underlying_price=473.0,
            contracts=sample_contracts,
        )

        strikes = chain.get_strikes()

        assert strikes == [470.0, 475.0]

    def test_get_atm_strike(self, sample_contracts):
        """Test getting ATM strike."""
        chain = OptionsChain(
            symbol="SPY",
            date=date(2024, 1, 15),
            expiry=date(2024, 1, 19),
            underlying_price=473.0,
            contracts=sample_contracts,
        )

        atm = chain.get_atm_strike()

        assert atm == 475.0  # Closer to 473 than 470

    def test_filter_near_money(self, sample_contracts):
        """Test filtering to near-money contracts."""
        chain = OptionsChain(
            symbol="SPY",
            date=date(2024, 1, 15),
            expiry=date(2024, 1, 19),
            underlying_price=473.0,
            contracts=sample_contracts,
        )

        # 10% range: 425.7 to 520.3 - all contracts within
        near = chain.filter_near_money(pct_range=0.10)
        assert len(near) == 4

        # 0.5% range: 470.6 to 475.4 - only 475 contracts
        near = chain.filter_near_money(pct_range=0.005)
        assert len(near) == 2
        assert all(c.strike == 475.0 for c in near)

    def test_empty_chain(self):
        """Test empty chain behavior."""
        chain = OptionsChain(
            symbol="SPY",
            date=date(2024, 1, 15),
            expiry=date(2024, 1, 19),
            underlying_price=473.0,
            contracts=[],
        )

        assert chain.total_oi == 0
        assert chain.get_strikes() == []
        assert chain.get_atm_strike() == 0.0


class TestDailyGEXSummary:
    """Tests for DailyGEXSummary dataclass."""

    def test_is_positive_gex(self):
        """Test positive GEX detection."""
        summary = DailyGEXSummary(
            symbol="SPY",
            date=date(2024, 1, 15),
            expiry=date(2024, 1, 19),
            underlying_price=475.0,
            total_gex=1000000.0,
            call_gex=1500000.0,
            put_gex=-500000.0,
            pin_strike=475.0,
            pin_strike_gex=500000.0,
            distance_to_pin=0.0,
            total_oi=50000,
            put_call_ratio=0.5,
        )

        assert summary.is_positive_gex is True

    def test_is_near_pin(self):
        """Test near-pin detection."""
        # Exactly at pin
        summary = DailyGEXSummary(
            symbol="SPY",
            date=date(2024, 1, 15),
            expiry=date(2024, 1, 19),
            underlying_price=475.0,
            total_gex=1000000.0,
            call_gex=1500000.0,
            put_gex=-500000.0,
            pin_strike=475.0,
            pin_strike_gex=500000.0,
            distance_to_pin=0.0,
            total_oi=50000,
            put_call_ratio=0.5,
        )
        assert summary.is_near_pin is True

        # 1.5% away (within 2%)
        summary.distance_to_pin = 0.015
        assert summary.is_near_pin is True

        # 3% away (outside 2%)
        summary.distance_to_pin = 0.03
        assert summary.is_near_pin is False
