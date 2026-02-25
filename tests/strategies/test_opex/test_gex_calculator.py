"""Tests for GEX Calculator."""

import pytest
from datetime import date
from unittest.mock import Mock

from src.strategies.opex.gex_calculator import GEXCalculator, GEXProfile
from src.data.options.options_schema import (
    OptionsChain,
    OptionSnapshot,
    StrikeGEX,
    OptionRight,
)


class TestGEXCalculatorContractGEX:
    """Tests for single contract GEX calculation."""

    def test_calculate_contract_gex_short_dealer(self):
        """Test GEX calculation with short dealer position (default).

        Standard SpotGamma convention:
        - Call GEX = gamma * OI * 100 * spot * (+1)
        - Put GEX = gamma * OI * 100 * spot * (-1)
        """
        calc = GEXCalculator(dealer_position="short")

        # Call: gamma=0.05, OI=1000, spot=$100
        # GEX = 0.05 * 1000 * 100 * 100 * (+1) = +500,000
        result = calc.calculate_contract_gex(
            gamma=0.05,
            open_interest=1000,
            spot_price=100.0,
            is_call=True,
        )
        assert result == 500_000.0

        # Put: gamma=0.05, OI=1000, spot=$100
        # GEX = 0.05 * 1000 * 100 * 100 * (-1) = -500,000
        result = calc.calculate_contract_gex(
            gamma=0.05,
            open_interest=1000,
            spot_price=100.0,
            is_call=False,
        )
        assert result == -500_000.0

    def test_calculate_contract_gex_long_dealer(self):
        """Test GEX calculation with long dealer position (signs flipped)."""
        calc = GEXCalculator(dealer_position="long")

        # Long dealer flips signs
        # Call: gamma=0.05, OI=1000, spot=$100
        # GEX = 0.05 * 1000 * 100 * 100 * (+1) * (-1) = -500,000
        result = calc.calculate_contract_gex(
            gamma=0.05,
            open_interest=1000,
            spot_price=100.0,
            is_call=True,
        )
        assert result == -500_000.0

        # Put: gamma=0.05, OI=1000, spot=$100
        # GEX = 0.05 * 1000 * 100 * 100 * (-1) * (-1) = +500,000
        result = calc.calculate_contract_gex(
            gamma=0.05,
            open_interest=1000,
            spot_price=100.0,
            is_call=False,
        )
        assert result == 500_000.0

    def test_calculate_contract_gex_zero_oi(self):
        """Test GEX with zero open interest."""
        calc = GEXCalculator()

        result = calc.calculate_contract_gex(
            gamma=0.05,
            open_interest=0,
            spot_price=100.0,
        )

        assert result == 0.0

    def test_calculate_contract_gex_zero_gamma(self):
        """Test GEX with zero gamma."""
        calc = GEXCalculator()

        result = calc.calculate_contract_gex(
            gamma=0.0,
            open_interest=1000,
            spot_price=100.0,
        )

        assert result == 0.0


class TestGEXCalculatorStrikeGEX:
    """Tests for strike-level GEX aggregation."""

    def _create_chain(self, contracts_data):
        """Helper to create options chain from contract data."""
        contracts = []
        for data in contracts_data:
            contracts.append(OptionSnapshot(
                symbol="SPY",
                date=date(2024, 1, 15),
                expiry=date(2024, 1, 19),
                strike=data["strike"],
                option_type=data["type"],
                open_interest=data["oi"],
                delta=data.get("delta", 0.5),
                gamma=data["gamma"],
                theta=data.get("theta", -0.05),
                vega=data.get("vega", 0.1),
                rho=data.get("rho", 0.01),
                bid=data.get("bid", 1.0),
                ask=data.get("ask", 1.10),
                underlying_price=475.0,
                volume=data.get("volume", 100),
            ))

        return OptionsChain(
            symbol="SPY",
            date=date(2024, 1, 15),
            expiry=date(2024, 1, 19),
            underlying_price=475.0,
            contracts=contracts,
        )

    def test_calculate_strike_gex_single_call(self):
        """Test strike GEX with single call.

        Standard convention: Calls have positive GEX contribution.
        """
        chain = self._create_chain([
            {"strike": 475.0, "type": OptionRight.CALL, "oi": 1000, "gamma": 0.02},
        ])
        calc = GEXCalculator()

        result = calc.calculate_strike_gex(chain)

        assert 475.0 in result
        assert result[475.0].call_gex > 0  # Calls = positive contribution
        assert result[475.0].put_gex == 0.0
        assert result[475.0].net_gex == result[475.0].call_gex

    def test_calculate_strike_gex_single_put(self):
        """Test strike GEX with single put.

        Standard convention: Puts have negative GEX contribution.
        """
        chain = self._create_chain([
            {"strike": 475.0, "type": OptionRight.PUT, "oi": 1000, "gamma": 0.02},
        ])
        calc = GEXCalculator()

        result = calc.calculate_strike_gex(chain)

        assert 475.0 in result
        assert result[475.0].call_gex == 0.0
        assert result[475.0].put_gex < 0  # Puts = negative contribution
        assert result[475.0].net_gex == result[475.0].put_gex

    def test_calculate_strike_gex_call_and_put_same_strike(self):
        """Test strike GEX with call and put at same strike.

        Net GEX = Call GEX (positive) + Put GEX (negative)
        With more call OI, net should be positive.
        """
        chain = self._create_chain([
            {"strike": 475.0, "type": OptionRight.CALL, "oi": 1000, "gamma": 0.02},
            {"strike": 475.0, "type": OptionRight.PUT, "oi": 800, "gamma": 0.02},
        ])
        calc = GEXCalculator()

        result = calc.calculate_strike_gex(chain)

        assert 475.0 in result
        assert result[475.0].call_gex > 0  # Calls positive
        assert result[475.0].put_gex < 0   # Puts negative
        assert result[475.0].net_gex == result[475.0].call_gex + result[475.0].put_gex
        assert result[475.0].net_gex > 0  # Net positive (more calls)
        assert result[475.0].call_oi == 1000
        assert result[475.0].put_oi == 800

    def test_calculate_strike_gex_multiple_strikes(self):
        """Test strike GEX with multiple strikes."""
        chain = self._create_chain([
            {"strike": 470.0, "type": OptionRight.CALL, "oi": 500, "gamma": 0.01},
            {"strike": 475.0, "type": OptionRight.CALL, "oi": 2000, "gamma": 0.02},
            {"strike": 480.0, "type": OptionRight.CALL, "oi": 500, "gamma": 0.01},
        ])
        calc = GEXCalculator()

        result = calc.calculate_strike_gex(chain)

        assert len(result) == 3
        assert 470.0 in result
        assert 475.0 in result
        assert 480.0 in result

        # ATM strike (475) should have highest absolute GEX
        assert abs(result[475.0].net_gex) > abs(result[470.0].net_gex)
        assert abs(result[475.0].net_gex) > abs(result[480.0].net_gex)

    def test_calculate_strike_gex_empty_chain(self):
        """Test strike GEX with empty chain."""
        chain = OptionsChain(
            symbol="SPY",
            date=date(2024, 1, 15),
            expiry=date(2024, 1, 19),
            underlying_price=475.0,
            contracts=[],
        )
        calc = GEXCalculator()

        result = calc.calculate_strike_gex(chain)

        assert result == {}


class TestGEXCalculatorTotalGEX:
    """Tests for total GEX calculation."""

    def _create_chain(self, contracts_data, spot=475.0):
        """Helper to create options chain."""
        contracts = []
        for data in contracts_data:
            contracts.append(OptionSnapshot(
                symbol="SPY",
                date=date(2024, 1, 15),
                expiry=date(2024, 1, 19),
                strike=data["strike"],
                option_type=data["type"],
                open_interest=data["oi"],
                delta=0.5,
                gamma=data["gamma"],
                theta=-0.05,
                vega=0.1,
                rho=0.01,
                bid=1.0,
                ask=1.10,
                underlying_price=spot,
                volume=100,
            ))

        return OptionsChain(
            symbol="SPY",
            date=date(2024, 1, 15),
            expiry=date(2024, 1, 19),
            underlying_price=spot,
            contracts=contracts,
        )

    def test_calculate_total_gex(self):
        """Test total GEX is sum of all strike GEX."""
        chain = self._create_chain([
            {"strike": 470.0, "type": OptionRight.CALL, "oi": 1000, "gamma": 0.01},
            {"strike": 475.0, "type": OptionRight.CALL, "oi": 2000, "gamma": 0.02},
            {"strike": 480.0, "type": OptionRight.PUT, "oi": 1500, "gamma": 0.015},
        ])
        calc = GEXCalculator()

        total = calc.calculate_total_gex(chain)
        strike_gex = calc.calculate_strike_gex(chain)
        expected = sum(sg.net_gex for sg in strike_gex.values())

        assert total == expected

    def test_calculate_total_gex_put_heavy_negative(self):
        """Test total GEX is negative when puts dominate.

        Standard convention: Calls +, Puts -
        Put-heavy chain should have negative net GEX.
        """
        chain = self._create_chain([
            {"strike": 475.0, "type": OptionRight.PUT, "oi": 2000, "gamma": 0.02},
        ])
        calc = GEXCalculator(dealer_position="short")

        total = calc.calculate_total_gex(chain)

        assert total < 0  # Puts = negative GEX


class TestGEXCalculatorPinStrike:
    """Tests for pin strike identification."""

    def _create_chain(self, contracts_data, spot=475.0):
        """Helper to create options chain."""
        contracts = []
        for data in contracts_data:
            contracts.append(OptionSnapshot(
                symbol="SPY",
                date=date(2024, 1, 15),
                expiry=date(2024, 1, 19),
                strike=data["strike"],
                option_type=data["type"],
                open_interest=data["oi"],
                delta=0.5,
                gamma=data["gamma"],
                theta=-0.05,
                vega=0.1,
                rho=0.01,
                bid=1.0,
                ask=1.10,
                underlying_price=spot,
                volume=100,
            ))

        return OptionsChain(
            symbol="SPY",
            date=date(2024, 1, 15),
            expiry=date(2024, 1, 19),
            underlying_price=spot,
            contracts=contracts,
        )

    def test_find_pin_strike_max_gex_method(self):
        """Test pin strike using max_gex method."""
        # Create chain where 475 has highest GEX
        chain = self._create_chain([
            {"strike": 470.0, "type": OptionRight.CALL, "oi": 500, "gamma": 0.01},
            {"strike": 475.0, "type": OptionRight.CALL, "oi": 5000, "gamma": 0.02},
            {"strike": 480.0, "type": OptionRight.CALL, "oi": 500, "gamma": 0.01},
        ])
        calc = GEXCalculator()

        pin = calc.find_pin_strike(chain, method="max_gex")

        assert pin == 475.0

    def test_find_pin_strike_max_oi_method(self):
        """Test pin strike using max_oi method."""
        # Create chain where 470 has highest OI
        chain = self._create_chain([
            {"strike": 470.0, "type": OptionRight.CALL, "oi": 10000, "gamma": 0.005},
            {"strike": 475.0, "type": OptionRight.CALL, "oi": 1000, "gamma": 0.02},
            {"strike": 480.0, "type": OptionRight.CALL, "oi": 500, "gamma": 0.01},
        ])
        calc = GEXCalculator()

        pin = calc.find_pin_strike(chain, method="max_oi")

        assert pin == 470.0

    def test_find_pin_strike_weighted_method(self):
        """Test pin strike using weighted method (GEX + proximity)."""
        # 475 is closest to spot (475), so should win with weighted method
        chain = self._create_chain([
            {"strike": 470.0, "type": OptionRight.CALL, "oi": 2000, "gamma": 0.015},
            {"strike": 475.0, "type": OptionRight.CALL, "oi": 2000, "gamma": 0.015},
            {"strike": 480.0, "type": OptionRight.CALL, "oi": 2000, "gamma": 0.015},
        ], spot=475.0)
        calc = GEXCalculator()

        pin = calc.find_pin_strike(chain, method="weighted")

        assert pin == 475.0

    def test_find_pin_strike_empty_chain(self):
        """Test pin strike with empty chain."""
        chain = OptionsChain(
            symbol="SPY",
            date=date(2024, 1, 15),
            expiry=date(2024, 1, 19),
            underlying_price=475.0,
            contracts=[],
        )
        calc = GEXCalculator()

        pin = calc.find_pin_strike(chain)

        assert pin is None

    def test_find_pin_strike_invalid_method(self):
        """Test pin strike with invalid method raises error."""
        chain = self._create_chain([
            {"strike": 475.0, "type": OptionRight.CALL, "oi": 1000, "gamma": 0.02},
        ])
        calc = GEXCalculator()

        with pytest.raises(ValueError, match="Unknown method"):
            calc.find_pin_strike(chain, method="invalid")

    def test_find_pin_strike_filters_to_near_money(self):
        """Test that only near-money strikes are considered."""
        # spot=475, near_strike_pct=0.10 -> only 427.5 to 522.5 considered
        chain = self._create_chain([
            {"strike": 400.0, "type": OptionRight.CALL, "oi": 50000, "gamma": 0.03},  # Far OTM
            {"strike": 475.0, "type": OptionRight.CALL, "oi": 1000, "gamma": 0.02},   # ATM
            {"strike": 550.0, "type": OptionRight.CALL, "oi": 50000, "gamma": 0.03},  # Far OTM
        ], spot=475.0)
        calc = GEXCalculator(near_strike_pct=0.10)

        pin = calc.find_pin_strike(chain, method="max_gex")

        # Should return ATM strike, not the far OTM with huge OI
        assert pin == 475.0


class TestGEXCalculatorProfile:
    """Tests for GEX profile calculation."""

    def _create_chain(self, contracts_data, spot=475.0):
        """Helper to create options chain."""
        contracts = []
        for data in contracts_data:
            contracts.append(OptionSnapshot(
                symbol="SPY",
                date=date(2024, 1, 15),
                expiry=date(2024, 1, 19),
                strike=data["strike"],
                option_type=data["type"],
                open_interest=data["oi"],
                delta=0.5,
                gamma=data["gamma"],
                theta=-0.05,
                vega=0.1,
                rho=0.01,
                bid=1.0,
                ask=1.10,
                underlying_price=spot,
                volume=100,
            ))

        return OptionsChain(
            symbol="SPY",
            date=date(2024, 1, 15),
            expiry=date(2024, 1, 19),
            underlying_price=spot,
            contracts=contracts,
        )

    def test_calculate_gex_profile_returns_correct_length(self):
        """Test GEX profile returns correct number of points."""
        chain = self._create_chain([
            {"strike": 475.0, "type": OptionRight.CALL, "oi": 1000, "gamma": 0.02},
        ])
        calc = GEXCalculator()

        profile = calc.calculate_gex_profile(chain, n_points=21)

        assert len(profile.prices) == 21
        assert len(profile.net_gex) == 21
        assert len(profile.call_gex) == 21
        assert len(profile.put_gex) == 21

    def test_calculate_gex_profile_price_range(self):
        """Test GEX profile spans correct price range."""
        chain = self._create_chain([
            {"strike": 475.0, "type": OptionRight.CALL, "oi": 1000, "gamma": 0.02},
        ], spot=475.0)
        calc = GEXCalculator()

        # 95% to 105% of spot
        profile = calc.calculate_gex_profile(chain, price_range=(0.95, 1.05))

        assert min(profile.prices) == pytest.approx(475.0 * 0.95, rel=0.01)
        assert max(profile.prices) == pytest.approx(475.0 * 1.05, rel=0.01)

    def test_calculate_gex_profile_to_dataframe(self):
        """Test GEX profile can convert to DataFrame."""
        chain = self._create_chain([
            {"strike": 475.0, "type": OptionRight.CALL, "oi": 1000, "gamma": 0.02},
        ])
        calc = GEXCalculator()

        profile = calc.calculate_gex_profile(chain, n_points=11)
        df = profile.to_dataframe()

        assert len(df) == 11
        assert "price" in df.columns
        assert "net_gex" in df.columns
        assert "call_gex" in df.columns
        assert "put_gex" in df.columns


class TestGEXCalculatorDailySummary:
    """Tests for daily GEX summary."""

    def _create_chain(self, contracts_data, spot=475.0):
        """Helper to create options chain."""
        contracts = []
        for data in contracts_data:
            contracts.append(OptionSnapshot(
                symbol="SPY",
                date=date(2024, 1, 15),
                expiry=date(2024, 1, 19),
                strike=data["strike"],
                option_type=data["type"],
                open_interest=data["oi"],
                delta=0.5,
                gamma=data["gamma"],
                theta=-0.05,
                vega=0.1,
                rho=0.01,
                bid=1.0,
                ask=1.10,
                underlying_price=spot,
                volume=100,
            ))

        return OptionsChain(
            symbol="SPY",
            date=date(2024, 1, 15),
            expiry=date(2024, 1, 19),
            underlying_price=spot,
            contracts=contracts,
        )

    def test_calculate_daily_summary(self):
        """Test daily summary contains all fields."""
        chain = self._create_chain([
            {"strike": 470.0, "type": OptionRight.CALL, "oi": 1000, "gamma": 0.01},
            {"strike": 475.0, "type": OptionRight.CALL, "oi": 2000, "gamma": 0.02},
            {"strike": 475.0, "type": OptionRight.PUT, "oi": 1500, "gamma": 0.018},
        ])
        calc = GEXCalculator()

        summary = calc.calculate_daily_summary(chain)

        assert summary.symbol == "SPY"
        assert summary.date == date(2024, 1, 15)
        assert summary.expiry == date(2024, 1, 19)
        assert summary.underlying_price == 475.0
        assert summary.total_gex != 0
        assert summary.call_gex != 0
        assert summary.put_gex != 0
        assert summary.pin_strike == 475.0  # Highest GEX strike
        assert summary.total_oi == 4500  # 1000 + 2000 + 1500

    def test_calculate_daily_summary_with_historical_percentile(self):
        """Test daily summary calculates GEX percentile."""
        chain = self._create_chain([
            {"strike": 475.0, "type": OptionRight.CALL, "oi": 1000, "gamma": 0.02},
        ])
        calc = GEXCalculator()

        # Historical GEX values for percentile calculation
        historical = [100_000, 200_000, 300_000, 400_000, 500_000]

        summary = calc.calculate_daily_summary(chain, historical_gex=historical)

        # Current GEX should be compared against historical
        assert 0 <= summary.gex_percentile <= 100


class TestGEXCalculatorRegime:
    """Tests for GEX regime classification."""

    def test_get_gex_regime_positive(self):
        """Test positive GEX regime."""
        calc = GEXCalculator()

        regime = calc.get_gex_regime(1_000_000)

        assert regime == "positive"

    def test_get_gex_regime_negative(self):
        """Test negative GEX regime."""
        calc = GEXCalculator()

        regime = calc.get_gex_regime(-1_000_000)

        assert regime == "negative"

    def test_get_gex_regime_neutral(self):
        """Test neutral GEX regime (zero)."""
        calc = GEXCalculator()

        regime = calc.get_gex_regime(0)

        assert regime == "neutral"


class TestGEXCalculatorPinProbability:
    """Tests for pin probability analysis."""

    def _create_chain(self, contracts_data, spot=475.0):
        """Helper to create options chain."""
        contracts = []
        for data in contracts_data:
            contracts.append(OptionSnapshot(
                symbol="SPY",
                date=date(2024, 1, 15),
                expiry=date(2024, 1, 19),
                strike=data["strike"],
                option_type=data["type"],
                open_interest=data["oi"],
                delta=0.5,
                gamma=data["gamma"],
                theta=-0.05,
                vega=0.1,
                rho=0.01,
                bid=1.0,
                ask=1.10,
                underlying_price=spot,
                volume=100,
            ))

        return OptionsChain(
            symbol="SPY",
            date=date(2024, 1, 15),
            expiry=date(2024, 1, 19),
            underlying_price=spot,
            contracts=contracts,
        )

    def test_analyze_pin_probability_sums_to_one(self):
        """Test pin probabilities sum to 1.0."""
        chain = self._create_chain([
            {"strike": 470.0, "type": OptionRight.CALL, "oi": 1000, "gamma": 0.01},
            {"strike": 475.0, "type": OptionRight.CALL, "oi": 2000, "gamma": 0.02},
            {"strike": 480.0, "type": OptionRight.CALL, "oi": 1000, "gamma": 0.01},
        ])
        calc = GEXCalculator()

        probs = calc.analyze_pin_probability(chain)

        # Only near-money strikes have non-zero probability
        total = sum(probs.values())
        assert total == pytest.approx(1.0, rel=0.01)

    def test_analyze_pin_probability_atm_highest(self):
        """Test ATM strike has highest probability (equal GEX)."""
        chain = self._create_chain([
            {"strike": 470.0, "type": OptionRight.CALL, "oi": 1000, "gamma": 0.02},
            {"strike": 475.0, "type": OptionRight.CALL, "oi": 1000, "gamma": 0.02},
            {"strike": 480.0, "type": OptionRight.CALL, "oi": 1000, "gamma": 0.02},
        ], spot=475.0)
        calc = GEXCalculator()

        probs = calc.analyze_pin_probability(chain)

        # ATM (475) should have highest probability due to proximity
        assert probs[475.0] > probs[470.0]
        assert probs[475.0] > probs[480.0]


class TestGEXCalculatorToDataFrame:
    """Tests for DataFrame conversion."""

    def test_to_dataframe_empty(self):
        """Test to_dataframe with empty input."""
        calc = GEXCalculator()

        df = calc.to_dataframe({})

        assert len(df) == 0

    def test_to_dataframe_with_data(self):
        """Test to_dataframe with strike GEX data."""
        calc = GEXCalculator()

        strike_gex = {
            470.0: StrikeGEX(
                strike=470.0,
                call_gex=-100_000,
                put_gex=-50_000,
                net_gex=-150_000,
                call_oi=1000,
                put_oi=500,
            ),
            475.0: StrikeGEX(
                strike=475.0,
                call_gex=-200_000,
                put_gex=-100_000,
                net_gex=-300_000,
                call_oi=2000,
                put_oi=1000,
            ),
        }

        df = calc.to_dataframe(strike_gex)

        assert len(df) == 2
        assert list(df.columns) == ["strike", "call_gex", "put_gex", "net_gex", "call_oi", "put_oi", "total_oi"]
        assert df.iloc[0]["strike"] == 470.0  # Sorted
        assert df.iloc[1]["strike"] == 475.0
