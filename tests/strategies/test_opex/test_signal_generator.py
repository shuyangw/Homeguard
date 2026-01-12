"""Tests for OpEx Signal Generator."""

import pytest
from datetime import date
from unittest.mock import Mock, patch

from src.strategies.opex.signal_generator import (
    OpExSignalGenerator,
    OpExSignal,
    SignalConfig,
    SignalDirection,
)
from src.strategies.opex.calendar import OpExPhase
from src.data.options.options_schema import (
    OptionsChain,
    OptionSnapshot,
    OptionRight,
)


class TestOpExSignalBasics:
    """Tests for OpExSignal dataclass."""

    def test_signal_is_actionable_long(self):
        """Test LONG signal is actionable."""
        signal = OpExSignal(
            symbol="SPY",
            date=date(2024, 1, 17),
            direction=SignalDirection.LONG,
            strength=0.7,
            phase=OpExPhase.PINNING,
            pin_strike=475.0,
            distance_to_pin=-0.01,
            net_gex=1_000_000,
            rationale="Test",
        )

        assert signal.is_actionable is True
        assert signal.direction_str == "LONG"

    def test_signal_is_actionable_short(self):
        """Test SHORT signal is actionable."""
        signal = OpExSignal(
            symbol="SPY",
            date=date(2024, 1, 17),
            direction=SignalDirection.SHORT,
            strength=0.7,
            phase=OpExPhase.PINNING,
            pin_strike=475.0,
            distance_to_pin=0.01,
            net_gex=1_000_000,
            rationale="Test",
        )

        assert signal.is_actionable is True
        assert signal.direction_str == "SHORT"

    def test_signal_not_actionable_neutral(self):
        """Test NEUTRAL signal is not actionable."""
        signal = OpExSignal(
            symbol="SPY",
            date=date(2024, 1, 17),
            direction=SignalDirection.NEUTRAL,
            strength=0.0,
            phase=OpExPhase.NORMAL,
            pin_strike=475.0,
            distance_to_pin=0.0,
            net_gex=0,
            rationale="Skip",
        )

        assert signal.is_actionable is False
        assert signal.direction_str == "NEUTRAL"


class TestSignalConfig:
    """Tests for SignalConfig defaults."""

    def test_default_config(self):
        """Test default configuration values."""
        config = SignalConfig()

        assert config.min_distance_pct == 0.005  # 0.5%
        assert config.max_distance_pct == 0.03   # 3%
        assert config.min_gex_threshold == 0.0
        assert OpExPhase.PINNING in config.active_phases
        assert OpExPhase.POST_OPEX in config.active_phases
        assert config.fade_on_pinning is True
        assert config.momentum_on_release is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = SignalConfig(
            min_distance_pct=0.01,
            max_distance_pct=0.05,
            active_phases=(OpExPhase.PINNING,),
        )

        assert config.min_distance_pct == 0.01
        assert config.max_distance_pct == 0.05
        assert config.active_phases == (OpExPhase.PINNING,)


class TestSignalGeneratorPinning:
    """Tests for PINNING phase signals."""

    def _create_chain(self, spot, pin_strike):
        """Helper to create options chain for testing."""
        contracts = [
            OptionSnapshot(
                symbol="SPY",
                date=date(2024, 1, 17),  # PINNING phase
                expiry=date(2024, 1, 19),  # OpEx day
                strike=pin_strike,
                option_type=OptionRight.CALL,
                open_interest=10000,
                delta=0.5,
                gamma=0.02,
                theta=-0.05,
                vega=0.1,
                rho=0.01,
                bid=2.0,
                ask=2.10,
                underlying_price=spot,
                volume=1000,
            ),
        ]

        return OptionsChain(
            symbol="SPY",
            date=date(2024, 1, 17),  # During PINNING phase
            expiry=date(2024, 1, 19),
            underlying_price=spot,
            contracts=contracts,
        )

    def test_pinning_price_above_pin_short(self):
        """Test PINNING: price above pin generates SHORT signal."""
        # Spot = 480, Pin = 475 -> 1.05% above pin
        chain = self._create_chain(spot=480.0, pin_strike=475.0)
        generator = OpExSignalGenerator()

        # Mock positive GEX (stabilizing environment)
        with patch.object(generator.gex_calc, 'calculate_total_gex', return_value=5_000_000):
            signal = generator.generate_signal(chain, date(2024, 1, 17))

        assert signal.direction == SignalDirection.SHORT
        assert signal.phase == OpExPhase.PINNING
        assert "SHORT" in signal.rationale
        assert signal.distance_to_pin > 0

    def test_pinning_price_below_pin_long(self):
        """Test PINNING: price below pin generates LONG signal."""
        # Spot = 470, Pin = 475 -> 1.05% below pin
        chain = self._create_chain(spot=470.0, pin_strike=475.0)
        generator = OpExSignalGenerator()

        # Mock positive GEX (stabilizing environment)
        with patch.object(generator.gex_calc, 'calculate_total_gex', return_value=5_000_000):
            signal = generator.generate_signal(chain, date(2024, 1, 17))

        assert signal.direction == SignalDirection.LONG
        assert signal.phase == OpExPhase.PINNING
        assert "LONG" in signal.rationale
        assert signal.distance_to_pin < 0

    def test_pinning_strength_increases_with_distance(self):
        """Test signal strength increases with distance from pin."""
        generator = OpExSignalGenerator()

        # Small distance: 0.63%
        chain_near = self._create_chain(spot=478.0, pin_strike=475.0)
        with patch.object(generator.gex_calc, 'calculate_total_gex', return_value=5_000_000):
            signal_near = generator.generate_signal(chain_near, date(2024, 1, 17))

        # Larger distance: 2.1%
        chain_far = self._create_chain(spot=485.0, pin_strike=475.0)
        with patch.object(generator.gex_calc, 'calculate_total_gex', return_value=5_000_000):
            signal_far = generator.generate_signal(chain_far, date(2024, 1, 17))

        assert signal_far.strength > signal_near.strength


class TestSignalGeneratorPostOpex:
    """Tests for POST_OPEX phase signals."""

    def _create_chain(self, spot, pin_strike):
        """Helper to create options chain for testing."""
        contracts = [
            OptionSnapshot(
                symbol="SPY",
                date=date(2024, 1, 22),  # POST_OPEX phase (Monday after)
                expiry=date(2024, 2, 16),  # Next monthly OpEx
                strike=pin_strike,
                option_type=OptionRight.CALL,
                open_interest=10000,
                delta=0.5,
                gamma=0.02,
                theta=-0.05,
                vega=0.1,
                rho=0.01,
                bid=2.0,
                ask=2.10,
                underlying_price=spot,
                volume=1000,
            ),
        ]

        return OptionsChain(
            symbol="SPY",
            date=date(2024, 1, 22),  # During POST_OPEX phase
            expiry=date(2024, 2, 16),
            underlying_price=spot,
            contracts=contracts,
        )

    def test_post_opex_price_above_pin_long(self):
        """Test POST_OPEX: price above pin generates LONG (momentum)."""
        chain = self._create_chain(spot=480.0, pin_strike=475.0)
        generator = OpExSignalGenerator()

        # Mock positive GEX
        with patch.object(generator.gex_calc, 'calculate_total_gex', return_value=5_000_000):
            signal = generator.generate_signal(chain, date(2024, 1, 22))

        assert signal.direction == SignalDirection.LONG
        assert signal.phase == OpExPhase.POST_OPEX
        assert "LONG" in signal.rationale
        assert "momentum" in signal.rationale.lower()

    def test_post_opex_price_below_pin_short(self):
        """Test POST_OPEX: price below pin generates SHORT (momentum)."""
        chain = self._create_chain(spot=470.0, pin_strike=475.0)
        generator = OpExSignalGenerator()

        # Mock positive GEX
        with patch.object(generator.gex_calc, 'calculate_total_gex', return_value=5_000_000):
            signal = generator.generate_signal(chain, date(2024, 1, 22))

        assert signal.direction == SignalDirection.SHORT
        assert signal.phase == OpExPhase.POST_OPEX
        assert "SHORT" in signal.rationale


class TestSignalGeneratorSkipConditions:
    """Tests for skip conditions."""

    def _create_chain(self, spot, pin_strike, signal_date):
        """Helper to create options chain."""
        contracts = [
            OptionSnapshot(
                symbol="SPY",
                date=signal_date,
                expiry=date(2024, 1, 19),
                strike=pin_strike,
                option_type=OptionRight.CALL,
                open_interest=10000,
                delta=0.5,
                gamma=0.02,
                theta=-0.05,
                vega=0.1,
                rho=0.01,
                bid=2.0,
                ask=2.10,
                underlying_price=spot,
                volume=1000,
            ),
        ]

        return OptionsChain(
            symbol="SPY",
            date=signal_date,
            expiry=date(2024, 1, 19),
            underlying_price=spot,
            contracts=contracts,
        )

    def test_skip_normal_phase(self):
        """Test signal skipped during NORMAL phase."""
        # Jan 5 is NORMAL phase (far from OpEx)
        chain = self._create_chain(spot=480.0, pin_strike=475.0, signal_date=date(2024, 1, 5))
        generator = OpExSignalGenerator()

        signal = generator.generate_signal(chain, date(2024, 1, 5))

        assert signal.direction == SignalDirection.NEUTRAL
        assert "not in active phases" in signal.rationale

    def test_skip_opex_day(self):
        """Test signal skipped on OPEX_DAY."""
        # Jan 19 is OPEX_DAY
        chain = self._create_chain(spot=480.0, pin_strike=475.0, signal_date=date(2024, 1, 19))
        config = SignalConfig(active_phases=(OpExPhase.PINNING, OpExPhase.POST_OPEX))
        generator = OpExSignalGenerator(config=config)

        signal = generator.generate_signal(chain, date(2024, 1, 19))

        assert signal.direction == SignalDirection.NEUTRAL
        assert "not in active phases" in signal.rationale

    def test_skip_too_close_to_pin(self):
        """Test signal skipped when too close to pin."""
        # Spot = 475.5, Pin = 475 -> 0.1% (below 0.5% min)
        chain = self._create_chain(spot=475.5, pin_strike=475.0, signal_date=date(2024, 1, 17))
        generator = OpExSignalGenerator()

        # Mock positive GEX so we get to the distance check
        with patch.object(generator.gex_calc, 'calculate_total_gex', return_value=5_000_000):
            signal = generator.generate_signal(chain, date(2024, 1, 17))

        assert signal.direction == SignalDirection.NEUTRAL
        assert "below minimum" in signal.rationale

    def test_skip_too_far_from_pin(self):
        """Test signal skipped when too far from pin."""
        # Spot = 495, Pin = 475 -> 4.2% (above 3% max)
        chain = self._create_chain(spot=495.0, pin_strike=475.0, signal_date=date(2024, 1, 17))
        generator = OpExSignalGenerator()

        # Mock positive GEX so we get to the distance check
        with patch.object(generator.gex_calc, 'calculate_total_gex', return_value=5_000_000):
            signal = generator.generate_signal(chain, date(2024, 1, 17))

        assert signal.direction == SignalDirection.NEUTRAL
        assert "above maximum" in signal.rationale

    def test_skip_negative_gex(self):
        """Test signal skipped with negative GEX (mocked)."""
        chain = self._create_chain(spot=480.0, pin_strike=475.0, signal_date=date(2024, 1, 17))
        generator = OpExSignalGenerator()

        # Mock GEX calculator to return negative GEX
        with patch.object(generator.gex_calc, 'calculate_total_gex', return_value=-1_000_000):
            signal = generator.generate_signal(chain, date(2024, 1, 17))

        assert signal.direction == SignalDirection.NEUTRAL
        assert "below threshold" in signal.rationale or "destabilizing" in signal.rationale


class TestSignalGeneratorBatch:
    """Tests for batch signal generation."""

    def _create_chain(self, symbol, spot, pin_strike, signal_date):
        """Helper to create options chain."""
        contracts = [
            OptionSnapshot(
                symbol=symbol,
                date=signal_date,
                expiry=date(2024, 1, 19),
                strike=pin_strike,
                option_type=OptionRight.CALL,
                open_interest=10000,
                delta=0.5,
                gamma=0.02,
                theta=-0.05,
                vega=0.1,
                rho=0.01,
                bid=2.0,
                ask=2.10,
                underlying_price=spot,
                volume=1000,
            ),
        ]

        return OptionsChain(
            symbol=symbol,
            date=signal_date,
            expiry=date(2024, 1, 19),
            underlying_price=spot,
            contracts=contracts,
        )

    def test_generate_signals_batch(self):
        """Test batch signal generation."""
        chains = [
            self._create_chain("SPY", 480.0, 475.0, date(2024, 1, 17)),
            self._create_chain("QQQ", 400.0, 395.0, date(2024, 1, 17)),
            self._create_chain("IWM", 200.0, 198.0, date(2024, 1, 17)),
        ]
        generator = OpExSignalGenerator()

        signals = generator.generate_signals_batch(chains, date(2024, 1, 17))

        assert len(signals) == 3
        assert signals[0].symbol == "SPY"
        assert signals[1].symbol == "QQQ"
        assert signals[2].symbol == "IWM"

    def test_get_actionable_signals(self):
        """Test filtering to actionable signals only."""
        chains = [
            # Actionable: 1% above pin
            self._create_chain("SPY", 480.0, 475.0, date(2024, 1, 17)),
            # Not actionable: too close to pin (0.1%)
            self._create_chain("QQQ", 400.5, 400.0, date(2024, 1, 17)),
        ]
        generator = OpExSignalGenerator()

        # Mock positive GEX
        with patch.object(generator.gex_calc, 'calculate_total_gex', return_value=5_000_000):
            signals = generator.get_actionable_signals(chains, date(2024, 1, 17))

        assert len(signals) == 1
        assert signals[0].symbol == "SPY"
        assert signals[0].is_actionable


class TestSignalGeneratorConfig:
    """Tests for custom configuration."""

    def _create_chain(self, spot, pin_strike):
        """Helper to create options chain."""
        contracts = [
            OptionSnapshot(
                symbol="SPY",
                date=date(2024, 1, 17),
                expiry=date(2024, 1, 19),
                strike=pin_strike,
                option_type=OptionRight.CALL,
                open_interest=10000,
                delta=0.5,
                gamma=0.02,
                theta=-0.05,
                vega=0.1,
                rho=0.01,
                bid=2.0,
                ask=2.10,
                underlying_price=spot,
                volume=1000,
            ),
        ]

        return OptionsChain(
            symbol="SPY",
            date=date(2024, 1, 17),
            expiry=date(2024, 1, 19),
            underlying_price=spot,
            contracts=contracts,
        )

    def test_custom_distance_thresholds(self):
        """Test custom distance thresholds."""
        # Default would skip at 0.3% but custom min is 0.1%
        chain = self._create_chain(spot=476.5, pin_strike=475.0)  # 0.32%
        config = SignalConfig(min_distance_pct=0.001, max_distance_pct=0.05)
        generator = OpExSignalGenerator(config=config)

        # Mock positive GEX
        with patch.object(generator.gex_calc, 'calculate_total_gex', return_value=5_000_000):
            signal = generator.generate_signal(chain, date(2024, 1, 17))

        assert signal.is_actionable

    def test_pinning_only_mode(self):
        """Test configuration for PINNING phase only."""
        config = SignalConfig(
            active_phases=(OpExPhase.PINNING,),
            momentum_on_release=False,
        )
        generator = OpExSignalGenerator(config=config)

        # PINNING phase - should work
        chain_pinning = self._create_chain(spot=480.0, pin_strike=475.0)
        # Mock positive GEX
        with patch.object(generator.gex_calc, 'calculate_total_gex', return_value=5_000_000):
            signal_pinning = generator.generate_signal(chain_pinning, date(2024, 1, 17))
        assert signal_pinning.is_actionable

    def test_fade_disabled(self):
        """Test with fade_on_pinning disabled."""
        config = SignalConfig(fade_on_pinning=False)
        generator = OpExSignalGenerator(config=config)

        chain = self._create_chain(spot=480.0, pin_strike=475.0)
        # Mock positive GEX
        with patch.object(generator.gex_calc, 'calculate_total_gex', return_value=5_000_000):
            signal = generator.generate_signal(chain, date(2024, 1, 17))

        # Should be neutral since fading is disabled
        assert signal.direction == SignalDirection.NEUTRAL
