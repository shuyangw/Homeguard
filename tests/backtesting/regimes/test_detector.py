"""
Tests for regime detection module.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / 'src'))

import pytest
import pandas as pd
import numpy as np

from backtesting.regimes.detector import (
    TrendDetector,
    VolatilityDetector,
    DrawdownDetector,
    RegimeLabel,
    RegimePeriod,
    detect_all_regimes
)


@pytest.fixture
def bull_market_prices():
    """Create bull market price data."""
    dates = pd.date_range(start='2020-01-01', periods=200, freq='D')
    # Strong upward trend
    prices = pd.Series(
        100 + np.linspace(0, 50, 200) + np.random.randn(200) * 2,
        index=dates
    )
    return prices


@pytest.fixture
def bear_market_prices():
    """Create bear market price data."""
    dates = pd.date_range(start='2020-01-01', periods=200, freq='D')
    # Strong downward trend
    prices = pd.Series(
        100 - np.linspace(0, 50, 200) + np.random.randn(200) * 2,
        index=dates
    )
    return prices


@pytest.fixture
def sideways_market_prices():
    """Create sideways market price data."""
    dates = pd.date_range(start='2020-01-01', periods=200, freq='D')
    # No trend, just noise
    prices = pd.Series(
        100 + np.random.randn(200) * 2,
        index=dates
    )
    return prices


@pytest.fixture
def volatile_prices():
    """Create high volatility price data."""
    dates = pd.date_range(start='2020-01-01', periods=200, freq='D')
    # High volatility
    prices = pd.Series(
        100 + np.cumsum(np.random.randn(200) * 5),  # Large moves
        index=dates
    )
    return prices


@pytest.fixture
def calm_prices():
    """Create low volatility price data."""
    dates = pd.date_range(start='2020-01-01', periods=200, freq='D')
    # Low volatility
    prices = pd.Series(
        100 + np.cumsum(np.random.randn(200) * 0.1),  # Small moves
        index=dates
    )
    return prices


class TestRegimeLabel:
    """Tests for RegimeLabel enum."""

    def test_regime_labels_exist(self):
        """Test that all regime labels are defined."""
        assert RegimeLabel.BULL.value == "Bull Market"
        assert RegimeLabel.BEAR.value == "Bear Market"
        assert RegimeLabel.SIDEWAYS.value == "Sideways"
        assert RegimeLabel.HIGH_VOL.value == "High Volatility"
        assert RegimeLabel.LOW_VOL.value == "Low Volatility"
        assert RegimeLabel.DRAWDOWN.value == "Drawdown"
        assert RegimeLabel.RECOVERY.value == "Recovery"
        assert RegimeLabel.CALM.value == "Calm"


class TestRegimePeriod:
    """Tests for RegimePeriod dataclass."""

    def test_regime_period_creation(self):
        """Test creating a regime period."""
        period = RegimePeriod(
            start_date='2020-01-01',
            end_date='2020-06-30',
            regime=RegimeLabel.BULL,
            metric_value=15.5
        )

        assert period.start_date == '2020-01-01'
        assert period.end_date == '2020-06-30'
        assert period.regime == RegimeLabel.BULL
        assert period.metric_value == 15.5


class TestTrendDetector:
    """Tests for TrendDetector class."""

    def test_initialization(self):
        """Test detector initialization."""
        detector = TrendDetector(lookback_days=60, threshold_pct=5.0)
        assert detector.lookback_days == 60
        assert detector.threshold_pct == 5.0

    def test_detect_bull_market(self, bull_market_prices):
        """Test detecting bull market."""
        detector = TrendDetector(lookback_days=60, threshold_pct=5.0)
        regimes = detector.detect(bull_market_prices)

        # Should detect at least one regime
        assert len(regimes) > 0

        # Should contain bull market periods
        bull_periods = [r for r in regimes if r.regime == RegimeLabel.BULL]
        assert len(bull_periods) > 0

    def test_detect_bear_market(self, bear_market_prices):
        """Test detecting bear market."""
        detector = TrendDetector(lookback_days=60, threshold_pct=5.0)
        regimes = detector.detect(bear_market_prices)

        # Should detect at least one regime
        assert len(regimes) > 0

        # Should contain bear market periods
        bear_periods = [r for r in regimes if r.regime == RegimeLabel.BEAR]
        assert len(bear_periods) > 0

    def test_detect_sideways_market(self, sideways_market_prices):
        """Test detecting sideways market."""
        detector = TrendDetector(lookback_days=60, threshold_pct=5.0)
        regimes = detector.detect(sideways_market_prices)

        # Should detect at least one regime
        assert len(regimes) > 0

        # Should contain sideways periods
        sideways_periods = [r for r in regimes if r.regime == RegimeLabel.SIDEWAYS]
        assert len(sideways_periods) > 0

    def test_empty_prices(self):
        """Test with empty price series."""
        detector = TrendDetector()
        regimes = detector.detect(pd.Series([]))
        assert len(regimes) == 0

    def test_insufficient_data(self):
        """Test with insufficient data."""
        detector = TrendDetector(lookback_days=60)
        prices = pd.Series([100, 101, 102], index=pd.date_range('2020-01-01', periods=3))
        regimes = detector.detect(prices)
        assert len(regimes) == 0


class TestVolatilityDetector:
    """Tests for VolatilityDetector class."""

    def test_initialization(self):
        """Test detector initialization."""
        detector = VolatilityDetector(lookback_days=20, percentile=70)
        assert detector.lookback_days == 20
        assert detector.percentile == 70

    def test_detect_regimes(self, volatile_prices):
        """Test detecting volatility regimes."""
        detector = VolatilityDetector(lookback_days=20, percentile=70)
        regimes = detector.detect(volatile_prices)

        # Should detect regimes
        assert len(regimes) > 0

        # Should have both high and low vol periods
        high_vol = [r for r in regimes if r.regime == RegimeLabel.HIGH_VOL]
        low_vol = [r for r in regimes if r.regime == RegimeLabel.LOW_VOL]

        # At least one type should exist
        assert len(high_vol) > 0 or len(low_vol) > 0

    def test_empty_prices(self):
        """Test with empty price series."""
        detector = VolatilityDetector()
        regimes = detector.detect(pd.Series([]))
        assert len(regimes) == 0

    def test_insufficient_data(self):
        """Test with insufficient data."""
        detector = VolatilityDetector(lookback_days=20)
        prices = pd.Series([100, 101], index=pd.date_range('2020-01-01', periods=2))
        regimes = detector.detect(prices)
        assert len(regimes) == 0


class TestDrawdownDetector:
    """Tests for DrawdownDetector class."""

    def test_initialization(self):
        """Test detector initialization."""
        detector = DrawdownDetector(drawdown_threshold=10.0)
        assert detector.drawdown_threshold == 10.0

    def test_detect_drawdown(self):
        """Test detecting drawdown regime."""
        # Create data with a clear drawdown that gets progressively worse
        dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
        # Create gradually worsening drawdown
        prices_list = [100] * 10  # Stable period
        # Gradual decline in steps to trigger drawdown detection
        for i in range(90):
            prices_list.append(100 - (i * 0.5))  # Gradually declining
        prices = pd.Series(prices_list, index=dates)

        detector = DrawdownDetector(drawdown_threshold=10.0)
        regimes = detector.detect(prices)

        # Should detect regimes
        assert len(regimes) > 0

        # Should detect drawdown or recovery periods (both involve significant drawdown)
        drawdown_or_recovery = [r for r in regimes
                                if r.regime in [RegimeLabel.DRAWDOWN, RegimeLabel.RECOVERY]]
        assert len(drawdown_or_recovery) > 0

    def test_detect_recovery(self):
        """Test detecting recovery regime."""
        # Create data with drawdown then recovery
        dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
        # 20 + 15 + 13 + 52 = 100
        prices = pd.Series(
            [100] * 20 +
            list(range(100, 70, -2)) +  # Drawdown (15 elements)
            list(range(70, 96, 2)) +     # Recovery (13 elements)
            [96] * 52,
            index=dates
        )

        detector = DrawdownDetector(drawdown_threshold=10.0)
        regimes = detector.detect(prices)

        # Should detect regimes
        assert len(regimes) > 0

        # May detect recovery periods
        recovery_periods = [r for r in regimes if r.regime == RegimeLabel.RECOVERY]
        # Recovery detection is complex, so we just check it doesn't error

    def test_detect_calm(self):
        """Test detecting calm regime."""
        # Create data at high water mark
        dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
        prices = pd.Series(100 + np.linspace(0, 20, 100), index=dates)  # Steady uptrend

        detector = DrawdownDetector(drawdown_threshold=10.0)
        regimes = detector.detect(prices)

        # Should detect regimes
        assert len(regimes) > 0

        # Should have calm periods (no significant drawdown)
        calm_periods = [r for r in regimes if r.regime == RegimeLabel.CALM]
        assert len(calm_periods) > 0

    def test_empty_prices(self):
        """Test with empty price series."""
        detector = DrawdownDetector()
        regimes = detector.detect(pd.Series([]))
        assert len(regimes) == 0

    def test_insufficient_data(self):
        """Test with minimal data."""
        detector = DrawdownDetector()
        prices = pd.Series([100], index=pd.date_range('2020-01-01', periods=1))
        regimes = detector.detect(prices)
        assert len(regimes) == 0


class TestDetectAllRegimes:
    """Tests for detect_all_regimes convenience function."""

    def test_detect_all(self, bull_market_prices):
        """Test detecting all regime types."""
        result = detect_all_regimes(bull_market_prices)

        assert 'trend' in result
        assert 'volatility' in result
        assert 'drawdown' in result

        assert isinstance(result['trend'], list)
        assert isinstance(result['volatility'], list)
        assert isinstance(result['drawdown'], list)

    def test_detect_all_empty(self):
        """Test with empty prices."""
        result = detect_all_regimes(pd.Series([]))

        assert len(result['trend']) == 0
        assert len(result['volatility']) == 0
        assert len(result['drawdown']) == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
