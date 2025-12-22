"""
Tests for technical indicators module.
"""

import numpy as np
import pandas as pd
import pytest

from src.screening.indicators import (
    IndicatorResult,
    TechnicalIndicators,
    compute_indicators_for_screening,
)


class TestTechnicalIndicators:
    """Tests for TechnicalIndicators class."""

    def test_init_with_valid_data(self, sample_historical_bars):
        """Test initialization with valid OHLCV data."""
        indicators = TechnicalIndicators(sample_historical_bars)
        assert len(indicators.df) == len(sample_historical_bars)

    def test_init_normalizes_column_names(self):
        """Test that column names are normalized to lowercase."""
        df = pd.DataFrame(
            {
                "Open": [100, 101],
                "High": [102, 103],
                "Low": [99, 100],
                "Close": [101, 102],
                "Volume": [1000, 2000],
            }
        )
        indicators = TechnicalIndicators(df)
        assert "close" in indicators.df.columns
        assert "Close" not in indicators.df.columns

    def test_init_missing_columns_raises(self):
        """Test that missing required columns raises error."""
        df = pd.DataFrame({"close": [100, 101], "volume": [1000, 2000]})
        with pytest.raises(ValueError) as exc:
            TechnicalIndicators(df)
        assert "Missing required columns" in str(exc.value)


class TestRSI:
    """Tests for RSI calculation."""

    def test_rsi_range(self, sample_historical_bars):
        """Test that RSI is within 0-100 range."""
        indicators = TechnicalIndicators(sample_historical_bars)
        rsi = indicators.rsi(14)
        assert rsi is not None
        assert 0 <= rsi <= 100

    def test_rsi_insufficient_data(self):
        """Test RSI returns None with insufficient data."""
        df = pd.DataFrame(
            {
                "open": [100, 101, 102],
                "high": [101, 102, 103],
                "low": [99, 100, 101],
                "close": [100, 101, 102],
            }
        )
        indicators = TechnicalIndicators(df)
        rsi = indicators.rsi(14)
        assert rsi is None

    def test_rsi_oversold(self, oversold_historical_bars):
        """Test RSI in oversold territory for declining prices."""
        indicators = TechnicalIndicators(oversold_historical_bars)
        rsi = indicators.rsi(14)
        # Declining prices should produce lower RSI
        assert rsi is not None
        assert rsi < 50  # Should be relatively low

    def test_rsi_series(self, sample_historical_bars):
        """Test RSI series calculation."""
        indicators = TechnicalIndicators(sample_historical_bars)
        rsi_series = indicators.rsi_series(14)
        assert len(rsi_series) == len(sample_historical_bars)


class TestMovingAverages:
    """Tests for moving average calculations."""

    def test_sma_calculation(self, sample_historical_bars):
        """Test SMA calculation."""
        indicators = TechnicalIndicators(sample_historical_bars)
        sma_20 = indicators.sma(20)
        assert sma_20 is not None

    def test_sma_insufficient_data(self):
        """Test SMA returns None with insufficient data."""
        df = pd.DataFrame(
            {
                "open": [100, 101],
                "high": [101, 102],
                "low": [99, 100],
                "close": [100, 101],
            }
        )
        indicators = TechnicalIndicators(df)
        sma = indicators.sma(20)
        assert sma is None

    def test_ema_calculation(self, sample_historical_bars):
        """Test EMA calculation."""
        indicators = TechnicalIndicators(sample_historical_bars)
        ema_12 = indicators.ema(12)
        ema_26 = indicators.ema(26)
        assert ema_12 is not None
        assert ema_26 is not None

    def test_sma_crossover_golden(self, sample_historical_bars):
        """Test SMA crossover detection."""
        indicators = TechnicalIndicators(sample_historical_bars)
        # Just verify method works without error
        result = indicators.is_sma_crossover(20, 50, "above")
        assert isinstance(result, bool)

    def test_sma_crossover_death(self, sample_historical_bars):
        """Test death cross detection."""
        indicators = TechnicalIndicators(sample_historical_bars)
        result = indicators.is_sma_crossover(20, 50, "below")
        assert isinstance(result, bool)


class TestMACD:
    """Tests for MACD calculation."""

    def test_macd_calculation(self, sample_historical_bars):
        """Test MACD line, signal, and histogram calculation."""
        indicators = TechnicalIndicators(sample_historical_bars)
        macd_line, signal_line, histogram = indicators.macd()
        assert macd_line is not None
        assert signal_line is not None
        assert histogram is not None

    def test_macd_insufficient_data(self):
        """Test MACD returns None with insufficient data."""
        df = pd.DataFrame(
            {
                "open": list(range(20)),
                "high": [x + 1 for x in range(20)],
                "low": [x - 1 for x in range(20)],
                "close": list(range(20)),
            }
        )
        indicators = TechnicalIndicators(df)
        macd_line, signal_line, histogram = indicators.macd()
        assert macd_line is None

    def test_macd_signal_type(self, sample_historical_bars):
        """Test MACD signal type detection."""
        indicators = TechnicalIndicators(sample_historical_bars)
        signal_type = indicators.macd_signal_type()
        assert signal_type in [None, "bullish", "bearish"]


class TestBollingerBands:
    """Tests for Bollinger Bands calculation."""

    def test_bollinger_bands_calculation(self, sample_historical_bars):
        """Test Bollinger Bands calculation."""
        indicators = TechnicalIndicators(sample_historical_bars)
        upper, middle, lower = indicators.bollinger_bands()
        assert upper is not None
        assert middle is not None
        assert lower is not None
        assert upper > middle > lower

    def test_bollinger_position(self, sample_historical_bars):
        """Test Bollinger position detection."""
        indicators = TechnicalIndicators(sample_historical_bars)
        position = indicators.bollinger_position()
        assert position in [None, "above_upper", "below_lower", "middle"]


class TestATR:
    """Tests for ATR calculation."""

    def test_atr_calculation(self, sample_historical_bars):
        """Test ATR calculation."""
        indicators = TechnicalIndicators(sample_historical_bars)
        atr = indicators.atr(14)
        assert atr is not None
        assert atr > 0

    def test_atr_percent(self, sample_historical_bars):
        """Test ATR percentage calculation."""
        indicators = TechnicalIndicators(sample_historical_bars)
        atr_pct = indicators.atr_percent(14)
        assert atr_pct is not None
        assert atr_pct > 0


class TestVolatility:
    """Tests for volatility calculations."""

    def test_historical_volatility(self, sample_historical_bars):
        """Test historical volatility calculation."""
        indicators = TechnicalIndicators(sample_historical_bars)
        vol = indicators.historical_volatility(20)
        assert vol is not None
        assert vol > 0


class TestVolumeMetrics:
    """Tests for volume metric calculations."""

    def test_avg_volume(self, sample_historical_bars):
        """Test average volume calculation."""
        indicators = TechnicalIndicators(sample_historical_bars)
        avg_vol = indicators.avg_volume(20)
        assert avg_vol is not None
        assert avg_vol > 0

    def test_relative_volume(self, sample_historical_bars):
        """Test relative volume calculation."""
        indicators = TechnicalIndicators(sample_historical_bars)
        rel_vol = indicators.relative_volume(20)
        # Relative volume could be any positive number
        assert rel_vol is not None


class TestComputeAll:
    """Tests for compute_all method."""

    def test_compute_all_returns_result(self, sample_historical_bars):
        """Test that compute_all returns IndicatorResult."""
        indicators = TechnicalIndicators(sample_historical_bars)
        result = indicators.compute_all()
        assert isinstance(result, IndicatorResult)

    def test_compute_all_populates_fields(self, sample_historical_bars):
        """Test that compute_all populates indicator fields."""
        indicators = TechnicalIndicators(sample_historical_bars)
        result = indicators.compute_all()
        assert result.rsi_14 is not None
        assert result.sma_20 is not None
        assert result.atr_14 is not None


class TestComputeIndicatorsForScreening:
    """Tests for compute_indicators_for_screening function."""

    def test_valid_data(self, sample_historical_bars):
        """Test with valid data."""
        result = compute_indicators_for_screening(sample_historical_bars)
        assert isinstance(result, IndicatorResult)

    def test_empty_data(self):
        """Test with empty DataFrame."""
        df = pd.DataFrame()
        result = compute_indicators_for_screening(df)
        assert isinstance(result, IndicatorResult)
        # All fields should be None
        assert result.rsi_14 is None
