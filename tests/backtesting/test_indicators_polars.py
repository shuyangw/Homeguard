"""
Unit tests for Polars indicators ensuring numeric equivalence with pandas.

These tests verify that Polars implementations produce identical results
to pandas implementations within floating-point tolerance (1e-6).
"""

import pytest
import numpy as np
import pandas as pd
import polars as pl

from src.backtesting.utils.indicators import Indicators
from src.backtesting.utils.indicators_polars import IndicatorsPolars


# Test data size
N_SAMPLES = 500
TOLERANCE = 1e-6


@pytest.fixture
def sample_prices():
    """Generate sample price data for testing."""
    np.random.seed(42)
    # Generate realistic price data with random walk
    returns = np.random.normal(0.0005, 0.02, N_SAMPLES)
    prices = 100 * np.exp(np.cumsum(returns))
    return prices


@pytest.fixture
def sample_ohlcv():
    """Generate sample OHLCV data for testing."""
    np.random.seed(42)
    returns = np.random.normal(0.0005, 0.02, N_SAMPLES)
    close = 100 * np.exp(np.cumsum(returns))
    # Generate OHLC from close
    high = close * (1 + np.abs(np.random.normal(0, 0.01, N_SAMPLES)))
    low = close * (1 - np.abs(np.random.normal(0, 0.01, N_SAMPLES)))
    open_price = (high + low) / 2
    volume = np.random.uniform(1000000, 10000000, N_SAMPLES)
    return open_price, high, low, close, volume


class TestIndicatorsParity:
    """Test numeric equivalence between pandas and polars implementations."""

    def test_sma_parity(self, sample_prices):
        """Test SMA produces identical results."""
        pd_prices = pd.Series(sample_prices)
        pl_prices = pl.Series(sample_prices)

        window = 20
        pd_result = Indicators.sma(pd_prices, window)
        pl_result = Indicators.sma(pl_prices, window)

        # Convert to numpy for comparison
        pd_values = pd_result.dropna().values
        pl_values = pl_result.drop_nulls().to_numpy()

        np.testing.assert_allclose(pd_values, pl_values, rtol=TOLERANCE)

    def test_ema_parity(self, sample_prices):
        """Test EMA produces identical results."""
        pd_prices = pd.Series(sample_prices)
        pl_prices = pl.Series(sample_prices)

        window = 20
        pd_result = Indicators.ema(pd_prices, window)
        pl_result = Indicators.ema(pl_prices, window)

        pd_values = pd_result.dropna().values
        pl_values = pl_result.drop_nulls().to_numpy()

        np.testing.assert_allclose(pd_values, pl_values, rtol=TOLERANCE)

    def test_rsi_parity(self, sample_prices):
        """Test RSI produces identical results."""
        pd_prices = pd.Series(sample_prices)
        pl_prices = pl.Series(sample_prices)

        window = 14
        pd_result = Indicators.rsi(pd_prices, window)
        pl_result = Indicators.rsi(pl_prices, window)

        # RSI can have NaN/inf values, filter them out
        pd_values = pd_result.replace([np.inf, -np.inf], np.nan).dropna().values
        pl_values = pl_result.to_pandas().replace([np.inf, -np.inf], np.nan).dropna().values

        # Polars and pandas have slightly different null handling for diff() + rolling
        # The values are equivalent but may have different null positions at the start
        # Compare the common portion that both produce
        min_len = min(len(pd_values), len(pl_values))
        # Skip first element if there's a length mismatch (edge null handling)
        if len(pd_values) > len(pl_values):
            pd_values = pd_values[1:min_len + 1]
            pl_values = pl_values[:min_len]
        elif len(pl_values) > len(pd_values):
            pl_values = pl_values[1:min_len + 1]
            pd_values = pd_values[:min_len]

        np.testing.assert_allclose(pd_values, pl_values, rtol=TOLERANCE)

    def test_bollinger_bands_parity(self, sample_prices):
        """Test Bollinger Bands produce identical results."""
        pd_prices = pd.Series(sample_prices)
        pl_prices = pl.Series(sample_prices)

        window = 20
        num_std = 2.0

        pd_upper, pd_middle, pd_lower = Indicators.bollinger_bands(pd_prices, window, num_std)
        pl_upper, pl_middle, pl_lower = Indicators.bollinger_bands(pl_prices, window, num_std)

        # Compare upper band
        np.testing.assert_allclose(
            pd_upper.dropna().values,
            pl_upper.drop_nulls().to_numpy(),
            rtol=TOLERANCE
        )
        # Compare middle band
        np.testing.assert_allclose(
            pd_middle.dropna().values,
            pl_middle.drop_nulls().to_numpy(),
            rtol=TOLERANCE
        )
        # Compare lower band
        np.testing.assert_allclose(
            pd_lower.dropna().values,
            pl_lower.drop_nulls().to_numpy(),
            rtol=TOLERANCE
        )

    def test_macd_parity(self, sample_prices):
        """Test MACD produces identical results."""
        pd_prices = pd.Series(sample_prices)
        pl_prices = pl.Series(sample_prices)

        pd_macd, pd_signal, pd_hist = Indicators.macd(pd_prices)
        pl_macd, pl_signal, pl_hist = Indicators.macd(pl_prices)

        # MACD line
        np.testing.assert_allclose(
            pd_macd.dropna().values,
            pl_macd.drop_nulls().to_numpy(),
            rtol=TOLERANCE
        )
        # Signal line
        np.testing.assert_allclose(
            pd_signal.dropna().values,
            pl_signal.drop_nulls().to_numpy(),
            rtol=TOLERANCE
        )
        # Histogram
        np.testing.assert_allclose(
            pd_hist.dropna().values,
            pl_hist.drop_nulls().to_numpy(),
            rtol=TOLERANCE
        )

    def test_atr_parity(self, sample_ohlcv):
        """Test ATR produces identical results."""
        _, high, low, close, _ = sample_ohlcv

        pd_high = pd.Series(high)
        pd_low = pd.Series(low)
        pd_close = pd.Series(close)

        pl_high = pl.Series(high)
        pl_low = pl.Series(low)
        pl_close = pl.Series(close)

        window = 14
        pd_result = Indicators.atr(pd_high, pd_low, pd_close, window)
        pl_result = Indicators.atr(pl_high, pl_low, pl_close, window)

        pd_values = pd_result.dropna().values
        pl_values = pl_result.drop_nulls().to_numpy()

        np.testing.assert_allclose(pd_values, pl_values, rtol=TOLERANCE)

    def test_stochastic_parity(self, sample_ohlcv):
        """Test Stochastic Oscillator produces identical results."""
        _, high, low, close, _ = sample_ohlcv

        pd_high = pd.Series(high)
        pd_low = pd.Series(low)
        pd_close = pd.Series(close)

        pl_high = pl.Series(high)
        pl_low = pl.Series(low)
        pl_close = pl.Series(close)

        k_window = 14
        d_window = 3
        pd_k, pd_d = Indicators.stochastic(pd_high, pd_low, pd_close, k_window, d_window)
        pl_k, pl_d = Indicators.stochastic(pl_high, pl_low, pl_close, k_window, d_window)

        # %K
        np.testing.assert_allclose(
            pd_k.dropna().values,
            pl_k.drop_nulls().to_numpy(),
            rtol=TOLERANCE
        )
        # %D
        np.testing.assert_allclose(
            pd_d.dropna().values,
            pl_d.drop_nulls().to_numpy(),
            rtol=TOLERANCE
        )

    def test_donchian_channel_parity(self, sample_ohlcv):
        """Test Donchian Channel produces identical results."""
        _, high, low, _, _ = sample_ohlcv

        pd_high = pd.Series(high)
        pd_low = pd.Series(low)

        pl_high = pl.Series(high)
        pl_low = pl.Series(low)

        window = 20
        pd_upper, pd_lower = Indicators.donchian_channel(pd_high, pd_low, window)
        pl_upper, pl_lower = Indicators.donchian_channel(pl_high, pl_low, window)

        # Upper channel
        np.testing.assert_allclose(
            pd_upper.dropna().values,
            pl_upper.drop_nulls().to_numpy(),
            rtol=TOLERANCE
        )
        # Lower channel
        np.testing.assert_allclose(
            pd_lower.dropna().values,
            pl_lower.drop_nulls().to_numpy(),
            rtol=TOLERANCE
        )

    def test_returns_parity(self, sample_prices):
        """Test returns calculation produces identical results."""
        pd_prices = pd.Series(sample_prices)
        pl_prices = pl.Series(sample_prices)

        for periods in [1, 5, 20]:
            pd_result = Indicators.returns(pd_prices, periods)
            pl_result = Indicators.returns(pl_prices, periods)

            pd_values = pd_result.dropna().values
            pl_values = pl_result.drop_nulls().to_numpy()

            np.testing.assert_allclose(pd_values, pl_values, rtol=TOLERANCE)

    def test_rolling_volatility_parity(self, sample_prices):
        """Test rolling volatility produces identical results."""
        pd_prices = pd.Series(sample_prices)
        pl_prices = pl.Series(sample_prices)

        window = 20
        # Test with annualization
        pd_result = Indicators.rolling_volatility(pd_prices, window, annualize=True)
        pl_result = Indicators.rolling_volatility(pl_prices, window, annualize=True)

        pd_values = pd_result.dropna().values
        pl_values = pl_result.drop_nulls().to_numpy()

        np.testing.assert_allclose(pd_values, pl_values, rtol=TOLERANCE)

        # Test without annualization
        pd_result = Indicators.rolling_volatility(pd_prices, window, annualize=False)
        pl_result = Indicators.rolling_volatility(pl_prices, window, annualize=False)

        pd_values = pd_result.dropna().values
        pl_values = pl_result.drop_nulls().to_numpy()

        np.testing.assert_allclose(pd_values, pl_values, rtol=TOLERANCE)

    def test_vwap_parity(self, sample_ohlcv):
        """Test VWAP produces identical results."""
        _, _, _, close, volume = sample_ohlcv

        pd_prices = pd.Series(close)
        pd_volume = pd.Series(volume)

        pl_prices = pl.Series(close)
        pl_volume = pl.Series(volume)

        pd_result = Indicators.vwap(pd_prices, pd_volume)
        pl_result = Indicators.vwap(pl_prices, pl_volume)

        pd_values = pd_result.dropna().values
        pl_values = pl_result.drop_nulls().to_numpy()

        np.testing.assert_allclose(pd_values, pl_values, rtol=TOLERANCE)

    def test_z_score_parity(self, sample_prices):
        """Test Z-score produces identical results."""
        pd_prices = pd.Series(sample_prices)
        pl_prices = pl.Series(sample_prices)

        window = 20
        pd_result = Indicators.z_score(pd_prices, window)
        pl_result = Indicators.z_score(pl_prices, window)

        pd_values = pd_result.dropna().values
        pl_values = pl_result.drop_nulls().to_numpy()

        np.testing.assert_allclose(pd_values, pl_values, rtol=TOLERANCE)


class TestAutoDetection:
    """Test that the hybrid interface auto-detects input type correctly."""

    def test_pandas_input_stays_pandas(self, sample_prices):
        """Test that pandas input returns pandas output."""
        pd_prices = pd.Series(sample_prices)
        result = Indicators.sma(pd_prices, 20)
        assert isinstance(result, pd.Series)

    def test_polars_input_stays_polars(self, sample_prices):
        """Test that polars input returns polars output."""
        pl_prices = pl.Series(sample_prices)
        result = Indicators.sma(pl_prices, 20)
        assert isinstance(result, pl.Series)

    def test_multi_output_functions_preserve_type(self, sample_prices):
        """Test that functions returning tuples preserve input type."""
        pd_prices = pd.Series(sample_prices)
        pl_prices = pl.Series(sample_prices)

        # Test Bollinger Bands
        pd_result = Indicators.bollinger_bands(pd_prices, 20)
        pl_result = Indicators.bollinger_bands(pl_prices, 20)

        assert all(isinstance(r, pd.Series) for r in pd_result)
        assert all(isinstance(r, pl.Series) for r in pl_result)

        # Test MACD
        pd_result = Indicators.macd(pd_prices)
        pl_result = Indicators.macd(pl_prices)

        assert all(isinstance(r, pd.Series) for r in pd_result)
        assert all(isinstance(r, pl.Series) for r in pl_result)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_small_window(self, sample_prices):
        """Test with minimum window size."""
        pd_prices = pd.Series(sample_prices)
        pl_prices = pl.Series(sample_prices)

        window = 2
        pd_result = Indicators.sma(pd_prices, window)
        pl_result = Indicators.sma(pl_prices, window)

        pd_values = pd_result.dropna().values
        pl_values = pl_result.drop_nulls().to_numpy()

        np.testing.assert_allclose(pd_values, pl_values, rtol=TOLERANCE)

    def test_large_window(self, sample_prices):
        """Test with window approaching data size."""
        pd_prices = pd.Series(sample_prices)
        pl_prices = pl.Series(sample_prices)

        window = N_SAMPLES - 10
        pd_result = Indicators.sma(pd_prices, window)
        pl_result = Indicators.sma(pl_prices, window)

        pd_values = pd_result.dropna().values
        pl_values = pl_result.drop_nulls().to_numpy()

        np.testing.assert_allclose(pd_values, pl_values, rtol=TOLERANCE)

    def test_constant_prices(self):
        """Test with constant price series."""
        constant_prices = np.ones(100) * 100.0
        pd_prices = pd.Series(constant_prices)
        pl_prices = pl.Series(constant_prices)

        # SMA of constant should equal constant
        pd_result = Indicators.sma(pd_prices, 20)
        pl_result = Indicators.sma(pl_prices, 20)

        pd_values = pd_result.dropna().values
        pl_values = pl_result.drop_nulls().to_numpy()

        np.testing.assert_allclose(pd_values, pl_values, rtol=TOLERANCE)
        np.testing.assert_allclose(pd_values, 100.0, rtol=TOLERANCE)
