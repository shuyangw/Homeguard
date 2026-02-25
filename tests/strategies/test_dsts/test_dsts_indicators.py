"""
Unit tests for DSTS indicators.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from src.strategies.advanced.dsts_indicators import DSTSIndicators, DSTSSignal


class TestDSTSIndicatorsBasic:
    """Tests for basic indicator calculations."""

    def create_sample_data(self, days: int = 100) -> pd.Series:
        """Create sample close price series for testing."""
        dates = pd.date_range(start='2020-01-01', periods=days, freq='D')
        np.random.seed(42)
        base_price = 10000
        returns = np.random.randn(days) * 0.03  # 3% daily volatility
        prices = base_price * np.exp(np.cumsum(returns))
        return pd.Series(prices, index=dates)

    def test_calculate_ema(self):
        """Test EMA calculation matches pandas ewm."""
        close = self.create_sample_data()
        period = 65

        ema = DSTSIndicators.calculate_ema(close, period)
        expected_ema = close.ewm(span=period, adjust=False).mean()

        pd.testing.assert_series_equal(ema, expected_ema)

    def test_calculate_ema_shorter_period(self):
        """Test EMA with shorter period."""
        close = self.create_sample_data()
        ema_10 = DSTSIndicators.calculate_ema(close, 10)
        ema_65 = DSTSIndicators.calculate_ema(close, 65)

        # Shorter EMA should be more responsive (closer to price)
        # Check variance of difference from price
        diff_10 = (close - ema_10).std()
        diff_65 = (close - ema_65).std()
        assert diff_10 < diff_65

    def test_calculate_std(self):
        """Test StdDev calculation matches pandas rolling."""
        close = self.create_sample_data()
        period = 65

        std = DSTSIndicators.calculate_std(close, period)
        expected_std = close.rolling(window=period).std()

        pd.testing.assert_series_equal(std, expected_std)

    def test_calculate_std_nan_values(self):
        """Test StdDev has NaN for insufficient data."""
        close = self.create_sample_data(days=100)
        period = 65

        std = DSTSIndicators.calculate_std(close, period)

        # First (period - 1) values should be NaN
        assert std.iloc[:period - 1].isna().all()
        # Rest should be valid
        assert not std.iloc[period - 1:].isna().any()

    def test_calculate_zscore_formula(self):
        """Test Z-score calculation matches manual formula."""
        close = self.create_sample_data()
        period = 65

        zscore = DSTSIndicators.calculate_zscore(close, period)

        # Manual calculation
        ema = close.ewm(span=period, adjust=False).mean()
        std = close.rolling(window=period).std()
        expected = (close - ema) / std

        pd.testing.assert_series_equal(zscore, expected)

    def test_calculate_zscore_range(self):
        """Test Z-score is typically in reasonable range."""
        close = self.create_sample_data(days=500)
        zscore = DSTSIndicators.calculate_zscore(close, 65)

        # For normally distributed returns, Z-score should mostly be within [-3, 3]
        valid_zscore = zscore.dropna()
        within_range = ((valid_zscore >= -4) & (valid_zscore <= 4)).mean()
        assert within_range > 0.95, f"Only {within_range:.1%} of Z-scores in [-4, 4]"

    def test_calculate_all_indicators(self):
        """Test calculate_all_indicators returns correct columns."""
        close = self.create_sample_data()
        result = DSTSIndicators.calculate_all_indicators(close, 65)

        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ['close', 'ema', 'std', 'zscore']
        assert len(result) == len(close)

    def test_calculate_all_indicators_values(self):
        """Test calculate_all_indicators returns consistent values."""
        close = self.create_sample_data()
        period = 65

        result = DSTSIndicators.calculate_all_indicators(close, period)

        # Values should match individual calculations
        pd.testing.assert_series_equal(
            result['ema'],
            DSTSIndicators.calculate_ema(close, period),
            check_names=False
        )
        pd.testing.assert_series_equal(
            result['std'],
            DSTSIndicators.calculate_std(close, period),
            check_names=False
        )
        pd.testing.assert_series_equal(
            result['zscore'],
            DSTSIndicators.calculate_zscore(close, period),
            check_names=False
        )


class TestDSTSSignalGeneration:
    """Tests for signal generation logic."""

    def create_trending_data(self, direction: str = 'up', days: int = 200) -> pd.Series:
        """Create data with clear trend for testing signals."""
        dates = pd.date_range(start='2020-01-01', periods=days, freq='D')
        base_price = 10000

        if direction == 'up':
            # Strong uptrend with small noise
            trend = np.linspace(0, 1, days)
            np.random.seed(42)
            noise = np.random.randn(days) * 0.01
            prices = base_price * (1 + trend + noise)
        elif direction == 'down':
            # Strong downtrend
            trend = np.linspace(1, 0, days)
            np.random.seed(42)
            noise = np.random.randn(days) * 0.01
            prices = base_price * (1 + trend + noise)
        else:
            # Flat/sideways
            np.random.seed(42)
            noise = np.random.randn(days) * 0.01
            prices = base_price * (1 + noise)

        return pd.Series(prices, index=dates)

    def test_generate_signals_return_types(self):
        """Test generate_signals returns correct types."""
        close = self.create_trending_data('up')
        entries, exits, zscore = DSTSIndicators.generate_signals(close)

        assert isinstance(entries, pd.Series)
        assert isinstance(exits, pd.Series)
        assert isinstance(zscore, pd.Series)
        assert entries.dtype == bool
        assert exits.dtype == bool

    def test_generate_signals_lengths(self):
        """Test generate_signals returns same length as input."""
        close = self.create_trending_data('up')
        entries, exits, zscore = DSTSIndicators.generate_signals(close)

        assert len(entries) == len(close)
        assert len(exits) == len(close)
        assert len(zscore) == len(close)

    def test_uptrend_generates_entry(self):
        """Test strong uptrend generates at least one entry."""
        close = self.create_trending_data('up', days=200)
        entries, exits, _ = DSTSIndicators.generate_signals(
            close, period=65, bull_threshold=0.75, bear_threshold=0.0
        )

        assert entries.any(), "Expected entry signal in uptrend"

    def test_entry_exit_mutual_exclusivity(self):
        """Test entry and exit never occur on same bar."""
        close = self.create_trending_data('up')
        entries, exits, _ = DSTSIndicators.generate_signals(close)

        # Entry and exit should never both be True on same bar
        assert not (entries & exits).any()

    def test_state_machine_alternation(self):
        """Test entries and exits alternate correctly."""
        close = self.create_trending_data('up')
        entries, exits, _ = DSTSIndicators.generate_signals(close)

        entry_indices = entries[entries].index.tolist()
        exit_indices = exits[exits].index.tolist()

        if len(entry_indices) >= 2 and len(exit_indices) >= 1:
            # Each entry (except possibly last) should have following exit
            for i, entry_date in enumerate(entry_indices[:-1]):
                exits_after = [e for e in exit_indices if e > entry_date]
                assert len(exits_after) > 0, f"No exit after entry on {entry_date}"

    def test_threshold_validation(self):
        """Test that higher bull threshold leads to fewer entries."""
        close = self.create_trending_data('up', days=300)

        entries_low, _, _ = DSTSIndicators.generate_signals(
            close, bull_threshold=0.5, bear_threshold=0.0
        )
        entries_high, _, _ = DSTSIndicators.generate_signals(
            close, bull_threshold=1.0, bear_threshold=0.0
        )

        # Higher threshold should result in fewer entries
        assert entries_low.sum() >= entries_high.sum()

    def test_vectorized_matches_iterative(self):
        """Test vectorized signal generation matches iterative."""
        close = self.create_trending_data('up', days=200)

        entries_iter, exits_iter, zscore_iter = DSTSIndicators.generate_signals(close)
        entries_vec, exits_vec, zscore_vec = DSTSIndicators.generate_signals_vectorized(close)

        pd.testing.assert_series_equal(entries_iter, entries_vec, check_names=False)
        pd.testing.assert_series_equal(exits_iter, exits_vec, check_names=False)
        pd.testing.assert_series_equal(zscore_iter, zscore_vec, check_names=False)


class TestDSTSGetCurrentSignal:
    """Tests for get_current_signal method."""

    def create_sample_data(self, days: int = 100) -> pd.Series:
        """Create sample close price series."""
        dates = pd.date_range(start='2020-01-01', periods=days, freq='D')
        np.random.seed(42)
        base_price = 10000
        returns = np.random.randn(days) * 0.03
        prices = base_price * np.exp(np.cumsum(returns))
        return pd.Series(prices, index=dates)

    def test_insufficient_data_raises(self):
        """Test insufficient data raises ValueError."""
        close = self.create_sample_data(days=30)  # Less than period
        with pytest.raises(ValueError, match="Insufficient data"):
            DSTSIndicators.get_current_signal(close, period=65)

    def test_returns_dsts_signal(self):
        """Test get_current_signal returns DSTSSignal dataclass."""
        close = self.create_sample_data(days=100)
        signal = DSTSIndicators.get_current_signal(close, period=65)

        assert isinstance(signal, DSTSSignal)
        assert hasattr(signal, 'timestamp')
        assert hasattr(signal, 'close')
        assert hasattr(signal, 'ema')
        assert hasattr(signal, 'std')
        assert hasattr(signal, 'zscore')
        assert hasattr(signal, 'state')

    def test_state_values(self):
        """Test state is one of expected values."""
        close = self.create_sample_data(days=100)
        signal = DSTSIndicators.get_current_signal(close, period=65)

        assert signal.state in ['BULLISH', 'BEARISH', 'NEUTRAL']

    def test_is_long_affects_state(self):
        """Test is_long parameter affects state determination."""
        close = self.create_sample_data(days=100)

        # Get Z-score to understand what state should be
        zscore = DSTSIndicators.calculate_zscore(close, 65).iloc[-1]

        signal_cash = DSTSIndicators.get_current_signal(close, is_long=False)
        signal_long = DSTSIndicators.get_current_signal(close, is_long=True)

        # If Z-score is between thresholds, behavior differs
        if 0.0 < zscore < 0.75:
            assert signal_cash.state == 'BEARISH'  # Below bull threshold
            assert signal_long.state == 'NEUTRAL'  # In position, not exiting


class TestDSTSTradeStatistics:
    """Tests for trade statistics calculation."""

    def create_trade_scenario(self) -> tuple:
        """Create a simple trade scenario for testing."""
        dates = pd.date_range(start='2020-01-01', periods=20, freq='D')
        close = pd.Series(
            [100, 105, 110, 115, 120, 118, 115, 112, 110, 108,
             106, 108, 110, 115, 120, 125, 130, 128, 125, 122],
            index=dates
        )

        entries = pd.Series(False, index=dates)
        exits = pd.Series(False, index=dates)

        # First trade: enter at 100, exit at 110 (10% gain)
        entries.iloc[0] = True
        exits.iloc[4] = True

        # Second trade: enter at 106, exit at 130 (22.6% gain)
        entries.iloc[10] = True
        exits.iloc[16] = True

        return close, entries, exits

    def test_trade_statistics_basic(self):
        """Test basic trade statistics calculation."""
        close, entries, exits = self.create_trade_scenario()
        stats = DSTSIndicators.calculate_trade_statistics(entries, exits, close)

        assert stats['trade_count'] == 2
        assert stats['win_count'] == 2
        assert stats['loss_count'] == 0
        assert stats['win_rate'] == 1.0

    def test_trade_statistics_no_trades(self):
        """Test statistics with no trades."""
        dates = pd.date_range(start='2020-01-01', periods=20, freq='D')
        close = pd.Series(100, index=dates)
        entries = pd.Series(False, index=dates)
        exits = pd.Series(False, index=dates)

        stats = DSTSIndicators.calculate_trade_statistics(entries, exits, close)

        assert stats['trade_count'] == 0
        assert stats['win_rate'] == 0.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
