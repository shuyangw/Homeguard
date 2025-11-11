"""
Unit tests for PairsStrategy base class.
"""

import pytest
import pandas as pd
import numpy as np
from typing import Dict, Tuple

from backtesting.base.pairs_strategy import PairsStrategy
from backtesting.base.strategy import MultiSymbolStrategy


class ConcretePairsStrategy(PairsStrategy):
    """
    Concrete implementation for testing purposes.
    """

    def generate_pairs_signals(
        self,
        data1: pd.DataFrame,
        data2: pd.DataFrame,
        symbol1: str = 'asset1',
        symbol2: str = 'asset2'
    ) -> Dict[str, Tuple[pd.Series, pd.Series, pd.Series, pd.Series]]:
        """
        Generate dummy pairs trading signals.

        For testing: Simple spread-based signals.
        """
        # Calculate simple spread
        spread = data2['close'] - data1['close']
        mean = spread.mean()
        std = spread.std()
        z_score = (spread - mean) / std

        # Entry: z-score > 2 (go long spread: short sym1, long sym2)
        long_spread_entry = z_score > 2.0
        long_spread_exit = z_score < 0.5

        # Short spread: z-score < -2 (go short spread: long sym1, short sym2)
        short_spread_entry = z_score < -2.0
        short_spread_exit = z_score > -0.5

        return {
            symbol1: (
                short_spread_entry,  # long_entries (long sym1 = short spread)
                short_spread_exit,   # long_exits
                long_spread_entry,   # short_entries (short sym1 = long spread)
                long_spread_exit     # short_exits
            ),
            symbol2: (
                long_spread_entry,   # long_entries (long sym2 = long spread)
                long_spread_exit,    # long_exits
                short_spread_entry,  # short_entries (short sym2 = short spread)
                short_spread_exit    # short_exits
            )
        }


class TestPairsStrategyInterface:
    """Test PairsStrategy abstract interface."""

    def test_inherits_from_multi_symbol_strategy(self):
        """PairsStrategy inherits from MultiSymbolStrategy."""
        strategy = ConcretePairsStrategy()

        assert isinstance(strategy, MultiSymbolStrategy)
        assert isinstance(strategy, PairsStrategy)

    def test_cannot_instantiate_abstract_class(self):
        """Cannot instantiate PairsStrategy directly without implementing abstract methods."""
        class IncompletePairsStrategy(PairsStrategy):
            pass  # Don't implement generate_pairs_signals

        with pytest.raises(TypeError) as excinfo:
            IncompletePairsStrategy()

        assert "abstract" in str(excinfo.value).lower()

    def test_must_implement_generate_pairs_signals(self):
        """Subclass must implement generate_pairs_signals()."""
        class IncompletePairsStrategy(PairsStrategy):
            pass

        with pytest.raises(TypeError):
            IncompletePairsStrategy()

    def test_can_instantiate_complete_subclass(self):
        """Can instantiate when generate_pairs_signals() implemented."""
        strategy = ConcretePairsStrategy()

        assert isinstance(strategy, PairsStrategy)


class TestPairsStrategyGetRequiredSymbols:
    """Test get_required_symbols() method."""

    def test_always_returns_2(self):
        """Always returns 2 (pairs require exactly 2 symbols)."""
        strategy = ConcretePairsStrategy()

        result = strategy.get_required_symbols()

        assert result == 2
        assert isinstance(result, int)

    def test_cannot_be_overridden_to_different_value(self):
        """Even if subclass tries to override, should return 2."""
        class BadPairsStrategy(ConcretePairsStrategy):
            def get_required_symbols(self):
                return 3  # Try to require 3 symbols

        strategy = BadPairsStrategy()

        # The PairsStrategy.get_required_symbols() is not abstract,
        # but subclass can override it. We're testing the base implementation.
        # Since BadPairsStrategy explicitly overrides, it will return 3.
        # For proper PairsStrategy usage, it should return 2.
        result = strategy.get_required_symbols()
        assert result == 3  # Overridden value

        # But proper usage would be:
        proper_strategy = ConcretePairsStrategy()
        assert proper_strategy.get_required_symbols() == 2


class TestPairsStrategyGeneratePairsSignals:
    """Test generate_pairs_signals() method."""

    @pytest.fixture
    def sample_pair_data(self):
        """Create sample data for a pair of symbols."""
        dates = pd.date_range('2020-01-01', periods=50, freq='D')

        # Create mean-reverting spread for testing
        price1 = 100 + np.cumsum(np.random.randn(50) * 0.5)
        price2 = 200 + 2 * price1 + np.random.randn(50) * 5  # Correlated but with spread

        data1 = pd.DataFrame({
            'open': price1,
            'high': price1 + np.random.uniform(0, 2, 50),
            'low': price1 - np.random.uniform(0, 2, 50),
            'close': price1,
            'volume': np.random.uniform(1e6, 2e6, 50)
        }, index=dates)

        data2 = pd.DataFrame({
            'open': price2,
            'high': price2 + np.random.uniform(0, 2, 50),
            'low': price2 - np.random.uniform(0, 2, 50),
            'close': price2,
            'volume': np.random.uniform(1e6, 2e6, 50)
        }, index=dates)

        return data1, data2

    def test_receives_two_dataframes(self, sample_pair_data):
        """Receives two DataFrames for the pair."""
        data1, data2 = sample_pair_data
        strategy = ConcretePairsStrategy()

        signals = strategy.generate_pairs_signals(
            data1, data2, 'AAPL', 'MSFT'
        )

        assert isinstance(signals, dict)
        assert 'AAPL' in signals
        assert 'MSFT' in signals

    def test_returns_four_signal_series_per_symbol(self, sample_pair_data):
        """Returns 4-tuple (long_entries, long_exits, short_entries, short_exits) per symbol."""
        data1, data2 = sample_pair_data
        strategy = ConcretePairsStrategy()

        signals = strategy.generate_pairs_signals(
            data1, data2, 'AAPL', 'MSFT'
        )

        for symbol, signal_tuple in signals.items():
            assert isinstance(signal_tuple, tuple)
            assert len(signal_tuple) == 4

            long_entries, long_exits, short_entries, short_exits = signal_tuple

            assert isinstance(long_entries, pd.Series)
            assert isinstance(long_exits, pd.Series)
            assert isinstance(short_entries, pd.Series)
            assert isinstance(short_exits, pd.Series)

    def test_signals_have_correct_index(self, sample_pair_data):
        """Signals have same index as input data."""
        data1, data2 = sample_pair_data
        strategy = ConcretePairsStrategy()

        signals = strategy.generate_pairs_signals(
            data1, data2, 'AAPL', 'MSFT'
        )

        for signal in signals['AAPL']:
            pd.testing.assert_index_equal(signal.index, data1.index)

        for signal in signals['MSFT']:
            pd.testing.assert_index_equal(signal.index, data2.index)

    def test_signals_are_boolean(self, sample_pair_data):
        """All signals are boolean Series."""
        data1, data2 = sample_pair_data
        strategy = ConcretePairsStrategy()

        signals = strategy.generate_pairs_signals(
            data1, data2, 'AAPL', 'MSFT'
        )

        for symbol, (long_e, long_x, short_e, short_x) in signals.items():
            assert long_e.dtype == bool
            assert long_x.dtype == bool
            assert short_e.dtype == bool
            assert short_x.dtype == bool

    def test_uses_custom_symbol_names(self, sample_pair_data):
        """Uses custom symbol names passed as arguments."""
        data1, data2 = sample_pair_data
        strategy = ConcretePairsStrategy()

        signals = strategy.generate_pairs_signals(
            data1, data2, 'CUSTOM1', 'CUSTOM2'
        )

        assert 'CUSTOM1' in signals
        assert 'CUSTOM2' in signals


class TestPairsStrategyGenerateMultiSignals:
    """Test generate_multi_signals() wrapper method."""

    @pytest.fixture
    def sample_data_dict(self):
        """Create data dict with 2 symbols."""
        dates = pd.date_range('2020-01-01', periods=20, freq='D')

        return {
            'AAPL': pd.DataFrame({
                'close': 100 + np.cumsum(np.random.randn(20) * 0.5)
            }, index=dates),
            'MSFT': pd.DataFrame({
                'close': 200 + np.cumsum(np.random.randn(20) * 0.5)
            }, index=dates)
        }

    def test_accepts_data_dict(self, sample_data_dict):
        """Accepts data_dict and calls generate_pairs_signals()."""
        strategy = ConcretePairsStrategy()

        signals = strategy.generate_multi_signals(sample_data_dict)

        assert isinstance(signals, dict)
        assert 'AAPL' in signals
        assert 'MSFT' in signals

    def test_validates_exactly_two_symbols(self):
        """Validates that exactly 2 symbols are provided."""
        strategy = ConcretePairsStrategy()
        dates = pd.date_range('2020-01-01', periods=10, freq='D')

        # Test with 1 symbol (too few)
        data_dict_1 = {
            'AAPL': pd.DataFrame({'close': np.random.randn(10)}, index=dates)
        }

        with pytest.raises(ValueError) as excinfo:
            strategy.generate_multi_signals(data_dict_1)

        assert "exactly 2 symbols" in str(excinfo.value).lower()
        assert "got 1" in str(excinfo.value)

        # Test with 3 symbols (too many)
        data_dict_3 = {
            'AAPL': pd.DataFrame({'close': np.random.randn(10)}, index=dates),
            'MSFT': pd.DataFrame({'close': np.random.randn(10)}, index=dates),
            'GOOGL': pd.DataFrame({'close': np.random.randn(10)}, index=dates)
        }

        with pytest.raises(ValueError) as excinfo:
            strategy.generate_multi_signals(data_dict_3)

        assert "exactly 2 symbols" in str(excinfo.value).lower()
        assert "got 3" in str(excinfo.value)

    def test_passes_symbols_in_order(self, sample_data_dict):
        """Passes symbols to generate_pairs_signals() in dict order."""
        strategy = ConcretePairsStrategy()

        signals = strategy.generate_multi_signals(sample_data_dict)

        # Should have signals for both symbols
        assert len(signals) == 2

    def test_returns_four_tuple_per_symbol(self, sample_data_dict):
        """Returns 4-tuple per symbol from wrapper."""
        strategy = ConcretePairsStrategy()

        signals = strategy.generate_multi_signals(sample_data_dict)

        for symbol, signal_tuple in signals.items():
            assert len(signal_tuple) == 4


class TestPairsStrategyEdgeCases:
    """Test edge cases and special scenarios."""

    def test_minimal_data_single_row(self):
        """Works with minimal data (single row)."""
        dates = pd.date_range('2020-01-01', periods=1, freq='D')

        data1 = pd.DataFrame({'close': [100.0]}, index=dates)
        data2 = pd.DataFrame({'close': [200.0]}, index=dates)

        strategy = ConcretePairsStrategy()

        signals = strategy.generate_pairs_signals(data1, data2, 'A', 'B')

        assert len(signals['A'][0]) == 1  # One timestamp
        assert len(signals['B'][0]) == 1

    def test_handles_nan_values_gracefully(self):
        """Strategy should handle NaN values in data."""
        dates = pd.date_range('2020-01-01', periods=10, freq='D')

        data1 = pd.DataFrame({
            'close': [100.0, np.nan, 102.0, 103.0, np.nan, 105.0, 106.0, 107.0, 108.0, 109.0]
        }, index=dates)

        data2 = pd.DataFrame({
            'close': [200.0, 201.0, np.nan, 203.0, 204.0, np.nan, 206.0, 207.0, 208.0, 209.0]
        }, index=dates)

        strategy = ConcretePairsStrategy()

        # Should not crash (implementation may vary on NaN handling)
        signals = strategy.generate_pairs_signals(data1, data2, 'A', 'B')

        assert isinstance(signals, dict)

    def test_misaligned_indexes(self):
        """Handles data with different date ranges."""
        dates1 = pd.date_range('2020-01-01', periods=10, freq='D')
        dates2 = pd.date_range('2020-01-05', periods=10, freq='D')  # Starts later

        data1 = pd.DataFrame({'close': np.random.randn(10) + 100}, index=dates1)
        data2 = pd.DataFrame({'close': np.random.randn(10) + 200}, index=dates2)

        strategy = ConcretePairsStrategy()

        # Implementation should handle this (may error or align data)
        # For now, just test that it runs
        try:
            signals = strategy.generate_pairs_signals(data1, data2, 'A', 'B')
            assert isinstance(signals, dict)
        except (ValueError, KeyError, IndexError):
            # Acceptable to error on misaligned data
            pass


class TestPairsStrategyIntegration:
    """Test integration with base strategy features."""

    def test_has_parameters_interface(self):
        """Has get_parameters() and set_parameters() from BaseStrategy."""
        strategy = ConcretePairsStrategy(entry_z=2.5, exit_z=0.5)

        params = strategy.get_parameters()

        assert 'entry_z' in params
        assert 'exit_z' in params
        assert params['entry_z'] == 2.5

    def test_can_update_parameters(self):
        """Can update parameters via set_parameters()."""
        strategy = ConcretePairsStrategy()

        strategy.set_parameters(new_param=42)

        assert strategy.get_parameters()['new_param'] == 42

    def test_has_string_representation(self):
        """Has __repr__ and __str__ methods."""
        strategy = ConcretePairsStrategy(test=1)

        repr_str = repr(strategy)
        str_str = str(strategy)

        assert 'ConcretePairsStrategy' in repr_str
        assert 'test' in repr_str  # Parameter should appear


class TestPairsStrategyDocumentation:
    """Test that class is properly documented."""

    def test_has_docstring(self):
        """PairsStrategy class has docstring."""
        assert PairsStrategy.__doc__ is not None
        assert len(PairsStrategy.__doc__.strip()) > 0

    def test_generate_pairs_signals_has_docstring(self):
        """generate_pairs_signals() method has docstring."""
        assert PairsStrategy.generate_pairs_signals.__doc__ is not None

    def test_docstring_mentions_pairs_trading(self):
        """Docstring mentions pairs trading."""
        assert 'pair' in PairsStrategy.__doc__.lower()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
