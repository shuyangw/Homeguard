"""
Unit tests for MultiSymbolStrategy base class.
"""

import pytest
import pandas as pd
import numpy as np
from typing import Dict, Union, Tuple, List

from backtesting.base.strategy import MultiSymbolStrategy


class ConcreteMultiSymbolStrategy(MultiSymbolStrategy):
    """
    Concrete implementation for testing purposes.
    """

    def __init__(self, required_symbols=2, **kwargs):
        self._required_symbols = required_symbols
        super().__init__(**kwargs)

    def get_required_symbols(self) -> Union[int, List[str]]:
        return self._required_symbols

    def generate_multi_signals(
        self,
        data_dict: Dict[str, pd.DataFrame]
    ) -> Dict[str, Union[
        Tuple[pd.Series, pd.Series],
        Tuple[pd.Series, pd.Series, pd.Series, pd.Series]
    ]]:
        """Generate dummy signals for testing."""
        signals = {}
        for symbol, data in data_dict.items():
            # Simple dummy signals: entries at even indices, exits at odd
            entries = pd.Series(
                [i % 2 == 0 for i in range(len(data))],
                index=data.index
            )
            exits = pd.Series(
                [i % 2 == 1 for i in range(len(data))],
                index=data.index
            )
            signals[symbol] = (entries, exits)
        return signals


class TestMultiSymbolStrategyInterface:
    """Test MultiSymbolStrategy abstract interface."""

    def test_cannot_instantiate_abstract_class(self):
        """Cannot instantiate MultiSymbolStrategy directly."""
        with pytest.raises(TypeError) as excinfo:
            MultiSymbolStrategy()

        assert "abstract" in str(excinfo.value).lower()

    def test_must_implement_get_required_symbols(self):
        """Subclass must implement get_required_symbols()."""
        class IncompleteStrategy(MultiSymbolStrategy):
            def generate_multi_signals(self, data_dict):
                return {}

        with pytest.raises(TypeError) as excinfo:
            IncompleteStrategy()

        assert "abstract" in str(excinfo.value).lower()

    def test_must_implement_generate_multi_signals(self):
        """Subclass must implement generate_multi_signals()."""
        class IncompleteStrategy(MultiSymbolStrategy):
            def get_required_symbols(self):
                return 2

        with pytest.raises(TypeError) as excinfo:
            IncompleteStrategy()

        assert "abstract" in str(excinfo.value).lower()

    def test_can_instantiate_complete_subclass(self):
        """Can instantiate when all abstract methods implemented."""
        strategy = ConcreteMultiSymbolStrategy(required_symbols=2)

        assert isinstance(strategy, MultiSymbolStrategy)
        assert strategy.get_required_symbols() == 2


class TestMultiSymbolStrategyGetRequiredSymbols:
    """Test get_required_symbols() method."""

    def test_returns_integer_count(self):
        """Can return integer count of required symbols."""
        strategy = ConcreteMultiSymbolStrategy(required_symbols=2)

        result = strategy.get_required_symbols()

        assert isinstance(result, int)
        assert result == 2

    def test_returns_list_of_specific_symbols(self):
        """Can return list of specific symbol names."""
        class SpecificSymbolStrategy(ConcreteMultiSymbolStrategy):
            def get_required_symbols(self) -> List[str]:
                return ['SPY', 'TLT']

        strategy = SpecificSymbolStrategy()
        result = strategy.get_required_symbols()

        assert isinstance(result, list)
        assert result == ['SPY', 'TLT']

    def test_different_symbol_counts(self):
        """Can require different numbers of symbols."""
        for count in [2, 3, 5, 10]:
            strategy = ConcreteMultiSymbolStrategy(required_symbols=count)
            assert strategy.get_required_symbols() == count


class TestMultiSymbolStrategyGenerateMultiSignals:
    """Test generate_multi_signals() method."""

    @pytest.fixture
    def sample_data_dict(self):
        """Create sample data for multiple symbols."""
        dates = pd.date_range('2020-01-01', periods=10, freq='D')

        data_dict = {
            'AAPL': pd.DataFrame({
                'open': np.random.uniform(100, 110, 10),
                'high': np.random.uniform(110, 120, 10),
                'low': np.random.uniform(90, 100, 10),
                'close': np.random.uniform(100, 110, 10),
                'volume': np.random.uniform(1e6, 2e6, 10)
            }, index=dates),
            'MSFT': pd.DataFrame({
                'open': np.random.uniform(200, 210, 10),
                'high': np.random.uniform(210, 220, 10),
                'low': np.random.uniform(190, 200, 10),
                'close': np.random.uniform(200, 210, 10),
                'volume': np.random.uniform(1e6, 2e6, 10)
            }, index=dates)
        }

        return data_dict

    def test_receives_dict_of_dataframes(self, sample_data_dict):
        """Receives dict mapping symbols to DataFrames."""
        strategy = ConcreteMultiSymbolStrategy()

        signals = strategy.generate_multi_signals(sample_data_dict)

        assert isinstance(signals, dict)
        assert 'AAPL' in signals
        assert 'MSFT' in signals

    def test_returns_dict_of_signal_tuples(self, sample_data_dict):
        """Returns dict mapping symbols to signal tuples."""
        strategy = ConcreteMultiSymbolStrategy()

        signals = strategy.generate_multi_signals(sample_data_dict)

        for symbol, signal_tuple in signals.items():
            assert isinstance(signal_tuple, tuple)
            assert len(signal_tuple) >= 2  # At least entries and exits

            for signal in signal_tuple:
                assert isinstance(signal, pd.Series)

    def test_signals_have_same_index_as_data(self, sample_data_dict):
        """Generated signals have same index as input data."""
        strategy = ConcreteMultiSymbolStrategy()

        signals = strategy.generate_multi_signals(sample_data_dict)

        for symbol, (entries, exits) in signals.items():
            expected_index = sample_data_dict[symbol].index
            pd.testing.assert_index_equal(entries.index, expected_index)
            pd.testing.assert_index_equal(exits.index, expected_index)

    def test_works_with_different_symbol_counts(self):
        """Works with different numbers of symbols."""
        dates = pd.date_range('2020-01-01', periods=5, freq='D')

        for n_symbols in [2, 3, 5]:
            data_dict = {}
            for i in range(n_symbols):
                symbol = f'SYM{i}'
                data_dict[symbol] = pd.DataFrame({
                    'close': np.random.uniform(100, 110, 5)
                }, index=dates)

            strategy = ConcreteMultiSymbolStrategy(required_symbols=n_symbols)
            signals = strategy.generate_multi_signals(data_dict)

            assert len(signals) == n_symbols


class TestMultiSymbolStrategyGenerateSignals:
    """Test that generate_signals() raises NotImplementedError."""

    @pytest.fixture
    def single_symbol_data(self):
        """Create sample data for single symbol."""
        dates = pd.date_range('2020-01-01', periods=10, freq='D')
        return pd.DataFrame({
            'open': np.random.uniform(100, 110, 10),
            'high': np.random.uniform(110, 120, 10),
            'low': np.random.uniform(90, 100, 10),
            'close': np.random.uniform(100, 110, 10),
            'volume': np.random.uniform(1e6, 2e6, 10)
        }, index=dates)

    def test_raises_not_implemented_error(self, single_symbol_data):
        """Calling generate_signals() raises NotImplementedError."""
        strategy = ConcreteMultiSymbolStrategy()

        with pytest.raises(NotImplementedError) as excinfo:
            strategy.generate_signals(single_symbol_data)

        error_msg = str(excinfo.value)
        assert "multi-symbol strategy" in error_msg.lower()
        assert "multiple symbols" in error_msg.lower()

    def test_error_message_includes_strategy_name(self, single_symbol_data):
        """Error message includes strategy class name."""
        strategy = ConcreteMultiSymbolStrategy()

        with pytest.raises(NotImplementedError) as excinfo:
            strategy.generate_signals(single_symbol_data)

        assert "ConcreteMultiSymbolStrategy" in str(excinfo.value)

    def test_error_message_includes_usage_hint(self, single_symbol_data):
        """Error message includes hint about correct usage."""
        strategy = ConcreteMultiSymbolStrategy()

        with pytest.raises(NotImplementedError) as excinfo:
            strategy.generate_signals(single_symbol_data)

        error_msg = str(excinfo.value)
        assert "BacktestEngine.run()" in error_msg
        assert "symbols=[" in error_msg


class TestMultiSymbolStrategyInheritance:
    """Test that MultiSymbolStrategy inherits from BaseStrategy."""

    def test_inherits_from_base_strategy(self):
        """MultiSymbolStrategy inherits from BaseStrategy."""
        from backtesting.base.strategy import BaseStrategy

        strategy = ConcreteMultiSymbolStrategy()

        assert isinstance(strategy, BaseStrategy)
        assert isinstance(strategy, MultiSymbolStrategy)

    def test_has_base_strategy_methods(self):
        """Has methods inherited from BaseStrategy."""
        strategy = ConcreteMultiSymbolStrategy(test_param=42)

        # BaseStrategy methods
        assert hasattr(strategy, 'get_parameters')
        assert hasattr(strategy, 'set_parameters')
        assert hasattr(strategy, 'validate_parameters')

        # Can use get_parameters()
        params = strategy.get_parameters()
        assert 'test_param' in params
        assert params['test_param'] == 42

    def test_can_set_parameters(self):
        """Can use set_parameters() from BaseStrategy."""
        strategy = ConcreteMultiSymbolStrategy()

        strategy.set_parameters(new_param=100)

        params = strategy.get_parameters()
        assert 'new_param' in params
        assert params['new_param'] == 100


class TestMultiSymbolStrategyEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_data_dict(self):
        """Handles empty data dict."""
        strategy = ConcreteMultiSymbolStrategy()

        signals = strategy.generate_multi_signals({})

        assert isinstance(signals, dict)
        assert len(signals) == 0

    def test_single_timestamp_data(self):
        """Handles data with single timestamp."""
        dates = pd.date_range('2020-01-01', periods=1, freq='D')
        data_dict = {
            'AAPL': pd.DataFrame({'close': [100.0]}, index=dates),
            'MSFT': pd.DataFrame({'close': [200.0]}, index=dates)
        }

        strategy = ConcreteMultiSymbolStrategy()
        signals = strategy.generate_multi_signals(data_dict)

        assert len(signals['AAPL'][0]) == 1  # entries
        assert len(signals['AAPL'][1]) == 1  # exits

    def test_very_large_symbol_count(self):
        """Handles large number of symbols."""
        dates = pd.date_range('2020-01-01', periods=5, freq='D')
        data_dict = {}

        n_symbols = 100
        for i in range(n_symbols):
            data_dict[f'SYM{i}'] = pd.DataFrame({
                'close': np.random.uniform(100, 110, 5)
            }, index=dates)

        strategy = ConcreteMultiSymbolStrategy(required_symbols=n_symbols)
        signals = strategy.generate_multi_signals(data_dict)

        assert len(signals) == n_symbols


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
