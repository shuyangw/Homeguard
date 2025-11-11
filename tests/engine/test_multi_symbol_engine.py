"""
Unit tests for BacktestEngine multi-symbol strategy support (Phase 2).
"""

import pytest
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Union, List
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from backtesting.engine.backtest_engine import BacktestEngine
from backtesting.base.strategy import BaseStrategy, MultiSymbolStrategy
from backtesting.base.pairs_strategy import PairsStrategy
from strategies.base_strategies.moving_average import MovingAverageCrossover


class TestMultiSymbolStrategy(MultiSymbolStrategy):
    """Concrete multi-symbol strategy for testing."""

    def __init__(self, required_symbols=2, **kwargs):
        self._required_symbols = required_symbols
        super().__init__(**kwargs)

    def get_required_symbols(self) -> Union[int, List[str]]:
        return self._required_symbols

    def generate_multi_signals(
        self,
        data_dict: Dict[str, pd.DataFrame]
    ) -> Dict[str, Tuple[pd.Series, pd.Series, pd.Series, pd.Series]]:
        """Generate dummy signals for testing."""
        signals = {}
        for symbol, data in data_dict.items():
            # Simple spread-based signals
            entries = pd.Series([i % 5 == 0 for i in range(len(data))], index=data.index)
            exits = pd.Series([i % 5 == 3 for i in range(len(data))], index=data.index)
            short_entries = pd.Series([i % 7 == 0 for i in range(len(data))], index=data.index)
            short_exits = pd.Series([i % 7 == 4 for i in range(len(data))], index=data.index)
            signals[symbol] = (entries, exits, short_entries, short_exits)
        return signals


class TestPairsTestStrategy(PairsStrategy):
    """Concrete pairs strategy for testing."""

    def generate_pairs_signals(
        self,
        data1: pd.DataFrame,
        data2: pd.DataFrame,
        symbol1: str = 'asset1',
        symbol2: str = 'asset2'
    ) -> Dict[str, Tuple[pd.Series, pd.Series, pd.Series, pd.Series]]:
        """Generate simple spread-based signals."""
        spread = data2['close'] - data1['close']
        mean = spread.mean()
        std = spread.std()
        z_score = (spread - mean) / std

        long_spread_entry = z_score > 1.0
        long_spread_exit = z_score < 0.5
        short_spread_entry = z_score < -1.0
        short_spread_exit = z_score > -0.5

        return {
            symbol1: (
                short_spread_entry, short_spread_exit,
                long_spread_entry, long_spread_exit
            ),
            symbol2: (
                long_spread_entry, long_spread_exit,
                short_spread_entry, short_spread_exit
            )
        }


class TestStrategyTypeDetection:
    """Test _detect_strategy_type() method."""

    def test_detects_multi_symbol_strategy(self):
        """Detects MultiSymbolStrategy correctly."""
        engine = BacktestEngine()
        strategy = TestMultiSymbolStrategy(required_symbols=2)

        result = engine._detect_strategy_type(strategy)

        assert result is True

    def test_detects_pairs_strategy(self):
        """Detects PairsStrategy (subclass of MultiSymbolStrategy)."""
        engine = BacktestEngine()
        strategy = TestPairsTestStrategy()

        result = engine._detect_strategy_type(strategy)

        assert result is True

    def test_detects_regular_strategy(self):
        """Detects regular single-symbol strategy."""
        engine = BacktestEngine()
        strategy = MovingAverageCrossover(fast_window=10, slow_window=20)

        result = engine._detect_strategy_type(strategy)

        assert result is False

    def test_detection_is_not_fooled_by_naming(self):
        """Type detection uses isinstance(), not naming."""
        engine = BacktestEngine()

        # Regular strategy with misleading name
        class MultiSymbolNamed(BaseStrategy):
            def generate_signals(self, data):
                entries = pd.Series(False, index=data.index)
                exits = pd.Series(False, index=data.index)
                return entries, exits

        strategy = MultiSymbolNamed()
        result = engine._detect_strategy_type(strategy)

        assert result is False


class TestSymbolDataSynchronization:
    """Test _synchronize_symbol_data() method."""

    @pytest.fixture
    def multi_symbol_data_aligned(self):
        """Create multi-symbol data with aligned timestamps."""
        dates = pd.date_range('2023-01-01', periods=50, freq='D')
        symbols = ['AAPL', 'MSFT']
        dfs = []

        for symbol in symbols:
            df = pd.DataFrame({
                'open': np.random.uniform(100, 110, 50),
                'high': np.random.uniform(110, 120, 50),
                'low': np.random.uniform(90, 100, 50),
                'close': np.random.uniform(100, 110, 50),
                'volume': np.random.uniform(1e6, 2e6, 50),
                'symbol': symbol
            }, index=dates)
            dfs.append(df)

        combined = pd.concat(dfs)
        combined = combined.set_index('symbol', append=True)
        combined = combined.swaplevel()
        combined = combined.sort_index()

        return combined

    @pytest.fixture
    def multi_symbol_data_misaligned(self):
        """Create multi-symbol data with misaligned timestamps."""
        dates1 = pd.date_range('2023-01-01', periods=50, freq='D')
        dates2 = pd.date_range('2023-01-05', periods=50, freq='D')  # Starts 4 days later

        df1 = pd.DataFrame({
            'open': np.random.uniform(100, 110, 50),
            'high': np.random.uniform(110, 120, 50),
            'low': np.random.uniform(90, 100, 50),
            'close': np.random.uniform(100, 110, 50),
            'volume': np.random.uniform(1e6, 2e6, 50),
            'symbol': 'AAPL'
        }, index=dates1)

        df2 = pd.DataFrame({
            'open': np.random.uniform(200, 210, 50),
            'high': np.random.uniform(210, 220, 50),
            'low': np.random.uniform(190, 200, 50),
            'close': np.random.uniform(200, 210, 50),
            'volume': np.random.uniform(1e6, 2e6, 50),
            'symbol': 'MSFT'
        }, index=dates2)

        combined = pd.concat([df1, df2])
        combined = combined.set_index('symbol', append=True)
        combined = combined.swaplevel()
        combined = combined.sort_index()

        return combined

    def test_synchronizes_aligned_data(self, multi_symbol_data_aligned):
        """Synchronizes data with aligned timestamps."""
        engine = BacktestEngine()
        symbols = ['AAPL', 'MSFT']

        result = engine._synchronize_symbol_data(multi_symbol_data_aligned, symbols)

        assert isinstance(result, dict)
        assert len(result) == 2
        assert 'AAPL' in result
        assert 'MSFT' in result

        # All DataFrames should have same index
        aapl_index = result['AAPL'].index
        msft_index = result['MSFT'].index
        assert len(aapl_index) == len(msft_index)
        assert aapl_index.equals(msft_index)

    def test_synchronizes_misaligned_data(self, multi_symbol_data_misaligned):
        """Synchronizes data with misaligned timestamps (finds intersection)."""
        engine = BacktestEngine()
        symbols = ['AAPL', 'MSFT']

        result = engine._synchronize_symbol_data(multi_symbol_data_misaligned, symbols)

        # Should find common timestamps (intersection)
        assert isinstance(result, dict)
        assert len(result) == 2

        aapl_index = result['AAPL'].index
        msft_index = result['MSFT'].index
        assert aapl_index.equals(msft_index)

        # Common period should be smaller than original
        assert len(aapl_index) < 50

    def test_raises_on_no_overlap(self):
        """Raises ValueError when no overlapping timestamps."""
        dates1 = pd.date_range('2023-01-01', periods=10, freq='D')
        dates2 = pd.date_range('2023-02-01', periods=10, freq='D')  # No overlap

        df1 = pd.DataFrame({
            'close': np.random.uniform(100, 110, 10),
            'symbol': 'AAPL'
        }, index=dates1)

        df2 = pd.DataFrame({
            'close': np.random.uniform(200, 210, 10),
            'symbol': 'MSFT'
        }, index=dates2)

        combined = pd.concat([df1, df2])
        combined = combined.set_index('symbol', append=True)
        combined = combined.swaplevel()
        combined = combined.sort_index()

        engine = BacktestEngine()

        with pytest.raises(ValueError) as excinfo:
            engine._synchronize_symbol_data(combined, ['AAPL', 'MSFT'])

        assert "no overlapping timestamps" in str(excinfo.value).lower()

    def test_raises_on_missing_symbol(self, multi_symbol_data_aligned):
        """Raises ValueError when requested symbol not in data."""
        engine = BacktestEngine()

        with pytest.raises(ValueError) as excinfo:
            engine._synchronize_symbol_data(
                multi_symbol_data_aligned,
                ['AAPL', 'GOOGL']  # GOOGL not in data
            )

        assert "googl" in str(excinfo.value).lower()
        assert "not found" in str(excinfo.value).lower()

    def test_returns_all_ohlcv_columns(self, multi_symbol_data_aligned):
        """Synchronized data preserves all OHLCV columns."""
        engine = BacktestEngine()
        result = engine._synchronize_symbol_data(
            multi_symbol_data_aligned,
            ['AAPL', 'MSFT']
        )

        for symbol, df in result.items():
            assert 'open' in df.columns
            assert 'high' in df.columns
            assert 'low' in df.columns
            assert 'close' in df.columns
            assert 'volume' in df.columns


class TestSymbolRequirementValidation:
    """Test symbol requirement validation in run() method."""

    @pytest.fixture
    def aligned_data(self):
        """Create aligned multi-symbol data."""
        dates = pd.date_range('2023-01-01', periods=30, freq='D')
        symbols = ['AAPL', 'MSFT', 'GOOGL']
        dfs = []

        for symbol in symbols:
            df = pd.DataFrame({
                'open': np.random.uniform(100, 110, 30),
                'high': np.random.uniform(110, 120, 30),
                'low': np.random.uniform(90, 100, 30),
                'close': np.random.uniform(100, 110, 30),
                'volume': np.random.uniform(1e6, 2e6, 30),
                'symbol': symbol
            }, index=dates)
            dfs.append(df)

        combined = pd.concat(dfs)
        combined = combined.set_index('symbol', append=True)
        combined = combined.swaplevel()
        combined = combined.sort_index()

        return combined

    def test_validates_integer_requirement_exact(self, aligned_data):
        """Validates exact symbol count requirement (int)."""
        engine = BacktestEngine()
        strategy = TestMultiSymbolStrategy(required_symbols=2)

        # Should succeed with exactly 2 symbols
        symbols = ['AAPL', 'MSFT']
        # Extract subset of data for these symbols
        subset_data = aligned_data.loc[symbols, :]

        # Should not raise
        portfolio = engine._run_multi_symbol_strategy(strategy, subset_data, symbols, 'close')
        assert portfolio is not None

    def test_rejects_wrong_symbol_count(self, aligned_data):
        """Rejects wrong number of symbols for integer requirement."""
        engine = BacktestEngine()
        strategy = TestMultiSymbolStrategy(required_symbols=2)

        symbols = ['AAPL', 'MSFT', 'GOOGL']  # 3 symbols, but strategy requires 2

        with pytest.raises(ValueError) as excinfo:
            engine.run(
                strategy,
                symbols=symbols,
                start_date='2023-01-01',
                end_date='2023-01-31'
            )

        assert "requires exactly 2 symbols" in str(excinfo.value)
        assert "got 3" in str(excinfo.value)

    def test_validates_list_requirement_specific_symbols(self):
        """Validates specific symbol list requirement."""
        class SpecificSymbolStrategy(MultiSymbolStrategy):
            def get_required_symbols(self) -> List[str]:
                return ['SPY', 'TLT']

            def generate_multi_signals(self, data_dict):
                return {
                    'SPY': (
                        pd.Series(False, index=data_dict['SPY'].index),
                        pd.Series(False, index=data_dict['SPY'].index),
                        pd.Series(False, index=data_dict['SPY'].index),
                        pd.Series(False, index=data_dict['SPY'].index)
                    ),
                    'TLT': (
                        pd.Series(False, index=data_dict['TLT'].index),
                        pd.Series(False, index=data_dict['TLT'].index),
                        pd.Series(False, index=data_dict['TLT'].index),
                        pd.Series(False, index=data_dict['TLT'].index)
                    )
                }

        engine = BacktestEngine()
        strategy = SpecificSymbolStrategy()

        with pytest.raises(ValueError) as excinfo:
            engine.run(
                strategy,
                symbols=['AAPL', 'MSFT'],  # Wrong symbols
                start_date='2023-01-01',
                end_date='2023-01-31'
            )

        assert "requires specific symbols" in str(excinfo.value).lower()
        assert "spy" in str(excinfo.value).lower()
        assert "tlt" in str(excinfo.value).lower()

    def test_rejects_single_symbol_for_multi_strategy(self):
        """Rejects single symbol for multi-symbol strategy."""
        engine = BacktestEngine()
        strategy = TestMultiSymbolStrategy(required_symbols=2)

        with pytest.raises(ValueError) as excinfo:
            engine.run(
                strategy,
                symbols=['AAPL'],  # Only 1 symbol
                start_date='2023-01-01',
                end_date='2023-01-31'
            )

        assert "multi-symbol strategy" in str(excinfo.value).lower()
        assert "requires at least 2 symbols" in str(excinfo.value).lower()


class TestPairsStrategyIntegration:
    """Test integration with PairsStrategy."""

    @pytest.fixture
    def pairs_data(self):
        """Create correlated pair data."""
        dates = pd.date_range('2023-01-01', periods=50, freq='D')
        np.random.seed(42)

        # Create correlated prices
        base = 100 + np.cumsum(np.random.randn(50) * 0.5)
        price1 = base + np.random.randn(50) * 2
        price2 = 2 * base + np.random.randn(50) * 3

        df1 = pd.DataFrame({
            'open': price1,
            'high': price1 + np.abs(np.random.randn(50) * 0.5),
            'low': price1 - np.abs(np.random.randn(50) * 0.5),
            'close': price1,
            'volume': np.random.uniform(1e6, 2e6, 50),
            'symbol': 'AAPL'
        }, index=dates)

        df2 = pd.DataFrame({
            'open': price2,
            'high': price2 + np.abs(np.random.randn(50) * 0.5),
            'low': price2 - np.abs(np.random.randn(50) * 0.5),
            'close': price2,
            'volume': np.random.uniform(1e6, 2e6, 50),
            'symbol': 'MSFT'
        }, index=dates)

        combined = pd.concat([df1, df2])
        combined = combined.set_index('symbol', append=True)
        combined = combined.swaplevel()
        combined = combined.sort_index()

        return combined

    def test_pairs_strategy_requires_exactly_2_symbols(self):
        """PairsStrategy requires exactly 2 symbols."""
        engine = BacktestEngine()
        strategy = TestPairsTestStrategy()

        # Should reject 3 symbols
        with pytest.raises(ValueError) as excinfo:
            engine.run(
                strategy,
                symbols=['AAPL', 'MSFT', 'GOOGL'],
                start_date='2023-01-01',
                end_date='2023-01-31'
            )

        assert "requires exactly 2 symbols" in str(excinfo.value)

    def test_pairs_strategy_executes_with_2_symbols(self, pairs_data):
        """PairsStrategy executes successfully with 2 symbols."""
        engine = BacktestEngine(initial_capital=10000.0, fees=0.001)
        strategy = TestPairsTestStrategy()

        # Should execute without error
        portfolio = engine._run_multi_symbol_strategy(
            strategy,
            pairs_data,
            ['AAPL', 'MSFT'],
            'close'
        )

        assert portfolio is not None
        assert hasattr(portfolio, 'stats')

    def test_pairs_strategy_generates_synchronized_signals(self, pairs_data):
        """PairsStrategy generates signals for both symbols."""
        engine = BacktestEngine()
        strategy = TestPairsTestStrategy()

        data_dict = engine._synchronize_symbol_data(pairs_data, ['AAPL', 'MSFT'])
        signals = strategy.generate_multi_signals(data_dict)

        assert 'AAPL' in signals
        assert 'MSFT' in signals

        # Both should have 4-tuple (long/short signals)
        assert len(signals['AAPL']) == 4
        assert len(signals['MSFT']) == 4

        # Signals should have same length
        assert len(signals['AAPL'][0]) == len(signals['MSFT'][0])


class TestMultiSymbolEdgeCases:
    """Test edge cases for multi-symbol strategies."""

    def test_empty_signals_no_crash(self, multi_symbol_data):
        """Handles multi-symbol strategy with no signals."""
        class NoSignalStrategy(MultiSymbolStrategy):
            def get_required_symbols(self):
                return 2

            def generate_multi_signals(self, data_dict):
                signals = {}
                for symbol, data in data_dict.items():
                    empty = pd.Series(False, index=data.index)
                    signals[symbol] = (empty, empty, empty, empty)
                return signals

        engine = BacktestEngine(initial_capital=10000.0)
        strategy = NoSignalStrategy()

        # Should not crash
        portfolio = engine._run_multi_symbol_strategy(
            strategy,
            multi_symbol_data,
            ['AAPL', 'MSFT'],
            'close'
        )

        assert portfolio is not None

    def test_minimal_common_timestamps(self):
        """Handles minimal overlapping timestamps."""
        dates1 = pd.date_range('2023-01-01', periods=10, freq='D')
        dates2 = pd.date_range('2023-01-09', periods=10, freq='D')  # Only 2 days overlap

        df1 = pd.DataFrame({
            'open': np.random.uniform(100, 110, 10),
            'high': np.random.uniform(110, 120, 10),
            'low': np.random.uniform(90, 100, 10),
            'close': np.random.uniform(100, 110, 10),
            'volume': np.random.uniform(1e6, 2e6, 10),
            'symbol': 'AAPL'
        }, index=dates1)

        df2 = pd.DataFrame({
            'open': np.random.uniform(200, 210, 10),
            'high': np.random.uniform(210, 220, 10),
            'low': np.random.uniform(190, 200, 10),
            'close': np.random.uniform(200, 210, 10),
            'volume': np.random.uniform(1e6, 2e6, 10),
            'symbol': 'MSFT'
        }, index=dates2)

        combined = pd.concat([df1, df2])
        combined = combined.set_index('symbol', append=True)
        combined = combined.swaplevel()
        combined = combined.sort_index()

        engine = BacktestEngine()
        result = engine._synchronize_symbol_data(combined, ['AAPL', 'MSFT'])

        # Should find 2-day intersection
        assert len(result['AAPL']) == 2
        assert len(result['MSFT']) == 2


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
