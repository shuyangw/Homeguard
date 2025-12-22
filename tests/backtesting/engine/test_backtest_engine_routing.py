"""
Tests for BacktestEngine routing logic.

Tests the correct routing of backtests based on strategy type,
portfolio mode, and execution mode.
"""

import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from datetime import datetime
import pandas as pd
import numpy as np

from src.backtesting.engine.backtest_engine import BacktestEngine
from src.backtesting.base.strategy import BaseStrategy, MultiSymbolStrategy, LongShortStrategy
from src.backtesting.base.pairs_strategy import PairsStrategy
from src.backtesting.utils.risk_config import RiskConfig


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def engine():
    """Create a BacktestEngine with default settings."""
    return BacktestEngine(
        initial_capital=100000,
        fees=0.001,
        slippage=0.0,
        timeframe='1min'
    )


@pytest.fixture
def simple_ohlcv_data():
    """Create simple OHLCV data for testing."""
    dates = pd.date_range(
        start='2024-01-02 09:35:00',
        periods=100,
        freq='1min',
        tz='US/Eastern'
    )
    return pd.DataFrame({
        'open': np.random.uniform(100, 110, 100),
        'high': np.random.uniform(105, 115, 100),
        'low': np.random.uniform(95, 105, 100),
        'close': np.random.uniform(100, 110, 100),
        'volume': np.random.uniform(1000, 10000, 100)
    }, index=dates)


@pytest.fixture
def mock_polars_df():
    """Create a mock Polars DataFrame."""
    mock_df = MagicMock()
    mock_df.is_empty.return_value = False

    dates = pd.date_range(
        start='2024-01-02 09:35:00',
        periods=100,
        freq='1min',
        tz='US/Eastern'
    )
    pandas_df = pd.DataFrame({
        'timestamp': dates,
        'open': np.random.uniform(100, 110, 100),
        'high': np.random.uniform(105, 115, 100),
        'low': np.random.uniform(95, 105, 100),
        'close': np.random.uniform(100, 110, 100),
        'volume': np.random.uniform(1000, 10000, 100)
    })
    mock_df.to_pandas.return_value = pandas_df
    return mock_df


# =============================================================================
# STRATEGY TEST CLASSES
# =============================================================================

class MockBaseStrategy(BaseStrategy):
    """Mock single-symbol strategy for testing."""

    def generate_signals(self, data):
        entries = pd.Series(False, index=data.index)
        exits = pd.Series(False, index=data.index)
        # Add a few signals
        if len(data) > 10:
            entries.iloc[5] = True
            exits.iloc[15] = True
        return entries, exits


class MockLongShortStrategy(LongShortStrategy):
    """Mock long-short strategy for testing."""

    def generate_long_short_signals(self, data):
        """Implementation of abstract method."""
        long_entries = pd.Series(False, index=data.index)
        long_exits = pd.Series(False, index=data.index)
        short_entries = pd.Series(False, index=data.index)
        short_exits = pd.Series(False, index=data.index)
        if len(data) > 20:
            long_entries.iloc[5] = True
            long_exits.iloc[15] = True
            short_entries.iloc[25] = True
            short_exits.iloc[35] = True
        return long_entries, long_exits, short_entries, short_exits


class MockMultiSymbolStrategy(MultiSymbolStrategy):
    """Mock multi-symbol strategy for testing."""

    def get_required_symbols(self):
        return 2

    def generate_multi_signals(self, data_dict):
        signals = {}
        for symbol, data in data_dict.items():
            entries = pd.Series(False, index=data.index)
            exits = pd.Series(False, index=data.index)
            if len(data) > 10:
                entries.iloc[5] = True
                exits.iloc[15] = True
            signals[symbol] = (entries, exits)
        return signals


class MockPairsStrategy(PairsStrategy):
    """Mock pairs strategy for testing."""

    def get_required_symbols(self):
        return 2

    def generate_pairs_signals(self, data1, data2, symbol1, symbol2):
        """Implementation of abstract method."""
        long_entries = pd.Series(False, index=data1.index)
        long_exits = pd.Series(False, index=data1.index)
        short_entries = pd.Series(False, index=data1.index)
        short_exits = pd.Series(False, index=data1.index)
        if len(data1) > 10:
            long_entries.iloc[5] = True
            short_entries.iloc[10] = True
        return long_entries, long_exits, short_entries, short_exits


# =============================================================================
# INITIALIZATION TESTS
# =============================================================================

class TestBacktestEngineInit:
    """Tests for BacktestEngine initialization."""

    def test_init_with_defaults(self):
        """Test engine initializes with default values."""
        engine = BacktestEngine()

        assert engine.initial_capital == 100000.0
        assert engine.fees == 0.001
        assert engine.slippage == 0.0
        assert engine.timeframe == '1min'

    def test_init_with_custom_values(self):
        """Test engine initializes with custom values."""
        engine = BacktestEngine(
            initial_capital=50000,
            fees=0.002,
            slippage=0.001,
            timeframe='1hour'
        )

        assert engine.initial_capital == 50000
        assert engine.fees == 0.002
        assert engine.slippage == 0.001
        assert engine.timeframe == '1hour'

    def test_init_rejects_negative_capital(self):
        """Test engine rejects negative initial capital."""
        with pytest.raises(ValueError, match="initial_capital must be positive"):
            BacktestEngine(initial_capital=-1000)

    def test_init_rejects_zero_capital(self):
        """Test engine rejects zero initial capital."""
        with pytest.raises(ValueError, match="initial_capital must be positive"):
            BacktestEngine(initial_capital=0)

    def test_init_rejects_negative_fees(self):
        """Test engine rejects negative fees."""
        with pytest.raises(ValueError, match="fees must be non-negative"):
            BacktestEngine(fees=-0.001)

    def test_init_rejects_negative_slippage(self):
        """Test engine rejects negative slippage."""
        with pytest.raises(ValueError, match="slippage must be non-negative"):
            BacktestEngine(slippage=-0.001)

    def test_init_crypto_timeframe_disables_market_filter(self):
        """Test crypto timeframe disables market day filtering."""
        engine = BacktestEngine(timeframe='crypto_1hour')

        assert engine.fractional_shares == True

    def test_init_creates_risk_config_if_none(self):
        """Test engine creates default risk config if none provided."""
        engine = BacktestEngine()

        assert engine.risk_config is not None
        assert isinstance(engine.risk_config, RiskConfig)


# =============================================================================
# STRATEGY TYPE DETECTION TESTS
# =============================================================================

class TestStrategyTypeDetection:
    """Tests for strategy type detection."""

    def test_detect_base_strategy(self, engine):
        """Test detection of base strategy (single-symbol)."""
        strategy = MockBaseStrategy()

        is_multi = engine._detect_strategy_type(strategy)

        assert is_multi == False

    def test_detect_long_short_strategy(self, engine):
        """Test detection of long-short strategy (still single-symbol)."""
        strategy = MockLongShortStrategy()

        # LongShortStrategy inherits from BaseStrategy, not MultiSymbolStrategy
        is_multi = engine._detect_strategy_type(strategy)

        assert is_multi == False

    def test_detect_multi_symbol_strategy(self, engine):
        """Test detection of multi-symbol strategy."""
        strategy = MockMultiSymbolStrategy()

        is_multi = engine._detect_strategy_type(strategy)

        assert is_multi == True

    def test_detect_pairs_strategy(self, engine):
        """Test detection of pairs strategy (multi-symbol)."""
        strategy = MockPairsStrategy()

        is_multi = engine._detect_strategy_type(strategy)

        assert is_multi == True


# =============================================================================
# ROUTING LOGIC TESTS
# =============================================================================

class TestRoutingLogic:
    """Tests for backtest routing based on strategy and mode."""

    def test_routes_to_single_symbol_for_one_symbol(self, engine, mock_polars_df):
        """Test routing to _run_single_symbol for single symbol."""
        strategy = MockBaseStrategy()

        with patch.object(engine.data_loader, 'load_symbol', return_value=mock_polars_df):
            with patch.object(engine, '_run_single_symbol', wraps=engine._run_single_symbol) as mock_method:
                with patch.object(engine, '_print_summary'):
                    try:
                        engine.run(
                            strategy=strategy,
                            symbols='AAPL',
                            start_date='2024-01-01',
                            end_date='2024-01-31',
                            portfolio_mode='single'
                        )
                    except Exception:
                        pass  # Ignore downstream errors

                    # Verify routing method was called
                    assert mock_method.called

    def test_routes_to_multiple_symbols_for_list(self, engine, mock_polars_df):
        """Test routing to _run_multiple_symbols for multiple symbols in single mode."""
        strategy = MockBaseStrategy()

        with patch.object(engine.data_loader, 'load_symbol', return_value=mock_polars_df):
            with patch.object(engine, '_run_multiple_symbols', wraps=engine._run_multiple_symbols) as mock_method:
                with patch.object(engine, '_print_summary'):
                    try:
                        engine.run(
                            strategy=strategy,
                            symbols=['AAPL', 'MSFT'],
                            start_date='2024-01-01',
                            end_date='2024-01-31',
                            portfolio_mode='single'
                        )
                    except Exception:
                        pass

                    assert mock_method.called

    def test_routes_to_multi_asset_portfolio(self, engine, mock_polars_df):
        """Test routing to _run_multi_asset_portfolio for multi mode."""
        strategy = MockBaseStrategy()

        with patch.object(engine.data_loader, 'load_symbols_synchronized') as mock_load:
            mock_load.return_value = {'AAPL': mock_polars_df, 'MSFT': mock_polars_df}
            with patch.object(engine, '_run_multi_asset_portfolio', wraps=engine._run_multi_asset_portfolio) as mock_method:
                with patch.object(engine, '_print_summary'):
                    try:
                        engine.run(
                            strategy=strategy,
                            symbols=['AAPL', 'MSFT'],
                            start_date='2024-01-01',
                            end_date='2024-01-31',
                            portfolio_mode='multi'
                        )
                    except Exception:
                        pass

                    assert mock_method.called

    def test_routes_to_multi_symbol_strategy(self, engine, mock_polars_df):
        """Test routing to _run_multi_symbol_strategy for multi-symbol strategies."""
        strategy = MockMultiSymbolStrategy()

        with patch.object(engine.data_loader, 'load_symbols_synchronized') as mock_load:
            mock_load.return_value = {'AAPL': mock_polars_df, 'MSFT': mock_polars_df}
            with patch.object(engine, '_run_multi_symbol_strategy', wraps=engine._run_multi_symbol_strategy) as mock_method:
                with patch.object(engine, '_print_summary'):
                    try:
                        engine.run(
                            strategy=strategy,
                            symbols=['AAPL', 'MSFT'],
                            start_date='2024-01-01',
                            end_date='2024-01-31'
                        )
                    except Exception:
                        pass

                    assert mock_method.called

    def test_routes_to_rolling_mode(self, engine, mock_polars_df):
        """Test routing to _run_rolling for rolling mode."""
        strategy = MockBaseStrategy()

        with patch.object(engine, '_run_rolling', return_value=MagicMock()) as mock_method:
            engine.run(
                strategy=strategy,
                symbols='AAPL',
                start_date='2024-01-01',
                end_date='2024-06-30',
                mode='rolling',
                window_days=60
            )

            assert mock_method.called

    def test_invalid_portfolio_mode_raises_error(self, engine, mock_polars_df):
        """Test invalid portfolio mode raises ValueError."""
        strategy = MockBaseStrategy()

        with patch.object(engine.data_loader, 'load_symbol', return_value=mock_polars_df):
            with pytest.raises(ValueError, match="Invalid portfolio_mode"):
                engine.run(
                    strategy=strategy,
                    symbols=['AAPL', 'MSFT'],
                    start_date='2024-01-01',
                    end_date='2024-01-31',
                    portfolio_mode='invalid_mode'
                )


# =============================================================================
# MULTI-SYMBOL VALIDATION TESTS
# =============================================================================

class TestMultiSymbolValidation:
    """Tests for multi-symbol strategy validation."""

    def test_multi_symbol_requires_at_least_two_symbols(self, engine):
        """Test multi-symbol strategy requires at least 2 symbols."""
        strategy = MockMultiSymbolStrategy()

        with pytest.raises(ValueError, match="requires at least 2 symbols"):
            engine.run(
                strategy=strategy,
                symbols=['AAPL'],  # Only 1 symbol
                start_date='2024-01-01',
                end_date='2024-01-31'
            )

    def test_multi_symbol_validates_exact_count(self, engine):
        """Test multi-symbol strategy validates exact symbol count."""
        strategy = MockMultiSymbolStrategy()  # Returns 2 as required

        with pytest.raises(ValueError, match="requires exactly 2 symbols"):
            engine.run(
                strategy=strategy,
                symbols=['AAPL', 'MSFT', 'GOOGL'],  # 3 symbols, expects 2
                start_date='2024-01-01',
                end_date='2024-01-31'
            )


# =============================================================================
# DATA LOADING TESTS
# =============================================================================

class TestDataLoading:
    """Tests for data loading and format conversion."""

    def test_load_data_converts_polars_to_pandas(self, engine, mock_polars_df):
        """Test data loading converts Polars to pandas correctly."""
        def mock_iter_symbols(symbols, start, end):
            for symbol in symbols:
                yield symbol, mock_polars_df

        with patch.object(engine.data_loader, 'iter_symbols', side_effect=mock_iter_symbols):
            result = engine._load_data_as_pandas(
                symbols=['AAPL'],
                start_date='2024-01-01',
                end_date='2024-01-31'
            )

            assert isinstance(result, pd.DataFrame)
            assert 'symbol' in result.index.names
            assert 'timestamp' in result.index.names

    def test_load_data_raises_on_no_data(self, engine):
        """Test data loading raises ValueError when no data found."""
        def mock_iter_symbols(symbols, start, end):
            return iter([])  # Empty iterator

        with patch.object(engine.data_loader, 'iter_symbols', side_effect=mock_iter_symbols):
            with pytest.raises(ValueError, match="No data found"):
                engine._load_data_as_pandas(
                    symbols=['INVALID'],
                    start_date='2024-01-01',
                    end_date='2024-01-31'
                )

    def test_load_data_creates_multi_index(self, engine, mock_polars_df):
        """Test data loading creates proper MultiIndex structure."""
        def mock_iter_symbols(symbols, start, end):
            for symbol in symbols:
                yield symbol, mock_polars_df

        with patch.object(engine.data_loader, 'iter_symbols', side_effect=mock_iter_symbols):
            result = engine._load_data_as_pandas(
                symbols=['AAPL', 'MSFT'],
                start_date='2024-01-01',
                end_date='2024-01-31'
            )

            assert isinstance(result.index, pd.MultiIndex)
            assert result.index.nlevels == 2


# =============================================================================
# SYNCHRONIZATION TESTS
# =============================================================================

class TestDataSynchronization:
    """Tests for multi-symbol data synchronization."""

    def test_synchronize_symbol_data_aligns_timestamps(self, engine):
        """Test symbol data synchronization aligns timestamps."""
        # Create data with slightly different timestamps
        dates1 = pd.date_range('2024-01-02 09:30:00', periods=10, freq='1min')
        dates2 = pd.date_range('2024-01-02 09:32:00', periods=10, freq='1min')

        data1 = pd.DataFrame({'close': np.arange(10)}, index=dates1)
        data2 = pd.DataFrame({'close': np.arange(10) + 10}, index=dates2)

        # Create MultiIndex DataFrame
        data1['symbol'] = 'AAPL'
        data2['symbol'] = 'MSFT'
        combined = pd.concat([data1, data2])
        combined = combined.reset_index().rename(columns={'index': 'timestamp'})
        combined = combined.set_index(['symbol', 'timestamp']).sort_index()

        result = engine._synchronize_symbol_data(combined, ['AAPL', 'MSFT'])

        assert 'AAPL' in result
        assert 'MSFT' in result
        # Common timestamps should be 09:32-09:39 (8 minutes)
        assert len(result['AAPL']) == 8
        assert len(result['MSFT']) == 8
        assert result['AAPL'].index.equals(result['MSFT'].index)

    def test_synchronize_symbol_data_raises_on_missing_symbol(self, engine):
        """Test synchronization raises error for missing symbol."""
        dates = pd.date_range('2024-01-02 09:30:00', periods=10, freq='1min')
        data = pd.DataFrame({'close': np.arange(10)}, index=dates)
        data['symbol'] = 'AAPL'
        data = data.reset_index().rename(columns={'index': 'timestamp'})
        data = data.set_index(['symbol', 'timestamp'])

        with pytest.raises(ValueError, match="Symbol MSFT not found"):
            engine._synchronize_symbol_data(data, ['AAPL', 'MSFT'])

    def test_synchronize_symbol_data_raises_on_no_overlap(self, engine):
        """Test synchronization raises error when no timestamp overlap."""
        dates1 = pd.date_range('2024-01-02 09:30:00', periods=5, freq='1min')
        dates2 = pd.date_range('2024-01-02 10:00:00', periods=5, freq='1min')  # No overlap

        data1 = pd.DataFrame({'close': np.arange(5)}, index=dates1)
        data2 = pd.DataFrame({'close': np.arange(5) + 10}, index=dates2)

        data1['symbol'] = 'AAPL'
        data2['symbol'] = 'MSFT'
        combined = pd.concat([data1, data2])
        combined = combined.reset_index().rename(columns={'index': 'timestamp'})
        combined = combined.set_index(['symbol', 'timestamp']).sort_index()

        with pytest.raises(ValueError, match="No overlapping timestamps"):
            engine._synchronize_symbol_data(combined, ['AAPL', 'MSFT'])


# =============================================================================
# RUN WITH DATA TESTS
# =============================================================================

class TestRunWithData:
    """Tests for run_with_data method."""

    def test_run_with_data_accepts_dataframe(self, engine, simple_ohlcv_data):
        """Test run_with_data accepts pre-loaded DataFrame."""
        strategy = MockBaseStrategy()

        with patch.object(engine, '_print_summary'):
            portfolio = engine.run_with_data(
                strategy=strategy,
                data=simple_ohlcv_data,
                price_type='close'
            )

            assert portfolio is not None

    def test_run_with_data_removes_duplicate_timestamps(self, engine):
        """Test run_with_data handles duplicate timestamps."""
        # Create data with duplicates
        dates = pd.date_range(
            start='2024-01-02 09:35:00',
            periods=50,
            freq='1min',
            tz='US/Eastern'
        )
        duplicated_dates = dates.append(dates[:10])  # Add 10 duplicates

        data = pd.DataFrame({
            'open': np.random.uniform(100, 110, 60),
            'high': np.random.uniform(105, 115, 60),
            'low': np.random.uniform(95, 105, 60),
            'close': np.random.uniform(100, 110, 60),
            'volume': np.random.uniform(1000, 10000, 60)
        }, index=duplicated_dates)

        strategy = MockBaseStrategy()

        with patch.object(engine, '_print_summary'):
            portfolio = engine.run_with_data(
                strategy=strategy,
                data=data,
                price_type='close'
            )

            # Should still work after removing duplicates
            assert portfolio is not None


# =============================================================================
# ROLLING WINDOW TESTS
# =============================================================================

class TestRollingMode:
    """Tests for rolling window backtesting."""

    def test_generate_windows_creates_correct_count(self, engine):
        """Test window generation creates correct number of windows."""
        with patch('src.backtesting.engine.backtest_engine.MarketCalendar') as mock_calendar:
            # Mock 120 trading days
            trading_days = pd.date_range('2024-01-02', periods=120, freq='B')
            mock_calendar.return_value.get_trading_days.return_value = trading_days

            windows = engine._generate_windows(
                start_date='2024-01-02',
                end_date='2024-06-30',
                window_days=60,
                step_days=30
            )

            # With 120 days, window=60, step=30: should get 3 windows
            # Window 1: days 0-59, Window 2: days 30-89, Window 3: days 60-119
            assert len(windows) == 3

    def test_generate_windows_returns_empty_for_insufficient_data(self, engine):
        """Test window generation handles insufficient data."""
        with patch('src.backtesting.engine.backtest_engine.MarketCalendar') as mock_calendar:
            # Only 30 trading days, but window is 60
            trading_days = pd.date_range('2024-01-02', periods=30, freq='B')
            mock_calendar.return_value.get_trading_days.return_value = trading_days

            windows = engine._generate_windows(
                start_date='2024-01-02',
                end_date='2024-02-15',
                window_days=60,
                step_days=30
            )

            # Should return single window with all available data
            assert len(windows) == 1


# =============================================================================
# PAIRS STRATEGY ROUTING TESTS
# =============================================================================

class TestPairsStrategyRouting:
    """Tests for pairs strategy specific routing."""

    def test_pairs_strategy_routes_to_multi_symbol_handler(self, engine, mock_polars_df):
        """Test pairs strategy routes to _run_multi_symbol_strategy."""
        strategy = MockPairsStrategy()

        with patch.object(engine.data_loader, 'load_symbols_synchronized') as mock_load:
            mock_load.return_value = {'AAPL': mock_polars_df, 'MSFT': mock_polars_df}
            with patch.object(engine, '_run_multi_symbol_strategy', wraps=engine._run_multi_symbol_strategy) as mock_method:
                with patch.object(engine, '_print_summary'):
                    try:
                        engine.run(
                            strategy=strategy,
                            symbols=['AAPL', 'MSFT'],
                            start_date='2024-01-01',
                            end_date='2024-01-31'
                        )
                    except Exception:
                        pass

                    # Should route to multi-symbol handler
                    assert mock_method.called


# =============================================================================
# STRING TO LIST CONVERSION TESTS
# =============================================================================

class TestSymbolListConversion:
    """Tests for symbol string to list conversion."""

    def test_single_symbol_string_converted_to_list(self, engine, mock_polars_df):
        """Test single symbol string is converted to list."""
        strategy = MockBaseStrategy()

        with patch.object(engine.data_loader, 'load_symbol', return_value=mock_polars_df):
            with patch.object(engine, '_run_single_symbol') as mock_method:
                mock_method.return_value = MagicMock()
                with patch.object(engine, '_print_summary'):
                    engine.run(
                        strategy=strategy,
                        symbols='AAPL',  # String, not list
                        start_date='2024-01-01',
                        end_date='2024-01-31'
                    )

                    # Should be called with 'AAPL' as the symbol argument
                    assert mock_method.call_args[0][1] == 'AAPL'
