"""
Tests for Signal validation and StrategySignals data validation.

Tests the Signal dataclass validation rules and the validate_data
method on StrategySignals base class.
"""

import pytest
from datetime import datetime
from typing import Dict, List

import pandas as pd
import numpy as np

from src.strategies.core.signal import Signal, SignalBatch
from src.strategies.core.base_strategy import StrategySignals


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def valid_signal():
    """Create a valid signal for testing."""
    return Signal(
        timestamp=datetime(2024, 1, 2, 10, 0, 0),
        symbol='AAPL',
        direction='BUY',
        confidence=0.75,
        price=150.50
    )


@pytest.fixture
def valid_ohlcv_data():
    """Create valid OHLCV data for testing."""
    dates = pd.date_range('2024-01-02 09:30:00', periods=100, freq='1min')
    return pd.DataFrame({
        'open': np.random.uniform(100, 105, 100),
        'high': np.random.uniform(105, 110, 100),
        'low': np.random.uniform(95, 100, 100),
        'close': np.random.uniform(100, 105, 100),
        'volume': np.random.uniform(1000, 10000, 100)
    }, index=dates)


@pytest.fixture
def mock_strategy():
    """Create a concrete strategy implementation for testing."""
    class TestStrategy(StrategySignals):
        def __init__(self, lookback: int = 20):
            self._lookback = lookback

        def generate_signals(
            self,
            market_data: Dict[str, pd.DataFrame],
            timestamp: datetime
        ) -> List[Signal]:
            return []

        def get_required_lookback(self) -> int:
            return self._lookback

    return TestStrategy()


# =============================================================================
# SIGNAL VALIDATION TESTS
# =============================================================================

class TestSignalDirectionValidation:
    """Tests for Signal direction validation."""

    def test_valid_buy_direction(self):
        """Test BUY direction is valid."""
        signal = Signal(
            timestamp=datetime.now(),
            symbol='AAPL',
            direction='BUY',
            confidence=0.5,
            price=100.0
        )
        assert signal.direction == 'BUY'

    def test_valid_sell_direction(self):
        """Test SELL direction is valid."""
        signal = Signal(
            timestamp=datetime.now(),
            symbol='AAPL',
            direction='SELL',
            confidence=0.5,
            price=100.0
        )
        assert signal.direction == 'SELL'

    def test_valid_hold_direction(self):
        """Test HOLD direction is valid."""
        signal = Signal(
            timestamp=datetime.now(),
            symbol='AAPL',
            direction='HOLD',
            confidence=0.5,
            price=100.0
        )
        assert signal.direction == 'HOLD'

    def test_invalid_direction_raises_error(self):
        """Test invalid direction raises ValueError."""
        with pytest.raises(ValueError, match="Invalid direction"):
            Signal(
                timestamp=datetime.now(),
                symbol='AAPL',
                direction='INVALID',
                confidence=0.5,
                price=100.0
            )

    def test_lowercase_direction_invalid(self):
        """Test lowercase direction is invalid (must be uppercase)."""
        with pytest.raises(ValueError, match="Invalid direction"):
            Signal(
                timestamp=datetime.now(),
                symbol='AAPL',
                direction='buy',
                confidence=0.5,
                price=100.0
            )

    def test_empty_direction_invalid(self):
        """Test empty direction is invalid."""
        with pytest.raises(ValueError, match="Invalid direction"):
            Signal(
                timestamp=datetime.now(),
                symbol='AAPL',
                direction='',
                confidence=0.5,
                price=100.0
            )


class TestSignalConfidenceValidation:
    """Tests for Signal confidence validation."""

    def test_confidence_zero_is_valid(self):
        """Test 0.0 confidence is valid."""
        signal = Signal(
            timestamp=datetime.now(),
            symbol='AAPL',
            direction='BUY',
            confidence=0.0,
            price=100.0
        )
        assert signal.confidence == 0.0

    def test_confidence_one_is_valid(self):
        """Test 1.0 confidence is valid."""
        signal = Signal(
            timestamp=datetime.now(),
            symbol='AAPL',
            direction='BUY',
            confidence=1.0,
            price=100.0
        )
        assert signal.confidence == 1.0

    def test_confidence_mid_range_is_valid(self):
        """Test mid-range confidence is valid."""
        signal = Signal(
            timestamp=datetime.now(),
            symbol='AAPL',
            direction='BUY',
            confidence=0.75,
            price=100.0
        )
        assert signal.confidence == 0.75

    def test_negative_confidence_raises_error(self):
        """Test negative confidence raises ValueError."""
        with pytest.raises(ValueError, match="Invalid confidence"):
            Signal(
                timestamp=datetime.now(),
                symbol='AAPL',
                direction='BUY',
                confidence=-0.1,
                price=100.0
            )

    def test_confidence_above_one_raises_error(self):
        """Test confidence > 1 raises ValueError."""
        with pytest.raises(ValueError, match="Invalid confidence"):
            Signal(
                timestamp=datetime.now(),
                symbol='AAPL',
                direction='BUY',
                confidence=1.1,
                price=100.0
            )


class TestSignalPriceValidation:
    """Tests for Signal price validation."""

    def test_positive_price_is_valid(self):
        """Test positive price is valid."""
        signal = Signal(
            timestamp=datetime.now(),
            symbol='AAPL',
            direction='BUY',
            confidence=0.5,
            price=100.0
        )
        assert signal.price == 100.0

    def test_small_positive_price_is_valid(self):
        """Test small positive price is valid."""
        signal = Signal(
            timestamp=datetime.now(),
            symbol='PENNY',
            direction='BUY',
            confidence=0.5,
            price=0.01
        )
        assert signal.price == 0.01

    def test_zero_price_raises_error(self):
        """Test zero price raises ValueError."""
        with pytest.raises(ValueError, match="Invalid price"):
            Signal(
                timestamp=datetime.now(),
                symbol='AAPL',
                direction='BUY',
                confidence=0.5,
                price=0.0
            )

    def test_negative_price_raises_error(self):
        """Test negative price raises ValueError."""
        with pytest.raises(ValueError, match="Invalid price"):
            Signal(
                timestamp=datetime.now(),
                symbol='AAPL',
                direction='BUY',
                confidence=0.5,
                price=-50.0
            )


class TestSignalSerialization:
    """Tests for Signal serialization/deserialization."""

    def test_to_dict_returns_dict(self, valid_signal):
        """Test to_dict returns proper dictionary."""
        result = valid_signal.to_dict()

        assert isinstance(result, dict)
        assert 'timestamp' in result
        assert 'symbol' in result
        assert 'direction' in result
        assert 'confidence' in result
        assert 'price' in result
        assert 'metadata' in result

    def test_to_dict_serializes_timestamp(self, valid_signal):
        """Test timestamp is serialized to ISO format."""
        result = valid_signal.to_dict()

        assert isinstance(result['timestamp'], str)
        # Should be parseable back to datetime
        parsed = datetime.fromisoformat(result['timestamp'])
        assert parsed == valid_signal.timestamp

    def test_from_dict_creates_signal(self, valid_signal):
        """Test from_dict creates equivalent Signal."""
        data = valid_signal.to_dict()
        reconstructed = Signal.from_dict(data)

        assert reconstructed.timestamp == valid_signal.timestamp
        assert reconstructed.symbol == valid_signal.symbol
        assert reconstructed.direction == valid_signal.direction
        assert reconstructed.confidence == valid_signal.confidence
        assert reconstructed.price == valid_signal.price
        assert reconstructed.metadata == valid_signal.metadata

    def test_roundtrip_preserves_all_fields(self):
        """Test to_dict -> from_dict roundtrip preserves all data."""
        original = Signal(
            timestamp=datetime(2024, 6, 15, 14, 30, 0),
            symbol='TSLA',
            direction='SELL',
            confidence=0.85,
            price=250.00,
            metadata={'indicator': 'RSI', 'value': 75.5}
        )

        roundtrip = Signal.from_dict(original.to_dict())

        assert roundtrip.timestamp == original.timestamp
        assert roundtrip.symbol == original.symbol
        assert roundtrip.direction == original.direction
        assert roundtrip.confidence == original.confidence
        assert roundtrip.price == original.price
        assert roundtrip.metadata == original.metadata

    def test_from_dict_with_missing_metadata(self):
        """Test from_dict handles missing metadata field."""
        data = {
            'timestamp': datetime.now().isoformat(),
            'symbol': 'AAPL',
            'direction': 'BUY',
            'confidence': 0.5,
            'price': 100.0
            # metadata intentionally omitted
        }

        signal = Signal.from_dict(data)

        assert signal.metadata == {}


# =============================================================================
# SIGNAL BATCH TESTS
# =============================================================================

class TestSignalBatch:
    """Tests for SignalBatch functionality."""

    def test_batch_creation(self):
        """Test batch can be created."""
        batch = SignalBatch(timestamp=datetime.now())

        assert len(batch) == 0
        assert batch.signals == []

    def test_add_signal_with_matching_timestamp(self):
        """Test adding signal with matching timestamp."""
        ts = datetime(2024, 1, 2, 10, 0, 0)
        batch = SignalBatch(timestamp=ts)
        signal = Signal(
            timestamp=ts,
            symbol='AAPL',
            direction='BUY',
            confidence=0.7,
            price=150.0
        )

        batch.add_signal(signal)

        assert len(batch) == 1
        assert batch.signals[0] == signal

    def test_add_signal_with_mismatched_timestamp_raises(self):
        """Test adding signal with wrong timestamp raises error."""
        batch_ts = datetime(2024, 1, 2, 10, 0, 0)
        signal_ts = datetime(2024, 1, 2, 11, 0, 0)  # Different

        batch = SignalBatch(timestamp=batch_ts)
        signal = Signal(
            timestamp=signal_ts,
            symbol='AAPL',
            direction='BUY',
            confidence=0.7,
            price=150.0
        )

        with pytest.raises(ValueError, match="doesn't match batch timestamp"):
            batch.add_signal(signal)

    def test_get_buy_signals(self):
        """Test filtering buy signals."""
        ts = datetime(2024, 1, 2, 10, 0, 0)
        batch = SignalBatch(timestamp=ts)

        batch.signals = [
            Signal(timestamp=ts, symbol='AAPL', direction='BUY', confidence=0.7, price=150.0),
            Signal(timestamp=ts, symbol='MSFT', direction='SELL', confidence=0.6, price=300.0),
            Signal(timestamp=ts, symbol='GOOGL', direction='BUY', confidence=0.8, price=140.0),
        ]

        buy_signals = batch.get_buy_signals()

        assert len(buy_signals) == 2
        assert all(s.direction == 'BUY' for s in buy_signals)

    def test_get_sell_signals(self):
        """Test filtering sell signals."""
        ts = datetime(2024, 1, 2, 10, 0, 0)
        batch = SignalBatch(timestamp=ts)

        batch.signals = [
            Signal(timestamp=ts, symbol='AAPL', direction='BUY', confidence=0.7, price=150.0),
            Signal(timestamp=ts, symbol='MSFT', direction='SELL', confidence=0.6, price=300.0),
            Signal(timestamp=ts, symbol='GOOGL', direction='SELL', confidence=0.8, price=140.0),
        ]

        sell_signals = batch.get_sell_signals()

        assert len(sell_signals) == 2
        assert all(s.direction == 'SELL' for s in sell_signals)

    def test_get_signals_for_symbol(self):
        """Test filtering by symbol."""
        ts = datetime(2024, 1, 2, 10, 0, 0)
        batch = SignalBatch(timestamp=ts)

        batch.signals = [
            Signal(timestamp=ts, symbol='AAPL', direction='BUY', confidence=0.7, price=150.0),
            Signal(timestamp=ts, symbol='MSFT', direction='SELL', confidence=0.6, price=300.0),
            Signal(timestamp=ts, symbol='AAPL', direction='SELL', confidence=0.5, price=145.0),
        ]

        aapl_signals = batch.get_signals_for_symbol('AAPL')

        assert len(aapl_signals) == 2
        assert all(s.symbol == 'AAPL' for s in aapl_signals)

    def test_batch_iteration(self):
        """Test iterating over batch signals."""
        ts = datetime(2024, 1, 2, 10, 0, 0)
        batch = SignalBatch(timestamp=ts)

        batch.signals = [
            Signal(timestamp=ts, symbol='AAPL', direction='BUY', confidence=0.7, price=150.0),
            Signal(timestamp=ts, symbol='MSFT', direction='SELL', confidence=0.6, price=300.0),
        ]

        signals_list = list(batch)

        assert len(signals_list) == 2


# =============================================================================
# STRATEGY DATA VALIDATION TESTS
# =============================================================================

class TestStrategyDataValidation:
    """Tests for StrategySignals.validate_data method."""

    def test_valid_data_passes(self, mock_strategy, valid_ohlcv_data):
        """Test valid OHLCV data passes validation."""
        is_valid, error = mock_strategy.validate_data(valid_ohlcv_data)

        assert is_valid == True
        assert error == ""

    def test_missing_columns_fails(self, mock_strategy):
        """Test missing required columns fails validation."""
        # Missing 'volume' column
        dates = pd.date_range('2024-01-02', periods=50, freq='1min')
        data = pd.DataFrame({
            'open': [100] * 50,
            'high': [105] * 50,
            'low': [95] * 50,
            'close': [102] * 50
            # volume missing
        }, index=dates)

        is_valid, error = mock_strategy.validate_data(data)

        assert is_valid == False
        assert "Missing columns" in error
        assert "volume" in error

    def test_non_datetime_index_fails(self, mock_strategy):
        """Test non-DatetimeIndex fails validation."""
        data = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [105, 106, 107],
            'low': [95, 96, 97],
            'close': [102, 103, 104],
            'volume': [1000, 1100, 1200]
        }, index=[0, 1, 2])  # Integer index instead of DatetimeIndex

        is_valid, error = mock_strategy.validate_data(data)

        assert is_valid == False
        assert "DatetimeIndex" in error

    def test_insufficient_data_length_fails(self, mock_strategy):
        """Test insufficient data length fails validation."""
        # Strategy requires 20 periods but only 10 provided
        dates = pd.date_range('2024-01-02', periods=10, freq='1min')
        data = pd.DataFrame({
            'open': [100] * 10,
            'high': [105] * 10,
            'low': [95] * 10,
            'close': [102] * 10,
            'volume': [1000] * 10
        }, index=dates)

        is_valid, error = mock_strategy.validate_data(data)

        assert is_valid == False
        assert "Insufficient data" in error

    def test_nan_values_fails(self, mock_strategy):
        """Test NaN values in required columns fails validation."""
        dates = pd.date_range('2024-01-02', periods=50, freq='1min')
        data = pd.DataFrame({
            'open': [100] * 50,
            'high': [105] * 50,
            'low': [95] * 50,
            'close': [102] * 48 + [np.nan, np.nan],  # 2 NaN values
            'volume': [1000] * 50
        }, index=dates)

        is_valid, error = mock_strategy.validate_data(data)

        assert is_valid == False
        assert "NaN values" in error

    def test_zero_price_fails(self, mock_strategy):
        """Test zero price fails validation."""
        dates = pd.date_range('2024-01-02', periods=50, freq='1min')
        data = pd.DataFrame({
            'open': [100] * 49 + [0],  # One zero
            'high': [105] * 50,
            'low': [95] * 50,
            'close': [102] * 50,
            'volume': [1000] * 50
        }, index=dates)

        is_valid, error = mock_strategy.validate_data(data)

        assert is_valid == False
        assert "Non-positive" in error

    def test_negative_price_fails(self, mock_strategy):
        """Test negative price fails validation."""
        dates = pd.date_range('2024-01-02', periods=50, freq='1min')
        data = pd.DataFrame({
            'open': [100] * 50,
            'high': [105] * 50,
            'low': [95] * 49 + [-5],  # One negative
            'close': [102] * 50,
            'volume': [1000] * 50
        }, index=dates)

        is_valid, error = mock_strategy.validate_data(data)

        assert is_valid == False
        assert "Non-positive" in error

    def test_high_less_than_low_fails(self, mock_strategy):
        """Test high < low fails validation."""
        dates = pd.date_range('2024-01-02', periods=50, freq='1min')
        data = pd.DataFrame({
            'open': [100] * 50,
            'high': [95] * 50,  # High is LOWER than low
            'low': [100] * 50,
            'close': [97] * 50,
            'volume': [1000] * 50
        }, index=dates)

        is_valid, error = mock_strategy.validate_data(data)

        assert is_valid == False
        assert "high < low" in error

    def test_close_outside_high_low_fails(self, mock_strategy):
        """Test close outside [low, high] fails validation."""
        dates = pd.date_range('2024-01-02', periods=50, freq='1min')
        data = pd.DataFrame({
            'open': [100] * 50,
            'high': [105] * 50,
            'low': [95] * 50,
            'close': [110] * 50,  # Close > high
            'volume': [1000] * 50
        }, index=dates)

        is_valid, error = mock_strategy.validate_data(data)

        assert is_valid == False
        assert "close outside" in error

    def test_open_outside_high_low_fails(self, mock_strategy):
        """Test open outside [low, high] fails validation."""
        dates = pd.date_range('2024-01-02', periods=50, freq='1min')
        data = pd.DataFrame({
            'open': [90] * 50,  # Open < low
            'high': [105] * 50,
            'low': [95] * 50,
            'close': [100] * 50,
            'volume': [1000] * 50
        }, index=dates)

        is_valid, error = mock_strategy.validate_data(data)

        assert is_valid == False
        assert "open outside" in error

    def test_symbol_included_in_error_message(self, mock_strategy):
        """Test symbol name is included in error messages."""
        dates = pd.date_range('2024-01-02', periods=10, freq='1min')
        data = pd.DataFrame({
            'open': [100] * 10,
            'high': [105] * 10,
            'low': [95] * 10,
            'close': [102] * 10,
            'volume': [1000] * 10
        }, index=dates)

        is_valid, error = mock_strategy.validate_data(data, symbol='AAPL')

        assert is_valid == False
        assert "AAPL" in error


class TestValidateMarketData:
    """Tests for validate_market_data method."""

    def test_all_valid_data(self, mock_strategy, valid_ohlcv_data):
        """Test all valid data returns no errors."""
        market_data = {
            'AAPL': valid_ohlcv_data.copy(),
            'MSFT': valid_ohlcv_data.copy()
        }

        is_valid, errors = mock_strategy.validate_market_data(market_data)

        assert is_valid == True
        assert len(errors) == 0

    def test_one_invalid_returns_error(self, mock_strategy, valid_ohlcv_data):
        """Test one invalid symbol returns its error."""
        invalid_data = valid_ohlcv_data.copy()
        invalid_data.loc[invalid_data.index[0], 'close'] = np.nan

        market_data = {
            'AAPL': valid_ohlcv_data.copy(),  # Valid
            'MSFT': invalid_data  # Invalid
        }

        is_valid, errors = mock_strategy.validate_market_data(market_data)

        assert is_valid == False
        assert len(errors) == 1
        assert "MSFT" in errors[0]

    def test_multiple_invalid_returns_all_errors(self, mock_strategy):
        """Test multiple invalid symbols return all errors."""
        dates = pd.date_range('2024-01-02', periods=5, freq='1min')
        invalid_data1 = pd.DataFrame({
            'open': [100] * 5,
            'high': [105] * 5,
            'low': [95] * 5,
            'close': [102] * 5,
            'volume': [1000] * 5
        }, index=dates)  # Too short

        invalid_data2 = invalid_data1.copy()

        market_data = {
            'AAPL': invalid_data1,
            'MSFT': invalid_data2
        }

        is_valid, errors = mock_strategy.validate_market_data(market_data)

        assert is_valid == False
        assert len(errors) == 2


# =============================================================================
# STRATEGY UTILITY METHOD TESTS
# =============================================================================

class TestStrategyUtilityMethods:
    """Tests for strategy utility methods."""

    def test_get_name_returns_class_name(self, mock_strategy):
        """Test get_name returns class name by default."""
        name = mock_strategy.get_name()

        assert name == 'TestStrategy'

    def test_get_parameters_returns_empty_dict_by_default(self, mock_strategy):
        """Test get_parameters returns empty dict by default."""
        params = mock_strategy.get_parameters()

        assert params == {}

    def test_str_representation(self, mock_strategy):
        """Test string representation includes name and params."""
        result = str(mock_strategy)

        assert 'TestStrategy' in result

    def test_repr_representation(self, mock_strategy):
        """Test repr representation includes class name."""
        result = repr(mock_strategy)

        assert 'TestStrategy' in result


# =============================================================================
# DATA REQUIREMENTS TESTS
# =============================================================================

class TestDataRequirements:
    """Tests for DataRequirements class."""

    def test_add_daily_data(self):
        """Test adding daily data requirement."""
        from src.strategies.core.base_strategy import DataRequirements

        reqs = DataRequirements()
        reqs.add_daily_data('AAPL', 200)

        assert len(reqs.daily_data) == 1
        assert reqs.daily_data[0] == ('AAPL', 200)

    def test_add_intraday_data(self):
        """Test adding intraday data requirement."""
        from src.strategies.core.base_strategy import DataRequirements

        reqs = DataRequirements()
        reqs.add_intraday_data('AAPL', '1Min', 100)

        assert len(reqs.intraday_data) == 1
        assert reqs.intraday_data[0] == ('AAPL', '1Min', 100)

    def test_get_all_symbols_returns_unique(self):
        """Test get_all_symbols returns unique symbol list."""
        from src.strategies.core.base_strategy import DataRequirements

        reqs = DataRequirements()
        reqs.add_daily_data('AAPL', 200)
        reqs.add_daily_data('AAPL', 50)  # Duplicate
        reqs.add_intraday_data('AAPL', '1Min', 100)  # Same symbol
        reqs.add_intraday_data('MSFT', '5Min', 50)

        symbols = reqs.get_all_symbols()

        assert 'AAPL' in symbols
        assert 'MSFT' in symbols
        assert len(symbols) == 2  # Unique only
