"""
Test Phase 1: Core Abstractions

Tests for the newly created core strategy abstractions:
- Signal and SignalBatch
- StrategySignals and DataRequirements
- Universe management classes
"""

import pytest
from datetime import datetime
import pandas as pd
import numpy as np

from src.strategies.core import Signal, SignalBatch, StrategySignals, DataRequirements
from src.strategies.universe import ETFUniverse, EquityUniverse


class TestSignal:
    """Test Signal data structure."""

    def test_valid_signal_creation(self):
        """Test creating a valid signal."""
        signal = Signal(
            timestamp=datetime(2025, 11, 12, 15, 50),
            symbol='AAPL',
            direction='BUY',
            confidence=0.85,
            price=150.25,
            metadata={'strategy': 'MA_Crossover'}
        )

        assert signal.symbol == 'AAPL'
        assert signal.direction == 'BUY'
        assert signal.confidence == 0.85
        assert signal.price == 150.25
        assert signal.metadata['strategy'] == 'MA_Crossover'

    def test_invalid_direction(self):
        """Test that invalid direction raises error."""
        with pytest.raises(ValueError, match="Invalid direction"):
            Signal(
                timestamp=datetime.now(),
                symbol='AAPL',
                direction='INVALID',  # Invalid
                confidence=0.85,
                price=150.25
            )

    def test_invalid_confidence(self):
        """Test that confidence outside [0, 1] raises error."""
        with pytest.raises(ValueError, match="Invalid confidence"):
            Signal(
                timestamp=datetime.now(),
                symbol='AAPL',
                direction='BUY',
                confidence=1.5,  # > 1.0
                price=150.25
            )

        with pytest.raises(ValueError, match="Invalid confidence"):
            Signal(
                timestamp=datetime.now(),
                symbol='AAPL',
                direction='BUY',
                confidence=-0.1,  # < 0.0
                price=150.25
            )

    def test_invalid_price(self):
        """Test that non-positive price raises error."""
        with pytest.raises(ValueError, match="Invalid price"):
            Signal(
                timestamp=datetime.now(),
                symbol='AAPL',
                direction='BUY',
                confidence=0.85,
                price=-10.0  # Negative
            )

    def test_signal_string_representation(self):
        """Test signal string formatting."""
        signal = Signal(
            timestamp=datetime(2025, 11, 12, 15, 50),
            symbol='TQQQ',
            direction='SELL',
            confidence=0.75,
            price=110.50
        )

        str_repr = str(signal)
        assert 'TQQQ' in str_repr
        assert 'SELL' in str_repr
        assert '110.50' in str_repr
        assert '75' in str_repr  # 75% confidence

    def test_signal_serialization(self):
        """Test signal to_dict and from_dict."""
        original = Signal(
            timestamp=datetime(2025, 11, 12, 15, 50),
            symbol='NVDA',
            direction='BUY',
            confidence=0.90,
            price=500.75,
            metadata={'indicator': 'RSI', 'value': 35}
        )

        # Serialize
        data = original.to_dict()
        assert data['symbol'] == 'NVDA'
        assert data['direction'] == 'BUY'
        assert data['confidence'] == 0.90

        # Deserialize
        reconstructed = Signal.from_dict(data)
        assert reconstructed.symbol == original.symbol
        assert reconstructed.direction == original.direction
        assert reconstructed.confidence == original.confidence
        assert reconstructed.price == original.price
        assert reconstructed.metadata == original.metadata


class TestSignalBatch:
    """Test SignalBatch."""

    def test_signal_batch_creation(self):
        """Test creating a batch of signals."""
        timestamp = datetime(2025, 11, 12, 15, 50)
        batch = SignalBatch(timestamp=timestamp)

        signal1 = Signal(timestamp, 'AAPL', 'BUY', 0.85, 150.25)
        signal2 = Signal(timestamp, 'MSFT', 'SELL', 0.70, 420.50)

        batch.add_signal(signal1)
        batch.add_signal(signal2)

        assert len(batch) == 2
        assert len(batch.get_buy_signals()) == 1
        assert len(batch.get_sell_signals()) == 1

    def test_batch_timestamp_validation(self):
        """Test that signals must match batch timestamp."""
        batch_time = datetime(2025, 11, 12, 15, 50)
        signal_time = datetime(2025, 11, 12, 16, 00)  # Different

        batch = SignalBatch(timestamp=batch_time)
        signal = Signal(signal_time, 'AAPL', 'BUY', 0.85, 150.25)

        with pytest.raises(ValueError, match="doesn't match batch timestamp"):
            batch.add_signal(signal)

    def test_batch_filtering(self):
        """Test filtering signals by direction."""
        timestamp = datetime.now()
        batch = SignalBatch(timestamp=timestamp)

        batch.add_signal(Signal(timestamp, 'AAPL', 'BUY', 0.85, 150.25))
        batch.add_signal(Signal(timestamp, 'MSFT', 'BUY', 0.80, 420.50))
        batch.add_signal(Signal(timestamp, 'GOOGL', 'SELL', 0.75, 140.30))
        batch.add_signal(Signal(timestamp, 'NVDA', 'HOLD', 0.60, 500.00))

        assert len(batch.get_buy_signals()) == 2
        assert len(batch.get_sell_signals()) == 1

        aapl_signals = batch.get_signals_for_symbol('AAPL')
        assert len(aapl_signals) == 1
        assert aapl_signals[0].symbol == 'AAPL'


class TestDataRequirements:
    """Test DataRequirements."""

    def test_data_requirements(self):
        """Test specifying data requirements."""
        req = DataRequirements()

        req.add_daily_data('SPY', 200)
        req.add_daily_data('VIX', 365)
        req.add_intraday_data('TQQQ', '1Min', 390)

        assert len(req.daily_data) == 2
        assert len(req.intraday_data) == 1

        symbols = req.get_all_symbols()
        assert 'SPY' in symbols
        assert 'VIX' in symbols
        assert 'TQQQ' in symbols


class TestETFUniverse:
    """Test ETF Universe management."""

    def test_leveraged_3x_list(self):
        """Test leveraged 3x ETF list."""
        etfs = ETFUniverse.LEVERAGED_3X

        assert 'TQQQ' in etfs
        assert 'SQQQ' in etfs
        assert 'UPRO' in etfs
        assert 'SPXU' in etfs
        assert len(etfs) >= 18  # At least 18 leveraged 3x ETFs

    def test_get_inverse_etf(self):
        """Test getting inverse ETF."""
        assert ETFUniverse.get_inverse_etf('TQQQ') == 'SQQQ'
        assert ETFUniverse.get_inverse_etf('SQQQ') == 'TQQQ'
        assert ETFUniverse.get_inverse_etf('UPRO') == 'SPXU'
        assert ETFUniverse.get_inverse_etf('TMF') == 'TMV'

    def test_is_leveraged(self):
        """Test checking if ETF is leveraged."""
        assert ETFUniverse.is_leveraged('TQQQ') == True
        assert ETFUniverse.is_leveraged('QLD') == True
        assert ETFUniverse.is_leveraged('SPY') == False
        assert ETFUniverse.is_leveraged('QQQ') == False

    def test_get_leverage_factor(self):
        """Test getting leverage factor."""
        assert ETFUniverse.get_leverage_factor('TQQQ') == 3
        assert ETFUniverse.get_leverage_factor('QLD') == 2
        assert ETFUniverse.get_leverage_factor('SPY') == 1

    def test_get_leveraged_bull_bear(self):
        """Test separating bull and bear leveraged ETFs."""
        bull = ETFUniverse.get_leveraged_bull()
        bear = ETFUniverse.get_leveraged_bear()

        assert 'TQQQ' in bull
        assert 'SQQQ' in bear
        assert 'UPRO' in bull
        assert 'SPXU' in bear

        # No overlap
        assert len(set(bull) & set(bear)) == 0

    def test_get_by_sector(self):
        """Test getting ETFs by sector."""
        tech = ETFUniverse.get_by_sector('Technology')
        assert 'XLK' in tech
        assert 'TECL' in tech

        financial = ETFUniverse.get_by_sector('Financial')
        assert 'XLF' in financial
        assert 'FAS' in financial


class TestEquityUniverse:
    """Test Equity Universe management."""

    def test_faang_list(self):
        """Test FAANG stock list."""
        faang = EquityUniverse.FAANG

        assert 'AAPL' in faang
        assert 'AMZN' in faang
        assert 'GOOGL' in faang
        assert 'META' in faang
        assert 'NFLX' in faang
        assert len(faang) == 5

    def test_mag7_list(self):
        """Test Magnificent 7 list."""
        mag7 = EquityUniverse.MAG7

        assert 'AAPL' in mag7
        assert 'MSFT' in mag7
        assert 'GOOGL' in mag7
        assert 'AMZN' in mag7
        assert 'NVDA' in mag7
        assert 'META' in mag7
        assert 'TSLA' in mag7
        assert len(mag7) == 7

    def test_get_by_sector(self):
        """Test getting stocks by sector."""
        semis = EquityUniverse.get_by_sector('Semiconductors')
        assert 'NVDA' in semis
        assert 'AMD' in semis
        assert 'INTC' in semis

        financials = EquityUniverse.get_by_sector('Financial')
        assert 'JPM' in financials
        assert 'BAC' in financials


class MockStrategy(StrategySignals):
    """Mock strategy for testing abstract interface."""

    def __init__(self, lookback: int = 50):
        self.lookback = lookback

    def generate_signals(self, market_data, timestamp):
        """Simple mock implementation."""
        signals = []

        for symbol, df in market_data.items():
            # Generate a BUY signal if close > previous close
            if len(df) >= 2:
                if df['close'].iloc[-1] > df['close'].iloc[-2]:
                    signals.append(Signal(
                        timestamp=timestamp,
                        symbol=symbol,
                        direction='BUY',
                        confidence=0.75,
                        price=df['close'].iloc[-1],
                        metadata={'strategy': 'Mock'}
                    ))

        return signals

    def get_required_lookback(self):
        return self.lookback


class TestStrategySignals:
    """Test StrategySignals abstract interface."""

    def test_strategy_implementation(self):
        """Test implementing a strategy."""
        strategy = MockStrategy(lookback=50)

        # Create mock market data
        dates = pd.date_range('2025-01-01', periods=100, freq='D')
        df = pd.DataFrame({
            'open': np.random.uniform(100, 110, 100),
            'high': np.random.uniform(110, 120, 100),
            'low': np.random.uniform(90, 100, 100),
            'close': np.random.uniform(100, 110, 100),
            'volume': np.random.randint(1000000, 10000000, 100)
        }, index=dates)

        market_data = {'AAPL': df}

        # Generate signals
        signals = strategy.generate_signals(market_data, datetime.now())

        assert isinstance(signals, list)
        # Signals may or may not be generated depending on random data

    def test_data_validation(self):
        """Test data validation."""
        strategy = MockStrategy()

        # Valid data
        dates = pd.date_range('2025-01-01', periods=100, freq='D')
        valid_df = pd.DataFrame({
            'open': np.random.uniform(100, 110, 100),
            'high': np.random.uniform(110, 120, 100),
            'low': np.random.uniform(90, 100, 100),
            'close': np.random.uniform(100, 110, 100),
            'volume': np.random.randint(1000000, 10000000, 100)
        }, index=dates)

        is_valid, error = strategy.validate_data(valid_df, 'AAPL')
        assert is_valid == True
        assert error == ""

        # Invalid data - missing columns
        invalid_df = pd.DataFrame({
            'close': [100, 101, 102]
        })

        is_valid, error = strategy.validate_data(invalid_df, 'AAPL')
        assert is_valid == False
        assert 'Missing columns' in error

        # Invalid data - not enough rows
        short_df = pd.DataFrame({
            'open': [100],
            'high': [110],
            'low': [90],
            'close': [105],
            'volume': [1000000]
        }, index=pd.date_range('2025-01-01', periods=1, freq='D'))

        is_valid, error = strategy.validate_data(short_df, 'AAPL')
        assert is_valid == False
        assert 'Insufficient data' in error


def test_phase1_imports():
    """Test that all Phase 1 components can be imported."""
    # Core abstractions
    from src.strategies.core import Signal, SignalBatch, StrategySignals, DataRequirements

    # Universes
    from src.strategies.universe import ETFUniverse, EquityUniverse

    # Adapters (empty for now)
    import src.backtesting.adapters
    import src.trading.adapters

    # All imports successful
    assert True


def test_no_breaking_changes():
    """Verify existing code still works."""
    # These imports should still work without modification
    from src.strategies.advanced.market_regime_detector import MarketRegimeDetector
    from src.strategies.advanced.bayesian_reversion_model import BayesianReversionModel
    from src.strategies.advanced.overnight_signal_generator import OvernightReversionSignals

    # All imports successful
    assert True


if __name__ == '__main__':
    # Run tests with pytest
    pytest.main([__file__, '-v', '--tb=short'])
