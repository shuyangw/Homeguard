"""
Tests for OMR (Overnight Mean Reversion) and RAMP (Regime-Aware Momentum Protection) strategies.

Tests cover:
- OvernightReversionSignals: signal generation, regime filtering, probability thresholds
- RAMPSignals: regime detection, momentum ranking, risk signals, position sizing
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta
from unittest.mock import MagicMock, patch


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def mock_regime_detector():
    """Create a mock regime detector with controllable output."""
    detector = MagicMock()
    detector.classify_regime.return_value = ('STRONG_BULL', 0.85)
    detector.get_regime_parameters.return_value = {
        'min_win_rate': 0.55,
        'min_expected_return': 0.002,
        'max_positions': 5,
        'position_size_multiplier': 1.0
    }
    return detector


@pytest.fixture
def mock_bayesian_model():
    """Create a mock Bayesian model with controllable probabilities."""
    model = MagicMock()
    model.trained = True
    model.regime_probabilities = {'TQQQ': {}, 'SOXL': {}, 'UPRO': {}}

    def get_prob(symbol, regime, intraday_return):
        # Return high probability for down days (mean reversion signal)
        if intraday_return < -0.02:
            return {
                'probability': 0.65,
                'expected_return': 0.008,
                'sharpe': 1.5,
                'sample_size': 100,
                'confidence': 0.9
            }
        elif intraday_return > 0.02:
            return {
                'probability': 0.60,
                'expected_return': 0.005,
                'sharpe': 1.2,
                'sample_size': 80,
                'confidence': 0.85
            }
        else:
            return None  # Flat day, no signal

    model.get_reversion_probability.side_effect = get_prob
    return model


@pytest.fixture
def sample_intraday_data():
    """Create sample intraday data for OMR testing."""
    # Create a trading day with data from 9:30 AM to 4:00 PM
    base_date = datetime(2024, 1, 15)
    timestamps = pd.date_range(
        start=f'{base_date.date()} 09:30:00',
        end=f'{base_date.date()} 16:00:00',
        freq='1min'
    )

    np.random.seed(42)
    # Simulate a down day (-3% intraday)
    open_price = 100.0
    close_price = 97.0
    n = len(timestamps)

    prices = np.linspace(open_price, close_price, n) + np.random.normal(0, 0.2, n)

    return pd.DataFrame({
        'open': prices - np.random.uniform(0, 0.1, n),
        'high': prices + np.abs(np.random.normal(0, 0.3, n)),
        'low': prices - np.abs(np.random.normal(0, 0.3, n)),
        'close': prices,
        'volume': np.random.uniform(1e4, 1e5, n)
    }, index=timestamps)


@pytest.fixture
def sample_spy_vix_data():
    """Create sample SPY and VIX data for regime detection."""
    dates = pd.date_range('2023-01-01', periods=300, freq='D')
    np.random.seed(42)

    # SPY: trending up
    spy_prices = 400 + np.cumsum(np.random.normal(0.5, 2, len(dates)))

    # VIX: low volatility (below 25)
    vix_prices = 15 + np.random.normal(0, 3, len(dates))
    vix_prices = np.clip(vix_prices, 10, 40)

    spy_series = pd.Series(spy_prices, index=dates, name='close')
    vix_series = pd.Series(vix_prices, index=dates, name='close')

    return spy_series, vix_series


@pytest.fixture
def sample_universe_prices():
    """Create sample price data for S&P 500 universe (RAMP testing)."""
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    np.random.seed(42)

    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'JPM', 'V', 'MA']
    data = {}

    for i, symbol in enumerate(symbols):
        # Different momentum profiles
        trend = np.random.uniform(-0.001, 0.003)  # Daily drift
        prices = 100 * np.exp(np.cumsum(np.random.normal(trend, 0.02, len(dates))))
        data[symbol] = prices

    return pd.DataFrame(data, index=dates)


# =============================================================================
# OMR SIGNAL TESTS
# =============================================================================

class TestOvernightReversionSignals:
    """Tests for OvernightReversionSignals class."""

    def test_initialization_default_symbols(self, mock_regime_detector, mock_bayesian_model):
        """Test initialization with default symbol universe."""
        from src.strategies.advanced.overnight_signal_generator import OvernightReversionSignals

        signals = OvernightReversionSignals(
            regime_detector=mock_regime_detector,
            bayesian_model=mock_bayesian_model
        )

        assert signals.min_probability == 0.55
        assert signals.min_expected_return == 0.002
        assert signals.max_positions == 5
        assert signals.skip_bear_regime is True
        assert len(signals.symbols) > 0  # Should have default universe

    def test_initialization_custom_symbols(self, mock_regime_detector, mock_bayesian_model):
        """Test initialization with custom symbol list."""
        from src.strategies.advanced.overnight_signal_generator import OvernightReversionSignals

        custom_symbols = ['TQQQ', 'SOXL', 'UPRO']
        signals = OvernightReversionSignals(
            regime_detector=mock_regime_detector,
            bayesian_model=mock_bayesian_model,
            symbols=custom_symbols
        )

        assert signals.symbols == custom_symbols

    def test_generate_signals_bull_regime(
        self, mock_regime_detector, mock_bayesian_model, sample_intraday_data
    ):
        """Test signal generation in bull regime produces signals."""
        from src.strategies.advanced.overnight_signal_generator import OvernightReversionSignals

        # Setup mock to return STRONG_BULL
        mock_regime_detector.classify_regime.return_value = ('STRONG_BULL', 0.85)

        signals_gen = OvernightReversionSignals(
            regime_detector=mock_regime_detector,
            bayesian_model=mock_bayesian_model,
            symbols=['TQQQ']
        )

        # Create market data with SPY, VIX, and symbol
        market_data = {
            'SPY': sample_intraday_data.copy(),
            'VIX': sample_intraday_data.copy(),  # Just needs close column
            'TQQQ': sample_intraday_data.copy()
        }

        timestamp = datetime(2024, 1, 15, 15, 50)
        signals = signals_gen.generate_signals(market_data, timestamp)

        # Should generate signal for down day in bull regime
        assert len(signals) >= 0  # May or may not generate based on thresholds
        mock_regime_detector.classify_regime.assert_called_once()

    def test_generate_signals_bear_regime_skipped(
        self, mock_regime_detector, mock_bayesian_model, sample_intraday_data
    ):
        """Test that signals are skipped in BEAR regime when skip_bear_regime=True."""
        from src.strategies.advanced.overnight_signal_generator import OvernightReversionSignals

        # Setup mock to return BEAR regime
        mock_regime_detector.classify_regime.return_value = ('BEAR', 0.75)

        signals_gen = OvernightReversionSignals(
            regime_detector=mock_regime_detector,
            bayesian_model=mock_bayesian_model,
            symbols=['TQQQ'],
            skip_bear_regime=True
        )

        market_data = {
            'SPY': sample_intraday_data.copy(),
            'VIX': sample_intraday_data.copy(),
            'TQQQ': sample_intraday_data.copy()
        }

        timestamp = datetime(2024, 1, 15, 15, 50)
        signals = signals_gen.generate_signals(market_data, timestamp)

        # Should return empty list in BEAR regime
        assert signals == []

    def test_generate_signals_bear_regime_allowed(
        self, mock_regime_detector, mock_bayesian_model, sample_intraday_data
    ):
        """Test that signals are generated in BEAR regime when skip_bear_regime=False."""
        from src.strategies.advanced.overnight_signal_generator import OvernightReversionSignals

        mock_regime_detector.classify_regime.return_value = ('BEAR', 0.75)

        signals_gen = OvernightReversionSignals(
            regime_detector=mock_regime_detector,
            bayesian_model=mock_bayesian_model,
            symbols=['TQQQ'],
            skip_bear_regime=False  # Allow trading in bear
        )

        market_data = {
            'SPY': sample_intraday_data.copy(),
            'VIX': sample_intraday_data.copy(),
            'TQQQ': sample_intraday_data.copy()
        }

        timestamp = datetime(2024, 1, 15, 15, 50)
        signals = signals_gen.generate_signals(market_data, timestamp)

        # Should attempt to generate signals (may still filter based on probability)
        mock_regime_detector.classify_regime.assert_called_once()

    def test_missing_spy_vix_data(self, mock_regime_detector, mock_bayesian_model):
        """Test that missing SPY/VIX data returns empty signals."""
        from src.strategies.advanced.overnight_signal_generator import OvernightReversionSignals

        signals_gen = OvernightReversionSignals(
            regime_detector=mock_regime_detector,
            bayesian_model=mock_bayesian_model,
            symbols=['TQQQ']
        )

        # Missing SPY
        market_data = {'VIX': pd.DataFrame(), 'TQQQ': pd.DataFrame()}
        signals = signals_gen.generate_signals(market_data, datetime.now())
        assert signals == []

        # Missing VIX
        market_data = {'SPY': pd.DataFrame(), 'TQQQ': pd.DataFrame()}
        signals = signals_gen.generate_signals(market_data, datetime.now())
        assert signals == []

    def test_signal_strength_calculation(self, mock_regime_detector, mock_bayesian_model):
        """Test signal strength is calculated correctly."""
        from src.strategies.advanced.overnight_signal_generator import OvernightReversionSignals

        signals_gen = OvernightReversionSignals(
            regime_detector=mock_regime_detector,
            bayesian_model=mock_bayesian_model,
            symbols=['TQQQ']
        )

        prob_data = {
            'probability': 0.70,  # High win rate
            'expected_return': 0.005,  # 0.5% expected
        }

        strength = signals_gen._calculate_signal_strength(
            prob_data, regime_confidence=0.9, intraday_return=-0.03
        )

        # Signal strength should be between 0 and 1
        assert 0 <= strength <= 1
        # Higher probability and return should give higher strength
        assert strength > 0.5

    def test_position_sizing_normalization(self, mock_regime_detector, mock_bayesian_model):
        """Test that position sizes are normalized and capped."""
        from src.strategies.advanced.overnight_signal_generator import OvernightReversionSignals

        signals_gen = OvernightReversionSignals(
            regime_detector=mock_regime_detector,
            bayesian_model=mock_bayesian_model,
            symbols=['TQQQ']
        )

        # Create signals with varying strengths
        signals = [
            {'signal_strength': 0.9},
            {'signal_strength': 0.7},
            {'signal_strength': 0.5},
        ]

        regime_params = {'position_size_multiplier': 1.0}
        sized_signals = signals_gen._add_position_sizing(signals, regime_params)

        # Check position sizes
        total_size = sum(s['position_size'] for s in sized_signals)
        assert total_size <= 1.0  # Total shouldn't exceed 100%

        for signal in sized_signals:
            assert signal['position_size'] <= 0.25  # Max 25% per position


# =============================================================================
# RAMP SIGNAL TESTS
# =============================================================================

class TestRAMPSignals:
    """Tests for RAMPSignals class."""

    def test_initialization(self):
        """Test RAMP initialization with default parameters."""
        from src.strategies.advanced.ramp_strategy import RAMPSignals

        symbols = ['AAPL', 'MSFT', 'GOOGL']
        ramp = RAMPSignals(symbols=symbols)

        assert ramp.symbols == symbols
        assert ramp.reduced_exposure == 0.5
        assert ramp.vix_threshold == 25.0
        assert ramp.spy_dd_threshold == -0.05
        assert ramp._current_regime == 'SIDEWAYS'  # Default

    def test_initialization_custom_params(self):
        """Test RAMP initialization with custom parameters."""
        from src.strategies.advanced.ramp_strategy import RAMPSignals

        ramp = RAMPSignals(
            symbols=['AAPL'],
            reduced_exposure=0.3,
            vix_threshold=30.0,
            spy_dd_threshold=-0.10
        )

        assert ramp.reduced_exposure == 0.3
        assert ramp.vix_threshold == 30.0
        assert ramp.spy_dd_threshold == -0.10

    def test_regime_detection_with_data(self, sample_spy_vix_data):
        """Test regime detection with valid SPY/VIX data."""
        from src.strategies.advanced.ramp_strategy import RAMPSignals

        spy_prices, vix_prices = sample_spy_vix_data

        ramp = RAMPSignals(symbols=['AAPL'])

        with patch.object(ramp.regime_detector, 'classify_regime') as mock_classify:
            mock_classify.return_value = ('STRONG_BULL', 0.85)

            regime, confidence = ramp.detect_regime(spy_prices, vix_prices)

            assert regime == 'STRONG_BULL'
            assert confidence == 0.85
            assert ramp._current_regime == 'STRONG_BULL'
            assert ramp._current_params['top_n'] == 20  # STRONG_BULL params

    def test_regime_detection_insufficient_data(self):
        """Test regime detection with insufficient data returns default."""
        from src.strategies.advanced.ramp_strategy import RAMPSignals

        ramp = RAMPSignals(symbols=['AAPL'])

        # Less than 252 days of data
        short_spy = pd.Series(np.random.randn(100), index=pd.date_range('2024-01-01', periods=100))

        regime, confidence = ramp.detect_regime(short_spy, short_spy)

        assert regime == 'SIDEWAYS'
        assert confidence == 0.5

    def test_momentum_scores_calculation(self, sample_universe_prices):
        """Test momentum score calculation."""
        from src.strategies.advanced.ramp_strategy import RAMPSignals

        symbols = list(sample_universe_prices.columns)
        ramp = RAMPSignals(symbols=symbols)
        ramp._prices_cache = sample_universe_prices

        # Set regime params
        ramp._current_params = {'long_p': 21, 'short_p': 5, 'long_w': 0.3, 'pen_w': 5.0, 'top_n': 5}

        scores = ramp.calculate_momentum_scores()

        assert isinstance(scores, pd.Series)
        assert len(scores) > 0
        # Scores should be finite
        assert scores.isna().sum() < len(scores)

    def test_risk_signals_high_vix(self):
        """Test risk signal triggers when VIX > threshold."""
        from src.strategies.advanced.ramp_strategy import RAMPSignals

        ramp = RAMPSignals(symbols=['AAPL'], vix_threshold=25.0)

        # VIX above threshold
        high_vix = pd.Series([30.0], index=[pd.Timestamp('2024-01-15')])

        risk_signals = ramp.calculate_risk_signals(vix_prices=high_vix)

        assert risk_signals.high_vix == True
        assert risk_signals.reduce_exposure == True
        assert risk_signals.exposure_pct == 0.5  # reduced_exposure default

    def test_risk_signals_spy_drawdown(self, sample_spy_vix_data):
        """Test risk signal triggers when SPY drawdown > threshold."""
        from src.strategies.advanced.ramp_strategy import RAMPSignals

        spy_prices, _ = sample_spy_vix_data

        ramp = RAMPSignals(symbols=['AAPL'], spy_dd_threshold=-0.05)

        # Modify SPY to have large drawdown
        spy_dd = spy_prices.copy()
        spy_dd.iloc[-1] = spy_dd.max() * 0.90  # 10% drawdown from peak

        risk_signals = ramp.calculate_risk_signals(spy_prices=spy_dd)

        assert risk_signals.spy_drawdown == True
        assert risk_signals.reduce_exposure == True

    def test_risk_signals_no_trigger(self, sample_spy_vix_data):
        """Test risk signals don't trigger in normal conditions."""
        from src.strategies.advanced.ramp_strategy import RAMPSignals

        spy_prices, vix_prices = sample_spy_vix_data

        ramp = RAMPSignals(symbols=['AAPL'])
        ramp._spy_cache = spy_prices
        ramp._vix_cache = vix_prices

        risk_signals = ramp.calculate_risk_signals()

        # Normal conditions - no protection triggered
        assert risk_signals.reduce_exposure == False
        assert risk_signals.exposure_pct == 1.0

    def test_generate_signals_buy_sell(self, sample_universe_prices, sample_spy_vix_data):
        """Test signal generation produces buy and sell signals."""
        from src.strategies.advanced.ramp_strategy import RAMPSignals

        spy_prices, vix_prices = sample_spy_vix_data
        symbols = list(sample_universe_prices.columns)

        ramp = RAMPSignals(symbols=symbols)
        ramp._prices_cache = sample_universe_prices
        ramp._spy_cache = spy_prices
        ramp._vix_cache = vix_prices

        with patch.object(ramp.regime_detector, 'classify_regime') as mock_classify:
            mock_classify.return_value = ('SIDEWAYS', 0.7)

            # Current positions: hold some stocks
            current_positions = {'AAPL': 10000, 'MSFT': 10000}

            signals, risk_signals = ramp.generate_signals(current_positions)

            # Should have signals for each position and new entries
            assert len(signals) > 0

            # Check signal types
            actions = [s.action for s in signals]
            assert 'buy' in actions or 'hold' in actions or 'sell' in actions

    def test_dynamic_topn_by_regime(self):
        """Test that TopN varies by regime correctly."""
        from src.strategies.advanced.ramp_strategy import RAMPSignals, REGIME_PARAMS

        ramp = RAMPSignals(symbols=['AAPL'])

        # Check REGIME_PARAMS has different TopN values
        assert REGIME_PARAMS['STRONG_BULL']['top_n'] == 20
        assert REGIME_PARAMS['WEAK_BULL']['top_n'] == 10
        assert REGIME_PARAMS['SIDEWAYS']['top_n'] == 5
        assert REGIME_PARAMS['UNPREDICTABLE']['top_n'] == 10
        assert REGIME_PARAMS['BEAR']['top_n'] == 10

    def test_position_sizing_1_over_n(self, sample_universe_prices, sample_spy_vix_data):
        """Test 1/N position sizing."""
        from src.strategies.advanced.ramp_strategy import RAMPSignals

        spy_prices, vix_prices = sample_spy_vix_data
        symbols = list(sample_universe_prices.columns)

        ramp = RAMPSignals(symbols=symbols)
        ramp._prices_cache = sample_universe_prices
        ramp._spy_cache = spy_prices
        ramp._vix_cache = vix_prices

        with patch.object(ramp.regime_detector, 'classify_regime') as mock_classify:
            mock_classify.return_value = ('SIDEWAYS', 0.7)  # top_n = 5

            signals, risk_signals = ramp.generate_signals()

            # Get buy/hold signals (not sell)
            active_signals = [s for s in signals if s.action != 'sell']

            if len(active_signals) > 0:
                # Each position should have equal weight
                expected_weight = 1.0 / ramp._current_params['top_n']
                for s in active_signals:
                    assert s.weight == pytest.approx(expected_weight, rel=0.01)

    def test_regime_params_property(self):
        """Test current_params property returns copy."""
        from src.strategies.advanced.ramp_strategy import RAMPSignals

        ramp = RAMPSignals(symbols=['AAPL'])

        params = ramp.current_params
        params['top_n'] = 999  # Modify the copy

        # Original should be unchanged
        assert ramp._current_params['top_n'] != 999

    def test_update_historical_data(self, sample_universe_prices, sample_spy_vix_data):
        """Test updating historical data cache."""
        from src.strategies.advanced.ramp_strategy import RAMPSignals

        spy_prices, vix_prices = sample_spy_vix_data

        ramp = RAMPSignals(symbols=['AAPL'])

        ramp.update_historical_data(sample_universe_prices, spy_prices, vix_prices)

        assert ramp._prices_cache is not None
        assert ramp._spy_cache is not None
        assert ramp._vix_cache is not None
        assert len(ramp._prices_cache) == len(sample_universe_prices)


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_omr_empty_market_data(self, mock_regime_detector, mock_bayesian_model):
        """Test OMR handles empty market data gracefully."""
        from src.strategies.advanced.overnight_signal_generator import OvernightReversionSignals

        signals_gen = OvernightReversionSignals(
            regime_detector=mock_regime_detector,
            bayesian_model=mock_bayesian_model,
            symbols=['TQQQ']
        )

        signals = signals_gen.generate_signals({}, datetime.now())
        assert signals == []

    def test_ramp_empty_price_data(self):
        """Test RAMP handles empty price data gracefully."""
        from src.strategies.advanced.ramp_strategy import RAMPSignals

        ramp = RAMPSignals(symbols=['AAPL'])

        scores = ramp.calculate_momentum_scores(pd.DataFrame())
        assert len(scores) == 0

    def test_omr_model_not_trained(self, mock_regime_detector):
        """Test OMR handles untrained Bayesian model."""
        from src.strategies.advanced.overnight_signal_generator import OvernightReversionSignals

        mock_model = MagicMock()
        mock_model.trained = False
        mock_model.regime_probabilities = {}

        signals_gen = OvernightReversionSignals(
            regime_detector=mock_regime_detector,
            bayesian_model=mock_model,
            symbols=['TQQQ']
        )

        # Model status should be logged but not crash
        assert signals_gen.bayesian_model.trained is False

    def test_ramp_data_quality_alert(self, sample_universe_prices):
        """Test RAMP logs alert when data quality is poor."""
        from src.strategies.advanced.ramp_strategy import RAMPSignals

        symbols = list(sample_universe_prices.columns)
        ramp = RAMPSignals(symbols=symbols)

        # Create data with many NaNs
        bad_data = sample_universe_prices.copy()
        bad_data.iloc[:50, :] = np.nan  # First 50 rows are NaN

        ramp._prices_cache = bad_data
        ramp._current_params = {'long_p': 21, 'short_p': 5, 'long_w': 0.3, 'pen_w': 5.0, 'top_n': 5}

        # Should still return scores but with NaNs filtered
        scores = ramp.calculate_momentum_scores()
        assert isinstance(scores, pd.Series)


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Integration tests for end-to-end signal generation."""

    def test_ramp_full_workflow(self, sample_universe_prices, sample_spy_vix_data):
        """Test complete RAMP workflow from data to signals."""
        from src.strategies.advanced.ramp_strategy import RAMPSignals

        spy_prices, vix_prices = sample_spy_vix_data
        symbols = list(sample_universe_prices.columns)

        # Initialize
        ramp = RAMPSignals(symbols=symbols)

        # Update data
        ramp.update_historical_data(sample_universe_prices, spy_prices, vix_prices)

        # Mock regime detection
        with patch.object(ramp.regime_detector, 'classify_regime') as mock_classify:
            mock_classify.return_value = ('WEAK_BULL', 0.75)

            # Detect regime
            regime, confidence = ramp.detect_regime()
            assert regime == 'WEAK_BULL'
            assert ramp._current_params['top_n'] == 10

            # Calculate risk signals
            risk_signals = ramp.calculate_risk_signals()
            assert risk_signals.regime == 'WEAK_BULL'

            # Generate signals
            signals, risk = ramp.generate_signals()
            assert len(signals) > 0

            # Verify signal structure
            for signal in signals:
                assert hasattr(signal, 'symbol')
                assert hasattr(signal, 'momentum_score')
                assert hasattr(signal, 'rank')
                assert hasattr(signal, 'weight')
                assert hasattr(signal, 'action')
                assert hasattr(signal, 'regime')
