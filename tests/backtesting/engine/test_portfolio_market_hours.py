"""
Tests for portfolio market hours filtering and Numba decision logic.

Tests the BasePortfolio market hours checking and Portfolio's
decision to use Numba or Python simulation.
"""

import pytest
from datetime import time, datetime
import pandas as pd
import numpy as np
import pytz

from src.backtesting.engine.base_portfolio import BasePortfolio
from src.backtesting.utils.risk_config import RiskConfig


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def eastern_tz():
    """Get Eastern timezone."""
    return pytz.timezone('US/Eastern')


@pytest.fixture
def utc_tz():
    """Get UTC timezone."""
    return pytz.UTC


@pytest.fixture
def simple_price_data():
    """Create simple price data for testing."""
    dates = pd.date_range(start='2024-01-02', periods=100, freq='1min')
    # Make sure it's a trading day (Tuesday) during market hours
    dates = pd.date_range(
        start='2024-01-02 09:35:00',
        periods=100,
        freq='1min',
        tz='US/Eastern'
    )
    prices = pd.Series(np.random.uniform(100, 110, 100), index=dates)
    return prices


# =============================================================================
# CONCRETE TEST CLASS
# =============================================================================

class ConcretePortfolio(BasePortfolio):
    """Concrete implementation for testing abstract base class."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._simulate()

    def _simulate(self):
        """Minimal simulation implementation."""
        # Create a simple equity curve
        self.equity_curve = pd.Series([self.init_cash] * 10)

    def stats(self):
        return {}

    def returns(self, freq=None):
        return pd.Series([0.0])


# =============================================================================
# MARKET HOURS FILTERING TESTS
# =============================================================================

class TestMarketHoursFiltering:
    """Tests for market hours filtering logic."""

    def test_market_hours_enabled_filters_weekend(self, eastern_tz):
        """Test that weekends are filtered when market_hours_only=True."""
        portfolio = ConcretePortfolio(
            init_cash=100000,
            fees=0.001,
            slippage=0.0,
            market_hours_only=True
        )

        # Saturday
        saturday = pd.Timestamp('2024-01-06 12:00:00', tz=eastern_tz)
        assert portfolio._is_market_hours(saturday) == False

        # Sunday
        sunday = pd.Timestamp('2024-01-07 12:00:00', tz=eastern_tz)
        assert portfolio._is_market_hours(sunday) == False

    def test_market_hours_enabled_allows_weekday(self, eastern_tz):
        """Test that weekdays within hours are allowed."""
        portfolio = ConcretePortfolio(
            init_cash=100000,
            fees=0.001,
            slippage=0.0,
            market_hours_only=True
        )

        # Tuesday at noon (within default hours 9:35 - 15:55)
        tuesday_noon = pd.Timestamp('2024-01-02 12:00:00', tz=eastern_tz)
        assert portfolio._is_market_hours(tuesday_noon) == True

    def test_market_hours_before_open(self, eastern_tz):
        """Test that time before market open is filtered."""
        portfolio = ConcretePortfolio(
            init_cash=100000,
            fees=0.001,
            slippage=0.0,
            market_hours_only=True
        )

        # Tuesday at 9:30 (before default open of 9:35)
        before_open = pd.Timestamp('2024-01-02 09:30:00', tz=eastern_tz)
        assert portfolio._is_market_hours(before_open) == False

    def test_market_hours_after_close(self, eastern_tz):
        """Test that time after market close is filtered."""
        portfolio = ConcretePortfolio(
            init_cash=100000,
            fees=0.001,
            slippage=0.0,
            market_hours_only=True
        )

        # Tuesday at 4:00 PM (after default close of 3:55)
        after_close = pd.Timestamp('2024-01-02 16:00:00', tz=eastern_tz)
        assert portfolio._is_market_hours(after_close) == False

    def test_market_hours_at_open(self, eastern_tz):
        """Test that exactly at market open is allowed."""
        portfolio = ConcretePortfolio(
            init_cash=100000,
            fees=0.001,
            slippage=0.0,
            market_hours_only=True
        )

        # Tuesday at exactly 9:35 (default open)
        at_open = pd.Timestamp('2024-01-02 09:35:00', tz=eastern_tz)
        assert portfolio._is_market_hours(at_open) == True

    def test_market_hours_at_close(self, eastern_tz):
        """Test that exactly at market close is allowed."""
        portfolio = ConcretePortfolio(
            init_cash=100000,
            fees=0.001,
            slippage=0.0,
            market_hours_only=True
        )

        # Tuesday at exactly 3:55 (default close)
        at_close = pd.Timestamp('2024-01-02 15:55:00', tz=eastern_tz)
        assert portfolio._is_market_hours(at_close) == True

    def test_market_hours_disabled_allows_all(self, eastern_tz):
        """Test that all times are allowed when market_hours_only=False."""
        portfolio = ConcretePortfolio(
            init_cash=100000,
            fees=0.001,
            slippage=0.0,
            market_hours_only=False
        )

        # Weekend
        saturday = pd.Timestamp('2024-01-06 12:00:00', tz=eastern_tz)
        assert portfolio._is_market_hours(saturday) == True

        # After hours
        after_close = pd.Timestamp('2024-01-02 20:00:00', tz=eastern_tz)
        assert portfolio._is_market_hours(after_close) == True


# =============================================================================
# TIMEZONE CONVERSION TESTS
# =============================================================================

class TestTimezoneConversion:
    """Tests for timezone handling in market hours."""

    def test_utc_timestamp_converted_to_eastern(self, utc_tz, eastern_tz):
        """Test that UTC timestamps are converted to Eastern time."""
        portfolio = ConcretePortfolio(
            init_cash=100000,
            fees=0.001,
            slippage=0.0,
            market_hours_only=True
        )

        # 17:00 UTC = 12:00 EST (winter)
        utc_noon_est = pd.Timestamp('2024-01-02 17:00:00', tz=utc_tz)
        assert portfolio._is_market_hours(utc_noon_est) == True

        # 02:00 UTC = 21:00 EST previous day (after hours)
        utc_after_hours = pd.Timestamp('2024-01-03 02:00:00', tz=utc_tz)
        assert portfolio._is_market_hours(utc_after_hours) == False

    def test_naive_timestamp_treated_as_eastern(self):
        """Test that naive timestamps are treated as Eastern time."""
        portfolio = ConcretePortfolio(
            init_cash=100000,
            fees=0.001,
            slippage=0.0,
            market_hours_only=True
        )

        # Naive timestamp at noon (assumed Eastern)
        naive_noon = pd.Timestamp('2024-01-02 12:00:00')
        assert portfolio._is_market_hours(naive_noon) == True

        # Naive timestamp at 8 AM (before market)
        naive_early = pd.Timestamp('2024-01-02 08:00:00')
        assert portfolio._is_market_hours(naive_early) == False


# =============================================================================
# CUSTOM MARKET HOURS TESTS
# =============================================================================

class TestCustomMarketHours:
    """Tests for custom market hours configuration."""

    def test_custom_market_open(self, eastern_tz):
        """Test custom market open time."""
        portfolio = ConcretePortfolio(
            init_cash=100000,
            fees=0.001,
            slippage=0.0,
            market_hours_only=True,
            market_open=time(10, 0)  # 10:00 AM
        )

        # 9:30 should be before custom open
        before_open = pd.Timestamp('2024-01-02 09:30:00', tz=eastern_tz)
        assert portfolio._is_market_hours(before_open) == False

        # 10:00 should be at custom open
        at_open = pd.Timestamp('2024-01-02 10:00:00', tz=eastern_tz)
        assert portfolio._is_market_hours(at_open) == True

    def test_custom_market_close(self, eastern_tz):
        """Test custom market close time."""
        portfolio = ConcretePortfolio(
            init_cash=100000,
            fees=0.001,
            slippage=0.0,
            market_hours_only=True,
            market_close=time(15, 0)  # 3:00 PM
        )

        # 3:30 should be after custom close
        after_close = pd.Timestamp('2024-01-02 15:30:00', tz=eastern_tz)
        assert portfolio._is_market_hours(after_close) == False

        # 3:00 should be at custom close
        at_close = pd.Timestamp('2024-01-02 15:00:00', tz=eastern_tz)
        assert portfolio._is_market_hours(at_close) == True


# =============================================================================
# FREQUENCY ANNUALIZATION TESTS
# =============================================================================

class TestFrequencyAnnualization:
    """Tests for period annualization based on frequency."""

    def test_minute_frequency_annualization(self):
        """Test 1-minute data annualization."""
        portfolio = ConcretePortfolio(
            init_cash=100000,
            fees=0.001,
            freq='1min'
        )

        periods = portfolio._get_periods_per_year()
        # 252 trading days * 6.5 hours * 60 minutes
        expected = 252 * 6.5 * 60
        assert periods == expected

    def test_hourly_frequency_annualization(self):
        """Test hourly data annualization."""
        portfolio = ConcretePortfolio(
            init_cash=100000,
            fees=0.001,
            freq='1h'
        )

        periods = portfolio._get_periods_per_year()
        # 252 trading days * 6.5 hours
        expected = 252 * 6.5
        assert periods == expected

    def test_daily_frequency_annualization(self):
        """Test daily data annualization."""
        portfolio = ConcretePortfolio(
            init_cash=100000,
            fees=0.001,
            freq='1d'
        )

        periods = portfolio._get_periods_per_year()
        assert periods == 252

    def test_crypto_daily_frequency(self):
        """Test crypto 24/7 daily annualization."""
        portfolio = ConcretePortfolio(
            init_cash=100000,
            fees=0.001,
            freq='crypto_1day'
        )

        periods = portfolio._get_periods_per_year()
        assert periods == 365

    def test_unknown_frequency_defaults_to_daily(self):
        """Test unknown frequency defaults to 252."""
        portfolio = ConcretePortfolio(
            init_cash=100000,
            fees=0.001,
            freq='unknown_freq'
        )

        periods = portfolio._get_periods_per_year()
        assert periods == 252


# =============================================================================
# NUMBA DECISION TESTS
# =============================================================================

class TestNumbaDecision:
    """Tests for Numba simulation decision logic."""

    def test_numba_enabled_with_fixed_percentage(self, simple_price_data):
        """Test Numba is used with fixed percentage sizing."""
        from src.backtesting.engine.portfolio_simulator import Portfolio

        entries = pd.Series([False] * len(simple_price_data), index=simple_price_data.index)
        exits = pd.Series([False] * len(simple_price_data), index=simple_price_data.index)

        risk_config = RiskConfig(
            position_sizing_method='fixed_percentage',
            position_size_pct=0.1,
            use_stop_loss=False
        )

        portfolio = Portfolio(
            price=simple_price_data,
            entries=entries,
            exits=exits,
            init_cash=100000,
            fees=0.001,
            slippage=0.0,
            risk_config=risk_config,
            use_numba=True
        )

        # Should use Numba
        assert portfolio._can_use_numba() == True

    def test_numba_disabled_with_atr_stops(self, simple_price_data):
        """Test Numba is disabled with ATR stop losses."""
        from src.backtesting.engine.portfolio_simulator import Portfolio

        entries = pd.Series([False] * len(simple_price_data), index=simple_price_data.index)
        exits = pd.Series([False] * len(simple_price_data), index=simple_price_data.index)

        # Create OHLCV DataFrame for ATR calculation
        price_data = pd.DataFrame({
            'open': simple_price_data,
            'high': simple_price_data * 1.01,
            'low': simple_price_data * 0.99,
            'close': simple_price_data,
            'volume': [1000000] * len(simple_price_data)
        }, index=simple_price_data.index)

        risk_config = RiskConfig(
            position_sizing_method='fixed_percentage',
            position_size_pct=0.1,
            use_stop_loss=True,
            stop_loss_type='atr',  # ATR requires price history
            atr_multiplier=2.0
        )

        portfolio = Portfolio(
            price=simple_price_data,
            entries=entries,
            exits=exits,
            init_cash=100000,
            fees=0.001,
            slippage=0.0,
            risk_config=risk_config,
            price_data=price_data,
            use_numba=True
        )

        # Should NOT use Numba because of ATR stops
        assert portfolio._can_use_numba() == False

    def test_numba_disabled_when_explicitly_off(self, simple_price_data):
        """Test Numba is disabled when use_numba=False."""
        from src.backtesting.engine.portfolio_simulator import Portfolio

        entries = pd.Series([False] * len(simple_price_data), index=simple_price_data.index)
        exits = pd.Series([False] * len(simple_price_data), index=simple_price_data.index)

        risk_config = RiskConfig(
            position_sizing_method='fixed_percentage',
            position_size_pct=0.1
        )

        portfolio = Portfolio(
            price=simple_price_data,
            entries=entries,
            exits=exits,
            init_cash=100000,
            fees=0.001,
            slippage=0.0,
            risk_config=risk_config,
            use_numba=False  # Explicitly disabled
        )

        assert portfolio._can_use_numba() == False


# =============================================================================
# BASIC METRICS CALCULATION TESTS
# =============================================================================

class TestBasicMetrics:
    """Tests for basic metrics calculation."""

    def test_calculate_basic_metrics_with_trades(self):
        """Test metrics calculation with closed trades."""
        portfolio = ConcretePortfolio(
            init_cash=100000,
            fees=0.001
        )

        # Create equity series
        equity = pd.Series([100000, 101000, 102000, 101500, 103000])

        # Create closed trades
        trades = [
            {'pnl': 1000},
            {'pnl': -500},
            {'pnl': 1500}
        ]

        metrics = portfolio._calculate_basic_metrics(equity, trades)

        assert 'Total Return [%]' in metrics
        assert 'Win Rate [%]' in metrics
        assert 'Total Trades' in metrics
        assert metrics['Total Trades'] == 3
        assert metrics['Win Rate [%]'] == pytest.approx(66.67, rel=0.1)

    def test_calculate_basic_metrics_no_trades(self):
        """Test metrics calculation without trades."""
        portfolio = ConcretePortfolio(
            init_cash=100000,
            fees=0.001
        )

        equity = pd.Series([100000, 100500, 101000])

        metrics = portfolio._calculate_basic_metrics(equity, None)

        assert 'Total Return [%]' in metrics
        assert 'Win Rate [%]' not in metrics

    def test_calculate_sharpe_ratio(self):
        """Test Sharpe ratio calculation."""
        portfolio = ConcretePortfolio(
            init_cash=100000,
            fees=0.001,
            freq='1d'
        )

        # Create consistent positive returns
        equity = pd.Series([100000, 101000, 102000, 103000, 104000])

        metrics = portfolio._calculate_basic_metrics(equity, None)

        assert 'Sharpe Ratio' in metrics
        # Positive returns should give positive Sharpe
        assert metrics['Sharpe Ratio'] > 0

    def test_calculate_max_drawdown(self):
        """Test max drawdown calculation."""
        portfolio = ConcretePortfolio(
            init_cash=100000,
            fees=0.001
        )

        # Create series with clear drawdown
        equity = pd.Series([100000, 110000, 95000, 100000])
        # Peak is 110000, trough is 95000, drawdown = (95000 - 110000) / 110000 = -13.6%

        metrics = portfolio._calculate_basic_metrics(equity, None)

        assert 'Max Drawdown [%]' in metrics
        assert metrics['Max Drawdown [%]'] < 0  # Drawdown is negative
        assert metrics['Max Drawdown [%]'] == pytest.approx(-13.636, rel=0.01)
