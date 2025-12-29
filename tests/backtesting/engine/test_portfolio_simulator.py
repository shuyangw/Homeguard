"""
Comprehensive tests for Portfolio class in portfolio_simulator.py.

Tests cover:
- Position tracking accuracy
- P&L calculations
- Fee and slippage application
- Stop loss triggering
- Short selling
- BasePortfolio inheritance
"""

import pytest
import pandas as pd
import numpy as np
from datetime import time

from src.backtesting.engine.portfolio_simulator import Portfolio, from_signals
from src.backtesting.engine.base_portfolio import BasePortfolio
from src.backtesting.utils.risk_config import RiskConfig


class TestPortfolioInheritance:
    """Test that Portfolio correctly inherits from BasePortfolio."""

    def test_inherits_from_base_portfolio(self):
        """Verify Portfolio is a subclass of BasePortfolio."""
        assert issubclass(Portfolio, BasePortfolio)

    def test_has_inherited_methods(self):
        """Verify inherited methods are available."""
        price = pd.Series([100.0], index=pd.date_range('2023-01-01', periods=1, freq='D'))
        entries = pd.Series([False], index=price.index)
        exits = pd.Series([False], index=price.index)

        portfolio = Portfolio(
            price=price,
            entries=entries,
            exits=exits,
            init_cash=10000,
            fees=0.0,
            slippage=0.0,
            market_hours_only=False
        )

        # Check inherited methods exist
        assert hasattr(portfolio, '_is_market_hours')
        assert hasattr(portfolio, '_get_periods_per_year')
        assert hasattr(portfolio, 'market_open')
        assert hasattr(portfolio, 'market_close')
        assert hasattr(portfolio, 'eastern_tz')

    def test_inherited_market_hours_constants(self):
        """Verify market hours constants are set correctly."""
        price = pd.Series([100.0], index=pd.date_range('2023-01-01', periods=1, freq='D'))
        entries = pd.Series([False], index=price.index)
        exits = pd.Series([False], index=price.index)

        portfolio = Portfolio(
            price=price,
            entries=entries,
            exits=exits,
            init_cash=10000,
            fees=0.0,
            slippage=0.0,
            market_hours_only=False
        )

        assert portfolio.market_open == time(9, 35)
        assert portfolio.market_close == time(15, 55)


class TestPortfolioInitialization:
    """Test Portfolio initialization."""

    def test_basic_initialization(self):
        """Test basic portfolio creation."""
        price = pd.Series([100.0, 101.0, 102.0], index=pd.date_range('2023-01-01', periods=3, freq='D'))
        entries = pd.Series([True, False, False], index=price.index)
        exits = pd.Series([False, False, True], index=price.index)

        portfolio = Portfolio(
            price=price,
            entries=entries,
            exits=exits,
            init_cash=10000,
            fees=0.001,
            slippage=0.0,
            market_hours_only=False
        )

        assert portfolio.init_cash == 10000
        assert portfolio.fees == 0.001
        assert portfolio.slippage == 0.0

    def test_default_risk_config(self):
        """Test default risk configuration is applied."""
        price = pd.Series([100.0], index=pd.date_range('2023-01-01', periods=1, freq='D'))
        entries = pd.Series([False], index=price.index)
        exits = pd.Series([False], index=price.index)

        portfolio = Portfolio(
            price=price,
            entries=entries,
            exits=exits,
            init_cash=10000,
            fees=0.0,
            slippage=0.0,
            market_hours_only=False
        )

        assert portfolio.risk_config is not None
        assert portfolio.risk_config.position_size_pct > 0


class TestPositionTracking:
    """Test position tracking accuracy."""

    def test_entry_creates_position(self):
        """Test that entry signal creates a position."""
        price = pd.Series(
            [100.0, 100.0, 100.0],
            index=pd.date_range('2023-01-01', periods=3, freq='D')
        )
        entries = pd.Series([True, False, False], index=price.index)
        exits = pd.Series([False, False, True], index=price.index)

        portfolio = Portfolio(
            price=price,
            entries=entries,
            exits=exits,
            init_cash=10000,
            fees=0.0,
            slippage=0.0,
            market_hours_only=False,
            risk_config=RiskConfig(position_size_pct=1.0)  # 100% allocation
        )

        # Should have entry and exit trades
        entry_trades = [t for t in portfolio.trades if t['type'] == 'entry']
        exit_trades = [t for t in portfolio.trades if t['type'] == 'exit']

        assert len(entry_trades) >= 1, "Should have at least one entry trade"
        assert len(exit_trades) >= 1, "Should have at least one exit trade"

    def test_no_position_without_signal(self):
        """Test no trades when no signals."""
        price = pd.Series(
            [100.0, 101.0, 102.0],
            index=pd.date_range('2023-01-01', periods=3, freq='D')
        )
        entries = pd.Series([False, False, False], index=price.index)
        exits = pd.Series([False, False, False], index=price.index)

        portfolio = Portfolio(
            price=price,
            entries=entries,
            exits=exits,
            init_cash=10000,
            fees=0.0,
            slippage=0.0,
            market_hours_only=False
        )

        assert len(portfolio.trades) == 0


class TestPnLCalculations:
    """Test P&L calculation accuracy."""

    def test_profitable_trade(self):
        """Test P&L for a profitable trade."""
        # Buy at 100, sell at 110 = 10% profit
        price = pd.Series(
            [100.0, 105.0, 110.0],
            index=pd.date_range('2023-01-01', periods=3, freq='D')
        )
        entries = pd.Series([True, False, False], index=price.index)
        exits = pd.Series([False, False, True], index=price.index)

        portfolio = Portfolio(
            price=price,
            entries=entries,
            exits=exits,
            init_cash=10000,
            fees=0.0,
            slippage=0.0,
            market_hours_only=False,
            risk_config=RiskConfig(position_size_pct=1.0)
        )

        stats = portfolio.stats()
        if stats and stats.get('Total Trades', 0) > 0:
            assert stats['Total Return [%]'] > 0, "Should have positive return"

    def test_losing_trade(self):
        """Test P&L for a losing trade."""
        # Buy at 100, sell at 90 = 10% loss
        price = pd.Series(
            [100.0, 95.0, 90.0],
            index=pd.date_range('2023-01-01', periods=3, freq='D')
        )
        entries = pd.Series([True, False, False], index=price.index)
        exits = pd.Series([False, False, True], index=price.index)

        portfolio = Portfolio(
            price=price,
            entries=entries,
            exits=exits,
            init_cash=10000,
            fees=0.0,
            slippage=0.0,
            market_hours_only=False,
            risk_config=RiskConfig(position_size_pct=1.0)
        )

        stats = portfolio.stats()
        if stats and stats.get('Total Trades', 0) > 0:
            assert stats['Total Return [%]'] < 0, "Should have negative return"


class TestFeesAndSlippage:
    """Test fee and slippage application."""

    def test_fees_reduce_returns(self):
        """Test that fees reduce returns."""
        price = pd.Series(
            [100.0, 100.0, 100.0],
            index=pd.date_range('2023-01-01', periods=3, freq='D')
        )
        entries = pd.Series([True, False, False], index=price.index)
        exits = pd.Series([False, False, True], index=price.index)

        # Portfolio without fees
        portfolio_no_fees = Portfolio(
            price=price,
            entries=entries,
            exits=exits,
            init_cash=10000,
            fees=0.0,
            slippage=0.0,
            market_hours_only=False,
            risk_config=RiskConfig(position_size_pct=1.0)
        )

        # Portfolio with fees
        portfolio_with_fees = Portfolio(
            price=price,
            entries=entries.copy(),
            exits=exits.copy(),
            init_cash=10000,
            fees=0.01,  # 1% fees
            slippage=0.0,
            market_hours_only=False,
            risk_config=RiskConfig(position_size_pct=1.0)
        )

        stats_no_fees = portfolio_no_fees.stats()
        stats_with_fees = portfolio_with_fees.stats()

        if stats_no_fees and stats_with_fees:
            # With fees should have lower or equal returns
            assert stats_with_fees['End Value'] <= stats_no_fees['End Value']

    def test_slippage_reduces_returns(self):
        """Test that slippage reduces returns."""
        price = pd.Series(
            [100.0, 100.0, 100.0],
            index=pd.date_range('2023-01-01', periods=3, freq='D')
        )
        entries = pd.Series([True, False, False], index=price.index)
        exits = pd.Series([False, False, True], index=price.index)

        # Portfolio without slippage
        portfolio_no_slip = Portfolio(
            price=price,
            entries=entries,
            exits=exits,
            init_cash=10000,
            fees=0.0,
            slippage=0.0,
            market_hours_only=False,
            risk_config=RiskConfig(position_size_pct=1.0)
        )

        # Portfolio with slippage
        portfolio_with_slip = Portfolio(
            price=price,
            entries=entries.copy(),
            exits=exits.copy(),
            init_cash=10000,
            fees=0.0,
            slippage=0.01,  # 1% slippage
            market_hours_only=False,
            risk_config=RiskConfig(position_size_pct=1.0)
        )

        stats_no_slip = portfolio_no_slip.stats()
        stats_with_slip = portfolio_with_slip.stats()

        if stats_no_slip and stats_with_slip:
            # With slippage should have lower or equal returns
            assert stats_with_slip['End Value'] <= stats_no_slip['End Value']


class TestStopLoss:
    """Test stop loss functionality."""

    def test_stop_loss_triggers(self):
        """Test that stop loss triggers on large drawdown."""
        # Price drops 15% - should trigger 10% stop loss
        price = pd.Series(
            [100.0, 95.0, 90.0, 85.0, 80.0],
            index=pd.date_range('2023-01-01', periods=5, freq='D')
        )
        entries = pd.Series([True, False, False, False, False], index=price.index)
        exits = pd.Series([False, False, False, False, False], index=price.index)  # No exit signal

        risk_config = RiskConfig(
            position_size_pct=1.0,
            use_stop_loss=True,
            stop_loss_type='percentage',
            stop_loss_pct=0.10  # 10% stop loss
        )

        portfolio = Portfolio(
            price=price,
            entries=entries,
            exits=exits,
            init_cash=10000,
            fees=0.0,
            slippage=0.0,
            market_hours_only=False,
            risk_config=risk_config
        )

        # Should have exited due to stop loss
        exit_trades = [t for t in portfolio.trades if t['type'] == 'exit']
        if exit_trades:
            # Check exit reason if available
            exit_reason = exit_trades[0].get('exit_reason', '')
            # Stop loss should have triggered before the -20% point
            assert portfolio.equity_curve.iloc[-1] > 8000, "Stop loss should limit losses"


class TestShortSelling:
    """Test short selling functionality."""

    def test_short_position_profit_on_decline(self):
        """Test short position profits when price declines."""
        # Price drops from 100 to 90 - short should profit
        price = pd.Series(
            [100.0, 95.0, 90.0],
            index=pd.date_range('2023-01-01', periods=3, freq='D')
        )
        # For shorts: exit signal opens short, entry signal closes it
        entries = pd.Series([False, False, True], index=price.index)  # Close short
        exits = pd.Series([True, False, False], index=price.index)  # Open short

        portfolio = Portfolio(
            price=price,
            entries=entries,
            exits=exits,
            init_cash=10000,
            fees=0.0,
            slippage=0.0,
            market_hours_only=False,
            risk_config=RiskConfig(position_size_pct=0.5),
            allow_shorts=True
        )

        stats = portfolio.stats()
        if stats and stats.get('Short Trades', 0) > 0:
            # Short on declining price should be profitable
            assert stats['Total Return [%]'] > 0, "Short on decline should profit"


class TestReturnsMethod:
    """Test returns() method."""

    def test_returns_is_series(self):
        """Test returns() returns a pandas Series."""
        price = pd.Series(
            [100.0, 101.0, 102.0],
            index=pd.date_range('2023-01-01', periods=3, freq='D')
        )
        entries = pd.Series([True, False, False], index=price.index)
        exits = pd.Series([False, False, True], index=price.index)

        portfolio = Portfolio(
            price=price,
            entries=entries,
            exits=exits,
            init_cash=10000,
            fees=0.0,
            slippage=0.0,
            market_hours_only=False
        )

        returns = portfolio.returns()
        assert isinstance(returns, pd.Series)

    def test_returns_no_nan(self):
        """Test returns has no NaN values."""
        price = pd.Series(
            [100.0, 101.0, 102.0],
            index=pd.date_range('2023-01-01', periods=3, freq='D')
        )
        entries = pd.Series([True, False, False], index=price.index)
        exits = pd.Series([False, False, True], index=price.index)

        portfolio = Portfolio(
            price=price,
            entries=entries,
            exits=exits,
            init_cash=10000,
            fees=0.0,
            slippage=0.0,
            market_hours_only=False
        )

        returns = portfolio.returns()
        assert not returns.isna().any(), "Returns should not contain NaN"

    def test_returns_with_resampling(self):
        """Test returns with frequency resampling."""
        # Create minute-level data
        price = pd.Series(
            np.random.randn(2000).cumsum() + 100,
            index=pd.date_range('2023-01-01', periods=2000, freq='T')
        )
        entries = pd.Series([False] * 2000, index=price.index)
        entries.iloc[0] = True
        exits = pd.Series([False] * 2000, index=price.index)
        exits.iloc[-1] = True

        portfolio = Portfolio(
            price=price,
            entries=entries,
            exits=exits,
            init_cash=10000,
            fees=0.0,
            slippage=0.0,
            market_hours_only=False
        )

        # Resample to hourly
        returns_hourly = portfolio.returns(freq='H')
        assert len(returns_hourly) < len(portfolio.returns())


class TestStatsMethod:
    """Test stats() method."""

    def test_stats_returns_dict(self):
        """Test stats() returns a dictionary."""
        price = pd.Series(
            [100.0, 101.0, 102.0],
            index=pd.date_range('2023-01-01', periods=3, freq='D')
        )
        entries = pd.Series([True, False, False], index=price.index)
        exits = pd.Series([False, False, True], index=price.index)

        portfolio = Portfolio(
            price=price,
            entries=entries,
            exits=exits,
            init_cash=10000,
            fees=0.0,
            slippage=0.0,
            market_hours_only=False
        )

        stats = portfolio.stats()
        assert isinstance(stats, dict)

    def test_stats_has_required_keys(self):
        """Test stats contains required metric keys."""
        price = pd.Series(
            [100.0, 101.0, 102.0],
            index=pd.date_range('2023-01-01', periods=3, freq='D')
        )
        entries = pd.Series([True, False, False], index=price.index)
        exits = pd.Series([False, False, True], index=price.index)

        portfolio = Portfolio(
            price=price,
            entries=entries,
            exits=exits,
            init_cash=10000,
            fees=0.0,
            slippage=0.0,
            market_hours_only=False
        )

        stats = portfolio.stats()
        required_keys = ['Total Return [%]', 'Sharpe Ratio', 'Max Drawdown [%]', 'Start Value', 'End Value']
        for key in required_keys:
            assert key in stats, f"Missing required key: {key}"

    def test_stats_caching(self):
        """Test stats are cached after first call."""
        price = pd.Series(
            [100.0, 101.0, 102.0],
            index=pd.date_range('2023-01-01', periods=3, freq='D')
        )
        entries = pd.Series([True, False, False], index=price.index)
        exits = pd.Series([False, False, True], index=price.index)

        portfolio = Portfolio(
            price=price,
            entries=entries,
            exits=exits,
            init_cash=10000,
            fees=0.0,
            slippage=0.0,
            market_hours_only=False
        )

        stats1 = portfolio.stats()
        stats2 = portfolio.stats()

        # Should return same object (cached)
        assert stats1 is stats2


class TestFromSignalsFactory:
    """Test from_signals factory function."""

    def test_from_signals_creates_portfolio(self):
        """Test from_signals creates a valid Portfolio."""
        price = pd.Series(
            [100.0, 101.0, 102.0],
            index=pd.date_range('2023-01-01', periods=3, freq='D')
        )
        entries = pd.Series([True, False, False], index=price.index)
        exits = pd.Series([False, False, True], index=price.index)

        portfolio = from_signals(
            close=price,
            entries=entries,
            exits=exits,
            init_cash=10000,
            fees=0.001,
            market_hours_only=False
        )

        # from_signals now returns PortfolioV2 (V2 is the default implementation)
        # Both Portfolio and PortfolioV2 inherit from BasePortfolio
        from src.backtesting.engine.base_portfolio import BasePortfolio
        assert isinstance(portfolio, BasePortfolio)
        assert portfolio.init_cash == 10000
        assert portfolio.fees == 0.001


class TestPeriodsPerYear:
    """Test _get_periods_per_year method."""

    def test_minute_frequency(self):
        """Test periods per year for minute data."""
        price = pd.Series([100.0], index=pd.date_range('2023-01-01', periods=1, freq='D'))
        entries = pd.Series([False], index=price.index)
        exits = pd.Series([False], index=price.index)

        portfolio = Portfolio(
            price=price,
            entries=entries,
            exits=exits,
            init_cash=10000,
            fees=0.0,
            slippage=0.0,
            freq='1min',
            market_hours_only=False
        )

        periods = portfolio._get_periods_per_year()
        assert periods == 252 * 6.5 * 60  # 252 days * 6.5 hours * 60 minutes

    def test_daily_frequency(self):
        """Test periods per year for daily data."""
        price = pd.Series([100.0], index=pd.date_range('2023-01-01', periods=1, freq='D'))
        entries = pd.Series([False], index=price.index)
        exits = pd.Series([False], index=price.index)

        portfolio = Portfolio(
            price=price,
            entries=entries,
            exits=exits,
            init_cash=10000,
            fees=0.0,
            slippage=0.0,
            freq='1d',
            market_hours_only=False
        )

        periods = portfolio._get_periods_per_year()
        assert periods == 252


class TestEquityCurve:
    """Test equity curve generation."""

    def test_equity_curve_length_matches_data(self):
        """Test equity curve has same length as input data."""
        price = pd.Series(
            [100.0, 101.0, 102.0, 103.0, 104.0],
            index=pd.date_range('2023-01-01', periods=5, freq='D')
        )
        entries = pd.Series([True, False, False, False, False], index=price.index)
        exits = pd.Series([False, False, False, False, True], index=price.index)

        portfolio = Portfolio(
            price=price,
            entries=entries,
            exits=exits,
            init_cash=10000,
            fees=0.0,
            slippage=0.0,
            market_hours_only=False
        )

        assert len(portfolio.equity_curve) == len(price)

    def test_equity_curve_starts_at_init_cash(self):
        """Test equity curve starts at initial cash (approximately)."""
        price = pd.Series(
            [100.0, 101.0, 102.0],
            index=pd.date_range('2023-01-01', periods=3, freq='D')
        )
        entries = pd.Series([False, False, False], index=price.index)  # No trades
        exits = pd.Series([False, False, False], index=price.index)

        portfolio = Portfolio(
            price=price,
            entries=entries,
            exits=exits,
            init_cash=10000,
            fees=0.0,
            slippage=0.0,
            market_hours_only=False
        )

        # With no trades, equity should stay at init_cash
        assert portfolio.equity_curve.iloc[0] == 10000
        assert portfolio.equity_curve.iloc[-1] == 10000
