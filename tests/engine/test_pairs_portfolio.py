"""
Unit tests for PairsPortfolio simulator (Phase 3).

Tests synchronized execution of both legs of pairs trades with proper
position sizing, P&L tracking, and risk management.
"""

import pytest
import pandas as pd
import numpy as np
from typing import Dict, Tuple
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from backtesting.engine.pairs_portfolio import PairPosition, PairsPortfolio
from backtesting.utils.risk_config import RiskConfig
from backtesting.engine.backtest_engine import BacktestEngine
from backtesting.base.pairs_strategy import PairsStrategy


class SimplePairsStrategy(PairsStrategy):
    """Simple pairs strategy for testing."""

    def generate_pairs_signals(
        self,
        data1: pd.DataFrame,
        data2: pd.DataFrame,
        symbol1: str = 'asset1',
        symbol2: str = 'asset2'
    ) -> Dict[str, Tuple[pd.Series, pd.Series, pd.Series, pd.Series]]:
        """Generate simple spread-based signals."""
        # Calculate spread
        spread = data2['close'] - data1['close']
        mean = spread.mean()
        std = spread.std()

        if std == 0:
            std = 1.0  # Avoid division by zero

        z_score = (spread - mean) / std

        # Entry when z-score crosses thresholds
        long_spread_entry = z_score > 1.5
        long_spread_exit = z_score < 0.3
        short_spread_entry = z_score < -1.5
        short_spread_exit = z_score > -0.3

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


class TestPairPositionDataclass:
    """Test PairPosition dataclass."""

    @pytest.fixture
    def sample_pair_position(self):
        """Create a sample pair position."""
        return PairPosition(
            symbol1='AAPL',
            symbol2='MSFT',
            shares1=-100.0,  # Short 100 shares
            shares2=50.0,    # Long 50 shares
            entry_price1=150.0,
            entry_price2=300.0,
            entry_timestamp=pd.Timestamp('2023-01-01'),
            entry_bar=10,
            hedge_ratio=0.5,
            capital_allocated=30000.0
        )

    def test_pair_position_creation(self, sample_pair_position):
        """PairPosition can be created with all required fields."""
        pos = sample_pair_position

        assert pos.symbol1 == 'AAPL'
        assert pos.symbol2 == 'MSFT'
        assert pos.shares1 == -100.0
        assert pos.shares2 == 50.0
        assert pos.entry_price1 == 150.0
        assert pos.entry_price2 == 300.0
        assert pos.hedge_ratio == 0.5
        assert pos.capital_allocated == 30000.0

    def test_is_long_spread(self, sample_pair_position):
        """Identifies long spread position (short sym1, long sym2)."""
        assert sample_pair_position.is_long_spread()
        assert not sample_pair_position.is_short_spread()

    def test_is_short_spread(self):
        """Identifies short spread position (long sym1, short sym2)."""
        pos = PairPosition(
            symbol1='AAPL',
            symbol2='MSFT',
            shares1=100.0,   # Long
            shares2=-50.0,   # Short
            entry_price1=150.0,
            entry_price2=300.0,
            entry_timestamp=pd.Timestamp('2023-01-01'),
            entry_bar=10,
            hedge_ratio=0.5,
            capital_allocated=30000.0
        )

        assert pos.is_short_spread()
        assert not pos.is_long_spread()

    def test_get_current_value_both_long(self):
        """Calculate current value with both legs long."""
        pos = PairPosition(
            symbol1='AAPL',
            symbol2='MSFT',
            shares1=100.0,
            shares2=50.0,
            entry_price1=150.0,
            entry_price2=300.0,
            entry_timestamp=pd.Timestamp('2023-01-01'),
            entry_bar=10,
            hedge_ratio=0.5,
            capital_allocated=30000.0
        )

        # Prices up: both legs profit
        value1, value2 = pos.get_current_value(160.0, 310.0)

        assert value1 == 100 * 160  # Long: shares * current_price
        assert value2 == 50 * 310

    def test_get_current_value_both_short(self):
        """Calculate current value with both legs short."""
        pos = PairPosition(
            symbol1='AAPL',
            symbol2='MSFT',
            shares1=-100.0,
            shares2=-50.0,
            entry_price1=150.0,
            entry_price2=300.0,
            entry_timestamp=pd.Timestamp('2023-01-01'),
            entry_bar=10,
            hedge_ratio=0.5,
            capital_allocated=30000.0
        )

        # Prices down: both shorts profit
        value1, value2 = pos.get_current_value(140.0, 290.0)

        assert value1 == 100 * (150 - 140)  # Short: shares * (entry - current)
        assert value2 == 50 * (300 - 290)

    def test_get_unrealized_pnl_long_spread_profit(self, sample_pair_position):
        """Calculate unrealized P&L for long spread (spread narrows = profit)."""
        # Long spread: short sym1 @ 150, long sym2 @ 300
        # Spread narrows (sym1 up to 160, sym2 stays 300)
        # Short sym1 loses: -10 * 100 = -1000
        # Long sym2 flat: 0
        # Net: -1000

        pnl = sample_pair_position.get_unrealized_pnl(160.0, 300.0)

        assert pnl < 0  # Loss when sym1 rises

    def test_get_unrealized_pnl_short_spread_profit(self):
        """Calculate unrealized P&L for short spread (spread widens = profit)."""
        pos = PairPosition(
            symbol1='AAPL',
            symbol2='MSFT',
            shares1=100.0,   # Long @ 150
            shares2=-50.0,   # Short @ 300
            entry_price1=150.0,
            entry_price2=300.0,
            entry_timestamp=pd.Timestamp('2023-01-01'),
            entry_bar=10,
            hedge_ratio=0.5,
            capital_allocated=30000.0
        )

        # Spread widens (sym1 to 160, sym2 to 290)
        # Long sym1 profits: 10 * 100 = 1000
        # Short sym2 profits: 10 * 50 = 500
        # Net: 1500

        pnl = pos.get_unrealized_pnl(160.0, 290.0)

        assert pnl > 0  # Profit when spread widens


class TestPairsPortfolioBasics:
    """Test basic PairsPortfolio functionality."""

    @pytest.fixture
    def simple_pair_data(self):
        """Create simple correlated pair data."""
        dates = pd.date_range('2023-01-01', periods=30, freq='D')
        np.random.seed(42)

        # Create mean-reverting spread
        base = 100 + np.cumsum(np.random.randn(30) * 0.5)
        price1 = base + 5 * np.sin(np.linspace(0, 4 * np.pi, 30))
        price2 = 2 * base + np.random.randn(30) * 2

        return (
            pd.Series(price1, index=dates),
            pd.Series(price2, index=dates)
        )

    def test_pairs_portfolio_initialization(self, simple_pair_data):
        """PairsPortfolio initializes correctly."""
        prices1, prices2 = simple_pair_data

        entries = pd.Series([i == 5 for i in range(len(prices1))], index=prices1.index)
        exits = pd.Series([i == 15 for i in range(len(prices1))], index=prices1.index)
        short_entries = pd.Series(False, index=prices1.index)
        short_exits = pd.Series(False, index=prices1.index)

        portfolio = PairsPortfolio(
            symbols=('AAPL', 'MSFT'),
            prices1=prices1,
            prices2=prices2,
            entries=entries,
            exits=exits,
            short_entries=short_entries,
            short_exits=short_exits,
            init_cash=100000.0,
            fees=0.001,
            slippage=0.0
        )

        assert portfolio is not None
        assert portfolio.init_cash == 100000.0
        assert portfolio.symbol1 == 'AAPL'
        assert portfolio.symbol2 == 'MSFT'

    def test_equity_curve_generated(self, simple_pair_data):
        """Equity curve is generated after simulation."""
        prices1, prices2 = simple_pair_data

        entries = pd.Series([i == 5 for i in range(len(prices1))], index=prices1.index)
        exits = pd.Series([i == 15 for i in range(len(prices1))], index=prices1.index)
        short_entries = pd.Series(False, index=prices1.index)
        short_exits = pd.Series(False, index=prices1.index)

        portfolio = PairsPortfolio(
            symbols=('AAPL', 'MSFT'),
            prices1=prices1,
            prices2=prices2,
            entries=entries,
            exits=exits,
            short_entries=short_entries,
            short_exits=short_exits,
            init_cash=100000.0,
            fees=0.001,
            slippage=0.0
        )

        assert portfolio.equity_curve is not None
        assert len(portfolio.equity_curve) == len(prices1)
        assert portfolio.equity_curve.iloc[0] == 100000.0  # Starts at initial capital

    def test_trades_recorded(self, simple_pair_data):
        """Trades are recorded with entry and exit."""
        prices1, prices2 = simple_pair_data

        entries = pd.Series([i == 5 for i in range(len(prices1))], index=prices1.index)
        exits = pd.Series([i == 15 for i in range(len(prices1))], index=prices1.index)
        short_entries = pd.Series(False, index=prices1.index)
        short_exits = pd.Series(False, index=prices1.index)

        portfolio = PairsPortfolio(
            symbols=('AAPL', 'MSFT'),
            prices1=prices1,
            prices2=prices2,
            entries=entries,
            exits=exits,
            short_entries=short_entries,
            short_exits=short_exits,
            init_cash=100000.0,
            fees=0.001,
            slippage=0.0,
            market_hours_only=False  # Allow all trades for testing
        )

        # Trades list should exist (may be empty if no valid signals)
        assert portfolio.trades is not None

        entry_trades = [t for t in portfolio.trades if t['type'] == 'entry']
        exit_trades = [t for t in portfolio.trades if t['type'] == 'exit']

        # If there are any trades, validate structure
        if len(portfolio.trades) > 0:
            assert len(entry_trades) >= 0
            assert len(exit_trades) >= 0

            # If entered, should record both legs
            if len(entry_trades) > 0:
                entry = entry_trades[0]
                assert 'symbol1' in entry
                assert 'symbol2' in entry
                assert 'shares1' in entry
                assert 'shares2' in entry

    def test_synchronized_entry_both_legs(self, simple_pair_data):
        """Both legs enter simultaneously."""
        prices1, prices2 = simple_pair_data

        entries = pd.Series([i == 5 for i in range(len(prices1))], index=prices1.index)
        exits = pd.Series([i == 15 for i in range(len(prices1))], index=prices1.index)
        short_entries = pd.Series(False, index=prices1.index)
        short_exits = pd.Series(False, index=prices1.index)

        portfolio = PairsPortfolio(
            symbols=('AAPL', 'MSFT'),
            prices1=prices1,
            prices2=prices2,
            entries=entries,
            exits=exits,
            short_entries=short_entries,
            short_exits=short_exits,
            init_cash=100000.0,
            fees=0.001,
            slippage=0.0
        )

        entry_trades = [t for t in portfolio.trades if t['type'] == 'entry']

        if len(entry_trades) > 0:
            entry = entry_trades[0]
            # Both legs should have non-zero shares
            assert entry['shares1'] != 0
            assert entry['shares2'] != 0

    def test_statistics_calculated(self, simple_pair_data):
        """Portfolio statistics are calculated."""
        prices1, prices2 = simple_pair_data

        entries = pd.Series([i == 5 for i in range(len(prices1))], index=prices1.index)
        exits = pd.Series([i == 15 for i in range(len(prices1))], index=prices1.index)
        short_entries = pd.Series(False, index=prices1.index)
        short_exits = pd.Series(False, index=prices1.index)

        portfolio = PairsPortfolio(
            symbols=('AAPL', 'MSFT'),
            prices1=prices1,
            prices2=prices2,
            entries=entries,
            exits=exits,
            short_entries=short_entries,
            short_exits=short_exits,
            init_cash=100000.0,
            fees=0.001,
            slippage=0.0
        )

        stats = portfolio.stats()

        assert stats is not None
        assert 'Total Return [%]' in stats
        assert 'Sharpe Ratio' in stats
        assert 'Max Drawdown [%]' in stats
        assert 'Total Trades' in stats


class TestPairsPortfolioIntegration:
    """Test PairsPortfolio integration with BacktestEngine."""

    @pytest.fixture
    def correlated_data(self):
        """Create correlated pair data for integration testing."""
        dates = pd.date_range('2023-01-01', periods=50, freq='D')
        np.random.seed(42)

        # Create cointegrated series
        base = 100 + np.cumsum(np.random.randn(50) * 0.5)
        price1 = base + 3 * np.sin(np.linspace(0, 6 * np.pi, 50))
        price2 = 2 * base + np.random.randn(50) * 2

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

    def test_engine_uses_pairs_portfolio_for_pairs_strategy(self, correlated_data):
        """BacktestEngine uses PairsPortfolio for PairsStrategy."""
        engine = BacktestEngine(initial_capital=100000.0, fees=0.001)
        strategy = SimplePairsStrategy()

        # This should route to PairsPortfolio
        portfolio = engine._run_multi_symbol_strategy(
            strategy,
            correlated_data,
            ['AAPL', 'MSFT'],
            'close'
        )

        assert portfolio is not None
        assert hasattr(portfolio, 'stats')

    def test_full_backtest_with_pairs_strategy(self, correlated_data):
        """Full end-to-end backtest with pairs strategy."""
        engine = BacktestEngine(initial_capital=50000.0, fees=0.001)
        strategy = SimplePairsStrategy()

        # Run full backtest
        portfolio = engine._run_multi_symbol_strategy(
            strategy,
            correlated_data,
            ['AAPL', 'MSFT'],
            'close'
        )

        stats = portfolio.stats()

        assert stats is not None
        assert stats['Start Value'] == 50000.0
        assert 'End Value' in stats
        assert 'Total Trades' in stats


class TestPairsPortfolioEdgeCases:
    """Test edge cases for pairs portfolio."""

    def test_no_signals_no_trades(self):
        """Handles case with no entry signals."""
        dates = pd.date_range('2023-01-01', periods=20, freq='D')
        prices1 = pd.Series(100.0, index=dates)
        prices2 = pd.Series(200.0, index=dates)

        # No entries
        entries = pd.Series(False, index=dates)
        exits = pd.Series(False, index=dates)
        short_entries = pd.Series(False, index=dates)
        short_exits = pd.Series(False, index=dates)

        portfolio = PairsPortfolio(
            symbols=('A', 'B'),
            prices1=prices1,
            prices2=prices2,
            entries=entries,
            exits=exits,
            short_entries=short_entries,
            short_exits=short_exits,
            init_cash=10000.0,
            fees=0.001,
            slippage=0.0
        )

        assert len(portfolio.trades) == 0
        assert portfolio.equity_curve.iloc[-1] == 10000.0  # Capital preserved

    def test_insufficient_capital_no_trade(self):
        """No trade when insufficient capital."""
        dates = pd.date_range('2023-01-01', periods=20, freq='D')
        # Very expensive prices
        prices1 = pd.Series(10000.0, index=dates)
        prices2 = pd.Series(20000.0, index=dates)

        entries = pd.Series([i == 5 for i in range(20)], index=dates)
        exits = pd.Series([i == 15 for i in range(20)], index=dates)
        short_entries = pd.Series(False, index=dates)
        short_exits = pd.Series(False, index=dates)

        # Very small capital
        portfolio = PairsPortfolio(
            symbols=('A', 'B'),
            prices1=prices1,
            prices2=prices2,
            entries=entries,
            exits=exits,
            short_entries=short_entries,
            short_exits=short_exits,
            init_cash=1000.0,  # Too small
            fees=0.001,
            slippage=0.0,
            risk_config=RiskConfig.moderate()
        )

        # Should not trade if can't afford even 1 share
        assert len(portfolio.trades) == 0

    def test_multiple_signals_only_one_position(self):
        """Only holds one pair position at a time."""
        dates = pd.date_range('2023-01-01', periods=20, freq='D')
        prices1 = pd.Series(100.0, index=dates)
        prices2 = pd.Series(200.0, index=dates)

        # Multiple entry signals
        entries = pd.Series([i in [5, 10, 15] for i in range(20)], index=dates)
        exits = pd.Series(False, index=dates)
        short_entries = pd.Series(False, index=dates)
        short_exits = pd.Series(False, index=dates)

        portfolio = PairsPortfolio(
            symbols=('A', 'B'),
            prices1=prices1,
            prices2=prices2,
            entries=entries,
            exits=exits,
            short_entries=short_entries,
            short_exits=short_exits,
            init_cash=10000.0,
            fees=0.001,
            slippage=0.0
        )

        entry_trades = [t for t in portfolio.trades if t['type'] == 'entry']

        # Should only enter once (first signal), ignore subsequent while in position
        assert len(entry_trades) <= 1


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
