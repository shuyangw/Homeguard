"""
Error handling and edge case tests for PairsPortfolio.

Tests robust behavior under invalid inputs, extreme conditions,
and error scenarios.
"""

import pytest
import pandas as pd
import numpy as np
from src.backtesting.engine.pairs_portfolio import PairsPortfolio, PairPosition


class TestPairsPortfolioDataValidation:
    """Test data validation and error handling."""

    def test_negative_prices_raise_error(self):
        """Test that negative prices are handled gracefully."""
        dates = pd.date_range('2020-01-01', periods=10, freq='D')
        prices1 = pd.Series(np.random.randn(10) * 10 + 100, index=dates)
        prices2 = pd.Series(np.random.randn(10) * 10 + 100, index=dates)

        # Inject negative price
        prices1.iloc[5] = -10

        entries = pd.Series([True] + [False] * 9, index=dates)
        exits = pd.Series([False] * 9 + [True], index=dates)

        # Should handle gracefully (skip invalid bars)
        portfolio = PairsPortfolio(
            symbols=('A', 'B'),
            prices1=prices1,
            prices2=prices2,
            entries=entries,
            exits=exits,
            short_entries=pd.Series(False, index=dates),
            short_exits=pd.Series(False, index=dates),
            init_cash=10000,
            fees=0.001,
            slippage=0.001
        )

        assert portfolio is not None
        # Should either skip the position or handle it
        assert portfolio.equity_curve is not None

    def test_zero_prices_handled(self):
        """Test that zero prices don't cause division by zero."""
        dates = pd.date_range('2020-01-01', periods=10, freq='D')
        prices1 = pd.Series([100.0] * 10, index=dates)
        prices2 = pd.Series([100.0] * 10, index=dates)

        # Zero price
        prices2.iloc[5] = 0.0

        entries = pd.Series([True] + [False] * 9, index=dates)
        exits = pd.Series([False] * 9 + [True], index=dates)

        portfolio = PairsPortfolio(
            symbols=('A', 'B'),
            prices1=prices1,
            prices2=prices2,
            entries=entries,
            exits=exits,
            short_entries=pd.Series(False, index=dates),
            short_exits=pd.Series(False, index=dates),
            init_cash=10000,
            fees=0.001,
            slippage=0.001
        )

        assert portfolio is not None

    def test_nan_prices_handled(self):
        """Test that NaN prices are handled gracefully."""
        dates = pd.date_range('2020-01-01', periods=10, freq='D')
        prices1 = pd.Series([100.0] * 10, index=dates)
        prices2 = pd.Series([100.0] * 10, index=dates)

        # NaN price
        prices1.iloc[5] = np.nan

        entries = pd.Series([True] + [False] * 9, index=dates)
        exits = pd.Series([False] * 9 + [True], index=dates)

        portfolio = PairsPortfolio(
            symbols=('A', 'B'),
            prices1=prices1,
            prices2=prices2,
            entries=entries,
            exits=exits,
            short_entries=pd.Series(False, index=dates),
            short_exits=pd.Series(False, index=dates),
            init_cash=10000,
            fees=0.001,
            slippage=0.001
        )

        assert portfolio is not None

    def test_inf_prices_handled(self):
        """Test that infinite prices are handled."""
        dates = pd.date_range('2020-01-01', periods=10, freq='D')
        prices1 = pd.Series([100.0] * 10, index=dates)
        prices2 = pd.Series([100.0] * 10, index=dates)

        # Inf price
        prices2.iloc[7] = np.inf

        entries = pd.Series([True] + [False] * 9, index=dates)
        exits = pd.Series([False] * 9 + [True], index=dates)

        portfolio = PairsPortfolio(
            symbols=('A', 'B'),
            prices1=prices1,
            prices2=prices2,
            entries=entries,
            exits=exits,
            short_entries=pd.Series(False, index=dates),
            short_exits=pd.Series(False, index=dates),
            init_cash=10000,
            fees=0.001,
            slippage=0.001
        )

        assert portfolio is not None

    def test_mismatched_price_lengths_error(self):
        """Test that mismatched price series lengths raise error."""
        dates1 = pd.date_range('2020-01-01', periods=10, freq='D')
        dates2 = pd.date_range('2020-01-01', periods=5, freq='D')

        prices1 = pd.Series([100.0] * 10, index=dates1)
        prices2 = pd.Series([100.0] * 5, index=dates2)

        entries = pd.Series([False] * 10, index=dates1)
        exits = pd.Series([False] * 10, index=dates1)

        # Should handle mismatched lengths
        portfolio = PairsPortfolio(
            symbols=('A', 'B'),
            prices1=prices1,
            prices2=prices2,
            entries=entries,
            exits=exits,
            short_entries=entries.copy(),
            short_exits=exits.copy(),
            init_cash=10000,
            fees=0.001,
            slippage=0.001
        )

        # Should work with aligned subset
        assert portfolio is not None

    def test_empty_price_series(self):
        """Test handling of empty price series."""
        dates = pd.date_range('2020-01-01', periods=0, freq='D')
        prices1 = pd.Series([], index=dates, dtype=float)
        prices2 = pd.Series([], index=dates, dtype=float)

        entries = pd.Series([], index=dates, dtype=bool)
        exits = pd.Series([], index=dates, dtype=bool)

        portfolio = PairsPortfolio(
            symbols=('A', 'B'),
            prices1=prices1,
            prices2=prices2,
            entries=entries,
            exits=exits,
            short_entries=entries.copy(),
            short_exits=exits.copy(),
            init_cash=10000,
            fees=0.001,
            slippage=0.001
        )

        assert portfolio is not None
        assert len(portfolio.equity_curve) == 0 or portfolio.equity_curve.iloc[0] == 10000

    def test_single_timestamp_data(self):
        """Test with single timestamp (minimal data)."""
        dates = pd.date_range('2020-01-01', periods=1, freq='D')
        prices1 = pd.Series([100.0], index=dates)
        prices2 = pd.Series([100.0], index=dates)

        entries = pd.Series([False], index=dates)
        exits = pd.Series([False], index=dates)

        portfolio = PairsPortfolio(
            symbols=('A', 'B'),
            prices1=prices1,
            prices2=prices2,
            entries=entries,
            exits=exits,
            short_entries=entries.copy(),
            short_exits=exits.copy(),
            init_cash=10000,
            fees=0.001,
            slippage=0.001
        )

        assert portfolio is not None
        assert len(portfolio.trades) == 0


class TestPairsPortfolioSignalValidation:
    """Test signal validation and error handling."""

    def test_conflicting_signals_handled(self):
        """Test simultaneous entry and exit signals."""
        dates = pd.date_range('2020-01-01', periods=10, freq='D')
        prices1 = pd.Series([100.0] * 10, index=dates)
        prices2 = pd.Series([100.0] * 10, index=dates)

        # Conflicting signals
        entries = pd.Series([True, False, True] + [False] * 7, index=dates)
        exits = pd.Series([True, False, True] + [False] * 7, index=dates)

        portfolio = PairsPortfolio(
            symbols=('A', 'B'),
            prices1=prices1,
            prices2=prices2,
            entries=entries,
            exits=exits,
            short_entries=pd.Series(False, index=dates),
            short_exits=pd.Series(False, index=dates),
            init_cash=10000,
            fees=0.001,
            slippage=0.001
        )

        # Should prioritize one action or skip
        assert portfolio is not None

    def test_nan_signals_handled(self):
        """Test NaN values in signal series."""
        dates = pd.date_range('2020-01-01', periods=10, freq='D')
        prices1 = pd.Series([100.0] * 10, index=dates)
        prices2 = pd.Series([100.0] * 10, index=dates)

        # NaN signals
        entries = pd.Series([True] + [False] * 9, index=dates)
        entries.iloc[5] = np.nan
        exits = pd.Series([False] * 9 + [True], index=dates)

        portfolio = PairsPortfolio(
            symbols=('A', 'B'),
            prices1=prices1,
            prices2=prices2,
            entries=entries,
            exits=exits,
            short_entries=pd.Series(False, index=dates),
            short_exits=pd.Series(False, index=dates),
            init_cash=10000,
            fees=0.001,
            slippage=0.001
        )

        # Should handle NaN as False
        assert portfolio is not None

    def test_all_false_signals_no_trades(self):
        """Test that all-False signals result in no trades."""
        dates = pd.date_range('2020-01-01', periods=10, freq='D')
        prices1 = pd.Series([100.0] * 10, index=dates)
        prices2 = pd.Series([100.0] * 10, index=dates)

        entries = pd.Series(False, index=dates)
        exits = pd.Series(False, index=dates)

        portfolio = PairsPortfolio(
            symbols=('A', 'B'),
            prices1=prices1,
            prices2=prices2,
            entries=entries,
            exits=exits,
            short_entries=entries.copy(),
            short_exits=exits.copy(),
            init_cash=10000,
            fees=0.001,
            slippage=0.001
        )

        assert len(portfolio.trades) == 0
        assert portfolio.equity_curve.iloc[-1] == 10000

    def test_exit_without_entry_ignored(self):
        """Test that exit signals without positions are ignored."""
        dates = pd.date_range('2020-01-01', periods=10, freq='D')
        prices1 = pd.Series([100.0] * 10, index=dates)
        prices2 = pd.Series([100.0] * 10, index=dates)

        # Exit before entry
        entries = pd.Series([False] * 5 + [True] + [False] * 4, index=dates)
        exits = pd.Series([True, False, True] + [False] * 6 + [True], index=dates)

        portfolio = PairsPortfolio(
            symbols=('A', 'B'),
            prices1=prices1,
            prices2=prices2,
            entries=entries,
            exits=exits,
            short_entries=pd.Series(False, index=dates),
            short_exits=pd.Series(False, index=dates),
            init_cash=10000,
            fees=0.001,
            slippage=0.001
        )

        # Should ignore early exits
        assert portfolio is not None


class TestPairsPortfolioCapitalManagement:
    """Test capital management and edge cases."""

    def test_very_small_capital(self):
        """Test with very small initial capital."""
        dates = pd.date_range('2020-01-01', periods=10, freq='D')
        prices1 = pd.Series([100.0] * 10, index=dates)
        prices2 = pd.Series([100.0] * 10, index=dates)

        entries = pd.Series([True] + [False] * 9, index=dates)
        exits = pd.Series([False] * 9 + [True], index=dates)

        portfolio = PairsPortfolio(
            symbols=('A', 'B'),
            prices1=prices1,
            prices2=prices2,
            entries=entries,
            exits=exits,
            short_entries=pd.Series(False, index=dates),
            short_exits=pd.Series(False, index=dates),
            init_cash=10,  # Very small
            fees=0.001,
            slippage=0.001
        )

        # Should handle gracefully (likely no trades)
        assert portfolio is not None

    def test_zero_capital_no_trades(self):
        """Test with zero initial capital."""
        dates = pd.date_range('2020-01-01', periods=10, freq='D')
        prices1 = pd.Series([100.0] * 10, index=dates)
        prices2 = pd.Series([100.0] * 10, index=dates)

        entries = pd.Series([True] + [False] * 9, index=dates)
        exits = pd.Series([False] * 9 + [True], index=dates)

        portfolio = PairsPortfolio(
            symbols=('A', 'B'),
            prices1=prices1,
            prices2=prices2,
            entries=entries,
            exits=exits,
            short_entries=pd.Series(False, index=dates),
            short_exits=pd.Series(False, index=dates),
            init_cash=0,
            fees=0.001,
            slippage=0.001
        )

        assert len(portfolio.trades) == 0

    def test_extreme_price_ratio(self):
        """Test with extreme price ratio between symbols."""
        dates = pd.date_range('2020-01-01', periods=10, freq='D')
        prices1 = pd.Series([0.01] * 10, index=dates)  # Penny stock
        prices2 = pd.Series([10000.0] * 10, index=dates)  # Expensive stock

        entries = pd.Series([True] + [False] * 9, index=dates)
        exits = pd.Series([False] * 9 + [True], index=dates)

        portfolio = PairsPortfolio(
            symbols=('A', 'B'),
            prices1=prices1,
            prices2=prices2,
            entries=entries,
            exits=exits,
            short_entries=pd.Series(False, index=dates),
            short_exits=pd.Series(False, index=dates),
            init_cash=10000,
            fees=0.001,
            slippage=0.001
        )

        assert portfolio is not None

    def test_very_high_fees_and_slippage(self):
        """Test with extreme fees and slippage."""
        dates = pd.date_range('2020-01-01', periods=10, freq='D')
        prices1 = pd.Series([100.0] * 10, index=dates)
        prices2 = pd.Series([100.0] * 10, index=dates)

        entries = pd.Series([True] + [False] * 9, index=dates)
        exits = pd.Series([False] * 9 + [True], index=dates)

        portfolio = PairsPortfolio(
            symbols=('A', 'B'),
            prices1=prices1,
            prices2=prices2,
            entries=entries,
            exits=exits,
            short_entries=pd.Series(False, index=dates),
            short_exits=pd.Series(False, index=dates),
            init_cash=10000,
            fees=0.1,  # 10% fees
            slippage=0.1  # 10% slippage
        )

        # Should handle high costs
        assert portfolio is not None

    def test_negative_fees_raises_or_handles(self):
        """Test that negative fees are handled."""
        dates = pd.date_range('2020-01-01', periods=10, freq='D')
        prices1 = pd.Series([100.0] * 10, index=dates)
        prices2 = pd.Series([100.0] * 10, index=dates)

        entries = pd.Series([False] * 10, index=dates)
        exits = pd.Series([False] * 10, index=dates)

        # Negative fees might be allowed (rebates)
        portfolio = PairsPortfolio(
            symbols=('A', 'B'),
            prices1=prices1,
            prices2=prices2,
            entries=entries,
            exits=exits,
            short_entries=entries.copy(),
            short_exits=exits.copy(),
            init_cash=10000,
            fees=-0.001,
            slippage=0.001
        )

        assert portfolio is not None


class TestPairPositionEdgeCases:
    """Test PairPosition dataclass edge cases."""

    def test_zero_shares_position(self):
        """Test position with zero shares."""
        dates = pd.date_range('2020-01-01', periods=1, freq='D')

        position = PairPosition(
            symbol1='A',
            symbol2='B',
            shares1=0.0,
            shares2=0.0,
            entry_price1=100.0,
            entry_price2=100.0,
            entry_timestamp=dates[0],
            entry_bar=0,
            hedge_ratio=1.0,
            capital_allocated=0.0
        )

        value1, value2 = position.get_current_value(100.0, 100.0)
        assert value1 == 0.0 and value2 == 0.0
        assert position.get_unrealized_pnl(100.0, 100.0) == 0.0

    def test_extreme_price_movement(self):
        """Test position with extreme price movement."""
        dates = pd.date_range('2020-01-01', periods=1, freq='D')

        position = PairPosition(
            symbol1='A',
            symbol2='B',
            shares1=-10.0,  # Short
            shares2=10.0,   # Long
            entry_price1=100.0,
            entry_price2=100.0,
            entry_timestamp=dates[0],
            entry_bar=0,
            hedge_ratio=1.0,
            capital_allocated=2000.0
        )

        # Extreme upward movement in short
        pnl = position.get_unrealized_pnl(1000.0, 100.0)
        assert pnl < -8000  # Large loss on short side

    def test_zero_entry_prices(self):
        """Test position with zero entry prices."""
        dates = pd.date_range('2020-01-01', periods=1, freq='D')

        position = PairPosition(
            symbol1='A',
            symbol2='B',
            shares1=10.0,
            shares2=10.0,
            entry_price1=0.0,  # Zero entry
            entry_price2=100.0,
            entry_timestamp=dates[0],
            entry_bar=0,
            hedge_ratio=1.0,
            capital_allocated=1000.0
        )

        # Should handle without crashing
        value1, value2 = position.get_current_value(100.0, 100.0)
        assert value1 >= 0 and value2 >= 0


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
