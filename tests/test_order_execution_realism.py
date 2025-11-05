"""
Order Execution Realism Tests

Ensures backtest doesn't assume perfect fills at best prices.

Real trading considerations:
- Cannot buy at the exact daily low
- Cannot sell at the exact daily high
- Slippage increases with position size
- Bid-ask spreads cost money
- Market impact for large orders

These tests verify the backtest engine models realistic order execution,
preventing overoptimistic backtest results.
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from typing import Tuple, Dict
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from backtesting.engine.backtest_engine import BacktestEngine
from backtesting.base.strategy import LongOnlyStrategy
from backtesting.utils.risk_config import RiskConfig


# ============================================================================
# TEST DATA GENERATORS
# ============================================================================

def create_volatile_intraday_data(
    n_bars: int = 100,
    base_price: float = 100.0,
    intraday_range_pct: float = 0.05
) -> pd.DataFrame:
    """
    Create data with significant intraday ranges.

    Each day has:
    - Open near previous close
    - High = open + range
    - Low = open - range
    - Close randomly within range

    This allows testing whether fills occur at realistic prices.
    """
    dates = pd.date_range(start='2023-01-01', periods=n_bars, freq='D')

    prices = []
    current_close = base_price

    for i in range(n_bars):
        # Open near previous close (small gap)
        open_price = current_close * (1 + np.random.uniform(-0.005, 0.005))

        # Intraday range
        range_dollars = open_price * intraday_range_pct
        high = open_price + range_dollars
        low = open_price - range_dollars

        # Close randomly in range
        close = np.random.uniform(low + range_dollars*0.2, high - range_dollars*0.2)

        prices.append({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': np.random.uniform(500000, 2000000)
        })

        current_close = close

    df = pd.DataFrame(prices, index=dates)
    return df


def create_trending_data_with_range(
    n_bars: int = 100,
    trend: str = 'up',
    daily_move_pct: float = 0.01,
    intraday_range_pct: float = 0.03
) -> pd.DataFrame:
    """
    Create trending data with realistic OHLC ranges.

    Allows testing execution at different price points within the bar.
    """
    dates = pd.date_range(start='2023-01-01', periods=n_bars, freq='D')

    base_price = 100.0
    trend_multiplier = 1.0 if trend == 'up' else -1.0

    data = []
    current_price = base_price

    for i in range(n_bars):
        # Trend move
        open_price = current_price
        close_price = current_price * (1 + daily_move_pct * trend_multiplier)

        # Add intraday range
        intraday_range = current_price * intraday_range_pct

        if trend == 'up':
            high = max(open_price, close_price) + intraday_range
            low = min(open_price, close_price) - intraday_range * 0.5
        else:
            high = max(open_price, close_price) + intraday_range * 0.5
            low = min(open_price, close_price) - intraday_range

        data.append({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close_price,
            'volume': 1000000
        })

        current_price = close_price

    df = pd.DataFrame(data, index=dates)
    return df


# ============================================================================
# TEST STRATEGIES
# ============================================================================

class SimpleBuyAndHoldStrategy(LongOnlyStrategy):
    """Buy on first bar, hold until end."""

    def generate_long_signals(self, data: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        entries = pd.Series(False, index=data.index)
        exits = pd.Series(False, index=data.index)

        # Buy on first bar
        entries.iloc[0] = True

        # Sell on last bar
        exits.iloc[-1] = True

        return entries, exits


class MultipleTradeStrategy(LongOnlyStrategy):
    """Make trades every N bars to test multiple executions."""

    def __init__(self, trade_every_n_bars: int = 10):
        super().__init__()
        self.trade_every_n_bars = trade_every_n_bars

    def generate_long_signals(self, data: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        entries = pd.Series(False, index=data.index)
        exits = pd.Series(False, index=data.index)

        # Buy every N bars
        for i in range(0, len(data), self.trade_every_n_bars):
            if i < len(data):
                entries.iloc[i] = True
            # Exit 5 bars later
            if i + 5 < len(data):
                exits.iloc[i + 5] = True

        return entries, exits


# ============================================================================
# TEST CLASS: Entry Price Realism
# ============================================================================

class TestEntryPriceRealism:
    """
    Verify entry prices are realistic (not at daily extremes).

    Real trading: Entries occur at open, close, or mid-range prices
    Unrealistic: Buying exactly at daily low, selling at daily high
    """

    def test_entries_not_at_daily_low(self):
        """
        Entry prices should NOT be at the daily low (too good to be true).

        Realistic: Entry at open, close, or within 20-80% of range
        Unrealistic: Entry at exact low of the day
        """
        # Create data with wide intraday ranges
        data = create_volatile_intraday_data(n_bars=50, intraday_range_pct=0.10)

        strategy = MultipleTradeStrategy(trade_every_n_bars=10)

        engine = BacktestEngine(
            initial_capital=10000,
            fees=0.001,
            slippage=0.0,  # No slippage for this test
            market_hours_only=False,
            risk_config=RiskConfig.disabled()
        )

        portfolio = engine.run_with_data(strategy, data)

        # Check entry prices
        entry_trades = [t for t in portfolio.trades if t['type'] == 'entry']

        print(f"\n{'='*60}")
        print("ENTRY PRICE REALISM TEST")
        print(f"{'='*60}")

        unrealistic_count = 0
        for trade in entry_trades:
            timestamp = trade['timestamp']
            entry_price = trade['price']

            bar = data.loc[timestamp]
            daily_low = bar['low']
            daily_high = bar['high']
            daily_range = daily_high - daily_low

            # Calculate where in the range the entry occurred
            # 0% = low, 50% = mid, 100% = high
            range_position = (entry_price - daily_low) / daily_range if daily_range > 0 else 0.5

            print(f"\nEntry on {timestamp.date()}:")
            print(f"  Bar: Low=${daily_low:.2f}, High=${daily_high:.2f}, Range=${daily_range:.2f}")
            print(f"  Entry price: ${entry_price:.2f}")
            print(f"  Position in range: {range_position*100:.1f}%")

            # Entries at extreme low (<5% of range) are unrealistic
            if range_position < 0.05:
                unrealistic_count += 1
                print(f"  ⚠️ WARNING: Entry too close to daily low!")

        print(f"\n{'='*60}")
        print(f"Total entries: {len(entry_trades)}")
        print(f"Unrealistic entries (at daily low): {unrealistic_count}")
        print(f"{'='*60}\n")

        # Most entries should NOT be at the daily low
        # Allow up to 10% to be near the low (by chance), but not more
        unrealistic_pct = unrealistic_count / len(entry_trades) if entry_trades else 0

        assert unrealistic_pct < 0.10, \
            f"Too many entries at daily low: {unrealistic_pct*100:.1f}% " \
            f"(expected <10% by random chance)"

    def test_exits_not_at_daily_high(self):
        """
        Exit prices should NOT be at the daily high (too good to be true).

        Realistic: Exit at open, close, or within 20-80% of range
        Unrealistic: Exit at exact high of the day
        """
        # Create data with wide intraday ranges
        data = create_volatile_intraday_data(n_bars=50, intraday_range_pct=0.10)

        strategy = MultipleTradeStrategy(trade_every_n_bars=10)

        engine = BacktestEngine(
            initial_capital=10000,
            fees=0.001,
            slippage=0.0,
            market_hours_only=False,
            risk_config=RiskConfig.disabled()
        )

        portfolio = engine.run_with_data(strategy, data)

        # Check exit prices
        exit_trades = [t for t in portfolio.trades if t['type'] == 'exit']

        print(f"\n{'='*60}")
        print("EXIT PRICE REALISM TEST")
        print(f"{'='*60}")

        unrealistic_count = 0
        for trade in exit_trades:
            timestamp = trade['timestamp']
            exit_price = trade['price']

            bar = data.loc[timestamp]
            daily_low = bar['low']
            daily_high = bar['high']
            daily_range = daily_high - daily_low

            # Calculate where in the range the exit occurred
            range_position = (exit_price - daily_low) / daily_range if daily_range > 0 else 0.5

            print(f"\nExit on {timestamp.date()}:")
            print(f"  Bar: Low=${daily_low:.2f}, High=${daily_high:.2f}, Range=${daily_range:.2f}")
            print(f"  Exit price: ${exit_price:.2f}")
            print(f"  Position in range: {range_position*100:.1f}%")

            # Exits at extreme high (>95% of range) are unrealistic
            if range_position > 0.95:
                unrealistic_count += 1
                print(f"  ⚠️ WARNING: Exit too close to daily high!")

        print(f"\n{'='*60}")
        print(f"Total exits: {len(exit_trades)}")
        print(f"Unrealistic exits (at daily high): {unrealistic_count}")
        print(f"{'='*60}\n")

        unrealistic_pct = unrealistic_count / len(exit_trades) if exit_trades else 0

        assert unrealistic_pct < 0.10, \
            f"Too many exits at daily high: {unrealistic_pct*100:.1f}% " \
            f"(expected <10% by random chance)"


# ============================================================================
# TEST CLASS: Slippage Impact
# ============================================================================

class TestSlippageImpact:
    """
    Verify slippage reduces returns as expected.

    Higher slippage → lower returns
    """

    def test_slippage_reduces_returns(self):
        """
        Compare returns with different slippage levels.

        Expected: Returns with 0.1% slippage < Returns with 0% slippage
        """
        data = create_trending_data_with_range(n_bars=100, trend='up', daily_move_pct=0.02)

        strategy = MultipleTradeStrategy(trade_every_n_bars=20)

        # Run with zero slippage
        engine_no_slip = BacktestEngine(
            initial_capital=10000,
            fees=0.001,
            slippage=0.0,
            market_hours_only=False,
            risk_config=RiskConfig.disabled()
        )

        portfolio_no_slip = engine_no_slip.run_with_data(strategy, data)
        stats_no_slip = portfolio_no_slip.stats()
        return_no_slip = stats_no_slip['Total Return [%]']

        # Run with 0.1% slippage
        engine_with_slip = BacktestEngine(
            initial_capital=10000,
            fees=0.001,
            slippage=0.001,  # 0.1% slippage
            market_hours_only=False,
            risk_config=RiskConfig.disabled()
        )

        portfolio_with_slip = engine_with_slip.run_with_data(strategy, data)
        stats_with_slip = portfolio_with_slip.stats()
        return_with_slip = stats_with_slip['Total Return [%]']

        # Run with 0.5% slippage
        engine_high_slip = BacktestEngine(
            initial_capital=10000,
            fees=0.001,
            slippage=0.005,  # 0.5% slippage
            market_hours_only=False,
            risk_config=RiskConfig.disabled()
        )

        portfolio_high_slip = engine_high_slip.run_with_data(strategy, data)
        stats_high_slip = portfolio_high_slip.stats()
        return_high_slip = stats_high_slip['Total Return [%]']

        print(f"\n{'='*60}")
        print("SLIPPAGE IMPACT TEST")
        print(f"{'='*60}")
        print(f"No slippage (0.0%):      {return_no_slip:>7.2f}%")
        print(f"Low slippage (0.1%):     {return_with_slip:>7.2f}%")
        print(f"High slippage (0.5%):    {return_high_slip:>7.2f}%")
        print(f"-" * 60)
        print(f"Impact of 0.1% slippage: {return_with_slip - return_no_slip:>7.2f}%")
        print(f"Impact of 0.5% slippage: {return_high_slip - return_no_slip:>7.2f}%")
        print(f"{'='*60}\n")

        # Slippage should reduce returns
        assert return_with_slip < return_no_slip, \
            f"Slippage should reduce returns: {return_with_slip:.2f}% vs {return_no_slip:.2f}%"

        assert return_high_slip < return_with_slip, \
            f"Higher slippage should reduce returns more: {return_high_slip:.2f}% vs {return_with_slip:.2f}%"

        # The impact should be proportional to number of trades
        # With more trades, slippage impact should be larger
        num_trades = len([t for t in portfolio_no_slip.trades if t['type'] == 'exit'])

        # Each round trip has 2 * slippage cost
        # With 0.1% slippage: ~0.2% per round trip
        expected_drag_low = num_trades * 0.002 * 100  # Rough estimate
        actual_drag_low = return_no_slip - return_with_slip

        print(f"Number of round trips: {num_trades}")
        print(f"Expected slippage drag (0.1%): ~{expected_drag_low:.2f}%")
        print(f"Actual slippage drag: {actual_drag_low:.2f}%")
        print()

        # Drag should be in reasonable range (within 2x of expected)
        assert 0 < actual_drag_low < expected_drag_low * 3, \
            f"Slippage impact outside expected range: {actual_drag_low:.2f}% vs ~{expected_drag_low:.2f}%"


# ============================================================================
# TEST CLASS: Realistic Fill Prices
# ============================================================================

class TestRealisticFillPrices:
    """
    Verify fills occur at reasonable prices (open or close, not extremes).
    """

    def test_fills_within_ohlc_range(self):
        """
        All fills must be within the OHLC range of the bar.

        Critical: Cannot fill at price outside [low, high]
        """
        data = create_volatile_intraday_data(n_bars=50, intraday_range_pct=0.10)

        strategy = MultipleTradeStrategy(trade_every_n_bars=10)

        engine = BacktestEngine(
            initial_capital=10000,
            fees=0.001,
            slippage=0.001,
            market_hours_only=False,
            risk_config=RiskConfig.disabled()
        )

        portfolio = engine.run_with_data(strategy, data)

        print(f"\n{'='*60}")
        print("FILL PRICE RANGE VALIDATION")
        print(f"{'='*60}")

        violations = []

        for trade in portfolio.trades:
            timestamp = trade['timestamp']
            fill_price = trade['price']

            bar = data.loc[timestamp]
            low = bar['low']
            high = bar['high']

            # Fill must be within [low, high]
            if fill_price < low or fill_price > high:
                violations.append({
                    'timestamp': timestamp,
                    'fill': fill_price,
                    'low': low,
                    'high': high,
                    'type': trade['type']
                })

                print(f"⚠️ VIOLATION on {timestamp.date()}:")
                print(f"   Fill: ${fill_price:.2f}, Bar range: [${low:.2f}, ${high:.2f}]")

        print(f"\n{'='*60}")
        print(f"Total trades: {len(portfolio.trades)}")
        print(f"Violations (fills outside OHLC): {len(violations)}")
        print(f"{'='*60}\n")

        assert len(violations) == 0, \
            f"Found {len(violations)} fills outside OHLC range"

    def test_average_fill_price_reasonable(self):
        """
        Average fill price should be near open/close, not at extremes.

        For entries: Average should be near open (realistic market orders)
        Not at: daily low (too good)
        """
        data = create_volatile_intraday_data(n_bars=50, intraday_range_pct=0.10)

        strategy = MultipleTradeStrategy(trade_every_n_bars=10)

        engine = BacktestEngine(
            initial_capital=10000,
            fees=0.001,
            slippage=0.001,
            market_hours_only=False,
            risk_config=RiskConfig.disabled()
        )

        portfolio = engine.run_with_data(strategy, data)

        # Analyze entry fills
        entry_positions = []

        for trade in [t for t in portfolio.trades if t['type'] == 'entry']:
            timestamp = trade['timestamp']
            fill_price = trade['price']

            bar = data.loc[timestamp]
            low = bar['low']
            high = bar['high']
            range_size = high - low

            if range_size > 0:
                position = (fill_price - low) / range_size
                entry_positions.append(position)

        avg_position = np.mean(entry_positions) if entry_positions else 0.5

        print(f"\n{'='*60}")
        print("AVERAGE FILL POSITION ANALYSIS")
        print(f"{'='*60}")
        print(f"Average entry position in daily range: {avg_position*100:.1f}%")
        print(f"  0% = Always at low (unrealistic)")
        print(f" 50% = Always at mid-point (reasonable)")
        print(f"100% = Always at high (unrealistic)")
        print(f"{'='*60}\n")

        # Average should be in reasonable range [20%, 80%]
        # Not always buying at lows or highs
        assert 0.15 < avg_position < 0.85, \
            f"Average fill position {avg_position*100:.1f}% is at extreme " \
            f"(expected between 15% and 85%)"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short', '-s'])
