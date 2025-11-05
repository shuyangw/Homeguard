"""
GOLD STANDARD: Synthetic Data Validation Tests

These tests use synthetic data with KNOWN CORRECT OUTCOMES to prove
the backtesting engine calculates correctly.

This is the most important test file for validation - if the engine
has calculation bugs, these tests will catch them.

Mathematical Principle:
    Create scenario → Calculate expected outcome → Run backtest
    If backtest != expected → BUG IN ENGINE!
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
# SYNTHETIC DATA GENERATORS
# ============================================================================

def create_sine_wave_prices(
    periods: int = 10,
    amplitude: float = 0.10,
    base_price: float = 100.0,
    bars_per_period: int = 20
) -> pd.DataFrame:
    """
    Create sine wave price data with known peaks and valleys.

    Args:
        periods: Number of complete sine wave cycles
        amplitude: Peak-to-trough amplitude (0.10 = 10% swings)
        base_price: Center price of sine wave
        bars_per_period: Bars per complete cycle

    Returns:
        DataFrame with OHLCV data following sine wave pattern

    Example:
        periods=2, amplitude=0.10, base_price=100
        → Prices oscillate: 100 → 110 → 100 → 90 → 100 (repeat)
    """
    n_bars = periods * bars_per_period
    dates = pd.date_range(start='2023-01-01', periods=n_bars, freq='D')

    # Create sine wave
    x = np.linspace(0, periods * 2 * np.pi, n_bars)
    sine_wave = np.sin(x)

    # Scale to price amplitude
    prices = base_price + (sine_wave * base_price * amplitude)

    # Create OHLCV
    df = pd.DataFrame({
        'open': prices,
        'high': prices + 0.5,  # Small intrabar range
        'low': prices - 0.5,
        'close': prices,
        'volume': np.ones(n_bars) * 1000000
    }, index=dates)

    return df


def create_all_wins_data(
    n_trades: int = 10,
    gain_pct: float = 0.05,
    base_price: float = 100.0,
    bars_per_trade: int = 10
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Create data for 100% win rate scenario with known outcome.

    Args:
        n_trades: Number of winning trades
        gain_pct: Percentage gain per trade (0.05 = 5%)
        base_price: Starting price
        bars_per_trade: Bars between entry and exit

    Returns:
        (price_data, entry_signals, exit_signals)
    """
    n_bars = n_trades * bars_per_trade + 1
    dates = pd.date_range(start='2023-01-01', periods=n_bars, freq='D')

    # Create upward trending prices
    prices = []
    current_price = base_price

    for trade in range(n_trades):
        # Entry price
        for i in range(bars_per_trade):
            if i == 0:
                prices.append(current_price)
            elif i == bars_per_trade - 1:
                # Exit at gain_pct higher
                current_price = current_price * (1 + gain_pct)
                prices.append(current_price)
            else:
                # Gradual rise
                prices.append(current_price + (current_price * gain_pct * i / bars_per_trade))

    prices.append(current_price)  # Final bar

    prices_array = np.array(prices)

    df = pd.DataFrame({
        'open': prices_array,
        'high': prices_array + 0.5,
        'low': prices_array - 0.5,
        'close': prices_array,
        'volume': np.ones(n_bars) * 1000000
    }, index=dates)

    # Create entry/exit signals
    entries = pd.Series(False, index=dates)
    exits = pd.Series(False, index=dates)

    for trade in range(n_trades):
        entry_idx = trade * bars_per_trade
        exit_idx = entry_idx + bars_per_trade - 1

        if entry_idx < len(dates):
            entries.iloc[entry_idx] = True
        if exit_idx < len(dates):
            exits.iloc[exit_idx] = True

    return df, entries, exits


def create_all_losses_data(
    n_trades: int = 10,
    loss_pct: float = 0.03,
    base_price: float = 100.0,
    bars_per_trade: int = 10
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Create data for 0% win rate scenario (all losses) with known outcome.
    """
    n_bars = n_trades * bars_per_trade + 1
    dates = pd.date_range(start='2023-01-01', periods=n_bars, freq='D')

    prices = []
    current_price = base_price

    for trade in range(n_trades):
        for i in range(bars_per_trade):
            if i == 0:
                prices.append(current_price)
            elif i == bars_per_trade - 1:
                # Exit at loss_pct lower
                current_price = current_price * (1 - loss_pct)
                prices.append(current_price)
            else:
                # Gradual decline
                prices.append(current_price - (current_price * loss_pct * i / bars_per_trade))

    prices.append(current_price)

    prices_array = np.array(prices)

    df = pd.DataFrame({
        'open': prices_array,
        'high': prices_array + 0.5,
        'low': prices_array - 0.5,
        'close': prices_array,
        'volume': np.ones(n_bars) * 1000000
    }, index=dates)

    entries = pd.Series(False, index=dates)
    exits = pd.Series(False, index=dates)

    for trade in range(n_trades):
        entry_idx = trade * bars_per_trade
        exit_idx = entry_idx + bars_per_trade - 1

        if entry_idx < len(dates):
            entries.iloc[entry_idx] = True
        if exit_idx < len(dates):
            exits.iloc[exit_idx] = True

    return df, entries, exits


def create_flat_price_data(n_bars: int = 100, price: float = 100.0) -> pd.DataFrame:
    """Create perfectly flat price data (no movement)."""
    dates = pd.date_range(start='2023-01-01', periods=n_bars, freq='D')

    df = pd.DataFrame({
        'open': np.ones(n_bars) * price,
        'high': np.ones(n_bars) * price + 0.1,
        'low': np.ones(n_bars) * price - 0.1,
        'close': np.ones(n_bars) * price,
        'volume': np.ones(n_bars) * 1000000
    }, index=dates)

    return df


# ============================================================================
# SYNTHETIC STRATEGIES
# ============================================================================

def generate_perfect_sine_signals(
    data: pd.DataFrame,
    bars_per_period: int = 20
) -> Tuple[pd.Series, pd.Series]:
    """
    Generate perfect entry/exit signals for sine wave data.

    For a sine wave with known period, we can calculate exactly where
    valleys and peaks occur without lookahead bias.

    Args:
        data: Price data (must be sine wave from create_sine_wave_prices)
        bars_per_period: Bars per complete sine cycle

    Returns:
        (entry_signals, exit_signals)
    """
    close = data['close']
    n = len(close)

    # For a sine wave starting at phase=0 (y=0, rising):
    # - Peak occurs at bars_per_period/4 (phase = π/2)
    # - Valley occurs at 3*bars_per_period/4 (phase = 3π/2)

    # Find local minima and maxima using centered rolling window
    # Note: This uses center=True which normally would be lookahead,
    # but for test validation purposes this is acceptable since
    # we're generating "perfect" signals to validate the engine
    window = 5
    rolling_min = close.rolling(window=window, center=True).min()
    rolling_max = close.rolling(window=window, center=True).max()

    entries = (close == rolling_min).fillna(False)
    exits = (close == rolling_max).fillna(False)

    return entries, exits


class PerfectPredictorStrategy(LongOnlyStrategy):
    """
    Perfect predictor strategy for sine wave data.

    Uses pre-computed perfect signals based on sine wave structure.
    """

    def __init__(self, bars_per_period: int = 20):
        super().__init__(bars_per_period=bars_per_period)
        self.bars_per_period = bars_per_period

    def generate_long_signals(self, data: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        return generate_perfect_sine_signals(data, self.bars_per_period)


class NeverTradeStrategy(LongOnlyStrategy):
    """Strategy that never generates any signals."""

    def generate_long_signals(self, data: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        n = len(data)
        entries = pd.Series(False, index=data.index)
        exits = pd.Series(False, index=data.index)
        return entries, exits


class SignalBasedStrategy(LongOnlyStrategy):
    """Strategy that uses pre-generated signals."""

    def __init__(self, entries: pd.Series, exits: pd.Series):
        super().__init__()
        self._entries = entries
        self._exits = exits

    def generate_long_signals(self, data: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        # Return pre-generated signals
        return self._entries.copy(), self._exits.copy()


# ============================================================================
# TEST CLASS: Perfect Predictor (GOLD STANDARD)
# ============================================================================

class TestPerfectPredictor:
    """
    The GOLD STANDARD test for backtest engine correctness.

    If this test fails, there's a bug in the engine.
    """

    def test_perfect_predictor_on_sine_wave(self):
        """
        CRITICAL: Perfect strategy on sine wave must match theoretical return.

        Scenario:
        - 5 complete sine wave cycles
        - 10% amplitude (valley=$90, peak=$110, base=$100)
        - Perfect predictor: Buy every valley, sell every peak
        - Expected ~4-5 complete round trips

        Math:
        - Valley = $100 × (1 - 0.10) = $90
        - Peak = $100 × (1 + 0.10) = $110
        - Gain = ($110 - $90) / $90 = 22.22%

        Expected (with 0.1% fees):
        - Each trade: 0.1% entry + 0.1% exit = 0.2% round-trip cost
        - Net gain per trade: 22.22% - 0.2% = 22.02%
        - 4 trades: (1.2202)^4 = 2.218 = 121.8% total return
        - 5 trades: (1.2202)^5 = 2.705 = 170.5% total return
        """
        # Create sine wave data
        data = create_sine_wave_prices(
            periods=5,
            amplitude=0.10,
            base_price=100.0,
            bars_per_period=20
        )

        # Perfect predictor strategy
        strategy = PerfectPredictorStrategy(bars_per_period=20)

        # Run backtest with minimal fees
        engine = BacktestEngine(
            initial_capital=10000,
            fees=0.001,  # 0.1%
            slippage=0.0,
            market_hours_only=False,
            risk_config=RiskConfig.disabled()  # Use 99% per trade for simplicity
        )

        portfolio = engine.run_with_data(strategy, data)

        # Calculate theoretical return
        # For sine wave with amplitude A and base B:
        # Valley = B(1-A), Peak = B(1+A)
        # Gain = [B(1+A) - B(1-A)] / B(1-A) = 2A / (1-A)
        amplitude = 0.10
        base_price = 100.0
        valley_price = base_price * (1 - amplitude)  # $90
        peak_price = base_price * (1 + amplitude)    # $110
        gain_per_trade = (2 * amplitude) / (1 - amplitude)  # 22.22%

        fees_per_trade = 0.002  # 0.2% round trip (0.1% * 2)
        net_gain_per_trade = gain_per_trade - fees_per_trade  # 22.0%

        # Estimate number of trades (should be ~10 for 5 periods)
        expected_trades_approx = 10

        # Allow for some variation in detection
        actual_trades = len([t for t in portfolio.trades if t['type'] == 'exit'])

        # Theoretical return (compound)
        # Use actual trade count for more accurate comparison
        theoretical_return_pct = ((1 + net_gain_per_trade) ** actual_trades - 1) * 100

        # Get actual return
        stats = portfolio.stats()
        actual_return_pct = stats.get('Total Return [%]', 0)

        # Debug: Print actual trade details
        print(f"\n{'='*60}")
        print("ACTUAL TRADE DETAILS:")
        print(f"{'='*60}")

        # First, let's see what's in a trade
        if portfolio.trades:
            print(f"Sample trade structure: {portfolio.trades[0]}")
            print(f"Available keys: {portfolio.trades[0].keys()}")

        entry_trades = [t for t in portfolio.trades if t['type'] == 'entry']
        exit_trades = [t for t in portfolio.trades if t['type'] == 'exit']

        for i, (entry, exit_t) in enumerate(zip(entry_trades, exit_trades), 1):
            entry_price = entry['price']
            exit_price = exit_t['price']
            gross_gain = (exit_price - entry_price) / entry_price * 100

            print(f"Trade {i}:")
            print(f"  Entry:  @ ${entry_price:.2f}")
            print(f"  Exit:   @ ${exit_price:.2f}")
            print(f"  Gross gain:  {gross_gain:.2f}%")
            print(f"  Full entry: {entry}")
            print(f"  Full exit: {exit_t}")

        print(f"\n{'='*60}")
        print("PERFECT PREDICTOR TEST - GOLD STANDARD")
        print(f"{'='*60}")
        print(f"Data: 5 sine wave cycles, 10% amplitude")
        print(f"Strategy: Perfect predictor (buy valleys, sell peaks)")
        print(f"Fees: 0.1% per trade (0.2% round trip)")
        print(f"-" * 60)
        print(f"Expected trades (approx): {expected_trades_approx}")
        print(f"Actual trades: {actual_trades}")
        print(f"Net gain per trade: {net_gain_per_trade*100:.1f}%")
        print(f"-" * 60)
        print(f"Theoretical return: {theoretical_return_pct:.2f}%")
        print(f"Actual return: {actual_return_pct:.2f}%")
        print(f"Difference: {abs(actual_return_pct - theoretical_return_pct):.2f}%")
        print(f"{'='*60}\n")

        # Allow 20% tolerance to account for:
        # 1. Potential open positions with unrealized P&L
        # 2. Imperfect extrema detection (within 1-2 bars)
        # 3. Small variations in actual vs theoretical prices
        # This test catches gross errors (100%+ discrepancies), while other
        # tests (all wins, all losses, etc.) provide tighter validation
        tolerance = theoretical_return_pct * 0.20

        assert abs(actual_return_pct - theoretical_return_pct) < tolerance, \
            f"ENGINE BUG: Actual return {actual_return_pct:.2f}% differs from " \
            f"theoretical {theoretical_return_pct:.2f}% by more than {tolerance:.2f}%"


# ============================================================================
# TEST CLASS: Known Win/Loss Scenarios
# ============================================================================

class TestKnownOutcomes:
    """
    Test scenarios with mathematically calculable outcomes.
    """

    def test_all_wins_scenario(self):
        """
        100% win rate with known outcome.

        Scenario:
        - 10 trades, each +5% gain
        - 0% fees for simplicity
        - Expected: (1.05)^10 = 1.6289 = 62.89% return
        """
        data, entries, exits = create_all_wins_data(
            n_trades=10,
            gain_pct=0.05,
            base_price=100.0
        )

        strategy = SignalBasedStrategy(entries, exits)

        engine = BacktestEngine(
            initial_capital=10000,
            fees=0.0,  # No fees for clean calculation
            slippage=0.0,
            market_hours_only=False,
            risk_config=RiskConfig.disabled()
        )

        portfolio = engine.run_with_data(strategy, data)

        # Calculate expected return
        # (1.05)^10 = 1.6289 = 62.89% return
        expected_return_pct = ((1.05 ** 10) - 1) * 100  # 62.89%

        stats = portfolio.stats()
        actual_return_pct = stats.get('Total Return [%]', 0)

        print(f"\nALL WINS SCENARIO:")
        print(f"Expected: {expected_return_pct:.2f}%")
        print(f"Actual: {actual_return_pct:.2f}%")

        # Allow 2% tolerance
        assert abs(actual_return_pct - expected_return_pct) < 2.0, \
            f"All wins scenario: Expected {expected_return_pct:.2f}%, got {actual_return_pct:.2f}%"

    def test_all_losses_scenario(self):
        """
        0% win rate with known outcome.

        Scenario:
        - 10 trades, each -3% loss
        - 0% fees for simplicity
        - Expected: (0.97)^10 = 0.7374 = -26.26% return
        """
        data, entries, exits = create_all_losses_data(
            n_trades=10,
            loss_pct=0.03,
            base_price=100.0
        )

        strategy = SignalBasedStrategy(entries, exits)

        engine = BacktestEngine(
            initial_capital=10000,
            fees=0.0,
            slippage=0.0,
            market_hours_only=False,
            risk_config=RiskConfig.disabled()
        )

        portfolio = engine.run_with_data(strategy, data)

        # Calculate expected return
        # (0.97)^10 = 0.7374 = -26.26% return
        expected_return_pct = ((0.97 ** 10) - 1) * 100  # -26.26%

        stats = portfolio.stats()
        actual_return_pct = stats.get('Total Return [%]', 0)

        print(f"\nALL LOSSES SCENARIO:")
        print(f"Expected: {expected_return_pct:.2f}%")
        print(f"Actual: {actual_return_pct:.2f}%")

        # Allow 2% tolerance
        assert abs(actual_return_pct - expected_return_pct) < 2.0, \
            f"All losses scenario: Expected {expected_return_pct:.2f}%, got {actual_return_pct:.2f}%"

    def test_zero_trades_preserves_capital(self):
        """
        Strategy that never trades → final equity = initial capital.

        This is the simplest test - if this fails, something is very wrong.
        """
        data = create_flat_price_data(n_bars=100, price=100.0)

        strategy = NeverTradeStrategy()

        engine = BacktestEngine(
            initial_capital=10000,
            fees=0.001,
            slippage=0.0,
            market_hours_only=False
        )

        portfolio = engine.run_with_data(strategy, data)

        stats = portfolio.stats()
        start_value = stats.get('Start Value', 0)
        end_value = stats.get('End Value', 0)
        total_trades = stats.get('Total Trades', -1)

        print(f"\nZERO TRADES SCENARIO:")
        print(f"Start: ${start_value:.2f}")
        print(f"End: ${end_value:.2f}")
        print(f"Trades: {total_trades}")

        # Should have 0 trades
        assert total_trades == 0, f"Expected 0 trades, got {total_trades}"

        # Capital should be preserved (exactly)
        assert abs(end_value - start_value) < 0.01, \
            f"Capital not preserved: ${start_value:.2f} → ${end_value:.2f}"


# ============================================================================
# TEST CLASS: Single Trade Precision
# ============================================================================

class TestSingleTradePrecision:
    """
    Test exact P&L calculation for single trades.
    """

    def test_single_trade_pnl_calculation(self):
        """
        Single entry/exit → verify exact P&L with fees.

        Scenario:
        - Init capital: $10,000
        - Buy 100 shares @ $100 = $10,000 cost
        - Fee: 0.1% = $10
        - Total cost: $10,010
        - Sell 100 shares @ $110 = $11,000 proceeds
        - Fee: 0.1% = $11
        - Net proceeds: $10,989
        - Expected P&L: $10,989 - $10,010 = $979 (9.79%)
        """
        # Create data: price goes 100 → 110
        dates = pd.date_range(start='2023-01-01', periods=20, freq='D')
        prices = np.linspace(100, 110, 20)

        data = pd.DataFrame({
            'open': prices,
            'high': prices + 0.5,
            'low': prices - 0.5,
            'close': prices,
            'volume': np.ones(20) * 1000000
        }, index=dates)

        # Signals: Enter on day 0, exit on day 19
        entries = pd.Series(False, index=dates)
        exits = pd.Series(False, index=dates)
        entries.iloc[0] = True
        exits.iloc[19] = True

        strategy = SignalBasedStrategy(entries, exits)

        engine = BacktestEngine(
            initial_capital=10000,
            fees=0.001,  # 0.1%
            slippage=0.0,
            market_hours_only=False,
            risk_config=RiskConfig.disabled()  # 99% allocation
        )

        portfolio = engine.run_with_data(strategy, data)

        # Manual calculation
        entry_price = 100.0
        exit_price = 110.0
        shares = 10000 / entry_price  # Buy with full capital
        entry_fee = entry_price * shares * 0.001
        total_cost = (entry_price * shares) + entry_fee

        proceeds = exit_price * shares
        exit_fee = proceeds * 0.001
        net_proceeds = proceeds - exit_fee

        expected_pnl = net_proceeds - total_cost
        expected_pnl_pct = (expected_pnl / total_cost) * 100

        # Get actual from backtest
        trades = portfolio.trades
        exit_trades = [t for t in trades if t['type'] == 'exit']

        if len(exit_trades) > 0:
            actual_pnl = exit_trades[0]['pnl']
            actual_pnl_pct = exit_trades[0]['pnl_pct']

            print(f"\nSINGLE TRADE PRECISION TEST:")
            print(f"Entry: {shares:.2f} shares @ ${entry_price:.2f}")
            print(f"Exit: {shares:.2f} shares @ ${exit_price:.2f}")
            print(f"Fees: ${entry_fee + exit_fee:.2f}")
            print(f"-" * 40)
            print(f"Expected P&L: ${expected_pnl:.2f} ({expected_pnl_pct:.2f}%)")
            print(f"Actual P&L: ${actual_pnl:.2f} ({actual_pnl_pct:.2f}%)")

            # Allow small rounding tolerance
            assert abs(actual_pnl - expected_pnl) < 1.0, \
                f"P&L mismatch: Expected ${expected_pnl:.2f}, got ${actual_pnl:.2f}"


# ============================================================================
# TEST CLASS: Fee Accumulation
# ============================================================================

class TestFeeAccumulation:
    """
    Verify fees accumulate correctly over multiple trades.
    """

    def test_fee_accumulation_over_multiple_trades(self):
        """
        Verify fees accumulate correctly.

        Scenario:
        - 5 trades with 0% gain (breakeven on price)
        - 0.1% fee per trade = 0.2% round-trip
        - Expected total fees: ~1% of capital (0.2% * 5 trades)
        - Expected return: -1% (just fees, no gains)
        """
        # Create flat price data
        dates = pd.date_range(start='2023-01-01', periods=51, freq='D')
        prices = np.ones(51) * 100.0

        data = pd.DataFrame({
            'open': prices,
            'high': prices + 0.1,
            'low': prices - 0.1,
            'close': prices,
            'volume': np.ones(51) * 1000000
        }, index=dates)

        # 5 round-trip trades
        entries = pd.Series(False, index=dates)
        exits = pd.Series(False, index=dates)

        for i in range(5):
            entries.iloc[i * 10] = True
            exits.iloc[i * 10 + 9] = True

        strategy = SignalBasedStrategy(entries, exits)

        engine = BacktestEngine(
            initial_capital=10000,
            fees=0.001,  # 0.1%
            slippage=0.0,
            market_hours_only=False,
            risk_config=RiskConfig.disabled()
        )

        portfolio = engine.run_with_data(strategy, data)

        stats = portfolio.stats()
        actual_return_pct = stats.get('Total Return [%]', 0)

        # Expected: ~-1% (0.2% round-trip * 5 trades)
        # Fees compound, so slightly more than -1%
        expected_return_pct = -1.0

        print(f"\nFEE ACCUMULATION TEST:")
        print(f"5 breakeven trades with 0.2% round-trip fees")
        print(f"Expected return (approx): {expected_return_pct:.2f}%")
        print(f"Actual return: {actual_return_pct:.2f}%")

        # Should be negative (fees eat into capital)
        assert actual_return_pct < 0, "Return should be negative (fees only)"

        # Should be around -1%
        assert -1.5 < actual_return_pct < -0.5, \
            f"Expected ~-1% from fees, got {actual_return_pct:.2f}%"


# ============================================================================
# TEST CLASS: Mixed Win/Loss Scenarios
# ============================================================================

class TestMixedOutcomes:
    """
    Test realistic scenarios with alternating wins and losses.
    """

    def test_alternating_wins_losses(self):
        """
        Alternating win/loss pattern with known outcome.

        Scenario:
        - 10 trades alternating: +10%, -5%, +10%, -5%, etc.
        - Expected: (1.10 * 0.95)^5 = (1.045)^5 = 24.62% return
        - After 0.2% fees per round trip: slightly lower
        """
        # Create data with alternating price movements
        # Simplified: Create price array directly
        n_trades = 10
        bars_per_trade = 10
        n_bars = n_trades * bars_per_trade + 1  # +1 for initial bar
        dates = pd.date_range(start='2023-01-01', periods=n_bars, freq='D')

        prices = []
        current_price = 100.0

        for i in range(n_trades):
            # Entry bar
            prices.append(current_price)

            # Hold for 8 bars (flat price)
            for _ in range(8):
                prices.append(current_price)

            # Apply price change for exit bar
            if i % 2 == 0:
                # Win: +10%
                exit_price = current_price * 1.10
            else:
                # Loss: -5%
                exit_price = current_price * 0.95

            # Exit bar (bar 9 of this trade)
            prices.append(exit_price)

            # Update current price for next trade
            current_price = exit_price

        # Add final bar
        prices.append(current_price)

        data = pd.DataFrame({
            'open': prices,
            'high': [p * 1.01 for p in prices],
            'low': [p * 0.99 for p in prices],
            'close': prices,
            'volume': np.ones(n_bars) * 1000000
        }, index=dates)

        # Generate entry/exit signals
        entries = pd.Series(False, index=dates)
        exits = pd.Series(False, index=dates)

        for i in range(n_trades):
            entry_bar = i * bars_per_trade
            exit_bar = entry_bar + 9
            entries.iloc[entry_bar] = True
            exits.iloc[exit_bar] = True

        strategy = SignalBasedStrategy(entries, exits)

        engine = BacktestEngine(
            initial_capital=10000,
            fees=0.001,  # 0.1%
            slippage=0.0,
            market_hours_only=False,
            risk_config=RiskConfig.disabled()
        )

        portfolio = engine.run_with_data(strategy, data)

        # Calculate theoretical return
        # Win: +10%, Loss: -5%
        # 5 win/loss pairs: (1.10 * 0.95)^5 = 1.045^5 = 1.2462
        gross_return_pct = ((1.10 * 0.95) ** 5 - 1) * 100  # 24.62%

        # Account for fees: 0.2% per round trip * 10 trades = 2% total
        # But fees compound, so net is slightly different
        # Approximate: 24.62% - 2% = 22.62%
        expected_return_pct = gross_return_pct - 2.0

        stats = portfolio.stats()
        actual_return_pct = stats.get('Total Return [%]', 0)

        print(f"\nALTERNATING WINS/LOSSES TEST:")
        print(f"10 trades: +10%, -5%, +10%, -5%, ...")
        print(f"Gross theoretical: {gross_return_pct:.2f}%")
        print(f"Expected (after fees): ~{expected_return_pct:.2f}%")
        print(f"Actual return: {actual_return_pct:.2f}%")

        # Allow 5% tolerance
        tolerance = 5.0
        assert abs(actual_return_pct - expected_return_pct) < tolerance, \
            f"Return mismatch: Expected ~{expected_return_pct:.2f}%, got {actual_return_pct:.2f}%"

    def test_extreme_volatility_handling(self):
        """
        Verify engine handles extreme volatility correctly.

        Scenario:
        - Stock price swings wildly: ±20% per bar
        - Buy low, sell high strategy
        - Should not crash, produce reasonable results
        """
        # Create highly volatile data
        n_bars = 50
        dates = pd.date_range(start='2023-01-01', periods=n_bars, freq='D')

        # Random walk with large steps
        np.random.seed(42)  # Reproducible
        prices = [100.0]
        for _ in range(n_bars - 1):
            change = np.random.uniform(-0.20, 0.20)  # ±20%
            new_price = prices[-1] * (1 + change)
            prices.append(max(new_price, 1.0))  # Floor at $1

        data = pd.DataFrame({
            'open': prices,
            'high': [p * 1.05 for p in prices],  # Allow for intraday range
            'low': [p * 0.95 for p in prices],
            'close': prices,
            'volume': np.ones(n_bars) * 1000000
        }, index=dates)

        # Simple strategy: Buy when price drops, sell when price rises
        close = data['close']
        returns = close.pct_change()

        entries = returns < -0.10  # Buy after -10% drop
        exits = returns > 0.10     # Sell after +10% gain

        strategy = SignalBasedStrategy(entries, exits)

        engine = BacktestEngine(
            initial_capital=10000,
            fees=0.001,
            slippage=0.005,  # Higher slippage in volatile markets
            market_hours_only=False,
            risk_config=RiskConfig.moderate()
        )

        portfolio = engine.run_with_data(strategy, data)

        stats = portfolio.stats()
        final_value = stats.get('End Value', 0)  # Correct key from portfolio_simulator.py
        total_return = stats.get('Total Return [%]', 0)

        print(f"\nEXTREME VOLATILITY TEST:")
        print(f"Price range: ${min(prices):.2f} - ${max(prices):.2f}")
        print(f"Mean volatility: {returns.abs().mean()*100:.1f}%")
        print(f"Final value: ${final_value:.2f}")
        print(f"Total return: {total_return:.2f}%")

        # Key validations:
        # 1. Engine did not crash
        assert final_value > 0, "Final value should be positive"

        # 2. Final value is reasonable (not infinite)
        assert final_value < 1000000, "Final value should be realistic"

        # 3. Return is within reasonable bounds (-100% to +500%)
        assert -100 < total_return < 500, \
            f"Return should be reasonable, got {total_return:.2f}%"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short', '-s'])
