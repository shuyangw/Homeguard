"""
Survivorship Bias Detection Tests

Ensures backtests don't overestimate performance by testing only on
surviving stocks.

Survivorship bias occurs when:
- Only testing on stocks that survived to present day
- Ignoring delisted, bankrupt, or acquired companies
- Missing the "losers" from the dataset

These tests validate the system handles failed companies correctly.
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from typing import Tuple

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from backtesting.engine.backtest_engine import BacktestEngine
from backtesting.base.strategy import LongOnlyStrategy
from backtesting.utils.risk_config import RiskConfig


# ============================================================================
# DATA GENERATORS FOR FAILED COMPANIES
# ============================================================================

def create_delisting_scenario(
    n_bars_before_delist: int = 50,
    decline_bars: int = 20,
    start_price: float = 100.0
) -> pd.DataFrame:
    """
    Create data for a stock that gets delisted.

    Price gradually declines to near-zero over decline_bars,
    then data stops (simulating delisting).
    """
    total_bars = n_bars_before_delist + decline_bars
    dates = pd.date_range(start='2023-01-01', periods=total_bars, freq='D')

    prices = []

    # Normal trading before decline
    for i in range(n_bars_before_delist):
        price = start_price + np.random.uniform(-2, 2)
        prices.append(max(price, 1.0))

    # Gradual decline to near-zero
    for i in range(decline_bars):
        # Linear decline to $0.10
        price = start_price * (1 - (i / decline_bars) * 0.999)
        prices.append(max(price, 0.10))

    df = pd.DataFrame({
        'open': prices,
        'high': [p * 1.02 for p in prices],
        'low': [p * 0.98 for p in prices],
        'close': prices,
        'volume': [100000] * total_bars
    }, index=dates)

    return df


def create_bankruptcy_scenario(
    n_bars: int = 50,
    bankruptcy_bar: int = 40,
    start_price: float = 50.0
) -> pd.DataFrame:
    """
    Create data for a stock that goes bankrupt.

    Price collapses to $0 at bankruptcy_bar.
    """
    dates = pd.date_range(start='2023-01-01', periods=n_bars, freq='D')

    prices = []

    for i in range(n_bars):
        if i < bankruptcy_bar - 5:
            # Normal trading
            price = start_price + np.random.uniform(-5, 5)
        elif i < bankruptcy_bar:
            # Rapid decline
            decline_factor = (bankruptcy_bar - i) / 5
            price = start_price * decline_factor * 0.2
        else:
            # Bankrupt - price at $0
            price = 0.01  # Near-zero (can't be exactly 0)

        prices.append(max(price, 0.01))

    df = pd.DataFrame({
        'open': prices,
        'high': [p * 1.05 for p in prices],
        'low': [p * 0.95 for p in prices],
        'close': prices,
        'volume': [50000] * n_bars
    }, index=dates)

    return df


def create_partial_data_scenario(
    total_bars: int = 100,
    data_ends_at: int = 60,
    start_price: float = 100.0
) -> pd.DataFrame:
    """
    Create data that stops mid-backtest (acquisition/delisting).

    Simulates a stock that has data up to data_ends_at, then nothing.
    """
    dates = pd.date_range(start='2023-01-01', periods=data_ends_at, freq='D')

    prices = [start_price + i * 0.5 for i in range(data_ends_at)]

    df = pd.DataFrame({
        'open': prices,
        'high': [p * 1.01 for p in prices],
        'low': [p * 0.99 for p in prices],
        'close': prices,
        'volume': [100000] * data_ends_at
    }, index=dates)

    return df


def create_mixed_survivors_and_failures(
    n_survivors: int = 3,
    n_failures: int = 2
) -> pd.DataFrame:
    """
    Create multi-symbol dataset with both surviving and failed stocks.

    Tests that strategy performs realistically with full universe.
    """
    # This would create a multi-index DataFrame
    # Simplified for now - would need multi-symbol support
    pass


# ============================================================================
# TEST STRATEGIES
# ============================================================================

class BuyAndHoldStrategy(LongOnlyStrategy):
    """Simple buy and hold for survivorship tests."""

    def generate_long_signals(self, data: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        entries = pd.Series(False, index=data.index)
        exits = pd.Series(False, index=data.index)

        # Buy on first bar
        entries.iloc[0] = True

        # Exit on last bar (if still have data)
        if len(data) > 1:
            exits.iloc[-1] = True

        return entries, exits


class EarlyExitStrategy(LongOnlyStrategy):
    """Exit after N bars regardless of price."""

    def __init__(self, hold_bars: int = 30):
        super().__init__()
        self.hold_bars = hold_bars

    def generate_long_signals(self, data: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        entries = pd.Series(False, index=data.index)
        exits = pd.Series(False, index=data.index)

        entries.iloc[0] = True

        if len(data) > self.hold_bars:
            exits.iloc[self.hold_bars] = True
        else:
            exits.iloc[-1] = True

        return entries, exits


# ============================================================================
# TEST CLASS: Delisting Scenarios
# ============================================================================

class TestDelistingScenarios:
    """
    Verify system handles delisted stocks correctly.
    """

    def test_delisting_shows_loss(self):
        """
        Stock that gets delisted should show realistic loss.

        If stock price goes from $100 → $0.10 before delisting,
        backtest should reflect this massive loss.
        """
        data = create_delisting_scenario(
            n_bars_before_delist=50,
            decline_bars=20,
            start_price=100.0
        )

        strategy = BuyAndHoldStrategy()

        print(f"\n{'='*60}")
        print("DELISTING SCENARIO TEST")
        print(f"{'='*60}")
        print(f"Stock declines from ${data.iloc[0]['close']:.2f} to ${data.iloc[-1]['close']:.2f}")
        print(f"Expected: Massive loss")

        # Use disabled risk (99% deployment) to test full impact
        engine = BacktestEngine(
            initial_capital=10000,
            fees=0.001,
            slippage=0.0,
            market_hours_only=False,
            risk_config=RiskConfig.disabled()  # 99% capital per trade
        )

        portfolio = engine.run_with_data(strategy, data)

        stats = portfolio.stats()
        total_return = stats.get('Total Return [%]', 0)

        print(f"\nActual return: {total_return:.2f}%")
        print(f"{'='*60}\n")

        # Should show massive loss (near -99% with full deployment)
        assert total_return < -90, \
            f"Delisting should cause massive loss, got {total_return:.2f}%"

    def test_early_exit_before_delisting(self):
        """
        If strategy exits before delisting, should capture partial decline.

        Exit after 30 bars (before full decline) should show smaller loss.
        """
        data = create_delisting_scenario(
            n_bars_before_delist=50,
            decline_bars=20,
            start_price=100.0
        )

        # Exit early (bar 30) before major decline
        strategy = EarlyExitStrategy(hold_bars=30)

        print(f"\n{'='*60}")
        print("EARLY EXIT BEFORE DELISTING TEST")
        print(f"{'='*60}")
        print("Exit at bar 30 (before major decline)")

        engine = BacktestEngine(
            initial_capital=10000,
            fees=0.001,
            slippage=0.0,
            market_hours_only=False,
            risk_config=RiskConfig.moderate()
        )

        portfolio = engine.run_with_data(strategy, data)

        stats = portfolio.stats()
        total_return = stats.get('Total Return [%]', 0)

        print(f"Return with early exit: {total_return:.2f}%")
        print(f"{'='*60}\n")

        # Should show small loss/gain, not massive loss
        assert total_return > -50, \
            f"Early exit should avoid massive loss, got {total_return:.2f}%"


# ============================================================================
# TEST CLASS: Bankruptcy Scenarios
# ============================================================================

class TestBankruptcyScenarios:
    """
    Verify system handles bankruptcy (price → $0) correctly.
    """

    def test_bankruptcy_reflects_total_loss(self):
        """
        Stock going to $0 should show near-total loss.

        Buy at $50, stock goes bankrupt → $0.01
        Should show ~99-100% loss.
        """
        data = create_bankruptcy_scenario(
            n_bars=50,
            bankruptcy_bar=40,
            start_price=50.0
        )

        strategy = BuyAndHoldStrategy()

        print(f"\n{'='*60}")
        print("BANKRUPTCY SCENARIO TEST")
        print(f"{'='*60}")
        print(f"Stock goes from ${data.iloc[0]['close']:.2f} to ${data.iloc[-1]['close']:.2f}")
        print("Expected: Near-total loss")

        # Use disabled risk (99% deployment) to test full impact
        engine = BacktestEngine(
            initial_capital=10000,
            fees=0.001,
            slippage=0.0,
            market_hours_only=False,
            risk_config=RiskConfig.disabled()  # 99% capital per trade
        )

        portfolio = engine.run_with_data(strategy, data)

        stats = portfolio.stats()
        total_return = stats.get('Total Return [%]', 0)
        final_value = stats.get('Final Portfolio Value [$]', 0)

        print(f"\nTotal return: {total_return:.2f}%")
        print(f"Final value: ${final_value:.2f}")
        print(f"{'='*60}\n")

        # Should show near-total loss (near -99% with full deployment)
        assert total_return < -95, \
            f"Bankruptcy should cause near-total loss, got {total_return:.2f}%"


# ============================================================================
# TEST CLASS: Partial Data Scenarios
# ============================================================================

class TestPartialDataScenarios:
    """
    Verify system handles stocks with incomplete data.
    """

    def test_partial_data_exits_at_end(self):
        """
        Stock with partial data should exit at last available bar.

        If data ends at bar 60 (acquisition), should exit there.
        """
        data = create_partial_data_scenario(
            total_bars=100,
            data_ends_at=60,
            start_price=100.0
        )

        strategy = BuyAndHoldStrategy()

        print(f"\n{'='*60}")
        print("PARTIAL DATA SCENARIO TEST")
        print(f"{'='*60}")
        print(f"Data ends at bar {len(data)} (acquisition/delisting)")
        print("Expected: Exit at last available bar")

        engine = BacktestEngine(
            initial_capital=10000,
            fees=0.001,
            slippage=0.0,
            market_hours_only=False,
            risk_config=RiskConfig.moderate()
        )

        portfolio = engine.run_with_data(strategy, data)

        # Check that exit occurred
        exit_trades = [t for t in portfolio.trades if t['type'] == 'exit']

        print(f"\nExit trades: {len(exit_trades)}")
        if exit_trades:
            exit_date = exit_trades[0]['timestamp']
            print(f"Exited on: {exit_date.date()}")
            print(f"Last data bar: {data.index[-1].date()}")

        print(f"{'='*60}\n")

        assert len(exit_trades) > 0, "Should have exited at end of data"


# ============================================================================
# TEST CLASS: Universe Selection Bias
# ============================================================================

class TestUniverseSelectionBias:
    """
    Test for bias from selecting only successful stocks.
    """

    def test_documenting_survivorship_bias_exists(self):
        """
        Document that survivorship bias exists if not using full universe.

        This test documents that using only surviving stocks will
        overestimate returns. It's a documentation test, not a fix.
        """
        print(f"\n{'='*60}")
        print("SURVIVORSHIP BIAS DOCUMENTATION")
        print(f"{'='*60}")
        print("\nSurvivorship bias occurs when:")
        print("1. Testing only on stocks that survived to present")
        print("2. Excluding delisted/bankrupt companies")
        print("3. Missing the 'losers' from the dataset")
        print()
        print("Impact:")
        print("- Overestimates strategy performance by 1-3% annually")
        print("- Hides strategies that work only on surviving stocks")
        print("- Creates unrealistic expectations")
        print()
        print("Mitigation:")
        print("1. Use datasets with delisted stocks included")
        print("2. Test on full historical universe (survivorship-free)")
        print("3. Apply penalty factor (reduce returns by 1-2%)")
        print("4. Be aware of the bias in results")
        print(f"{'='*60}\n")

        # This is a documentation test - always passes
        assert True, "Survivorship bias documented"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short', '-s'])
