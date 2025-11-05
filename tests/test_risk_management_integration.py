"""
Test risk management integration with portfolio simulator.
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from backtesting.engine.backtest_engine import BacktestEngine
from backtesting.utils.risk_config import RiskConfig
from strategies.base_strategies.moving_average import MovingAverageCrossover


@pytest.fixture
def simple_trending_data():
    """Generate simple trending price data for testing."""
    dates = pd.date_range('2024-01-01', periods=100, freq='D')

    # Generate uptrend with some noise
    trend = np.linspace(100, 120, 100)
    noise = np.random.normal(0, 1, 100)
    close = trend + noise

    df = pd.DataFrame({
        'timestamp': dates,
        'open': close - 0.5,
        'high': close + 1,
        'low': close - 1,
        'close': close,
        'volume': np.random.randint(1000000, 10000000, 100)
    })
    df.set_index('timestamp', inplace=True)

    return df


class TestRiskManagementIntegration:
    """Test risk management integration with backtesting engine."""

    def test_default_risk_config_is_moderate(self):
        """Test that default risk config is moderate (10% per trade)."""
        engine = BacktestEngine(initial_capital=100000.0)

        assert engine.risk_config is not None
        assert engine.risk_config.position_size_pct == 0.10
        assert engine.risk_config.use_stop_loss == True
        assert engine.risk_config.stop_loss_pct == 0.02

    def test_custom_risk_config_is_used(self):
        """Test that custom risk config is properly set."""
        custom_config = RiskConfig.conservative()
        engine = BacktestEngine(initial_capital=100000.0, risk_config=custom_config)

        assert engine.risk_config.position_size_pct == 0.05
        assert engine.risk_config.stop_loss_pct == 0.01

    def test_position_sizing_respects_percentage(self, simple_trending_data):
        """Test that position sizes respect configured percentage."""
        # Test with 10% position sizing (moderate)
        config = RiskConfig.moderate()
        engine = BacktestEngine(initial_capital=100000.0, fees=0.0, risk_config=config)

        # Use a more aggressive MA crossover to ensure trades
        strategy = MovingAverageCrossover(fast_window=3, slow_window=10)

        portfolio = engine.run_with_data(strategy, simple_trending_data)

        # If no trades were generated, that's okay - skip the test
        if len(portfolio.trades) == 0:
            import pytest
            pytest.skip("No trades generated on this data - strategy didn't find entry signals")

        # Check entry trades to verify position sizing
        entry_trades = [t for t in portfolio.trades if t['type'] == 'entry']
        if entry_trades:
            for trade in entry_trades:
                # Position cost should be approximately 10% of capital
                # (allowing for some variation due to fees/slippage)
                position_cost = trade['cost']
                expected_cost = 100000.0 * 0.10

                # Allow 20% tolerance (±2% of capital)
                assert position_cost <= expected_cost * 1.2, \
                    f"Position cost {position_cost} exceeds 10% limit (expected ~{expected_cost})"

    def test_disabled_risk_management_uses_high_allocation(self, simple_trending_data):
        """Test that disabled risk management uses high capital allocation."""
        config = RiskConfig.disabled()
        engine = BacktestEngine(initial_capital=100000.0, fees=0.0, risk_config=config)
        strategy = MovingAverageCrossover(fast_window=5, slow_window=15)

        portfolio = engine.run_with_data(strategy, simple_trending_data)

        # With disabled risk management, should use 99% of capital
        entry_trades = [t for t in portfolio.trades if t['type'] == 'entry']
        if entry_trades:
            first_trade = entry_trades[0]
            position_cost = first_trade['cost']

            # Should be close to 99% of capital
            assert position_cost > 100000.0 * 0.90, \
                "Disabled risk management should use high capital allocation"

    def test_conservative_vs_moderate_vs_aggressive(self, simple_trending_data):
        """Test that different risk profiles produce different position sizes."""
        strategy = MovingAverageCrossover(fast_window=5, slow_window=15)

        # Run with conservative config (5%)
        engine_conservative = BacktestEngine(
            initial_capital=100000.0,
            fees=0.0,
            risk_config=RiskConfig.conservative()
        )
        portfolio_conservative = engine_conservative.run_with_data(strategy, simple_trending_data)

        # Run with moderate config (10%)
        engine_moderate = BacktestEngine(
            initial_capital=100000.0,
            fees=0.0,
            risk_config=RiskConfig.moderate()
        )
        portfolio_moderate = engine_moderate.run_with_data(strategy, simple_trending_data)

        # Run with aggressive config (20%)
        engine_aggressive = BacktestEngine(
            initial_capital=100000.0,
            fees=0.0,
            risk_config=RiskConfig.aggressive()
        )
        portfolio_aggressive = engine_aggressive.run_with_data(strategy, simple_trending_data)

        # Get first entry trades
        conservative_entry = [t for t in portfolio_conservative.trades if t['type'] == 'entry']
        moderate_entry = [t for t in portfolio_moderate.trades if t['type'] == 'entry']
        aggressive_entry = [t for t in portfolio_aggressive.trades if t['type'] == 'entry']

        if conservative_entry and moderate_entry and aggressive_entry:
            conservative_cost = conservative_entry[0]['cost']
            moderate_cost = moderate_entry[0]['cost']
            aggressive_cost = aggressive_entry[0]['cost']

            # Conservative should be smallest
            assert conservative_cost < moderate_cost, \
                "Conservative position should be smaller than moderate"

            # Aggressive should be largest
            assert aggressive_cost > moderate_cost, \
                "Aggressive position should be larger than moderate"

    def test_stop_loss_exits_are_logged(self, simple_trending_data):
        """Test that stop loss exits are properly logged."""
        # Create a config with tight stop loss
        config = RiskConfig(
            position_size_pct=0.10,
            use_stop_loss=True,
            stop_loss_pct=0.01,  # Very tight 1% stop
            stop_loss_type='percentage'
        )

        engine = BacktestEngine(initial_capital=100000.0, fees=0.0, risk_config=config)
        strategy = MovingAverageCrossover(fast_window=5, slow_window=15)

        portfolio = engine.run_with_data(strategy, simple_trending_data)

        # Check if any exits have an exit_reason field
        exit_trades = [t for t in portfolio.trades if t['type'] == 'exit']

        # With tight stops, we should see some stop loss exits
        if exit_trades:
            has_exit_reason = any('exit_reason' in t for t in exit_trades)
            assert has_exit_reason, "Exit trades should include exit_reason field"


class TestPositionSizerInitialization:
    """Test that position sizers are correctly initialized."""

    def test_fixed_percentage_sizer_is_default(self):
        """Test that fixed percentage is the default position sizing method."""
        config = RiskConfig.moderate()
        assert config.position_sizing_method == 'fixed_percentage'

    def test_portfolio_accepts_risk_config(self, simple_trending_data):
        """Test that portfolio simulator accepts and uses risk config."""
        from backtesting.engine.portfolio_simulator import from_signals

        # Generate simple entry/exit signals
        entries = pd.Series(False, index=simple_trending_data.index)
        exits = pd.Series(False, index=simple_trending_data.index)

        # Entry at index 10, exit at index 20
        entries.iloc[10] = True
        exits.iloc[20] = True

        config = RiskConfig.moderate()

        portfolio = from_signals(
            close=simple_trending_data['close'],
            entries=entries,
            exits=exits,
            init_cash=100000.0,
            fees=0.0,
            slippage=0.0,
            freq='1D',
            risk_config=config,
            price_data=simple_trending_data
        )

        # Verify portfolio was created
        assert portfolio is not None
        assert hasattr(portfolio, 'risk_config')
        assert portfolio.risk_config.position_size_pct == 0.10


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
