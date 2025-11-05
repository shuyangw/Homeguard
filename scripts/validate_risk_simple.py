"""
Simple validation of risk management implementation using controlled test data.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import pandas as pd
import numpy as np
from backtesting.engine.portfolio_simulator import from_signals
from backtesting.utils.risk_config import RiskConfig
from utils import logger


def create_simple_signals(length=100):
    """Create simple entry/exit signals for testing."""
    dates = pd.date_range('2024-01-01', periods=length, freq='D')

    # Create upward trending prices
    prices = pd.Series(np.linspace(100, 120, length), index=dates)

    # Create signals: buy at index 10, sell at index 20
    # buy again at index 30, sell at index 40, etc.
    entries = pd.Series(False, index=dates)
    exits = pd.Series(False, index=dates)

    for i in range(10, length, 30):
        if i < length:
            entries.iloc[i] = True
        if i + 10 < length:
            exits.iloc[i + 10] = True

    return prices, entries, exits


def test_position_sizing():
    """Test position sizing with 99% vs 10%."""
    logger.blank()
    logger.separator("=", 80)
    logger.header("POSITION SIZING VALIDATION")
    logger.separator("=", 80)
    logger.blank()

    prices, entries, exits = create_simple_signals(100)

    # Test with 99% sizing
    logger.info("Test 1: 99% position sizing (DISABLED risk management)...")
    config_99 = RiskConfig.disabled()
    portfolio_99 = from_signals(
        close=prices,
        entries=entries,
        exits=exits,
        init_cash=100000.0,
        fees=0.0,
        slippage=0.0,
        freq='1D',
        market_hours_only=False,  # Disable for test data
        risk_config=config_99
    )

    # Test with 10% sizing
    logger.info("Test 2: 10% position sizing (MODERATE risk management)...")
    config_10 = RiskConfig.moderate()
    portfolio_10 = from_signals(
        close=prices,
        entries=entries,
        exits=exits,
        init_cash=100000.0,
        fees=0.0,
        slippage=0.0,
        freq='1D',
        market_hours_only=False,
        risk_config=config_10
    )

    # Analyze results
    logger.blank()
    logger.separator("-", 80)
    logger.header("RESULTS")
    logger.separator("-", 80)

    # Get first entry trade for each
    entry_99 = [t for t in portfolio_99.trades if t['type'] == 'entry']
    entry_10 = [t for t in portfolio_10.trades if t['type'] == 'entry']

    if entry_99 and entry_10:
        trade_99 = entry_99[0]
        trade_10 = entry_10[0]

        logger.metric(f"99% Sizing - First Trade:")
        logger.metric(f"  Shares: {trade_99['shares']:.2f}")
        logger.metric(f"  Cost: ${trade_99['cost']:,.2f}")
        logger.metric(f"  % of capital: {(trade_99['cost'] / 100000 * 100):.1f}%")
        logger.blank()

        logger.metric(f"10% Sizing - First Trade:")
        logger.metric(f"  Shares: {trade_10['shares']:.2f}")
        logger.metric(f"  Cost: ${trade_10['cost']:,.2f}")
        logger.metric(f"  % of capital: {(trade_10['cost'] / 100000 * 100):.1f}%")
        logger.blank()

        # Verify 10% is using correct sizing
        expected_cost = 100000 * 0.10
        actual_cost = trade_10['cost']
        tolerance = expected_cost * 0.05  # 5% tolerance

        if abs(actual_cost - expected_cost) < tolerance:
            logger.success(f"✓ 10% position sizing is CORRECT (${actual_cost:,.2f} ≈ ${expected_cost:,.2f})")
        else:
            logger.error(f"✗ 10% position sizing INCORRECT (${actual_cost:,.2f} ≠ ${expected_cost:,.2f})")

        logger.blank()

        # Verify 99% is using old sizing
        if trade_99['cost'] > 90000:
            logger.success(f"✓ 99% position sizing confirmed (${trade_99['cost']:,.2f} > $90,000)")
        else:
            logger.warning(f"⚠ 99% position sizing may be incorrect (${trade_99['cost']:,.2f})")

    else:
        logger.error("✗ No trades generated - cannot validate position sizing")

    logger.blank()


def test_stop_losses():
    """Test stop loss functionality."""
    logger.separator("=", 80)
    logger.header("STOP LOSS VALIDATION")
    logger.separator("=", 80)
    logger.blank()

    # Create data with a drop to trigger stop loss
    dates = pd.date_range('2024-01-01', periods=50, freq='D')
    prices = pd.Series([100] * 10 + list(range(100, 90, -1)) + [90] * 30, index=dates)

    # Entry at index 5, no explicit exit (let stop loss handle it)
    entries = pd.Series(False, index=dates)
    exits = pd.Series(False, index=dates)
    entries.iloc[5] = True

    logger.info("Test: 2% stop loss should trigger when price drops...")

    config_stop = RiskConfig(
        position_size_pct=0.10,
        use_stop_loss=True,
        stop_loss_pct=0.02,  # 2% stop loss
        stop_loss_type='percentage'
    )

    portfolio = from_signals(
        close=prices,
        entries=entries,
        exits=exits,
        init_cash=100000.0,
        fees=0.0,
        slippage=0.0,
        freq='1D',
        market_hours_only=False,
        risk_config=config_stop
    )

    logger.blank()
    logger.separator("-", 80)
    logger.header("STOP LOSS RESULTS")
    logger.separator("-", 80)

    exit_trades = [t for t in portfolio.trades if t['type'] == 'exit']

    if exit_trades:
        exit_trade = exit_trades[0]
        exit_reason = exit_trade.get('exit_reason', 'unknown')

        logger.metric(f"Exit triggered: {exit_reason}")

        if 'stop' in exit_reason.lower() or 'loss' in exit_reason.lower():
            logger.success("✓ Stop loss TRIGGERED correctly")

            # Check the loss amount
            pnl_pct = exit_trade.get('pnl_pct', 0)
            logger.metric(f"Loss: {pnl_pct:.2f}%")

            if -3.0 < pnl_pct < -1.0:  # Should be close to -2%
                logger.success(f"✓ Stop loss amount CORRECT (~2% loss)")
            else:
                logger.warning(f"⚠ Stop loss amount may be off: {pnl_pct:.2f}%")
        else:
            logger.error(f"✗ Stop loss did NOT trigger (exit reason: {exit_reason})")
    else:
        logger.error("✗ No exit trade found - stop loss may not be working")

    logger.blank()


def test_risk_profiles():
    """Test different risk profiles."""
    logger.separator("=", 80)
    logger.header("RISK PROFILE VALIDATION")
    logger.separator("=", 80)
    logger.blank()

    prices, entries, exits = create_simple_signals(100)

    profiles = {
        'Conservative': RiskConfig.conservative(),
        'Moderate': RiskConfig.moderate(),
        'Aggressive': RiskConfig.aggressive()
    }

    results = {}

    for name, config in profiles.items():
        logger.info(f"Testing {name} profile ({config.position_size_pct*100:.0f}% per trade)...")

        portfolio = from_signals(
            close=prices,
            entries=entries,
            exits=exits,
            init_cash=100000.0,
            fees=0.0,
            slippage=0.0,
            freq='1D',
            market_hours_only=False,
            risk_config=config
        )

        entry_trades = [t for t in portfolio.trades if t['type'] == 'entry']
        if entry_trades:
            first_trade = entry_trades[0]
            results[name] = {
                'position_size': first_trade['cost'],
                'shares': first_trade['shares'],
                'pct_of_capital': (first_trade['cost'] / 100000 * 100)
            }

    logger.blank()
    logger.separator("-", 80)
    logger.header("PROFILE COMPARISON")
    logger.separator("-", 80)

    logger.metric(f"{'Profile':<15} {'Position Size':>15} {'% of Capital':>15} {'Shares':>12}")
    logger.separator("-", 80)

    for name, data in results.items():
        logger.metric(
            f"{name:<15} ${data['position_size']:>14,.2f} "
            f"{data['pct_of_capital']:>14.1f}% {data['shares']:>12.0f}"
        )

    logger.blank()

    # Verify scaling
    if len(results) == 3:
        cons_size = results['Conservative']['position_size']
        mod_size = results['Moderate']['position_size']
        agg_size = results['Aggressive']['position_size']

        if cons_size < mod_size < agg_size:
            logger.success("✓ Risk profiles scale correctly (Conservative < Moderate < Aggressive)")
        else:
            logger.error("✗ Risk profiles NOT scaling correctly")

    logger.blank()


def main():
    """Run all validation tests."""
    logger.blank()
    logger.separator("=", 80)
    logger.header("RISK MANAGEMENT VALIDATION (SIMPLE)")
    logger.separator("=", 80)
    logger.blank()

    test_position_sizing()
    test_stop_losses()
    test_risk_profiles()

    logger.separator("=", 80)
    logger.header("VALIDATION COMPLETE ✓")
    logger.separator("=", 80)
    logger.blank()


if __name__ == '__main__':
    main()
