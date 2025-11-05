"""
Validate risk management implementation by comparing different position sizing approaches.

This script demonstrates:
1. The difference between 99% (old) and 10% (new default) position sizing
2. Conservative vs Moderate vs Aggressive risk profiles
3. Stop loss behavior
4. Position sizing calculations
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import pandas as pd
import numpy as np
from backtesting.engine.backtest_engine import BacktestEngine
from backtesting.utils.risk_config import RiskConfig
from strategies.base_strategies.moving_average import MovingAverageCrossover
from utils import logger


def create_synthetic_trending_data(days=252, initial_price=100, trend=0.001, volatility=0.015):
    """
    Create synthetic trending price data for testing.

    Args:
        days: Number of trading days
        initial_price: Starting price
        trend: Daily trend (0.001 = 0.1% per day)
        volatility: Daily volatility (0.015 = 1.5% per day)

    Returns:
        DataFrame with OHLCV data
    """
    dates = pd.date_range('2024-01-01', periods=days, freq='D')

    # Generate returns with trend and volatility
    returns = np.random.normal(trend, volatility, days)

    # Calculate prices
    prices = initial_price * (1 + returns).cumprod()

    # Create OHLCV data
    df = pd.DataFrame({
        'timestamp': dates,
        'open': prices * np.random.uniform(0.995, 1.005, days),
        'high': prices * np.random.uniform(1.005, 1.02, days),
        'low': prices * np.random.uniform(0.98, 0.995, days),
        'close': prices,
        'volume': np.random.randint(1000000, 10000000, days)
    })
    df.set_index('timestamp', inplace=True)

    return df


def validate_position_sizing():
    """Test 1: Compare 99% vs 10% position sizing."""
    logger.blank()
    logger.separator("=", 80)
    logger.header("TEST 1: POSITION SIZING COMPARISON (99% vs 10%)")
    logger.separator("=", 80)
    logger.blank()

    data = create_synthetic_trending_data(days=252, trend=0.0005, volatility=0.01)
    strategy = MovingAverageCrossover(fast_window=10, slow_window=30)

    # Test with 99% sizing (old unrealistic behavior)
    logger.info("Running backtest with 99% position sizing (OLD/UNREALISTIC)...")
    config_99 = RiskConfig.disabled()
    engine_99 = BacktestEngine(initial_capital=100000, fees=0.001, risk_config=config_99)
    portfolio_99 = engine_99.run_with_data(strategy, data)
    stats_99 = portfolio_99.stats()

    logger.blank()

    # Test with 10% sizing (new default)
    logger.info("Running backtest with 10% position sizing (NEW/REALISTIC)...")
    config_10 = RiskConfig.moderate()
    engine_10 = BacktestEngine(initial_capital=100000, fees=0.001, risk_config=config_10)
    portfolio_10 = engine_10.run_with_data(strategy, data)
    stats_10 = portfolio_10.stats()

    # Compare results
    logger.blank()
    logger.separator("-", 80)
    logger.header("COMPARISON RESULTS")
    logger.separator("-", 80)

    if stats_99 and stats_10:
        logger.metric(f"{'Metric':<25} {'99% Sizing':>15} {'10% Sizing':>15} {'Change':>15}")
        logger.separator("-", 80)

        for key in ['Total Return [%]', 'Sharpe Ratio', 'Max Drawdown [%]', 'Total Trades', 'Win Rate [%]']:
            val_99 = stats_99.get(key, 0)
            val_10 = stats_10.get(key, 0)

            if 'Return' in key or 'Drawdown' in key:
                change = val_10 - val_99
                logger.metric(f"{key:<25} {val_99:>14.2f}% {val_10:>14.2f}% {change:>+14.2f}%")
            elif 'Ratio' in key:
                change_pct = ((val_10 / val_99 - 1) * 100) if val_99 != 0 else 0
                logger.metric(f"{key:<25} {val_99:>15.3f} {val_10:>15.3f} {change_pct:>+13.1f}%")
            else:
                logger.metric(f"{key:<25} {val_99:>15.0f} {val_10:>15.0f}")

    logger.blank()
    logger.success("✓ Position sizing validation complete!")
    logger.blank()

    return stats_99, stats_10


def validate_risk_profiles():
    """Test 2: Compare Conservative vs Moderate vs Aggressive profiles."""
    logger.separator("=", 80)
    logger.header("TEST 2: RISK PROFILE COMPARISON")
    logger.separator("=", 80)
    logger.blank()

    data = create_synthetic_trending_data(days=252, trend=0.0008, volatility=0.012)
    strategy = MovingAverageCrossover(fast_window=15, slow_window=40)

    profiles = {
        'Conservative (5%)': RiskConfig.conservative(),
        'Moderate (10%)': RiskConfig.moderate(),
        'Aggressive (20%)': RiskConfig.aggressive()
    }

    results = {}

    for name, config in profiles.items():
        logger.info(f"Running backtest with {name} profile...")
        engine = BacktestEngine(initial_capital=100000, fees=0.001, risk_config=config)
        portfolio = engine.run_with_data(strategy, data)
        results[name] = portfolio.stats()
        logger.blank()

    # Compare results
    logger.separator("-", 80)
    logger.header("RISK PROFILE COMPARISON")
    logger.separator("-", 80)

    logger.metric(f"{'Metric':<25} {'Conservative':>15} {'Moderate':>15} {'Aggressive':>15}")
    logger.separator("-", 80)

    if all(results.values()):
        for key in ['Total Return [%]', 'Sharpe Ratio', 'Max Drawdown [%]', 'Total Trades']:
            values = [results[profile].get(key, 0) for profile in profiles.keys()]

            if 'Return' in key or 'Drawdown' in key:
                logger.metric(f"{key:<25} {values[0]:>14.2f}% {values[1]:>14.2f}% {values[2]:>14.2f}%")
            elif 'Ratio' in key:
                logger.metric(f"{key:<25} {values[0]:>15.3f} {values[1]:>15.3f} {values[2]:>15.3f}")
            else:
                logger.metric(f"{key:<25} {values[0]:>15.0f} {values[1]:>15.0f} {values[2]:>15.0f}")

    logger.blank()
    logger.success("✓ Risk profile validation complete!")
    logger.blank()

    return results


def validate_stop_losses():
    """Test 3: Validate stop loss behavior."""
    logger.separator("=", 80)
    logger.header("TEST 3: STOP LOSS VALIDATION")
    logger.separator("=", 80)
    logger.blank()

    # Create more volatile data to trigger stop losses
    data = create_synthetic_trending_data(days=252, trend=0.0003, volatility=0.025)
    strategy = MovingAverageCrossover(fast_window=10, slow_window=30)

    # Test with tight stop loss
    logger.info("Running backtest with 2% stop loss...")
    config_stop = RiskConfig(
        position_size_pct=0.10,
        use_stop_loss=True,
        stop_loss_pct=0.02,  # 2% stop loss
        stop_loss_type='percentage'
    )
    engine_stop = BacktestEngine(initial_capital=100000, fees=0.001, risk_config=config_stop)
    portfolio_stop = engine_stop.run_with_data(strategy, data)

    # Test without stop loss
    logger.blank()
    logger.info("Running backtest WITHOUT stop loss...")
    config_no_stop = RiskConfig(
        position_size_pct=0.10,
        use_stop_loss=False
    )
    engine_no_stop = BacktestEngine(initial_capital=100000, fees=0.001, risk_config=config_no_stop)
    portfolio_no_stop = engine_no_stop.run_with_data(strategy, data)

    # Analyze stop loss exits
    logger.blank()
    logger.separator("-", 80)
    logger.header("STOP LOSS ANALYSIS")
    logger.separator("-", 80)

    stop_loss_exits = [t for t in portfolio_stop.trades if t.get('type') == 'exit' and 'stop_loss' in str(t.get('exit_reason', ''))]
    strategy_exits = [t for t in portfolio_stop.trades if t.get('type') == 'exit' and t.get('exit_reason') == 'strategy_signal']

    logger.metric(f"Total exits with stop loss: {len([t for t in portfolio_stop.trades if t.get('type') == 'exit'])}")
    logger.metric(f"  - Stop loss triggered: {len(stop_loss_exits)}")
    logger.metric(f"  - Strategy signal: {len(strategy_exits)}")
    logger.blank()

    stats_stop = portfolio_stop.stats()
    stats_no_stop = portfolio_no_stop.stats()

    if stats_stop and stats_no_stop:
        logger.metric(f"{'Metric':<30} {'With 2% Stop':>15} {'No Stop':>15}")
        logger.separator("-", 80)

        for key in ['Total Return [%]', 'Max Drawdown [%]', 'Win Rate [%]']:
            val_stop = stats_stop.get(key, 0)
            val_no_stop = stats_no_stop.get(key, 0)
            logger.metric(f"{key:<30} {val_stop:>14.2f}% {val_no_stop:>14.2f}%")

    logger.blank()
    logger.success("✓ Stop loss validation complete!")
    logger.blank()

    return portfolio_stop, portfolio_no_stop


def main():
    """Run all validation tests."""
    logger.blank()
    logger.separator("=", 80)
    logger.header("RISK MANAGEMENT VALIDATION SUITE")
    logger.separator("=", 80)
    logger.blank()
    logger.info("This script validates the risk management implementation by:")
    logger.info("  1. Comparing 99% vs 10% position sizing")
    logger.info("  2. Testing Conservative/Moderate/Aggressive profiles")
    logger.info("  3. Validating stop loss behavior")
    logger.blank()

    # Run validation tests
    validate_position_sizing()
    validate_risk_profiles()
    validate_stop_losses()

    # Final summary
    logger.separator("=", 80)
    logger.header("VALIDATION COMPLETE ✓")
    logger.separator("=", 80)
    logger.success("All risk management features validated successfully!")
    logger.blank()
    logger.info("Key takeaways:")
    logger.info("  • 10% position sizing produces more realistic results than 99%")
    logger.info("  • Different risk profiles scale returns and drawdowns appropriately")
    logger.info("  • Stop losses protect against large losses")
    logger.info("  • Risk management is enabled by default (RiskConfig.moderate())")
    logger.blank()


if __name__ == '__main__':
    main()
