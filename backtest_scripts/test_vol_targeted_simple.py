"""
Simple test of VolatilityTargetedMomentum with minimal parameters.
Quick diagnostic to see if strategy works at all.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from backtesting.engine.backtest_engine import BacktestEngine
from strategies.advanced.volatility_targeted_momentum import VolatilityTargetedMomentum
from backtesting.utils.risk_config import RiskConfig
from utils import logger

def simple_test():
    """Run a single backtest with default parameters."""
    logger.header("SIMPLE TEST - VolatilityTargetedMomentum")
    logger.blank()

    # Create engine
    engine = BacktestEngine(
        initial_capital=100000,
        fees=0.001,
        slippage=0.0005
    )
    engine.risk_config = RiskConfig.moderate()

    # Create strategy with default parameters
    strategy = VolatilityTargetedMomentum()

    logger.info("Testing with default parameters:")
    logger.info(f"  lookback_period: {strategy.params['lookback_period']}")
    logger.info(f"  ma_window: {strategy.params['ma_window']}")
    logger.info(f"  vol_window: {strategy.params['vol_window']}")
    logger.info(f"  target_vol: {strategy.params['target_vol']}")
    logger.info(f"  max_leverage: {strategy.params['max_leverage']}")
    logger.blank()

    # Run backtest
    logger.info("Running backtest on AAPL (2019-2021)...")

    try:
        result = engine.run(
            strategy=strategy,
            symbols='AAPL',
            start_date='2019-01-01',
            end_date='2021-12-31'
        )

        logger.success("Backtest completed successfully!")
        logger.blank()

        # Get stats
        stats = result.stats()
        if stats:
            logger.header("RESULTS")
            logger.profit(f"Total Return: {stats.get('total_return', 0):.2%}")
            logger.profit(f"Sharpe Ratio: {stats.get('sharpe_ratio', 0):.4f}")
            logger.info(f"Max Drawdown: {stats.get('max_drawdown', 0):.2%}")
            logger.info(f"Number of Trades: {stats.get('num_trades', 0)}")
            logger.info(f"Win Rate: {stats.get('win_rate', 0):.2%}")
        else:
            logger.warning("No stats available")

        logger.blank()

        # Decision
        sharpe = stats.get('sharpe_ratio', -999) if stats else -999
        if sharpe > 0.3:
            logger.success(f"PROMISING: Sharpe {sharpe:.4f} > 0.3")
        elif sharpe > 0.0:
            logger.info(f"MARGINAL: Sharpe {sharpe:.4f} > 0 but < 0.3")
        else:
            logger.error(f"FAILED: Sharpe {sharpe:.4f} < 0")

    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True

if __name__ == '__main__':
    try:
        success = simple_test()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
