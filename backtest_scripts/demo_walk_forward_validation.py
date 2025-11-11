"""
Demonstration of Walk-Forward Validation Framework

This script demonstrates how to use walk-forward validation to:
1. Test parameter stability across different time periods
2. Detect overfitting (train Sharpe >> test Sharpe)
3. Find robust parameters that work across multiple windows

Walk-forward validation is ESSENTIAL for production trading systems.
It prevents the "optimization on entire dataset" trap that leads to
strategy failure in live trading.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from backtesting.engine.backtest_engine import BacktestEngine
from backtesting.optimization import GridSearchOptimizer, WalkForwardOptimizer
from strategies.base_strategies.moving_average import MovingAverageCrossover
from strategies.base_strategies.mean_reversion import RSIMeanReversion
from backtesting.utils.risk_config import RiskConfig
from utils import logger


def demo_walk_forward_ma_crossover():
    """
    Demonstrate walk-forward validation on MA Crossover strategy.

    Tests parameter stability from 2020-2024 using 12-month training
    windows and 6-month test windows, rolling forward every 6 months.
    """
    logger.blank()
    logger.separator()
    logger.header("WALK-FORWARD VALIDATION DEMO")
    logger.info("Strategy: Moving Average Crossover")
    logger.info("Period: 2020-01-01 to 2024-01-01")
    logger.separator()
    logger.blank()

    # Create engine with moderate risk
    engine = BacktestEngine(
        initial_capital=100000,
        fees=0.001,
        slippage=0.0005
    )
    engine.risk_config = RiskConfig.moderate()

    # Create base optimizer (Grid Search)
    base_optimizer = GridSearchOptimizer(engine)

    # Create walk-forward optimizer
    wf_optimizer = WalkForwardOptimizer(engine, base_optimizer)

    # Parameter grid (small for demo speed)
    param_grid = {
        'fast_window': [20, 30, 40],
        'slow_window': [100, 150, 200],
        'ma_type': ['sma', 'ema']
    }

    logger.info("Parameter Grid:")
    logger.info(f"  fast_window: {param_grid['fast_window']}")
    logger.info(f"  slow_window: {param_grid['slow_window']}")
    logger.info(f"  ma_type: {param_grid['ma_type']}")
    logger.info(f"  Total combinations: {3 * 3 * 2} = 18")
    logger.blank()

    logger.info("Window Configuration:")
    logger.info("  Training: 12 months")
    logger.info("  Testing: 6 months")
    logger.info("  Step: 6 months (no overlap)")
    logger.blank()

    # Run walk-forward analysis
    results = wf_optimizer.analyze(
        strategy_class=MovingAverageCrossover,
        param_space=param_grid,
        symbols='AAPL',
        start_date='2020-01-01',
        end_date='2024-01-01',
        train_months=12,
        test_months=6,
        step_months=6,
        metric='sharpe_ratio',
        export_results=True,
        max_workers=4  # Parallel optimization
    )

    # Interpretation guide
    logger.blank()
    logger.separator()
    logger.header("INTERPRETATION GUIDE")
    logger.separator()
    logger.blank()

    logger.info("Degradation Thresholds:")
    logger.blank()
    logger.profit("  < 0.3: ✅ EXCELLENT (parameters are robust)")
    logger.success("  0.3-0.5: ✅ GOOD (acceptable degradation)")
    logger.warning("  0.5-0.8: ⚠️  CONCERNING (overfitting risk)")
    logger.error("  > 0.8: ❌ SEVERE (parameters overfit to training)")
    logger.blank()

    logger.info("What to do based on results:")
    logger.blank()
    if results.get('avg_degradation', float('inf')) < 0.5:
        logger.profit("✅ Parameters are stable - safe for live trading")
        logger.info("   → Use 'best_stable_params' for production")
        logger.info("   → Monitor performance monthly")
        logger.info("   → Re-optimize if Sharpe degrades by > 0.3")
    else:
        logger.error("❌ Parameters are not stable - DO NOT trade live")
        logger.info("   → Try different parameter ranges")
        logger.info("   → Test on different symbols")
        logger.info("   → Consider simpler strategy")
        logger.info("   → May need more data for optimization")

    logger.blank()
    logger.separator()


def demo_walk_forward_rsi():
    """
    Demonstrate walk-forward validation on RSI Mean Reversion strategy.

    Uses shorter windows (6 months train, 3 months test) suitable for
    faster-moving mean reversion strategies.
    """
    logger.blank()
    logger.separator()
    logger.header("WALK-FORWARD VALIDATION: RSI MEAN REVERSION")
    logger.info("Period: 2020-01-01 to 2024-01-01")
    logger.separator()
    logger.blank()

    # Create engine
    engine = BacktestEngine(
        initial_capital=100000,
        fees=0.001,
        slippage=0.0005
    )
    engine.risk_config = RiskConfig.moderate()

    # Create optimizers
    base_optimizer = GridSearchOptimizer(engine)
    wf_optimizer = WalkForwardOptimizer(engine, base_optimizer)

    # Parameter grid (small for demo)
    param_grid = {
        'rsi_window': [10, 14, 21],
        'oversold': [25, 30],
        'overbought': [70, 75]
    }

    logger.info("Parameter Grid:")
    logger.info(f"  rsi_window: {param_grid['rsi_window']}")
    logger.info(f"  oversold: {param_grid['oversold']}")
    logger.info(f"  overbought: {param_grid['overbought']}")
    logger.info(f"  Total combinations: {3 * 2 * 2} = 12")
    logger.blank()

    logger.info("Window Configuration (Mean Reversion):")
    logger.info("  Training: 6 months (shorter for faster strategies)")
    logger.info("  Testing: 3 months")
    logger.info("  Step: 3 months")
    logger.blank()

    # Run walk-forward analysis
    results = wf_optimizer.analyze(
        strategy_class=RSIMeanReversion,
        param_space=param_grid,
        symbols='NVDA',  # High volatility stock good for mean reversion
        start_date='2020-01-01',
        end_date='2024-01-01',
        train_months=6,   # Shorter for mean reversion
        test_months=3,
        step_months=3,
        metric='sharpe_ratio',
        export_results=True,
        max_workers=4
    )

    logger.blank()
    logger.info("Mean reversion strategies often show higher degradation")
    logger.info("than trend-following because market regimes change faster.")
    logger.info("Degradation < 0.6 is acceptable for mean reversion.")
    logger.blank()
    logger.separator()


def compare_with_without_walk_forward():
    """
    Demonstrate the danger of optimizing on entire dataset vs walk-forward.
    """
    logger.blank()
    logger.separator()
    logger.header("COMPARISON: Standard Optimization vs Walk-Forward")
    logger.separator()
    logger.blank()

    engine = BacktestEngine(
        initial_capital=100000,
        fees=0.001,
        slippage=0.0005
    )
    engine.risk_config = RiskConfig.moderate()

    param_grid = {
        'fast_window': [20, 30, 40],
        'slow_window': [100, 150, 200],
        'ma_type': ['sma', 'ema']
    }

    # Method 1: Standard optimization on entire period (WRONG!)
    logger.header("❌ METHOD 1: Optimize on Entire Period (WRONG)")
    logger.blank()
    logger.warning("This is what most beginners do...")
    logger.warning("It leads to overfit parameters that fail in live trading!")
    logger.blank()

    optimizer = GridSearchOptimizer(engine)
    entire_result = optimizer.optimize_parallel(
        strategy_class=MovingAverageCrossover,
        param_grid=param_grid,
        symbols='AAPL',
        start_date='2020-01-01',
        end_date='2024-01-01',
        metric='sharpe_ratio',
        max_workers=4,
        export_results=False
    )

    logger.success(f"Sharpe Ratio: {entire_result['best_value']:.4f}")
    logger.metric(f"Best Parameters: {entire_result['best_params']}")
    logger.blank()
    logger.error("BUT... this Sharpe is OVERFIT to the entire period!")
    logger.error("In live trading, it will likely be much worse.")
    logger.blank()

    # Method 2: Walk-forward validation (CORRECT!)
    logger.header("✅ METHOD 2: Walk-Forward Validation (CORRECT)")
    logger.blank()
    logger.success("This is the professional way...")
    logger.success("It gives realistic expectations for live trading!")
    logger.blank()

    wf_optimizer = WalkForwardOptimizer(engine, optimizer)
    wf_result = wf_optimizer.analyze(
        strategy_class=MovingAverageCrossover,
        param_space=param_grid,
        symbols='AAPL',
        start_date='2020-01-01',
        end_date='2024-01-01',
        train_months=12,
        test_months=6,
        step_months=6,
        metric='sharpe_ratio',
        export_results=True,
        max_workers=4
    )

    logger.blank()
    logger.separator()
    logger.header("REALITY CHECK")
    logger.separator()
    logger.blank()

    entire_sharpe = entire_result['best_value']
    if 'avg_degradation' in wf_result:
        avg_test_sharpe = entire_sharpe - wf_result['avg_degradation']

        logger.metric(f"Optimized on entire period: {entire_sharpe:.4f}")
        logger.metric(f"Walk-forward avg test Sharpe: {avg_test_sharpe:.4f}")
        logger.metric(f"Degradation: {wf_result['avg_degradation']:.4f}")
        logger.blank()

        if wf_result['avg_degradation'] > 0.5:
            logger.error("⚠️  Large degradation! Parameters are overfit.")
            logger.error("   The 'entire period' Sharpe is misleading.")
            logger.error(f"   Realistic live trading Sharpe: ~{avg_test_sharpe:.2f}")
        else:
            logger.profit("✅ Low degradation! Parameters are robust.")
            logger.profit(f"   Expected live trading Sharpe: ~{avg_test_sharpe:.2f}")

    logger.blank()
    logger.separator()


if __name__ == '__main__':
    # Choose which demo to run
    import sys

    if len(sys.argv) > 1:
        demo = sys.argv[1]
        if demo == 'ma':
            demo_walk_forward_ma_crossover()
        elif demo == 'rsi':
            demo_walk_forward_rsi()
        elif demo == 'compare':
            compare_with_without_walk_forward()
        else:
            logger.error(f"Unknown demo: {demo}")
            logger.info("Usage: python demo_walk_forward_validation.py [ma|rsi|compare]")
    else:
        # Run all demos
        demo_walk_forward_ma_crossover()
        logger.blank()
        logger.blank()
        demo_walk_forward_rsi()
        logger.blank()
        logger.blank()
        compare_with_without_walk_forward()
