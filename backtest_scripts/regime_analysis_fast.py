"""
FAST proof-of-concept script for regime-based testing and walk-forward validation.

Demonstrates:
1. Walk-forward validation to prevent overfitting
2. Regime-based analysis to identify failure conditions
3. Robustness scoring across market conditions

FAST VERSION:
- Uses DAILY data instead of intraday (500 bars instead of 387,732)
- Completes all 3 examples in ~2-3 minutes instead of 15+ minutes
- Identical functionality, just faster execution
- Perfect for demonstrations and quick validation

NOTE: Daily data is less precise than intraday but sufficient for:
- Strategy validation
- Parameter optimization
- Regime detection
- Overfitting assessment
"""

import sys
from pathlib import Path

import time
import pandas as pd
from backtesting.engine.backtest_engine import BacktestEngine
from backtesting.chunking.walk_forward import WalkForwardValidator

from utils.path_setup import setup_project_paths
ROOT_DIR = setup_project_paths()
from backtesting.regimes.analyzer import RegimeAnalyzer
from strategies.base_strategies.moving_average import MovingAverageCrossover
from utils import logger

# Fast parameters
START_DATE = '2022-01-01'
END_DATE = '2023-12-31'
SYMBOL = 'AAPL'

# Global cache
_cached_engine = None
_cached_daily_data = None


def get_engine():
    """Get cached engine instance."""
    global _cached_engine
    if _cached_engine is None:
        logger.info("Initializing backtesting engine...")
        _cached_engine = BacktestEngine(initial_capital=10000, fees=0.001)
    return _cached_engine


def get_daily_data():
    """Get cached daily data (resampled from intraday)."""
    global _cached_daily_data
    if _cached_daily_data is None:
        logger.info(f"Loading market data for {SYMBOL} ({START_DATE} to {END_DATE})...")
        start_time = time.time()

        # Load intraday data
        engine = get_engine()
        intraday_data = engine.data_loader.load_symbols(
            [SYMBOL],
            START_DATE,
            END_DATE
        )

        logger.info(f"Resampling {len(intraday_data)} intraday bars to daily...")

        # Resample to daily OHLCV
        # Group by symbol level, then resample on timestamp level
        daily_dfs = []
        for symbol in intraday_data.index.get_level_values('symbol').unique():
            symbol_data = intraday_data.xs(symbol, level='symbol')

            # Resample OHLCV appropriately
            daily = symbol_data.resample('D').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()

            # Add symbol level back
            daily['symbol'] = symbol
            daily = daily.reset_index().set_index(['symbol', 'timestamp'])
            daily_dfs.append(daily)

        _cached_daily_data = pd.concat(daily_dfs).sort_index()

        elapsed = time.time() - start_time
        logger.success(f"Daily data ready: {len(_cached_daily_data)} bars in {elapsed:.1f}s")

    return _cached_daily_data


def example_1_walk_forward_validation():
    """
    Example 1: Walk-Forward Validation

    Prevents overfitting by testing on out-of-sample data.
    FAST: Uses daily data for quick execution.
    """
    logger.blank()
    logger.separator()
    logger.header("EXAMPLE 1: WALK-FORWARD VALIDATION (DAILY DATA)")
    logger.separator()
    logger.blank()

    start_time = time.time()

    # Get engine
    engine = get_engine()

    # Pre-load daily data
    daily_data = get_daily_data()

    # Temporarily replace engine's data loader to return our daily data
    original_load_method = engine.data_loader.load_symbols

    def load_daily_data(symbols, start, end, timeframe='1min'):
        """Return cached daily data."""
        return daily_data

    engine.data_loader.load_symbols = load_daily_data

    # Create walk-forward validator
    logger.info("Setting up walk-forward validator...")
    validator = WalkForwardValidator(
        engine=engine,
        train_months=6,
        test_months=3,
        step_months=3
    )

    # Define parameter grid
    param_grid = {
        'fast_window': [10, 20],
        'slow_window': [50, 100]
    }

    num_combinations = len(param_grid['fast_window']) * len(param_grid['slow_window'])
    logger.info(f"Testing {num_combinations} parameter combinations")
    logger.info("Running walk-forward validation...")

    results = validator.validate(
        strategy_class=MovingAverageCrossover,
        param_grid=param_grid,
        symbols=SYMBOL,
        start_date=START_DATE,
        end_date=END_DATE,
        metric='sharpe_ratio'
    )

    # Restore original method
    engine.data_loader.load_symbols = original_load_method

    elapsed = time.time() - start_time

    # Results printed automatically
    logger.blank()
    logger.success(f"Walk-forward validation complete in {elapsed:.1f}s!")
    logger.info(f"Out-of-sample Sharpe: {results.out_of_sample_sharpe:.2f}")
    logger.info(f"Performance degradation: {results.degradation_pct:.1f}%")

    if abs(results.degradation_pct) > 20:
        logger.warning("WARNING: Strategy may be overfit!")
    else:
        logger.success("Strategy appears robust to out-of-sample data")

    logger.blank()
    return results


def example_2_regime_analysis():
    """
    Example 2: Regime-Based Analysis

    Analyzes performance across different market conditions.
    FAST: Uses daily data.
    """
    logger.blank()
    logger.separator()
    logger.header("EXAMPLE 2: REGIME-BASED ANALYSIS (DAILY DATA)")
    logger.separator()
    logger.blank()

    start_time = time.time()

    # Get engine
    engine = get_engine()

    # Get daily data
    daily_data = get_daily_data()

    # Temporarily replace data loader
    original_load_method = engine.data_loader.load_symbols

    def load_daily_data(symbols, start, end, timeframe='1min'):
        return daily_data

    engine.data_loader.load_symbols = load_daily_data

    # Create strategy
    strategy = MovingAverageCrossover(fast_window=20, slow_window=100)

    # Run backtest
    logger.info("Running backtest...")
    portfolio = engine.run(
        strategy=strategy,
        symbols=[SYMBOL],
        start_date=START_DATE,
        end_date=END_DATE,
        price_type='close'
    )

    # Restore original method
    engine.data_loader.load_symbols = original_load_method

    # Get returns
    returns = portfolio.returns()

    # Get market prices
    market_prices = daily_data.xs(SYMBOL, level='symbol')['close']

    # Create regime analyzer
    logger.info("Analyzing performance by market regime...")
    analyzer = RegimeAnalyzer(
        trend_lookback=60,
        vol_lookback=20,
        drawdown_threshold=10.0
    )

    # Analyze by regime
    regime_results = analyzer.analyze(
        portfolio_returns=returns,
        market_prices=market_prices,
        trades=None
    )

    elapsed = time.time() - start_time

    # Results printed automatically
    logger.blank()
    logger.success(f"Regime analysis complete in {elapsed:.1f}s!")
    logger.info(f"Robustness score: {regime_results.robustness_score:.1f}/100")
    logger.info(f"Best regime: {regime_results.best_regime}")
    logger.info(f"Worst regime: {regime_results.worst_regime}")

    if regime_results.robustness_score >= 70:
        logger.success("Strategy is highly robust across market conditions")
    elif regime_results.robustness_score >= 50:
        logger.info("Strategy shows reasonable consistency")
    else:
        logger.warning("Strategy performance varies significantly by regime")

    logger.blank()
    return regime_results


def example_3_combined_analysis():
    """
    Example 3: Combined Walk-Forward + Regime Analysis

    The ultimate validation.
    FAST: Uses daily data.
    """
    logger.blank()
    logger.separator()
    logger.header("EXAMPLE 3: COMBINED ANALYSIS (DAILY DATA)")
    logger.separator()
    logger.blank()

    start_time = time.time()

    # Get engine and data
    engine = get_engine()
    daily_data = get_daily_data()

    # Temporarily replace data loader
    original_load_method = engine.data_loader.load_symbols

    def load_daily_data(symbols, start, end, timeframe='1min'):
        return daily_data

    engine.data_loader.load_symbols = load_daily_data

    # Step 1: Walk-forward validation
    logger.header("Step 1: Walk-Forward Validation")
    logger.info("Running walk-forward validation...")

    validator = WalkForwardValidator(
        engine=engine,
        train_months=6,
        test_months=3,
        step_months=3
    )

    param_grid = {
        'fast_window': [10, 20],
        'slow_window': [50, 100]
    }

    wf_results = validator.validate(
        strategy_class=MovingAverageCrossover,
        param_grid=param_grid,
        symbols=SYMBOL,
        start_date=START_DATE,
        end_date=END_DATE,
        metric='sharpe_ratio'
    )

    logger.blank()

    # Step 2: Regime analysis on out-of-sample returns
    logger.header("Step 2: Regime Analysis on Out-of-Sample Performance")
    logger.info("Analyzing only the out-of-sample test periods...")

    # Get market prices
    market_prices = daily_data.xs(SYMBOL, level='symbol')['close']

    # Create regime analyzer
    analyzer = RegimeAnalyzer()

    # Analyze out-of-sample returns by regime
    regime_results = analyzer.analyze(
        portfolio_returns=wf_results.oos_returns,
        market_prices=market_prices,
        trades=None
    )

    # Restore original method
    engine.data_loader.load_symbols = original_load_method

    elapsed = time.time() - start_time
    logger.blank()

    # Step 3: Final verdict
    logger.separator()
    logger.header("FINAL VERDICT")
    logger.separator()
    logger.blank()

    logger.info(f"Total Time: {elapsed:.1f}s")
    logger.info(f"Out-of-Sample Sharpe: {wf_results.out_of_sample_sharpe:.2f}")
    logger.info(f"Performance Degradation: {wf_results.degradation_pct:.1f}%")
    logger.info(f"Robustness Score: {regime_results.robustness_score:.1f}/100")
    logger.blank()

    # Determine if strategy is production-ready
    is_robust = abs(wf_results.degradation_pct) < 20
    is_consistent = regime_results.robustness_score >= 60

    if is_robust and is_consistent:
        logger.success("✓ PASS: Strategy is production-ready!")
        logger.success("  - Low overfitting risk")
        logger.success("  - Consistent across market regimes")
    elif is_robust:
        logger.warning("⚠ CONDITIONAL: Low overfitting but inconsistent across regimes")
        logger.warning("  - Consider regime-specific position sizing")
    elif is_consistent:
        logger.warning("⚠ CONDITIONAL: Consistent but may be overfit")
        logger.warning("  - Consider longer walk-forward windows")
    else:
        logger.error("✗ FAIL: Strategy not production-ready")
        logger.error("  - High overfitting risk")
        logger.error("  - Inconsistent across regimes")

    logger.blank()
    logger.separator()
    return wf_results, regime_results


def main():
    """Run all examples."""
    logger.blank()
    logger.separator()
    logger.header("REGIME-BASED TESTING PROOF-OF-CONCEPT (FAST VERSION)")
    logger.separator()
    logger.info("This script demonstrates advanced validation techniques:")
    logger.info("  1. Walk-Forward Validation (prevents overfitting)")
    logger.info("  2. Regime-Based Analysis (identifies failure conditions)")
    logger.info("  3. Combined Analysis (production readiness assessment)")
    logger.blank()
    logger.info("FAST VERSION FEATURES:")
    logger.info("  - Uses DAILY data instead of intraday")
    logger.info("  - ~500 bars instead of 387,732 bars")
    logger.info("  - Completes in 2-3 minutes instead of 15+ minutes")
    logger.info("  - Perfect for demonstrations and quick validation")
    logger.separator()
    logger.blank()

    # Let user select which examples to run
    if len(sys.argv) > 1:
        example_num = sys.argv[1]
        if example_num == '1':
            examples_to_run = [1]
        elif example_num == '2':
            examples_to_run = [2]
        elif example_num == '3':
            examples_to_run = [3]
        else:
            logger.error(f"Invalid example number: {example_num}")
            logger.info("Usage: python regime_analysis_fast.py [1|2|3|all]")
            logger.info("  1 = Walk-forward validation only")
            logger.info("  2 = Regime analysis only")
            logger.info("  3 = Combined analysis only")
            logger.info("  all or no argument = Run all examples")
            return 1
    else:
        examples_to_run = [1, 2, 3]

    logger.info(f"Running example(s): {examples_to_run}")
    logger.blank()

    total_start_time = time.time()

    try:
        if 1 in examples_to_run:
            example_1_walk_forward_validation()

        if 2 in examples_to_run:
            example_2_regime_analysis()

        if 3 in examples_to_run:
            example_3_combined_analysis()

        total_elapsed = time.time() - total_start_time

        logger.blank()
        logger.separator()
        logger.success(f"All examples completed successfully in {total_elapsed:.1f}s!")
        logger.info(f"Average time per example: {total_elapsed / len(examples_to_run):.1f}s")
        logger.separator()
        logger.blank()

    except Exception as e:
        logger.error(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
