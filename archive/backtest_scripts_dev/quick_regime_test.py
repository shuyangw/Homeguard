"""
Quick test of regime-based modules with minimal data.
"""

import sys
from pathlib import Path

import pandas as pd
import numpy as np
from utils import logger

from utils.path_setup import setup_project_paths
ROOT_DIR = setup_project_paths()

logger.info("Starting quick regime module test...")

# Test 1: Regime Detection
logger.blank()
logger.header("TEST 1: Regime Detection")
logger.blank()

from backtesting.regimes.detector import TrendDetector, VolatilityDetector

# Create simple price data
dates = pd.date_range('2020-01-01', periods=200, freq='D')
prices = pd.Series(
    100 + np.linspace(0, 50, 200) + np.random.randn(200) * 2,
    index=dates
)

logger.info("Detecting trend regimes...")
trend_detector = TrendDetector(lookback_days=60, threshold_pct=5.0)
trend_regimes = trend_detector.detect(prices)

logger.success(f"Found {len(trend_regimes)} trend regime periods:")
for regime in trend_regimes[:3]:  # Show first 3
    logger.info(f"  {regime.regime.value}: {regime.start_date} to {regime.end_date}")

logger.info("Detecting volatility regimes...")
vol_detector = VolatilityDetector(lookback_days=20)
vol_regimes = vol_detector.detect(prices)

logger.success(f"Found {len(vol_regimes)} volatility regime periods:")
for regime in vol_regimes[:3]:  # Show first 3
    logger.info(f"  {regime.regime.value}: {regime.start_date} to {regime.end_date}")

# Test 2: Regime Analysis
logger.blank()
logger.header("TEST 2: Regime Analysis")
logger.blank()

from backtesting.regimes.analyzer import RegimeAnalyzer

# Create simple returns
returns = pd.Series(np.random.randn(200) * 0.01, index=dates)

logger.info("Running regime analysis...")
analyzer = RegimeAnalyzer()
results = analyzer.analyze(
    portfolio_returns=returns,
    market_prices=prices,
    trades=None
)

logger.success("Analysis complete!")
logger.info(f"Robustness Score: {results.robustness_score:.1f}/100")
logger.info(f"Best Regime: {results.best_regime}")
logger.info(f"Worst Regime: {results.worst_regime}")

# Test 3: Walk-Forward Window Generation (no actual backtesting)
logger.blank()
logger.header("TEST 3: Walk-Forward Window Generation")
logger.blank()

from backtesting.chunking.walk_forward import WalkForwardValidator
from backtesting.engine.backtest_engine import BacktestEngine

logger.info("Creating walk-forward validator...")
engine = BacktestEngine(initial_capital=10000, fees=0.001)
validator = WalkForwardValidator(
    engine=engine,
    train_months=6,
    test_months=3,
    step_months=3
)

logger.info("Generating windows...")
windows = validator.generate_windows(
    start_date='2020-01-01',
    end_date='2021-12-31'
)

logger.success(f"Generated {len(windows)} walk-forward windows:")
for i, window in enumerate(windows[:3], 1):  # Show first 3
    logger.info(f"  Window {i}:")
    logger.info(f"    Train: {window.train_start} to {window.train_end}")
    logger.info(f"    Test:  {window.test_start} to {window.test_end}")

logger.blank()
logger.separator()
logger.success("All quick tests passed! ✓")
logger.separator()
logger.blank()
