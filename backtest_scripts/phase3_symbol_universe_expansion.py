"""
Phase 3: Test expanded symbol universe with optimal parameters from Phase 2.

Tests:
1. Best config on all 46 symbols (excluding TSLL, NVDL due to limited history)
2. Individual symbol quality assessment
3. Sector-based portfolios
4. Top N symbol selection
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import warnings

from utils.path_setup import setup_project_paths
ROOT_DIR = setup_project_paths()
warnings.filterwarnings('ignore')

from src.utils.logger import logger

# Import from optimize script
exec(open('backtest_scripts/optimize_overnight_strategy.py').read())

# Symbol categories
BASELINE_23 = [
    'TQQQ', 'SQQQ', 'UPRO', 'SPXU', 'UDOW', 'SDOW',
    'TNA', 'TZA', 'SOXL', 'SOXS', 'FAS', 'FAZ',
    'LABU', 'LABD', 'TECL', 'TECS',
    'QLD', 'QID', 'SSO', 'SDS',
    'UVXY', 'SVXY', 'VIXY'
]

ADDITIONAL_23 = [
    'CURE', 'CUT', 'ERX', 'ERY', 'RETL', 'WEBL',
    'DPST', 'DFEN', 'NAIL', 'DUST',
    'UWM', 'TWM', 'UGL', 'GLL', 'UYG', 'SKF', 'USD',
    'UCO', 'SCO', 'BIB', 'BIS'
]

# Exclude TSLL and NVDL due to limited history
ALL_46_SYMBOLS = BASELINE_23 + ADDITIONAL_23

# Sector groupings
SECTORS = {
    'Tech': ['TQQQ', 'SQQQ', 'SOXL', 'SOXS', 'TECL', 'TECS', 'QLD', 'QID', 'USD'],
    'Broad Market': ['UPRO', 'SPXU', 'UDOW', 'SDOW', 'TNA', 'TZA', 'SSO', 'SDS'],
    'Financials': ['FAS', 'FAZ', 'UYG', 'SKF', 'DPST'],
    'Healthcare': ['LABU', 'LABD', 'CURE', 'CUT', 'BIB', 'BIS'],
    'Energy': ['ERX', 'ERY', 'UCO', 'SCO'],
    'Commodities': ['UGL', 'GLL', 'DUST'],
    'Real Estate': ['UWM', 'TWM'],
    'Retail': ['RETL', 'NAIL'],
    'Volatility': ['UVXY', 'SVXY', 'VIXY'],
    'Other': ['WEBL', 'DFEN']
}

# Optimal config from Phase 2
OPTIMAL_CONFIG = {
    'max_position_size': 0.15,
    'max_total_exposure': 0.50,
    'max_loss_per_trade': -0.02,
    'max_concurrent_positions': 3,
    'vix_threshold': 35,
    'min_win_rate': 0.58,
    'min_expected_return': 0.002,
    'min_sample_size': 15,
    'skip_regimes': ['BEAR'],
    'sizing_method': 'fixed'
}


def test_individual_symbols(data, regime_detector, bayesian_model, test_start, test_end):
    """Test each symbol individually to identify high-quality ones."""

    logger.info("\n" + "="*80)
    logger.info("TESTING INDIVIDUAL SYMBOLS")
    logger.info("="*80)

    individual_results = []

    for symbol in ALL_46_SYMBOLS:
        if symbol not in data:
            logger.warning(f"  {symbol}: Data not available")
            continue

        trades_df = backtest_strategy(
            data, regime_detector, bayesian_model,
            test_start, test_end, [symbol], OPTIMAL_CONFIG, symbol
        )

        if trades_df.empty:
            logger.info(f"  {symbol}: No qualifying trades")
            continue

        result = analyze_results(trades_df, symbol, OPTIMAL_CONFIG)

        if result:
            individual_results.append(result)
            logger.info(f"  {symbol}: Return {result['total_return']:.1%}, "
                       f"Sharpe {result['sharpe_ratio']:.2f}, "
                       f"Win Rate {result['win_rate']:.1%}, "
                       f"Trades {result['total_trades']}")

    # Sort by Sharpe ratio
    individual_results.sort(key=lambda x: x['sharpe_ratio'], reverse=True)

    logger.info("\n" + "="*80)
    logger.info("TOP 10 INDIVIDUAL SYMBOLS BY SHARPE")
    logger.info("="*80)

    for i, r in enumerate(individual_results[:10], 1):
        logger.success(f"{i}. {r['name']}: Return {r['total_return']:.1%}, "
                      f"Sharpe {r['sharpe_ratio']:.2f}, Win Rate {r['win_rate']:.1%}")

    return individual_results


def test_portfolio_variations(data, regime_detector, bayesian_model, test_start, test_end, individual_results):
    """Test different portfolio compositions."""

    logger.info("\n" + "="*80)
    logger.info("TESTING PORTFOLIO VARIATIONS")
    logger.info("="*80)

    portfolio_results = []

    # 1. All 46 symbols
    logger.info("\n[1] Testing all 46 symbols...")
    trades_df = backtest_strategy(
        data, regime_detector, bayesian_model,
        test_start, test_end, ALL_46_SYMBOLS, OPTIMAL_CONFIG, "All 46 ETFs"
    )
    result = analyze_results(trades_df, "All 46 ETFs", OPTIMAL_CONFIG)
    if result:
        portfolio_results.append(result)
        logger.info(f"  Return: {result['total_return']:.1%}, Sharpe: {result['sharpe_ratio']:.2f}, "
                   f"Max DD: {result['max_drawdown']:.1%}, Trades: {result['total_trades']}")

    # 2. Top 10 symbols by individual Sharpe
    top_10_symbols = [r['name'] for r in individual_results[:10]]
    logger.info(f"\n[2] Testing top 10 symbols by Sharpe: {', '.join(top_10_symbols)}")
    trades_df = backtest_strategy(
        data, regime_detector, bayesian_model,
        test_start, test_end, top_10_symbols, OPTIMAL_CONFIG, "Top 10 by Sharpe"
    )
    result = analyze_results(trades_df, "Top 10 by Sharpe", OPTIMAL_CONFIG)
    if result:
        portfolio_results.append(result)
        logger.info(f"  Return: {result['total_return']:.1%}, Sharpe: {result['sharpe_ratio']:.2f}, "
                   f"Max DD: {result['max_drawdown']:.1%}, Trades: {result['total_trades']}")

    # 3. Top 20 symbols
    top_20_symbols = [r['name'] for r in individual_results[:20]]
    logger.info(f"\n[3] Testing top 20 symbols by Sharpe...")
    trades_df = backtest_strategy(
        data, regime_detector, bayesian_model,
        test_start, test_end, top_20_symbols, OPTIMAL_CONFIG, "Top 20 by Sharpe"
    )
    result = analyze_results(trades_df, "Top 20 by Sharpe", OPTIMAL_CONFIG)
    if result:
        portfolio_results.append(result)
        logger.info(f"  Return: {result['total_return']:.1%}, Sharpe: {result['sharpe_ratio']:.2f}, "
                   f"Max DD: {result['max_drawdown']:.1%}, Trades: {result['total_trades']}")

    # 4. Top 30 symbols
    top_30_symbols = [r['name'] for r in individual_results[:30]]
    logger.info(f"\n[4] Testing top 30 symbols by Sharpe...")
    trades_df = backtest_strategy(
        data, regime_detector, bayesian_model,
        test_start, test_end, top_30_symbols, OPTIMAL_CONFIG, "Top 30 by Sharpe"
    )
    result = analyze_results(trades_df, "Top 30 by Sharpe", OPTIMAL_CONFIG)
    if result:
        portfolio_results.append(result)
        logger.info(f"  Return: {result['total_return']:.1%}, Sharpe: {result['sharpe_ratio']:.2f}, "
                   f"Max DD: {result['max_drawdown']:.1%}, Trades: {result['total_trades']}")

    # 5. Baseline 23 for comparison
    logger.info(f"\n[5] Testing baseline 23 symbols (for comparison)...")
    trades_df = backtest_strategy(
        data, regime_detector, bayesian_model,
        test_start, test_end, BASELINE_23, OPTIMAL_CONFIG, "Baseline 23"
    )
    result = analyze_results(trades_df, "Baseline 23", OPTIMAL_CONFIG)
    if result:
        portfolio_results.append(result)
        logger.info(f"  Return: {result['total_return']:.1%}, Sharpe: {result['sharpe_ratio']:.2f}, "
                   f"Max DD: {result['max_drawdown']:.1%}, Trades: {result['total_trades']}")

    # 6. Test each sector
    for sector_name, sector_symbols in SECTORS.items():
        available_symbols = [s for s in sector_symbols if s in data]
        if len(available_symbols) < 2:
            continue

        logger.info(f"\n[Sector] Testing {sector_name} ({len(available_symbols)} symbols)...")
        trades_df = backtest_strategy(
            data, regime_detector, bayesian_model,
            test_start, test_end, available_symbols, OPTIMAL_CONFIG, f"Sector: {sector_name}"
        )
        result = analyze_results(trades_df, f"Sector: {sector_name}", OPTIMAL_CONFIG)
        if result:
            portfolio_results.append(result)
            logger.info(f"  Return: {result['total_return']:.1%}, Sharpe: {result['sharpe_ratio']:.2f}, "
                       f"Max DD: {result['max_drawdown']:.1%}, Trades: {result['total_trades']}")

    return portfolio_results


def main():
    logger.info("\n" + "="*80)
    logger.info("PHASE 3: SYMBOL UNIVERSE EXPANSION")
    logger.info("="*80)
    logger.info(f"\nTesting with optimal config from Phase 2:")
    logger.info(f"  Position Size: {OPTIMAL_CONFIG['max_position_size']:.0%}")
    logger.info(f"  Stop-Loss: {OPTIMAL_CONFIG['max_loss_per_trade']:.0%}")
    logger.info(f"  Min Win Rate: {OPTIMAL_CONFIG['min_win_rate']:.0%}")
    logger.info(f"  VIX Threshold: {OPTIMAL_CONFIG['vix_threshold']}")
    logger.info(f"  Max Positions: {OPTIMAL_CONFIG['max_concurrent_positions']}")

    # Load data for all symbols
    logger.info(f"\nLoading data for {len(ALL_46_SYMBOLS)} symbols...")
    data = load_data(ALL_46_SYMBOLS)
    if data is None:
        return

    # Initialize models
    regime_detector = SimpleRegimeDetector()

    # Train model on all symbols
    train_end = pd.Timestamp('2023-12-31')
    test_start = pd.Timestamp('2024-01-01')
    test_end = data['SPY'].index[-1]

    logger.info(f"\nTraining Bayesian model on all {len(ALL_46_SYMBOLS)} symbols...")
    bayesian_model = SimpleBayesianModel()
    bayesian_model.train(data, regime_detector, data['SPY'], data['^VIX'], train_end, ALL_46_SYMBOLS)

    # Test individual symbols
    individual_results = test_individual_symbols(
        data, regime_detector, bayesian_model, test_start, test_end
    )

    # Save individual results
    save_progress(individual_results, "phase3_individual_symbols.csv")

    # Test portfolio variations
    portfolio_results = test_portfolio_variations(
        data, regime_detector, bayesian_model, test_start, test_end, individual_results
    )

    # Save portfolio results
    save_progress(portfolio_results, "phase3_portfolio_variations.csv")

    # Combined results
    all_results = individual_results + portfolio_results

    # Summary report
    logger.info("\n" + "="*80)
    logger.info("PHASE 3 SUMMARY")
    logger.info("="*80)

    portfolio_results.sort(key=lambda x: x['sharpe_ratio'], reverse=True)

    logger.info("\nTop 5 Portfolios by Sharpe:")
    for i, r in enumerate(portfolio_results[:5], 1):
        logger.success(f"{i}. {r['name']}: Return {r['total_return']:.1%}, "
                      f"Sharpe {r['sharpe_ratio']:.2f}, Max DD {r['max_drawdown']:.1%}")

    # Save detailed report
    summary_path = REPORTS_DIR / "20251112_PHASE3_SYMBOL_UNIVERSE.md"
    with open(summary_path, 'w') as f:
        f.write("# Phase 3: Symbol Universe Expansion\n\n")
        f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Symbols Tested**: {len(ALL_46_SYMBOLS)}\n")
        f.write(f"**Optimal Config**: Pos 15%, Stop -2%, WR 58%, VIX 35, Max Pos 3\n\n")

        f.write("## Top 10 Individual Symbols\n\n")
        f.write("| Rank | Symbol | Return | Sharpe | Win Rate | Max DD | Trades |\n")
        f.write("|------|--------|--------|--------|----------|--------|--------|\n")

        for i, r in enumerate(individual_results[:10], 1):
            f.write(f"| {i} | {r['name']} | {r['total_return']:.1%} | {r['sharpe_ratio']:.2f} | "
                   f"{r['win_rate']:.1%} | {r['max_drawdown']:.1%} | {r['total_trades']} |\n")

        f.write("\n## Portfolio Results\n\n")
        f.write("| Portfolio | Return | Sharpe | Win Rate | Max DD | Trades |\n")
        f.write("|-----------|--------|--------|----------|--------|--------|\n")

        for r in portfolio_results:
            f.write(f"| {r['name']} | {r['total_return']:.1%} | {r['sharpe_ratio']:.2f} | "
                   f"{r['win_rate']:.1%} | {r['max_drawdown']:.1%} | {r['total_trades']} |\n")

    logger.success(f"\nSaved detailed report to {summary_path}")

    logger.info("\n" + "="*80)
    logger.info("PHASE 3 COMPLETE")
    logger.info("="*80)


if __name__ == "__main__":
    main()
