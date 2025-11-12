"""
Phase 4: Test advanced position sizing methods.

Methods to test:
1. Fixed (baseline from Phase 2)
2. Probability-weighted (scale by win probability)
3. Expected return weighted (scale by expected return)
4. Kelly Criterion (optimal sizing based on edge)
5. Volatility-adjusted (larger positions in stable symbols)
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.logger import logger

# Import from optimize script
exec(open('backtest_scripts/optimize_overnight_strategy.py').read())

# Use Top 20 symbols from Phase 3
TOP_20_SYMBOLS = [
    'FAZ', 'USD', 'UDOW', 'UYG', 'SOXL', 'TECL', 'UPRO', 'SVXY', 'TQQQ', 'SSO',
    'DFEN', 'WEBL', 'UCO', 'NAIL', 'LABU', 'TNA', 'SQQQ', 'ERX', 'RETL', 'CUT'
]

# Base config from Phase 2
BASE_CONFIG = {
    'max_position_size': 0.15,
    'max_total_exposure': 0.50,
    'max_loss_per_trade': -0.02,
    'max_concurrent_positions': 3,
    'vix_threshold': 35,
    'min_win_rate': 0.58,
    'min_expected_return': 0.002,
    'min_sample_size': 15,
    'skip_regimes': ['BEAR'],
}


def main():
    logger.info("\n" + "="*80)
    logger.info("PHASE 4: ADVANCED POSITION SIZING")
    logger.info("="*80)
    logger.info(f"\nTesting on Top 20 symbols from Phase 3")
    logger.info(f"Base position size: {BASE_CONFIG['max_position_size']:.0%}")

    # Load data
    logger.info(f"\nLoading data for {len(TOP_20_SYMBOLS)} symbols...")
    data = load_data(TOP_20_SYMBOLS)
    if data is None:
        return

    # Initialize models
    regime_detector = SimpleRegimeDetector()

    # Train model
    train_end = pd.Timestamp('2023-12-31')
    test_start = pd.Timestamp('2024-01-01')
    test_end = data['SPY'].index[-1]

    logger.info(f"\nTraining Bayesian model on Top 20 symbols...")
    bayesian_model = SimpleBayesianModel()
    bayesian_model.train(data, regime_detector, data['SPY'], data['^VIX'], train_end, TOP_20_SYMBOLS)

    # Test different sizing methods
    sizing_methods = [
        ('fixed', 'Fixed 15%'),
        ('probability_weighted', 'Probability-Weighted'),
        ('expected_return_weighted', 'Expected Return Weighted'),
        ('kelly', 'Kelly Criterion'),
    ]

    results = []

    for method, name in sizing_methods:
        logger.info(f"\n{'='*80}")
        logger.info(f"Testing: {name}")
        logger.info("="*80)

        config = BASE_CONFIG.copy()
        config['sizing_method'] = method

        trades_df = backtest_strategy(
            data, regime_detector, bayesian_model,
            test_start, test_end, TOP_20_SYMBOLS, config, name
        )

        result = analyze_results(trades_df, name, config)

        if result:
            results.append(result)
            logger.info(f"  Return: {result['total_return']:.1%}, Sharpe: {result['sharpe_ratio']:.2f}, "
                       f"Win Rate: {result['win_rate']:.1%}, Max DD: {result['max_drawdown']:.1%}, "
                       f"Trades: {result['total_trades']}")

            # Show position size statistics
            if not trades_df.empty:
                avg_pos = trades_df['position_size'].mean()
                min_pos = trades_df['position_size'].min()
                max_pos = trades_df['position_size'].max()
                logger.info(f"  Position sizes - Avg: {avg_pos:.1%}, Min: {min_pos:.1%}, Max: {max_pos:.1%}")

    # Comparison
    logger.info("\n" + "="*80)
    logger.info("SIZING METHOD COMPARISON")
    logger.info("="*80)

    results.sort(key=lambda x: x['sharpe_ratio'], reverse=True)

    logger.info(f"{'Method':<30} {'Return':<10} {'Sharpe':<8} {'Win%':<7} {'MaxDD':<8} {'Trades':<8}")
    logger.info("-"*80)

    for r in results:
        logger.info(
            f"{r['name']:<30} {r['total_return']:<10.1%} {r['sharpe_ratio']:<8.2f} "
            f"{r['win_rate']:<7.1%} {r['max_drawdown']:<8.1%} {r['total_trades']:<8}"
        )

    # Save results
    save_progress(results, "phase4_position_sizing.csv")

    # Best method
    best = results[0]
    logger.info("\n" + "="*80)
    logger.info("BEST SIZING METHOD")
    logger.info("="*80)
    logger.success(f"  Method: {best['name']}")
    logger.success(f"  Return: {best['total_return']:.1%}")
    logger.success(f"  Sharpe: {best['sharpe_ratio']:.2f}")
    logger.success(f"  Max DD: {best['max_drawdown']:.1%}")

    # Save detailed report
    summary_path = REPORTS_DIR / "20251112_PHASE4_POSITION_SIZING.md"
    with open(summary_path, 'w') as f:
        f.write("# Phase 4: Advanced Position Sizing\n\n")
        f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Symbols**: Top 20 from Phase 3\n")
        f.write(f"**Base Config**: Pos 15%, Stop -2%, WR 58%, VIX 35, Max Pos 3\n\n")

        f.write("## Results\n\n")
        f.write("| Method | Return | Sharpe | Win Rate | Max DD | Trades |\n")
        f.write("|--------|--------|--------|----------|--------|--------|\n")

        for r in results:
            f.write(f"| {r['name']} | {r['total_return']:.1%} | {r['sharpe_ratio']:.2f} | "
                   f"{r['win_rate']:.1%} | {r['max_drawdown']:.1%} | {r['total_trades']} |\n")

        f.write(f"\n## Best Method\n\n")
        f.write(f"**{best['name']}**\n\n")
        f.write(f"- Total Return: {best['total_return']:.1%}\n")
        f.write(f"- Sharpe Ratio: {best['sharpe_ratio']:.2f}\n")
        f.write(f"- Win Rate: {best['win_rate']:.1%}\n")
        f.write(f"- Max Drawdown: {best['max_drawdown']:.1%}\n")
        f.write(f"- Total Trades: {best['total_trades']}\n")

    logger.success(f"\nSaved detailed report to {summary_path}")

    logger.info("\n" + "="*80)
    logger.info("PHASE 4 COMPLETE")
    logger.info("="*80)


if __name__ == "__main__":
    main()
