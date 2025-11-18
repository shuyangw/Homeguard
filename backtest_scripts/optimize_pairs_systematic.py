"""
Systematic Pairs Trading Optimization

Goal: Achieve production-ready Sharpe >= 0.8 for top performing pairs.

Methodology:
1. Start with sensitivity analysis (one parameter at a time)
2. Full grid search around promising regions
3. Document all findings incrementally
4. Save best parameters for each pair

This script is designed for LONG-RUNNING OPTIMIZATION (2-8+ hours).
Progress is saved incrementally to avoid data loss.
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
from datetime import datetime
import json
from itertools import product
import time

from utils.path_setup import setup_project_paths
ROOT_DIR = setup_project_paths()

# Add project root to path
project_root = Path(__file__).parent.parent

from src.backtesting.engine.backtest_engine import BacktestEngine
from src.strategies.advanced.pairs_trading import PairsTrading
from src.utils import logger
from src.config import get_backtest_results_dir, PROJECT_ROOT


# Top 8 pairs from validation (ordered by Sharpe ratio)
TOP_PAIRS = [
    ("XLY", "UVXY", 0.551, 34.91),  # (symbol1, symbol2, baseline_sharpe, baseline_return)
    ("XLI", "UVXY", 0.518, 34.62),
    ("DIA", "UVXY", 0.517, 34.75),
    ("VTI", "UVXY", 0.507, 30.76),
    ("VOO", "UVXY", 0.498, 34.68),
    ("UNG", "UVXY", 0.475, 42.41),
    ("XLK", "UVXY", 0.469, 27.78),
    ("XLF", "UVXY", 0.457, 32.55),
]

# Parameter grid
PARAM_GRID = {
    'entry_zscore': [1.5, 1.75, 2.0, 2.25, 2.5],
    'exit_zscore': [0.0, 0.25, 0.5, 0.75],
    'stop_loss_zscore': [3.0, 3.5, 4.0],
    'zscore_window': [15, 20, 25, 30]
}

# Backtest configuration
START_DATE = "2023-01-01"
END_DATE = "2024-11-11"
INITIAL_CAPITAL = 100000
FEES = 0.0001  # 0.01%
SLIPPAGE = 0.001  # 0.10%


def run_single_backtest(symbol1, symbol2, params):
    """
    Run a single backtest with given parameters.

    Args:
        symbol1: First symbol
        symbol2: Second symbol
        params: Dictionary with strategy parameters

    Returns:
        Dictionary with results or None if failed
    """
    try:
        # Create strategy
        strategy = PairsTrading(
            pair_selection_window=252,
            cointegration_pvalue=0.05,
            entry_zscore=params['entry_zscore'],
            exit_zscore=params['exit_zscore'],
            stop_loss_zscore=params['stop_loss_zscore'],
            zscore_window=params['zscore_window']
        )

        # Create engine
        engine = BacktestEngine(
            initial_capital=INITIAL_CAPITAL,
            fees=FEES,
            slippage=SLIPPAGE
        )

        # Run backtest
        portfolio = engine.run(
            strategy=strategy,
            symbols=[symbol1, symbol2],
            start_date=START_DATE,
            end_date=END_DATE
        )

        # Get statistics
        stats = portfolio.stats()

        if stats is None:
            return None

        return {
            'sharpe': stats['Sharpe Ratio'],
            'return': stats['Total Return [%]'],
            'annual_return': stats['Annual Return [%]'],
            'max_dd': stats['Max Drawdown [%]'],
            'win_rate': stats['Win Rate [%]'],
            'trades': stats['Total Trades'],
            'final_equity': stats['End Value'],
            'success': True
        }

    except Exception as e:
        logger.error(f"Error in backtest: {str(e)}")
        return None


def sensitivity_analysis(symbol1, symbol2, baseline_sharpe, baseline_return):
    """
    Perform sensitivity analysis - vary one parameter at a time.

    This helps identify which parameters have the most impact.

    Args:
        symbol1: First symbol
        symbol2: Second symbol
        baseline_sharpe: Baseline Sharpe ratio (for comparison)
        baseline_return: Baseline return (for comparison)

    Returns:
        DataFrame with sensitivity results
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"SENSITIVITY ANALYSIS: {symbol1}/{symbol2}")
    logger.info(f"Baseline: Sharpe {baseline_sharpe:.3f}, Return {baseline_return:.2f}%")
    logger.info(f"{'='*80}\n")

    # Baseline parameters
    baseline = {
        'entry_zscore': 2.0,
        'exit_zscore': 0.5,
        'stop_loss_zscore': 3.5,
        'zscore_window': 20
    }

    results = []

    # Vary each parameter one at a time
    for param_name, param_values in PARAM_GRID.items():
        logger.info(f"\nTesting {param_name}...")

        for value in param_values:
            # Create test params (baseline + one changed param)
            test_params = baseline.copy()
            test_params[param_name] = value

            # Run backtest
            result = run_single_backtest(symbol1, symbol2, test_params)

            if result:
                results.append({
                    'pair': f"{symbol1}/{symbol2}",
                    'varied_param': param_name,
                    'varied_value': value,
                    **test_params,
                    **result
                })
                logger.info(f"  {param_name}={value}: Sharpe={result['sharpe']:.3f}, Return={result['return']:.2f}%, Trades={result['trades']}")
            else:
                logger.error(f"  {param_name}={value}: FAILED")

    df = pd.DataFrame(results)

    # Save sensitivity results
    output_path = get_backtest_results_dir() / f"sensitivity_{symbol1}_{symbol2}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"\n[+] Sensitivity results saved: {output_path}")

    # Analyze sensitivity
    logger.info(f"\nSENSITIVITY SUMMARY:")
    for param_name in PARAM_GRID.keys():
        param_df = df[df['varied_param'] == param_name]
        if len(param_df) > 0:
            best = param_df.loc[param_df['sharpe'].idxmax()]
            logger.info(f"  {param_name}: Best={best['varied_value']} (Sharpe={best['sharpe']:.3f})")

    return df


def grid_search_optimization(symbol1, symbol2, baseline_sharpe, baseline_return):
    """
    Perform full grid search optimization.

    Tests all parameter combinations systematically.

    Args:
        symbol1: First symbol
        symbol2: Second symbol
        baseline_sharpe: Baseline Sharpe ratio
        baseline_return: Baseline return

    Returns:
        DataFrame with all results
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"GRID SEARCH OPTIMIZATION: {symbol1}/{symbol2}")
    logger.info(f"Baseline: Sharpe {baseline_sharpe:.3f}, Return {baseline_return:.2f}%")
    logger.info(f"{'='*80}\n")

    # Generate all parameter combinations
    param_names = list(PARAM_GRID.keys())
    param_values = [PARAM_GRID[name] for name in param_names]
    combinations = list(product(*param_values))

    total_combos = len(combinations)
    logger.info(f"Total combinations to test: {total_combos}")
    logger.info(f"Estimated time: {total_combos * 30 / 60:.1f} minutes\n")

    results = []
    start_time = time.time()

    for idx, combo in enumerate(combinations, 1):
        # Create parameter dict
        params = dict(zip(param_names, combo))

        # Progress update every 10 tests
        if idx % 10 == 0:
            elapsed = time.time() - start_time
            tests_per_sec = idx / elapsed
            remaining = (total_combos - idx) / tests_per_sec
            pct_complete = 100 * idx / total_combos
            logger.info(f"[{idx}/{total_combos}] {pct_complete:.1f}% complete, ETA: {remaining/60:.1f} min")

        # Run backtest
        result = run_single_backtest(symbol1, symbol2, params)

        if result:
            results.append({
                'pair': f"{symbol1}/{symbol2}",
                **params,
                **result
            })

            # Highlight if production ready
            if result['sharpe'] >= 0.8:
                logger.info(f"  [PRODUCTION READY] Sharpe={result['sharpe']:.3f} with {params}")

    df = pd.DataFrame(results)

    # Sort by Sharpe ratio
    df = df.sort_values('sharpe', ascending=False)

    # Save full results
    output_path = get_backtest_results_dir() / f"grid_search_{symbol1}_{symbol2}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"\n[+] Grid search results saved: {output_path}")

    # Show top 10
    logger.info(f"\nTOP 10 CONFIGURATIONS:")
    for idx, row in df.head(10).iterrows():
        logger.info(f"  {idx+1}. Sharpe={row['sharpe']:.3f}, Return={row['return']:.2f}%, "
                   f"Entry={row['entry_zscore']}, Exit={row['exit_zscore']}, "
                   f"Stop={row['stop_loss_zscore']}, Window={row['zscore_window']}, "
                   f"Trades={row['trades']}")

    # Check if any production ready
    production_ready = df[df['sharpe'] >= 0.8]
    if len(production_ready) > 0:
        logger.info(f"\n[+] {len(production_ready)} PRODUCTION READY configurations found!")
        for idx, row in production_ready.iterrows():
            logger.info(f"  - Sharpe={row['sharpe']:.3f}: Entry={row['entry_zscore']}, "
                       f"Exit={row['exit_zscore']}, Stop={row['stop_loss_zscore']}, "
                       f"Window={row['zscore_window']}")
    else:
        best = df.iloc[0]
        gap = 0.8 - best['sharpe']
        pct_gap = 100 * gap / 0.8
        logger.info(f"\n[~] No production ready configs. Best Sharpe={best['sharpe']:.3f} "
                   f"({pct_gap:.1f}% below target)")

    return df


def update_progress_chronicle(pair, phase, results_summary):
    """
    Update the progress chronicle document.

    Args:
        pair: Pair identifier (e.g., "XLY/UVXY")
        phase: Phase name (e.g., "Sensitivity Analysis")
        results_summary: Dictionary with key findings
    """
    chronicle_path = PROJECT_ROOT / "docs" / "reports" / "OPTIMIZATION_PROGRESS.md"
    chronicle_path.parent.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    update = f"""
### {timestamp} - {pair} - {phase}

**Status**: {results_summary.get('status', 'Complete')}
**Best Sharpe**: {results_summary.get('best_sharpe', 'N/A')}
**Best Return**: {results_summary.get('best_return', 'N/A')}
**Best Parameters**: {results_summary.get('best_params', 'N/A')}
**Total Tests**: {results_summary.get('total_tests', 'N/A')}
**Production Ready**: {results_summary.get('production_ready', 'No')}

{results_summary.get('notes', '')}

---
"""

    # Append to chronicle
    with open(chronicle_path, 'a') as f:
        f.write(update)

    logger.info(f"[+] Progress chronicle updated: {chronicle_path}")


def optimize_single_pair(symbol1, symbol2, baseline_sharpe, baseline_return,
                         sensitivity_only=False):
    """
    Optimize a single pair.

    Args:
        symbol1: First symbol
        symbol2: Second symbol
        baseline_sharpe: Baseline Sharpe ratio
        baseline_return: Baseline return
        sensitivity_only: If True, only run sensitivity analysis

    Returns:
        Dictionary with optimization results
    """
    pair_name = f"{symbol1}/{symbol2}"
    logger.info(f"\n{'#'*80}")
    logger.info(f"# OPTIMIZING PAIR: {pair_name}")
    logger.info(f"# Baseline: Sharpe {baseline_sharpe:.3f}, Return {baseline_return:.2f}%")
    logger.info(f"# Target: Sharpe >= 0.8")
    logger.info(f"{'#'*80}\n")

    start_time = time.time()

    # Phase 1: Sensitivity Analysis
    logger.info("[PHASE 1] Sensitivity Analysis")
    sensitivity_df = sensitivity_analysis(symbol1, symbol2, baseline_sharpe, baseline_return)

    # Update chronicle
    best_sens = sensitivity_df.loc[sensitivity_df['sharpe'].idxmax()]
    update_progress_chronicle(
        pair_name,
        "Phase 1: Sensitivity Analysis",
        {
            'status': 'Complete',
            'best_sharpe': f"{best_sens['sharpe']:.3f}",
            'best_return': f"{best_sens['return']:.2f}%",
            'best_params': f"Entry={best_sens['entry_zscore']}, Exit={best_sens['exit_zscore']}, "
                          f"Stop={best_sens['stop_loss_zscore']}, Window={best_sens['zscore_window']}",
            'total_tests': len(sensitivity_df),
            'production_ready': 'Yes' if best_sens['sharpe'] >= 0.8 else 'No',
            'notes': 'Varied one parameter at a time to identify most impactful parameters.'
        }
    )

    if sensitivity_only:
        elapsed = time.time() - start_time
        logger.info(f"\n[+] Sensitivity analysis complete in {elapsed/60:.1f} minutes")
        return {'sensitivity_df': sensitivity_df}

    # Phase 2: Full Grid Search
    logger.info("\n[PHASE 2] Full Grid Search Optimization")
    grid_df = grid_search_optimization(symbol1, symbol2, baseline_sharpe, baseline_return)

    # Update chronicle
    best_grid = grid_df.iloc[0]
    production_ready = grid_df[grid_df['sharpe'] >= 0.8]

    update_progress_chronicle(
        pair_name,
        "Phase 2: Full Grid Search",
        {
            'status': 'Complete',
            'best_sharpe': f"{best_grid['sharpe']:.3f}",
            'best_return': f"{best_grid['return']:.2f}%",
            'best_params': f"Entry={best_grid['entry_zscore']}, Exit={best_grid['exit_zscore']}, "
                          f"Stop={best_grid['stop_loss_zscore']}, Window={best_grid['zscore_window']}",
            'total_tests': len(grid_df),
            'production_ready': f"Yes ({len(production_ready)} configs)" if len(production_ready) > 0 else 'No',
            'notes': f"Tested all {len(grid_df)} parameter combinations. "
                    f"Best improved from baseline {baseline_sharpe:.3f} to {best_grid['sharpe']:.3f} "
                    f"({100*(best_grid['sharpe']-baseline_sharpe)/baseline_sharpe:.1f}% improvement)."
        }
    )

    elapsed = time.time() - start_time
    logger.info(f"\n[+] Full optimization complete in {elapsed/60:.1f} minutes")

    return {
        'sensitivity_df': sensitivity_df,
        'grid_df': grid_df,
        'best_params': {
            'entry_zscore': best_grid['entry_zscore'],
            'exit_zscore': best_grid['exit_zscore'],
            'stop_loss_zscore': best_grid['stop_loss_zscore'],
            'zscore_window': best_grid['zscore_window']
        },
        'best_sharpe': best_grid['sharpe'],
        'best_return': best_grid['return'],
        'production_ready': len(production_ready) > 0
    }


def main():
    """Main optimization routine."""
    logger.info("="*80)
    logger.info("SYSTEMATIC PAIRS TRADING OPTIMIZATION")
    logger.info("Goal: Achieve Sharpe >= 0.8 for production readiness")
    logger.info("="*80)

    # Start with top 4 pairs (closest to target)
    priority_pairs = TOP_PAIRS[:4]

    logger.info(f"\nPRIORITY PAIRS (Top 4):")
    for idx, (sym1, sym2, sharpe, ret) in enumerate(priority_pairs, 1):
        logger.info(f"  {idx}. {sym1}/{sym2}: Sharpe {sharpe:.3f}, Return {ret:.2f}%")

    # Optimization strategy
    logger.info(f"\nOPTIMIZATION STRATEGY:")
    logger.info("  1. Start with XLY/UVXY (best baseline Sharpe)")
    logger.info("  2. Full sensitivity + grid search")
    logger.info("  3. If production ready found, test next pair")
    logger.info("  4. Document all findings incrementally")
    logger.info("\nStarting optimization (estimated 2-8 hours)...")

    all_results = []

    # Optimize priority pairs
    for idx, (sym1, sym2, baseline_sharpe, baseline_return) in enumerate(priority_pairs, 1):
        logger.info(f"\n{'='*80}")
        logger.info(f"PAIR {idx}/{len(priority_pairs)}: {sym1}/{sym2}")
        logger.info(f"{'='*80}")

        result = optimize_single_pair(sym1, sym2, baseline_sharpe, baseline_return)

        all_results.append({
            'pair': f"{sym1}/{sym2}",
            'baseline_sharpe': baseline_sharpe,
            'baseline_return': baseline_return,
            'optimized_sharpe': result['best_sharpe'],
            'optimized_return': result['best_return'],
            'improvement_pct': 100 * (result['best_sharpe'] - baseline_sharpe) / baseline_sharpe,
            'production_ready': result['production_ready'],
            **result['best_params']
        })

        # Save incremental results
        results_df = pd.DataFrame(all_results)
        output_path = get_backtest_results_dir() / "pairs_optimization_results.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(output_path, index=False)
        logger.info(f"\n[+] Incremental results saved: {output_path}")

    # Final summary
    logger.info(f"\n{'='*80}")
    logger.info("OPTIMIZATION COMPLETE")
    logger.info(f"{'='*80}\n")

    results_df = pd.DataFrame(all_results)

    logger.info("FINAL RESULTS:")
    for _, row in results_df.iterrows():
        status = "[PRODUCTION READY]" if row['production_ready'] else "[NEEDS WORK]"
        logger.info(f"  {status} {row['pair']}: Sharpe {row['baseline_sharpe']:.3f} -> {row['optimized_sharpe']:.3f} "
                   f"({row['improvement_pct']:.1f}% improvement)")

    production_ready = results_df[results_df['production_ready']]
    logger.info(f"\nProduction ready pairs: {len(production_ready)}/{len(results_df)}")

    if len(production_ready) > 0:
        logger.info("\n[+] SUCCESS! Production-ready configurations found:")
        for _, row in production_ready.iterrows():
            logger.info(f"  {row['pair']}: Sharpe {row['optimized_sharpe']:.3f}")
            logger.info(f"    Parameters: Entry={row['entry_zscore']}, Exit={row['exit_zscore']}, "
                       f"Stop={row['stop_loss_zscore']}, Window={row['zscore_window']}")
    else:
        logger.info("\n[~] No production-ready pairs found. Additional optimization needed.")
        logger.info("Consider:")
        logger.info("  - Testing remaining 4 pairs (UNG, XLK, XLF)")
        logger.info("  - Implementing Kalman filter for dynamic hedge ratios")
        logger.info("  - Adding regime detection")
        logger.info("  - Multi-pair portfolio optimization")

    logger.info(f"\n[+] All results saved: {output_path}")
    logger.info(f"[+] Chronicle updated: {PROJECT_ROOT / 'docs' / 'reports' / 'OPTIMIZATION_PROGRESS.md'}")


if __name__ == "__main__":
    main()
