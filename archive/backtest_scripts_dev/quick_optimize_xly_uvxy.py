"""
Quick XLY/UVXY Optimization

Tests a focused parameter grid around the best sensitivity results.
This is faster than full grid search for initial exploration.
"""

import pandas as pd
import sys
from pathlib import Path
from itertools import product
import time

from utils.path_setup import setup_project_paths
ROOT_DIR = setup_project_paths()

# Add project root to path
project_root = Path(__file__).parent.parent

from src.backtesting.engine.backtest_engine import BacktestEngine
from src.strategies.advanced.pairs_trading import PairsTrading
from src.utils import logger


def run_single_backtest(symbol1, symbol2, params):
    """Run a single backtest with given parameters."""
    try:
        strategy = PairsTrading(
            pair_selection_window=252,
            cointegration_pvalue=0.05,
            entry_zscore=params['entry_zscore'],
            exit_zscore=params['exit_zscore'],
            stop_loss_zscore=params['stop_loss_zscore'],
            zscore_window=params['zscore_window']
        )

        engine = BacktestEngine(
            initial_capital=100000,
            fees=0.0001,  # 0.01%
            slippage=0.001  # 0.10%
        )

        portfolio = engine.run(
            strategy=strategy,
            symbols=[symbol1, symbol2],
            start_date="2023-01-01",
            end_date="2024-11-11"
        )

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
            'final_equity': stats['End Value']
        }

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return None


def main():
    """Quick focused optimization."""
    logger.info("QUICK XLY/UVXY OPTIMIZATION")
    logger.info(" Focused grid around best sensitivity results\n")

    # Focused parameter grid (based on sensitivity analysis)
    param_grid = {
        'entry_zscore': [2.0, 2.25, 2.5, 2.75],  # Best was 2.5
        'exit_zscore': [0.25, 0.5, 0.75],  # Best was 0.5
        'stop_loss_zscore': [3.5],  # No impact, keep at 3.5
        'zscore_window': [20, 25]  # Best was 20
    }

    param_names = list(param_grid.keys())
    param_values = [param_grid[name] for name in param_names]
    combinations = list(product(*param_values))

    total = len(combinations)
    logger.info(f"Testing {total} combinations\n")

    results = []
    start_time = time.time()

    for idx, combo in enumerate(combinations, 1):
        params = dict(zip(param_names, combo))

        result = run_single_backtest("XLY", "UVXY", params)

        if result:
            results.append({
                'pair': 'XLY/UVXY',
                **params,
                **result
            })

            pct = 100 * idx / total
            logger.info(f"[{idx}/{total}] {pct:.0f}% - Entry={params['entry_zscore']}, Exit={params['exit_zscore']}, "
                       f"Win={params['zscore_window']}: Sharpe={result['sharpe']:.3f}")

            if result['sharpe'] >= 0.8:
                logger.info(f"  [PRODUCTION READY!] Sharpe={result['sharpe']:.3f}")

    df = pd.DataFrame(results)
    df = df.sort_values('sharpe', ascending=False)

    output_path = project_root / "output" / "quick_optimize_XLY_UVXY.csv"
    df.to_csv(output_path, index=False)

    elapsed = time.time() - start_time
    logger.info(f"\nCompleted in {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    logger.info(f"Results saved: {output_path}\n")

    logger.info("TOP 10 RESULTS:")
    for idx, row in df.head(10).iterrows():
        production_ready = "[PRODUCTION READY]" if row['sharpe'] >= 0.8 else ""
        logger.info(f"  {idx+1}. Sharpe={row['sharpe']:.3f}, Return={row['return']:.2f}%, "
                   f"Entry={row['entry_zscore']}, Exit={row['exit_zscore']}, "
                   f"Window={row['zscore_window']}, Trades={row['trades']} {production_ready}")

    best = df.iloc[0]
    if best['sharpe'] >= 0.8:
        logger.info(f"\n[+] SUCCESS! Production-ready configuration found")
        logger.info(f"  Sharpe: {best['sharpe']:.3f}")
        logger.info(f"  Parameters: Entry={best['entry_zscore']}, Exit={best['exit_zscore']}, "
                   f"Stop={best['stop_loss_zscore']}, Window={best['zscore_window']}")
    else:
        gap = 0.8 - best['sharpe']
        pct_gap = 100 * gap / 0.8
        logger.info(f"\n[~] Best Sharpe={best['sharpe']:.3f} ({pct_gap:.1f}% below 0.8 target)")
        logger.info("  May need additional strategies (Kalman filter, regime detection, etc.)")


if __name__ == "__main__":
    main()
