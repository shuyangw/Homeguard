"""
Optimize All Top 8 Pairs

Based on XLY/UVXY optimization results, test the best parameter configurations
across all top performing pairs to find which can achieve Sharpe >= 0.8.

From XLY/UVXY optimization:
- Best: Entry=2.75, Exit=0.25, Window=25 (Sharpe 0.735)
- Runner-up: Entry=2.75, Exit=0.5, Window=25 (Sharpe 0.692)
- Third: Entry=2.5, Exit=0.25, Window=25 (Sharpe 0.666)
"""

import pandas as pd
import sys
from pathlib import Path
from itertools import product
import time

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.backtesting.engine.backtest_engine import BacktestEngine
from src.strategies.advanced.pairs_trading import PairsTrading
from src.utils import logger


# Top 8 pairs
TOP_PAIRS = [
    ("XLY", "UVXY", 0.551, 34.91),
    ("XLI", "UVXY", 0.518, 34.62),
    ("DIA", "UVXY", 0.517, 34.75),
    ("VTI", "UVXY", 0.507, 30.76),
    ("VOO", "UVXY", 0.498, 34.68),
    ("UNG", "UVXY", 0.475, 42.41),
    ("XLK", "UVXY", 0.469, 27.78),
    ("XLF", "UVXY", 0.457, 32.55),
]

# Best parameter configurations from XLY/UVXY optimization
BEST_CONFIGS = [
    {'entry_zscore': 2.75, 'exit_zscore': 0.25, 'stop_loss_zscore': 3.5, 'zscore_window': 25, 'name': 'Config 1 (Best XLY)'},
    {'entry_zscore': 2.75, 'exit_zscore': 0.5, 'stop_zscore': 3.5, 'zscore_window': 25, 'name': 'Config 2'},
    {'entry_zscore': 2.5, 'exit_zscore': 0.25, 'stop_loss_zscore': 3.5, 'zscore_window': 25, 'name': 'Config 3'},
    {'entry_zscore': 2.5, 'exit_zscore': 0.75, 'stop_loss_zscore': 3.5, 'zscore_window': 25, 'name': 'Config 4'},
    {'entry_zscore': 2.5, 'exit_zscore': 0.25, 'stop_loss_zscore': 3.5, 'zscore_window': 20, 'name': 'Config 5'},
    # Also test baseline for comparison
    {'entry_zscore': 2.0, 'exit_zscore': 0.5, 'stop_loss_zscore': 3.5, 'zscore_window': 20, 'name': 'Baseline'},
]


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
    """Test best configurations across all pairs."""
    logger.info("="*80)
    logger.info("OPTIMIZE ALL TOP 8 PAIRS")
    logger.info("Testing best parameter configurations from XLY/UVXY optimization")
    logger.info("="*80)

    total_tests = len(TOP_PAIRS) * len(BEST_CONFIGS)
    logger.info(f"\nTotal tests: {total_tests} ({len(TOP_PAIRS)} pairs × {len(BEST_CONFIGS)} configs)")
    logger.info(f"Estimated time: {total_tests * 30 / 60:.1f} minutes\n")

    results = []
    start_time = time.time()
    test_num = 0

    for pair_idx, (sym1, sym2, baseline_sharpe, baseline_return) in enumerate(TOP_PAIRS, 1):
        logger.info(f"\n{'='*80}")
        logger.info(f"PAIR {pair_idx}/{len(TOP_PAIRS)}: {sym1}/{sym2}")
        logger.info(f"Baseline: Sharpe {baseline_sharpe:.3f}, Return {baseline_return:.2f}%")
        logger.info(f"{'='*80}\n")

        for config in BEST_CONFIGS:
            test_num += 1
            pct_complete = 100 * test_num / total_tests

            logger.info(f"[{test_num}/{total_tests}] {pct_complete:.0f}% - {config['name']}: "
                       f"Entry={config['entry_zscore']}, Exit={config['exit_zscore']}, "
                       f"Window={config['zscore_window']}...")

            result = run_single_backtest(sym1, sym2, config)

            if result:
                results.append({
                    'pair': f"{sym1}/{sym2}",
                    'symbol1': sym1,
                    'symbol2': sym2,
                    'config_name': config['name'],
                    'baseline_sharpe': baseline_sharpe,
                    'baseline_return': baseline_return,
                    **config,
                    **result,
                    'sharpe_improvement': result['sharpe'] - baseline_sharpe,
                    'sharpe_improvement_pct': 100 * (result['sharpe'] - baseline_sharpe) / baseline_sharpe if baseline_sharpe > 0 else 0,
                    'production_ready': result['sharpe'] >= 0.8
                })

                production_status = "[PRODUCTION READY!]" if result['sharpe'] >= 0.8 else ""
                logger.info(f"  Sharpe: {result['sharpe']:.3f} (Δ{result['sharpe']-baseline_sharpe:+.3f}), "
                           f"Return: {result['return']:.2f}%, Trades: {result['trades']} {production_status}")
            else:
                logger.error("  FAILED")

            # Progress update every 10 tests
            if test_num % 10 == 0:
                elapsed = time.time() - start_time
                tests_per_sec = test_num / elapsed
                remaining = (total_tests - test_num) / tests_per_sec
                logger.info(f"  [Progress: {pct_complete:.0f}% complete, ETA: {remaining/60:.1f} min]")

    # Create results DataFrame
    df = pd.DataFrame(results)
    df = df.sort_values('sharpe', ascending=False)

    # Save all results
    output_path = project_root / "output" / "all_pairs_optimization_results.csv"
    df.to_csv(output_path, index=False)

    elapsed = time.time() - start_time
    logger.info(f"\n{'='*80}")
    logger.info(f"OPTIMIZATION COMPLETE in {elapsed/60:.1f} minutes")
    logger.info(f"{'='*80}\n")

    logger.info(f"Results saved: {output_path}\n")

    # Summary statistics
    logger.info("="*80)
    logger.info("SUMMARY STATISTICS")
    logger.info("="*80)

    production_ready = df[df['production_ready']]
    logger.info(f"\nProduction Ready (Sharpe >= 0.8): {len(production_ready)}/{len(df)} ({100*len(production_ready)/len(df):.1f}%)")

    if len(production_ready) > 0:
        logger.info("\n[+] PRODUCTION READY CONFIGURATIONS:")
        for _, row in production_ready.iterrows():
            logger.info(f"  {row['pair']} - {row['config_name']}: Sharpe {row['sharpe']:.3f}")
            logger.info(f"    Parameters: Entry={row['entry_zscore']}, Exit={row['exit_zscore']}, "
                       f"Stop={row['stop_loss_zscore']}, Window={row['zscore_window']}")
            logger.info(f"    Performance: Return {row['return']:.2f}%, MaxDD {row['max_dd']:.2f}%, "
                       f"WinRate {row['win_rate']:.1f}%, Trades {row['trades']}")
    else:
        logger.info("\n[~] No production-ready configurations found (Sharpe >= 0.8)")

    # Best configuration per pair
    logger.info("\n" + "="*80)
    logger.info("BEST CONFIGURATION PER PAIR")
    logger.info("="*80 + "\n")

    for pair_name in df['pair'].unique():
        pair_df = df[df['pair'] == pair_name]
        best = pair_df.iloc[0]

        baseline_sharpe = best['baseline_sharpe']
        improvement = best['sharpe'] - baseline_sharpe
        improvement_pct = 100 * improvement / baseline_sharpe

        production_status = "[PRODUCTION READY]" if best['sharpe'] >= 0.8 else "[NEEDS WORK]"

        logger.info(f"{production_status} {pair_name}:")
        logger.info(f"  Baseline: Sharpe {baseline_sharpe:.3f}")
        logger.info(f"  Optimized: Sharpe {best['sharpe']:.3f} ({improvement:+.3f}, {improvement_pct:+.1f}%)")
        logger.info(f"  Best Config: {best['config_name']}")
        logger.info(f"  Parameters: Entry={best['entry_zscore']}, Exit={best['exit_zscore']}, Window={best['zscore_window']}")
        logger.info(f"  Performance: Return {best['return']:.2f}%, MaxDD {best['max_dd']:.2f}%, Trades {best['trades']}")
        logger.info("")

    # Configuration effectiveness
    logger.info("="*80)
    logger.info("CONFIGURATION EFFECTIVENESS")
    logger.info("="*80 + "\n")

    for config_name in BEST_CONFIGS:
        config_df = df[df['config_name'] == config_name['name']]
        if len(config_df) > 0:
            avg_sharpe = config_df['sharpe'].mean()
            avg_improvement = config_df['sharpe_improvement'].mean()
            production_count = config_df['production_ready'].sum()

            logger.info(f"{config_name['name']}:")
            logger.info(f"  Avg Sharpe: {avg_sharpe:.3f}")
            logger.info(f"  Avg Improvement: {avg_improvement:+.3f}")
            logger.info(f"  Production Ready: {production_count}/{len(config_df)}")
            logger.info("")

    # Overall improvement
    logger.info("="*80)
    logger.info("OVERALL IMPROVEMENT")
    logger.info("="*80 + "\n")

    logger.info(f"Average baseline Sharpe: {df['baseline_sharpe'].mean():.3f}")
    logger.info(f"Average optimized Sharpe: {df.groupby('pair')['sharpe'].max().mean():.3f}")
    logger.info(f"Average improvement: {df.groupby('pair')['sharpe_improvement'].max().mean():.3f} "
               f"({df.groupby('pair')['sharpe_improvement_pct'].max().mean():.1f}%)")

    gap_to_target = 0.8 - df.groupby('pair')['sharpe'].max().mean()
    logger.info(f"\nAverage gap to Sharpe 0.8: {gap_to_target:.3f} ({100*gap_to_target/0.8:.1f}%)")

    if len(production_ready) == 0:
        logger.info("\n[!] RECOMMENDATION:")
        logger.info("  No pairs achieved Sharpe >= 0.8 with parameter optimization alone.")
        logger.info("  Consider advanced enhancements:")
        logger.info("    1. Kalman filter for dynamic hedge ratios")
        logger.info("    2. Regime detection for adaptive position sizing")
        logger.info("    3. Multi-pair portfolio diversification")
        logger.info("    4. Alternative entry/exit signals (Bollinger bands, RSI)")
        logger.info("    5. Higher leverage (15-20% position sizing)")


if __name__ == "__main__":
    main()
