"""
Position Limit Optimization for OMR Strategy.

Tests different max_positions settings to find optimal configuration.
Compares performance metrics across:
- 1 position (ultra conservative)
- 3 positions (current production)
- 5 positions (original documentation)
- 7 positions (aggressive)
- 10 positions (maximum diversification)

Run this to determine if increasing position limits improves performance.
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

# Import the validation script components
from validate_overnight_strategy_v3_full_universe import (
    SimpleRegimeDetector,
    SimpleBayesianModel,
    load_data,
    backtest_strategy,
    analyze_results
)

DATA_DIR = Path('data/leveraged_etfs')

# Production symbol universe (20 ETFs from config)
PRODUCTION_SYMBOLS = [
    'FAZ', 'USD', 'UDOW', 'UYG', 'SOXL', 'TECL', 'UPRO', 'SVXY',
    'TQQQ', 'SSO', 'DFEN', 'WEBL', 'UCO', 'FAS', 'TNA', 'LABU',
    'SPXU', 'QLD', 'SQQQ', 'NAIL'
]

# Test configurations
POSITION_LIMIT_TESTS = [
    {
        'name': 'Ultra Conservative',
        'max_positions': 1,
        'position_size': 0.20,  # 20% in single position
        'max_exposure': 0.20
    },
    {
        'name': 'Conservative',
        'max_positions': 2,
        'position_size': 0.15,  # 15% per position
        'max_exposure': 0.30    # 2 × 15% = 30%
    },
    {
        'name': 'Current Production',
        'max_positions': 3,
        'position_size': 0.15,  # 15% per position
        'max_exposure': 0.45    # 3 × 15% = 45%
    },
    {
        'name': 'Moderate',
        'max_positions': 4,
        'position_size': 0.15,  # 15% per position
        'max_exposure': 0.60    # 4 × 15% = 60%
    },
    {
        'name': 'Original Spec',
        'max_positions': 5,
        'position_size': 0.15,  # 15% per position
        'max_exposure': 0.75    # 5 × 15% = 75%
    },
    {
        'name': 'Aggressive',
        'max_positions': 7,
        'position_size': 0.12,  # 12% per position
        'max_exposure': 0.84    # 7 × 12% = 84%
    },
    {
        'name': 'Maximum Diversification',
        'max_positions': 10,
        'position_size': 0.10,  # 10% per position
        'max_exposure': 1.00    # 10 × 10% = 100%
    }
]


def run_position_limit_test(config, data, spy_data, vix_data):
    """
    Run backtest with specific position limit configuration.

    Args:
        config: Test configuration dict
        data: Historical data for all symbols
        spy_data: SPY data for regime detection
        vix_data: VIX data for regime detection

    Returns:
        Results dictionary with metrics
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"Testing: {config['name']}")
    logger.info(f"  Max Positions: {config['max_positions']}")
    logger.info(f"  Position Size: {config['position_size']:.1%}")
    logger.info(f"  Max Exposure: {config['max_exposure']:.1%}")
    logger.info(f"{'='*80}")

    # Modify the global parameters for this test
    # NOTE: This is a hack for testing - in production, pass as parameters
    import validate_overnight_strategy_v3_full_universe as v3
    original_max_pos = v3.MAX_CONCURRENT_POSITIONS
    original_pos_size = v3.MAX_POSITION_SIZE
    original_exposure = v3.MAX_TOTAL_EXPOSURE

    try:
        v3.MAX_CONCURRENT_POSITIONS = config['max_positions']
        v3.MAX_POSITION_SIZE = config['position_size']
        v3.MAX_TOTAL_EXPOSURE = config['max_exposure']

        # Create regime detector and bayesian model
        regime_detector = SimpleRegimeDetector(
            bull_ma_period=200,
            vix_threshold=25
        )
        bayesian_model = SimpleBayesianModel()

        # Run backtest
        start_date = pd.Timestamp('2024-01-01')
        end_date = pd.Timestamp('2024-11-14')

        trades = backtest_strategy(
            data=data,
            regime_detector=regime_detector,
            bayesian_model=bayesian_model,
            start_date=start_date,
            end_date=end_date,
            symbols=PRODUCTION_SYMBOLS,
            name=config['name']
        )

        # Calculate metrics
        metrics = analyze_results(trades, name=config['name'])

        return {
            'config': config,
            'trades': trades,
            'metrics': metrics
        }

    finally:
        # Restore original values
        v3.MAX_CONCURRENT_POSITIONS = original_max_pos
        v3.MAX_POSITION_SIZE = original_pos_size
        v3.MAX_TOTAL_EXPOSURE = original_exposure


def compare_results(results):
    """
    Compare results across all position limit configurations.

    Args:
        results: List of result dictionaries
    """
    print("\n" + "="*100)
    print("POSITION LIMIT COMPARISON RESULTS")
    print("="*100)

    # Create comparison table
    comparison = []
    for result in results:
        config = result['config']
        metrics = result['metrics']

        comparison.append({
            'Configuration': config['name'],
            'Max Pos': config['max_positions'],
            'Pos Size': f"{config['position_size']:.1%}",
            'Total Trades': len(result['trades']),
            'Win Rate': f"{metrics.get('win_rate', 0):.1%}",
            'Total Return': f"{metrics.get('total_return', 0):.1%}",
            'Sharpe': f"{metrics.get('sharpe_ratio', 0):.2f}",
            'Max DD': f"{metrics.get('max_drawdown', 0):.1%}",
            'Avg Return': f"{metrics.get('avg_return', 0):.2%}"
        })

    df = pd.DataFrame(comparison)
    print("\n" + df.to_string(index=False))

    # Find best configurations
    print("\n" + "="*100)
    print("BEST CONFIGURATIONS BY METRIC")
    print("="*100)

    # Extract numeric values for ranking
    for result in results:
        result['total_return'] = result['metrics'].get('total_return', 0)
        result['sharpe'] = result['metrics'].get('sharpe_ratio', 0)
        result['win_rate'] = result['metrics'].get('win_rate', 0)
        result['max_dd'] = result['metrics'].get('max_drawdown', 0)

    best_sharpe = max(results, key=lambda x: x['sharpe'])
    best_return = max(results, key=lambda x: x['total_return'])
    best_winrate = max(results, key=lambda x: x['win_rate'])
    best_drawdown = min(results, key=lambda x: abs(x['max_dd']))  # Lowest DD

    print(f"\n{'='*50}")
    print(f"Best Sharpe Ratio: {best_sharpe['config']['name']}")
    print(f"  Sharpe: {best_sharpe['sharpe']:.2f}")
    print(f"  Max Positions: {best_sharpe['config']['max_positions']}")
    print(f"  Total Return: {best_sharpe['total_return']:.1%}")

    print(f"\n{'='*50}")
    print(f"Best Total Return: {best_return['config']['name']}")
    print(f"  Return: {best_return['total_return']:.1%}")
    print(f"  Max Positions: {best_return['config']['max_positions']}")
    print(f"  Sharpe: {best_return['sharpe']:.2f}")

    print(f"\n{'='*50}")
    print(f"Best Win Rate: {best_winrate['config']['name']}")
    print(f"  Win Rate: {best_winrate['win_rate']:.1%}")
    print(f"  Max Positions: {best_winrate['config']['max_positions']}")
    print(f"  Sharpe: {best_winrate['sharpe']:.2f}")

    print(f"\n{'='*50}")
    print(f"Best Drawdown Control: {best_drawdown['config']['name']}")
    print(f"  Max Drawdown: {best_drawdown['max_dd']:.1%}")
    print(f"  Max Positions: {best_drawdown['config']['max_positions']}")
    print(f"  Sharpe: {best_drawdown['sharpe']:.2f}")

    # Recommendations
    print("\n" + "="*100)
    print("RECOMMENDATIONS")
    print("="*100)

    # Find balanced configuration (best Sharpe with acceptable drawdown)
    acceptable_results = [r for r in results if abs(r['max_dd']) < 0.10]  # DD < 10%
    if acceptable_results:
        recommended = max(acceptable_results, key=lambda x: x['sharpe'])
        print(f"\nRecommended Configuration: {recommended['config']['name']}")
        print(f"  Reason: Best Sharpe ({recommended['sharpe']:.2f}) with acceptable drawdown")
        print(f"  Max Positions: {recommended['config']['max_positions']}")
        print(f"  Position Size: {recommended['config']['position_size']:.1%}")
        print(f"  Total Return: {recommended['total_return']:.1%}")
        print(f"  Max Drawdown: {recommended['max_dd']:.1%}")
        print(f"  Win Rate: {recommended['win_rate']:.1%}")

        # Compare to current production
        current = [r for r in results if r['config']['name'] == 'Current Production'][0]
        print(f"\nCurrent Production Performance:")
        print(f"  Max Positions: {current['config']['max_positions']}")
        print(f"  Sharpe: {current['sharpe']:.2f}")
        print(f"  Total Return: {current['total_return']:.1%}")
        print(f"  Max Drawdown: {current['max_dd']:.1%}")

        if recommended['config']['name'] != 'Current Production':
            sharpe_improvement = ((recommended['sharpe'] - current['sharpe']) / current['sharpe']) * 100
            return_improvement = recommended['total_return'] - current['total_return']
            print(f"\nPotential Improvement:")
            print(f"  Sharpe: {sharpe_improvement:+.1f}%")
            print(f"  Return: {return_improvement:+.1%}")
        else:
            print(f"\nCurrent production config is already optimal!")

    return df


def main():
    """Run position limit optimization tests."""
    print("\n" + "="*100)
    print("OMR STRATEGY - POSITION LIMIT OPTIMIZATION")
    print("="*100)
    print(f"Testing {len(POSITION_LIMIT_TESTS)} different position limit configurations")
    print(f"Symbol Universe: {len(PRODUCTION_SYMBOLS)} ETFs (production config)")
    print(f"Backtest Period: 2024-01-01 to 2024-11-14")
    print("="*100)

    # Load data
    logger.info("\nLoading historical data...")
    data = load_data(DATA_DIR, PRODUCTION_SYMBOLS + ['SPY', '^VIX'])

    if 'SPY' not in data or '^VIX' not in data:
        logger.error("SPY or VIX data not found!")
        return 1

    spy_data = data['SPY']
    vix_data = data['^VIX']

    # Run tests for each configuration
    results = []
    for config in POSITION_LIMIT_TESTS:
        try:
            result = run_position_limit_test(config, data, spy_data, vix_data)
            results.append(result)
        except Exception as e:
            logger.error(f"Error testing {config['name']}: {e}")
            import traceback
            traceback.print_exc()

    if not results:
        logger.error("No successful tests!")
        return 1

    # Compare and analyze results
    comparison_df = compare_results(results)

    # Save results
    output_dir = Path('reports')
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = output_dir / f'{timestamp}_POSITION_LIMIT_OPTIMIZATION.csv'
    comparison_df.to_csv(output_file, index=False)
    logger.info(f"\nResults saved to: {output_file}")

    print("\n" + "="*100)
    print("TESTING COMPLETE")
    print("="*100)

    return 0


if __name__ == "__main__":
    sys.exit(main())
