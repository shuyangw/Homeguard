"""Walk-forward grid search for sentiment parameters."""

import subprocess
import sys
import yaml
import os
from pathlib import Path
import re

# Grid search parameters
sentiment_scores = [0.0, 0.1, 0.2, 0.3, 0.4]
confidences = [0.5, 0.6, 0.7, 0.8]

# Periods
train_period = ("2024-01-01", "2024-03-31")
test_period = ("2024-04-01", "2024-06-30")

# 50 symbols with good data coverage
SYMBOLS = [
    'AAPL', 'ABBV', 'ABNB', 'ABT', 'ADBE', 'ADM', 'ALB', 'AMAT', 'AMD', 'AMZN',
    'ANET', 'APA', 'AVGO', 'AXP', 'BA', 'BAC', 'BBY', 'BKR', 'BMY', 'BX',
    'C', 'CCL', 'CMCSA', 'COIN', 'COP', 'CRM', 'CRWD', 'CSCO', 'CSX', 'CTRA',
    'CVS', 'CVX', 'CZR', 'DAL', 'DASH', 'DDOG', 'DELL', 'DIS', 'EBAY', 'F',
    'FCX', 'FTNT', 'GE', 'GILD', 'GM', 'GOOG', 'GOOGL', 'HD', 'IBM', 'INTC'
]

# Base config
base_config = {
    'mode': 'single',
    'strategy': {
        'name': 'HVORBStrategy',
        'parameters': {
            'opening_range_minutes': 5,
            'sip_lookback_days': 14,
            'sip_min_score': 1.5,
            'min_gap_pct': 0.02,
            'max_gap_pct': 0.15,
            'gap_direction_filter': True,
            'atr_period': 14,
            'min_atr_pct': 0.03,
            'max_atr_pct': 3.0,
            'target1_multiplier': 1.0,
            'target1_exit_pct': 0.5,
            'target2_multiplier': 2.0,
            'use_trailing_stop': True,
            'trailing_offset_pct': 0.02,
            'eod_exit_hour': 15,
            'eod_exit_minute': 55,
            'daily_max_loss_pct': 0.03,
            'max_daily_trades': 10,
            'max_concurrent_positions': 5,
            'entry_cutoff_hour': 11,
            'entry_cutoff_minute': 0,
            'rvol_threshold': 1.5,
            'use_sentiment': True,
            'skip_earnings_days': True,
            'use_regime': True,
            'long_only': True,
        }
    },
    'symbols': {'list': SYMBOLS},
    'backtest': {
        'initial_capital': 100000,
        'fees': 0.001,
        'slippage': 0.001,
        'benchmark': 'SPY',
        'market_hours_only': True,
        'allow_shorts': False,
        'portfolio_mode': 'multi'
    },
    'risk': {
        'enabled': True,
        'position_sizing_method': 'fixed_percent',
        'position_size_pct': 0.10,
        'max_positions': 5
    },
    'output': {
        'save_trades': False,
        'save_reports': False,
        'quantstats': False,
        'visualize': False,
        'verbosity': 0
    }
}


def run_backtest(config_path):
    """Run backtest and extract results."""
    result = subprocess.run(
        [sys.executable, '-m', 'src.backtest_runner', '--config', config_path],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).parent.parent)
    )

    output = result.stdout + result.stderr

    # Parse results
    ret_match = re.search(r'Total Return:\s+([-\d.]+)%', output)
    sharpe_match = re.search(r'Sharpe Ratio:\s+([-\d.]+)', output)
    trades_match = re.search(r'Total Trades:\s+([\d.]+)', output)
    winrate_match = re.search(r'Win Rate:\s+([\d.]+)%', output)

    return {
        'return': float(ret_match.group(1)) if ret_match else 0.0,
        'sharpe': float(sharpe_match.group(1)) if sharpe_match else 0.0,
        'trades': int(float(trades_match.group(1))) if trades_match else 0,
        'winrate': float(winrate_match.group(1)) if winrate_match else 0.0
    }


def main():
    results = []
    config_path = Path(__file__).parent.parent / 'config' / 'backtesting' / '_grid_search_temp.yaml'

    print("=" * 70)
    print("WALK-FORWARD SENTIMENT PARAMETER OPTIMIZATION")
    print("=" * 70)
    print(f"Train: {train_period[0]} to {train_period[1]}")
    print(f"Test:  {test_period[0]} to {test_period[1]}")
    print(f"Grid:  {len(sentiment_scores)} scores x {len(confidences)} confidences = {len(sentiment_scores) * len(confidences)} combinations")
    print("=" * 70)
    print()

    total = len(sentiment_scores) * len(confidences)
    count = 0

    for score in sentiment_scores:
        for conf in confidences:
            count += 1
            print(f"[{count}/{total}] Testing score={score}, conf={conf}...", end=" ", flush=True)

            # Deep copy config
            import copy
            config = copy.deepcopy(base_config)
            config['strategy']['parameters']['min_sentiment_score'] = score
            config['strategy']['parameters']['min_sentiment_confidence'] = conf

            # Train period
            config['dates'] = {'start': train_period[0], 'end': train_period[1]}
            with open(config_path, 'w') as f:
                yaml.dump(config, f)
            train_results = run_backtest(str(config_path))

            # Test period
            config['dates'] = {'start': test_period[0], 'end': test_period[1]}
            with open(config_path, 'w') as f:
                yaml.dump(config, f)
            test_results = run_backtest(str(config_path))

            results.append({
                'score': score,
                'conf': conf,
                'train_return': train_results['return'],
                'train_sharpe': train_results['sharpe'],
                'train_trades': train_results['trades'],
                'test_return': test_results['return'],
                'test_sharpe': test_results['sharpe'],
                'test_trades': test_results['trades'],
            })

            print(f"Train: {train_results['return']:+.2f}% | Test: {test_results['return']:+.2f}%")

    # Clean up temp file
    if config_path.exists():
        os.remove(config_path)

    # Print summary
    print()
    print("=" * 70)
    print("RESULTS SUMMARY (sorted by test return)")
    print("=" * 70)
    print(f"{'Score':<6} {'Conf':<5} | {'Train Ret':>10} {'Train#':>7} | {'Test Ret':>10} {'Test#':>6} | {'Stable':<6}")
    print("-" * 70)

    # Sort by test return
    sorted_results = sorted(results, key=lambda x: x['test_return'], reverse=True)

    for r in sorted_results:
        # Stability check
        stable = "No"
        if r['train_return'] > 0 and r['test_return'] > 0:
            stable = "YES"
        elif r['train_return'] > -2 and r['test_return'] > -2 and r['train_trades'] > 3 and r['test_trades'] > 3:
            stable = "Maybe"

        print(f"{r['score']:<6} {r['conf']:<5} | {r['train_return']:>+9.2f}% {r['train_trades']:>6} | {r['test_return']:>+9.2f}% {r['test_trades']:>5} | {stable:<6}")

    print()
    print("=" * 70)
    print("BEST PARAMETERS (positive in both train AND test)")
    print("=" * 70)

    best = [r for r in results if r['train_return'] > 0 and r['test_return'] > 0]
    if best:
        for r in sorted(best, key=lambda x: x['test_return'], reverse=True):
            print(f"  score={r['score']}, confidence={r['conf']}")
            print(f"    Train: {r['train_return']:+.2f}% ({r['train_trades']} trades)")
            print(f"    Test:  {r['test_return']:+.2f}% ({r['test_trades']} trades)")
    else:
        print("  No parameters were profitable in both periods.")
        print()
        print("  Best test performance:")
        best_test = sorted_results[0]
        print(f"    score={best_test['score']}, confidence={best_test['conf']}")
        print(f"    Train: {best_test['train_return']:+.2f}% | Test: {best_test['test_return']:+.2f}%")


if __name__ == '__main__':
    main()
