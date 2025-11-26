"""
Analyze Kelly Criterion potential for Overnight Mean Reversion Strategy.

Compares current fixed position sizing (15%) vs. Kelly-based sizing.
"""

import pandas as pd
import numpy as np
from typing import Dict
import sys
import os
from pathlib import Path

from utils.path_setup import setup_project_paths
ROOT_DIR = setup_project_paths()

# Add src to path
from src.config import get_backtest_results_dir

# Load trades
trades = pd.read_csv(get_backtest_results_dir() / 'overnight_validation_trades.csv')
trades['date'] = pd.to_datetime(trades['date'])

print('='*80)
print('KELLY CRITERION ANALYSIS - OVERNIGHT MEAN REVERSION')
print('='*80)
print()

# Current configuration
CURRENT_POSITION_SIZE = 0.15  # 15% per trade
MAX_POSITIONS = 3
INITIAL_CAPITAL = 100000

print('CURRENT CONFIGURATION:')
print('-'*80)
print(f'Position Size: {CURRENT_POSITION_SIZE*100:.0f}% per trade')
print(f'Max Positions: {MAX_POSITIONS}')
print(f'Max Deployment: {CURRENT_POSITION_SIZE * MAX_POSITIONS * 100:.0f}% of capital')
print()

# Calculate Kelly by regime
print('STEP 1: CALCULATE KELLY CRITERION BY REGIME')
print('-'*80)

kelly_by_regime = {}

for regime in trades['regime'].unique():
    regime_trades = trades[trades['regime'] == regime]

    # Win probability
    win_rate = (regime_trades['actual_return'] > 0).mean()
    lose_rate = 1 - win_rate

    # Average win and loss
    wins = regime_trades[regime_trades['actual_return'] > 0]['actual_return']
    losses = regime_trades[regime_trades['actual_return'] < 0]['actual_return']

    if len(wins) > 0 and len(losses) > 0:
        avg_win = wins.mean()
        avg_loss = abs(losses.mean())

        # Win/loss ratio
        win_loss_ratio = avg_win / avg_loss

        # Kelly formula: (bp - q) / b
        # where b = win/loss ratio, p = win prob, q = loss prob
        kelly = (win_loss_ratio * win_rate - lose_rate) / win_loss_ratio
        half_kelly = kelly / 2  # Conservative Kelly

        kelly_by_regime[regime] = {
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'win_loss_ratio': win_loss_ratio,
            'kelly': kelly,
            'half_kelly': half_kelly,
            'trades': len(regime_trades)
        }

        print(f'{regime:15} Win: {win_rate*100:5.1f}%  Avg W/L: {win_loss_ratio:.2f}  '
              f'Kelly: {kelly*100:6.1f}%  Half-Kelly: {half_kelly*100:5.1f}%  ({len(regime_trades)} trades)')

print()

# Calculate regime-specific Sharpe ratios for reference
print('STEP 2: REGIME PERFORMANCE METRICS')
print('-'*80)

regime_stats = {}
for regime in trades['regime'].unique():
    regime_trades = trades[trades['regime'] == regime]
    returns = regime_trades['actual_return']

    if len(returns) > 1 and returns.std() > 0:
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252)
    else:
        sharpe = 0

    regime_stats[regime] = {
        'sharpe': sharpe,
        'avg_return': returns.mean(),
        'trades': len(regime_trades)
    }

    print(f'{regime:15} Sharpe: {sharpe:6.2f}  Avg Return: {returns.mean()*100:6.2f}%  '
          f'Trades: {len(regime_trades)}')

print()

# Simulate portfolio performance with different strategies
print('STEP 3: SIMULATE PORTFOLIO PERFORMANCE')
print('-'*80)
print()

def simulate_portfolio(trades_df, position_sizing_func, name):
    """Simulate portfolio with given position sizing function."""
    portfolio_value = INITIAL_CAPITAL
    portfolio_history = [portfolio_value]
    daily_returns = []

    # Group trades by date
    for date in trades_df['date'].unique():
        day_trades = trades_df[trades_df['date'] == date]

        # Calculate position sizes for this day's trades
        day_return = 0
        for _, trade in day_trades.iterrows():
            position_size = position_sizing_func(trade)
            trade_return = trade['actual_return'] * position_size
            day_return += trade_return

        daily_returns.append(day_return)
        portfolio_value *= (1 + day_return)
        portfolio_history.append(portfolio_value)

    # Calculate metrics
    daily_returns = pd.Series(daily_returns)

    total_return = (portfolio_value - INITIAL_CAPITAL) / INITIAL_CAPITAL

    if len(daily_returns) > 1 and daily_returns.std() > 0:
        sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
    else:
        sharpe = 0

    # Max drawdown
    portfolio_series = pd.Series(portfolio_history)
    cumulative = portfolio_series / INITIAL_CAPITAL
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_dd = drawdown.min()

    return {
        'name': name,
        'final_value': portfolio_value,
        'total_return': total_return,
        'sharpe': sharpe,
        'max_dd': max_dd,
        'avg_daily_return': daily_returns.mean(),
        'daily_std': daily_returns.std()
    }

# Strategy 1: Current fixed sizing (15% per trade)
def fixed_sizing(trade):
    return CURRENT_POSITION_SIZE

# Strategy 2: Half-Kelly by regime
def half_kelly_sizing(trade):
    regime = trade['regime']
    if regime in kelly_by_regime:
        kelly_val = kelly_by_regime[regime]['half_kelly']
        # Cap at 25% per trade for safety
        return min(max(kelly_val, 0.01), 0.25)
    return 0.10  # Default if regime not found

# Strategy 3: Full Kelly by regime (risky)
def full_kelly_sizing(trade):
    regime = trade['regime']
    if regime in kelly_by_regime:
        kelly_val = kelly_by_regime[regime]['kelly']
        # Cap at 40% per trade for safety
        return min(max(kelly_val, 0.01), 0.40)
    return 0.15

# Strategy 4: Kelly with regime filter (skip BEAR)
def kelly_filtered_sizing(trade):
    regime = trade['regime']
    if regime == 'BEAR':
        return 0.0  # Skip BEAR trades
    if regime in kelly_by_regime:
        kelly_val = kelly_by_regime[regime]['half_kelly']
        return min(max(kelly_val, 0.01), 0.25)
    return 0.10

# Run simulations
print('Strategy 1: Fixed 15% Position Sizing (Current)')
result_fixed = simulate_portfolio(trades, fixed_sizing, 'Fixed 15%')
print(f"  Final Value:    ${result_fixed['final_value']:,.0f}")
print(f"  Total Return:   {result_fixed['total_return']*100:,.1f}%")
print(f"  Sharpe Ratio:   {result_fixed['sharpe']:.2f}")
print(f"  Max Drawdown:   {result_fixed['max_dd']*100:.1f}%")
print()

print('Strategy 2: Half-Kelly by Regime')
result_half_kelly = simulate_portfolio(trades, half_kelly_sizing, 'Half-Kelly')
print(f"  Final Value:    ${result_half_kelly['final_value']:,.0f}")
print(f"  Total Return:   {result_half_kelly['total_return']*100:,.1f}%")
print(f"  Sharpe Ratio:   {result_half_kelly['sharpe']:.2f}")
print(f"  Max Drawdown:   {result_half_kelly['max_dd']*100:.1f}%")
print()

print('Strategy 3: Full Kelly by Regime (Aggressive)')
result_full_kelly = simulate_portfolio(trades, full_kelly_sizing, 'Full Kelly')
print(f"  Final Value:    ${result_full_kelly['final_value']:,.0f}")
print(f"  Total Return:   {result_full_kelly['total_return']*100:,.1f}%")
print(f"  Sharpe Ratio:   {result_full_kelly['sharpe']:.2f}")
print(f"  Max Drawdown:   {result_full_kelly['max_dd']*100:.1f}%")
print()

print('Strategy 4: Half-Kelly + BEAR Filter (Recommended)')
result_kelly_filtered = simulate_portfolio(
    trades, kelly_filtered_sizing, 'Half-Kelly + Filter'
)
print(f"  Final Value:    ${result_kelly_filtered['final_value']:,.0f}")
print(f"  Total Return:   {result_kelly_filtered['total_return']*100:,.1f}%")
print(f"  Sharpe Ratio:   {result_kelly_filtered['sharpe']:.2f}")
print(f"  Max Drawdown:   {result_kelly_filtered['max_dd']*100:.1f}%")
print()

# Comparison table
print('='*80)
print('COMPARISON SUMMARY')
print('='*80)
print()

results = [result_fixed, result_half_kelly, result_full_kelly, result_kelly_filtered]

print(f"{'Strategy':<30} {'Return':<12} {'Sharpe':<10} {'Max DD':<12} {'Final Value':<15}")
print('-'*80)
for result in results:
    print(f"{result['name']:<30} "
          f"{result['total_return']*100:>10.1f}%  "
          f"{result['sharpe']:>8.2f}  "
          f"{result['max_dd']*100:>10.1f}%  "
          f"${result['final_value']:>13,.0f}")

print()
print('='*80)
print('KEY FINDINGS')
print('='*80)
print()

# Calculate improvement
base_return = result_fixed['total_return']
kelly_return = result_kelly_filtered['total_return']
improvement = (kelly_return - base_return) / base_return

base_sharpe = result_fixed['sharpe']
kelly_sharpe = result_kelly_filtered['sharpe']
sharpe_improvement = (kelly_sharpe - base_sharpe) / base_sharpe

base_dd = abs(result_fixed['max_dd'])
kelly_dd = abs(result_kelly_filtered['max_dd'])
dd_improvement = (base_dd - kelly_dd) / base_dd

print(f'1. Return Improvement:   {improvement*100:+.1f}%')
print(f'   ({base_return*100:.1f}% -> {kelly_return*100:.1f}%)')
print()

print(f'2. Sharpe Improvement:   {sharpe_improvement*100:+.1f}%')
print(f'   ({base_sharpe:.2f} -> {kelly_sharpe:.2f})')
print()

print(f'3. Drawdown Reduction:   {dd_improvement*100:+.1f}%')
print(f'   ({base_dd*100:.1f}% -> {kelly_dd*100:.1f}%)')
print()

print(f'4. Final Portfolio Value Increase: ${result_kelly_filtered["final_value"] - result_fixed["final_value"]:,.0f}')
print(f'   (${result_fixed["final_value"]:,.0f} -> ${result_kelly_filtered["final_value"]:,.0f})')
print()

# Risk-adjusted return comparison
base_return_per_dd = result_fixed['total_return'] / abs(result_fixed['max_dd'])
kelly_return_per_dd = result_kelly_filtered['total_return'] / abs(result_kelly_filtered['max_dd'])

print(f'5. Return per Unit of Max Drawdown:')
print(f'   Fixed:  {base_return_per_dd:.2f}x')
print(f'   Kelly:  {kelly_return_per_dd:.2f}x')
print(f'   Improvement: {(kelly_return_per_dd / base_return_per_dd - 1)*100:+.1f}%')
print()

print('='*80)
print('RECOMMENDATION')
print('='*80)
print()

if kelly_sharpe > base_sharpe and kelly_dd < base_dd:
    print('✓ IMPLEMENT KELLY CRITERION')
    print()
    print('Half-Kelly with BEAR filter shows:')
    print(f'  • Higher Sharpe ratio ({kelly_sharpe:.2f} vs {base_sharpe:.2f})')
    print(f'  • Lower maximum drawdown ({kelly_dd*100:.1f}% vs {base_dd*100:.1f}%)')
    print(f'  • Better risk-adjusted returns')
    print()
    print('Recommended implementation:')
    print('  • Use half-Kelly sizing by regime')
    print('  • Cap position size at 25% per trade')
    print('  • Skip BEAR regime entirely (0% allocation)')
    print('  • Adjust position sizes daily based on regime')
else:
    print('⚠ MARGINAL BENEFIT')
    print()
    print('Kelly criterion shows limited improvement over fixed sizing.')
    print('Consider keeping fixed sizing for simplicity.')

print()
print('='*80)
