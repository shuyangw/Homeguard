"""Debug script for short position parity issues."""
import sys
sys.path.insert(0, '.')
import pandas as pd
import numpy as np
from src.backtesting.engine.portfolio_simulator import Portfolio
from src.backtesting.utils.risk_config import RiskConfig

# Create downtrend data
dates = pd.date_range('2022-01-03 10:00:00', periods=50, freq='D', tz='US/Eastern')
prices = pd.Series([100 - i * 0.3 for i in range(50)], index=dates)
n = len(prices)

# Short entry on day 2, cover on day 40
entries = pd.Series([i == 40 for i in range(n)], index=prices.index)
exits = pd.Series([i == 2 for i in range(n)], index=prices.index)

# Disable stop loss
risk_config = RiskConfig(use_stop_loss=False)

print('Dates and prices for bars 0-10:')
for i in range(11):
    dow = dates[i].strftime("%a")
    print(f'  Bar {i}: {dates[i].date()} ({dow}) price={prices.iloc[i]:.2f}, entry={entries.iloc[i]}, exit={exits.iloc[i]}')

# Run both simulations
portfolio_py = Portfolio(prices, entries, exits, 10000, 0, 0, allow_shorts=True, risk_config=risk_config, market_hours_only=False, use_numba=False)
portfolio_nb = Portfolio(prices, entries, exits, 10000, 0, 0, allow_shorts=True, risk_config=risk_config, market_hours_only=False, use_numba=True)

print('\nEquity comparison all bars with differences:')
for i in range(n):
    py_eq = portfolio_py.equity_curve.iloc[i]
    nb_eq = portfolio_nb.equity_curve.iloc[i]
    diff = py_eq - nb_eq
    if abs(diff) > 0.001:
        print(f'  Bar {i}: Python={py_eq:.4f}, Numba={nb_eq:.4f}, diff={diff:.4f}')

print('\nPython trades:', len(portfolio_py.trades))
for t in portfolio_py.trades:
    print(f"  {t['type']} price={t['price']:.4f} shares={t['shares']:.4f}")

print('\nNumba trades:', len(portfolio_nb.trades))
for t in portfolio_nb.trades:
    print(f"  {t['type']} price={t['price']:.4f} shares={t['shares']:.4f}")

# Calculate expected shares at bar 40
print('\n--- Position sizing analysis at bar 40 ---')
# Short entry at bar 2, price 99.40, 10 shares
short_entry_proceeds = 10 * 99.40
print(f'Short entry proceeds: {short_entry_proceeds:.2f}')
# Cash after short = 10000 + 994 = 10994
cash_after_short = 10000 + short_entry_proceeds
print(f'Cash after short: {cash_after_short:.2f}')
# Cover at bar 40, price 88.00, 10 shares
cover_cost = 10 * 88.00
print(f'Cover cost: {cover_cost:.2f}')
# Cash after cover
cash_after_cover = cash_after_short - cover_cost
print(f'Cash after cover: {cash_after_cover:.2f}')
# Portfolio value for long entry
print(f'Portfolio value for long entry: {cash_after_cover:.2f}')
# Target value = 10% of portfolio
target_value = cash_after_cover * 0.10
print(f'Target value (10%): {target_value:.2f}')
# Expected shares
expected_shares = int(target_value / 88.0)
print(f'Expected shares: int({target_value:.2f} / 88.00) = {expected_shares}')
