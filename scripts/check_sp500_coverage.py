"""Check S&P 500 coverage in local Alpaca data."""
import pandas as pd
from pathlib import Path

sp500_file = Path(__file__).parent.parent / 'backtest_lists' / 'sp500-2025.csv'
local_data = Path(r'F:\Stock_Data\equities_1min')

sp500 = pd.read_csv(sp500_file)['Symbol'].tolist()
# Directories are named "symbol=AAPL" (hive partitioning)
local = [d.name.replace('symbol=', '') for d in local_data.iterdir() if d.is_dir()]
matched = [s for s in sp500 if s in local]
missing = [s for s in sp500 if s not in local]

print(f'S&P 500 symbols: {len(sp500)}')
print(f'Local symbols: {len(local)}')
print(f'Matched: {len(matched)} ({100*len(matched)/len(sp500):.1f}%)')
print(f'Missing: {len(missing)}')
if missing:
    print(f'Missing symbols: {missing[:20]}...')
