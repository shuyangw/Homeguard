"""Diagnose RAMP data issues - check how many symbols have valid momentum scores."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv
load_dotenv()

import pandas as pd
from datetime import datetime, timedelta
import pytz
from pathlib import Path

# Get S&P 500 symbols
project_root = Path(__file__).resolve().parent.parent.parent
csv_path = project_root / 'backtest_lists' / 'sp500-2025.csv'
symbols_df = pd.read_csv(csv_path)
sp500_symbols = symbols_df['Symbol'].tolist()

print("=" * 80)
print("RAMP DATA DIAGNOSIS")
print("=" * 80)
print(f"\nS&P 500 universe: {len(sp500_symbols)} symbols")

# Connect to Alpaca and fetch historical data for a sample
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed

api_key = os.getenv('ALPACA_PAPER_KEY_ID')
secret_key = os.getenv('ALPACA_PAPER_SECRET_KEY')
data_client = StockHistoricalDataClient(api_key, secret_key)

ET = pytz.timezone('America/New_York')
end_date = datetime.now(ET)
start_date = end_date - timedelta(days=60)  # Only need 60 days for momentum calc

print(f"\nFetching daily bars from {start_date.date()} to {end_date.date()}...")
print("This may take 1-2 minutes...")

# Fetch data in batches
prices_dict = {}
failed_symbols = []
batch_size = 100

for i in range(0, len(sp500_symbols), batch_size):
    batch = sp500_symbols[i:i + batch_size]
    try:
        request = StockBarsRequest(
            symbol_or_symbols=batch,
            timeframe=TimeFrame.Day,
            start=start_date,
            end=end_date,
            feed=DataFeed.IEX  # Use IEX feed (free with paper account)
        )
        bars = data_client.get_stock_bars(request)

        for symbol in batch:
            if symbol in bars.data and bars.data[symbol]:
                df = pd.DataFrame([{
                    'timestamp': b.timestamp,
                    'close': b.close
                } for b in bars.data[symbol]])
                if not df.empty:
                    df.set_index('timestamp', inplace=True)
                    prices_dict[symbol] = df['close']
            else:
                failed_symbols.append(symbol)
    except Exception as e:
        print(f"  Batch {i//batch_size + 1} error: {e}")
        failed_symbols.extend(batch)

    print(f"  Fetched {min(i + batch_size, len(sp500_symbols))}/{len(sp500_symbols)} symbols...")

print(f"\nSymbols with data: {len(prices_dict)}")
print(f"Symbols failed: {len(failed_symbols)}")

if failed_symbols:
    print(f"\nFailed symbols (first 20): {failed_symbols[:20]}")

if not prices_dict:
    print("\nERROR: No data fetched!")
    sys.exit(1)

# Create prices DataFrame
prices_df = pd.DataFrame(prices_dict)
print(f"\nPrices DataFrame:")
print(f"  Symbols: {len(prices_df.columns)}")
print(f"  Days: {len(prices_df)}")
print(f"  Date range: {prices_df.index[0]} to {prices_df.index[-1]}")

# Calculate momentum scores like RAMP does
print("\n" + "-" * 40)
print("MOMENTUM SCORE CALCULATION (RAMP formula)")
print("-" * 40)

long_p = 21
short_p = 5
long_w = 0.3
pen_w = 5.0

long_returns = prices_df.pct_change(long_p, fill_method=None)
short_returns = prices_df.pct_change(short_p, fill_method=None)
momentum = (long_w * long_returns) - (pen_w * short_returns)

latest_scores = momentum.iloc[-1]
valid_scores = latest_scores.dropna()

print(f"\nTotal momentum scores: {len(latest_scores)}")
print(f"Valid (non-NaN) scores: {len(valid_scores)}")
print(f"NaN scores: {len(latest_scores) - len(valid_scores)}")

# Why are some NaN?
nan_symbols = latest_scores[latest_scores.isna()].index.tolist()
print(f"\nSymbols with NaN momentum (first 20):")
for sym in nan_symbols[:20]:
    col = prices_df[sym]
    non_nan_count = col.notna().sum()
    last_valid = col.last_valid_index()
    first_valid = col.first_valid_index()
    print(f"  {sym}: {non_nan_count} valid days, first: {first_valid}, last: {last_valid}")

# Check if missing data is recent
print("\n" + "-" * 40)
print("DATA RECENCY CHECK")
print("-" * 40)

latest_date = prices_df.index[-1]
print(f"\nLatest date in data: {latest_date}")

# Check how many symbols have data on the last day
last_day_valid = prices_df.iloc[-1].notna().sum()
print(f"Symbols with data on last day: {last_day_valid}/{len(prices_df.columns)}")

# Check how many symbols have data on last 5 days
for lookback in [1, 5, 10, 21]:
    if len(prices_df) >= lookback:
        valid_in_window = (prices_df.iloc[-lookback:].notna().sum(axis=0) == lookback).sum()
        print(f"Symbols with complete data for last {lookback} days: {valid_in_window}/{len(prices_df.columns)}")

# Show top momentum stocks
if len(valid_scores) > 0:
    print("\n" + "-" * 40)
    print("TOP 20 MOMENTUM STOCKS (would be selected in STRONG_BULL)")
    print("-" * 40)

    ranked = valid_scores.sort_values(ascending=False)
    for i, (sym, score) in enumerate(ranked.head(20).items(), 1):
        print(f"  #{i:2}: {sym:5} ({score:+.2%})")

# Check for special cases - symbols that got delisted, changed tickers, etc
print("\n" + "-" * 40)
print("POTENTIAL ISSUES")
print("-" * 40)

# Check for symbols in S&P 500 list but not in fetched data
missing_from_fetch = set(sp500_symbols) - set(prices_dict.keys())
if missing_from_fetch:
    print(f"\nSymbols in SP500 list but failed to fetch ({len(missing_from_fetch)}):")
    print(f"  {list(missing_from_fetch)[:30]}")

# Check for very low price stocks that might have issues
low_price_stocks = []
for sym in prices_dict:
    if prices_dict[sym].iloc[-1] < 5:
        low_price_stocks.append((sym, prices_dict[sym].iloc[-1]))

if low_price_stocks:
    print(f"\nLow price stocks (< $5) that may have liquidity issues:")
    for sym, price in sorted(low_price_stocks, key=lambda x: x[1]):
        print(f"  {sym}: ${price:.2f}")

print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)
if len(valid_scores) >= 10:
    print(f"\n[OK] {len(valid_scores)} symbols have valid momentum scores")
    print(f"     RAMP should be able to select top 10 in WEAK_BULL regime")
else:
    print(f"\n[WARNING] Only {len(valid_scores)} symbols have valid momentum scores")
    print(f"          RAMP cannot fill all {10} positions in WEAK_BULL regime!")
