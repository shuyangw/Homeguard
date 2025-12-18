"""Check data completeness for RAMP."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv
load_dotenv()

import pandas as pd
from pathlib import Path
import pickle

# Check disk cache
CACHE_DIR = Path.home() / '.homeguard' / 'cache'
CACHE_FILE_PREFIX = 'ramp_historical_cache'

print("=" * 80)
print("DATA COMPLETENESS CHECK")
print("=" * 80)

# Check if cache exists locally
print(f"\nCache directory: {CACHE_DIR}")
print(f"Exists: {CACHE_DIR.exists()}")

if CACHE_DIR.exists():
    cache_files = list(CACHE_DIR.glob(f'{CACHE_FILE_PREFIX}_*.pkl'))
    print(f"Cache files found: {len(cache_files)}")

    if cache_files:
        latest = sorted(cache_files)[-1]
        print(f"Latest cache: {latest.name}")

        # Load and inspect
        with open(latest, 'rb') as f:
            cached = pickle.load(f)

        data = cached.get('data', {})
        prices_df = data.get('prices')

        if prices_df is not None:
            print(f"\nPrices DataFrame:")
            print(f"  Symbols: {len(prices_df.columns)}")
            print(f"  Days: {len(prices_df)}")
            print(f"  Date range: {prices_df.index[0]} to {prices_df.index[-1]}")

            # Check for NaN
            nan_counts = prices_df.isna().sum()
            symbols_with_nan = (nan_counts > 0).sum()
            print(f"  Symbols with any NaN: {symbols_with_nan}")

            # Calculate momentum scores like RAMP does
            print("\n  Calculating momentum scores like RAMP...")
            long_p = 21
            short_p = 5
            long_w = 0.3
            pen_w = 5.0

            long_returns = prices_df.pct_change(long_p, fill_method=None)
            short_returns = prices_df.pct_change(short_p, fill_method=None)
            momentum = (long_w * long_returns) - (pen_w * short_returns)

            latest_scores = momentum.iloc[-1]
            valid_scores = latest_scores.dropna()

            print(f"  Total momentum scores: {len(latest_scores)}")
            print(f"  Valid (non-NaN) scores: {len(valid_scores)}")
            print(f"  NaN scores: {len(latest_scores) - len(valid_scores)}")

            # Show why some are NaN
            nan_symbols = latest_scores[latest_scores.isna()].index.tolist()
            if nan_symbols:
                print(f"\n  First 10 symbols with NaN momentum:")
                for sym in nan_symbols[:10]:
                    col = prices_df[sym]
                    non_nan_count = col.notna().sum()
                    last_valid = col.last_valid_index()
                    print(f"    {sym}: {non_nan_count} valid prices, last valid: {last_valid}")

            # Show top momentum stocks
            if len(valid_scores) > 0:
                ranked = valid_scores.sort_values(ascending=False)
                print(f"\n  Top 10 momentum stocks:")
                for i, (sym, score) in enumerate(ranked.head(10).items(), 1):
                    print(f"    #{i}: {sym} ({score:.2%})")
        else:
            print("  No prices DataFrame in cache!")
else:
    print("\nNo cache directory - data hasn't been cached to disk yet")
    print("This is expected since the caching code hasn't been deployed")
