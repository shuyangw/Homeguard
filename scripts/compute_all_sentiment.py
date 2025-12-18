"""Compute sentiment for all available news data."""
import sys
sys.path.insert(0, '.')

from src.data.news import SentimentCache

def main():
    # Check for --recompute flag
    recompute = '--recompute' in sys.argv

    cache = SentimentCache()

    # List available news
    available = cache.list_available_news()
    print('Available news data:')
    for item in available:
        status = 'YES' if item['has_sentiment'] else 'NO'
        print(f"  {item['symbol']}/{item['year']}: sentiment={status}")

    if recompute:
        # Recompute ALL sentiment (overwrite existing)
        print(f'\nRecomputing ALL sentiment for {len(available)} symbol/year combos...')
        print('Using headline + summary for analysis\n')

        for item in available:
            symbol = item['symbol']
            year = item['year']
            print(f'{symbol}/{year}...', end=' ', flush=True)
            try:
                count = cache.compute_and_save(symbol, year, overwrite=True)
                print(f'{count} articles')
            except Exception as e:
                print(f'ERROR: {e}')
    else:
        # Only compute missing sentiment
        missing = cache.list_missing_sentiment()
        print(f'\nMissing sentiment: {len(missing)} symbol/year combos')

        for item in missing:
            symbol = item['symbol']
            year = item['year']
            print(f'\nComputing sentiment for {symbol}/{year}...')
            try:
                count = cache.compute_and_save(symbol, year, overwrite=False)
                print(f'  Processed {count} articles')
            except Exception as e:
                print(f'  ERROR: {e}')

if __name__ == '__main__':
    main()
