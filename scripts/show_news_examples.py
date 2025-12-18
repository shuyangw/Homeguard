"""Show news headline examples with sentiment scores."""
import sys
sys.path.insert(0, '.')

import pandas as pd
import pyarrow.parquet as pq

def show_news_for_symbol(symbol, date_start, date_end, description):
    """Show news headlines and sentiment for a date range."""
    print(f"\n{'='*70}")
    print(f"{symbol} NEWS: {description}")
    print(f"Date range: {date_start} to {date_end}")
    print('='*70)

    # Load news
    news_path = f'H:\\Stock_Data\\news\\symbol={symbol}\\year=2024\\news.parquet'
    pf = pq.ParquetFile(news_path)
    df = pf.read().to_pandas()

    # Load sentiment
    sent_path = f'H:\\Stock_Data\\sentiment\\symbol={symbol}\\year=2024\\sentiment.parquet'
    pf2 = pq.ParquetFile(sent_path)
    sent = pf2.read().to_pandas()

    # Filter to date range
    df['date'] = pd.to_datetime(df['timestamp']).dt.date
    sent['date'] = pd.to_datetime(sent['timestamp']).dt.date

    start = pd.to_datetime(date_start).date()
    end = pd.to_datetime(date_end).date()

    filtered = sent[(sent['date'] >= start) & (sent['date'] <= end)]

    if filtered.empty:
        print("No news in this date range")
        return

    # Calculate daily average
    daily_avg = filtered.groupby('date')['sentiment_score'].mean()
    print(f"\nDaily average sentiment:")
    for date, avg in daily_avg.items():
        print(f"  {date}: {avg:+.3f}")

    print(f"\nHeadlines ({len(filtered)} articles):")
    print("-"*70)

    for _, row in filtered.iterrows():
        score = row['sentiment_score']
        label = row['sentiment_label']
        ts = row['timestamp'].strftime('%m-%d %H:%M')
        headline = row['headline'][:60]

        # Color indicator
        if score >= 0.3:
            indicator = "[++]"
        elif score >= 0.1:
            indicator = "[+ ]"
        elif score <= -0.3:
            indicator = "[--]"
        elif score <= -0.1:
            indicator = "[- ]"
        else:
            indicator = "[  ]"

        print(f"{indicator} {score:+.2f} | {ts} | {headline}")


def main():
    # AMD - Jan 16 (winning trade +$344)
    show_news_for_symbol('AMD', '2024-01-15', '2024-01-17',
                         'Around winning trade on Jan 16 (+$344)')

    # AMD - Jan 18 (losing trade filtered out)
    show_news_for_symbol('AMD', '2024-01-17', '2024-01-19',
                         'Around filtered trade on Jan 18 (would have lost -$308)')

    # TSLA - Apr 24 (losing trade that passed filter)
    show_news_for_symbol('TSLA', '2024-04-23', '2024-04-25',
                         'Around losing trade on Apr 24 (-$604)')


if __name__ == '__main__':
    main()
