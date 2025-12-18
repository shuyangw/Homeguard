"""Show news with full summaries."""
import sys
sys.path.insert(0, '.')

import pyarrow.parquet as pq
import pandas as pd

def main():
    pf = pq.ParquetFile(r'H:\Stock_Data\news\symbol=AMD\year=2024\news.parquet')
    df = pf.read().to_pandas()

    print("NEWS DATA FIELDS:")
    print(f"  Total articles: {len(df)}")
    print(f"  With summaries: {(df['summary'].notna() & (df['summary'] != '')).sum()}")
    print(f"  With content:   {(df['content'].notna() & (df['content'] != '')).sum()}")
    print()

    # Filter to articles with summaries around Jan 16
    df['date'] = pd.to_datetime(df['timestamp']).dt.date
    jan16 = df[(df['date'] >= pd.to_datetime('2024-01-16').date()) &
               (df['date'] <= pd.to_datetime('2024-01-16').date())]

    with_summary = jan16[jan16['summary'].notna() & (jan16['summary'] != '')]

    print("="*80)
    print("AMD NEWS ON JAN 16, 2024 (Day of winning trade)")
    print("="*80)

    for _, row in with_summary.iterrows():
        print(f"\n[{row['source'].upper()}] {row['timestamp'].strftime('%H:%M')}")
        print(f"HEADLINE: {row['headline']}")
        print(f"AUTHOR: {row['author']}")
        if row['summary']:
            print(f"SUMMARY: {row['summary']}")
        print(f"URL: {row['url']}")
        print("-"*80)

    # Also show one with negative sentiment
    print("\n" + "="*80)
    print("EXAMPLE NEGATIVE NEWS (from Jan 18)")
    print("="*80)

    jan18 = df[(df['date'] >= pd.to_datetime('2024-01-18').date()) &
               (df['date'] <= pd.to_datetime('2024-01-18').date())]
    with_summary = jan18[jan18['summary'].notna() & (jan18['summary'] != '')]

    for _, row in with_summary.head(3).iterrows():
        print(f"\n[{row['source'].upper()}] {row['timestamp'].strftime('%H:%M')}")
        print(f"HEADLINE: {row['headline']}")
        if row['summary']:
            print(f"SUMMARY: {row['summary']}")
        print("-"*80)


if __name__ == '__main__':
    main()
