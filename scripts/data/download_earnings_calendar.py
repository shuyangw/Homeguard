"""
Download earnings calendar for backtesting.

Uses yfinance to get historical earnings dates for a list of symbols.
Saves to CSV for use in ORB strategy earnings filter.
"""

import os
import sys
import time
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def get_earnings_dates(symbol: str, start_year: int = 2017, end_year: int = 2024) -> pd.DataFrame:
    """
    Get historical earnings dates for a symbol using yfinance.

    Args:
        symbol: Stock ticker symbol
        start_year: Start year for earnings data
        end_year: End year for earnings data

    Returns:
        DataFrame with columns: symbol, earnings_date, earnings_time (BMO/AMC)
    """
    try:
        ticker = yf.Ticker(symbol)

        # Try to get earnings dates from the earnings_dates property
        earnings_df = ticker.earnings_dates

        if earnings_df is None or len(earnings_df) == 0:
            print(f"  [{symbol}] No earnings data found via earnings_dates")
            return pd.DataFrame()

        # Reset index to get the date as a column
        earnings_df = earnings_df.reset_index()

        # The index is typically 'Earnings Date'
        if 'Earnings Date' in earnings_df.columns:
            date_col = 'Earnings Date'
        elif 'index' in earnings_df.columns:
            date_col = 'index'
        else:
            date_col = earnings_df.columns[0]

        # Rename and select columns
        result = pd.DataFrame({
            'symbol': symbol,
            'earnings_date': pd.to_datetime(earnings_df[date_col]).dt.date,
        })

        # Filter to date range
        start_date = datetime(start_year, 1, 1).date()
        end_date = datetime(end_year, 12, 31).date()

        result = result[
            (result['earnings_date'] >= start_date) &
            (result['earnings_date'] <= end_date)
        ]

        # Remove duplicates
        result = result.drop_duplicates()

        print(f"  [{symbol}] Found {len(result)} earnings dates ({start_year}-{end_year})")
        return result

    except Exception as e:
        print(f"  [{symbol}] Error: {e}")
        return pd.DataFrame()


def download_earnings_calendar(symbols: list, start_year: int = 2017, end_year: int = 2024) -> pd.DataFrame:
    """
    Download earnings calendar for multiple symbols.

    Args:
        symbols: List of stock symbols
        start_year: Start year
        end_year: End year

    Returns:
        Combined DataFrame with all earnings dates
    """
    all_earnings = []

    print(f"\nDownloading earnings data for {len(symbols)} symbols ({start_year}-{end_year})...")
    print("=" * 60)

    for i, symbol in enumerate(symbols):
        earnings_df = get_earnings_dates(symbol, start_year, end_year)
        if len(earnings_df) > 0:
            all_earnings.append(earnings_df)

        # Rate limiting - yfinance can be sensitive
        if (i + 1) % 10 == 0:
            print(f"  Progress: {i + 1}/{len(symbols)} symbols processed")
            time.sleep(1)

    if not all_earnings:
        print("No earnings data found!")
        return pd.DataFrame()

    combined = pd.concat(all_earnings, ignore_index=True)
    combined = combined.sort_values(['symbol', 'earnings_date'])

    return combined


def validate_earnings_data(df: pd.DataFrame, symbols: list, start_year: int, end_year: int) -> dict:
    """
    Validate earnings calendar data quality.

    Returns:
        dict with validation metrics
    """
    if len(df) == 0:
        return {'valid': False, 'reason': 'No data'}

    validation = {
        'total_records': len(df),
        'symbols_with_data': df['symbol'].nunique(),
        'symbols_missing': [],
        'expected_quarters': (end_year - start_year + 1) * 4,
        'coverage_by_symbol': {},
        'date_range': {
            'min': df['earnings_date'].min(),
            'max': df['earnings_date'].max()
        }
    }

    # Check coverage per symbol
    for symbol in symbols:
        symbol_data = df[df['symbol'] == symbol]
        count = len(symbol_data)
        validation['coverage_by_symbol'][symbol] = count

        if count == 0:
            validation['symbols_missing'].append(symbol)

    # Calculate coverage stats
    coverages = list(validation['coverage_by_symbol'].values())
    validation['avg_earnings_per_symbol'] = sum(coverages) / len(coverages) if coverages else 0
    validation['min_coverage'] = min(coverages) if coverages else 0
    validation['max_coverage'] = max(coverages) if coverages else 0

    # Determine if data is valid
    validation['valid'] = (
        validation['symbols_with_data'] >= len(symbols) * 0.8 and  # 80% of symbols have data
        validation['avg_earnings_per_symbol'] >= validation['expected_quarters'] * 0.5  # 50% of expected quarters
    )

    return validation


def main():
    # Read high-vol symbol list
    symbols_file = project_root / "config/universes" / "sp500_highvol.csv"

    if not symbols_file.exists():
        print(f"Symbol list not found: {symbols_file}")
        return

    symbols_df = pd.read_csv(symbols_file)
    if 'symbol' in symbols_df.columns:
        symbols = symbols_df['symbol'].tolist()
    else:
        symbols = symbols_df.iloc[:, 0].tolist()

    print(f"Loaded {len(symbols)} symbols from {symbols_file}")

    # Download earnings data
    start_year = 2017
    end_year = 2024

    earnings_df = download_earnings_calendar(symbols, start_year, end_year)

    if len(earnings_df) == 0:
        print("\nFailed to download earnings data!")
        return

    # Validate data
    print("\n" + "=" * 60)
    print("VALIDATION RESULTS")
    print("=" * 60)

    validation = validate_earnings_data(earnings_df, symbols, start_year, end_year)

    print(f"\nTotal earnings records: {validation['total_records']}")
    print(f"Symbols with data: {validation['symbols_with_data']}/{len(symbols)}")
    print(f"Symbols missing data: {len(validation['symbols_missing'])}")
    if validation['symbols_missing']:
        print(f"  Missing: {', '.join(validation['symbols_missing'][:10])}")
    print(f"Date range: {validation['date_range']['min']} to {validation['date_range']['max']}")
    print(f"Avg earnings per symbol: {validation['avg_earnings_per_symbol']:.1f}")
    print(f"Expected (4 quarters x {end_year - start_year + 1} years): {validation['expected_quarters']}")
    print(f"Coverage range: {validation['min_coverage']} - {validation['max_coverage']} per symbol")

    # Show coverage by symbol
    print("\nCoverage by symbol:")
    for symbol, count in sorted(validation['coverage_by_symbol'].items(), key=lambda x: x[1]):
        status = "[OK]" if count >= 20 else "[LOW]" if count >= 10 else "[!]"
        print(f"  {status} {symbol}: {count} earnings dates")

    # Save to CSV
    output_dir = project_root / "config/universes"
    output_file = output_dir / "earnings_calendar_sp500_highvol.csv"

    earnings_df.to_csv(output_file, index=False)
    print(f"\nSaved earnings calendar to: {output_file}")
    print(f"Total records: {len(earnings_df)}")

    # Also create a summary
    summary_file = output_dir / "earnings_calendar_validation.txt"
    with open(summary_file, 'w') as f:
        f.write(f"Earnings Calendar Validation Report\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"=" * 60 + "\n\n")
        f.write(f"Total records: {validation['total_records']}\n")
        f.write(f"Symbols with data: {validation['symbols_with_data']}/{len(symbols)}\n")
        f.write(f"Date range: {validation['date_range']['min']} to {validation['date_range']['max']}\n")
        f.write(f"Avg earnings per symbol: {validation['avg_earnings_per_symbol']:.1f}\n")
        f.write(f"Data valid: {validation['valid']}\n\n")
        f.write("Coverage by symbol:\n")
        for symbol, count in sorted(validation['coverage_by_symbol'].items(), key=lambda x: -x[1]):
            f.write(f"  {symbol}: {count}\n")

    print(f"Saved validation report to: {summary_file}")

    if validation['valid']:
        print("\n[+] Earnings data is VALID and ready for backtesting!")
    else:
        print("\n[!] Earnings data has coverage issues - review before using")


if __name__ == "__main__":
    main()
