#!/usr/bin/env python3
"""
Data Ingestion Runner Script

Execute this script to run the data ingestion pipeline for stock market data.
Supports ingesting single symbols, lists of symbols, or symbols from CSV files.
"""

from pathlib import Path
from alpaca.data import TimeFrame

from data_engine import IngestionPipeline


def ingest_single_symbol(symbol, start_date, end_date, timeframe=TimeFrame.Minute):
    """
    Ingest data for a single symbol.

    Args:
        symbol: Stock ticker symbol (e.g., 'AAPL')
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        timeframe: Data timeframe (default: Minute)
    """
    pipeline = IngestionPipeline()
    pipeline.ingest_symbols(
        symbols=[symbol],
        start_date=start_date,
        end_date=end_date,
        timeframe=timeframe
    )


def ingest_symbol_list(symbols, start_date, end_date, timeframe=TimeFrame.Minute, index_name=None):
    """
    Ingest data for a list of symbols.

    Args:
        symbols: List of stock ticker symbols (e.g., ['AAPL', 'GOOGL', 'MSFT'])
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        timeframe: Data timeframe (default: Minute)
        index_name: Optional index name for metadata tracking
    """
    pipeline = IngestionPipeline()
    pipeline.ingest_symbols(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        timeframe=timeframe,
        index_name=index_name
    )


def ingest_from_csv(csv_path, start_date, end_date, timeframe=TimeFrame.Minute, index_name=None):
    """
    Ingest data for symbols from a CSV file.

    Args:
        csv_path: Path to CSV file containing symbols
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        timeframe: Data timeframe (default: Minute)
        index_name: Optional index name for metadata tracking
    """
    pipeline = IngestionPipeline()
    pipeline.ingest_from_csv(
        csv_path=csv_path,
        start_date=start_date,
        end_date=end_date,
        timeframe=timeframe,
        index_name=index_name
    )


def main():
    """
    Main entry point for the data ingestion pipeline.
    """
    start_date = "2016-01-01"
    end_date = "2025-01-01"
    timeframe = TimeFrame.Minute

    # Example 1: Ingest a single symbol
    # ingest_single_symbol('AAPL', start_date, end_date, timeframe)

    # Example 2: Ingest a list of symbols
    # ingest_symbol_list(['AAPL', 'GOOGL', 'MSFT'], start_date, end_date, timeframe, index_name='tech_stocks')

    # # Example 3: Ingest from CSV file (default behavior)
    base_path = Path(__file__).parent.parent / "backtest_lists"
    sp500_csv = base_path / "iex_random_1000-2025.csv"
    ingest_from_csv(sp500_csv, start_date, end_date, timeframe)

    # Additional CSV examples (uncomment as needed)
    # nasdaq100_csv = base_path / "nasdaq100-2025.csv"
    # ingest_from_csv(nasdaq100_csv, start_date, end_date, timeframe)
    #
    # djia_csv = base_path / "djia-2025.csv"
    # ingest_from_csv(djia_csv, start_date, end_date, timeframe)


if __name__ == "__main__":
    main()
