"""
Symbol loading utilities for reading stock symbols from various sources.
"""

import pandas as pd
from pathlib import Path


class SymbolLoader:
    """
    Loads stock symbols from CSV files or other sources.
    """

    @staticmethod
    def from_csv(csv_path, symbol_column='Symbol'):
        """
        Load stock symbols from a CSV file.

        Args:
            csv_path: Path to the CSV file containing stock symbols
            symbol_column: Name of the column containing symbols (default: 'Symbol')

        Returns:
            List of stock symbols

        Raises:
            FileNotFoundError: If CSV file doesn't exist
            ValueError: If specified column not found in CSV
        """
        csv_file = Path(csv_path)
        if not csv_file.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        df = pd.read_csv(csv_file)

        if symbol_column not in df.columns:
            raise ValueError(
                f"Column '{symbol_column}' not found in CSV. "
                f"Available columns: {df.columns.tolist()}"
            )

        symbols = df[symbol_column].tolist()
        print(f"Loaded {len(symbols)} symbols from {csv_path}")
        return symbols

    @staticmethod
    def from_list(symbols):
        """
        Use a predefined list of symbols.

        Args:
            symbols: List of stock symbols

        Returns:
            List of stock symbols
        """
        print(f"Using {len(symbols)} symbols from provided list")
        return symbols

    @staticmethod
    def from_text_file(file_path):
        """
        Load symbols from a text file (one symbol per line).

        Args:
            file_path: Path to text file

        Returns:
            List of stock symbols

        Raises:
            FileNotFoundError: If text file doesn't exist
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Text file not found: {file_path}")

        with open(file_path, 'r') as f:
            symbols = [line.strip() for line in f if line.strip()]

        print(f"Loaded {len(symbols)} symbols from {file_path}")
        return symbols
