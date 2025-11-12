"""
Metadata storage and retrieval for stock indices.
"""

import pandas as pd
from pathlib import Path

from src.config import settings, OS_ENVIRONMENT


def get_base_path():
    """
    Get the base storage path from settings based on detected OS.

    Returns:
        Path object for the base storage directory
    """
    base_path = settings[OS_ENVIRONMENT]["local_storage_dir"]
    return Path(base_path)


class MetadataStore:
    """
    Manages metadata for stock indices (S&P 500, NASDAQ 100, DJIA, etc.).
    """

    def __init__(self, base_path=None):
        """
        Initialize the metadata store.

        Args:
            base_path: Base directory for metadata storage (defaults to OS-specific setting)
        """
        self.base_path = Path(base_path) if base_path else get_base_path()
        self.metadata_path = self.base_path / "metadata" / "indices"

    def store(self, index_name, symbols_list):
        """
        Store metadata for a stock index.

        Args:
            index_name: Name of the index (e.g., 's_and_p_500', 'nasdaq_100', 'djia')
            symbols_list: List of stock symbols in the index

        Returns:
            Path to the saved metadata file
        """
        # Create metadata directory structure
        self.metadata_path.mkdir(parents=True, exist_ok=True)

        # Create DataFrame with symbol list and timestamp
        metadata_df = pd.DataFrame({
            'symbol': symbols_list,
            'added_date': pd.Timestamp.now(tz='UTC')
        })

        # Save to CSV
        file_path = self.metadata_path / f"{index_name}.csv"
        metadata_df.to_csv(file_path, index=False)

        print(f"Stored metadata for {index_name} with {len(symbols_list)} symbols at {file_path}")
        return file_path

    def load(self, index_name):
        """
        Load metadata for a stock index.

        Args:
            index_name: Name of the index (e.g., 's_and_p_500', 'nasdaq_100', 'djia')

        Returns:
            pandas DataFrame with index metadata, or None if not found
        """
        file_path = self.metadata_path / f"{index_name}.csv"

        if file_path.exists():
            metadata_df = pd.read_csv(file_path)
            print(f"Loaded metadata for {index_name} with {len(metadata_df)} symbols")
            return metadata_df
        else:
            print(f"Metadata file not found: {file_path}")
            return None

    def exists(self, index_name):
        """
        Check if metadata exists for a given index.

        Args:
            index_name: Name of the index

        Returns:
            bool: True if metadata exists, False otherwise
        """
        file_path = self.metadata_path / f"{index_name}.csv"
        return file_path.exists()
