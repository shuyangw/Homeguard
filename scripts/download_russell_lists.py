"""
Download Russell 1000 and Russell 2000 stock lists from multiple sources.
Converts to backtest_lists schema: Ranking, Company Name, Symbol
"""

import pandas as pd
import requests
from io import StringIO
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import get_logger

logger = get_logger(__name__)

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
}


def fetch_russell1000_github() -> pd.DataFrame | None:
    """Fetch Russell 1000 from GitHub source."""
    url = "https://raw.githubusercontent.com/asafravid/sss/master/Indices/russell1000.csv"
    logger.info(f"Downloading Russell 1000 from GitHub: {url}")

    try:
        response = requests.get(url, headers=HEADERS, timeout=30)
        response.raise_for_status()

        df = pd.read_csv(StringIO(response.text))
        logger.info(f"Downloaded {len(df)} Russell 1000 stocks")
        return df
    except Exception as e:
        logger.error(f"GitHub Russell 1000 fetch failed: {e}")
        return None


def fetch_russell2000_github() -> pd.DataFrame | None:
    """Fetch Russell 2000 from GitHub source."""
    url = "https://raw.githubusercontent.com/ikoniaris/Russell2000/master/russell_2000_components.csv"
    logger.info(f"Downloading Russell 2000 from GitHub: {url}")

    try:
        response = requests.get(url, headers=HEADERS, timeout=30)
        response.raise_for_status()

        df = pd.read_csv(StringIO(response.text))
        logger.info(f"Downloaded {len(df)} Russell 2000 stocks")
        return df
    except Exception as e:
        logger.error(f"GitHub Russell 2000 fetch failed: {e}")
        return None


def normalize_dataframe(df: pd.DataFrame, index_name: str) -> pd.DataFrame:
    """Normalize dataframe to standard schema: Ranking, Company Name, Symbol."""
    # Identify ticker and name columns
    ticker_col = None
    name_col = None

    for col in df.columns:
        col_lower = str(col).lower()
        if 'symbol' in col_lower or 'ticker' in col_lower:
            ticker_col = col
        elif 'name' in col_lower:
            name_col = col

    if ticker_col is None or name_col is None:
        logger.error(f"Could not identify columns in {df.columns.tolist()}")
        return pd.DataFrame()

    # Filter valid tickers
    df = df[df[ticker_col].notna()].copy()
    df['_ticker'] = df[ticker_col].astype(str).str.strip().str.upper()

    # Create output in schema format
    result = pd.DataFrame({
        'Ranking': range(1, len(df) + 1),
        'Company Name': df[name_col].astype(str).str.strip(),
        'Symbol': df['_ticker'].values
    })

    # Remove duplicates
    result = result.drop_duplicates(subset=['Symbol'])
    result['Ranking'] = range(1, len(result) + 1)

    logger.info(f"Normalized {len(result)} {index_name} stocks")
    return result


def download_russell_index(index_name: str) -> pd.DataFrame:
    """Download Russell index constituents."""
    logger.info(f"Downloading {index_name} constituents...")

    if index_name == "Russell 1000":
        df = fetch_russell1000_github()
    else:
        df = fetch_russell2000_github()

    if df is None or len(df) < 100:
        raise ValueError(f"Could not download {index_name}")

    return normalize_dataframe(df, index_name)


def main():
    output_dir = project_root / 'backtest_lists'
    output_dir.mkdir(exist_ok=True)

    # Download Russell 1000
    logger.info("=" * 50)
    logger.info("Downloading Russell 1000")
    logger.info("=" * 50)
    try:
        r1000 = download_russell_index("Russell 1000")
        output_file = output_dir / 'russell1000-2025.csv'
        r1000.to_csv(output_file, index=False)
        logger.info(f"Saved Russell 1000 list to {output_file}")
        logger.info(f"Total symbols: {len(r1000)}")
    except Exception as e:
        logger.error(f"Failed to download Russell 1000: {e}")
        raise

    # Download Russell 2000
    logger.info("=" * 50)
    logger.info("Downloading Russell 2000")
    logger.info("=" * 50)
    try:
        r2000 = download_russell_index("Russell 2000")
        output_file = output_dir / 'russell2000-2025.csv'
        r2000.to_csv(output_file, index=False)
        logger.info(f"Saved Russell 2000 list to {output_file}")
        logger.info(f"Total symbols: {len(r2000)}")
    except Exception as e:
        logger.error(f"Failed to download Russell 2000: {e}")
        raise

    logger.info("=" * 50)
    logger.info("Download complete!")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
