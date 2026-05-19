"""Alpaca SIP universe panel loader.

Reads daily closes from canonical Homeguard DuckDB/Parquet storage.
Auxiliary series SPY and VIX are appended automatically.

Convention: NaNs preserved (no forward-fill). Engine.py is responsible
for handling NaN -> forced exit for held symbols.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import List
import pandas as pd

try:
    from src.settings import get_local_storage_dir
except ImportError:
    def get_local_storage_dir():
        return 'data'


AUX_SYMBOLS = ['SPY', 'VIX']


def _read_universe_symbols(universe_csv: Path) -> List[str]:
    """Read symbol list from a CSV with header 'symbol'."""
    df = pd.read_csv(universe_csv)
    if 'symbol' not in df.columns:
        raise ValueError(f"Universe CSV missing 'symbol' column: {universe_csv}")
    return df['symbol'].dropna().astype(str).tolist()


def _read_closes_from_parquet(symbols: List[str], start: datetime, end: datetime) -> pd.DataFrame:
    """Read close prices for `symbols` from Alpaca SIP daily Parquet.

    Returns a wide DataFrame indexed by date with columns = symbols.
    Symbols with no data in window produce all-NaN columns.

    Storage layout under get_local_storage_dir():
        daily/1day/<SYMBOL>.parquet  (canonical schema, 'close' column)
    """
    storage = Path(get_local_storage_dir()) / 'daily' / '1day'
    series_by_symbol: dict[str, pd.Series] = {}
    for sym in symbols:
        path = storage / f'{sym}.parquet'
        if not path.exists():
            continue
        df = pd.read_parquet(path, columns=['timestamp', 'close'])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp').sort_index()
        mask = (df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))
        series_by_symbol[sym] = df.loc[mask, 'close']
    if not series_by_symbol:
        return pd.DataFrame()
    panel = pd.DataFrame(series_by_symbol)
    return panel


def load_universe_panel(
    universe_csv: Path,
    start: datetime,
    end: datetime,
) -> pd.DataFrame:
    """Load wide panel of daily closes for the universe + SPY + VIX.

    Args:
        universe_csv: path to a CSV with header 'symbol'.
        start: inclusive earliest date.
        end: inclusive latest date.

    Returns:
        Wide DataFrame indexed by date, columns = universe symbols + ['SPY', 'VIX'].
        NaNs preserved.
    """
    symbols = _read_universe_symbols(universe_csv)
    all_symbols = list(dict.fromkeys(symbols + AUX_SYMBOLS))
    return _read_closes_from_parquet(all_symbols, start, end)
