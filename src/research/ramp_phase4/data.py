"""Alpaca SIP universe panel loader.

Reads daily closes from the wide Homeguard equities cache:
    H:/Stock_Data/equities_daily_cache.parquet
(long format: columns 'symbol', 'trade_date', 'close', ...)

Auxiliary series:
    SPY  -- from the same cache.
    VIX  -- the equities cache does NOT contain VIX; fetched from yfinance
            once per call. (This is the only place yfinance is used for the
            Phase B harness; equities data is always from the Alpaca SIP cache.)

Convention: NaNs preserved (no forward-fill). engine.py is responsible
for handling NaN -> forced exit for held symbols.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional
import pandas as pd

try:
    from src.settings import get_local_storage_dir
except ImportError:
    def get_local_storage_dir():
        return 'data'


AUX_SYMBOLS = ['SPY', 'VIX']


def _read_universe_symbols(universe_csv: Path) -> List[str]:
    """Read symbol list from a CSV.

    Accepts any capitalization of the 'symbol' header
    (`config/universes/sp500-2025.csv` uses `Symbol`).
    """
    df = pd.read_csv(universe_csv)
    col_map = {c.lower(): c for c in df.columns}
    if 'symbol' not in col_map:
        raise ValueError(
            f"Universe CSV missing 'symbol' column (case-insensitive): "
            f"{universe_csv} (cols={list(df.columns)})"
        )
    return df[col_map['symbol']].dropna().astype(str).tolist()


def _fetch_vix_yfinance(start: datetime, end: datetime) -> Optional[pd.Series]:
    """Fetch ^VIX close prices from yfinance.

    Returned series is indexed by tz-naive midnight timestamps. This is the only
    place the Phase B harness uses yfinance; equities data is always from the
    Alpaca SIP cache.
    """
    try:
        import yfinance as yf
    except ImportError:
        return None

    # yfinance end is exclusive.
    df = yf.download(
        '^VIX',
        start=start.strftime('%Y-%m-%d'),
        end=(end + timedelta(days=1)).strftime('%Y-%m-%d'),
        progress=False,
        auto_adjust=True,
    )
    if df is None or df.empty:
        return None
    # yfinance returns multi-level columns; pick close, single ticker.
    if isinstance(df.columns, pd.MultiIndex):
        close = df[('Close', '^VIX')]
    else:
        close = df['Close']
    close = close.copy()
    close.name = 'VIX'
    # Normalize index to tz-naive midnight (matches our equities panel).
    idx = pd.DatetimeIndex(close.index)
    if idx.tz is not None:
        idx = idx.tz_convert('America/New_York').tz_localize(None)
    close.index = idx.normalize()
    return close


def _read_closes_from_parquet(symbols: List[str], start: datetime, end: datetime) -> pd.DataFrame:
    """Read close prices for `symbols` from the daily Alpaca SIP cache.

    The on-disk layout (verified) is a single long-form Parquet at
    `<storage>/equities_daily_cache.parquet` with columns
    (`symbol`, `trade_date`, `close`, `low`).

    Returns a wide DataFrame indexed by tz-naive date with columns = symbols.
    Symbols missing from the cache appear as all-NaN columns.

    VIX is NOT in the cache; if 'VIX' is requested it is fetched from yfinance
    (single round-trip per call).
    """
    storage = Path(get_local_storage_dir())
    cache_path = storage / 'equities_daily_cache.parquet'
    if not cache_path.exists():
        raise RuntimeError(f'Daily cache not found at {cache_path}')

    df = pd.read_parquet(cache_path, columns=['symbol', 'trade_date', 'close'])
    # Drop tz from trade_date so the index is comparable with naive `start`/`end`.
    ts = pd.to_datetime(df['trade_date'])
    if getattr(ts.dt, 'tz', None) is not None:
        ts = ts.dt.tz_convert('America/New_York').dt.tz_localize(None)
    df = df.assign(trade_date=ts.dt.normalize())

    mask = (df['trade_date'] >= pd.Timestamp(start).normalize()) & \
           (df['trade_date'] <= pd.Timestamp(end).normalize())
    df_window = df.loc[mask, ['trade_date', 'symbol', 'close']]

    # Restrict to symbols we will actually need before pivoting (huge memory win).
    cache_symbols = [s for s in symbols if s != 'VIX']
    df_window = df_window[df_window['symbol'].isin(cache_symbols)]

    panel = df_window.pivot_table(
        index='trade_date', columns='symbol', values='close', aggfunc='last'
    ).sort_index()

    # VIX comes from yfinance if requested.
    if 'VIX' in symbols:
        vix = _fetch_vix_yfinance(start, end)
        if vix is not None:
            # Align VIX to panel index (panel may have dates VIX doesn't and vice versa).
            panel = panel.join(vix, how='outer').sort_index()
        else:
            panel['VIX'] = pd.NA

    # Ensure every requested symbol exists as a column (NaN if absent).
    for sym in symbols:
        if sym not in panel.columns:
            panel[sym] = pd.NA

    return panel[symbols]


def load_universe_panel(
    universe_csv: Path,
    start: datetime,
    end: datetime,
) -> pd.DataFrame:
    """Load wide panel of daily closes for the universe + SPY + VIX.

    Args:
        universe_csv: path to a CSV with a 'symbol' column (case-insensitive).
        start: inclusive earliest date.
        end: inclusive latest date.

    Returns:
        Wide DataFrame indexed by date, columns = universe symbols + ['SPY', 'VIX'].
        NaNs preserved.
    """
    symbols = _read_universe_symbols(universe_csv)
    all_symbols = list(dict.fromkeys(symbols + AUX_SYMBOLS))
    return _read_closes_from_parquet(all_symbols, start, end)
