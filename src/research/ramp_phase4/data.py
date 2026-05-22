"""Alpaca SIP universe panel loader.

Primary source: aggregated daily closes derived from the FRESH 1-min SIP
split tree at `H:/Stock_Data/equities_1min_sip_split/symbol=<SYM>/year=<Y>/
month=<M>/data.parquet`. Aggregation result is cached at
`<storage>/cache/ramp_phase4/equities_daily_from_sip.parquet` (long-form
panel, columns `symbol`, `trade_date`, `close`).

Fallback source (legacy, stale at 2025-12-04): single long-form
Parquet at `<storage>/equities_daily_cache.parquet`.

Auxiliary series:
    SPY  -- from the same panel as the equities (SIP aggregation -> fallback).
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

try:
    from src.utils.logger import logger
except ImportError:  # pragma: no cover
    import logging
    logger = logging.getLogger(__name__)


AUX_SYMBOLS = ['SPY', 'VIX']

SIP_SPLIT_REL = 'equities_1min_sip_split'
SIP_DAILY_CACHE_REL = 'cache/ramp_phase4/equities_daily_from_sip.parquet'
LEGACY_DAILY_CACHE_REL = 'equities_daily_cache.parquet'


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


def _aggregate_symbol_daily(symbol_dir: Path, start: datetime, end: datetime) -> Optional[pd.Series]:
    """Aggregate 1-min bars for a single symbol's partition tree to daily closes.

    Walks `symbol=<SYM>/year=<Y>/month=<M>/data.parquet`, keeps only the
    year/month partitions overlapping `[start, end]`, restricts to RTH
    (09:30-16:00 ET) when possible, and returns the last close per trade date.

    Returns None if no partitions exist or all reads are empty.
    """
    year_dirs = sorted(symbol_dir.glob('year=*'))
    if not year_dirs:
        return None

    start_ym = (start.year, start.month)
    end_ym = (end.year, end.month)

    dfs: List[pd.DataFrame] = []
    for year_dir in year_dirs:
        try:
            year = int(year_dir.name.split('=', 1)[1])
        except (ValueError, IndexError):
            continue
        if year < start.year or year > end.year:
            continue
        for month_dir in sorted(year_dir.glob('month=*')):
            try:
                month = int(month_dir.name.split('=', 1)[1])
            except (ValueError, IndexError):
                continue
            ym = (year, month)
            if ym < start_ym or ym > end_ym:
                continue
            parquet_path = month_dir / 'data.parquet'
            if not parquet_path.exists():
                continue
            try:
                dfs.append(pd.read_parquet(parquet_path, columns=['timestamp', 'close']))
            except Exception as exc:
                logger.warning(f'[phase4] failed reading {parquet_path}: {exc}')
                continue

    if not dfs:
        return None

    bars = pd.concat(dfs, ignore_index=True)
    if bars.empty:
        return None

    ts = bars['timestamp']
    if getattr(ts.dt, 'tz', None) is None:
        ts_et = ts.dt.tz_localize('UTC').dt.tz_convert('America/New_York')
    else:
        ts_et = ts.dt.tz_convert('America/New_York')

    hr = ts_et.dt.hour
    mn = ts_et.dt.minute
    rth_mask = ((hr > 9) | ((hr == 9) & (mn >= 30))) & (hr < 16)
    rth_bars = bars.loc[rth_mask].copy()
    if rth_bars.empty:
        # Fallback: no bars matched the RTH window (rare, malformed month). Use
        # all bars and take last close per ET date.
        rth_bars = bars.copy()
        rth_dates = ts_et.dt.normalize().dt.tz_localize(None)
    else:
        rth_dates = ts_et.loc[rth_mask].dt.normalize().dt.tz_localize(None)
    rth_bars = rth_bars.assign(_date=rth_dates.values)

    daily = rth_bars.groupby('_date')['close'].last()
    daily.index = pd.DatetimeIndex(daily.index)
    # Window filter applied post-aggregation in the parent fn; return all
    # available daily closes here so the cache holds the full symbol history.
    return daily


def _aggregate_to_daily_from_sip_split(
    symbols: List[str], start: datetime, end: datetime
) -> pd.DataFrame:
    """Aggregate the 1-min SIP split tree to a wide daily-close panel.

    Walks `H:/Stock_Data/equities_1min_sip_split/symbol=<SYM>/year=<Y>/month=<M>/data.parquet`
    for each requested symbol, restricts to RTH where possible, and groups by
    trade date taking the LAST close.

    Returns a wide DataFrame indexed by date (tz-naive midnight) with columns
    = requested symbols. Symbols whose `symbol=<SYM>` partition is absent
    contribute an all-NaN column.
    """
    storage = Path(get_local_storage_dir())
    root = storage / SIP_SPLIT_REL
    if not root.exists():
        raise RuntimeError(f'SIP split tree not found at {root}')

    series_map: dict = {}
    missing: List[str] = []
    for i, sym in enumerate(symbols):
        symbol_dir = root / f'symbol={sym}'
        if not symbol_dir.exists():
            missing.append(sym)
            series_map[sym] = pd.Series(dtype='float64', name=sym)
            continue
        daily = _aggregate_symbol_daily(symbol_dir, start, end)
        if daily is None or daily.empty:
            missing.append(sym)
            series_map[sym] = pd.Series(dtype='float64', name=sym)
        else:
            daily.name = sym
            series_map[sym] = daily
        if (i + 1) % 50 == 0:
            logger.info(f'[phase4] SIP daily aggregation progress: {i + 1}/{len(symbols)}')

    if missing:
        logger.warning(f'[phase4] SIP partitions missing for {len(missing)} symbol(s); '
                       f'first few: {missing[:5]}')

    panel = pd.concat(series_map.values(), axis=1)
    panel.columns = list(series_map.keys())
    panel = panel.sort_index()
    return panel


def _load_or_build_sip_daily_cache(
    symbols: List[str], start: datetime, end: datetime
) -> Optional[pd.DataFrame]:
    """Try the SIP-aggregated daily cache; (re)build it if missing or stale.

    Cache layout: long-form Parquet at `<storage>/cache/ramp_phase4/
    equities_daily_from_sip.parquet` with columns (`symbol`, `trade_date`,
    `close`). Easy to merge new dates in without rewriting the whole file.

    Invalidation policy: if the cache exists and its max `trade_date` is at
    least `end`, use it as-is; otherwise rebuild from scratch for the requested
    symbols.

    Returns a wide DataFrame indexed by date with the requested symbols as
    columns (NaN columns for symbols missing from SIP). Returns None if the
    SIP split tree is unavailable.
    """
    storage = Path(get_local_storage_dir())
    cache_path = storage / SIP_DAILY_CACHE_REL
    sip_root = storage / SIP_SPLIT_REL

    if not sip_root.exists():
        return None

    end_ts = pd.Timestamp(end).normalize()
    # Allow up to 7 calendar days of slack so weekends/holidays don't trigger
    # a full rebuild (Friday close is good for the following weekend).
    CACHE_SLACK_DAYS = 7
    use_cache = False
    if cache_path.exists():
        try:
            head = pd.read_parquet(cache_path, columns=['trade_date'])
            max_date = pd.to_datetime(head['trade_date']).max()
            if pd.notna(max_date) and max_date >= end_ts - pd.Timedelta(days=CACHE_SLACK_DAYS):
                use_cache = True
            else:
                logger.info(f'[phase4] SIP daily cache stale (max={max_date.date() if pd.notna(max_date) else None}'
                            f' < end-{CACHE_SLACK_DAYS}d={(end_ts - pd.Timedelta(days=CACHE_SLACK_DAYS)).date()}); rebuilding.')
        except Exception as exc:
            logger.warning(f'[phase4] failed inspecting SIP daily cache; rebuilding: {exc}')

    if not use_cache:
        logger.info(f'[phase4] building SIP daily cache for {len(symbols)} symbols '
                    f'(window {start.date()}..{end.date()}); first run may take 5-15 min.')
        wide = _aggregate_to_daily_from_sip_split(symbols, start, end)
        if wide.empty:
            return None
        # Persist as long-form (small file, easy to extend later).
        long_df = (
            wide.stack(future_stack=True)
            .dropna()
            .reset_index()
        )
        long_df.columns = ['trade_date', 'symbol', 'close']
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        long_df.to_parquet(cache_path, index=False)
        logger.info(f'[phase4] wrote SIP daily cache: {cache_path} '
                    f'({len(long_df):,} rows, {wide.shape[1]} symbols)')

    # Read what we need from the cache.
    df = pd.read_parquet(cache_path, columns=['symbol', 'trade_date', 'close'])
    df['trade_date'] = pd.to_datetime(df['trade_date']).dt.normalize()
    mask = (df['trade_date'] >= pd.Timestamp(start).normalize()) & \
           (df['trade_date'] <= end_ts)
    needed_syms = [s for s in symbols if s != 'VIX']
    df = df.loc[mask & df['symbol'].isin(needed_syms), ['trade_date', 'symbol', 'close']]
    if df.empty:
        return None
    panel = df.pivot_table(
        index='trade_date', columns='symbol', values='close', aggfunc='last'
    ).sort_index()
    return panel


def _read_closes_from_legacy_cache(
    symbols: List[str], start: datetime, end: datetime
) -> pd.DataFrame:
    """Read close prices for `symbols` from the legacy daily Alpaca SIP cache.

    The on-disk layout (verified) is a single long-form Parquet at
    `<storage>/equities_daily_cache.parquet` with columns
    (`symbol`, `trade_date`, `close`, ...). This is the pre-2026-05 path and
    is stale at 2025-12-04; kept here as a fallback only.
    """
    storage = Path(get_local_storage_dir())
    cache_path = storage / LEGACY_DAILY_CACHE_REL
    if not cache_path.exists():
        raise RuntimeError(f'Legacy daily cache not found at {cache_path}')

    df = pd.read_parquet(cache_path, columns=['symbol', 'trade_date', 'close'])
    ts = pd.to_datetime(df['trade_date'])
    if getattr(ts.dt, 'tz', None) is not None:
        ts = ts.dt.tz_convert('America/New_York').dt.tz_localize(None)
    df = df.assign(trade_date=ts.dt.normalize())

    mask = (df['trade_date'] >= pd.Timestamp(start).normalize()) & \
           (df['trade_date'] <= pd.Timestamp(end).normalize())
    df_window = df.loc[mask, ['trade_date', 'symbol', 'close']]
    cache_symbols = [s for s in symbols if s != 'VIX']
    df_window = df_window[df_window['symbol'].isin(cache_symbols)]

    panel = df_window.pivot_table(
        index='trade_date', columns='symbol', values='close', aggfunc='last'
    ).sort_index()
    return panel


def _read_closes_from_parquet(symbols: List[str], start: datetime, end: datetime) -> pd.DataFrame:
    """Read close prices for `symbols` and return a wide DataFrame.

    Source resolution (in order):
      1. Aggregated daily cache from the FRESH 1-min SIP split tree
         (`<storage>/cache/ramp_phase4/equities_daily_from_sip.parquet`),
         rebuilt on demand if missing or its max date is before `end`.
      2. Legacy `<storage>/equities_daily_cache.parquet` (stale ~2025-12-04),
         used only if the SIP path fails or yields no rows.

    VIX is not present in either equities cache; if 'VIX' is requested it is
    fetched from yfinance (single round-trip per call).

    Returns wide DataFrame indexed by tz-naive date with columns = symbols.
    Symbols missing from the resolved source appear as all-NaN columns.
    """
    panel: Optional[pd.DataFrame] = None

    # Source 1: FRESH SIP aggregation.
    try:
        sip_panel = _load_or_build_sip_daily_cache(symbols, start, end)
        if sip_panel is not None and not sip_panel.empty:
            panel = sip_panel
            logger.info(f'[phase4] using FRESH SIP-aggregated daily panel '
                        f'({panel.shape[0]} dates x {panel.shape[1]} symbols).')
    except Exception as exc:
        logger.warning(f'[phase4] SIP daily aggregation failed; '
                       f'falling back to legacy cache: {exc}')

    # Source 2: legacy stale cache.
    if panel is None or panel.empty:
        legacy = _read_closes_from_legacy_cache(symbols, start, end)
        panel = legacy
        logger.warning(f'[phase4] using LEGACY (stale) daily cache '
                       f'({panel.shape[0]} dates x {panel.shape[1]} symbols).')

    # VIX comes from yfinance if requested.
    if 'VIX' in symbols:
        vix = _fetch_vix_yfinance(start, end)
        if vix is not None:
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
