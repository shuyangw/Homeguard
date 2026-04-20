"""
IBKR Historical Data Provider - Implements DataProviderInterface.

Slots into CompositeDataProvider as a data source alongside Alpaca
and yfinance.  Handles IBKR's duration/barSize constraints transparently.

DataFrame contract (from DataProviderInterface):
    - Index: DatetimeIndex with America/New_York timezone
    - Columns: open, high, low, close, volume (lowercase)
    - Returns: pd.DataFrame or None on failure (enables fallback)
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd
import pytz

from ib_async import util as ib_util

from src.data.providers.base import DataProviderInterface
from src.trading.brokers.ibkr.connection import IBKRConnectionManager
from src.trading.brokers.ibkr.contracts import ContractResolver
from src.trading.brokers.ibkr.pacing import PacingManager
from src.utils.logger import get_logger

logger = get_logger(__name__)

ET = pytz.timezone('America/New_York')

# IBKR max duration per barSize for auto-chunking
_TIMEFRAME_MAP = {
    "1Min": ("1 min", timedelta(days=1), "1 D"),
    "5Min": ("5 mins", timedelta(weeks=1), "1 W"),
    "15Min": ("15 mins", timedelta(weeks=2), "2 W"),
    "1Hour": ("1 hour", timedelta(days=30), "1 M"),
    "1H": ("1 hour", timedelta(days=30), "1 M"),
    "1D": ("1 day", timedelta(days=365), "1 Y"),
    "1Day": ("1 day", timedelta(days=365), "1 Y"),
}


class IBKRDataProvider(DataProviderInterface):
    """
    Historical data provider backed by IBKR.

    Implements DataProviderInterface exactly (including force_refresh param).
    Also provides options-specific methods beyond the base interface.
    """

    def __init__(
        self,
        connection: IBKRConnectionManager,
        resolver: Optional[ContractResolver] = None,
        pacer: Optional[PacingManager] = None,
    ):
        self._conn = connection
        self._resolver = resolver or ContractResolver(connection)
        self._pacer = pacer or PacingManager(
            max_per_10min=connection.config.historical_pacing_max_per_10min,
            identical_cooldown=connection.config.identical_request_cooldown_seconds,
        )

    @property
    def name(self) -> str:
        return "IBKR"

    def is_available(self) -> bool:
        return self._conn.is_connected

    def supports_timeframe(self, timeframe: str) -> bool:
        return timeframe in _TIMEFRAME_MAP

    # ---- DataProviderInterface ----

    def get_historical_bars(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: str = '1D',
        force_refresh: bool = False,
    ) -> Optional[pd.DataFrame]:
        """
        Fetch historical bars with automatic chunking and pacing.

        Returns DataFrame with America/New_York timezone index,
        lowercase columns (open, high, low, close, volume).
        Returns None on failure.
        """
        if timeframe not in _TIMEFRAME_MAP:
            logger.warning(f"[IBKR] Unsupported timeframe '{timeframe}'")
            return None

        bar_size, max_chunk_delta, _ = _TIMEFRAME_MAP[timeframe]

        try:
            contract = self._resolver.resolve_stock(symbol)
        except Exception as e:
            logger.warning(f"[IBKR] Cannot resolve {symbol}: {e}")
            return None

        chunks = self._compute_chunks(start, end, max_chunk_delta)
        all_bars = []

        for chunk_start, chunk_end in chunks:
            chunk_duration = self._delta_to_duration(chunk_end - chunk_start)

            pacing_key = self._pacer.request_key(
                symbol, chunk_duration, bar_size, "TRADES",
                chunk_end.strftime("%Y%m%d %H:%M:%S"),
            )
            self._pacer.acquire(pacing_key)

            try:
                bars = self._conn.run_sync(
                    self._conn.ib.reqHistoricalDataAsync(
                        contract,
                        endDateTime=chunk_end.strftime("%Y%m%d %H:%M:%S"),
                        durationStr=chunk_duration,
                        barSizeSetting=bar_size,
                        whatToShow="TRADES",
                        useRTH=True,
                        formatDate=2,
                    ),
                    timeout=60,
                )
                if bars:
                    all_bars.extend(bars)
            except Exception as e:
                logger.warning(
                    f"[IBKR] Failed to fetch {symbol} "
                    f"{chunk_start.date()} -> {chunk_end.date()}: {e}"
                )
                continue

        if not all_bars:
            return None

        df = ib_util.df(all_bars)
        return self._normalize_dataframe(df, start, end)

    def get_historical_bars_batch(
        self,
        symbols: List[str],
        start: datetime,
        end: datetime,
        timeframe: str = '1D',
        force_refresh: bool = False,
    ) -> Dict[str, pd.DataFrame]:
        """Fetch bars for multiple symbols. Returns {symbol: DataFrame}."""
        results = {}
        for symbol in symbols:
            df = self.get_historical_bars(symbol, start, end, timeframe, force_refresh)
            if df is not None and not df.empty:
                results[symbol] = df
        return results

    # ---- Options-specific (beyond DataProviderInterface) ----

    def get_options_chain_data(
        self,
        underlying: str,
        expiration: Optional[str] = None,
    ) -> List[Dict]:
        """
        Fetch option chain data via IBKR.

        Returns List[Dict] matching OptionsTradingInterface.get_options_chain()
        format so IBKRBroker can delegate directly.
        """
        try:
            contract = self._resolver.resolve_stock(underlying)
        except Exception as e:
            logger.warning(f"[IBKR] Cannot resolve {underlying}: {e}")
            return []

        chains = self._conn.run_sync(
            self._conn.ib.reqSecDefOptParamsAsync(
                contract.symbol, "", contract.secType, contract.conId
            )
        )

        if not chains:
            logger.warning(f"[IBKR] No option chain found for {underlying}")
            return []

        # Find SMART exchange chain
        smart_chain = next((c for c in chains if c.exchange == "SMART"), chains[0])

        expirations = sorted(smart_chain.expirations)
        if expiration:
            exp_str = expiration if isinstance(expiration, str) else expiration.strftime("%Y%m%d")
            expirations = [e for e in expirations if e == exp_str]

        # Return chain metadata (strikes/expirations available)
        # Full quote data requires per-contract snapshot (expensive)
        results = []
        for exp in expirations:
            for strike in sorted(smart_chain.strikes):
                for opt_type in ("call", "put"):
                    results.append({
                        'contract_id': f"{underlying}_{exp}_{strike}_{opt_type[0].upper()}",
                        'underlying': underlying,
                        'expiration': exp,
                        'strike': strike,
                        'option_type': opt_type,
                        'bid': None,
                        'ask': None,
                        'last': None,
                        'volume': None,
                        'open_interest': None,
                        'implied_volatility': None,
                    })

        logger.info(f"[IBKR] Option chain for {underlying}: {len(results)} contracts")
        return results

    # ---- Internal ----

    @staticmethod
    def _compute_chunks(
        start: datetime, end: datetime, max_delta: timedelta,
    ) -> list:
        chunks = []
        current = start
        while current < end:
            chunk_end = min(current + max_delta, end)
            chunks.append((current, chunk_end))
            current = chunk_end
        return chunks

    @staticmethod
    def _delta_to_duration(delta: timedelta) -> str:
        days = delta.days
        if days <= 1:
            return "1 D"
        elif days <= 7:
            return f"{days} D"
        elif days <= 30:
            return f"{max(1, days // 7)} W"
        elif days <= 365:
            return f"{max(1, days // 30)} M"
        else:
            return f"{max(1, days // 365)} Y"

    def _normalize_dataframe(
        self, df: pd.DataFrame, start: datetime, end: datetime,
    ) -> pd.DataFrame:
        """
        Normalize ib_async DataFrame to Homeguard contract.

        Contract: DatetimeIndex in America/New_York, lowercase columns.
        """
        if df is None or df.empty:
            return pd.DataFrame()

        # ib_async uses 'date' column
        if 'date' in df.columns:
            df = df.set_index('date')

        # ib_async returns date objects for daily bars (plain Index, no .tz)
        # and tz-aware datetimes for intraday. Coerce to DatetimeIndex before
        # any timezone arithmetic.
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        # Ensure ET timezone (matching AlpacaBroker contract). Daily bars are
        # naive market-close snapshots -> localize directly to ET. Intraday
        # bars arrive in UTC -> convert.
        if df.index.tz is None:
            df.index = df.index.tz_localize(ET)
        else:
            df.index = df.index.tz_convert(ET)

        # Lowercase columns (ib_async already uses lowercase)
        df.columns = [c.lower() for c in df.columns]

        # Deduplicate and sort
        df = df[~df.index.duplicated(keep='last')]
        df = df.sort_index()

        return df
