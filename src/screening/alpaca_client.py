"""
Alpaca Client Wrapper for Stock Screener.

Provides methods for fetching:
- All tradable assets
- Snapshots (current price/volume)
- Historical bars (for technical indicators)

Supports both IEX (free) and SIP (paid) data feeds.
"""

import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetAssetsRequest
from alpaca.trading.enums import AssetClass, AssetStatus
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import (
    StockBarsRequest,
    StockSnapshotRequest,
)
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.enums import DataFeed

from src.api_key import API_KEY, API_SECRET
from src.utils.logger import get_logger
from src.utils.timezone import tz

logger = get_logger()


@dataclass
class SnapshotData:
    """
    Parsed snapshot data for a single symbol.

    Contains current price, volume, and derived metrics.
    """

    symbol: str
    latest_trade_price: float
    latest_trade_size: int
    latest_trade_time: Optional[datetime]
    bid: Optional[float]
    ask: Optional[float]
    daily_bar_open: Optional[float]
    daily_bar_high: Optional[float]
    daily_bar_low: Optional[float]
    daily_bar_close: Optional[float]
    daily_bar_volume: Optional[int]
    prev_daily_close: Optional[float]
    minute_bar_close: Optional[float]
    minute_bar_volume: Optional[int]

    @property
    def change_pct(self) -> float:
        """Calculate 1-day change percentage from previous close."""
        if self.prev_daily_close and self.prev_daily_close > 0:
            return (self.latest_trade_price / self.prev_daily_close - 1) * 100
        return 0.0

    @property
    def gap_pct(self) -> float:
        """Calculate gap percentage (open vs previous close)."""
        if self.prev_daily_close and self.daily_bar_open and self.prev_daily_close > 0:
            return (self.daily_bar_open / self.prev_daily_close - 1) * 100
        return 0.0

    @property
    def spread_pct(self) -> Optional[float]:
        """Calculate bid-ask spread as percentage."""
        if self.bid and self.ask and self.bid > 0:
            return (self.ask - self.bid) / self.bid * 100
        return None


class AlpacaScreenerClient:
    """
    Client for Alpaca data APIs optimized for screening.

    Supports both IEX (free, 15-min delay) and SIP (paid, real-time) feeds.

    Usage:
        client = AlpacaScreenerClient(paper=True)  # IEX feed

        # Get all tradable symbols
        symbols = client.get_all_tradable_assets()

        # Get snapshots for screening
        snapshots = client.get_snapshots(['AAPL', 'MSFT', 'GOOGL'])

        # Get historical bars for technical indicators
        bars = client.get_historical_bars(['AAPL'], days=50)
    """

    # Rate limiting / retry settings
    MAX_RETRIES = 3
    RETRY_DELAY = 2.0  # seconds
    BATCH_SIZE = 200  # Max symbols per snapshot request

    def __init__(
        self,
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        paper: bool = True,
    ):
        """
        Initialize the Alpaca screener client.

        Args:
            api_key: Alpaca API key (default: from environment)
            secret_key: Alpaca secret key (default: from environment)
            paper: Use IEX feed (True) or SIP feed (False)
        """
        self.api_key = api_key or API_KEY
        self.secret_key = secret_key or API_SECRET

        if not self.api_key or not self.secret_key:
            raise ValueError("Alpaca API credentials not provided or found in environment")

        self.paper = paper
        self.feed = DataFeed.IEX if paper else DataFeed.SIP

        # Initialize clients
        self.trading_client = TradingClient(
            self.api_key, self.secret_key, paper=paper
        )
        self.data_client = StockHistoricalDataClient(
            self.api_key, self.secret_key
        )

        logger.info(
            f"Initialized AlpacaScreenerClient (paper={paper}, feed={self.feed.value})"
        )

    # -------------------------------------------------------------------------
    # Tradable Assets
    # -------------------------------------------------------------------------

    def get_all_tradable_assets(
        self,
        tradable_only: bool = True,
        fractionable_only: bool = False,
    ) -> List[str]:
        """
        Fetch all tradable US equity symbols from Alpaca.

        Args:
            tradable_only: Only include tradable assets
            fractionable_only: Only include fractionable assets

        Returns:
            List of symbol strings (e.g., ['AAPL', 'MSFT', ...])
        """
        try:
            request = GetAssetsRequest(
                status=AssetStatus.ACTIVE,
                asset_class=AssetClass.US_EQUITY,
            )
            assets = self.trading_client.get_all_assets(request)

            symbols = []
            for asset in assets:
                # Filter based on criteria
                if tradable_only and not asset.tradable:
                    continue
                if fractionable_only and not asset.fractionable:
                    continue
                # Skip assets with weird symbols (warrants, units, etc.)
                if " " in asset.symbol or "." in asset.symbol:
                    continue
                symbols.append(asset.symbol)

            logger.info(f"Fetched {len(symbols)} tradable assets from Alpaca")
            return sorted(symbols)

        except Exception as e:
            logger.error(f"Failed to fetch tradable assets: {e}")
            raise

    # -------------------------------------------------------------------------
    # Snapshots
    # -------------------------------------------------------------------------

    def get_snapshots(
        self, symbols: List[str]
    ) -> Dict[str, SnapshotData]:
        """
        Fetch snapshots for multiple symbols.

        Batches requests to avoid rate limits (max 200 per request).

        Args:
            symbols: List of symbols to fetch

        Returns:
            Dict of symbol -> SnapshotData
        """
        if not symbols:
            return {}

        results = {}

        # Process in batches
        for i in range(0, len(symbols), self.BATCH_SIZE):
            batch = symbols[i : i + self.BATCH_SIZE]
            batch_results = self._fetch_snapshots_batch(batch)
            results.update(batch_results)

            # Small delay between batches to avoid rate limiting
            if i + self.BATCH_SIZE < len(symbols):
                time.sleep(0.1)

        return results

    def _fetch_snapshots_batch(
        self, symbols: List[str]
    ) -> Dict[str, SnapshotData]:
        """Fetch snapshots for a single batch of symbols."""
        for attempt in range(self.MAX_RETRIES):
            try:
                request = StockSnapshotRequest(
                    symbol_or_symbols=symbols,
                    feed=self.feed,
                )
                snapshots = self.data_client.get_stock_snapshot(request)

                results = {}
                for symbol, snap in snapshots.items():
                    results[symbol] = self._parse_snapshot(symbol, snap)

                return results

            except Exception as e:
                if attempt < self.MAX_RETRIES - 1:
                    delay = self.RETRY_DELAY * (attempt + 1)
                    logger.warning(
                        f"Snapshot fetch failed (attempt {attempt + 1}), retrying in {delay}s: {e}"
                    )
                    time.sleep(delay)
                else:
                    logger.error(f"Failed to fetch snapshots after {self.MAX_RETRIES} attempts: {e}")
                    raise

        return {}

    def _parse_snapshot(self, symbol: str, snap: Any) -> SnapshotData:
        """Parse Alpaca snapshot response into SnapshotData."""
        # Latest trade
        latest_trade = snap.latest_trade
        trade_price = float(latest_trade.price) if latest_trade else 0.0
        trade_size = int(latest_trade.size) if latest_trade else 0
        trade_time = latest_trade.timestamp if latest_trade else None

        # Latest quote
        latest_quote = snap.latest_quote
        bid = float(latest_quote.bid_price) if latest_quote else None
        ask = float(latest_quote.ask_price) if latest_quote else None

        # Daily bar
        daily_bar = snap.daily_bar
        daily_open = float(daily_bar.open) if daily_bar else None
        daily_high = float(daily_bar.high) if daily_bar else None
        daily_low = float(daily_bar.low) if daily_bar else None
        daily_close = float(daily_bar.close) if daily_bar else None
        daily_volume = int(daily_bar.volume) if daily_bar else None

        # Previous daily bar
        prev_bar = snap.previous_daily_bar
        prev_close = float(prev_bar.close) if prev_bar else None

        # Minute bar
        minute_bar = snap.minute_bar
        minute_close = float(minute_bar.close) if minute_bar else None
        minute_volume = int(minute_bar.volume) if minute_bar else None

        return SnapshotData(
            symbol=symbol,
            latest_trade_price=trade_price,
            latest_trade_size=trade_size,
            latest_trade_time=trade_time,
            bid=bid,
            ask=ask,
            daily_bar_open=daily_open,
            daily_bar_high=daily_high,
            daily_bar_low=daily_low,
            daily_bar_close=daily_close,
            daily_bar_volume=daily_volume,
            prev_daily_close=prev_close,
            minute_bar_close=minute_close,
            minute_bar_volume=minute_volume,
        )

    # -------------------------------------------------------------------------
    # Historical Bars
    # -------------------------------------------------------------------------

    def get_historical_bars(
        self,
        symbols: List[str],
        days: int = 50,
        timeframe: str = "1D",
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch historical bars for multiple symbols.

        Args:
            symbols: List of symbols to fetch
            days: Number of days of history (default: 50)
            timeframe: Bar timeframe: '1D', '1H', '15Min', etc.

        Returns:
            Dict of symbol -> DataFrame with OHLCV columns
        """
        if not symbols:
            return {}

        # Parse timeframe
        tf = self._parse_timeframe(timeframe)

        # Calculate date range
        end_date = tz.now()
        start_date = end_date - timedelta(days=days + 10)  # Extra buffer for weekends

        results = {}

        # Process in batches
        for i in range(0, len(symbols), self.BATCH_SIZE):
            batch = symbols[i : i + self.BATCH_SIZE]
            batch_results = self._fetch_bars_batch(batch, start_date, end_date, tf)
            results.update(batch_results)

            # Small delay between batches
            if i + self.BATCH_SIZE < len(symbols):
                time.sleep(0.1)

        return results

    def _fetch_bars_batch(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        timeframe: TimeFrame,
    ) -> Dict[str, pd.DataFrame]:
        """Fetch historical bars for a single batch of symbols."""
        for attempt in range(self.MAX_RETRIES):
            try:
                request = StockBarsRequest(
                    symbol_or_symbols=symbols,
                    timeframe=timeframe,
                    start=start_date,
                    end=end_date,
                    feed=self.feed,
                )
                bars = self.data_client.get_stock_bars(request)

                results = {}
                for symbol in symbols:
                    if symbol in bars.data:
                        df = self._bars_to_dataframe(bars.data[symbol])
                        if not df.empty:
                            results[symbol] = df

                return results

            except Exception as e:
                if attempt < self.MAX_RETRIES - 1:
                    delay = self.RETRY_DELAY * (attempt + 1)
                    logger.warning(
                        f"Bars fetch failed (attempt {attempt + 1}), retrying in {delay}s: {e}"
                    )
                    time.sleep(delay)
                else:
                    logger.error(f"Failed to fetch bars after {self.MAX_RETRIES} attempts: {e}")
                    raise

        return {}

    def _bars_to_dataframe(self, bars: List[Any]) -> pd.DataFrame:
        """Convert Alpaca bars to pandas DataFrame."""
        if not bars:
            return pd.DataFrame()

        data = []
        for bar in bars:
            data.append(
                {
                    "timestamp": bar.timestamp,
                    "open": float(bar.open),
                    "high": float(bar.high),
                    "low": float(bar.low),
                    "close": float(bar.close),
                    "volume": float(bar.volume),
                    "vwap": float(bar.vwap) if hasattr(bar, "vwap") and bar.vwap else None,
                    "trade_count": int(bar.trade_count) if hasattr(bar, "trade_count") and bar.trade_count else None,
                }
            )

        df = pd.DataFrame(data)
        df.set_index("timestamp", inplace=True)
        df.sort_index(inplace=True)

        return df

    def _parse_timeframe(self, timeframe: str) -> TimeFrame:
        """Parse timeframe string to Alpaca TimeFrame."""
        tf_upper = timeframe.upper()

        if tf_upper in ["1D", "D", "DAY", "DAILY"]:
            return TimeFrame(1, TimeFrameUnit.Day)
        elif tf_upper in ["1H", "H", "HOUR", "HOURLY"]:
            return TimeFrame(1, TimeFrameUnit.Hour)
        elif tf_upper in ["15MIN", "15M"]:
            return TimeFrame(15, TimeFrameUnit.Minute)
        elif tf_upper in ["5MIN", "5M"]:
            return TimeFrame(5, TimeFrameUnit.Minute)
        elif tf_upper in ["1MIN", "1M", "MIN", "MINUTE"]:
            return TimeFrame(1, TimeFrameUnit.Minute)
        else:
            logger.warning(f"Unknown timeframe '{timeframe}', defaulting to 1D")
            return TimeFrame(1, TimeFrameUnit.Day)

    # -------------------------------------------------------------------------
    # Feed Management
    # -------------------------------------------------------------------------

    def set_feed(self, feed: str) -> None:
        """
        Set the data feed.

        Args:
            feed: 'iex' or 'sip'
        """
        feed_lower = feed.lower()
        if feed_lower == "iex":
            self.feed = DataFeed.IEX
        elif feed_lower == "sip":
            self.feed = DataFeed.SIP
        else:
            raise ValueError(f"Unknown feed: {feed}. Must be 'iex' or 'sip'")

        logger.info(f"Changed data feed to {self.feed.value}")

    @property
    def is_sip(self) -> bool:
        """Check if using SIP (real-time) feed."""
        return self.feed == DataFeed.SIP

    @property
    def is_iex(self) -> bool:
        """Check if using IEX (free/delayed) feed."""
        return self.feed == DataFeed.IEX
