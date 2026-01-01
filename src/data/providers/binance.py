"""
Binance Data Provider - Live crypto price data from Binance.

Provides real-time and historical OHLCV data via Binance public REST API.
No authentication required for public market data endpoints.

Features:
- REST API for historical klines and current prices
- Rate limiting with exponential backoff (1200 req/min limit)
- Automatic failover support (used by CryptoDataProviderWithFallback)
- Symbol normalization: BTC/USD -> BTCUSDT

Usage:
    provider = BinanceDataProvider()
    df = provider.get_historical_bars('BTC/USD', start, end, '1D')
    price = provider.get_current_price('BTC/USD')
"""

import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set

import pandas as pd
import requests

from src.data.providers.base import DataProviderInterface
from src.utils.logger import logger


# Binance kline interval mapping
_TIMEFRAME_MAP = {
    '1D': '1d',
    '1d': '1d',
    'day': '1d',
    '1H': '1h',
    '1h': '1h',
    'hour': '1h',
    '1Min': '1m',
    '1min': '1m',
    'minute': '1m',
    '5Min': '5m',
    '5min': '5m',
    '15Min': '15m',
    '15min': '15m',
    '4H': '4h',
    '4h': '4h',
}


def _to_binance_symbol(symbol: str) -> str:
    """
    Convert internal format to Binance format.

    BTC/USD -> BTCUSDT
    ETH/USD -> ETHUSDT
    BTC_USD -> BTCUSDT
    """
    # Remove slashes and underscores, extract base currency
    base = symbol.replace('/USD', '').replace('_USD', '').replace('/USDT', '').replace('_USDT', '')
    return f"{base}USDT"


def _from_binance_symbol(symbol: str) -> str:
    """
    Convert Binance format to internal format.

    BTCUSDT -> BTC/USD
    ETHUSDT -> ETH/USD
    """
    if symbol.endswith('USDT'):
        return f"{symbol[:-4]}/USD"
    return symbol


class BinanceDataProvider(DataProviderInterface):
    """
    Binance data provider for live crypto prices.

    Uses Binance public REST API (no authentication required).

    Rate Limits:
    - 1200 requests per minute for raw requests
    - 10 orders per second for IP
    - This provider only uses public endpoints, no order limits apply

    Symbol Format:
    - Internal: BTC/USD, ETH/USD (matches Alpaca format)
    - Binance: BTCUSDT, ETHUSDT

    Contract:
    - Index: DatetimeIndex with America/New_York timezone
    - Columns: open, high, low, close, volume (lowercase)
    - Returns: pd.DataFrame or None on failure
    """

    BASE_URL = "https://api.binance.com"

    # Rate limiting
    MAX_RETRIES = 3
    RETRY_DELAY = 5  # seconds base delay
    RATE_LIMIT_DELAY = 0.1  # 100ms between requests
    MAX_KLINES_PER_REQUEST = 1000  # Binance limit

    def __init__(self, timeout: int = 30):
        """
        Initialize Binance data provider.

        Args:
            timeout: Request timeout in seconds
        """
        self._timeout = timeout
        self._session = requests.Session()
        self._session.headers.update({
            'Content-Type': 'application/json',
            'User-Agent': 'Homeguard/1.0',
        })
        self._last_request_time = 0.0

    @property
    def name(self) -> str:
        return "Binance"

    def _rate_limit(self) -> None:
        """Enforce rate limiting between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.RATE_LIMIT_DELAY:
            time.sleep(self.RATE_LIMIT_DELAY - elapsed)
        self._last_request_time = time.time()

    def _request_with_retry(
        self,
        endpoint: str,
        params: Optional[Dict] = None,
        max_retries: Optional[int] = None
    ) -> Optional[Dict]:
        """
        Make API request with exponential backoff retry.

        Args:
            endpoint: API endpoint (e.g., '/api/v3/klines')
            params: Query parameters
            max_retries: Override default max retries

        Returns:
            JSON response or None on failure
        """
        retries = max_retries or self.MAX_RETRIES
        url = f"{self.BASE_URL}{endpoint}"

        for attempt in range(1, retries + 1):
            try:
                self._rate_limit()

                response = self._session.get(
                    url,
                    params=params,
                    timeout=self._timeout
                )

                # Handle rate limit errors
                if response.status_code == 429:
                    retry_after = int(response.headers.get('Retry-After', 60))
                    logger.warning(
                        f"[Binance] Rate limited, waiting {retry_after}s"
                    )
                    time.sleep(retry_after)
                    continue

                # Handle server errors
                if response.status_code >= 500:
                    raise requests.RequestException(
                        f"Server error: {response.status_code}"
                    )

                response.raise_for_status()
                return response.json()

            except requests.RequestException as e:
                if attempt < retries:
                    delay = self.RETRY_DELAY * (2 ** (attempt - 1))  # Exponential
                    logger.warning(
                        f"[Binance] Request failed (attempt {attempt}/{retries}): {e}. "
                        f"Retrying in {delay}s..."
                    )
                    time.sleep(delay)
                else:
                    logger.error(f"[Binance] Request failed after {retries} attempts: {e}")
                    return None

        return None

    def get_current_price(self, symbol: str) -> Optional[float]:
        """
        Get current price for a symbol.

        Args:
            symbol: Crypto symbol (e.g., 'BTC/USD')

        Returns:
            Current price as float, or None on failure
        """
        binance_symbol = _to_binance_symbol(symbol)

        data = self._request_with_retry(
            '/api/v3/ticker/price',
            params={'symbol': binance_symbol}
        )

        if data and 'price' in data:
            return float(data['price'])

        return None

    def get_current_prices(self, symbols: List[str]) -> Dict[str, float]:
        """
        Get current prices for multiple symbols.

        Args:
            symbols: List of crypto symbols (e.g., ['BTC/USD', 'ETH/USD'])

        Returns:
            Dict mapping symbol -> price
        """
        # Fetch all prices in one request (more efficient)
        data = self._request_with_retry('/api/v3/ticker/price')

        if not data:
            logger.warning("[Binance] Failed to fetch prices")
            return {}

        # Build lookup by Binance symbol
        price_map = {item['symbol']: float(item['price']) for item in data}

        # Map back to internal symbols
        results = {}
        for symbol in symbols:
            binance_symbol = _to_binance_symbol(symbol)
            if binance_symbol in price_map:
                results[symbol] = price_map[binance_symbol]
            else:
                logger.debug(f"[Binance] Symbol {symbol} ({binance_symbol}) not found")

        return results

    def get_historical_bars(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: str = '1D',
        force_refresh: bool = False
    ) -> Optional[pd.DataFrame]:
        """
        Get historical OHLCV bars for a crypto symbol.

        Args:
            symbol: Crypto symbol (e.g., 'BTC/USD')
            start: Start datetime
            end: End datetime
            timeframe: '1D', '1H', '1Min', etc.
            force_refresh: Ignored (always fetches live)

        Returns:
            DataFrame with OHLCV data, or None on failure
        """
        binance_symbol = _to_binance_symbol(symbol)
        interval = _TIMEFRAME_MAP.get(timeframe, '1d')

        # Convert to milliseconds timestamp
        start_ms = int(start.timestamp() * 1000)
        end_ms = int(end.timestamp() * 1000)

        all_klines = []

        # Binance returns max 1000 klines per request, paginate if needed
        current_start = start_ms
        while current_start < end_ms:
            params = {
                'symbol': binance_symbol,
                'interval': interval,
                'startTime': current_start,
                'endTime': end_ms,
                'limit': self.MAX_KLINES_PER_REQUEST,
            }

            data = self._request_with_retry('/api/v3/klines', params=params)

            if not data:
                logger.warning(f"[Binance] Failed to fetch klines for {symbol}")
                break

            if len(data) == 0:
                break

            all_klines.extend(data)

            # Move to next batch
            last_timestamp = data[-1][0]
            current_start = last_timestamp + 1  # +1ms to avoid duplicates

            # Stop if we got fewer than limit (no more data)
            if len(data) < self.MAX_KLINES_PER_REQUEST:
                break

        if not all_klines:
            return None

        # Convert to DataFrame
        # Binance kline format: [open_time, open, high, low, close, volume, close_time, ...]
        df = pd.DataFrame(all_klines, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])

        # Convert types
        df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms', utc=True)
        df['open'] = df['open'].astype(float)
        df['high'] = df['high'].astype(float)
        df['low'] = df['low'].astype(float)
        df['close'] = df['close'].astype(float)
        df['volume'] = df['volume'].astype(float)

        # Set index and convert to Eastern Time
        df = df.set_index('timestamp')
        df.index = df.index.tz_convert('America/New_York')

        # Keep only OHLCV columns
        df = df[['open', 'high', 'low', 'close', 'volume']]

        # Sort by index
        df = df.sort_index()

        logger.debug(f"[Binance] Loaded {len(df)} bars for {symbol}")
        return df

    def get_historical_bars_batch(
        self,
        symbols: List[str],
        start: datetime,
        end: datetime,
        timeframe: str = '1D',
        force_refresh: bool = False
    ) -> Dict[str, pd.DataFrame]:
        """
        Get historical bars for multiple crypto symbols.

        Args:
            symbols: List of crypto symbols (e.g., ['BTC/USD', 'ETH/USD'])
            start: Start datetime
            end: End datetime
            timeframe: Timeframe string
            force_refresh: Ignored (always fetches live)

        Returns:
            Dict mapping symbol -> DataFrame
        """
        results = {}

        for symbol in symbols:
            df = self.get_historical_bars(symbol, start, end, timeframe, force_refresh)
            if df is not None:
                results[symbol] = df

        logger.info(f"[Binance] Loaded {len(results)}/{len(symbols)} symbols")
        return results

    def get_available_symbols(self) -> Set[str]:
        """
        Get list of available trading pairs on Binance.

        Returns:
            Set of symbols in internal format (e.g., {'BTC/USD', 'ETH/USD'})
        """
        data = self._request_with_retry('/api/v3/exchangeInfo')

        if not data or 'symbols' not in data:
            return set()

        symbols = set()
        for item in data['symbols']:
            # Only include USDT pairs (mapped to /USD internally)
            if item['symbol'].endswith('USDT') and item['status'] == 'TRADING':
                internal_symbol = _from_binance_symbol(item['symbol'])
                symbols.add(internal_symbol)

        return symbols

    def is_available(self) -> bool:
        """Check if Binance API is reachable."""
        try:
            data = self._request_with_retry('/api/v3/ping', max_retries=1)
            return data is not None
        except Exception:
            return False

    def supports_timeframe(self, timeframe: str) -> bool:
        """Check if the timeframe is supported."""
        return timeframe in _TIMEFRAME_MAP


class CryptoDataProviderWithFallback:
    """
    Composite crypto data provider with Binance primary and Alpaca fallback.

    Automatically switches to Alpaca after 3 consecutive Binance failures,
    with a 5-minute fallback window before retrying Binance.

    Usage:
        provider = CryptoDataProviderWithFallback()
        df = provider.get_historical_bars('BTC/USD', start, end, '1D')
        prices = provider.get_current_prices(['BTC/USD', 'ETH/USD'])
    """

    FAILURE_THRESHOLD = 3
    FALLBACK_DURATION_MINUTES = 5

    def __init__(self):
        """Initialize with Binance primary and Alpaca fallback."""
        from src.data.providers.crypto import CryptoDataProvider

        self._binance = BinanceDataProvider()
        self._alpaca = CryptoDataProvider()  # Local parquet storage
        self._failure_count = 0
        self._fallback_until: Optional[datetime] = None
        self._using_fallback = False

    @property
    def name(self) -> str:
        return "Binance+Alpaca"

    def _should_use_fallback(self) -> bool:
        """Check if we should use Alpaca fallback."""
        if self._fallback_until and datetime.now() < self._fallback_until:
            return True
        if self._fallback_until and datetime.now() >= self._fallback_until:
            # Fallback window expired, reset and try Binance again
            self._fallback_until = None
            self._failure_count = 0
            self._using_fallback = False
        return False

    def _record_failure(self) -> None:
        """Record a Binance failure and trigger fallback if threshold reached."""
        self._failure_count += 1
        if self._failure_count >= self.FAILURE_THRESHOLD:
            self._fallback_until = datetime.now() + timedelta(
                minutes=self.FALLBACK_DURATION_MINUTES
            )
            self._using_fallback = True
            logger.warning(
                f"[CryptoFallback] Switching to Alpaca for {self.FALLBACK_DURATION_MINUTES} minutes "
                f"after {self._failure_count} Binance failures"
            )

    def _record_success(self) -> None:
        """Record a successful Binance request."""
        self._failure_count = 0

    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price with fallback."""
        if self._should_use_fallback():
            # Alpaca local storage doesn't have real-time prices
            # Fall back to last known price from historical data
            df = self._alpaca.get_historical_bars(
                symbol,
                datetime.now() - timedelta(days=7),
                datetime.now(),
                '1D'
            )
            if df is not None and not df.empty:
                return float(df['close'].iloc[-1])
            return None

        try:
            price = self._binance.get_current_price(symbol)
            if price is not None:
                self._record_success()
                return price
            self._record_failure()
        except Exception as e:
            logger.error(f"[CryptoFallback] Binance error: {e}")
            self._record_failure()

        # Try Alpaca as immediate fallback
        df = self._alpaca.get_historical_bars(
            symbol,
            datetime.now() - timedelta(days=7),
            datetime.now(),
            '1D'
        )
        if df is not None and not df.empty:
            return float(df['close'].iloc[-1])
        return None

    def get_current_prices(self, symbols: List[str]) -> Dict[str, float]:
        """Get current prices for multiple symbols with fallback."""
        if self._should_use_fallback():
            results = {}
            for symbol in symbols:
                price = self.get_current_price(symbol)
                if price is not None:
                    results[symbol] = price
            return results

        try:
            prices = self._binance.get_current_prices(symbols)
            if prices:
                self._record_success()
                return prices
            self._record_failure()
        except Exception as e:
            logger.error(f"[CryptoFallback] Binance error: {e}")
            self._record_failure()

        # Fallback to Alpaca
        results = {}
        for symbol in symbols:
            price = self.get_current_price(symbol)
            if price is not None:
                results[symbol] = price
        return results

    def get_historical_bars(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: str = '1D',
        force_refresh: bool = False
    ) -> Optional[pd.DataFrame]:
        """Get historical bars with fallback."""
        if self._should_use_fallback():
            return self._alpaca.get_historical_bars(
                symbol, start, end, timeframe, force_refresh
            )

        try:
            df = self._binance.get_historical_bars(
                symbol, start, end, timeframe, force_refresh
            )
            if df is not None:
                self._record_success()
                return df
            self._record_failure()
        except Exception as e:
            logger.error(f"[CryptoFallback] Binance error: {e}")
            self._record_failure()

        # Fallback to Alpaca
        return self._alpaca.get_historical_bars(
            symbol, start, end, timeframe, force_refresh
        )

    def get_historical_bars_batch(
        self,
        symbols: List[str],
        start: datetime,
        end: datetime,
        timeframe: str = '1D',
        force_refresh: bool = False
    ) -> Dict[str, pd.DataFrame]:
        """Get historical bars for multiple symbols with fallback."""
        if self._should_use_fallback():
            return self._alpaca.get_historical_bars_batch(
                symbols, start, end, timeframe, force_refresh
            )

        try:
            data = self._binance.get_historical_bars_batch(
                symbols, start, end, timeframe, force_refresh
            )
            if data:
                self._record_success()
                return data
            self._record_failure()
        except Exception as e:
            logger.error(f"[CryptoFallback] Binance error: {e}")
            self._record_failure()

        # Fallback to Alpaca
        return self._alpaca.get_historical_bars_batch(
            symbols, start, end, timeframe, force_refresh
        )

    def is_available(self) -> bool:
        """Check if at least one provider is available."""
        return self._binance.is_available() or self._alpaca.is_available()
