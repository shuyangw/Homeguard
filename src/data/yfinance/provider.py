"""
YFinance Fundamentals Provider - Fetches fundamental data from Yahoo Finance.

Provides market cap, P/E ratios, sector, and other fundamental metrics
that Alpaca doesn't offer. Includes persistent caching for efficiency.
"""

import time
from typing import Any, Dict, List, Optional

from src.data.yfinance.cache import FundamentalsCache
from src.data.yfinance.fundamentals import FundamentalData
from src.utils.logger import logger


class YFinanceFundamentalsProvider:
    """
    Provider for fundamental stock data via yfinance.

    Features:
    - Fetches market cap, P/E, sector, dividends, etc.
    - Persistent parquet caching (24-hour TTL)
    - Rate limiting to avoid Yahoo Finance blocks
    - Batch operations with progress logging

    Usage:
        provider = YFinanceFundamentalsProvider()
        fundamentals = provider.get_fundamentals(['AAPL', 'MSFT'])
        apple = fundamentals['AAPL']
        print(f"Market Cap: ${apple.market_cap:.1f}B")
    """

    def __init__(
        self,
        cache_ttl_hours: float = 24,
        rate_limit_delay: float = 0.25,
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize YFinance fundamentals provider.

        Args:
            cache_ttl_hours: Cache TTL in hours (default: 24)
            rate_limit_delay: Delay between requests in seconds (default: 0.25)
            cache_dir: Directory for cache file (default: .cache in project root)
        """
        self._cache = FundamentalsCache(
            cache_dir=cache_dir,
            ttl_hours=cache_ttl_hours,
        )
        self._rate_limit_delay = rate_limit_delay
        self._last_request_time: Optional[float] = None

    def _wait_for_rate_limit(self) -> None:
        """Wait if needed to respect rate limit."""
        if self._last_request_time is not None:
            elapsed = time.time() - self._last_request_time
            if elapsed < self._rate_limit_delay:
                time.sleep(self._rate_limit_delay - elapsed)
        self._last_request_time = time.time()

    def get_fundamentals(
        self,
        symbols: List[str],
        skip_cache: bool = False,
    ) -> Dict[str, FundamentalData]:
        """
        Fetch fundamental data for multiple symbols.

        Args:
            symbols: List of stock symbols
            skip_cache: If True, bypass cache and fetch fresh data

        Returns:
            Dict mapping symbol to FundamentalData
        """
        result = {}
        symbols_to_fetch = []

        # Check cache first
        if not skip_cache:
            for symbol in symbols:
                cached = self._cache.get(symbol)
                if cached is not None:
                    result[symbol] = cached
                else:
                    symbols_to_fetch.append(symbol)
        else:
            symbols_to_fetch = list(symbols)

        # Fetch uncached symbols
        if symbols_to_fetch:
            logger.info(
                f"[YFinanceFundamentals] Fetching {len(symbols_to_fetch)} symbols "
                f"({len(result)} cached)"
            )

            for i, symbol in enumerate(symbols_to_fetch):
                try:
                    self._wait_for_rate_limit()
                    data = self._fetch_single(symbol)
                    if data is not None:
                        result[symbol] = data
                        self._cache.set(symbol, data)

                    # Log progress every 50 symbols
                    if (i + 1) % 50 == 0:
                        logger.info(
                            f"[YFinanceFundamentals] Progress: {i + 1}/{len(symbols_to_fetch)}"
                        )

                except Exception as e:
                    logger.warning(f"[YFinanceFundamentals] Failed to fetch {symbol}: {e}")

            logger.info(
                f"[YFinanceFundamentals] Fetched {len(result)}/{len(symbols)} symbols"
            )

        return result

    def _fetch_single(self, symbol: str) -> Optional[FundamentalData]:
        """
        Fetch fundamental data for a single symbol.

        Args:
            symbol: Stock symbol

        Returns:
            FundamentalData or None on failure
        """
        try:
            import yfinance as yf

            ticker = yf.Ticker(symbol)
            info = ticker.info

            if not info or info.get("symbol") is None:
                logger.debug(f"[YFinanceFundamentals] No data for {symbol}")
                return None

            return FundamentalData.from_yfinance_info(symbol, info)

        except Exception as e:
            logger.debug(f"[YFinanceFundamentals] Error fetching {symbol}: {e}")
            return None

    def get_single(
        self,
        symbol: str,
        skip_cache: bool = False,
    ) -> Optional[FundamentalData]:
        """
        Fetch fundamental data for a single symbol.

        Args:
            symbol: Stock symbol
            skip_cache: If True, bypass cache

        Returns:
            FundamentalData or None
        """
        result = self.get_fundamentals([symbol], skip_cache=skip_cache)
        return result.get(symbol)

    def get_market_cap(self, symbol: str) -> Optional[float]:
        """
        Get market cap in billions for a symbol.

        Args:
            symbol: Stock symbol

        Returns:
            Market cap in billions or None
        """
        data = self.get_single(symbol)
        return data.market_cap if data else None

    def get_sector(self, symbol: str) -> Optional[str]:
        """
        Get sector for a symbol.

        Args:
            symbol: Stock symbol

        Returns:
            Sector string or None
        """
        data = self.get_single(symbol)
        return data.sector if data else None

    def get_pe_ratio(self, symbol: str) -> Optional[float]:
        """
        Get trailing P/E ratio for a symbol.

        Args:
            symbol: Stock symbol

        Returns:
            P/E ratio or None
        """
        data = self.get_single(symbol)
        return data.pe_ratio if data else None

    def get_info(self, symbol: str) -> Dict[str, Any]:
        """
        Get raw yfinance info dict for a symbol.

        Bypasses caching and returns full info dict.

        Args:
            symbol: Stock symbol

        Returns:
            Raw info dict from yfinance
        """
        try:
            import yfinance as yf

            self._wait_for_rate_limit()
            ticker = yf.Ticker(symbol)
            return ticker.info or {}

        except Exception as e:
            logger.error(f"[YFinanceFundamentals] Error fetching info for {symbol}: {e}")
            return {}

    def filter_by_market_cap(
        self,
        symbols: List[str],
        min_cap: Optional[float] = None,
        max_cap: Optional[float] = None,
    ) -> List[str]:
        """
        Filter symbols by market cap.

        Args:
            symbols: List of symbols to filter
            min_cap: Minimum market cap in billions
            max_cap: Maximum market cap in billions

        Returns:
            Filtered list of symbols
        """
        fundamentals = self.get_fundamentals(symbols)

        result = []
        for symbol in symbols:
            data = fundamentals.get(symbol)
            if data is None or data.market_cap is None:
                continue

            if min_cap is not None and data.market_cap < min_cap:
                continue
            if max_cap is not None and data.market_cap > max_cap:
                continue

            result.append(symbol)

        return result

    def filter_by_sector(
        self,
        symbols: List[str],
        include_sectors: Optional[List[str]] = None,
        exclude_sectors: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Filter symbols by sector.

        Args:
            symbols: List of symbols to filter
            include_sectors: Only include these sectors
            exclude_sectors: Exclude these sectors

        Returns:
            Filtered list of symbols
        """
        fundamentals = self.get_fundamentals(symbols)

        result = []
        for symbol in symbols:
            data = fundamentals.get(symbol)
            if data is None or data.sector is None:
                continue

            if include_sectors and data.sector not in include_sectors:
                continue
            if exclude_sectors and data.sector in exclude_sectors:
                continue

            result.append(symbol)

        return result

    def clear_cache(self) -> None:
        """Clear the fundamentals cache."""
        self._cache.clear()

    def get_cache_stats(self) -> Dict:
        """Get cache statistics."""
        return self._cache.get_stats()

    def evict_expired(self) -> int:
        """Evict expired cache entries."""
        return self._cache.evict_expired()
