"""
Composite Data Provider - Orchestrates fallback chain.

Follows VIXProvider pattern:
1. Try primary provider
2. On failure, try fallback providers in order
3. Cache successful results for resilience
4. Use cache as last resort (or first for cache-first strategy)

Cache-first strategy (for intraday data):
- Check cache first before making ANY API calls
- If cache is fresh (within TTL), return immediately
- Reduces API calls dramatically, avoiding rate limits
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd

from src.data.providers.base import DataProviderInterface
from src.data.providers.cache import DataCache
from src.utils.logger import logger
from src.utils.timezone import tz


class CompositeDataProvider(DataProviderInterface):
    """
    Composite provider with fallback chain and caching.

    Usage:
        providers = [AlpacaDataProvider(broker), YFinanceDataProvider()]
        composite = CompositeDataProvider(providers, cache_enabled=True)

        # Will try Alpaca first, then yfinance, then cache
        df = composite.get_historical_bars('TQQQ', start, end, '1Min')

    Cache-first mode (default for intraday):
        # Check cache first, only call API if cache is stale
        composite = CompositeDataProvider(
            providers,
            cache_first_for_intraday=True,
            intraday_cache_ttl_minutes=5
        )
    """

    def __init__(
        self,
        providers: List[DataProviderInterface],
        cache_enabled: bool = True,
        cache_max_age_hours: int = 24,
        cache_first_for_intraday: bool = True,
        intraday_cache_ttl_minutes: int = 5
    ):
        """
        Initialize composite provider.

        Args:
            providers: List of providers in priority order
            cache_enabled: Enable persistent caching
            cache_max_age_hours: Maximum age for cached daily data before warning
            cache_first_for_intraday: Check cache before API for intraday data
            intraday_cache_ttl_minutes: How long intraday cache is considered fresh
        """
        self._providers = providers
        self._cache_enabled = cache_enabled
        self._cache = DataCache() if cache_enabled else None
        self._cache_max_age_hours = cache_max_age_hours
        self._cache_first_for_intraday = cache_first_for_intraday
        self._intraday_cache_ttl = timedelta(minutes=intraday_cache_ttl_minutes)

        # Track last successful source
        self.last_source: Optional[str] = None
        self.last_fetch_time: Optional[datetime] = None

        provider_names = [p.name for p in providers]
        logger.info(f"[Composite] Initialized with providers: {provider_names}")
        if cache_first_for_intraday:
            logger.info(f"[Composite] Cache-first enabled for intraday (TTL: {intraday_cache_ttl_minutes}min)")

    @property
    def name(self) -> str:
        return "Composite"

    def _is_intraday(self, timeframe: str) -> bool:
        """Check if timeframe is intraday."""
        return timeframe.lower() in ['1min', '5min', '15min', '1hour', '1h']

    def _is_cache_fresh(self, symbol: str, timeframe: str) -> Tuple[bool, Optional[pd.DataFrame]]:
        """
        Check if cache is fresh for given symbol/timeframe.

        Returns:
            Tuple of (is_fresh, cached_data) - cached_data is None if not found
        """
        if not self._cache_enabled or not self._cache:
            return False, None

        # Determine TTL based on timeframe
        if self._is_intraday(timeframe):
            max_age = self._intraday_cache_ttl.total_seconds() / 3600  # Convert to hours
        else:
            max_age = self._cache_max_age_hours

        cached = self._cache.retrieve(symbol, timeframe, max_age_hours=int(max_age) or 1)

        if cached is not None and not cached.empty:
            return True, cached

        return False, None

    def get_historical_bars(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: str = '1D',
        force_refresh: bool = False
    ) -> Optional[pd.DataFrame]:
        """
        Fetch bars with fallback chain and optional cache-first.

        Args:
            symbol: Stock symbol
            start: Start datetime
            end: End datetime
            timeframe: Bar timeframe
            force_refresh: If True, bypass cache and fetch fresh data (use at execution time)
        """

        # Cache-first for intraday data (unless force_refresh)
        if self._cache_first_for_intraday and self._is_intraday(timeframe) and not force_refresh:
            is_fresh, cached = self._is_cache_fresh(symbol, timeframe)
            if is_fresh and cached is not None:
                self.last_source = "cache"
                self.last_fetch_time = datetime.now()
                logger.debug(f"[Composite] {symbol} from cache (fresh, skipping API)")
                return cached

        # Try each provider in order
        for provider in self._providers:
            if not provider.is_available():
                logger.debug(f"[Composite] {provider.name} unavailable, skipping")
                continue

            if not provider.supports_timeframe(timeframe):
                logger.debug(f"[Composite] {provider.name} doesn't support {timeframe}")
                continue

            df = provider.get_historical_bars(symbol, start, end, timeframe)

            if df is not None and not df.empty:
                self.last_source = provider.name
                self.last_fetch_time = datetime.now()

                # Cache successful result
                if self._cache_enabled and self._cache:
                    self._cache.store(symbol, timeframe, df)

                logger.debug(f"[Composite] {symbol} from {provider.name} ({len(df)} bars)")
                return df
            else:
                logger.warning(f"[!] [{provider.name}] No data for {symbol}")

        # All providers failed - try cache as last resort (even if stale)
        if self._cache_enabled and self._cache:
            cached = self._cache.retrieve(
                symbol, timeframe,
                max_age_hours=self._cache_max_age_hours * 24  # Allow very stale data as last resort
            )
            if cached is not None:
                self.last_source = "cache"
                logger.warning(f"[Composite] {symbol} from cache (stale data)")
                return cached

        logger.error(f"[Composite] All sources failed for {symbol}")
        return None

    def get_historical_bars_batch(
        self,
        symbols: List[str],
        start: datetime,
        end: datetime,
        timeframe: str = '1D',
        force_refresh: bool = False
    ) -> Dict[str, pd.DataFrame]:
        """
        Batch fetch with per-symbol fallback and cache-first.

        Args:
            symbols: List of stock symbols
            start: Start datetime
            end: End datetime
            timeframe: Bar timeframe
            force_refresh: If True, bypass cache and fetch fresh data (use at execution time)
        """
        results = {}
        remaining = list(symbols)

        # Cache-first for intraday data (unless force_refresh)
        if self._cache_first_for_intraday and self._is_intraday(timeframe) and not force_refresh:
            cached_count = 0
            for symbol in list(remaining):
                is_fresh, cached = self._is_cache_fresh(symbol, timeframe)
                if is_fresh and cached is not None:
                    results[symbol] = cached
                    remaining.remove(symbol)
                    cached_count += 1

            if cached_count > 0:
                logger.info(f"[Composite] Cache-first: {cached_count} symbols from fresh cache")

            # If all symbols served from cache, we're done
            if not remaining:
                logger.info(f"[Composite] All {len(symbols)} symbols served from cache")
                return results
        elif force_refresh:
            logger.info(f"[Composite] Force refresh: bypassing cache for {len(symbols)} symbols")

        # Try batch fetch from each provider for remaining symbols
        for provider in self._providers:
            if not remaining:
                break

            if not provider.is_available():
                continue

            if not provider.supports_timeframe(timeframe):
                continue

            batch_results = provider.get_historical_bars_batch(
                remaining, start, end, timeframe
            )

            for symbol, df in batch_results.items():
                if df is not None and not df.empty:
                    results[symbol] = df
                    if symbol in remaining:
                        remaining.remove(symbol)

                    # Cache result
                    if self._cache_enabled and self._cache:
                        self._cache.store(symbol, timeframe, df)

        # Try cache for remaining symbols (stale data as last resort)
        if remaining and self._cache_enabled and self._cache:
            for symbol in list(remaining):
                cached = self._cache.retrieve(
                    symbol, timeframe,
                    max_age_hours=self._cache_max_age_hours * 24
                )
                if cached is not None:
                    results[symbol] = cached
                    remaining.remove(symbol)
                    logger.warning(f"[Composite] {symbol} from stale cache")

        if remaining:
            logger.warning(f"[Composite] Failed to fetch: {remaining}")

        logger.info(f"[Composite] Batch complete: {len(results)}/{len(symbols)} symbols")
        return results

    def get_source_info(self) -> Tuple[Optional[str], Optional[datetime]]:
        """Get info about last data source used."""
        return self.last_source, self.last_fetch_time

    def clear_cache(self, symbol: Optional[str] = None) -> None:
        """Clear cache for symbol or all symbols."""
        if self._cache:
            if symbol:
                self._cache.clear_symbol(symbol)
                logger.info(f"[Composite] Cleared cache for {symbol}")
            else:
                self._cache.clear_all()
                logger.info("[Composite] Cleared all cache")
