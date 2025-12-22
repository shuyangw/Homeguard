"""
Stock Screener - Main Screener Class.

Provides Finviz-like stock screening using Alpaca's APIs.
Supports price, volume, technical, and fundamental filters.

Usage:
    from src.screening import StockScreener, ScreenerConfig, PriceFilter

    screener = StockScreener(paper=True)

    # Screen all tradable symbols
    config = ScreenerConfig(
        universe=None,  # Fetch all from Alpaca
        price=PriceFilter(min_price=10, max_price=500),
        volume=VolumeFilter(min_avg_volume_20d=500_000),
        max_results=50
    )

    symbols = screener.screen(config)
    # Returns: ['AAPL', 'MSFT', ...]
"""

from typing import Any, Dict, List, Optional

import pandas as pd

from src.screening.alpaca_client import AlpacaScreenerClient, SnapshotData
from src.screening.cache import ScreenerCache, hash_config
from src.screening.filters import (
    ComparisonFilter,
    FundamentalFilter,
    PriceFilter,
    ScreenerConfig,
    ScreenerResult,
    TechnicalFilter,
    VolumeFilter,
    evaluate_comparison,
)
from src.screening.indicators import TechnicalIndicators
from src.utils.logger import get_logger

logger = get_logger()


class StockScreener:
    """
    Main stock screener class.

    Screens symbols based on price, volume, technical, and fundamental filters.
    Fetches data from Alpaca APIs and caches results for performance.

    Usage:
        screener = StockScreener(paper=True)

        config = ScreenerConfig(
            universe=None,  # None = fetch all tradable from Alpaca
            price=PriceFilter(min_price=10),
            technical=TechnicalFilter(rsi_14=('lt', 30)),
            max_results=20
        )

        # Get list of matching symbols
        symbols = screener.screen(config)

        # Get detailed results with metrics
        results = screener.screen_with_details(config)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        paper: bool = True,
        cache_ttl_seconds: int = 60,
        yfinance_provider: Optional[Any] = None,
    ):
        """
        Initialize the stock screener.

        Args:
            api_key: Alpaca API key (default: from environment)
            secret_key: Alpaca secret key (default: from environment)
            paper: Use IEX feed (True) or SIP feed (False)
            cache_ttl_seconds: Cache TTL for snapshot data
            yfinance_provider: Optional YFinanceProvider for fundamental data
        """
        self.client = AlpacaScreenerClient(
            api_key=api_key,
            secret_key=secret_key,
            paper=paper,
        )
        self.cache = ScreenerCache(
            snapshot_ttl=cache_ttl_seconds,
            historical_ttl=3600,  # 1 hour for bars
            result_ttl=30,  # 30 seconds for results
            assets_ttl=3600,  # 1 hour for asset list
        )
        self.yfinance_provider = yfinance_provider

        logger.info(
            f"Initialized StockScreener (paper={paper}, cache_ttl={cache_ttl_seconds}s)"
        )

    def screen(self, config: ScreenerConfig) -> List[str]:
        """
        Screen symbols based on configuration.

        Args:
            config: ScreenerConfig with filter criteria

        Returns:
            List of matching symbol strings, sorted and limited by config
        """
        results = self.screen_with_details(config)
        return [r.symbol for r in results]

    def screen_with_details(self, config: ScreenerConfig) -> List[ScreenerResult]:
        """
        Screen symbols and return detailed results.

        Args:
            config: ScreenerConfig with filter criteria

        Returns:
            List of ScreenerResult objects with all computed metrics
        """
        # 1. Check result cache
        config_hash = hash_config(config.model_dump_json())
        cached = self.cache.get_result(config_hash)
        if cached is not None:
            logger.debug(f"Returning cached result for config hash {config_hash}")
            # Note: cached results are symbol lists, need to re-fetch details
            # For simplicity, we skip result caching for detailed results
            pass

        # 2. Determine universe
        universe = self._get_universe(config)
        if not universe:
            logger.warning("Empty universe after filtering")
            return []

        logger.info(f"Screening {len(universe)} symbols...")

        # 3. Fetch snapshots
        snapshots = self._get_snapshots(universe)
        logger.debug(f"Fetched {len(snapshots)} snapshots")

        # 4. Apply fast filters (price, current volume)
        candidates = self._apply_fast_filters(snapshots, config)
        logger.debug(f"{len(candidates)} symbols passed fast filters")

        if not candidates:
            return []

        # 5. If fundamental filters needed, apply them
        if config.has_fundamental_filters():
            candidates = self._apply_fundamental_filters(candidates, config)
            logger.debug(f"{len(candidates)} symbols passed fundamental filters")

        if not candidates:
            return []

        # 6. If technical filters needed, fetch historical and apply
        if config.has_technical_filters() or self._needs_historical_for_volume(config):
            candidates = self._apply_technical_filters(candidates, snapshots, config)
            logger.debug(f"{len(candidates)} symbols passed technical filters")

        if not candidates:
            return []

        # 7. Build result objects
        results = self._build_results(candidates, snapshots, config)

        # 8. Sort and limit
        results = self._sort_results(results, config)
        results = results[: config.max_results]

        # 9. Cache result (symbol list only)
        self.cache.set_result(config_hash, [r.symbol for r in results])

        logger.info(f"Screen complete: {len(results)} matches")
        return results

    # -------------------------------------------------------------------------
    # Universe
    # -------------------------------------------------------------------------

    def _get_universe(self, config: ScreenerConfig) -> List[str]:
        """Get the universe of symbols to screen."""
        if config.universe is not None:
            return config.universe

        # Fetch all tradable from Alpaca (cached)
        cached_assets = self.cache.get_assets()
        if cached_assets is not None:
            return cached_assets

        assets = self.client.get_all_tradable_assets()
        self.cache.set_assets(assets)
        return assets

    # -------------------------------------------------------------------------
    # Snapshots
    # -------------------------------------------------------------------------

    def _get_snapshots(self, symbols: List[str]) -> Dict[str, SnapshotData]:
        """Fetch snapshots, using cache where possible."""
        # Check cache for existing snapshots
        cached = self.cache.get_snapshots_batch(symbols)
        uncached = [s for s in symbols if s not in cached]

        if uncached:
            # Fetch missing snapshots
            new_snapshots = self.client.get_snapshots(uncached)
            self.cache.set_snapshots_batch(new_snapshots)
            cached.update(new_snapshots)

        return cached

    # -------------------------------------------------------------------------
    # Fast Filters (Price, Current Volume)
    # -------------------------------------------------------------------------

    def _apply_fast_filters(
        self, snapshots: Dict[str, SnapshotData], config: ScreenerConfig
    ) -> List[str]:
        """Apply fast filters that only need snapshot data."""
        passing = []

        for symbol, snap in snapshots.items():
            if self._passes_price_filters(snap, config.price):
                if self._passes_volume_filters_fast(snap, config.volume):
                    passing.append(symbol)

        return passing

    def _passes_price_filters(
        self, snap: SnapshotData, price_filter: Optional[PriceFilter]
    ) -> bool:
        """Check if snapshot passes price filters."""
        if price_filter is None:
            return True

        price = snap.latest_trade_price

        # Price range
        if price_filter.min_price is not None and price < price_filter.min_price:
            return False
        if price_filter.max_price is not None and price > price_filter.max_price:
            return False

        # 1-day change
        if price_filter.change_1d_pct is not None:
            change = snap.change_pct
            if not evaluate_comparison(change, price_filter.change_1d_pct):
                return False

        # Gap
        if price_filter.gap_pct is not None:
            gap = snap.gap_pct
            if not evaluate_comparison(gap, price_filter.gap_pct):
                return False

        return True

    def _passes_volume_filters_fast(
        self, snap: SnapshotData, volume_filter: Optional[VolumeFilter]
    ) -> bool:
        """Check volume filters that only need current data."""
        if volume_filter is None:
            return True

        volume = snap.daily_bar_volume or 0

        if volume_filter.min_volume is not None and volume < volume_filter.min_volume:
            return False
        if volume_filter.max_volume is not None and volume > volume_filter.max_volume:
            return False

        # Note: min_avg_volume_20d and relative_volume need historical data
        return True

    def _needs_historical_for_volume(self, config: ScreenerConfig) -> bool:
        """Check if volume filters need historical data."""
        if config.volume is None:
            return False
        return (
            config.volume.min_avg_volume_20d is not None
            or config.volume.relative_volume is not None
        )

    # -------------------------------------------------------------------------
    # Fundamental Filters
    # -------------------------------------------------------------------------

    def _apply_fundamental_filters(
        self, symbols: List[str], config: ScreenerConfig
    ) -> List[str]:
        """Apply fundamental filters using YFinance provider."""
        if self.yfinance_provider is None:
            logger.warning(
                "Fundamental filters requested but YFinance provider not configured"
            )
            return symbols

        fundamental_filter = config.fundamental
        if fundamental_filter is None:
            return symbols

        passing = []

        # Fetch fundamentals in batch
        fundamentals = self.yfinance_provider.get_fundamentals(symbols)

        for symbol in symbols:
            fund = fundamentals.get(symbol)
            if fund is None:
                continue

            if self._passes_fundamental_filter(fund, fundamental_filter):
                passing.append(symbol)

        return passing

    def _passes_fundamental_filter(
        self, fund: Any, fundamental_filter: FundamentalFilter
    ) -> bool:
        """Check if fundamentals pass the filter."""
        # Market cap
        if fundamental_filter.min_market_cap is not None:
            if fund.market_cap is None or fund.market_cap < fundamental_filter.min_market_cap:
                return False
        if fundamental_filter.max_market_cap is not None:
            if fund.market_cap is None or fund.market_cap > fundamental_filter.max_market_cap:
                return False

        # P/E ratio
        if fundamental_filter.min_pe_ratio is not None:
            if fund.pe_ratio is None or fund.pe_ratio < fundamental_filter.min_pe_ratio:
                return False
        if fundamental_filter.max_pe_ratio is not None:
            if fund.pe_ratio is None or fund.pe_ratio > fundamental_filter.max_pe_ratio:
                return False

        # Forward P/E
        if fundamental_filter.min_forward_pe is not None:
            if fund.forward_pe is None or fund.forward_pe < fundamental_filter.min_forward_pe:
                return False
        if fundamental_filter.max_forward_pe is not None:
            if fund.forward_pe is None or fund.forward_pe > fundamental_filter.max_forward_pe:
                return False

        # Sectors
        if fundamental_filter.sectors is not None:
            if fund.sector is None or fund.sector not in fundamental_filter.sectors:
                return False
        if fundamental_filter.exclude_sectors is not None:
            if fund.sector is not None and fund.sector in fundamental_filter.exclude_sectors:
                return False

        return True

    # -------------------------------------------------------------------------
    # Technical Filters
    # -------------------------------------------------------------------------

    def _apply_technical_filters(
        self,
        symbols: List[str],
        snapshots: Dict[str, SnapshotData],
        config: ScreenerConfig,
    ) -> List[str]:
        """Apply technical filters using historical data."""
        # Fetch historical bars
        cached_bars = self.cache.get_historical_batch(symbols)
        uncached = [s for s in symbols if s not in cached_bars]

        if uncached:
            new_bars = self.client.get_historical_bars(uncached, days=60)
            self.cache.set_historical_batch(new_bars)
            cached_bars.update(new_bars)

        passing = []

        for symbol in symbols:
            bars = cached_bars.get(symbol)
            if bars is None or bars.empty:
                continue

            snap = snapshots.get(symbol)

            if self._passes_technical_filters(bars, snap, config):
                passing.append(symbol)

        return passing

    def _passes_technical_filters(
        self,
        bars: pd.DataFrame,
        snap: Optional[SnapshotData],
        config: ScreenerConfig,
    ) -> bool:
        """Check if symbol passes all technical filters."""
        try:
            indicators = TechnicalIndicators(bars)
        except Exception:
            return False

        # Volume filters that need historical
        if config.volume:
            if not self._passes_volume_filters_historical(
                indicators, snap, config.volume
            ):
                return False

        # Technical filters
        if config.technical:
            if not self._passes_technical_filter_checks(indicators, config.technical):
                return False

        return True

    def _passes_volume_filters_historical(
        self,
        indicators: TechnicalIndicators,
        snap: Optional[SnapshotData],
        volume_filter: VolumeFilter,
    ) -> bool:
        """Check volume filters that need historical data."""
        avg_volume = indicators.avg_volume(20)

        if volume_filter.min_avg_volume_20d is not None:
            if avg_volume is None or avg_volume < volume_filter.min_avg_volume_20d:
                return False

        if volume_filter.relative_volume is not None:
            rel_vol = indicators.relative_volume(20)
            if rel_vol is None:
                return False
            if not evaluate_comparison(rel_vol, volume_filter.relative_volume):
                return False

        return True

    def _passes_technical_filter_checks(
        self, indicators: TechnicalIndicators, tech_filter: TechnicalFilter
    ) -> bool:
        """Check all technical indicator filters."""
        # RSI
        if tech_filter.rsi_14 is not None:
            rsi = indicators.rsi(14)
            if rsi is None:
                return False
            if not evaluate_comparison(rsi, tech_filter.rsi_14):
                return False

        # SMA Crossover
        if tech_filter.sma_crossover is not None:
            fast, slow, direction = tech_filter.sma_crossover
            if not indicators.is_sma_crossover(fast, slow, direction):
                return False

        # MACD Signal
        if tech_filter.macd_signal is not None:
            signal_type = indicators.macd_signal_type()
            if signal_type is None:
                return False
            if tech_filter.macd_signal != "any" and signal_type != tech_filter.macd_signal:
                return False

        # Bollinger Position
        if tech_filter.bollinger_position is not None:
            position = indicators.bollinger_position()
            if position is None:
                return False
            if position != tech_filter.bollinger_position:
                return False

        # ATR%
        if tech_filter.atr_pct is not None:
            atr_pct = indicators.atr_percent(14)
            if atr_pct is None:
                return False
            if not evaluate_comparison(atr_pct, tech_filter.atr_pct):
                return False

        # Historical Volatility
        if tech_filter.historical_volatility_20d is not None:
            vol = indicators.historical_volatility(20)
            if vol is None:
                return False
            if not evaluate_comparison(vol, tech_filter.historical_volatility_20d):
                return False

        return True

    # -------------------------------------------------------------------------
    # Build Results
    # -------------------------------------------------------------------------

    def _build_results(
        self,
        symbols: List[str],
        snapshots: Dict[str, SnapshotData],
        config: ScreenerConfig,
    ) -> List[ScreenerResult]:
        """Build ScreenerResult objects for passing symbols."""
        results = []

        # Get historical for computing metrics
        bars_cache = self.cache.get_historical_batch(symbols)

        for symbol in symbols:
            snap = snapshots.get(symbol)
            if snap is None:
                continue

            bars = bars_cache.get(symbol)

            result = self._build_single_result(symbol, snap, bars, config)
            results.append(result)

        return results

    def _build_single_result(
        self,
        symbol: str,
        snap: SnapshotData,
        bars: Optional[pd.DataFrame],
        config: ScreenerConfig,
    ) -> ScreenerResult:
        """Build a single ScreenerResult."""
        # Compute indicator values if we have bars
        rsi_14 = None
        atr_pct = None
        avg_volume_20d = None
        relative_volume = None

        if bars is not None and not bars.empty:
            try:
                indicators = TechnicalIndicators(bars)
                rsi_14 = indicators.rsi(14)
                atr_pct = indicators.atr_percent(14)
                avg_volume_20d = indicators.avg_volume(20)
                if avg_volume_20d and avg_volume_20d > 0:
                    current_vol = snap.daily_bar_volume or 0
                    relative_volume = current_vol / avg_volume_20d
            except Exception:
                pass

        # Get fundamentals if available
        market_cap = None
        pe_ratio = None
        sector = None
        if self.yfinance_provider is not None:
            try:
                fund = self.yfinance_provider.get_fundamentals([symbol]).get(symbol)
                if fund:
                    market_cap = fund.market_cap
                    pe_ratio = fund.pe_ratio
                    sector = fund.sector
            except Exception:
                pass

        return ScreenerResult(
            symbol=symbol,
            price=snap.latest_trade_price,
            change_1d_pct=snap.change_pct,
            volume=snap.daily_bar_volume or 0,
            relative_volume=relative_volume,
            avg_volume_20d=int(avg_volume_20d) if avg_volume_20d else None,
            gap_pct=snap.gap_pct,
            rsi_14=rsi_14,
            atr_pct=atr_pct,
            market_cap=market_cap,
            pe_ratio=pe_ratio,
            sector=sector,
        )

    # -------------------------------------------------------------------------
    # Sorting
    # -------------------------------------------------------------------------

    def _sort_results(
        self, results: List[ScreenerResult], config: ScreenerConfig
    ) -> List[ScreenerResult]:
        """Sort results by the specified field."""
        sort_field = config.sort_by
        reverse = config.sort_descending

        def get_sort_key(r: ScreenerResult) -> float:
            value = getattr(r, sort_field, None)
            if value is None:
                return float("-inf") if reverse else float("inf")
            return float(value)

        return sorted(results, key=get_sort_key, reverse=reverse)

    # -------------------------------------------------------------------------
    # Cache Management
    # -------------------------------------------------------------------------

    def clear_cache(self) -> None:
        """Clear all caches."""
        self.cache.clear()
        logger.info("Screener cache cleared")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return self.cache.get_detailed_stats()
