"""DataAcquisitionManager - unified orchestrator for all data downloads."""

from typing import Any, Dict, List, Optional

from src.data.acquisition.plugins import PLUGIN_REGISTRY, get_plugin_class
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataAcquisitionManager:
    """Single entry point for all data acquisition operations."""

    def list_sources(self) -> List[str]:
        return list(PLUGIN_REGISTRY.keys())

    def download(
        self,
        source: str,
        symbols: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        skip_existing: bool = False,
        **kwargs,
    ) -> Any:
        """Download data from a specific source.

        Args:
            source: Data source name (equities, crypto, futures, news)
            symbols: List of symbols to download
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            skip_existing: Skip symbols with existing data
            **kwargs: Additional args passed to plugin constructor
        """
        if source not in PLUGIN_REGISTRY:
            raise ValueError(
                f"Unknown source '{source}'. Available: {self.list_sources()}"
            )

        plugin_cls = get_plugin_class(source)
        plugin = plugin_cls(**kwargs)

        download_kwargs: Dict[str, Any] = {"skip_existing": skip_existing}
        if symbols:
            download_kwargs["symbols"] = symbols
        if start_date:
            download_kwargs["start_date"] = start_date
        if end_date:
            download_kwargs["end_date"] = end_date

        logger.info(f"Starting download: source={source}, symbols={symbols}")
        result = plugin.download(**download_kwargs)
        logger.info(
            f"Download complete: {result.succeeded}/{result.total_symbols} succeeded, "
            f"{result.failed} failed, {result.total_rows:,} rows"
        )
        return result

    def get_status(self, source: str) -> Dict[str, int]:
        """Get manifest status for a source."""
        from src.data.acquisition.manifest import DownloadManifest
        from src.settings import (
            CRYPTO_ALPACA_1MIN,
            EQUITIES_IEX_1MIN,
            FUTURES_DATABENTO_TRADES,
            NEWS_ALPACA,
            get_local_storage_dir,
        )

        storage_subdir = {
            "equities": EQUITIES_IEX_1MIN,
            "crypto": CRYPTO_ALPACA_1MIN,
            "futures": FUTURES_DATABENTO_TRADES,
            "news": NEWS_ALPACA,
        }.get(source, source)

        manifest = DownloadManifest(get_local_storage_dir(), storage_subdir)
        return manifest.summary()
