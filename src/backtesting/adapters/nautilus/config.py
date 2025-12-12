"""
Configuration dataclasses for NautilusTrader adapters.

These configurations do not depend on NautilusTrader being installed.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any


@dataclass
class NautilusStrategyConfig:
    """
    Configuration for running a Homeguard strategy in NautilusTrader.

    Attributes:
        strategy_class: Name of the Homeguard strategy class (e.g., "RAMPSignals")
        strategy_params: Parameters to initialize the strategy
        symbols: List of symbols to trade
        lookback_periods: Number of historical periods needed for indicators
        position_size_pct: Position size as fraction of portfolio (default: 0.1 = 10%)
        venue: Trading venue identifier (default: "XNAS" for NASDAQ)
    """
    strategy_class: str
    strategy_params: Dict[str, Any]
    symbols: List[str]
    lookback_periods: int
    position_size_pct: float = 0.1
    venue: str = "XNAS"

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if not self.symbols:
            raise ValueError("symbols cannot be empty")
        if self.lookback_periods < 1:
            raise ValueError("lookback_periods must be at least 1")
        if not 0 < self.position_size_pct <= 1:
            raise ValueError("position_size_pct must be between 0 and 1")


@dataclass
class DataConversionConfig:
    """
    Configuration for converting Homeguard parquet data to NautilusTrader catalog.

    Attributes:
        source_dir: Root directory of Homeguard data (e.g., F:\\Stock_Data)
        symbols: List of symbols to convert
        catalog_path: Output path for NautilusTrader catalog.
                      Defaults to {source_dir}/nautilus_catalog/
        start_date: Optional start date filter (YYYY-MM-DD)
        end_date: Optional end date filter (YYYY-MM-DD)
        timeframes: Timeframes to convert (default: minute and day)
        venue: Trading venue identifier (default: "XNAS")
    """
    source_dir: Path
    symbols: List[str]
    catalog_path: Optional[Path] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    timeframes: List[str] = field(default_factory=lambda: ['1min', '1day'])
    venue: str = "XNAS"

    def __post_init__(self) -> None:
        """Set defaults and validate configuration."""
        # Ensure source_dir is a Path
        if isinstance(self.source_dir, str):
            self.source_dir = Path(self.source_dir)

        # Default catalog_path to source_dir/nautilus_catalog
        if self.catalog_path is None:
            self.catalog_path = self.source_dir / 'nautilus_catalog'
        elif isinstance(self.catalog_path, str):
            self.catalog_path = Path(self.catalog_path)

        # Validate
        if not self.symbols:
            raise ValueError("symbols cannot be empty")
        if not self.source_dir.exists():
            raise ValueError(f"source_dir does not exist: {self.source_dir}")

        # Validate timeframes
        valid_timeframes = {'1min', '1hour', '1day'}
        for tf in self.timeframes:
            if tf not in valid_timeframes:
                raise ValueError(
                    f"Invalid timeframe '{tf}'. Must be one of: {valid_timeframes}"
                )

        # Validate date format if provided
        import re
        date_pattern = re.compile(r'^\d{4}-\d{2}-\d{2}$')
        if self.start_date and not date_pattern.match(self.start_date):
            raise ValueError(f"start_date must be YYYY-MM-DD format: {self.start_date}")
        if self.end_date and not date_pattern.match(self.end_date):
            raise ValueError(f"end_date must be YYYY-MM-DD format: {self.end_date}")

    @property
    def timeframe_dir_map(self) -> Dict[str, str]:
        """Map timeframe names to Homeguard directory names."""
        return {
            '1min': 'equities_1min',
            '1hour': 'equities_1hour',
            '1day': 'equities_1day',
        }

    def get_source_pattern(self, symbol: str, timeframe: str) -> str:
        """
        Get glob pattern for finding source parquet files.

        Args:
            symbol: Stock symbol
            timeframe: One of '1min', '1hour', '1day'

        Returns:
            Glob pattern string for matching parquet files
        """
        base_dir = self.timeframe_dir_map.get(timeframe, f'equities_{timeframe}')
        return f"{base_dir}/symbol={symbol}/year=*/month=*/data.parquet"
