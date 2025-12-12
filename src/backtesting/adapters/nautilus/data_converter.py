"""
Data Converter for NautilusTrader.

Converts Homeguard Hive-partitioned parquet data to NautilusTrader's
ParquetDataCatalog format.

Key Design: COPY, NOT TRANSFORM
- Original Homeguard parquet files remain unchanged
- NautilusTrader catalog is created as a new directory
- Both formats coexist in the same Stock_Data directory
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import pandas as pd

from src.backtesting.adapters.nautilus.config import DataConversionConfig
from src.utils.logger import logger

# Check if NautilusTrader is available
try:
    from nautilus_trader.persistence.catalog import ParquetDataCatalog
    from nautilus_trader.persistence.wranglers import BarDataWrangler
    from nautilus_trader.model.data import Bar, BarType, BarSpecification
    from nautilus_trader.model.instruments import Equity
    from nautilus_trader.model.identifiers import InstrumentId, Symbol, Venue
    from nautilus_trader.model.enums import AggregationSource, BarAggregation, PriceType
    from nautilus_trader.model.objects import Price, Quantity, Money
    from nautilus_trader.model.currencies import USD
    NAUTILUS_AVAILABLE = True
except ImportError:
    NAUTILUS_AVAILABLE = False


# Nanosecond deltas for different timeframes
TS_INIT_DELTAS = {
    '1min': 60_000_000_000,         # 60 seconds in nanoseconds
    '1hour': 3_600_000_000_000,     # 3600 seconds in nanoseconds
    '1day': 86_400_000_000_000,     # 86400 seconds in nanoseconds
}

# Bar aggregation mapping
BAR_AGGREGATION_MAP = {
    '1min': BarAggregation.MINUTE if NAUTILUS_AVAILABLE else None,
    '1hour': BarAggregation.HOUR if NAUTILUS_AVAILABLE else None,
    '1day': BarAggregation.DAY if NAUTILUS_AVAILABLE else None,
}


def load_homeguard_bars(
    source_dir: Path,
    symbol: str,
    timeframe: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> pd.DataFrame:
    """
    Load bars from Hive-partitioned parquet structure.

    Args:
        source_dir: Root data directory (e.g., F:\\Stock_Data)
        symbol: Stock symbol (e.g., "AAPL")
        timeframe: One of '1min', '1hour', '1day'
        start_date: Optional start date filter (YYYY-MM-DD)
        end_date: Optional end date filter (YYYY-MM-DD)

    Returns:
        DataFrame with columns: timestamp, open, high, low, close, volume
        (and optionally trade_count, vwap if present)

    Raises:
        FileNotFoundError: If no parquet files found for symbol
    """
    # Map timeframe to directory name
    timeframe_dirs = {
        '1min': 'equities_1min',
        '1hour': 'equities_1hour',
        '1day': 'equities_1day',
    }
    base_dir = timeframe_dirs.get(timeframe, f'equities_{timeframe}')

    # Find all parquet files for this symbol
    pattern = source_dir / base_dir / f'symbol={symbol}' / 'year=*' / 'month=*' / 'data.parquet'
    parquet_files = list(source_dir.glob(f'{base_dir}/symbol={symbol}/year=*/month=*/data.parquet'))

    if not parquet_files:
        raise FileNotFoundError(
            f"No parquet files found for {symbol} in {source_dir / base_dir}"
        )

    # Load and concatenate all parquet files
    dfs = []
    for pf in parquet_files:
        try:
            df = pd.read_parquet(pf)
            dfs.append(df)
        except Exception as e:
            logger.warning(f"Failed to read {pf}: {e}")
            continue

    if not dfs:
        raise FileNotFoundError(f"No valid parquet files found for {symbol}")

    # Concatenate and sort
    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.sort_values('timestamp').reset_index(drop=True)

    # Apply date filters
    if start_date:
        combined = combined[combined['timestamp'] >= pd.Timestamp(start_date, tz='UTC')]
    if end_date:
        combined = combined[combined['timestamp'] <= pd.Timestamp(end_date, tz='UTC')]

    # Ensure canonical schema (lowercase columns)
    combined.columns = [c.lower() for c in combined.columns]

    return combined


def create_equity_instrument(
    symbol: str,
    venue: str = "XNAS",
    price_precision: int = 2,
    size_precision: int = 0
) -> Any:
    """
    Create NautilusTrader Equity instrument for symbol.

    Args:
        symbol: Stock symbol (e.g., "AAPL")
        venue: Trading venue (default: "XNAS" for NASDAQ)
        price_precision: Decimal places for prices (default: 2)
        size_precision: Decimal places for quantities (default: 0)

    Returns:
        NautilusTrader Equity instrument

    Raises:
        RuntimeError: If NautilusTrader is not installed
    """
    if not NAUTILUS_AVAILABLE:
        raise RuntimeError("NautilusTrader is not installed")

    instrument_id = InstrumentId(Symbol(symbol), Venue(venue))

    return Equity(
        instrument_id=instrument_id,
        raw_symbol=Symbol(symbol),
        currency=USD,
        price_precision=price_precision,
        price_increment=Price.from_str(f"0.{'0' * (price_precision - 1)}1"),
        lot_size=Quantity.from_str("1"),
        max_quantity=None,
        min_quantity=Quantity.from_str("1"),
        max_price=None,
        min_price=None,
        ts_event=0,
        ts_init=0,
    )


def create_bar_type(
    symbol: str,
    timeframe: str,
    venue: str = "XNAS"
) -> Any:
    """
    Create NautilusTrader BarType for given symbol and timeframe.

    Args:
        symbol: Stock symbol
        timeframe: One of '1min', '1hour', '1day'
        venue: Trading venue

    Returns:
        NautilusTrader BarType

    Raises:
        RuntimeError: If NautilusTrader is not installed
        ValueError: If invalid timeframe
    """
    if not NAUTILUS_AVAILABLE:
        raise RuntimeError("NautilusTrader is not installed")

    aggregation = BAR_AGGREGATION_MAP.get(timeframe)
    if aggregation is None:
        raise ValueError(f"Invalid timeframe: {timeframe}")

    instrument_id = InstrumentId(Symbol(symbol), Venue(venue))

    return BarType(
        instrument_id=instrument_id,
        bar_spec=BarSpecification(1, aggregation, PriceType.LAST),
        aggregation_source=AggregationSource.EXTERNAL,
    )


def convert_to_nautilus_bars(
    df: pd.DataFrame,
    bar_type: Any,
    instrument: Any
) -> List[Any]:
    """
    Convert pandas DataFrame to NautilusTrader Bar objects.

    Args:
        df: DataFrame with columns: timestamp, open, high, low, close, volume
        bar_type: NautilusTrader BarType
        instrument: NautilusTrader Equity instrument

    Returns:
        List of NautilusTrader Bar objects

    Raises:
        RuntimeError: If NautilusTrader is not installed
    """
    if not NAUTILUS_AVAILABLE:
        raise RuntimeError("NautilusTrader is not installed")

    # Prepare DataFrame for wrangler
    # BarDataWrangler expects: timestamp index, columns: open, high, low, close, volume
    df_copy = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
    df_copy.set_index('timestamp', inplace=True)

    # Determine ts_init_delta based on bar_type
    # Extract timeframe from bar_spec
    bar_spec = bar_type.bar_spec
    if bar_spec.aggregation == BarAggregation.MINUTE:
        ts_init_delta = TS_INIT_DELTAS['1min']
    elif bar_spec.aggregation == BarAggregation.HOUR:
        ts_init_delta = TS_INIT_DELTAS['1hour']
    elif bar_spec.aggregation == BarAggregation.DAY:
        ts_init_delta = TS_INIT_DELTAS['1day']
    else:
        ts_init_delta = TS_INIT_DELTAS['1min']  # Default

    # Use BarDataWrangler to convert
    wrangler = BarDataWrangler(bar_type=bar_type, instrument=instrument)
    bars = wrangler.process(df_copy, ts_init_delta=ts_init_delta)

    return bars


def homeguard_to_nautilus_catalog(
    config: DataConversionConfig
) -> Any:
    """
    Convert Homeguard parquet data to NautilusTrader ParquetDataCatalog.

    This is the main conversion function. It:
    1. Creates the catalog directory
    2. Creates instruments for each symbol
    3. Loads Homeguard data and converts to Nautilus bars
    4. Writes all data to the catalog

    Args:
        config: DataConversionConfig specifying source, symbols, timeframes

    Returns:
        ParquetDataCatalog instance ready for backtesting

    Raises:
        RuntimeError: If NautilusTrader is not installed
    """
    if not NAUTILUS_AVAILABLE:
        raise RuntimeError("NautilusTrader is not installed")

    logger.info(f"Converting Homeguard data to NautilusTrader catalog")
    logger.info(f"  Source: {config.source_dir}")
    logger.info(f"  Catalog: {config.catalog_path}")
    logger.info(f"  Symbols: {len(config.symbols)}")
    logger.info(f"  Timeframes: {config.timeframes}")

    # Create catalog directory
    config.catalog_path.mkdir(parents=True, exist_ok=True)

    # Initialize catalog
    catalog = ParquetDataCatalog(str(config.catalog_path))

    # Track conversion stats
    total_bars = 0
    converted_symbols = 0
    failed_symbols = []

    # Process each symbol
    all_instruments = []
    all_bars = []

    for i, symbol in enumerate(config.symbols):
        try:
            # Create instrument
            instrument = create_equity_instrument(symbol, config.venue)
            all_instruments.append(instrument)

            # Process each timeframe
            for timeframe in config.timeframes:
                try:
                    # Load Homeguard data
                    df = load_homeguard_bars(
                        config.source_dir,
                        symbol,
                        timeframe,
                        config.start_date,
                        config.end_date
                    )

                    if df.empty:
                        logger.warning(f"No data for {symbol} {timeframe}")
                        continue

                    # Create bar type
                    bar_type = create_bar_type(symbol, timeframe, config.venue)

                    # Convert to Nautilus bars
                    bars = convert_to_nautilus_bars(df, bar_type, instrument)
                    all_bars.extend(bars)
                    total_bars += len(bars)

                except FileNotFoundError:
                    logger.debug(f"No {timeframe} data for {symbol}")
                except Exception as e:
                    logger.warning(f"Error converting {symbol} {timeframe}: {e}")

            converted_symbols += 1

            # Progress logging
            if (i + 1) % 50 == 0:
                logger.info(f"Processed {i + 1}/{len(config.symbols)} symbols...")

        except Exception as e:
            logger.error(f"Failed to process {symbol}: {e}")
            failed_symbols.append(symbol)

    # Write instruments to catalog
    if all_instruments:
        logger.info(f"Writing {len(all_instruments)} instruments to catalog...")
        catalog.write_data(all_instruments)

    # Write bars to catalog
    if all_bars:
        logger.info(f"Writing {total_bars:,} bars to catalog...")
        catalog.write_data(all_bars)

    # Summary
    logger.info("=" * 60)
    logger.info("Conversion complete:")
    logger.info(f"  Symbols converted: {converted_symbols}/{len(config.symbols)}")
    logger.info(f"  Total bars: {total_bars:,}")
    logger.info(f"  Catalog path: {config.catalog_path}")
    if failed_symbols:
        logger.warning(f"  Failed symbols: {len(failed_symbols)}")

    return catalog


def create_backtest_data_configs(
    catalog: Any,
    symbols: List[str],
    start: str,
    end: str,
    timeframe: str = '1min',
    venue: str = "XNAS"
) -> List[Any]:
    """
    Create BacktestDataConfig objects for NautilusTrader BacktestNode.

    Args:
        catalog: ParquetDataCatalog instance
        symbols: List of symbols to include
        start: Start date (YYYY-MM-DD)
        end: End date (YYYY-MM-DD)
        timeframe: Bar timeframe (default: '1min')
        venue: Trading venue (default: "XNAS")

    Returns:
        List of BacktestDataConfig objects

    Raises:
        RuntimeError: If NautilusTrader is not installed
    """
    if not NAUTILUS_AVAILABLE:
        raise RuntimeError("NautilusTrader is not installed")

    from nautilus_trader.config import BacktestDataConfig

    configs = []
    for symbol in symbols:
        instrument_id = f"{symbol}.{venue}"
        bar_type = f"{instrument_id}-1-{timeframe.upper()}-LAST-EXTERNAL"

        config = BacktestDataConfig(
            catalog_path=str(catalog.path),
            data_cls="Bar",
            instrument_id=instrument_id,
            bar_type=bar_type,
            start_time=start,
            end_time=end,
        )
        configs.append(config)

    return configs


# Utility function for quick conversion
def convert_symbols_to_catalog(
    symbols: List[str],
    source_dir: Optional[Path] = None,
    catalog_path: Optional[Path] = None,
    timeframes: Optional[List[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Any:
    """
    Convenience function to convert symbols to NautilusTrader catalog.

    Args:
        symbols: List of stock symbols
        source_dir: Source data directory (default: from settings)
        catalog_path: Output catalog path (default: source_dir/nautilus_catalog)
        timeframes: Timeframes to convert (default: ['1min', '1day'])
        start_date: Optional start date filter
        end_date: Optional end date filter

    Returns:
        ParquetDataCatalog instance
    """
    # Get source_dir from settings if not provided
    if source_dir is None:
        from src.settings import get_local_storage_dir
        source_dir = get_local_storage_dir()

    config = DataConversionConfig(
        source_dir=source_dir,
        symbols=symbols,
        catalog_path=catalog_path,
        start_date=start_date,
        end_date=end_date,
        timeframes=timeframes or ['1min', '1day'],
    )

    return homeguard_to_nautilus_catalog(config)
