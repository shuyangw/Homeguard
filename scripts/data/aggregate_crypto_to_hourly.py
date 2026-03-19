"""Aggregate crypto minute data to hourly data."""
import pandas as pd
from pathlib import Path
from src.settings import get_local_storage_dir
from src.utils import logger
import numpy as np

def aggregate_to_hourly(minute_df):
    """Aggregate OHLCV minute data to hourly."""
    if minute_df.empty:
        return pd.DataFrame()
    
    # Set timestamp as index for resampling
    minute_df = minute_df.set_index('timestamp')
    
    # Resample to hourly
    hourly = minute_df.resample('1H').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
        'trade_count': 'sum',
        'vwap': 'mean',  # or could use volume-weighted mean
    })
    
    # Remove rows where all OHLC are NaN
    hourly = hourly.dropna(subset=['close'])
    
    # Reset timestamp index to column
    hourly = hourly.reset_index()
    
    return hourly

def create_hourly_data():
    """Create hourly crypto data from minute data."""
    storage_dir = get_local_storage_dir()
    
    minute_dir = Path(storage_dir) / 'crypto_1min'
    hourly_dir = Path(storage_dir) / 'crypto_1hour'
    
    # Create hourly directory
    hourly_dir.mkdir(exist_ok=True)
    
    # Process each symbol
    for symbol_dir in minute_dir.glob('symbol=*'):
        symbol_name = symbol_dir.name.split('=')[1]
        logger.info(f"Processing {symbol_name}...")
        
        # Create symbol directory in hourly
        hourly_symbol_dir = hourly_dir / symbol_dir.name
        hourly_symbol_dir.mkdir(exist_ok=True)
        
        # Collect all minute data for symbol across all dates
        all_data = []
        for parquet_file in sorted(symbol_dir.glob('year=*/month=*/data.parquet')):
            logger.debug(f"  Reading {parquet_file}")
            df = pd.read_parquet(str(parquet_file))
            all_data.append(df)
        
        if not all_data:
            logger.warning(f"  No data found for {symbol_name}")
            continue
        
        # Combine all minute data
        combined = pd.concat(all_data, ignore_index=True)
        logger.info(f"  {symbol_name}: Loaded {len(combined):,} minute bars")
        
        # Sort by timestamp
        combined = combined.sort_values('timestamp').reset_index(drop=True)
        
        # Aggregate to hourly
        hourly = aggregate_to_hourly(combined)
        logger.info(f"  {symbol_name}: Aggregated to {len(hourly):,} hourly bars")
        
        # Group by year and month and save
        hourly['year'] = hourly['timestamp'].dt.year
        hourly['month'] = hourly['timestamp'].dt.month
        
        for (year, month), group in hourly.groupby(['year', 'month']):
            year_month_dir = hourly_symbol_dir / f'year={year}' / f'month={month:02d}'
            year_month_dir.mkdir(parents=True, exist_ok=True)
            
            parquet_path = year_month_dir / 'data.parquet'
            
            # Drop year/month columns before saving
            save_data = group.drop(['year', 'month'], axis=1)
            save_data.to_parquet(str(parquet_path), index=False)
            logger.debug(f"    Saved {len(save_data)} bars to {parquet_path}")
        
        logger.info(f"  {symbol_name}: Complete")

if __name__ == '__main__':
    logger.info("Starting crypto data aggregation to hourly...")
    create_hourly_data()
    logger.info("Aggregation complete!")
