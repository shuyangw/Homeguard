"""
Fix corrupted ETF data files with RangeIndex instead of DatetimeIndex.

9 symbols need repair: USD, UYG, DFEN, WEBL, UCO, NAIL, ERX, RETL, CUT
"""

import os
import sys
from pathlib import Path

from utils.path_setup import setup_project_paths
ROOT_DIR = setup_project_paths()


import pandas as pd
from src.utils.logger import logger

DATA_DIR = Path('data/leveraged_etfs')

# Symbols that need fixing (identified from walk-forward validation)
CORRUPTED_SYMBOLS = [
    'USD', 'UYG', 'DFEN', 'WEBL', 'UCO',
    'NAIL', 'ERX', 'RETL', 'CUT'
]


def diagnose_file(symbol):
    """Check what's wrong with a parquet file."""
    file_path = DATA_DIR / f'{symbol}_1d.parquet'

    if not file_path.exists():
        return {'exists': False, 'error': 'File not found'}

    try:
        df = pd.read_parquet(file_path)

        return {
            'exists': True,
            'index_type': type(df.index).__name__,
            'is_datetime': isinstance(df.index, pd.DatetimeIndex),
            'columns': df.columns.tolist(),
            'is_multiindex_cols': isinstance(df.columns, pd.MultiIndex),
            'shape': df.shape,
            'first_rows': df.head(3).to_dict() if len(df) > 0 else None
        }
    except Exception as e:
        return {'exists': True, 'error': str(e)}


def fix_corrupted_file(symbol):
    """Fix a corrupted parquet file."""
    file_path = DATA_DIR / f'{symbol}_1d.parquet'

    logger.info(f"\nFixing {symbol}...")

    try:
        # Load data
        df = pd.read_parquet(file_path)
        logger.info(f"  Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        logger.info(f"  Current index type: {type(df.index).__name__}")
        logger.info(f"  Current columns: {df.columns.tolist()}")

        # Fix 1: Flatten MultiIndex columns if present
        if isinstance(df.columns, pd.MultiIndex):
            logger.info(f"  [FIX 1] Flattening MultiIndex columns...")
            df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
            logger.info(f"  New columns: {df.columns.tolist()}")

        # Fix 2: Set Date as index if it's a column
        if 'Date' in df.columns:
            logger.info(f"  [FIX 2] Setting 'Date' column as index...")
            df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)  # Remove timezone
            df.set_index('Date', inplace=True)
            logger.info(f"  New index type: {type(df.index).__name__}")

        # Fix 3: If index is RangeIndex, check if there's a date column with different name
        elif isinstance(df.index, pd.RangeIndex):
            # Look for any column that might be dates
            date_candidates = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]

            if date_candidates:
                date_col = date_candidates[0]
                logger.info(f"  [FIX 3] Found date column: {date_col}, setting as index...")
                df[date_col] = pd.to_datetime(df[date_col])
                # Remove timezone if present
                if hasattr(df[date_col].dt, 'tz') and df[date_col].dt.tz is not None:
                    logger.info(f"    Removing timezone info for compatibility...")
                    df[date_col] = df[date_col].dt.tz_localize(None)
                df.set_index(date_col, inplace=True)
                df.index.name = 'Date'
                logger.info(f"  New index type: {type(df.index).__name__}")
            else:
                logger.error(f"  [ERROR] No date column found! Columns: {df.columns.tolist()}")
                return False

        # Fix 4: If index is already DatetimeIndex but timezone-aware, make it naive
        if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
            logger.info(f"  [FIX 4] Removing timezone from DatetimeIndex...")
            df.index = df.index.tz_localize(None)

        # Verify we now have DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            logger.error(f"  [ERROR] Still not DatetimeIndex after fixes: {type(df.index).__name__}")
            return False

        # Ensure required columns exist
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            logger.error(f"  [ERROR] Missing required columns: {missing_cols}")
            logger.error(f"  Available columns: {df.columns.tolist()}")
            return False

        # Sort by date
        df = df.sort_index()

        # Remove duplicates
        if df.index.duplicated().any():
            logger.warning(f"  [WARNING] Removing {df.index.duplicated().sum()} duplicate dates")
            df = df[~df.index.duplicated(keep='first')]

        # Save backup
        backup_path = DATA_DIR / f'{symbol}_1d_BACKUP.parquet'
        logger.info(f"  Creating backup: {backup_path.name}")
        original_df = pd.read_parquet(file_path)
        original_df.to_parquet(backup_path)

        # Save fixed version
        logger.info(f"  Saving fixed version...")
        df.to_parquet(file_path)

        # Verify fix worked
        test_df = pd.read_parquet(file_path)
        if isinstance(test_df.index, pd.DatetimeIndex):
            logger.info(f"  [SUCCESS] {symbol} fixed! Index is now DatetimeIndex with {len(test_df)} rows")
            return True
        else:
            logger.error(f"  [ERROR] Fix failed - index is still {type(test_df.index).__name__}")
            return False

    except Exception as e:
        logger.error(f"  [ERROR] Failed to fix {symbol}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Fix all corrupted ETF data files."""

    logger.info("="*80)
    logger.info("ETF DATA REPAIR UTILITY")
    logger.info("="*80)
    logger.info(f"\nTarget directory: {DATA_DIR.absolute()}")
    logger.info(f"Symbols to fix: {len(CORRUPTED_SYMBOLS)}")

    # Step 1: Diagnose all files
    logger.info("\n" + "="*80)
    logger.info("STEP 1: DIAGNOSING FILES")
    logger.info("="*80)

    diagnoses = {}
    for symbol in CORRUPTED_SYMBOLS:
        diagnosis = diagnose_file(symbol)
        diagnoses[symbol] = diagnosis

        if not diagnosis.get('exists'):
            logger.error(f"[X] {symbol}: File not found")
        elif diagnosis.get('error'):
            logger.error(f"[X] {symbol}: {diagnosis['error']}")
        elif diagnosis.get('is_datetime'):
            logger.info(f"[OK] {symbol}: Already has DatetimeIndex (false positive?)")
        else:
            logger.warning(f"[!] {symbol}: {diagnosis['index_type']} (needs fixing)")

    # Step 2: Fix corrupted files
    logger.info("\n" + "="*80)
    logger.info("STEP 2: FIXING CORRUPTED FILES")
    logger.info("="*80)

    fixed_count = 0
    failed_count = 0
    skipped_count = 0

    for symbol in CORRUPTED_SYMBOLS:
        diagnosis = diagnoses[symbol]

        if not diagnosis.get('exists'):
            logger.warning(f"\nSkipping {symbol}: File doesn't exist")
            skipped_count += 1
            continue

        if diagnosis.get('is_datetime'):
            logger.info(f"\nSkipping {symbol}: Already fixed")
            skipped_count += 1
            continue

        # Attempt fix
        success = fix_corrupted_file(symbol)
        if success:
            fixed_count += 1
        else:
            failed_count += 1

    # Summary
    logger.info("\n" + "="*80)
    logger.info("REPAIR SUMMARY")
    logger.info("="*80)
    logger.info(f"Total symbols: {len(CORRUPTED_SYMBOLS)}")
    logger.info(f"Successfully fixed: {fixed_count}")
    logger.info(f"Failed to fix: {failed_count}")
    logger.info(f"Skipped (already OK): {skipped_count}")

    if fixed_count > 0:
        logger.info(f"\n[SUCCESS] Fixed {fixed_count} files!")
        logger.info("Backups saved with '_BACKUP' suffix")
        logger.info("\nNext step: Re-run walk-forward validation:")
        logger.info("  python backtest_scripts/overnight_walk_forward_validation.py")

    if failed_count > 0:
        logger.error(f"\n[WARNING] {failed_count} files failed to fix")
        logger.error("Manual intervention may be required")

    return fixed_count, failed_count, skipped_count


if __name__ == '__main__':
    fixed, failed, skipped = main()

    if failed > 0:
        sys.exit(1)  # Exit with error code
    else:
        sys.exit(0)  # Success
