"""
Migrate Files to New Output Directory Structure

Reorganizes all log and output files from:
  C:/Users/qwqw1/Dropbox/cs/stonk/logs/*

To:
  C:/Users/qwqw1/Dropbox/cs/stonk/output/
  ├── backtesting/
  │   ├── results/    # Backtest results, quantstats reports, tearsheets
  │   └── tests/      # Unit test logs, test outputs
  └── live_trading/
      ├── paper/      # Paper trading logs (currently active)
      └── production/ # Live production trading (future)
"""

import sys
from pathlib import Path
import shutil
import configparser
import platform
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import logger
from src.config import get_output_dir, get_backtest_results_dir, get_backtest_tests_dir, get_live_trading_dir


def categorize_file(file_path: Path) -> str:
    """
    Determine the category for a file based on its path and name.

    Returns:
        str: Path component like 'backtesting/results', 'backtesting/tests', 'live_trading/paper'
    """
    name = file_path.name.lower()
    parent = file_path.parent.name.lower()

    # Live trading logs (already in live_trading/ subdirectory)
    if 'live_trading' in str(file_path):
        return 'live_trading/paper'

    # Backtest test files (in backtest_tests/ or contain 'test' in name)
    if parent == 'backtest_tests' or 'test' in name or 'debug' in name:
        return 'backtesting/tests'

    # Everything else is backtest results
    return 'backtesting/results'


def get_target_directory(category: str, output_base: Path) -> Path:
    """Get target directory for a given category."""
    if category == 'backtesting/results':
        return output_base / 'backtesting' / 'results'
    elif category == 'backtesting/tests':
        return output_base / 'backtesting' / 'tests'
    elif category == 'live_trading/paper':
        return output_base / 'live_trading' / 'paper'
    elif category == 'live_trading/production':
        return output_base / 'live_trading' / 'production'
    else:
        raise ValueError(f"Unknown category: {category}")


def main():
    """Migrate all files to new directory structure."""
    logger.info("=" * 80)
    logger.info("MIGRATING TO NEW OUTPUT DIRECTORY STRUCTURE")
    logger.info("=" * 80)

    # Get directories
    output_dir = get_output_dir()
    old_log_dir = output_dir.parent / "logs"  # Old location

    logger.info(f"Old location: {old_log_dir}")
    logger.info(f"New location: {output_dir}")
    logger.info("")

    # Check if old directory exists
    if not old_log_dir.exists():
        logger.info("Old logs directory does not exist - nothing to migrate")
        logger.info("Creating new directory structure...")
    else:
        logger.info(f"Found old logs directory: {old_log_dir}")

    # Create new directory structure
    get_backtest_results_dir().mkdir(parents=True, exist_ok=True)
    get_backtest_tests_dir().mkdir(parents=True, exist_ok=True)
    get_live_trading_dir('paper').mkdir(parents=True, exist_ok=True)
    get_live_trading_dir('production').mkdir(parents=True, exist_ok=True)

    logger.info("Created directory structure:")
    logger.info(f"  {get_backtest_results_dir()}")
    logger.info(f"  {get_backtest_tests_dir()}")
    logger.info(f"  {get_live_trading_dir('paper')}")
    logger.info(f"  {get_live_trading_dir('production')}")
    logger.info("")

    if not old_log_dir.exists():
        logger.info("Migration complete - new structure ready!")
        return 0

    # Find all files to migrate
    all_files = [f for f in old_log_dir.rglob("*") if f.is_file()]

    if not all_files:
        logger.info("No files to migrate")
        return 0

    logger.info(f"Found {len(all_files)} files to migrate")
    logger.info("")

    # Categorize files
    categorized = {}
    for file_path in all_files:
        category = categorize_file(file_path)
        if category not in categorized:
            categorized[category] = []
        categorized[category].append(file_path)

    # Show categorization
    logger.info("File categorization:")
    for category, files in categorized.items():
        target_dir = get_target_directory(category, output_dir)
        logger.info(f"  {category}: {len(files)} files -> {target_dir}")
    logger.info("")

    # Move files
    moved_count = 0
    skipped_count = 0

    for category, files in categorized.items():
        target_dir = get_target_directory(category, output_dir)

        logger.info(f"Migrating {len(files)} files to {category}/")

        for file_path in files:
            target_path = target_dir / file_path.name

            # Check if file already exists
            if target_path.exists():
                logger.warning(f"  File already exists, skipping: {file_path.name}")
                skipped_count += 1
                continue

            try:
                shutil.move(str(file_path), str(target_path))
                logger.info(f"  Moved: {file_path.name}")
                moved_count += 1
            except Exception as e:
                logger.error(f"  Error moving {file_path.name}: {e}")

    logger.info("")
    logger.info("=" * 80)
    logger.info("MIGRATION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Files moved:   {moved_count}")
    logger.info(f"Files skipped: {skipped_count}")
    logger.info(f"Total files:   {len(all_files)}")
    logger.info("")

    # Check if old directory is empty
    remaining_files = [f for f in old_log_dir.rglob("*") if f.is_file()]
    if not remaining_files:
        logger.info("Old logs directory is empty")
        logger.info(f"Safe to delete: {old_log_dir}")
        logger.info("")
        logger.info("New directory structure:")
        logger.info(f"  {output_dir}/")
        logger.info("  ├── backtesting/")
        logger.info("  │   ├── results/    # Backtest results, quantstats reports")
        logger.info("  │   └── tests/      # Unit test logs, debug output")
        logger.info("  └── live_trading/")
        logger.info("      ├── paper/      # Paper trading logs")
        logger.info("      └── production/ # Live production trading")
    else:
        logger.warning(f"Old directory still contains {len(remaining_files)} files")
        logger.info("Remaining files:")
        for f in remaining_files[:10]:
            logger.info(f"  - {f.relative_to(old_log_dir)}")
        if len(remaining_files) > 10:
            logger.info(f"  ... and {len(remaining_files) - 10} more")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
