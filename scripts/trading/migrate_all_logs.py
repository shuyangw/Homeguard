"""
Migrate All Logs to Settings.ini Directory

Categorizes and moves all log files from repository to the log directory
specified in settings.ini, organizing them by type.
"""

import sys
from pathlib import Path
import shutil
import configparser
import platform

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import logger


def get_log_directory_from_settings():
    """Read log directory from settings.ini."""
    config = configparser.ConfigParser()
    settings_file = project_root / "settings.ini"

    if not settings_file.exists():
        logger.error(f"Settings file not found: {settings_file}")
        return None

    config.read(settings_file)
    system = platform.system().lower()

    # Map system names
    if system == 'darwin':
        section = 'macos'
    elif system == 'windows':
        section = 'windows'
    else:
        section = 'linux'

    # Get log output directory from settings
    if section in config and 'log_output_dir' in config[section]:
        log_base = Path(config[section]['log_output_dir'])
        return log_base
    else:
        logger.error(f"No log_output_dir found in [{section}] section")
        return None


def categorize_file(file_path: Path) -> str:
    """
    Determine the category for a log file based on its name and content.

    Returns:
        str: Category name ('backtest_results', 'test_logs', 'live_trading', etc.)
    """
    name = file_path.name.lower()

    # Already categorized - live trading logs
    if file_path.parent.name == 'live_trading':
        return 'live_trading'

    # Test logs
    if 'test' in name or 'debug' in name:
        return 'backtest_tests'

    # Strategy-specific backtest results
    if 'bollinger' in name or 'long_short' in name or 'ma_' in name:
        return 'backtest_results'

    # Short selling related
    if 'short_selling' in name or 'shorting' in name:
        return 'backtest_tests'

    # Default to backtest_results
    return 'backtest_results'


def main():
    """Categorize and move all logs from repository to settings.ini directory."""
    logger.info("=" * 80)
    logger.info("MIGRATING ALL LOGS TO SETTINGS.INI DIRECTORY")
    logger.info("=" * 80)

    # Get target directory from settings
    target_base = get_log_directory_from_settings()
    if not target_base:
        logger.error("Could not determine log directory from settings.ini")
        return 1

    source_dir = project_root / "logs"

    logger.info(f"Source directory: {source_dir}")
    logger.info(f"Target base:      {target_base}")
    logger.info("")

    # Check if source directory exists
    if not source_dir.exists():
        logger.info("Source directory does not exist - no logs to migrate")
        return 0

    # Find all files recursively
    all_files = []
    for pattern in ['*.log', '*.csv', '*.json', '*.md', '*']:
        all_files.extend(list(source_dir.rglob(pattern)))

    # Filter to only files (not directories)
    all_files = [f for f in all_files if f.is_file()]

    if not all_files:
        logger.info("No log files found to migrate")
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
        logger.info(f"  {category}: {len(files)} files")
    logger.info("")

    # Move each category
    moved_count = 0
    skipped_count = 0

    for category, files in categorized.items():
        target_dir = target_base / category
        target_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Moving {len(files)} files to {category}/")

        for file_path in files:
            target_path = target_dir / file_path.name

            # Check if file already exists in target
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
    logger.info(f"MIGRATION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Files moved:   {moved_count}")
    logger.info(f"Files skipped: {skipped_count}")
    logger.info(f"Total files:   {len(all_files)}")
    logger.info("")

    # Check if source directory is empty
    remaining_files = [f for f in source_dir.rglob("*") if f.is_file()]
    if not remaining_files:
        logger.info("Source logs directory is empty - safe to delete")
        logger.info(f"You can manually delete: {source_dir}")
        logger.info("")
        logger.info("Suggested categories created:")
        for category in categorized.keys():
            logger.info(f"  - {target_base / category}")
    else:
        logger.warning(f"Source directory still contains {len(remaining_files)} files")
        logger.info("Remaining files:")
        for f in remaining_files[:10]:  # Show first 10
            logger.info(f"  - {f.relative_to(source_dir)}")
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
