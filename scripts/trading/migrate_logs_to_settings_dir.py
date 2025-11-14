"""
Migrate Existing Logs to Settings.ini Directory

Moves logs from repository to the log directory specified in settings.ini
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


def main():
    """Move logs from repository to settings.ini directory."""
    logger.info("=" * 80)
    logger.info("MIGRATING LOGS TO SETTINGS.INI DIRECTORY")
    logger.info("=" * 80)

    # Get target directory from settings
    target_base = get_log_directory_from_settings()
    if not target_base:
        logger.error("Could not determine log directory from settings.ini")
        return 1

    target_dir = target_base / "live_trading"
    source_dir = project_root / "logs" / "live_trading"

    logger.info(f"Source directory: {source_dir}")
    logger.info(f"Target directory: {target_dir}")
    logger.info("")

    # Create target directory
    target_dir.mkdir(parents=True, exist_ok=True)
    logger.success(f"Created target directory: {target_dir}")

    # Check if source directory exists
    if not source_dir.exists():
        logger.info("Source directory does not exist - no logs to migrate")
        return 0

    # Find all log files
    log_files = list(source_dir.glob("*.log"))
    csv_files = list(source_dir.glob("*.csv"))
    json_files = list(source_dir.glob("*.json"))
    md_files = list(source_dir.glob("*.md"))

    all_files = log_files + csv_files + json_files + md_files

    if not all_files:
        logger.info("No log files found to migrate")
        return 0

    logger.info(f"Found {len(all_files)} files to migrate:")
    logger.info(f"  - {len(log_files)} .log files")
    logger.info(f"  - {len(csv_files)} .csv files")
    logger.info(f"  - {len(json_files)} .json files")
    logger.info(f"  - {len(md_files)} .md files")
    logger.info("")

    # Move each file
    moved_count = 0
    for file_path in all_files:
        target_path = target_dir / file_path.name

        # Check if file already exists in target
        if target_path.exists():
            logger.warning(f"File already exists in target, skipping: {file_path.name}")
            continue

        try:
            shutil.move(str(file_path), str(target_path))
            logger.info(f"Moved: {file_path.name}")
            moved_count += 1
        except Exception as e:
            logger.error(f"Error moving {file_path.name}: {e}")

    logger.info("")
    logger.info("=" * 80)
    logger.info(f"MIGRATION COMPLETE: {moved_count}/{len(all_files)} files moved")
    logger.info("=" * 80)
    logger.info(f"Logs now in: {target_dir}")
    logger.info("")

    # Check if source directory is empty
    remaining_files = list(source_dir.glob("*"))
    if not remaining_files:
        logger.info("Source directory is empty - safe to delete")
        logger.info(f"You can manually delete: {source_dir}")
    else:
        logger.warning(f"Source directory still contains {len(remaining_files)} files")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
