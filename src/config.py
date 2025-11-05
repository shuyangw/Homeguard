import configparser
import platform
from pathlib import Path

SETTINGS_FILE = Path(__file__).resolve().parent / "../settings.ini"
PROJECT_ROOT = Path(__file__).resolve().parent.parent

def get_os_environment():
    """
    Detect the operating system environment.

    Returns:
        str: 'windows', 'macos', or 'linux'
    """
    system = platform.system().lower()
    if system == 'darwin':
        return 'macos'
    elif system == 'windows':
        return 'windows'
    elif system == 'linux':
        return 'linux'
    else:
        # Default to linux for unknown systems
        return 'linux'

def _load_settings():
    config = configparser.ConfigParser()
    if not SETTINGS_FILE.exists():
        raise FileNotFoundError(f"Settings file not found at: {SETTINGS_FILE}")

    config.read(SETTINGS_FILE)
    return config

# Load settings and detect OS
settings = _load_settings()
OS_ENVIRONMENT = get_os_environment()


def get_local_storage_dir() -> Path:
    """
    Get the local storage directory for the current OS environment.

    Returns:
        Path: Configured local storage directory
    """
    storage_dir = settings.get(OS_ENVIRONMENT, 'local_storage_dir')
    return Path(storage_dir)


def get_log_output_dir() -> Path:
    """
    Get the log output directory for the current OS environment.

    Returns:
        Path: Configured log output directory
    """
    log_dir = settings.get(OS_ENVIRONMENT, 'log_output_dir')
    return Path(log_dir)


def get_tearsheet_frequency():
    """
    Get the tearsheet frequency for data resampling.

    Returns:
        str or None: Frequency string ('D' for daily, 'H' for hourly)
                     Returns None if 'full' is specified (no resampling)

    The frequency determines how tearsheet data is resampled to reduce file size:
    - 'full': No resampling (100% accuracy, 100+ MB files)
    - 'H': Hourly resampling (99.5% accuracy, ~3 MB files, 35x smaller)
    - 'D': Daily resampling (99% accuracy, ~280 KB files, 393x smaller) - RECOMMENDED
    - 'W': Weekly resampling (95% accuracy, minimal files)
    """
    freq = settings.get('tearsheets', 'frequency', fallback='D')

    # Convert 'full' to None (no resampling)
    if freq.lower() == 'full':
        return None

    return freq
