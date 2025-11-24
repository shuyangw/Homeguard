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


def get_output_dir() -> Path:
    """
    Get the base output directory for the current OS environment.

    Returns:
        Path: Configured output directory
    """
    # Try new 'output_dir' key first, fall back to 'log_output_dir' for compatibility
    try:
        output_dir = settings.get(OS_ENVIRONMENT, 'output_dir')
    except:
        output_dir = settings.get(OS_ENVIRONMENT, 'log_output_dir')
    return Path(output_dir)


def get_log_output_dir() -> Path:
    """
    Get the log output directory for the current OS environment.

    DEPRECATED: Use get_output_dir(), get_backtest_results_dir(),
    get_backtest_tests_dir(), or get_live_trading_dir() instead.

    Returns:
        Path: Configured output directory (for backward compatibility)
    """
    return get_output_dir()


def get_backtest_results_dir() -> Path:
    """
    Get the directory for backtest results and reports.

    Returns:
        Path: output/backtesting/results
    """
    return get_output_dir() / "backtesting" / "results"


def get_backtest_tests_dir() -> Path:
    """
    Get the directory for backtest unit tests and debug logs.

    Returns:
        Path: output/backtesting/tests
    """
    return get_output_dir() / "backtesting" / "tests"


def get_live_trading_dir(mode: str = 'paper') -> Path:
    """
    Get the directory for live trading logs.

    Args:
        mode: 'paper' for paper trading, 'production' for live trading

    Returns:
        Path: output/live_trading/{mode}
    """
    if mode not in ['paper', 'production']:
        raise ValueError(f"Invalid mode: {mode}. Must be 'paper' or 'production'")

    return get_output_dir() / "live_trading" / mode


def get_models_dir() -> Path:
    """
    Get the directory for trained model artifacts (pickle files, etc.).

    Returns:
        Path: models/ (in project root)
    """
    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir


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
