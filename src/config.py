import configparser
import platform
from pathlib import Path

SETTINGS_FILE = Path(__file__).resolve().parent / "../settings.ini"

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
