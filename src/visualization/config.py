"""
Configuration for visualization engine.
"""

from pathlib import Path
from typing import Optional
from enum import Enum

from src.settings import settings, OS_ENVIRONMENT


class LogLevel(Enum):
    """Logging verbosity levels."""
    MINIMAL = 0    # Only summary statistics
    NORMAL = 1     # Trade entries/exits
    DETAILED = 2   # Trade details + portfolio changes
    VERBOSE = 3    # Everything including intermediate calculations


class VisualizationConfig:
    """
    Configuration for visualization and logging.
    """

    def __init__(
        self,
        log_dir: Optional[Path] = None,
        log_level: LogLevel = LogLevel.NORMAL,
        save_charts: bool = True,
        save_logs: bool = True
    ):
        """
        Initialize visualization configuration.

        Args:
            log_dir: Base directory for logs (defaults to OS-specific setting)
            log_level: Logging verbosity level
            save_charts: Whether to save chart visualizations (static PNG/SVG format)
            save_logs: Whether to save text logs
        """
        if log_dir is None:
            log_dir_str = settings[OS_ENVIRONMENT].get("log_output_dir")
            if log_dir_str:
                log_dir = Path(log_dir_str)
            else:
                # Fallback to data directory / logs
                base_dir = settings[OS_ENVIRONMENT]["local_storage_dir"]
                log_dir = Path(base_dir) / "logs"

        self.log_dir = Path(log_dir)
        self.log_level = log_level
        self.save_charts = save_charts
        self.save_logs = save_logs

    @classmethod
    def from_args(
        cls,
        verbosity: int = 1,
        enable_charts: bool = True,
        enable_logs: bool = True
    ):
        """
        Create config from command-line style arguments.

        Args:
            verbosity: 0=minimal, 1=normal, 2=detailed, 3=verbose
            enable_charts: Whether to save charts (always HTML)
            enable_logs: Whether to save logs
        """
        log_level = LogLevel(min(verbosity, 3))

        return cls(
            log_level=log_level,
            save_charts=enable_charts,
            save_logs=enable_logs
        )

    def is_enabled(self) -> bool:
        """Check if any visualization/logging is enabled."""
        return self.save_charts or self.save_logs
