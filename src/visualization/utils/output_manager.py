"""
Output directory management for visualization engine.
"""

from pathlib import Path
from datetime import datetime
from typing import Optional, List
from src.utils import logger as main_logger


class OutputManager:
    """
    Manages output directory structure for backtest visualizations and logs.

    Creates human-readable timestamped folders with strategy names.
    """

    def __init__(self, base_dir: Path):
        """
        Initialize output manager.

        Args:
            base_dir: Base directory for all output
        """
        self.base_dir = Path(base_dir)
        self.run_dir: Optional[Path] = None

    def create_run_directory(
        self,
        strategy_name: str,
        symbols: List[str],
        timestamp: Optional[datetime] = None
    ) -> Path:
        """
        Create a timestamped directory for this backtest run.

        Format: {base_dir}/{YYYYMMDD_HHMMSS}_{strategy}_{symbols}/

        Args:
            strategy_name: Name of the strategy being tested
            symbols: List of symbols being tested
            timestamp: Optional specific timestamp (defaults to now)

        Returns:
            Path to the created run directory
        """
        if timestamp is None:
            timestamp = datetime.now()

        timestamp_str = timestamp.strftime('%Y%m%d_%H%M%S')

        symbols_str = '_'.join(symbols) if len(symbols) <= 3 else f"{len(symbols)}symbols"

        dir_name = f"{timestamp_str}_{strategy_name}_{symbols_str}"

        self.run_dir = self.base_dir / dir_name
        self.run_dir.mkdir(parents=True, exist_ok=True)

        (self.run_dir / "charts").mkdir(exist_ok=True)
        (self.run_dir / "logs").mkdir(exist_ok=True)
        (self.run_dir / "reports").mkdir(exist_ok=True)

        main_logger.success(f"Created output directory: {self.run_dir}")

        return self.run_dir

    def get_chart_path(self, filename: str) -> Path:
        """Get path for a chart file."""
        if self.run_dir is None:
            raise RuntimeError("Run directory not created. Call create_run_directory() first.")
        return self.run_dir / "charts" / filename

    def get_log_path(self, filename: str) -> Path:
        """Get path for a log file."""
        if self.run_dir is None:
            raise RuntimeError("Run directory not created. Call create_run_directory() first.")
        return self.run_dir / "logs" / filename

    def get_report_path(self, filename: str) -> Path:
        """Get path for a report file."""
        if self.run_dir is None:
            raise RuntimeError("Run directory not created. Call create_run_directory() first.")
        return self.run_dir / "reports" / filename

    def get_run_directory(self) -> Path:
        """Get the current run directory."""
        if self.run_dir is None:
            raise RuntimeError("Run directory not created. Call create_run_directory() first.")
        return self.run_dir
