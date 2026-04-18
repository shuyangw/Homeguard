"""
JSON snapshot writer/reader for offline fallback.

Writes periodic snapshots of MetricsRegistry state to disk.
Can be read by external tools when Grafana/VictoriaMetrics are down.
"""

import json
import os
import time
import threading
from pathlib import Path
from typing import Optional, TYPE_CHECKING

from src.utils.logger import get_logger

if TYPE_CHECKING:
    from src.monitoring.registry import MetricsRegistry

logger = get_logger()


class SnapshotWriter:
    """
    Periodically writes registry state to a JSON file.

    Args:
        registry: MetricsRegistry to snapshot
        snapshot_dir: Directory to write snapshots
        interval_seconds: Seconds between snapshots (default 30)
    """

    def __init__(self, registry: 'MetricsRegistry',
                 snapshot_dir: str,
                 interval_seconds: int = 30):
        self.registry = registry
        self.snapshot_dir = Path(snapshot_dir)
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self.interval_seconds = interval_seconds
        self._file_path = self.snapshot_dir / f'{registry.strategy}_snapshot.json'

    def write_once(self) -> None:
        """Write a single snapshot to disk."""
        snapshot = self.registry.snapshot()
        tmp_path = self._file_path.with_suffix('.tmp')
        with open(tmp_path, 'w') as f:
            json.dump(snapshot, f, indent=2)
        # Atomic rename
        tmp_path.replace(self._file_path)

    def start_background(self) -> threading.Thread:
        """Start a daemon thread that writes snapshots periodically."""
        def _loop():
            while True:
                try:
                    self.write_once()
                except Exception as e:
                    logger.error(f"Snapshot write failed: {e}")
                time.sleep(self.interval_seconds)

        thread = threading.Thread(
            target=_loop,
            name=f'snapshot-{self.registry.strategy}',
            daemon=True,
        )
        thread.start()
        logger.info(
            f"Snapshot writer started: {self._file_path} "
            f"every {self.interval_seconds}s"
        )
        return thread


def read_snapshot(snapshot_dir: str,
                  max_age_seconds: float = 120.0) -> Optional[dict]:
    """
    Read the most recent snapshot from a directory.

    Args:
        snapshot_dir: Directory containing snapshot files
        max_age_seconds: Maximum age before snapshot is considered stale

    Returns:
        Parsed snapshot dict, or None if stale/missing
    """
    snapshot_dir = Path(snapshot_dir)
    if not snapshot_dir.exists():
        return None

    # Find most recent snapshot file
    snapshots = sorted(snapshot_dir.glob('*_snapshot.json'),
                       key=lambda p: p.stat().st_mtime, reverse=True)
    if not snapshots:
        return None

    latest = snapshots[0]
    age = time.time() - latest.stat().st_mtime
    if age > max_age_seconds:
        return None

    with open(latest) as f:
        return json.load(f)
