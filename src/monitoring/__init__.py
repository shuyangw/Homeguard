"""
Homeguard Monitoring - In-process metrics exporter.

Provides per-strategy Prometheus-compatible metrics via a background HTTP thread.
Each strategy process runs its own MetricsRegistry + server on a unique port.

Usage:
    from src.monitoring import MetricsRegistry, start_metrics_server

    registry = MetricsRegistry(strategy='omr')
    thread = start_metrics_server(registry, port=8081)
"""

from src.monitoring.registry import MetricsRegistry
from src.monitoring.server import start_metrics_server
from src.monitoring.snapshot import SnapshotWriter, read_snapshot

__all__ = [
    'MetricsRegistry',
    'start_metrics_server',
    'SnapshotWriter',
    'read_snapshot',
]
