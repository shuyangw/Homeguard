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


def start_metrics_server(registry, port: int):
    """
    Start the HTTP metrics server in a background daemon thread.

    Lazy import of server module so this package can be imported before
    Task 4 (server.py) is implemented.

    Args:
        registry: MetricsRegistry instance to serve metrics from.
        port: TCP port to listen on.

    Returns:
        threading.Thread (daemon, already started).
    """
    from src.monitoring.server import start_metrics_server as _start
    return _start(registry, port)


__all__ = ['MetricsRegistry', 'start_metrics_server']
