"""
Lightweight HTTP server for Prometheus metrics exposition.

Runs on a background daemon thread. Serves two endpoints:
  /metrics - Prometheus text format (scraped by VictoriaMetrics)
  /health  - JSON liveness check

Uses stdlib http.server to avoid adding dependencies.
"""

import json
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import TYPE_CHECKING

from src.utils.logger import get_logger

if TYPE_CHECKING:
    from src.monitoring.registry import MetricsRegistry

logger = get_logger()


class _MetricsHandler(BaseHTTPRequestHandler):
    """HTTP request handler for /metrics and /health."""

    # Set by the factory function below
    registry: 'MetricsRegistry' = None  # type: ignore

    def do_GET(self):
        try:
            if self.path == '/metrics':
                body = self.registry.prometheus_format()
                self.send_response(200)
                self.send_header('Content-Type',
                                 'text/plain; version=0.0.4; charset=utf-8')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            elif self.path == '/health':
                data = {
                    'status': 'ok',
                    'strategy': self.registry.strategy,
                    'uptime_seconds': round(time.time() - self.registry._created_at, 1),
                }
                body = json.dumps(data).encode('utf-8')
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            else:
                self.send_error(404)
        except Exception as e:
            logger.error(f"[metrics-server] handler error on {self.path}: {e}")
            try:
                self.send_error(500)
            except Exception:
                pass  # Connection may already be broken

    def log_request(self, code='-', size='-'):
        """Suppress successful access logs (would flood at scrape interval)."""
        pass

    def log_error(self, format, *args):
        """Route server errors through the Homeguard logger."""
        logger.error(f"[metrics-server] {format % args}")


def start_metrics_server(
    registry: 'MetricsRegistry',
    host: str = '127.0.0.1',
    port: int = 8081,
) -> threading.Thread:
    """
    Start a background daemon thread serving metrics over HTTP.

    Args:
        registry: MetricsRegistry instance to serve
        host: Bind address (default 127.0.0.1, localhost only)
        port: Listen port (default 8081)

    Returns:
        The daemon Thread (already started).
    """
    # Create a handler class bound to this specific registry
    handler_class = type(
        '_BoundHandler',
        (_MetricsHandler,),
        {'registry': registry}
    )

    server = HTTPServer((host, port), handler_class)

    thread = threading.Thread(
        target=server.serve_forever,
        name=f'metrics-server-{port}',
        daemon=True,
    )
    thread.start()
    logger.info(f"Metrics server started on {host}:{port}")
    return thread
