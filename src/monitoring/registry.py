"""
Thread-safe metrics registry for Prometheus-compatible metric storage.

Supports gauges, counters, and histograms with arbitrary label sets.
All operations are protected by a single lock for simplicity --
contention is negligible at 15-second scrape intervals.
"""

import threading
import time
from typing import Dict, Optional, Any


# Label set key: frozenset of (key, value) pairs for hashability
LabelKey = Optional[frozenset]


def _label_key(labels: Optional[Dict[str, str]] = None) -> LabelKey:
    """Convert label dict to hashable key."""
    if not labels:
        return None
    return frozenset(labels.items())


def _format_labels(labels: Optional[Dict[str, str]] = None) -> str:
    """Format labels for Prometheus text exposition."""
    if not labels:
        return ''
    pairs = ','.join(f'{k}="{v}"' for k, v in sorted(labels.items()))
    return '{' + pairs + '}'


class MetricsRegistry:
    """
    Thread-safe in-memory metrics registry.

    Stores gauges, counters, and histograms keyed by (metric_name, label_set).
    Exports to Prometheus text exposition format for VictoriaMetrics scraping.

    Args:
        strategy: Strategy name (omr, ramp, mp, cscm). Used as default
                  label value and for process-level metrics.
    """

    def __init__(self, strategy: str):
        self.strategy = strategy
        self._lock = threading.Lock()
        self._gauges: Dict[str, Dict[LabelKey, float]] = {}
        self._counters: Dict[str, Dict[LabelKey, int]] = {}
        self._histograms: Dict[str, Dict[LabelKey, Dict[str, Any]]] = {}
        self._created_at = time.time()

    # ---- Gauges ----

    def set_gauge(self, name: str, value: float,
                  labels: Optional[Dict[str, str]] = None) -> None:
        """Set a gauge to an absolute value."""
        key = _label_key(labels)
        with self._lock:
            if name not in self._gauges:
                self._gauges[name] = {}
            self._gauges[name][key] = value

    def get_gauge(self, name: str,
                  labels: Optional[Dict[str, str]] = None) -> Optional[float]:
        """Get current gauge value, or None if not set."""
        key = _label_key(labels)
        with self._lock:
            return self._gauges.get(name, {}).get(key)

    def remove_gauge(self, name: str,
                     labels: Optional[Dict[str, str]] = None) -> None:
        """Remove a gauge label set (e.g., when a position closes)."""
        key = _label_key(labels)
        with self._lock:
            if name in self._gauges:
                self._gauges[name].pop(key, None)

    # ---- Counters ----

    def inc_counter(self, name: str,
                    labels: Optional[Dict[str, str]] = None,
                    amount: int = 1) -> None:
        """Increment a counter."""
        key = _label_key(labels)
        with self._lock:
            if name not in self._counters:
                self._counters[name] = {}
            self._counters[name][key] = self._counters[name].get(key, 0) + amount

    def get_counter(self, name: str,
                    labels: Optional[Dict[str, str]] = None) -> int:
        """Get current counter value (0 if never incremented)."""
        key = _label_key(labels)
        with self._lock:
            return self._counters.get(name, {}).get(key, 0)

    # ---- Histograms ----

    def observe_histogram(self, name: str, value: float,
                          labels: Optional[Dict[str, str]] = None) -> None:
        """Record an observation in a histogram."""
        key = _label_key(labels)
        with self._lock:
            if name not in self._histograms:
                self._histograms[name] = {}
            if key not in self._histograms[name]:
                self._histograms[name][key] = {'count': 0, 'sum': 0.0}
            self._histograms[name][key]['count'] += 1
            self._histograms[name][key]['sum'] += value

    def get_histogram(self, name: str,
                      labels: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Get histogram count and sum."""
        key = _label_key(labels)
        with self._lock:
            return self._histograms.get(name, {}).get(
                key, {'count': 0, 'sum': 0.0}
            ).copy()

    # ---- Export ----

    def prometheus_format(self) -> bytes:
        """
        Export all metrics in Prometheus text exposition format.

        Returns UTF-8 encoded bytes suitable for HTTP response body.
        """
        lines = []
        with self._lock:
            # Gauges
            for name, label_values in self._gauges.items():
                for lk, value in label_values.items():
                    labels_str = _format_labels(dict(lk) if lk else None)
                    lines.append(f'{name}{labels_str} {value}')

            # Counters
            for name, label_values in self._counters.items():
                for lk, value in label_values.items():
                    labels_str = _format_labels(dict(lk) if lk else None)
                    lines.append(f'{name}{labels_str} {value}')

            # Histograms (count + sum)
            for name, label_values in self._histograms.items():
                for lk, hist in label_values.items():
                    labels_str = _format_labels(dict(lk) if lk else None)
                    lines.append(f'{name}_count{labels_str} {hist["count"]}')
                    lines.append(f'{name}_sum{labels_str} {hist["sum"]}')

        lines.append('')  # trailing newline
        return '\n'.join(lines).encode('utf-8')

    def snapshot(self) -> dict:
        """
        Return a JSON-serializable snapshot of all current metric values.

        Used by snapshot.py for offline fallback.
        """
        with self._lock:
            return {
                'strategy': self.strategy,
                'timestamp': time.time(),
                'gauges': {
                    name: {
                        str(dict(lk) if lk else '{}'): v
                        for lk, v in label_values.items()
                    }
                    for name, label_values in self._gauges.items()
                },
                'counters': {
                    name: {
                        str(dict(lk) if lk else '{}'): v
                        for lk, v in label_values.items()
                    }
                    for name, label_values in self._counters.items()
                },
                'histograms': {
                    name: {
                        str(dict(lk) if lk else '{}'): h.copy()
                        for lk, h in label_values.items()
                    }
                    for name, label_values in self._histograms.items()
                },
            }
