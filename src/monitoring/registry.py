"""
Thread-safe metrics registry for Prometheus-compatible metric storage.

Supports gauges, counters, and histograms with arbitrary label sets.
All operations are protected by a single lock for simplicity --
contention is negligible at 15-second scrape intervals.
"""

import json
import platform
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
                        json.dumps(dict(lk), sort_keys=True) if lk else '{}': v
                        for lk, v in label_values.items()
                    }
                    for name, label_values in self._gauges.items()
                },
                'counters': {
                    name: {
                        json.dumps(dict(lk), sort_keys=True) if lk else '{}': v
                        for lk, v in label_values.items()
                    }
                    for name, label_values in self._counters.items()
                },
                'histograms': {
                    name: {
                        json.dumps(dict(lk), sort_keys=True) if lk else '{}': h.copy()
                        for lk, h in label_values.items()
                    }
                    for name, label_values in self._histograms.items()
                },
            }

    # ---- High-level convenience methods ----
    # These are called from trading code. Each maps to set_gauge/inc_counter.

    def update_portfolio(self, equity_usd: float, cash_usd: float,
                         buying_power_usd: float, broker: str) -> None:
        """Update portfolio-level gauges."""
        b = {'broker': broker}
        self.set_gauge('hg_portfolio_equity_usd', equity_usd, b)
        self.set_gauge('hg_portfolio_cash_usd', cash_usd, b)
        self.set_gauge('hg_portfolio_buying_power_usd', buying_power_usd, b)

    def update_drawdown(self, drawdown_pct: float) -> None:
        self.set_gauge('hg_portfolio_drawdown_pct', drawdown_pct)

    def update_day_pnl(self, day_pnl_usd: float) -> None:
        self.set_gauge('hg_portfolio_day_pnl_usd', day_pnl_usd)

    def update_strategy(self, realized_pnl: float, unrealized_pnl: float,
                        positions: int, capital_allocated: float,
                        last_signal_ts: float) -> None:
        """Update strategy-level gauges."""
        s = {'strategy': self.strategy}
        self.set_gauge('hg_strategy_realized_pnl_usd', realized_pnl, s)
        self.set_gauge('hg_strategy_unrealized_pnl_usd', unrealized_pnl, s)
        self.set_gauge('hg_strategy_positions_count', float(positions), s)
        self.set_gauge('hg_strategy_capital_allocated_usd', capital_allocated, s)
        self.set_gauge('hg_strategy_last_signal_timestamp', last_signal_ts, s)

    def update_signal_missing(self, symbols_missing: int) -> None:
        self.set_gauge('hg_strategy_signal_symbols_missing', float(symbols_missing),
                       {'strategy': self.strategy})

    def update_strategy_initial_capital(self, initial_capital_usd: float) -> None:
        """Report the strategy's starting/budgeted capital (static per session)."""
        self.set_gauge('hg_strategy_initial_capital_usd', float(initial_capital_usd),
                       {'strategy': self.strategy})

    def update_strategy_equity(self, equity_usd: float) -> None:
        """Report the strategy's current equity in USD.

        The caller is responsible for computing equity as
        `initial_capital + attributed_unrealized_pnl` for the strategy's tagged
        positions. Passing broker.get_account().portfolio_value overstates equity
        when multiple strategies share a broker account (e.g. OMR + RAMP both on
        IBKR paper). See run_live_paper_trading._compute_strategy_equity.

        Drawdown is NOT emitted as a separate gauge -- it is derived from this
        gauge's history in PromQL via
            (equity - max_over_time(equity[Wd])) / max_over_time(equity[Wd])
        See the "Drawdown % by Strategy" panel in portfolio_overview.json.
        """
        self.set_gauge('hg_strategy_equity_usd', float(equity_usd),
                       {'strategy': self.strategy})

    def update_position(self, symbol: str, qty: float,
                        unrealized_pnl: float) -> None:
        """Update per-position gauges."""
        labels = {'symbol': symbol, 'strategy': self.strategy}
        self.set_gauge('hg_position_qty', qty, labels)
        self.set_gauge('hg_position_unrealized_pnl_usd', unrealized_pnl, labels)

    def close_position(self, symbol: str) -> None:
        """Remove position metrics when a position is closed."""
        labels = {'symbol': symbol, 'strategy': self.strategy}
        self.remove_gauge('hg_position_qty', labels)
        self.remove_gauge('hg_position_unrealized_pnl_usd', labels)

    def update_regime(self, state_code: int, sma_20: float, sma_50: float,
                      sma_200: float, time_in_state_seconds: float) -> None:
        """Update regime detection gauges."""
        self.set_gauge('hg_regime_state_code', float(state_code))
        self.set_gauge('hg_regime_sma_signal', sma_20, {'period': '20'})
        self.set_gauge('hg_regime_sma_signal', sma_50, {'period': '50'})
        self.set_gauge('hg_regime_sma_signal', sma_200, {'period': '200'})
        self.set_gauge('hg_regime_time_in_state_seconds', time_in_state_seconds)

    def record_order_submitted(self, side: str, broker: str) -> None:
        self.inc_counter('hg_orders_submitted_total',
                         {'strategy': self.strategy, 'side': side, 'broker': broker})

    def record_order_filled(self, broker: str, slippage_bps: float) -> None:
        self.inc_counter('hg_orders_filled_total',
                         {'strategy': self.strategy, 'broker': broker})
        self.observe_histogram('hg_fill_slippage_bps', slippage_bps,
                               {'strategy': self.strategy})

    def record_order_rejected(self, reason: str, broker: str) -> None:
        self.inc_counter('hg_orders_rejected_total',
                         {'strategy': self.strategy, 'reason': reason, 'broker': broker})

    def inc_rebalance_error(self, phase: str = 'other') -> None:
        """Record a rebalance step failure.

        Phase values: 'buy', 'sell', 'close', 'reconcile', 'other'. Useful
        signal that a rebalance attempt fired but individual orders failed --
        e.g. the 2026-04-24 incident where RAMP called a missing broker
        method and 12 consecutive buy orders silently errored out.
        """
        self.inc_counter('hg_strategy_rebalance_errors_total',
                         {'strategy': self.strategy, 'phase': phase})

    def update_broker_heartbeat(self, broker: str) -> None:
        """Record successful broker API call by stamping the current Unix time.
        Consumers compute age in PromQL via `time() - hg_broker_last_heartbeat_timestamp`.
        """
        self.set_gauge('hg_broker_last_heartbeat_timestamp', time.time(),
                       {'broker': broker})

    def update_websocket(self, provider: str, connected: bool,
                         symbols: int) -> None:
        self.set_gauge('hg_websocket_connected',
                       1.0 if connected else 0.0, {'provider': provider})
        self.set_gauge('hg_websocket_symbols_subscribed',
                       float(symbols), {'provider': provider})

    def update_market_open(self, is_open: bool) -> None:
        self.set_gauge('hg_market_open', 1.0 if is_open else 0.0)

    def update_process_rss(self) -> None:
        """Update process RSS gauge using platform-appropriate method."""
        try:
            if platform.system() == 'Linux':
                with open('/proc/self/status') as f:
                    for line in f:
                        if line.startswith('VmRSS:'):
                            rss_kb = int(line.split()[1])
                            self.set_gauge('hg_process_rss_bytes',
                                           float(rss_kb * 1024),
                                           {'strategy': self.strategy})
                            return
            # Fallback: use psutil if available (works on Windows/macOS/Linux)
            try:
                import psutil
                rss_bytes = psutil.Process().memory_info().rss
                self.set_gauge('hg_process_rss_bytes', float(rss_bytes),
                               {'strategy': self.strategy})
                return
            except ImportError:
                pass
            # Last resort: resource.getrusage (Unix only)
            try:
                import resource
                rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                # macOS returns bytes, Linux returns KB
                if platform.system() == 'Darwin':
                    rss_bytes = rss
                else:
                    rss_bytes = rss * 1024
                self.set_gauge('hg_process_rss_bytes', float(rss_bytes),
                               {'strategy': self.strategy})
            except (ImportError, AttributeError):
                pass
        except (OSError, IOError, ValueError):
            pass  # Non-critical metric: /proc parse or unexpected I/O

    def update_ramp_cache(self, age_seconds: float, hit: bool) -> None:
        self.set_gauge('hg_ramp_cache_age_seconds', age_seconds)
        if hit:
            self.inc_counter('hg_ramp_cache_hit_total')
