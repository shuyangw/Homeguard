"""
Instrumentation hooks for trading code.

These functions accept an optional MetricsRegistry and are no-ops when
registry is None. This lets trading code call hooks unconditionally
without if-guards at every call site.

Usage in trading code:
    from src.monitoring.hooks import update_portfolio_metrics
    update_portfolio_metrics(registry, account, broker_name)
"""

import time
from typing import Optional, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from src.monitoring.registry import MetricsRegistry


def update_portfolio_metrics(
    registry: Optional['MetricsRegistry'],
    account: Dict,
    broker_name: str,
) -> None:
    """Update portfolio gauges from a broker account dict."""
    if not registry:
        return
    equity = float(account.get('portfolio_value', 0))
    cash = float(account.get('cash', 0))
    buying_power = float(account.get('buying_power', 0))
    registry.update_portfolio(equity, cash, buying_power, broker_name)
    registry.update_broker_heartbeat(broker_name)


def update_market_status(
    registry: Optional['MetricsRegistry'],
    is_open: bool,
) -> None:
    """Update market open/closed gauge."""
    if not registry:
        return
    registry.update_market_open(is_open)


def update_process_metrics(
    registry: Optional['MetricsRegistry'],
) -> None:
    """Update process-level metrics (RSS)."""
    if not registry:
        return
    registry.update_process_rss()


def update_strategy_metrics(
    registry: Optional['MetricsRegistry'],
    realized_pnl: float,
    unrealized_pnl: float,
    positions: int,
    capital_allocated: float,
) -> None:
    """Update strategy-level gauges after a run_once cycle."""
    if not registry:
        return
    registry.update_strategy(
        realized_pnl=realized_pnl,
        unrealized_pnl=unrealized_pnl,
        positions=positions,
        capital_allocated=capital_allocated,
        last_signal_ts=time.time(),
    )


def update_position_metrics(
    registry: Optional['MetricsRegistry'],
    positions: list,
) -> None:
    """
    Update per-position gauges from a list of position dicts.

    Each position dict must have 'symbol', 'quantity', and 'unrealized_pnl'.
    """
    if not registry:
        return
    for pos in positions:
        registry.update_position(
            symbol=pos['symbol'],
            qty=float(pos.get('quantity', 0)),
            unrealized_pnl=float(pos.get('unrealized_pnl', 0)),
        )


def update_websocket_metrics(
    registry: Optional['MetricsRegistry'],
    provider: str,
    connected: bool,
    symbols: int,
) -> None:
    """Update WebSocket connectivity gauges."""
    if not registry:
        return
    registry.update_websocket(provider, connected, symbols)
