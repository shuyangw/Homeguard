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
    if registry is None:
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
    if registry is None:
        return
    registry.update_market_open(is_open)


def update_process_metrics(
    registry: Optional['MetricsRegistry'],
) -> None:
    """Update process-level metrics (RSS)."""
    if registry is None:
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
    if registry is None:
        return
    registry.update_strategy(
        realized_pnl=realized_pnl,
        unrealized_pnl=unrealized_pnl,
        positions=positions,
        capital_allocated=capital_allocated,
        last_signal_ts=time.time(),
    )


def update_strategy_initial_capital(
    registry: Optional['MetricsRegistry'],
    initial_capital_usd: float,
) -> None:
    """Report the strategy's starting/budgeted capital (emit once at startup)."""
    if registry is None:
        return
    registry.update_strategy_initial_capital(initial_capital_usd)


def update_strategy_equity(
    registry: Optional['MetricsRegistry'],
    equity_usd: float,
) -> None:
    """Report the strategy's current equity (cash + positions). Emit per-tick."""
    if registry is None:
        return
    registry.update_strategy_equity(equity_usd)


def update_strategy_drawdown(
    registry: Optional['MetricsRegistry'],
    drawdown_pct: float,
) -> None:
    """Report the strategy's drawdown from peak strategy equity, in percent.

    No-op if registry is None. Drawdown should be a non-positive number;
    callers compute it as `(equity - peak) / peak * 100` and clamp at 0.
    """
    if registry is None:
        return
    registry.update_strategy_drawdown(drawdown_pct)


def inc_rebalance_error(
    registry: Optional['MetricsRegistry'],
    phase: str = 'other',
) -> None:
    """Increment rebalance-error counter. No-op when registry is None."""
    if registry is None:
        return
    registry.inc_rebalance_error(phase)


def update_position_metrics(
    registry: Optional['MetricsRegistry'],
    positions: list,
) -> None:
    """
    Update per-position gauges from a list of position dicts.

    Each position dict must have 'symbol', 'quantity', and 'unrealized_pnl'.
    """
    if registry is None:
        return
    for pos in positions:
        # Use `or 0` rather than .get(k, 0) because IBKR can return None
        # (key present, value None) for unrealized_pnl on freshly opened
        # positions before market data is subscribed for the symbol.
        registry.update_position(
            symbol=pos['symbol'],
            qty=float(pos.get('quantity') or 0),
            unrealized_pnl=float(pos.get('unrealized_pnl') or 0),
        )


def update_websocket_metrics(
    registry: Optional['MetricsRegistry'],
    provider: str,
    connected: bool,
    symbols: int,
) -> None:
    """Update WebSocket connectivity gauges."""
    if registry is None:
        return
    registry.update_websocket(provider, connected, symbols)
