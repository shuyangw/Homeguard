"""
Numba JIT-compiled portfolio simulation v2 with state tracking.

Extends the original numba_sim with bar-by-bar state arrays for:
- Cash balance at each bar
- Position size at each bar
- Position entry price at each bar

These arrays enable post-simulation portfolio state analysis without
impacting simulation performance.
"""

import numba
import numpy as np

# Trade type constants
TRADE_ENTRY = np.int8(0)
TRADE_EXIT = np.int8(1)
TRADE_SHORT_ENTRY = np.int8(2)
TRADE_COVER_SHORT = np.int8(3)

# Exit reason constants
EXIT_SIGNAL = np.int8(0)
EXIT_STOP_LOSS = np.int8(1)
EXIT_TIME_STOP = np.int8(2)
EXIT_PROFIT_TARGET = np.int8(3)

# Trade type names for conversion
TRADE_TYPE_NAMES = {
    TRADE_ENTRY: "entry",
    TRADE_EXIT: "exit",
    TRADE_SHORT_ENTRY: "short_entry",
    TRADE_COVER_SHORT: "cover_short",
}

# Exit reason names for conversion
EXIT_REASON_NAMES = {
    EXIT_SIGNAL: "strategy_signal",
    EXIT_STOP_LOSS: "stop_loss",
    EXIT_TIME_STOP: "time_stop",
    EXIT_PROFIT_TARGET: "profit_target",
}

NUMBA_V2_AVAILABLE = True


@numba.jit(nopython=True)
def simulate_portfolio_numba_v2(
    prices: np.ndarray,
    entries: np.ndarray,
    exits: np.ndarray,
    market_hours: np.ndarray,
    init_cash: float,
    fees: float,
    slippage: float,
    position_size_pct: float,
    use_stop_loss: bool,
    stop_loss_pct: float,
    use_profit_target: bool,
    profit_target_pct: float,
    use_time_stop: bool,
    max_bars_in_position: int,
    allow_shorts: bool,
    fractional_shares: bool = False,
    max_trades: int = 10000,
) -> tuple:
    """
    JIT-compiled portfolio simulation v2 with state tracking.

    This extends the original simulate_portfolio_numba with additional
    state arrays that record cash, position, and entry price at each bar.

    Args:
        prices: Close price array
        entries: Boolean array of entry signals
        exits: Boolean array of exit signals
        market_hours: Boolean array indicating valid trading hours
        init_cash: Initial capital
        fees: Trading fees as decimal
        slippage: Slippage as decimal
        position_size_pct: Position size as fraction of portfolio
        use_stop_loss: Whether to use percentage stop loss
        stop_loss_pct: Stop loss percentage as decimal
        use_profit_target: Whether to use profit target
        profit_target_pct: Profit target percentage as decimal
        use_time_stop: Whether to use time-based stop
        max_bars_in_position: Maximum bars before forced exit
        allow_shorts: Whether short selling is enabled
        fractional_shares: If True, allow fractional share quantities
        max_trades: Maximum number of trades to track

    Returns:
        Tuple containing:
        - equity: np.ndarray - equity curve
        - trade_bars, trade_types, trade_prices, trade_shares: trade arrays
        - trade_pnls, trade_pnl_pcts, trade_exit_reasons: P&L arrays
        - trade_costs, trade_proceeds: cost/proceeds arrays
        - trade_count: total trades
        - state_cash: np.ndarray - cash at each bar
        - state_position: np.ndarray - position at each bar
        - state_position_price: np.ndarray - entry price at each bar
    """
    n = len(prices)
    equity = np.empty(n, dtype=np.float64)

    # Pre-allocate trade arrays
    trade_bars = np.empty(max_trades, dtype=np.int64)
    trade_types = np.empty(max_trades, dtype=np.int8)
    trade_prices = np.empty(max_trades, dtype=np.float64)
    trade_shares = np.empty(max_trades, dtype=np.float64)
    trade_pnls = np.empty(max_trades, dtype=np.float64)
    trade_pnl_pcts = np.empty(max_trades, dtype=np.float64)
    trade_exit_reasons = np.empty(max_trades, dtype=np.int8)
    trade_costs = np.empty(max_trades, dtype=np.float64)
    trade_proceeds = np.empty(max_trades, dtype=np.float64)

    # V2: State tracking arrays
    state_cash = np.empty(n, dtype=np.float64)
    state_position = np.empty(n, dtype=np.float64)
    state_position_price = np.empty(n, dtype=np.float64)

    # Position state
    cash = init_cash
    position = 0.0
    position_price = 0.0
    bars_in_position = 0
    trade_idx = 0

    for i in range(n):
        price = prices[i]

        # Calculate current portfolio value
        if position > 0:
            portfolio_value = cash + position * price
        elif position < 0:
            short_pnl = (position_price - price) * abs(position)
            portfolio_value = cash + short_pnl
        else:
            portfolio_value = cash

        # Record state at start of bar (before any trades)
        state_cash[i] = cash
        state_position[i] = position
        state_position_price[i] = position_price

        # Track equity even outside market hours
        if not market_hours[i]:
            equity[i] = portfolio_value
            continue

        # Track time in position (only during market hours)
        if position != 0:
            bars_in_position += 1

        # === RISK MANAGEMENT CHECKS ===
        exit_triggered = False
        exit_reason = np.int8(0)

        if position > 0:
            pnl_pct = (price - position_price) / position_price

            if use_stop_loss and pnl_pct <= -stop_loss_pct:
                exit_triggered = True
                exit_reason = np.int8(1)
            elif use_profit_target and pnl_pct >= profit_target_pct:
                exit_triggered = True
                exit_reason = np.int8(3)
            elif use_time_stop and bars_in_position >= max_bars_in_position:
                exit_triggered = True
                exit_reason = np.int8(2)

        elif position < 0:
            pnl_pct = (position_price - price) / position_price

            if use_stop_loss and pnl_pct <= -stop_loss_pct:
                exit_triggered = True
                exit_reason = np.int8(1)
            elif use_profit_target and pnl_pct >= profit_target_pct:
                exit_triggered = True
                exit_reason = np.int8(3)
            elif use_time_stop and bars_in_position >= max_bars_in_position:
                exit_triggered = True
                exit_reason = np.int8(2)

        # === EXECUTE RISK EXIT ===
        if exit_triggered and trade_idx < max_trades:
            if position > 0:
                slippage_adj = price * (1 - slippage)
                proceeds = position * slippage_adj
                fee = proceeds * fees
                net_proceeds = proceeds - fee

                pnl = net_proceeds - (position * position_price)
                pnl_pct_val = (pnl / (position * position_price)) * 100

                trade_bars[trade_idx] = i
                trade_types[trade_idx] = np.int8(1)
                trade_prices[trade_idx] = price
                trade_shares[trade_idx] = position
                trade_pnls[trade_idx] = pnl
                trade_pnl_pcts[trade_idx] = pnl_pct_val
                trade_exit_reasons[trade_idx] = exit_reason
                trade_proceeds[trade_idx] = net_proceeds
                trade_costs[trade_idx] = 0.0
                trade_idx += 1

                cash += net_proceeds
                position = 0.0
                position_price = 0.0
                bars_in_position = 0

            else:
                slippage_adj = price * (1 + slippage)
                cost_to_cover = abs(position) * slippage_adj
                fee = cost_to_cover * fees
                total_cost = cost_to_cover + fee

                proceeds_from_short = abs(position) * position_price
                pnl = proceeds_from_short - cost_to_cover - fee
                pnl_pct_val = (pnl / proceeds_from_short) * 100

                trade_bars[trade_idx] = i
                trade_types[trade_idx] = np.int8(3)
                trade_prices[trade_idx] = price
                trade_shares[trade_idx] = abs(position)
                trade_pnls[trade_idx] = pnl
                trade_pnl_pcts[trade_idx] = pnl_pct_val
                trade_exit_reasons[trade_idx] = exit_reason
                trade_costs[trade_idx] = total_cost
                trade_proceeds[trade_idx] = 0.0
                trade_idx += 1

                cash -= total_cost
                position = 0.0
                position_price = 0.0
                bars_in_position = 0

        # === ENTRY SIGNAL: want to go LONG ===
        if entries[i]:
            if position < 0 and not exit_triggered and trade_idx < max_trades:
                slippage_adj = price * (1 + slippage)
                cost_to_cover = abs(position) * slippage_adj
                fee = cost_to_cover * fees
                total_cost = cost_to_cover + fee

                proceeds_from_short = abs(position) * position_price
                pnl = proceeds_from_short - cost_to_cover - fee
                pnl_pct_val = (pnl / proceeds_from_short) * 100

                trade_bars[trade_idx] = i
                trade_types[trade_idx] = np.int8(3)
                trade_prices[trade_idx] = price
                trade_shares[trade_idx] = abs(position)
                trade_pnls[trade_idx] = pnl
                trade_pnl_pcts[trade_idx] = pnl_pct_val
                trade_exit_reasons[trade_idx] = np.int8(0)
                trade_costs[trade_idx] = total_cost
                trade_proceeds[trade_idx] = 0.0
                trade_idx += 1

                cash -= total_cost
                position = 0.0
                position_price = 0.0
                bars_in_position = 0
                portfolio_value = cash

            if position == 0 and cash > 0 and trade_idx < max_trades:
                target_value = portfolio_value * position_size_pct
                if fractional_shares:
                    shares = target_value / price
                else:
                    shares = np.floor(target_value / price)

                slippage_adj = price * (1 + slippage)
                cost = shares * slippage_adj
                fee = cost * fees
                total_cost = cost + fee

                if total_cost <= cash and shares > 0:
                    trade_bars[trade_idx] = i
                    trade_types[trade_idx] = np.int8(0)
                    trade_prices[trade_idx] = price
                    trade_shares[trade_idx] = shares
                    trade_pnls[trade_idx] = 0.0
                    trade_pnl_pcts[trade_idx] = 0.0
                    trade_exit_reasons[trade_idx] = np.int8(-1)
                    trade_costs[trade_idx] = total_cost
                    trade_proceeds[trade_idx] = 0.0
                    trade_idx += 1

                    position = shares
                    position_price = price
                    cash -= total_cost
                    bars_in_position = 0

        # === EXIT SIGNAL: want to go SHORT or FLAT ===
        if exits[i] and not entries[i]:
            if position > 0 and not exit_triggered and trade_idx < max_trades:
                slippage_adj = price * (1 - slippage)
                proceeds = position * slippage_adj
                fee = proceeds * fees
                net_proceeds = proceeds - fee

                pnl = net_proceeds - (position * position_price)
                pnl_pct_val = (pnl / (position * position_price)) * 100

                trade_bars[trade_idx] = i
                trade_types[trade_idx] = np.int8(1)
                trade_prices[trade_idx] = price
                trade_shares[trade_idx] = position
                trade_pnls[trade_idx] = pnl
                trade_pnl_pcts[trade_idx] = pnl_pct_val
                trade_exit_reasons[trade_idx] = np.int8(0)
                trade_proceeds[trade_idx] = net_proceeds
                trade_costs[trade_idx] = 0.0
                trade_idx += 1

                cash += net_proceeds
                position = 0.0
                position_price = 0.0
                bars_in_position = 0
                portfolio_value = cash

            if position == 0 and allow_shorts and cash > 0 and trade_idx < max_trades:
                target_value = portfolio_value * position_size_pct
                if fractional_shares:
                    shares = target_value / price
                else:
                    shares = np.floor(target_value / price)

                slippage_adj = price * (1 - slippage)
                proceeds = shares * slippage_adj
                fee = proceeds * fees
                net_proceeds = proceeds - fee

                if shares > 0:
                    trade_bars[trade_idx] = i
                    trade_types[trade_idx] = np.int8(2)
                    trade_prices[trade_idx] = price
                    trade_shares[trade_idx] = shares
                    trade_pnls[trade_idx] = 0.0
                    trade_pnl_pcts[trade_idx] = 0.0
                    trade_exit_reasons[trade_idx] = np.int8(-1)
                    trade_proceeds[trade_idx] = net_proceeds
                    trade_costs[trade_idx] = 0.0
                    trade_idx += 1

                    position = -shares
                    position_price = price
                    cash += net_proceeds
                    bars_in_position = 0

        # === FINAL EQUITY CALCULATION ===
        if position > 0:
            equity[i] = cash + position * price
        elif position < 0:
            equity[i] = cash + (position_price - price) * abs(position)
        else:
            equity[i] = cash

    # Return sliced arrays plus state arrays
    return (
        equity,
        trade_bars[:trade_idx],
        trade_types[:trade_idx],
        trade_prices[:trade_idx],
        trade_shares[:trade_idx],
        trade_pnls[:trade_idx],
        trade_pnl_pcts[:trade_idx],
        trade_exit_reasons[:trade_idx],
        trade_costs[:trade_idx],
        trade_proceeds[:trade_idx],
        trade_idx,
        state_cash,
        state_position,
        state_position_price,
    )
