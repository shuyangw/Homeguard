"""
Numba JIT-compiled portfolio simulation for high-performance backtesting.

This module provides a JIT-compiled version of the portfolio simulation loop
that achieves 10-100x speedup over pure Python while maintaining complete
feature parity with the original implementation.

Features supported:
- Long and short positions
- Slippage and fees
- Market hours filtering (via pre-computed mask)
- Fixed percentage position sizing
- Percentage stop loss
- Time-based stop
- Profit target
- Full trade logging with P&L

For features requiring dynamic data (ATR stops, volatility sizing),
the Python fallback is used automatically.
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
    TRADE_ENTRY: 'entry',
    TRADE_EXIT: 'exit',
    TRADE_SHORT_ENTRY: 'short_entry',
    TRADE_COVER_SHORT: 'cover_short'
}

# Exit reason names for conversion
EXIT_REASON_NAMES = {
    EXIT_SIGNAL: 'strategy_signal',
    EXIT_STOP_LOSS: 'stop_loss',
    EXIT_TIME_STOP: 'time_stop',
    EXIT_PROFIT_TARGET: 'profit_target'
}


@numba.jit(nopython=True)
def simulate_portfolio_numba(
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
    max_trades: int = 10000
) -> tuple:
    """
    JIT-compiled portfolio simulation with complete feature parity.

    This function simulates a trading portfolio based on entry/exit signals,
    handling long and short positions, stop losses, profit targets, and
    time-based exits.

    Args:
        prices: Close price array
        entries: Boolean array of entry signals (want to go long)
        exits: Boolean array of exit signals (want to go short or flat)
        market_hours: Boolean array indicating valid trading hours
        init_cash: Initial capital
        fees: Trading fees as decimal (e.g., 0.001 = 0.1%)
        slippage: Slippage as decimal (e.g., 0.0005 = 0.05%)
        position_size_pct: Position size as fraction of portfolio (e.g., 0.10 = 10%)
        use_stop_loss: Whether to use percentage stop loss
        stop_loss_pct: Stop loss percentage as decimal (e.g., 0.05 = 5%)
        use_profit_target: Whether to use profit target
        profit_target_pct: Profit target percentage as decimal
        use_time_stop: Whether to use time-based stop
        max_bars_in_position: Maximum bars before forced exit
        allow_shorts: Whether short selling is enabled
        max_trades: Maximum number of trades to track (pre-allocation)

    Returns:
        Tuple containing:
        - equity: np.ndarray - equity curve (length n)
        - trade_bars: np.ndarray - bar index of each trade
        - trade_types: np.ndarray - trade type codes (see TRADE_* constants)
        - trade_prices: np.ndarray - execution price for each trade
        - trade_shares: np.ndarray - shares traded
        - trade_pnls: np.ndarray - P&L for exits (0 for entries)
        - trade_pnl_pcts: np.ndarray - P&L % for exits (0 for entries)
        - trade_exit_reasons: np.ndarray - exit reason codes (-1 for entries)
        - trade_count: int - total number of trades executed

    Position encoding:
        - position > 0: Long shares held
        - position < 0: Short shares held (negative count)
        - position == 0: Flat (no position)
    """
    n = len(prices)
    equity = np.empty(n, dtype=np.float64)

    # Pre-allocate trade arrays (will slice to actual count at end)
    trade_bars = np.empty(max_trades, dtype=np.int64)
    trade_types = np.empty(max_trades, dtype=np.int8)
    trade_prices = np.empty(max_trades, dtype=np.float64)
    trade_shares = np.empty(max_trades, dtype=np.float64)
    trade_pnls = np.empty(max_trades, dtype=np.float64)
    trade_pnl_pcts = np.empty(max_trades, dtype=np.float64)
    trade_exit_reasons = np.empty(max_trades, dtype=np.int8)
    trade_costs = np.empty(max_trades, dtype=np.float64)  # For entry cost tracking
    trade_proceeds = np.empty(max_trades, dtype=np.float64)  # For exit proceeds tracking

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
            # Long position: cash + market value of shares
            portfolio_value = cash + position * price
        elif position < 0:
            # Short position: cash + unrealized P&L from short
            # P&L = (entry_price - current_price) * shares
            short_pnl = (position_price - price) * abs(position)
            portfolio_value = cash + short_pnl
        else:
            portfolio_value = cash

        # Track equity even outside market hours
        if not market_hours[i]:
            equity[i] = portfolio_value
            continue

        # Track time in position (only during market hours)
        if position != 0:
            bars_in_position += 1

        # === RISK MANAGEMENT CHECKS ===
        exit_triggered = False
        exit_reason = np.int8(0)  # EXIT_SIGNAL

        if position > 0:  # Long position checks
            pnl_pct = (price - position_price) / position_price

            # Check stop loss (price dropped too much)
            # Use <= to match Python's RiskManager behavior (includes boundary)
            if use_stop_loss and pnl_pct <= -stop_loss_pct:
                exit_triggered = True
                exit_reason = np.int8(1)  # EXIT_STOP_LOSS
            # Check profit target (price rose enough)
            # Use >= to match Python's RiskManager behavior (includes boundary)
            elif use_profit_target and pnl_pct >= profit_target_pct:
                exit_triggered = True
                exit_reason = np.int8(3)  # EXIT_PROFIT_TARGET
            # Check time stop (held too long)
            elif use_time_stop and bars_in_position >= max_bars_in_position:
                exit_triggered = True
                exit_reason = np.int8(2)  # EXIT_TIME_STOP

        elif position < 0:  # Short position checks
            # For shorts, profit when price drops
            pnl_pct = (position_price - price) / position_price

            # Use <= and >= to match Python's RiskManager behavior (includes boundary)
            if use_stop_loss and pnl_pct <= -stop_loss_pct:
                exit_triggered = True
                exit_reason = np.int8(1)  # EXIT_STOP_LOSS
            elif use_profit_target and pnl_pct >= profit_target_pct:
                exit_triggered = True
                exit_reason = np.int8(3)  # EXIT_PROFIT_TARGET
            elif use_time_stop and bars_in_position >= max_bars_in_position:
                exit_triggered = True
                exit_reason = np.int8(2)  # EXIT_TIME_STOP

        # === EXECUTE RISK EXIT ===
        if exit_triggered and trade_idx < max_trades:
            if position > 0:
                # Close long position (sell)
                slippage_adj = price * (1 - slippage)
                proceeds = position * slippage_adj
                fee = proceeds * fees
                net_proceeds = proceeds - fee

                pnl = net_proceeds - (position * position_price)
                pnl_pct_val = (pnl / (position * position_price)) * 100

                trade_bars[trade_idx] = i
                trade_types[trade_idx] = np.int8(1)  # TRADE_EXIT
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

            else:  # position < 0, close short
                # Buy to cover (buy at higher price due to slippage)
                slippage_adj = price * (1 + slippage)
                cost_to_cover = abs(position) * slippage_adj
                fee = cost_to_cover * fees
                total_cost = cost_to_cover + fee

                proceeds_from_short = abs(position) * position_price
                pnl = proceeds_from_short - cost_to_cover - fee
                pnl_pct_val = (pnl / proceeds_from_short) * 100

                trade_bars[trade_idx] = i
                trade_types[trade_idx] = np.int8(3)  # TRADE_COVER_SHORT
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

        # Note: We do NOT recalculate portfolio_value here to match Python's behavior.
        # Python uses the start-of-bar portfolio_value for position sizing, even after
        # a risk-triggered exit. This maintains exact parity.

        # === ENTRY SIGNAL: want to go LONG ===
        # Note: Entry signals should be processed even after a risk exit (matching Python)
        if entries[i]:
            # First, close any short position (only if not already closed by risk exit)
            if position < 0 and not exit_triggered and trade_idx < max_trades:
                slippage_adj = price * (1 + slippage)
                cost_to_cover = abs(position) * slippage_adj
                fee = cost_to_cover * fees
                total_cost = cost_to_cover + fee

                proceeds_from_short = abs(position) * position_price
                pnl = proceeds_from_short - cost_to_cover - fee
                pnl_pct_val = (pnl / proceeds_from_short) * 100

                trade_bars[trade_idx] = i
                trade_types[trade_idx] = np.int8(3)  # TRADE_COVER_SHORT
                trade_prices[trade_idx] = price
                trade_shares[trade_idx] = abs(position)
                trade_pnls[trade_idx] = pnl
                trade_pnl_pcts[trade_idx] = pnl_pct_val
                trade_exit_reasons[trade_idx] = np.int8(0)  # EXIT_SIGNAL
                trade_costs[trade_idx] = total_cost
                trade_proceeds[trade_idx] = 0.0
                trade_idx += 1

                cash -= total_cost
                position = 0.0
                position_price = 0.0
                bars_in_position = 0
                portfolio_value = cash

            # Now open long if flat and have capital
            if position == 0 and cash > 0 and trade_idx < max_trades:
                # Note: We keep the stale portfolio_value from start of bar to match Python's
                # behavior. Python only recalculates portfolio_value after signal-based exits,
                # not after risk-triggered exits. This may result in larger positions after a
                # loss, but maintains exact parity with the Python implementation.

                # Calculate shares based on position sizing (matches Python FixedPercentageSizer)
                # Note: Position sizing is done with raw price, slippage applied to cost
                target_value = portfolio_value * position_size_pct
                shares = np.floor(target_value / price)  # Round down like Python's int()

                # Apply slippage and fees to cost (same as Python)
                slippage_adj = price * (1 + slippage)
                cost = shares * slippage_adj
                fee = cost * fees
                total_cost = cost + fee

                if total_cost <= cash and shares > 0:
                    trade_bars[trade_idx] = i
                    trade_types[trade_idx] = np.int8(0)  # TRADE_ENTRY
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
        # Note: Unlike entry signals, exit signals should still be processed after
        # a risk exit to allow opening shorts (matching Python behavior)
        if exits[i] and not entries[i]:
            # First, close any long position (only if not already closed by risk exit)
            if position > 0 and not exit_triggered and trade_idx < max_trades:
                slippage_adj = price * (1 - slippage)
                proceeds = position * slippage_adj
                fee = proceeds * fees
                net_proceeds = proceeds - fee

                pnl = net_proceeds - (position * position_price)
                pnl_pct_val = (pnl / (position * position_price)) * 100

                trade_bars[trade_idx] = i
                trade_types[trade_idx] = np.int8(1)  # TRADE_EXIT
                trade_prices[trade_idx] = price
                trade_shares[trade_idx] = position
                trade_pnls[trade_idx] = pnl
                trade_pnl_pcts[trade_idx] = pnl_pct_val
                trade_exit_reasons[trade_idx] = np.int8(0)  # EXIT_SIGNAL
                trade_proceeds[trade_idx] = net_proceeds
                trade_costs[trade_idx] = 0.0
                trade_idx += 1

                cash += net_proceeds
                position = 0.0
                position_price = 0.0
                bars_in_position = 0
                portfolio_value = cash

            # Open short if flat, shorts allowed, and have capital
            # This can happen after both signal-based and risk-based exits
            if position == 0 and allow_shorts and cash > 0 and trade_idx < max_trades:
                # Note: We keep the stale portfolio_value from start of bar to match Python's
                # behavior. Python only recalculates portfolio_value after signal-based exits,
                # not after risk-triggered exits.

                # For shorts, we receive proceeds from the sale
                # Calculate shares based on position sizing (matches Python FixedPercentageSizer)
                # Note: Position sizing is done with raw price, slippage applied to proceeds
                target_value = portfolio_value * position_size_pct
                shares = np.floor(target_value / price)  # Round down like Python's int()

                # Apply slippage and fees to proceeds (same as Python)
                slippage_adj = price * (1 - slippage)
                proceeds = shares * slippage_adj
                fee = proceeds * fees
                net_proceeds = proceeds - fee

                if shares > 0:
                    trade_bars[trade_idx] = i
                    trade_types[trade_idx] = np.int8(2)  # TRADE_SHORT_ENTRY
                    trade_prices[trade_idx] = price
                    trade_shares[trade_idx] = shares
                    trade_pnls[trade_idx] = 0.0
                    trade_pnl_pcts[trade_idx] = 0.0
                    trade_exit_reasons[trade_idx] = np.int8(-1)
                    trade_proceeds[trade_idx] = net_proceeds
                    trade_costs[trade_idx] = 0.0
                    trade_idx += 1

                    position = -shares  # Negative = short
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

    # Return sliced arrays (only actual trades)
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
        trade_idx
    )
