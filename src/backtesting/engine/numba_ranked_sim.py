"""
Numba JIT-compiled ranked portfolio simulation for momentum/factor strategies.

This module provides a JIT-compiled simulation for strategies that:
1. Score/rank multiple symbols
2. Select top N by score
3. Hold equal-weight positions
4. Rebalance periodically

Designed for MP (Momentum Protection) but reusable for any ranking strategy.
"""

import numba
import numpy as np
from typing import Tuple


@numba.jit(nopython=True, cache=True)
def simulate_ranked_portfolio(
    returns_matrix: np.ndarray,      # (n_days, n_symbols) daily returns
    momentum_matrix: np.ndarray,     # (n_days, n_symbols) momentum scores
    vix_values: np.ndarray,          # (n_days,) VIX levels
    spy_drawdown: np.ndarray,        # (n_days,) SPY drawdown from peak
    top_n: int,
    position_size: float,
    vix_threshold: float,
    spy_dd_threshold: float,
    reduced_exposure: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate a ranked portfolio strategy using Numba JIT compilation.

    The strategy:
    1. Each day, rank symbols by momentum score from previous day (no lookahead)
    2. Select top N symbols
    3. Equal weight positions (position_size per symbol)
    4. Reduce exposure when VIX high or SPY in drawdown

    Args:
        returns_matrix: Daily returns for all symbols (n_days, n_symbols)
        momentum_matrix: Momentum scores for ranking (n_days, n_symbols)
        vix_values: VIX closing values
        spy_drawdown: SPY drawdown from rolling peak (negative values)
        top_n: Number of top symbols to hold
        position_size: Position size per symbol (e.g., 0.065 = 6.5%)
        vix_threshold: VIX level above which to reduce exposure
        spy_dd_threshold: SPY drawdown below which to reduce exposure (e.g., -0.05)
        reduced_exposure: Exposure multiplier when risk signals trigger (e.g., 0.5)

    Returns:
        Tuple of:
        - portfolio_returns: (n_days,) daily portfolio returns
        - exposures: (n_days,) exposure level each day (1.0 or reduced_exposure)
    """
    n_days, n_symbols = returns_matrix.shape
    portfolio_returns = np.zeros(n_days, dtype=np.float64)
    exposures = np.ones(n_days, dtype=np.float64)

    # Pre-allocate array for sorting indices
    sorted_indices = np.empty(n_symbols, dtype=np.int64)

    for i in range(1, n_days):
        # Use previous day's momentum scores (no lookahead bias)
        mom_scores = momentum_matrix[i - 1]

        # Count valid (non-NaN) scores
        valid_count = 0
        for j in range(n_symbols):
            if not np.isnan(mom_scores[j]):
                valid_count += 1

        if valid_count < top_n:
            # Not enough valid scores, skip this day
            portfolio_returns[i] = 0.0
            continue

        # Manual argsort for top N (Numba-compatible)
        # Copy scores and track indices
        score_copy = np.empty(n_symbols, dtype=np.float64)
        for j in range(n_symbols):
            if np.isnan(mom_scores[j]):
                score_copy[j] = -np.inf  # Push NaN to bottom
            else:
                score_copy[j] = mom_scores[j]

        # Get indices that would sort in descending order
        for j in range(n_symbols):
            sorted_indices[j] = j

        # Simple selection sort for top N (faster than full sort for small N)
        for j in range(min(top_n, n_symbols)):
            max_idx = j
            for k in range(j + 1, n_symbols):
                if score_copy[sorted_indices[k]] > score_copy[sorted_indices[max_idx]]:
                    max_idx = k
            # Swap
            sorted_indices[j], sorted_indices[max_idx] = sorted_indices[max_idx], sorted_indices[j]

        # Check risk signals from previous day
        risk_off = False
        if vix_values[i - 1] > vix_threshold:
            risk_off = True
        if spy_drawdown[i - 1] < spy_dd_threshold:
            risk_off = True

        exposure = reduced_exposure if risk_off else 1.0
        exposures[i] = exposure

        # Calculate portfolio return from top N symbols
        total_return = 0.0
        valid_positions = 0

        for j in range(top_n):
            sym_idx = sorted_indices[j]
            ret = returns_matrix[i, sym_idx]
            if not np.isnan(ret):
                total_return += ret
                valid_positions += 1

        if valid_positions > 0:
            avg_return = total_return / valid_positions
            # Portfolio return = avg return * position size * num positions * exposure
            portfolio_returns[i] = avg_return * position_size * valid_positions * exposure

    return portfolio_returns, exposures


@numba.jit(nopython=True, cache=True)
def calculate_momentum_scores(
    prices: np.ndarray,      # (n_days, n_symbols) close prices
    long_period: int,        # e.g., 252 for 12-month
    short_period: int,       # e.g., 21 for 1-month
    penalty_weight: float = 1.0,  # weight for short-term penalty
    long_weight: float = 1.0      # weight for long-term return
) -> np.ndarray:
    """
    Calculate momentum scores with configurable weights.

    Formula: momentum = (long_weight * long_return) - (penalty_weight * short_return)

    Args:
        prices: Close prices (n_days, n_symbols)
        long_period: Long lookback period in days
        short_period: Short lookback period in days
        penalty_weight: Weight for short-term return penalty (default 1.0)
            - 0.0: Pure long-term momentum (no exclusion)
            - 1.0: Standard momentum (equal subtraction)
            - >1.0: Stronger penalization of recent gains
        long_weight: Weight for long-term return (default 1.0)
            - <1.0: Dampen long-term signal
            - 1.0: Standard long-term weight
            - >1.0: Amplify long-term signal

    Returns:
        momentum_scores: (n_days, n_symbols) momentum values
    """
    n_days, n_symbols = prices.shape
    momentum = np.full((n_days, n_symbols), np.nan, dtype=np.float64)

    for i in range(max(long_period, short_period), n_days):
        for j in range(n_symbols):
            price_now = prices[i, j]
            price_long_ago = prices[i - long_period, j]
            price_short_ago = prices[i - short_period, j]

            if price_now > 0 and price_long_ago > 0 and price_short_ago > 0:
                long_return = (price_now - price_long_ago) / price_long_ago
                short_return = (price_now - price_short_ago) / price_short_ago
                momentum[i, j] = (long_weight * long_return) - (penalty_weight * short_return)

    return momentum


@numba.jit(nopython=True, cache=True)
def simulate_ranked_portfolio_with_stops(
    prices_matrix: np.ndarray,       # (n_days, n_symbols) close prices
    returns_matrix: np.ndarray,      # (n_days, n_symbols) daily returns
    momentum_matrix: np.ndarray,     # (n_days, n_symbols) momentum scores
    vix_values: np.ndarray,          # (n_days,) VIX levels
    spy_drawdown: np.ndarray,        # (n_days,) SPY drawdown from peak
    top_n: int,
    position_size: float,
    vix_threshold: float,
    spy_dd_threshold: float,
    stop_loss_pct: float,            # Stop loss percentage (e.g., -0.10 for -10%)
    reduced_exposure: float = 0.5
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate ranked portfolio with position-level stop losses.

    Key differences from basic simulation:
    1. Tracks entry prices for each position
    2. Monitors cumulative P&L from entry
    3. Closes position when stop loss hit
    4. Replaces stopped positions with next ranked symbol

    Args:
        prices_matrix: Close prices (n_days, n_symbols)
        returns_matrix: Daily returns for all symbols (n_days, n_symbols)
        momentum_matrix: Momentum scores for ranking (n_days, n_symbols)
        vix_values: VIX closing values
        spy_drawdown: SPY drawdown from rolling peak (negative values)
        top_n: Number of top symbols to hold
        position_size: Position size per symbol (e.g., 0.065 = 6.5%)
        vix_threshold: VIX level above which to reduce exposure
        spy_dd_threshold: SPY drawdown below which to reduce exposure
        stop_loss_pct: Stop loss threshold (negative, e.g., -0.10 for 10% loss)
        reduced_exposure: Exposure multiplier when risk signals trigger

    Returns:
        Tuple of:
        - portfolio_returns: (n_days,) daily portfolio returns
        - exposures: (n_days,) exposure level each day
        - stop_counts: (n_days,) number of stops triggered each day
    """
    n_days, n_symbols = returns_matrix.shape
    portfolio_returns = np.zeros(n_days, dtype=np.float64)
    exposures = np.ones(n_days, dtype=np.float64)
    stop_counts = np.zeros(n_days, dtype=np.int64)

    # Position tracking: -1 means empty slot
    current_positions = np.full(top_n, -1, dtype=np.int64)  # Symbol indices
    entry_prices = np.zeros(top_n, dtype=np.float64)        # Entry prices

    # Pre-allocate arrays
    sorted_indices = np.empty(n_symbols, dtype=np.int64)
    score_copy = np.empty(n_symbols, dtype=np.float64)

    for i in range(1, n_days):
        # Use previous day's momentum scores (no lookahead bias)
        mom_scores = momentum_matrix[i - 1]

        # Count valid scores
        valid_count = 0
        for j in range(n_symbols):
            if not np.isnan(mom_scores[j]):
                valid_count += 1

        if valid_count < top_n:
            portfolio_returns[i] = 0.0
            continue

        # Sort symbols by momentum (descending)
        for j in range(n_symbols):
            if np.isnan(mom_scores[j]):
                score_copy[j] = -np.inf
            else:
                score_copy[j] = mom_scores[j]

        for j in range(n_symbols):
            sorted_indices[j] = j

        # Full sort needed for stop loss replacement logic
        for j in range(n_symbols):
            max_idx = j
            for k in range(j + 1, n_symbols):
                if score_copy[sorted_indices[k]] > score_copy[sorted_indices[max_idx]]:
                    max_idx = k
            sorted_indices[j], sorted_indices[max_idx] = sorted_indices[max_idx], sorted_indices[j]

        # Check risk signals
        risk_off = False
        if vix_values[i - 1] > vix_threshold:
            risk_off = True
        if spy_drawdown[i - 1] < spy_dd_threshold:
            risk_off = True
        exposure = reduced_exposure if risk_off else 1.0
        exposures[i] = exposure

        # Step 1: Check existing positions for stop losses
        stops_today = 0
        for slot in range(top_n):
            sym_idx = current_positions[slot]
            if sym_idx >= 0:  # Position exists
                current_price = prices_matrix[i, sym_idx]
                entry_price = entry_prices[slot]
                if entry_price > 0 and current_price > 0:
                    position_return = (current_price - entry_price) / entry_price
                    if position_return <= stop_loss_pct:
                        # Stop loss triggered - realize the loss
                        current_positions[slot] = -1
                        entry_prices[slot] = 0.0
                        stops_today += 1

        stop_counts[i] = stops_today

        # Step 2: Fill empty slots with top ranked symbols not already held
        for slot in range(top_n):
            if current_positions[slot] < 0:  # Empty slot
                for rank in range(n_symbols):
                    candidate = sorted_indices[rank]
                    # Check if candidate not already in portfolio
                    already_held = False
                    for s in range(top_n):
                        if current_positions[s] == candidate:
                            already_held = True
                            break
                    if not already_held:
                        # Check if valid price
                        if prices_matrix[i, candidate] > 0:
                            current_positions[slot] = candidate
                            entry_prices[slot] = prices_matrix[i, candidate]
                            break

        # Step 3: Calculate today's portfolio return
        total_return = 0.0
        valid_positions = 0

        for slot in range(top_n):
            sym_idx = current_positions[slot]
            if sym_idx >= 0:
                ret = returns_matrix[i, sym_idx]
                if not np.isnan(ret):
                    total_return += ret
                    valid_positions += 1
                    # Update entry price for tracking (mark-to-market for next day's stop check)
                    # Keep original entry price for stop loss calculation

        if valid_positions > 0:
            avg_return = total_return / valid_positions
            portfolio_returns[i] = avg_return * position_size * valid_positions * exposure

    return portfolio_returns, exposures, stop_counts


@numba.jit(nopython=True, cache=True)
def simulate_ranked_portfolio_with_intraday_stops(
    prices_matrix: np.ndarray,       # (n_days, n_symbols) close prices
    lows_matrix: np.ndarray,         # (n_days, n_symbols) daily low prices
    returns_matrix: np.ndarray,      # (n_days, n_symbols) daily returns
    momentum_matrix: np.ndarray,     # (n_days, n_symbols) momentum scores
    vix_values: np.ndarray,          # (n_days,) VIX levels
    spy_drawdown: np.ndarray,        # (n_days,) SPY drawdown from peak
    top_n: int,
    position_size: float,
    vix_threshold: float,
    spy_dd_threshold: float,
    stop_loss_pct: float,            # Stop loss percentage (e.g., -0.10 for -10%)
    reduced_exposure: float = 0.5
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate ranked portfolio with INTRADAY stop losses using daily low prices.

    Key difference from end-of-day stop loss:
    - Checks if daily LOW breached the stop price (not close)
    - Exits at stop price when triggered (realistic execution)
    - Calculates partial-day return: open to stop price

    Args:
        prices_matrix: Close prices (n_days, n_symbols)
        lows_matrix: Daily LOW prices (n_days, n_symbols) - for intraday stop check
        returns_matrix: Daily returns for all symbols (n_days, n_symbols)
        momentum_matrix: Momentum scores for ranking (n_days, n_symbols)
        vix_values: VIX closing values
        spy_drawdown: SPY drawdown from rolling peak (negative values)
        top_n: Number of top symbols to hold
        position_size: Position size per symbol (e.g., 0.065 = 6.5%)
        vix_threshold: VIX level above which to reduce exposure
        spy_dd_threshold: SPY drawdown below which to reduce exposure
        stop_loss_pct: Stop loss threshold (negative, e.g., -0.10 for 10% loss)
        reduced_exposure: Exposure multiplier when risk signals trigger

    Returns:
        Tuple of:
        - portfolio_returns: (n_days,) daily portfolio returns
        - exposures: (n_days,) exposure level each day
        - stop_counts: (n_days,) number of stops triggered each day
    """
    n_days, n_symbols = returns_matrix.shape
    portfolio_returns = np.zeros(n_days, dtype=np.float64)
    exposures = np.ones(n_days, dtype=np.float64)
    stop_counts = np.zeros(n_days, dtype=np.int64)

    # Position tracking: -1 means empty slot
    current_positions = np.full(top_n, -1, dtype=np.int64)  # Symbol indices
    entry_prices = np.zeros(top_n, dtype=np.float64)        # Entry prices

    # Pre-allocate arrays
    sorted_indices = np.empty(n_symbols, dtype=np.int64)
    score_copy = np.empty(n_symbols, dtype=np.float64)

    for i in range(1, n_days):
        # Use previous day's momentum scores (no lookahead bias)
        mom_scores = momentum_matrix[i - 1]

        # Count valid scores
        valid_count = 0
        for j in range(n_symbols):
            if not np.isnan(mom_scores[j]):
                valid_count += 1

        if valid_count < top_n:
            portfolio_returns[i] = 0.0
            continue

        # Sort symbols by momentum (descending)
        for j in range(n_symbols):
            if np.isnan(mom_scores[j]):
                score_copy[j] = -np.inf
            else:
                score_copy[j] = mom_scores[j]

        for j in range(n_symbols):
            sorted_indices[j] = j

        # Full sort needed for stop loss replacement logic
        for j in range(n_symbols):
            max_idx = j
            for k in range(j + 1, n_symbols):
                if score_copy[sorted_indices[k]] > score_copy[sorted_indices[max_idx]]:
                    max_idx = k
            sorted_indices[j], sorted_indices[max_idx] = sorted_indices[max_idx], sorted_indices[j]

        # Check risk signals
        risk_off = False
        if vix_values[i - 1] > vix_threshold:
            risk_off = True
        if spy_drawdown[i - 1] < spy_dd_threshold:
            risk_off = True
        exposure = reduced_exposure if risk_off else 1.0
        exposures[i] = exposure

        # Step 1: Check existing positions for INTRADAY stop losses using daily low
        stops_today = 0
        stopped_returns = 0.0  # Accumulate returns from stopped positions

        for slot in range(top_n):
            sym_idx = current_positions[slot]
            if sym_idx >= 0:  # Position exists
                entry_price = entry_prices[slot]
                if entry_price > 0:
                    # Calculate stop price
                    stop_price = entry_price * (1.0 + stop_loss_pct)
                    daily_low = lows_matrix[i, sym_idx]

                    if daily_low > 0 and daily_low <= stop_price:
                        # INTRADAY stop triggered - exit at stop price
                        # Return for this position is exactly the stop loss %
                        stopped_returns += stop_loss_pct
                        current_positions[slot] = -1
                        entry_prices[slot] = 0.0
                        stops_today += 1

        stop_counts[i] = stops_today

        # Step 2: Fill empty slots with top ranked symbols not already held
        for slot in range(top_n):
            if current_positions[slot] < 0:  # Empty slot
                for rank in range(n_symbols):
                    candidate = sorted_indices[rank]
                    # Check if candidate not already in portfolio
                    already_held = False
                    for s in range(top_n):
                        if current_positions[s] == candidate:
                            already_held = True
                            break
                    if not already_held:
                        # Check if valid price
                        if prices_matrix[i, candidate] > 0:
                            current_positions[slot] = candidate
                            entry_prices[slot] = prices_matrix[i, candidate]
                            break

        # Step 3: Calculate today's portfolio return
        # Includes: stopped positions (at stop price) + held positions (at close)
        total_return = stopped_returns  # Start with stopped position returns
        active_positions = top_n - stops_today  # Positions still held

        for slot in range(top_n):
            sym_idx = current_positions[slot]
            if sym_idx >= 0:
                ret = returns_matrix[i, sym_idx]
                if not np.isnan(ret):
                    total_return += ret

        # Portfolio return = sum of all position returns * position_size * exposure
        # Each position contributes position_size to the portfolio
        portfolio_returns[i] = total_return * position_size * exposure

    return portfolio_returns, exposures, stop_counts


@numba.jit(nopython=True, cache=True)
def calculate_spy_drawdown(spy_prices: np.ndarray) -> np.ndarray:
    """
    Calculate SPY drawdown from rolling peak.

    Args:
        spy_prices: (n_days,) SPY close prices

    Returns:
        drawdown: (n_days,) drawdown values (negative, e.g., -0.05 = 5% drawdown)
    """
    n_days = len(spy_prices)
    drawdown = np.zeros(n_days, dtype=np.float64)
    running_max = spy_prices[0]

    for i in range(n_days):
        if spy_prices[i] > running_max:
            running_max = spy_prices[i]
        if running_max > 0:
            drawdown[i] = (spy_prices[i] - running_max) / running_max

    return drawdown


def run_mp_backtest_numba(
    prices_df,           # DataFrame with symbol columns (close prices)
    spy_series,          # Series of SPY prices
    vix_series,          # Series of VIX values
    long_period: int,
    short_period: int,
    top_n: int,
    position_size: float,
    vix_threshold: float,
    spy_dd_threshold: float,
    reduced_exposure: float = 0.5,
    penalty_weight: float = 1.0,
    long_weight: float = 1.0,
    stop_loss_pct: float = None,
    lows_df = None       # Optional DataFrame with daily low prices for intraday stops
) -> Tuple[np.ndarray, dict]:
    """
    High-level wrapper for Numba MP backtest.

    Converts pandas DataFrames to numpy arrays and runs the simulation.

    Args:
        prices_df: DataFrame of close prices (columns = symbols)
        spy_series: Series of SPY close prices
        vix_series: Series of VIX values
        long_period: Long momentum lookback period
        short_period: Short momentum lookback period
        top_n: Number of top symbols to hold
        position_size: Position size per symbol
        vix_threshold: VIX level for risk-off
        spy_dd_threshold: SPY drawdown threshold for risk-off
        reduced_exposure: Exposure multiplier in risk-off mode
        penalty_weight: Weight for short-term return penalty in momentum calculation.
            - 0.0: Pure long-term momentum
            - 1.0: Standard momentum (long - short)
            - >1.0: Stronger penalization of recent gains
        long_weight: Weight for long-term return in momentum calculation.
            - <1.0: Dampen long-term signal
            - 1.0: Standard long-term weight
            - >1.0: Amplify long-term signal
        stop_loss_pct: Optional stop loss percentage (negative, e.g., -0.10 for -10%).
            - None: No stop loss (original behavior)
            - -0.10: Close position if it drops 10% from entry
        lows_df: Optional DataFrame of daily LOW prices for intraday stop loss.
            - None: Use end-of-day stop loss (checks close price)
            - DataFrame: Use intraday stop loss (checks if low breached stop)

    Returns:
        Tuple of (portfolio_returns, metadata_dict)
    """
    # Convert to numpy arrays
    prices = prices_df.values.astype(np.float64)
    spy_prices = spy_series.values.astype(np.float64)
    vix_values = vix_series.values.astype(np.float64)

    # Calculate daily returns
    returns = np.zeros_like(prices)
    returns[1:] = (prices[1:] - prices[:-1]) / prices[:-1]
    returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)

    # Calculate momentum scores with both weights
    momentum = calculate_momentum_scores(prices, long_period, short_period, penalty_weight, long_weight)

    # Calculate SPY drawdown
    spy_dd = calculate_spy_drawdown(spy_prices)

    # Determine which stop loss mode to use
    use_intraday_stops = stop_loss_pct is not None and lows_df is not None

    # Run simulation
    if use_intraday_stops:
        # Intraday stop loss using daily lows
        lows = lows_df.values.astype(np.float64)
        portfolio_returns, exposures, stop_counts = simulate_ranked_portfolio_with_intraday_stops(
            prices_matrix=prices,
            lows_matrix=lows,
            returns_matrix=returns,
            momentum_matrix=momentum,
            vix_values=vix_values,
            spy_drawdown=spy_dd,
            top_n=top_n,
            position_size=position_size,
            vix_threshold=vix_threshold,
            spy_dd_threshold=spy_dd_threshold,
            stop_loss_pct=stop_loss_pct,
            reduced_exposure=reduced_exposure
        )
        total_stops = int(np.sum(stop_counts))
        stop_mode = 'intraday'
    elif stop_loss_pct is not None:
        # End-of-day stop loss using close prices
        portfolio_returns, exposures, stop_counts = simulate_ranked_portfolio_with_stops(
            prices_matrix=prices,
            returns_matrix=returns,
            momentum_matrix=momentum,
            vix_values=vix_values,
            spy_drawdown=spy_dd,
            top_n=top_n,
            position_size=position_size,
            vix_threshold=vix_threshold,
            spy_dd_threshold=spy_dd_threshold,
            stop_loss_pct=stop_loss_pct,
            reduced_exposure=reduced_exposure
        )
        total_stops = int(np.sum(stop_counts))
        stop_mode = 'eod'
    else:
        # No stop loss
        portfolio_returns, exposures = simulate_ranked_portfolio(
            returns_matrix=returns,
            momentum_matrix=momentum,
            vix_values=vix_values,
            spy_drawdown=spy_dd,
            top_n=top_n,
            position_size=position_size,
            vix_threshold=vix_threshold,
            spy_dd_threshold=spy_dd_threshold,
            reduced_exposure=reduced_exposure
        )
        total_stops = 0
        stop_mode = None

    metadata = {
        'n_days': len(portfolio_returns),
        'avg_exposure': float(np.mean(exposures)),
        'risk_off_days': int(np.sum(exposures < 1.0)),
        'penalty_weight': penalty_weight,
        'long_weight': long_weight,
        'stop_loss_pct': stop_loss_pct,
        'stop_mode': stop_mode,  # 'intraday', 'eod', or None
        'total_stops': total_stops,
    }

    return portfolio_returns, metadata
