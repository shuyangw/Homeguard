"""
Numba-optimized core functions for ORB strategy.

Provides JIT-compiled signal generation for the Opening Range Breakout strategy,
achieving 10-50x speedup over pure Python implementation.
"""

import numba
import numpy as np


@numba.jit(nopython=True)
def calculate_opening_range_numba(
    hour_of_day: np.ndarray,
    minute_of_day: np.ndarray,
    dates: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    or_end_minutes: int = 15
) -> tuple:
    """
    Calculate opening range for each day with Numba JIT.

    Args:
        hour_of_day: Hour of day (0-23) for each bar
        minute_of_day: Minute of hour (0-59) for each bar
        dates: Date index for each bar (days since epoch)
        highs: High prices
        lows: Low prices
        or_end_minutes: Minutes after market open for OR calculation

    Returns:
        Tuple of (or_highs, or_lows, unique_dates) arrays indexed by date
    """
    n = len(dates)
    if n == 0:
        return np.empty(0), np.empty(0), np.empty(0, dtype=np.int64)

    # Market open time
    market_open_hour = 9
    market_open_minute = 30

    # OR end time
    or_end_hour = 9
    or_end_minute = 30 + or_end_minutes
    if or_end_minute >= 60:
        or_end_hour = 10
        or_end_minute = or_end_minute - 60

    # Pre-allocate arrays for each unique date
    unique_dates = np.unique(dates)
    n_dates = len(unique_dates)

    or_highs = np.full(n_dates, np.nan)
    or_lows = np.full(n_dates, np.nan)

    for i in range(n_dates):
        date = unique_dates[i]
        day_mask = dates == date

        # Get bars for this day
        day_hours = hour_of_day[day_mask]
        day_minutes = minute_of_day[day_mask]
        day_highs = highs[day_mask]
        day_lows = lows[day_mask]

        if len(day_hours) == 0:
            continue

        # Find bars within opening range (9:30 to 9:30+or_end_minutes)
        or_highs_list = []
        or_lows_list = []

        for j in range(len(day_hours)):
            hour = day_hours[j]
            minute = day_minutes[j]

            # Check if within OR period
            after_open = (hour > market_open_hour) or (hour == market_open_hour and minute >= market_open_minute)
            before_end = (hour < or_end_hour) or (hour == or_end_hour and minute < or_end_minute)

            if after_open and before_end:
                or_highs_list.append(day_highs[j])
                or_lows_list.append(day_lows[j])

        if len(or_highs_list) > 0:
            or_highs[i] = max(or_highs_list)
            or_lows[i] = min(or_lows_list)

    return or_highs, or_lows, unique_dates


@numba.jit(nopython=True)
def calculate_or_avg_volumes_numba(
    hour_of_day: np.ndarray,
    minute_of_day: np.ndarray,
    dates: np.ndarray,
    volumes: np.ndarray,
    or_end_minutes: int = 15
) -> tuple:
    """
    Calculate average volume during opening range for each day.

    Returns:
        Tuple of (avg_volumes, unique_dates) arrays indexed by date
    """
    n = len(dates)
    if n == 0:
        return np.empty(0), np.empty(0, dtype=np.int64)

    # Market open time
    market_open_hour = 9
    market_open_minute = 30

    # OR end time
    or_end_hour = 9
    or_end_minute = 30 + or_end_minutes
    if or_end_minute >= 60:
        or_end_hour = 10
        or_end_minute = or_end_minute - 60

    # Pre-allocate arrays for each unique date
    unique_dates = np.unique(dates)
    n_dates = len(unique_dates)

    avg_volumes = np.zeros(n_dates)

    for i in range(n_dates):
        date = unique_dates[i]
        day_mask = dates == date

        # Get bars for this day
        day_hours = hour_of_day[day_mask]
        day_minutes = minute_of_day[day_mask]
        day_volumes = volumes[day_mask]

        if len(day_hours) == 0:
            continue

        # Find bars within opening range
        or_volumes = []

        for j in range(len(day_hours)):
            hour = day_hours[j]
            minute = day_minutes[j]

            # Check if within OR period
            after_open = (hour > market_open_hour) or (hour == market_open_hour and minute >= market_open_minute)
            before_end = (hour < or_end_hour) or (hour == or_end_hour and minute < or_end_minute)

            if after_open and before_end:
                or_volumes.append(day_volumes[j])

        if len(or_volumes) > 0:
            total = 0.0
            for v in or_volumes:
                total += v
            avg_volumes[i] = total / len(or_volumes)

    return avg_volumes, unique_dates


@numba.jit(nopython=True)
def generate_orb_signals_numba(
    hour_of_day: np.ndarray,
    minute_of_day: np.ndarray,
    dates: np.ndarray,
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    volumes: np.ndarray,
    vwaps: np.ndarray,
    fast_emas: np.ndarray,
    slow_emas: np.ndarray,
    rvols: np.ndarray,
    atrs: np.ndarray,
    or_avg_volumes: np.ndarray,
    or_end_minutes: int = 15,
    rvol_threshold: float = 1.5,
    target_multiplier: float = 1.0,
    long_only: bool = False,
    use_vwap_filter: bool = True,
    use_ema_filter: bool = True,
    exit_hour: int = 15,
    exit_minute: int = 45,
    one_trade_per_day: bool = False,
    min_or_width_pct: float = 0.0,
    breakout_buffer_pct: float = 0.0,
    entry_cutoff_hour: int = 15,
    entry_cutoff_minute: int = 30,
    use_atr_stops: bool = False,
    atr_stop_multiplier: float = 1.5,
    use_volume_confirmation: bool = False,
    volume_breakout_mult: float = 1.5,
    use_trailing_stop: bool = False
) -> tuple:
    """
    Generate ORB entry/exit signals with Numba JIT compilation.

    Args:
        hour_of_day: Hour of day (0-23) for each bar
        minute_of_day: Minute of hour (0-59) for each bar
        dates: Date index for each bar
        opens: Open prices
        highs: High prices
        lows: Low prices
        closes: Close prices
        volumes: Volume
        vwaps: VWAP values
        fast_emas: Fast EMA values
        slow_emas: Slow EMA values
        rvols: Relative volume values
        atrs: ATR values for ATR-based stops
        or_avg_volumes: Average volume during OR period for each day
        or_end_minutes: Minutes for opening range (default 15)
        rvol_threshold: Minimum RVOL for entry (default 1.5)
        target_multiplier: Target as multiple of OR height (default 1.0)
        long_only: Only take long trades (default False)
        use_vwap_filter: Require VWAP alignment (default True)
        use_ema_filter: Require EMA trend alignment (default True)
        exit_hour: Force exit hour (default 15 = 3 PM)
        exit_minute: Force exit minute (default 45)
        one_trade_per_day: If True, only take one trade per day (default False)
        min_or_width_pct: Minimum OR width as % of price to trade (default 0.0 = no filter)
                          Recommended: 0.5% for 2x ETFs, 0.75% for 3x ETFs
        breakout_buffer_pct: Buffer as % of OR height for breakout confirmation (default 0.0)
                             Recommended: 0.25 (25% of OR height)
        entry_cutoff_hour: No entries after this hour (default 15 = 3 PM)
        entry_cutoff_minute: No entries after this minute (default 30)
        use_atr_stops: Use ATR-based stops instead of OR-opposite (default False)
        atr_stop_multiplier: ATR multiplier for stop distance (default 1.5)
        use_volume_confirmation: Require above-avg volume on breakout bar (default False)
        volume_breakout_mult: Multiplier for breakout volume requirement (default 1.5)
        use_trailing_stop: Enable trailing stop after reaching 1R (default False)

    Returns:
        Tuple of (long_entries, long_exits, short_entries, short_exits) boolean arrays
    """
    n = len(dates)

    # Initialize signal arrays
    long_entries = np.zeros(n, dtype=np.bool_)
    long_exits = np.zeros(n, dtype=np.bool_)
    short_entries = np.zeros(n, dtype=np.bool_)
    short_exits = np.zeros(n, dtype=np.bool_)

    if n == 0:
        return long_entries, long_exits, short_entries, short_exits

    # Calculate opening ranges
    or_highs, or_lows, or_dates = calculate_opening_range_numba(
        hour_of_day, minute_of_day, dates, highs, lows, or_end_minutes
    )

    # Market times
    market_open_hour = 9
    market_open_minute = 30

    # OR end time
    or_end_hour = 9
    or_end_minute = 30 + or_end_minutes
    if or_end_minute >= 60:
        or_end_hour = 10
        or_end_minute = or_end_minute - 60

    # Track position state
    in_position = False
    position_direction = 0  # 1=long, -1=short
    entry_price = 0.0
    stop_loss = 0.0
    target = 0.0
    prev_close_inside_or = True
    highest_since_entry = 0.0  # For trailing stop
    lowest_since_entry = 0.0   # For trailing stop
    reached_1r = False         # For trailing stop activation
    or_height_for_position = 0.0  # Store OR height for trailing stop calc

    # Track daily trade status (for one_trade_per_day)
    last_trade_date = -1  # Date of last trade
    traded_today = False

    for i in range(n):
        # Get current time
        hour = hour_of_day[i]
        minute = minute_of_day[i]
        date = dates[i]

        # Reset daily trade flag at start of new day
        if date != last_trade_date:
            traded_today = False

        # Find OR for this date
        or_idx = -1
        for j in range(len(or_dates)):
            if or_dates[j] == date:
                or_idx = j
                break

        if or_idx == -1:
            continue

        or_high = or_highs[or_idx]
        or_low = or_lows[or_idx]

        if np.isnan(or_high) or np.isnan(or_low):
            continue

        or_height = or_high - or_low

        # Skip if OR too narrow (use min_or_width_pct if set, otherwise 0.1% minimum)
        or_width_pct = (or_height / or_high) * 100.0
        min_width = min_or_width_pct if min_or_width_pct > 0.0 else 0.1
        if or_width_pct < min_width:
            continue

        # Skip during OR formation
        before_or_end = (hour < or_end_hour) or (hour == or_end_hour and minute < or_end_minute)
        if before_or_end:
            prev_close_inside_or = (or_low <= closes[i] <= or_high)
            continue

        # === EXIT LOGIC (check first) ===
        if in_position:
            should_exit = False

            # Update highest/lowest since entry for trailing stop
            if position_direction == 1:  # Long
                if highs[i] > highest_since_entry:
                    highest_since_entry = highs[i]
            else:  # Short
                if lows[i] < lowest_since_entry or lowest_since_entry == 0.0:
                    lowest_since_entry = lows[i]

            # Check if reached 1R for trailing stop activation
            if use_trailing_stop and not reached_1r:
                if position_direction == 1:  # Long
                    unrealized = highest_since_entry - entry_price
                    if unrealized >= or_height_for_position:
                        reached_1r = True
                        # Move stop to breakeven + 0.5R
                        stop_loss = entry_price + (or_height_for_position * 0.5)
                else:  # Short
                    unrealized = entry_price - lowest_since_entry
                    if unrealized >= or_height_for_position:
                        reached_1r = True
                        # Move stop to breakeven + 0.5R
                        stop_loss = entry_price - (or_height_for_position * 0.5)

            # Time exit
            after_exit_time = (hour > exit_hour) or (hour == exit_hour and minute >= exit_minute)
            if after_exit_time:
                should_exit = True

            # Stop loss / target exits
            elif position_direction == 1:  # Long
                if lows[i] <= stop_loss:
                    should_exit = True
                elif highs[i] >= target:
                    should_exit = True
            else:  # Short
                if highs[i] >= stop_loss:
                    should_exit = True
                elif lows[i] <= target:
                    should_exit = True

            if should_exit:
                if position_direction == 1:
                    long_exits[i] = True
                else:
                    short_exits[i] = True
                in_position = False
                position_direction = 0
                reached_1r = False
                highest_since_entry = 0.0
                lowest_since_entry = 0.0

        # === ENTRY LOGIC ===
        # Skip if past entry cutoff
        after_entry_cutoff = (hour > entry_cutoff_hour) or (hour == entry_cutoff_hour and minute >= entry_cutoff_minute)
        if after_entry_cutoff:
            prev_close_inside_or = (or_low <= closes[i] <= or_high)
            continue

        # Skip if already traded today (one trade per day rule)
        if one_trade_per_day and traded_today:
            prev_close_inside_or = (or_low <= closes[i] <= or_high)
            continue

        # Skip if already in position
        if in_position:
            prev_close_inside_or = (or_low <= closes[i] <= or_high)
            continue

        # Detect breakout (with optional buffer)
        current_close_inside = (or_low <= closes[i] <= or_high)
        breakout_direction = 0

        # Calculate breakout threshold with buffer
        buffer = or_height * breakout_buffer_pct
        long_breakout_level = or_high + buffer
        short_breakout_level = or_low - buffer

        if closes[i] > long_breakout_level and prev_close_inside_or:
            breakout_direction = 1  # Long
        elif closes[i] < short_breakout_level and prev_close_inside_or:
            breakout_direction = -1  # Short

        if breakout_direction != 0:
            # Check entry filters
            should_enter = True

            # Long-only check
            if long_only and breakout_direction == -1:
                should_enter = False

            # RVOL filter
            if should_enter and rvols[i] < rvol_threshold:
                should_enter = False

            # VWAP filter
            if should_enter and use_vwap_filter:
                if breakout_direction == 1 and closes[i] <= vwaps[i]:
                    should_enter = False
                elif breakout_direction == -1 and closes[i] >= vwaps[i]:
                    should_enter = False

            # EMA trend filter
            if should_enter and use_ema_filter:
                if breakout_direction == 1 and fast_emas[i] <= slow_emas[i]:
                    should_enter = False
                elif breakout_direction == -1 and fast_emas[i] >= slow_emas[i]:
                    should_enter = False

            # Volume confirmation filter
            if should_enter and use_volume_confirmation:
                # Get avg volume for this day's OR period
                avg_or_vol = or_avg_volumes[or_idx]
                if avg_or_vol > 0 and volumes[i] < (avg_or_vol * volume_breakout_mult):
                    should_enter = False

            if should_enter:
                # Calculate stop and target
                if breakout_direction == 1:  # Long
                    entry_price = closes[i]
                    # ATR-based stop or OR-opposite stop
                    if use_atr_stops and atrs[i] > 0:
                        atr_stop = entry_price - (atrs[i] * atr_stop_multiplier)
                        stop_loss = max(atr_stop, or_low)  # Use wider of the two
                    else:
                        stop_loss = or_low
                    target = entry_price + (or_height * target_multiplier)
                    long_entries[i] = True
                else:  # Short
                    entry_price = closes[i]
                    # ATR-based stop or OR-opposite stop
                    if use_atr_stops and atrs[i] > 0:
                        atr_stop = entry_price + (atrs[i] * atr_stop_multiplier)
                        stop_loss = min(atr_stop, or_high)  # Use wider of the two
                    else:
                        stop_loss = or_high
                    target = entry_price - (or_height * target_multiplier)
                    short_entries[i] = True

                in_position = True
                position_direction = breakout_direction
                or_height_for_position = or_height
                highest_since_entry = closes[i]
                lowest_since_entry = closes[i]

                # Mark that we've traded today
                traded_today = True
                last_trade_date = date

        prev_close_inside_or = current_close_inside

    # Force exit at end of data if still in position
    if in_position:
        if position_direction == 1:
            long_exits[n-1] = True
        else:
            short_exits[n-1] = True

    return long_entries, long_exits, short_entries, short_exits
