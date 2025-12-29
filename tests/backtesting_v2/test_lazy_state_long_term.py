"""
Long-term test for lazy state tracking.

Verifies that track_state=True with lazy vectorized construction
performs well on large datasets (1.5M+ bars).
"""

import pandas as pd
import numpy as np
import time
import gc
import tracemalloc
import sys

from src.backtesting.engine.portfolio_simulator import from_signals
from src.backtesting_v2.engine.portfolio_simulator import from_signals_v2
from src.backtesting.utils.risk_config import RiskConfig
from src.backtesting.engine.streaming_data_loader import StreamingDataLoader


def main():
    print("=" * 80)
    print("LONG-TERM LAZY STATE TRACKING TEST (2017-2024)")
    print("=" * 80)

    # Load 7+ years of SPY data
    loader = StreamingDataLoader(timeframe="1min")
    data = None
    print("\nLoading SPY data (2017-2024)...")
    for sym, df in loader.iter_symbols(["SPY"], "2017-01-01", "2024-12-31"):
        data = df.to_pandas().set_index("timestamp")

    print(f"Loaded {len(data):,} bars")

    # Generate signals
    close = data["close"]
    ma_fast = close.rolling(20).mean()
    ma_slow = close.rolling(50).mean()
    entries = ((ma_fast > ma_slow) & (ma_fast.shift(1) <= ma_slow.shift(1))).fillna(False)
    exits = ((ma_fast < ma_slow) & (ma_fast.shift(1) >= ma_slow.shift(1))).fillna(False)

    params = dict(
        close=data["close"],
        entries=entries,
        exits=exits,
        init_cash=100000,
        fees=0.001,
        slippage=0.0005,
        freq="1min",
        market_hours_only=True,
        risk_config=RiskConfig.moderate(),
        price_data=data,
        allow_shorts=False,
        use_numba=True,
    )

    # Warm up
    print("Warming up Numba...")
    small_params = params.copy()
    small_params["close"] = params["close"].iloc[:1000]
    small_params["entries"] = params["entries"].iloc[:1000]
    small_params["exits"] = params["exits"].iloc[:1000]
    small_params["price_data"] = params["price_data"].iloc[:1000]
    _ = from_signals(**small_params)
    _ = from_signals_v2(**small_params, track_state=True)
    gc.collect()

    # Run V1 benchmark
    print("Running V1 benchmark...")
    gc.collect()
    tracemalloc.start()
    v1_start = time.perf_counter()
    pf_v1 = from_signals(**params)
    v1_time = time.perf_counter() - v1_start
    v1_mem = tracemalloc.get_traced_memory()[1] / 1024 / 1024
    tracemalloc.stop()

    # Run V2 (track_state=True) benchmark - simulation only
    print("Running V2 (track_state=True) benchmark (simulation)...")
    gc.collect()
    tracemalloc.start()
    v2_sim_start = time.perf_counter()
    pf_v2 = from_signals_v2(**params, track_state=True)
    v2_sim_time = time.perf_counter() - v2_sim_start
    v2_sim_mem = tracemalloc.get_traced_memory()[1] / 1024 / 1024
    tracemalloc.stop()

    # Now access portfolio_state to trigger lazy construction
    print("Building portfolio_state DataFrame (lazy)...")
    gc.collect()
    tracemalloc.start()
    state_start = time.perf_counter()
    state_df = pf_v2.portfolio_state
    state_time = time.perf_counter() - state_start
    state_mem = tracemalloc.get_traced_memory()[1] / 1024 / 1024
    tracemalloc.stop()

    # Verify parity
    eq_diff = np.abs(pf_v1.equity_curve.values - pf_v2.equity_curve.values).max()

    print()
    print("--- RESULTS ---")
    print(f"Bars: {len(data):,}")
    print(f"Equity diff: ${eq_diff:.6f}")
    print()
    print(f"V1 Time:                    {v1_time:.2f}s | Memory: {v1_mem:.1f} MB")
    print(f"V2 Simulation:              {v2_sim_time:.2f}s | Memory: {v2_sim_mem:.1f} MB")
    print(f"V2 State DataFrame build:  {state_time:.2f}s | Memory: {state_mem:.1f} MB")
    print(f"V2 Total (sim + state):     {v2_sim_time + state_time:.2f}s")
    print()
    print(f"State DataFrame rows: {len(state_df):,}")
    print(f"State DataFrame columns: {list(state_df.columns)}")
    print()

    v2_total_overhead = ((v2_sim_time + state_time) / v1_time - 1) * 100
    v2_sim_overhead = ((v2_sim_time) / v1_time - 1) * 100

    print(f"V2 simulation only overhead: {v2_sim_overhead:+.1f}%")
    print(f"V2 total (with state build) overhead: {v2_total_overhead:+.1f}%")

    print()
    if eq_diff < 0.01:
        print("[+] PARITY VERIFIED")
    else:
        print("[-] PARITY FAILED")

    if v2_total_overhead < 50:
        print(f"[+] PERFORMANCE VERIFIED: V2 total overhead is {v2_total_overhead:+.1f}%")
    else:
        print(f"[!] V2 total overhead is {v2_total_overhead:.1f}%")

    print("=" * 80)

    return 0 if eq_diff < 0.01 else 1


if __name__ == "__main__":
    sys.exit(main())
