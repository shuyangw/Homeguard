"""
Performance test: V1 vs V2 with track_state optimization.

Tests that V2 with track_state=False has similar performance to V1.
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
    print("PERFORMANCE TEST: V1 vs V2 (track_state=False) vs V2 (track_state=True)")
    print("=" * 80)

    # Load 6 months of SPY data
    loader = StreamingDataLoader(timeframe="1min")
    data = None
    for sym, df in loader.iter_symbols(["SPY"], "2023-01-01", "2023-06-30"):
        data = df.to_pandas().set_index("timestamp")

    print(f"Loaded {len(data):,} bars of SPY data")

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

    # Warm up (compile Numba)
    print("\nWarming up Numba...")
    _ = from_signals(**params)
    _ = from_signals_v2(**params, track_state=False)
    _ = from_signals_v2(**params, track_state=True)
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

    # Run V2 (track_state=False) benchmark
    print("Running V2 (track_state=False) benchmark...")
    gc.collect()
    tracemalloc.start()
    v2_nostate_start = time.perf_counter()
    pf_v2_nostate = from_signals_v2(**params, track_state=False)
    v2_nostate_time = time.perf_counter() - v2_nostate_start
    v2_nostate_mem = tracemalloc.get_traced_memory()[1] / 1024 / 1024
    tracemalloc.stop()

    # Run V2 (track_state=True) benchmark
    print("Running V2 (track_state=True) benchmark...")
    gc.collect()
    tracemalloc.start()
    v2_state_start = time.perf_counter()
    pf_v2_state = from_signals_v2(**params, track_state=True)
    v2_state_time = time.perf_counter() - v2_state_start
    v2_state_mem = tracemalloc.get_traced_memory()[1] / 1024 / 1024
    tracemalloc.stop()

    # Compare results
    s1 = pf_v1.stats()
    s2_nostate = pf_v2_nostate.stats()
    s2_state = pf_v2_state.stats()

    eq_diff_nostate = np.abs(pf_v1.equity_curve.values - pf_v2_nostate.equity_curve.values).max()
    eq_diff_state = np.abs(pf_v1.equity_curve.values - pf_v2_state.equity_curve.values).max()

    print()
    print("--- PARITY CHECK ---")
    print(f"V1 Return: {s1['Total Return [%]']:.4f}%")
    print(f"V2 (no state) Return: {s2_nostate['Total Return [%]']:.4f}%")
    print(f"V2 (with state) Return: {s2_state['Total Return [%]']:.4f}%")
    print(f"Max Equity Diff (no state): ${eq_diff_nostate:.6f}")
    print(f"Max Equity Diff (with state): ${eq_diff_state:.6f}")

    print()
    print("--- PERFORMANCE RESULTS ---")
    print(f"V1 Time:                  {v1_time:.3f}s | Memory: {v1_mem:.1f} MB")
    print(f"V2 (track_state=False):   {v2_nostate_time:.3f}s | Memory: {v2_nostate_mem:.1f} MB")
    print(f"V2 (track_state=True):    {v2_state_time:.3f}s | Memory: {v2_state_mem:.1f} MB")

    v2_nostate_overhead = ((v2_nostate_time / v1_time) - 1) * 100
    v2_state_overhead = ((v2_state_time / v1_time) - 1) * 100

    print()
    print("--- OVERHEAD vs V1 ---")
    print(f"V2 (track_state=False):  {v2_nostate_overhead:+.1f}% time")
    print(f"V2 (track_state=True):   {v2_state_overhead:+.1f}% time")

    print()
    print("--- STATE TRACKING ---")
    print(f"V2 (no state) portfolio_state rows: {len(pf_v2_nostate.portfolio_state)}")
    print(f"V2 (with state) portfolio_state rows: {len(pf_v2_state.portfolio_state)}")

    print()
    parity_ok = eq_diff_nostate < 0.01 and eq_diff_state < 0.01
    perf_ok = abs(v2_nostate_overhead) < 50  # Allow 50% variance (Python wrapper overhead)

    if parity_ok:
        print("[+] PARITY VERIFIED: All versions produce identical results")
    else:
        print("[-] PARITY FAILED")

    if perf_ok:
        print(f"[+] PERFORMANCE VERIFIED: V2 (no state) overhead is {v2_nostate_overhead:+.1f}%")
    else:
        print(f"[!] V2 (no state) has {v2_nostate_overhead:.1f}% overhead - investigate")

    print("=" * 80)

    return 0 if (parity_ok and perf_ok) else 1


if __name__ == "__main__":
    sys.exit(main())
