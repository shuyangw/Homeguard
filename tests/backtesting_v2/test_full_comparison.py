"""
Full Backtest Comparison: V1 vs V2 (no tracking) vs V2 (with tracking).

Compares all three approaches on long-term data (2017-2024):
1. V1: Original backtesting engine
2. V2 (track_state=False): V2 using V1 Numba, no state tracking
3. V2 (track_state=True): V2 with lazy vectorized state tracking
"""

import pandas as pd
import numpy as np
import time
import gc
import tracemalloc
import sys
from typing import Dict, Any, Tuple

from src.backtesting.engine.portfolio_simulator import from_signals
from src.backtesting_v2.engine.portfolio_simulator import from_signals_v2
from src.backtesting.utils.risk_config import RiskConfig
from src.backtesting.engine.streaming_data_loader import StreamingDataLoader


def generate_ma_signals(data: pd.DataFrame, fast: int = 20, slow: int = 50) -> Tuple[pd.Series, pd.Series]:
    """Generate MA crossover signals."""
    close = data["close"]
    ma_fast = close.rolling(fast).mean()
    ma_slow = close.rolling(slow).mean()
    entries = ((ma_fast > ma_slow) & (ma_fast.shift(1) <= ma_slow.shift(1))).fillna(False)
    exits = ((ma_fast < ma_slow) & (ma_fast.shift(1) >= ma_slow.shift(1))).fillna(False)
    return entries, exits


def generate_mean_reversion_signals(data: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """Generate Bollinger Band mean reversion signals."""
    close = data["close"]
    ma = close.rolling(20).mean()
    std = close.rolling(20).std()
    lower = ma - (std * 2.0)
    entries = ((close < lower) & (close.shift(1) >= lower.shift(1))).fillna(False)
    exits = ((close > ma) & (close.shift(1) <= ma.shift(1))).fillna(False)
    return entries, exits


def generate_momentum_signals(data: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """Generate momentum signals."""
    close = data["close"]
    momentum = close.pct_change(20)
    entries = ((momentum > 0.02) & (momentum.shift(1) <= 0.02)).fillna(False)
    exits = ((momentum < -0.01) & (momentum.shift(1) >= -0.01)).fillna(False)
    return entries, exits


def run_comparison(
    name: str,
    data: pd.DataFrame,
    entries: pd.Series,
    exits: pd.Series,
    risk_config: RiskConfig,
    allow_shorts: bool = False,
) -> Dict[str, Any]:
    """Run V1, V2 (no tracking), and V2 (with tracking) and compare."""

    params = dict(
        close=data["close"],
        entries=entries,
        exits=exits,
        init_cash=100000,
        fees=0.001,
        slippage=0.0005,
        freq="1min",
        market_hours_only=True,
        risk_config=risk_config,
        price_data=data,
        allow_shorts=allow_shorts,
        use_numba=True,
    )

    # ========== V1 ==========
    gc.collect()
    tracemalloc.start()
    v1_start = time.perf_counter()
    pf_v1 = from_signals(**params)
    v1_time = time.perf_counter() - v1_start
    v1_mem = tracemalloc.get_traced_memory()[1] / 1024 / 1024
    tracemalloc.stop()

    # ========== V2 (track_state=False) ==========
    gc.collect()
    tracemalloc.start()
    v2_notrack_start = time.perf_counter()
    pf_v2_notrack = from_signals_v2(**params, track_state=False)
    v2_notrack_time = time.perf_counter() - v2_notrack_start
    v2_notrack_mem = tracemalloc.get_traced_memory()[1] / 1024 / 1024
    tracemalloc.stop()

    # ========== V2 (track_state=True) - simulation only ==========
    gc.collect()
    tracemalloc.start()
    v2_track_start = time.perf_counter()
    pf_v2_track = from_signals_v2(**params, track_state=True)
    v2_track_sim_time = time.perf_counter() - v2_track_start
    v2_track_sim_mem = tracemalloc.get_traced_memory()[1] / 1024 / 1024
    tracemalloc.stop()

    # ========== V2 (track_state=True) - access state to trigger build ==========
    gc.collect()
    tracemalloc.start()
    state_build_start = time.perf_counter()
    state_df = pf_v2_track.portfolio_state
    state_build_time = time.perf_counter() - state_build_start
    state_build_mem = tracemalloc.get_traced_memory()[1] / 1024 / 1024
    tracemalloc.stop()

    v2_track_total_time = v2_track_sim_time + state_build_time

    # ========== Parity checks ==========
    eq_diff_notrack = np.abs(pf_v1.equity_curve.values - pf_v2_notrack.equity_curve.values).max()
    eq_diff_track = np.abs(pf_v1.equity_curve.values - pf_v2_track.equity_curve.values).max()

    s1 = pf_v1.stats()
    s2_notrack = pf_v2_notrack.stats()
    s2_track = pf_v2_track.stats()

    trades_match_notrack = len(pf_v1.trades) == len(pf_v2_notrack.trades)
    trades_match_track = len(pf_v1.trades) == len(pf_v2_track.trades)

    parity_ok = (
        eq_diff_notrack < 0.01 and
        eq_diff_track < 0.01 and
        trades_match_notrack and
        trades_match_track
    )

    return {
        "name": name,
        "bars": len(data),
        "trades": len(pf_v1.trades),
        "parity_ok": parity_ok,
        "eq_diff_notrack": eq_diff_notrack,
        "eq_diff_track": eq_diff_track,
        "return_v1": s1["Total Return [%]"],
        "return_v2_notrack": s2_notrack["Total Return [%]"],
        "return_v2_track": s2_track["Total Return [%]"],
        "v1_time": v1_time,
        "v2_notrack_time": v2_notrack_time,
        "v2_track_sim_time": v2_track_sim_time,
        "v2_track_state_time": state_build_time,
        "v2_track_total_time": v2_track_total_time,
        "v1_mem": v1_mem,
        "v2_notrack_mem": v2_notrack_mem,
        "v2_track_sim_mem": v2_track_sim_mem,
        "v2_track_state_mem": state_build_mem,
        "state_rows": len(state_df),
    }


def print_result(r: Dict[str, Any]):
    """Print a single test result."""
    status = "[+] PASS" if r["parity_ok"] else "[-] FAIL"

    v2_notrack_overhead = ((r["v2_notrack_time"] / r["v1_time"]) - 1) * 100
    v2_track_sim_overhead = ((r["v2_track_sim_time"] / r["v1_time"]) - 1) * 100
    v2_track_total_overhead = ((r["v2_track_total_time"] / r["v1_time"]) - 1) * 100

    print(f"  {status}")
    print(f"  Bars: {r['bars']:,} | Trades: {r['trades']:,}")
    print(f"  Equity diff (no track): ${r['eq_diff_notrack']:.6f} | (with track): ${r['eq_diff_track']:.6f}")
    print()
    print(f"  --- Timing ---")
    print(f"  V1:                      {r['v1_time']:.2f}s")
    print(f"  V2 (no tracking):        {r['v2_notrack_time']:.2f}s  ({v2_notrack_overhead:+.1f}%)")
    print(f"  V2 (tracking) sim only:  {r['v2_track_sim_time']:.2f}s  ({v2_track_sim_overhead:+.1f}%)")
    print(f"  V2 (tracking) state build: {r['v2_track_state_time']:.3f}s")
    print(f"  V2 (tracking) TOTAL:     {r['v2_track_total_time']:.2f}s  ({v2_track_total_overhead:+.1f}%)")
    print()
    print(f"  --- Memory (Peak MB) ---")
    print(f"  V1: {r['v1_mem']:.1f} | V2 no track: {r['v2_notrack_mem']:.1f} | V2 track sim: {r['v2_track_sim_mem']:.1f} | state build: {r['v2_track_state_mem']:.1f}")
    print(f"  State rows: {r['state_rows']:,}")


def main():
    print("=" * 100)
    print("FULL BACKTEST COMPARISON: V1 vs V2 (no tracking) vs V2 (with tracking)")
    print("Data: 2017-2024 (7+ years of minute-level data)")
    print("=" * 100)

    loader = StreamingDataLoader(timeframe="1min")
    results = []

    # Test symbols
    symbols = ["SPY", "QQQ"]

    for symbol in symbols:
        print(f"\n{'='*100}")
        print(f"TESTING {symbol} (2017-01-01 to 2024-12-31)")
        print("=" * 100)

        # Load data
        print(f"\nLoading {symbol} data...")
        load_start = time.perf_counter()
        symbol_data = None

        try:
            for sym, df in loader.iter_symbols([symbol], "2017-01-01", "2024-12-31"):
                symbol_data = df.to_pandas().set_index("timestamp")
        except Exception as e:
            print(f"  Error loading {symbol}: {e}")
            continue

        load_time = time.perf_counter() - load_start

        if symbol_data is None or len(symbol_data) < 1000:
            print(f"  Insufficient data for {symbol}")
            continue

        print(f"  Loaded {len(symbol_data):,} bars in {load_time:.2f}s")

        # Warm up Numba (once per symbol)
        print("  Warming up Numba...")
        small_data = symbol_data.iloc[:1000]
        small_entries, small_exits = generate_ma_signals(small_data)
        _ = from_signals(
            close=small_data["close"], entries=small_entries, exits=small_exits,
            init_cash=100000, fees=0.001, slippage=0.0005, freq="1min",
            market_hours_only=True, risk_config=RiskConfig.moderate(),
            price_data=small_data, use_numba=True,
        )
        _ = from_signals_v2(
            close=small_data["close"], entries=small_entries, exits=small_exits,
            init_cash=100000, fees=0.001, slippage=0.0005, freq="1min",
            market_hours_only=True, risk_config=RiskConfig.moderate(),
            price_data=small_data, use_numba=True, track_state=True,
        )
        gc.collect()

        # Strategy 1: MA Crossover
        print(f"\n--- {symbol} MA Crossover (20/50) ---")
        entries, exits = generate_ma_signals(symbol_data)
        result = run_comparison(f"{symbol} MA Crossover", symbol_data, entries, exits, RiskConfig.moderate())
        results.append(result)
        print_result(result)

        # Strategy 2: MA Crossover with Shorts
        print(f"\n--- {symbol} MA Crossover + Shorts ---")
        result = run_comparison(f"{symbol} MA + Shorts", symbol_data, entries, exits, RiskConfig.moderate(), allow_shorts=True)
        results.append(result)
        print_result(result)

        # Strategy 3: Mean Reversion
        print(f"\n--- {symbol} Mean Reversion ---")
        entries, exits = generate_mean_reversion_signals(symbol_data)
        result = run_comparison(f"{symbol} Mean Reversion", symbol_data, entries, exits, RiskConfig.moderate())
        results.append(result)
        print_result(result)

        # Strategy 4: Momentum
        print(f"\n--- {symbol} Momentum ---")
        entries, exits = generate_momentum_signals(symbol_data)
        result = run_comparison(f"{symbol} Momentum", symbol_data, entries, exits, RiskConfig.moderate())
        results.append(result)
        print_result(result)

        # Strategy 5: With Stop Loss
        print(f"\n--- {symbol} MA + 2% Stop Loss ---")
        entries, exits = generate_ma_signals(symbol_data)
        risk_sl = RiskConfig(
            enabled=True, position_size_pct=0.10,
            use_stop_loss=True, stop_loss_pct=0.02, stop_loss_type="percentage",
        )
        result = run_comparison(f"{symbol} MA + Stop Loss", symbol_data, entries, exits, risk_sl)
        results.append(result)
        print_result(result)

        # Strategy 6: With Profit Target
        print(f"\n--- {symbol} MA + Profit Target ---")
        risk_pt = RiskConfig(
            enabled=True, position_size_pct=0.10,
            use_stop_loss=True, stop_loss_pct=0.02,
            take_profit_pct=0.03, stop_loss_type="profit_target",
        )
        result = run_comparison(f"{symbol} MA + Profit Target", symbol_data, entries, exits, risk_pt)
        results.append(result)
        print_result(result)

        # Free memory
        del symbol_data
        gc.collect()

    # ==================== SUMMARY ====================
    print("\n" + "=" * 100)
    print("FULL COMPARISON SUMMARY")
    print("=" * 100)

    passed = sum(1 for r in results if r["parity_ok"])
    failed = len(results) - passed

    print(f"\nTotal Tests: {len(results)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")

    # Aggregate stats
    total_bars = sum(r["bars"] for r in results)
    total_trades = sum(r["trades"] for r in results)
    total_v1_time = sum(r["v1_time"] for r in results)
    total_v2_notrack_time = sum(r["v2_notrack_time"] for r in results)
    total_v2_track_sim_time = sum(r["v2_track_sim_time"] for r in results)
    total_v2_track_state_time = sum(r["v2_track_state_time"] for r in results)
    total_v2_track_total_time = sum(r["v2_track_total_time"] for r in results)

    avg_v1_mem = np.mean([r["v1_mem"] for r in results])
    avg_v2_notrack_mem = np.mean([r["v2_notrack_mem"] for r in results])
    avg_v2_track_sim_mem = np.mean([r["v2_track_sim_mem"] for r in results])

    print(f"\n--- Aggregate Stats ---")
    print(f"Total bars processed: {total_bars:,}")
    print(f"Total trades executed: {total_trades:,}")

    print(f"\n--- Total Time ---")
    print(f"V1:                      {total_v1_time:.2f}s")
    print(f"V2 (no tracking):        {total_v2_notrack_time:.2f}s  ({((total_v2_notrack_time/total_v1_time)-1)*100:+.1f}%)")
    print(f"V2 (tracking) sim only:  {total_v2_track_sim_time:.2f}s  ({((total_v2_track_sim_time/total_v1_time)-1)*100:+.1f}%)")
    print(f"V2 (tracking) state build: {total_v2_track_state_time:.3f}s")
    print(f"V2 (tracking) TOTAL:     {total_v2_track_total_time:.2f}s  ({((total_v2_track_total_time/total_v1_time)-1)*100:+.1f}%)")

    print(f"\n--- Average Peak Memory ---")
    print(f"V1: {avg_v1_mem:.1f} MB")
    print(f"V2 (no tracking): {avg_v2_notrack_mem:.1f} MB")
    print(f"V2 (tracking) sim: {avg_v2_track_sim_mem:.1f} MB")

    # Detailed results table
    print("\n--- Detailed Results ---")
    print("-" * 140)
    header = f"{'Test':<25} {'Pass':<6} {'Bars':>12} {'V1':>8} {'V2 NoTrk':>10} {'V2 Trk Sim':>12} {'V2 State':>10} {'V2 Total':>10} {'Overhead':>10}"
    print(header)
    print("-" * 140)

    for r in results:
        status = "[+]" if r["parity_ok"] else "[-]"
        overhead = ((r["v2_track_total_time"] / r["v1_time"]) - 1) * 100
        print(
            f"{r['name']:<25} {status:<6} {r['bars']:>12,} "
            f"{r['v1_time']:>7.2f}s {r['v2_notrack_time']:>9.2f}s "
            f"{r['v2_track_sim_time']:>11.2f}s {r['v2_track_state_time']:>9.3f}s "
            f"{r['v2_track_total_time']:>9.2f}s {overhead:>9.1f}%"
        )

    print("-" * 140)

    # Final verdict
    print("\n" + "=" * 100)
    if failed == 0:
        print("[+] ALL TESTS PASSED - Complete parity across all three versions!")
        print()
        v2_notrack_pct = ((total_v2_notrack_time/total_v1_time)-1)*100
        v2_track_pct = ((total_v2_track_total_time/total_v1_time)-1)*100
        print(f"    V2 (no tracking):    {v2_notrack_pct:+.1f}% vs V1")
        print(f"    V2 (with tracking):  {v2_track_pct:+.1f}% vs V1 (includes lazy state build)")
    else:
        print(f"[-] {failed} TEST(S) FAILED - Parity issues detected")
    print("=" * 100)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
