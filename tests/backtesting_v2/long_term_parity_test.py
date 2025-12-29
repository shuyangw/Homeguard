"""
Long-Term Parity Test: V1 vs V2 (2017-2024).

Tests backtesting engine parity with 7+ years of minute-level data.
Tracks execution time and memory usage.
"""

import pandas as pd
import numpy as np
import time
import sys
import gc
import tracemalloc
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


def generate_mean_reversion_signals(data: pd.DataFrame, window: int = 20, num_std: float = 2.0) -> Tuple[pd.Series, pd.Series]:
    """Generate Bollinger Band mean reversion signals."""
    close = data["close"]
    ma = close.rolling(window).mean()
    std = close.rolling(window).std()

    lower = ma - (std * num_std)

    entries = ((close < lower) & (close.shift(1) >= lower.shift(1))).fillna(False)
    exits = ((close > ma) & (close.shift(1) <= ma.shift(1))).fillna(False)

    return entries, exits


def generate_momentum_signals(data: pd.DataFrame, lookback: int = 20) -> Tuple[pd.Series, pd.Series]:
    """Generate momentum signals."""
    close = data["close"]
    momentum = close.pct_change(lookback)

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
    """Run V1 and V2 backtests and compare."""

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

    # Run V1
    gc.collect()
    tracemalloc.start()
    v1_start = time.perf_counter()
    pf_v1 = from_signals(**params)
    v1_time = time.perf_counter() - v1_start
    v1_mem = tracemalloc.get_traced_memory()[1] / 1024 / 1024  # Peak MB
    tracemalloc.stop()

    # Run V2
    gc.collect()
    tracemalloc.start()
    v2_start = time.perf_counter()
    pf_v2 = from_signals_v2(**params)
    v2_time = time.perf_counter() - v2_start
    v2_mem = tracemalloc.get_traced_memory()[1] / 1024 / 1024  # Peak MB
    tracemalloc.stop()

    # Compare results
    eq_diff = np.abs(pf_v1.equity_curve.values - pf_v2.equity_curve.values)
    max_eq_diff = eq_diff.max()
    mean_eq_diff = eq_diff.mean()

    s1 = pf_v1.stats()
    s2 = pf_v2.stats()

    ret_diff = abs(s1["Total Return [%]"] - s2["Total Return [%]"])
    trades_match = len(pf_v1.trades) == len(pf_v2.trades)

    # Check trade-by-trade PnL
    pnl_match = True
    max_pnl_diff = 0.0
    if trades_match and len(pf_v1.trades) > 0:
        for t1, t2 in zip(pf_v1.trades, pf_v2.trades):
            if "pnl" in t1 and "pnl" in t2:
                diff = abs(t1["pnl"] - t2["pnl"])
                max_pnl_diff = max(max_pnl_diff, diff)
                if diff > 0.01:
                    pnl_match = False

    passed = max_eq_diff < 0.01 and ret_diff < 0.0001 and trades_match and pnl_match

    return {
        "name": name,
        "passed": passed,
        "bars": len(data),
        "trades_v1": len(pf_v1.trades),
        "trades_v2": len(pf_v2.trades),
        "max_eq_diff": max_eq_diff,
        "mean_eq_diff": mean_eq_diff,
        "max_pnl_diff": max_pnl_diff,
        "return_v1": s1["Total Return [%]"],
        "return_v2": s2["Total Return [%]"],
        "sharpe_v1": s1["Sharpe Ratio"],
        "sharpe_v2": s2["Sharpe Ratio"],
        "final_v1": pf_v1.equity_curve.iloc[-1],
        "final_v2": pf_v2.equity_curve.iloc[-1],
        "v1_time": v1_time,
        "v2_time": v2_time,
        "v1_mem_mb": v1_mem,
        "v2_mem_mb": v2_mem,
        "state_rows": len(pf_v2.portfolio_state) if hasattr(pf_v2, 'portfolio_state') else 0,
    }


def main():
    print("=" * 80)
    print("LONG-TERM PARITY TEST: V1 vs V2 (2017-2024)")
    print("=" * 80)

    loader = StreamingDataLoader(timeframe="1min")

    # Test symbols
    symbols = ["SPY", "QQQ"]

    # Date range - 7+ years
    start_date = "2017-01-01"
    end_date = "2024-12-31"

    results = []

    for symbol in symbols:
        print(f"\n{'='*80}")
        print(f"TESTING {symbol} ({start_date} to {end_date})")
        print("=" * 80)

        # Load data
        print(f"\nLoading {symbol} data...")
        load_start = time.perf_counter()
        symbol_data = None

        try:
            for sym, df in loader.iter_symbols([symbol], start_date, end_date):
                symbol_data = df.to_pandas().set_index("timestamp")
        except Exception as e:
            print(f"  Error loading {symbol}: {e}")
            continue

        load_time = time.perf_counter() - load_start

        if symbol_data is None or len(symbol_data) < 1000:
            print(f"  Insufficient data for {symbol}")
            continue

        print(f"  Loaded {len(symbol_data):,} bars in {load_time:.2f}s")
        print(f"  Date range: {symbol_data.index[0]} to {symbol_data.index[-1]}")

        # Strategy 1: MA Crossover
        print(f"\n--- {symbol} MA Crossover (20/50) ---")
        entries, exits = generate_ma_signals(symbol_data)
        result = run_comparison(
            f"{symbol} MA Crossover",
            symbol_data, entries, exits,
            RiskConfig.moderate(),
        )
        results.append(result)
        print_result(result)

        # Strategy 2: MA Crossover with Shorts
        print(f"\n--- {symbol} MA Crossover + Shorts ---")
        result = run_comparison(
            f"{symbol} MA + Shorts",
            symbol_data, entries, exits,
            RiskConfig.moderate(),
            allow_shorts=True,
        )
        results.append(result)
        print_result(result)

        # Strategy 3: Mean Reversion
        print(f"\n--- {symbol} Mean Reversion (BB 20, 2std) ---")
        entries, exits = generate_mean_reversion_signals(symbol_data)
        result = run_comparison(
            f"{symbol} Mean Reversion",
            symbol_data, entries, exits,
            RiskConfig.moderate(),
        )
        results.append(result)
        print_result(result)

        # Strategy 4: Momentum
        print(f"\n--- {symbol} Momentum ---")
        entries, exits = generate_momentum_signals(symbol_data)
        result = run_comparison(
            f"{symbol} Momentum",
            symbol_data, entries, exits,
            RiskConfig.moderate(),
        )
        results.append(result)
        print_result(result)

        # Strategy 5: With Stop Loss
        print(f"\n--- {symbol} MA + 2% Stop Loss ---")
        entries, exits = generate_ma_signals(symbol_data)
        risk_sl = RiskConfig(
            enabled=True,
            position_size_pct=0.10,
            use_stop_loss=True,
            stop_loss_pct=0.02,
            stop_loss_type="percentage",
        )
        result = run_comparison(
            f"{symbol} MA + Stop Loss",
            symbol_data, entries, exits,
            risk_sl,
        )
        results.append(result)
        print_result(result)

        # Strategy 6: With Profit Target
        print(f"\n--- {symbol} MA + Profit Target ---")
        risk_pt = RiskConfig(
            enabled=True,
            position_size_pct=0.10,
            use_stop_loss=True,
            stop_loss_pct=0.02,
            take_profit_pct=0.03,
            stop_loss_type="profit_target",
        )
        result = run_comparison(
            f"{symbol} MA + Profit Target",
            symbol_data, entries, exits,
            risk_pt,
        )
        results.append(result)
        print_result(result)

        # Free memory
        del symbol_data
        gc.collect()

    # Summary
    print("\n" + "=" * 80)
    print("LONG-TERM PARITY TEST SUMMARY")
    print("=" * 80)

    passed = sum(1 for r in results if r["passed"])
    failed = len(results) - passed

    print(f"\nTotal Tests: {len(results)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")

    # Performance summary
    total_bars = sum(r["bars"] for r in results)
    total_trades = sum(r["trades_v1"] for r in results)
    total_v1_time = sum(r["v1_time"] for r in results)
    total_v2_time = sum(r["v2_time"] for r in results)
    avg_v1_mem = np.mean([r["v1_mem_mb"] for r in results])
    avg_v2_mem = np.mean([r["v2_mem_mb"] for r in results])

    print(f"\n--- Performance Summary ---")
    print(f"Total bars processed: {total_bars:,}")
    print(f"Total trades executed: {total_trades:,}")
    print(f"\nV1 total time: {total_v1_time:.2f}s")
    print(f"V2 total time: {total_v2_time:.2f}s")
    print(f"V2 overhead: {((total_v2_time / total_v1_time) - 1) * 100:.1f}%")
    print(f"\nV1 avg peak memory: {avg_v1_mem:.1f} MB")
    print(f"V2 avg peak memory: {avg_v2_mem:.1f} MB")
    print(f"V2 memory overhead: {((avg_v2_mem / avg_v1_mem) - 1) * 100:.1f}%")

    # Detailed results table
    print("\n--- Detailed Results ---")
    print("-" * 120)
    print(f"{'Test':<30} {'Pass':<6} {'Bars':>12} {'Trades':>10} {'Eq Diff':>12} {'V1 Time':>10} {'V2 Time':>10} {'Overhead':>10}")
    print("-" * 120)

    for r in results:
        status = "[+]" if r["passed"] else "[-]"
        overhead = ((r["v2_time"] / r["v1_time"]) - 1) * 100 if r["v1_time"] > 0 else 0
        print(f"{r['name']:<30} {status:<6} {r['bars']:>12,} {r['trades_v1']:>10,} ${r['max_eq_diff']:>10.4f} {r['v1_time']:>9.2f}s {r['v2_time']:>9.2f}s {overhead:>9.1f}%")

    print("-" * 120)

    # Final verdict
    print("\n" + "=" * 80)
    if failed == 0:
        print("[+] ALL LONG-TERM TESTS PASSED - V1 and V2 have COMPLETE PARITY!")
    else:
        print(f"[-] {failed} TEST(S) FAILED - Parity issues detected")
    print("=" * 80)

    return 0 if failed == 0 else 1


def print_result(r: Dict[str, Any]):
    """Print a single test result."""
    status = "[+] PASS" if r["passed"] else "[-] FAIL"
    overhead = ((r["v2_time"] / r["v1_time"]) - 1) * 100 if r["v1_time"] > 0 else 0

    print(f"  {status}")
    print(f"  Bars: {r['bars']:,} | Trades: {r['trades_v1']:,}")
    print(f"  Max Eq Diff: ${r['max_eq_diff']:.6f} | Max PnL Diff: ${r['max_pnl_diff']:.6f}")
    print(f"  Return V1: {r['return_v1']:.2f}% | Return V2: {r['return_v2']:.2f}%")
    print(f"  Final V1: ${r['final_v1']:,.2f} | Final V2: ${r['final_v2']:,.2f}")
    print(f"  V1 Time: {r['v1_time']:.2f}s | V2 Time: {r['v2_time']:.2f}s | Overhead: {overhead:.1f}%")
    print(f"  V1 Mem: {r['v1_mem_mb']:.1f}MB | V2 Mem: {r['v2_mem_mb']:.1f}MB | State rows: {r['state_rows']:,}")


if __name__ == "__main__":
    sys.exit(main())
